# ## Developer notes
#
# Our internal arrays are larger than needed - we have free slots. This
# is because we must be able to dynamically add and remove vertices and
# faces. Further, we have one reserved vertex slot at index zero,
# because we must be able to mark faces as unused and still point to a
# valid vertex index.
#
# In the faces array, empty faces are denoted by setting its values to 0.
# In the vertices array, empty vertices are denoted by setting its values to NaN.
#
# We always make a copy of the given data:
# - so we control the dtype.
# - we will change the values, avoid surprises by modifying given arrays.
# - we need the first vertex to be empty.
# - we may want to initialize with some extra size.
#
# Vertex indices are denoted with vi, face indices with fi.

import time
import warnings  # todo: use warnings or logger?
import queue

import numpy as np
import maybe_pylinalg
import maybe_pygfx

# We assume meshes with triangles (not quads) for now
vertices_per_face = 3


# todo: better name
class AbstractMesh:
    """
    A mesh object representing a closed mesh, plus utilities to work with such as mesh.
    Implemented in a generic way ... though not sure how yet :)
    """

    def __init__(self, vertices, faces, oversize_factor=1.5):
        # In this method we first initialize all used attributes. This
        # serves mostly as internal docs.

        # The counts of active faces and vertices.
        self._nvertices = 0
        self._nfaces = 0

        # The number of slots for the face and vertex arrays.
        self._nvertices_slots = 0  # >= _nvertices + 1
        self._nfaces_slots = 0  # >= _nfaces

        # Array placeholders.
        # Normals are used to determine surface curvature. Colors are
        # used as a feedback mechanism to show what vertices participate
        # in a morph.
        self._vertices = None  # Nx3 float32 arrray
        self._normals = None  # Nx3 float32 array
        self._colors = None  # Nx4 float32 array
        self._faces = None  # Nx3 int32 array

        # We keep track of free slots using sets.
        self._free_vertices = set()
        self._free_faces = set()

        # The reverse mapping. Is a list because the "rows" have variable size.
        # This is probably the most important data structure of this module ...
        self._vertex2faces = []  # list mapping vi -> [fi1, fi2, fi3, ...]

        # We keep track of what has changed, so we can update efficiently to the GPU
        self._dirty_vertices = set()
        self._dirty_colors = set()
        self._dirty_faces = set()

        # todo: other attributes on the original that I'm not sure of ...
        self._mesh_props = (
            {}
        )  # todo: this must be cleared whenever the topology has changed
        self._meta = {}
        self._gl_objects = None
        self._version_count = 0

        # Delegate initialization
        self.set_data(vertices, faces, oversize_factor)

    @property
    def is_edge_manifold(self):
        """Whether the mesh is edge_manifold.

        This is another way of saying that each edge is part of either
        1 or 2 faces. It is one of the two condition for a mesh to be
        manifold. You could also call it weak-manifold, I suppose.
        """
        if "is_edge_manifold" not in self._mesh_props:
            self.check_edge_manifold_and_closed()
        return self._mesh_props["is_edge_manifold"]

    @property
    def is_manifold(self):
        """Whether the mesh is manifold.

        The mesh being manifold means that:

        * Each edge is part of 1 or 2 faces.
        * The faces incident to a vertex form a closed or an open fan.
        """
        if "is_only_connected_by_edges" not in self._mesh_props:
            self.check_only_connected_by_edges()
        return self.is_edge_manifold and self._mesh_props["is_only_connected_by_edges"]

    @property
    def is_closed(self):
        """Whether the mesh is closed.

        If this tool is used in a GUI, you can use this flag to show a
        warning symbol. If this value is False, a warning with more
        info is shown in the console.
        """
        if "is_closed" not in self._mesh_props:
            self.check_edge_manifold_and_closed()
        return self._mesh_props["is_closed"]

    @property
    def is_oriented(self):
        """Whether the mesh is orientable.

        The mesh being orientable means that the face orientation (i.e. winding) is
        consistent - each two neighbouring faces have the same orientation.
        """
        # The way we calculate orientedness, also implies edge-manifoldness.
        # todo: I think that a mesh can be edge_manifold, connected, but not manifold.
        if "is_oriented" not in self._mesh_props:
            self.check_oriented()
        return self._mesh_props["is_oriented"]

    @property
    def is_volumetric(self):
        """Whether the mesh has a positive volume.

        If not, the mesh is probably inside-out. CCW winding is assumed.
        """
        # todo: ...

    @property
    def edges(self):
        """All edges of this mesh as pairs of vertex indices

        Returns
        -------
        ndarray, [n_faces, 3, 2]
            pairs of vertex-indices specifying an edge.
            the ith edge is the edge opposite from the ith vertex of the face

        """
        # todo: only use valid faces, maybe per component?
        array = self.faces[:, [[1, 2], [2, 0], [0, 1]]]
        array.setflags(write=False)
        return array

    @property
    def metadata(self):
        """A dict with metadata about the mesh."""
        arrays = self._faces, self._vertices, self._colors, self._normals
        nb = sum([a.nbytes for a in arrays if a is not None])

        return {
            "is_closed": self.is_closed,
            "components": self._component_count,
            "vertices": self._nvertices,
            "faces": self._nfaces,
            "free_vertices": len(self._free_vertices),
            "free_faces": len(self._free_faces),
            "approx_mem": f"{nb/2**20:0.2f} MiB"
            if nb > 2**20
            else f"{nb/2**10:0.2f} KiB",
        }

    # %%

    def set_data(self, vertices, faces, oversize_factor=1.5):
        """Set the (new) vertex and face data.

        The data is copied and the internal data structure is build up.
        The winding of faces is automatically corrected where needed, and a warning
        is shown if this was the case. The mesh is also validated for being closed.
        If the mesh is not closed, a warning is shown for each offending face,
        and the ``is_closed`` property is set to False.
        """

        # Check incoming arrays
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        if not (
            isinstance(vertices, np.ndarray)
            and vertices.ndim == 2
            and vertices.shape[1] == 3
        ):
            raise TypeError("Vertices must be a Nx3 array")
        if not (
            isinstance(faces, np.ndarray)
            and faces.ndim == 2
            and vertices.shape[1] == vertices_per_face
        ):
            raise TypeError("Faces must be a Nx3 array")

        # We want to be able to assume that there is at least one face, and 3 valid vertices
        # (or 4 faces and 4 vertices for a closed mesh)
        if not faces.size or not vertices.size:
            raise ValueError("The mesh must have nonzero faces and vertices.")

        # Check sanity of the faces
        if faces.min() < 0 or faces.max() >= vertices.shape[0]:
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        oversize_factor = max(1, oversize_factor or 1.5)

        # Set counts
        self._nvertices = vertices.shape[0]
        self._nfaces = faces.shape[0]
        self._nvertices_slots = int(oversize_factor * self._nvertices) + 1
        self._nfaces_slots = int(oversize_factor * self._nfaces)

        # Check-check, double-check
        assert self._nvertices_slots >= self._nvertices + 1
        assert self._nfaces_slots >= self._nfaces

        # Allocate arrays.
        self._vertices = np.empty((self._nvertices_slots, 3), np.float32)
        self._normals = np.empty((self._nvertices_slots, 3), np.float32)
        self._colors = np.empty((self._nvertices_slots, 3), np.float32)
        self._faces = np.empty((self._nfaces_slots, vertices_per_face), np.int32)

        # Copy the data
        self._vertices[1 : self._nvertices + 1, :] = vertices
        self._faces[0 : self._nfaces, :] = faces + 1

        # Nullify the free slots
        self._vertices[0, :] = 0
        self._vertices[self._nvertices + 1 :, :] = np.NaN
        self._faces[self._nfaces :, :] = 0

        # Collect the free slots
        self._free_vertices = set(range(self._nvertices + 1, self._nvertices_slots))
        self._free_faces = set(range(self._nfaces, self._nfaces_slots))

        # Update self._vertex2faces (the reverse mapping)
        self._calculate_vertex2faces()

        # Mark for full upload
        self._dirty_vertices.add("reset")
        self._dirty_colors.add("reset")
        self._dirty_faces.add("reset")

        # Check/ correct winding and check whether it's a solid
        # self._fix_stuff()

        # todo:
        # self.data_update()
        # self.metadata_update()

    def _calculate_vertex2faces(self):
        """Do a full reset of the vertex2faces backward mapping array."""
        faces = self._faces

        # todo: what offers the best performance (at the right moment)?
        # - A python list with lists?
        # - A python list with arrays (would have to do an extra step here)?
        # - An array with lists?
        # - An array with arrays?

        self._vertex2faces = vertex2faces = [[] for i in range(self._nvertices_slots)]

        # self._vertex2faces = vertex2faces = np.empty(self._nvertices_slots, dtype=list)
        # for vi in range(1, self._nvertices_slots):
        #     vertex2faces[vi] = []

        for fi in range(self._nfaces_slots):
            face = faces[fi]
            if face[0] == 0:
                continue  # empty slot
            # Loop unrolling helps performance a bit
            vertex2faces[face[0]].append(fi)
            vertex2faces[face[1]].append(fi)
            vertex2faces[face[2]].append(fi)

    def check_edge_manifold_and_closed(self):
        """Check whether the mesh is edge-manifold, and whether it is closed.

        This is a (probably) much faster way to check that the mesh is edge-manifold.
        It does not guarantee proper manifoldness though, since there can still be faces
        that are attached via a single vertex.
        """

        # Select faces
        (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
        faces = self._faces[valid_face_indices, :]

        # Select edges
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
        edges = edges.reshape(-1, 2)
        edges.sort(axis=1)

        # This line is the performance bottleneck. It is not worth
        # combining this method with e.g. check_oriented, because this
        # line needs to be applied to different data, so the gain would
        # be abour zero.
        _, counts_sorted = np.unique(edges, axis=0, return_counts=True)

        # The mesh is edge-manifold if edges are shared at most by 2 faces.
        is_edge_manifold = bool(counts_sorted.max() <= 2)

        # The mesh is closed if it has no edges incident to just once face.
        # The following is equivalent to np.all(counts_sorted == 2)
        is_closed = is_edge_manifold and bool(counts_sorted.min() == 2)

        # Store and return
        self._mesh_props["is_edge_manifold"] = is_edge_manifold
        self._mesh_props["is_closed"] = is_closed
        return is_edge_manifold, is_closed

    def check_oriented(self):
        """Check whether  the mesh is oriented. Also implies edge-manifoldness."""

        # Select faces
        (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
        faces = self._faces[valid_face_indices, :]

        # Select edges, no sorting!
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
        edges = edges.reshape(-1, 2)

        # The magic line
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)

        # If neighbouring faces have consistent winding, their edges
        # are in opposing directions, so the unsorted edges should have
        # no duplicates. Note that ths implies edge manifoldness,
        # because if an edge is incident to more than 2 faces, one of
        # the edges orientations would appear twice.
        is_oriented = edge_counts.max() == 1

        # Store and return
        self._mesh_props["is_oriented"] = is_oriented
        return is_oriented

    def check_only_connected_by_edges(self):
        """Check whether the mesh is only connected by edges.

        This helps checking for manifoldness. The second condition is:
        the faces incident to a vertex form a closed or an open fan.
        Another way to say this is that a group of connected faces can
        only connect by sharing an edge. It cannot connect to another
        group 1) via an edge that already has 2 faces, or 2) via only
        a vertex. The case (1) is covered by every edge having either
        1 or 2 faces. So we only have to check for the second failure.

        We do this by splitting components using vertex-connectedness,
        and then splitting each component using edge-connectedness. If
        a component has sub-components, then these are connected via a
        vertex, which violates thes rule (that faces around a vertex
        must be a fan or half-open fan).
        """

        components = self._split_components(None, via_edges_only=False)
        is_only_connected_by_edges = True

        for component in components:
            subcomponents = self._split_components(component, via_edges_only=True)
            if len(subcomponents) > 1:
                is_only_connected_by_edges = False
                break

        self._mesh_props["is_only_connected_by_edges"] = is_only_connected_by_edges
        return is_only_connected_by_edges

    def check_manifold_closed_deprecated(self):
        """Does a proper check that the mesh is manifold, and also whether its closed."""

        # This algorithm checks the faces around each vertex. This
        # allows checking all conditions for manifoldness, and also
        # whether the mesh is closed. It's not even that slow. But using
        # another combination of check-methods is faster to cover all
        # mesh-properties. Leaving this for a bit, beacause we may use
        # it to do repairs.

        # todo: clean this up

        fliebertjes = 0
        flobbertes = 0
        isopen = 0

        # todo: this selection could be done beforehand, of if we split components first, we may already have lists available?
        for vi1 in range(len(self._vertices)):
            if np.isnan(self._vertices[vi1][0]):
                continue

            # all_vertex_indices = []
            # for fi in self._vertex2faces[vi1]:
            #     vii = set(self._faces[fi])
            #     vii.remove(vi1)
            #     all_vertex_indices.extend(vii)
            #
            # # todo: the below check is not enough to detect all of the above, but it can detect fans, and if it does that considerably
            # # faster then we could do that first, and only to the slower check if its not a fan.
            # # That way we may get reasonable fast code for closed meshes, while being somewhat slower along the edges of open meshes.
            # unique_ids, counts = np.unique(all_vertex_indices, return_counts=True)
            # if (counts==2).all():
            #     continue  # a fan, all is well!
            #
            # if (counts > 2).any():
            #     fliebertjes += 1
            #     continue  # fail, next!
            #
            # if (counts==1).any():
            #     isopen += 1
            #     if (counts==1).sum() > 2:
            #         flobbertes += 1
            #
            # continue

            # Fom the current vertex (vi1), we select all incident faces
            # (all faces that use the vertex) and from that select all
            # neighbouring vertices.
            #
            # Our goal is to check whether these faces form a closed or an open fan.
            # There may not be fliebertjes (a face attached by an edge that is already
            # incident to 2 other faces), or flobbertjes (a face that is attached by only
            # a vertex.
            #
            #
            #  v2     /|v3
            #  |\    / |
            #  | \  /  |
            #  |  vi1  |
            #  |/   \  |
            #  v1    \ |
            #         \|v4
            #

            # vertex_indices_per_face = []
            vertex_counts = {}
            for fi in self._vertex2faces[vi1]:
                vii = set(self._faces[fi])
                vii.remove(vi1)
                for vi in vii:
                    # vertex_counts[vi] = vertex_counts.get(vi, 0) + 1
                    vertex_counts.setdefault(vi, []).append(fi)
                # vertex_indices_per_face.append(vii)

            # for vii in vertex_indices_per_face:
            #     xx = sum(vertex_counts[vi] for vi in vii)
            #     if xx == 0:
            #         self._is_manifold = False
            #     elif xx > 2:
            #         self.

            for vi2, fii in vertex_counts.items():
                if len(fii) == 2:
                    pass  # ok!
                elif len(fii) > 2:
                    fliebertjes += 1  # Bad!
                elif len(fii) == 1:
                    isopen += 1
                    # Might be an edge, or could be a flobbertje
                    fi = fii[0]
                    vii = set(self._faces[fi])
                    if len(vii) != 3:
                        flobbertes += 1
                        continue
                        # todo: a degenerate triangle, should we remove that from top instead?
                    vii.remove(vi1)
                    vii.remove(vi2)
                    vi = vii.pop()
                    if len(vertex_counts[vi]) == 1:
                        if len(vertex_counts) > 2:
                            # If this is the outer vertex on a triangle on the egde, we don't
                            # need to check this further
                            flobbertes += 1

        is_manifold = fliebertjes == 0 and flobbertes == 0
        is_closed = isopen == 0

        return is_manifold, is_closed

    def _split_components(self, face_indices=None, *, via_edges_only=True):
        """Split the mesh in one or more connected components."""

        # Performance notes:
        # * Using a deque for the front increases performance a tiny bit.
        # * Using set logic makes it a bit slower e.g.
        #   `new_neighbour_faces = neighbour_faces.intersection(faces_to_check)`
        # * Using a list for faces_in_this_component avoids having to cast to list later.

        # Collect faces to check
        if face_indices is None:
            (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
            faces_to_check = set(valid_face_indices)
        else:
            faces_to_check = set(face_indices)

        # List of components, with each component being a list of face indices.
        components = []

        while len(faces_to_check) > 0:
            # Create new front - once for each connected component in the mesh
            fi_next = faces_to_check.pop()
            faces_in_this_component = [fi_next]
            front = queue.deque()
            front.append(fi_next)

            # Walk along the front until we find no more neighbours
            while front:
                fi_check = front.popleft()
                neighbour_faces = self.get_neighbour_faces(
                    fi_check, via_edges_only=via_edges_only
                )
                for fi in neighbour_faces:
                    if fi in faces_to_check:
                        faces_to_check.remove(fi)
                        front.append(fi)
                        faces_in_this_component.append(fi)

            # We've now found one connected component (there may be more).
            components.append(faces_in_this_component)

        return components

    def _fix_stuff_deprecated(self):
        """Check that the mesh is closed.

        This method also autocorrects the winding of indidual faces,
        as well as of a whole sub-mesh (connected component) so that their
        volumes are positive.

        This method reports its findings by showing a warning and by setting
        a private attribute (which can e.g. be checked in tests and debugging).
        """

        # This alogotitm is a port of what we did in Arbiter to check the mesh.
        # This code can:
        # * Check edge-manifoldness, closed, orientability.
        # * Repair winding, repair meshes being inside-out.
        # Leaving for a bit, because we may use some of it for doing repairs.

        # todo: clean this up
        # todo: the _check_manifold_nr2 checks manifoldness and closeness, so in here we can restrict to the winding (detecting and fixing) and detecting connected components.
        # todo: if holes are detected, we can check the vertices at these locations for duplicates, and we can merge those duplicates to create a closed mesh from a "stitched" one.
        faces = self._faces

        # Collect faces to check
        (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
        faces_to_check = set(valid_face_indices)

        # Keep track of issues
        component_count = 0
        total_reversed_count = 0
        unclosed_case1 = []
        unclosed_case2 = []

        self._is_oriented = self._is_manifold = True

        while len(faces_to_check) > 0:
            # Create new front - once for each connected component in the mesh
            component_count += 1
            vi_next = faces_to_check.pop()
            faces_in_this_component = {vi_next}
            reversed_faces = []
            front = queue.deque()
            front.append(vi_next)

            # Walk along the front
            while len(front) > 0:
                fi_check = front.popleft()
                vi1, vi2, vi3 = faces[fi_check]
                neighbour_per_edge = [0, 0, 0]
                for fi in self.get_neighbour_faces(fi_check):
                    vj1, vj2, vj3 = faces[fi]
                    matching_vertices = {vj1, vj2, vj3} & {vi1, vi2, vi3}
                    if len(matching_vertices) == 3:
                        self._is_manifold = False
                    elif len(matching_vertices) == 2:
                        # todo: we now know that we have two matches, so we can write these three if-s better, I think
                        if vi1 in matching_vertices and vi2 in matching_vertices:
                            neighbour_per_edge[0] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in faces_in_this_component:
                                    front.append(fi)
                                    faces_in_this_component.add(fi)
                                if (
                                    (vi1 == vj1 and vi2 == vj2)
                                    or (vi1 == vj2 and vi2 == vj3)
                                    or (vi1 == vj3 and vi2 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )
                        elif vi2 in matching_vertices and vi3 in matching_vertices:
                            neighbour_per_edge[1] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in faces_in_this_component:
                                    front.append(fi)
                                    faces_in_this_component.add(fi)
                                if (
                                    (vi2 == vj1 and vi3 == vj2)
                                    or (vi2 == vj2 and vi3 == vj3)
                                    or (vi2 == vj3 and vi3 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )
                        elif vi3 in matching_vertices and vi1 in matching_vertices:
                            neighbour_per_edge[2] += 1
                            if fi in faces_to_check:
                                faces_to_check.remove(fi)
                                if fi not in faces_in_this_component:
                                    front.append(fi)
                                    faces_in_this_component.add(fi)
                                if (
                                    (vi3 == vj1 and vi1 == vj2)
                                    or (vi3 == vj2 and vi1 == vj3)
                                    or (vi3 == vj3 and vi1 == vj1)
                                ):
                                    reversed_faces.append(fi)
                                    faces[fi, 1], faces[fi, 2] = (
                                        faces[fi, 2],
                                        faces[fi, 1],
                                    )

                # Now that we checked all neighbours, check if we have a neighbour on each edge.
                # If this is the case for all faces, we know that the mesh is closed. The mesh
                # can still have weird crossovers or parts sticking out though (e.g. a Klein bottle).
                if not (
                    neighbour_per_edge[0] == 1
                    and neighbour_per_edge[1] == 1
                    and neighbour_per_edge[2] == 1
                ):
                    if (
                        neighbour_per_edge[0] == 0
                        or neighbour_per_edge[1] == 0
                        or neighbour_per_edge[2] == 0
                    ):
                        unclosed_case1.append((fi_check, neighbour_per_edge))
                    else:
                        self._is_manifold = False
                        unclosed_case2.append((fi_check, neighbour_per_edge))

            self._is_oriented = not reversed_faces

            # We've now found one connected component (there may be more)
            faces_in_this_component = list(faces_in_this_component)
            cfaces = self._faces[faces_in_this_component]
            volume = maybe_pylinalg.volume_of_closed_mesh(self._vertices, cfaces)

            if volume < 0:
                # Reverse the whole component
                cfaces[:, 1], cfaces[:, 2] = cfaces[:, 2].copy(), cfaces[:, 1].copy()
                self._faces[faces_in_this_component] = cfaces
                total_reversed_count += len(faces_in_this_component) - len(
                    reversed_faces
                )
                self._dirty_faces.update(faces_in_this_component)
            else:
                # Mark reversed faces as changed for GPU
                total_reversed_count += len(reversed_faces)
                self._dirty_faces.update(reversed_faces)

        self._component_count = component_count

        # Report on winding - all is well, we corrected it!
        self._n_reversed_faces = total_reversed_count
        # if total_reversed_count > 0:
        #     warnings.warn(
        #         f"Auto-corrected the winding of {total_reversed_count} faces."
        #     )

        # Report on the mesh not being closed - this is a problem!
        # todo: how bad is this problem again? Should we abort or still allow e.g. morphing the mesh?
        self._original_unclosed_faces = {fi for fi, _ in unclosed_case1} | {
            fi for fi, _ in unclosed_case2
        }
        if unclosed_case1:
            lines = [f"    - {fi} ({nbe})" for fi, nbe in unclosed_case1]
            n = len(unclosed_case1)
            # warnings.warn(
            #     f"There is a hole in the mesh at {n} faces:\n" + "\n".join(lines)
            # )
        if unclosed_case2:
            lines = [f"    - {fi} ({nbe})" for fi, nbe in unclosed_case2]
            n = len(unclosed_case2)
            # warnings.warn(
            #     f"Too many neighbour faces at {n} faces:\n" + "\n".join(lines)
            # )

    def validate_internals(self):
        raise NotImplementedError()
        # todo: xx

    def get_volume(self):
        """Calculate the volume of the mesh.

        It is assumed that the mesh is closed. If not, the result does
        not mean much. If the volume is negative, it is inside out.
        """
        # This code is surprisingly fast, over 10x faster than the other
        # checks. I also checked out skcg's computed_interior_volume,
        # which uses Gauss' theorem of calculus, but that is
        # considerably (~8x) slower.
        (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
        valid_faces = self._faces[valid_face_indices]
        return maybe_pylinalg.volume_of_closed_mesh(self._vertices, valid_faces)

    def get_surface_area(self):
        # see skcg computed_surface_area
        # Or simply calculate area of each triangle (vectorized)
        raise NotImplementedError()

    def split(self):
        """Return a list of Mesh objects, one for each connected component."""
        # I don't think we need this for our purpose, but this class is capable
        # of doing something like this, so it could be a nice util.
        raise NotImplementedError()

    def merge(self, other):
        raise NotImplementedError()

    # %% Walk over the surface

    def get_neighbour_vertices(self, vi):
        """Get a list of vertex indices that neighbour the given vertex index."""
        faces = self._faces
        vertices = self._vertices
        face_indices = self._vertex2faces[vi]

        neighbour_vertices = set(faces[face_indices].flat)
        neighbour_vertices.discard(vi)

        # todo: is it worth converting to array?
        return neighbour_vertices
        # return np.array(neighbour_vertices, np.int32)

    def get_neighbour_faces(self, fi, *, via_edges_only=False):
        """Get a list of face indices that neighbour the given face index."""

        vertex2faces = self._vertex2faces

        if via_edges_only:
            neighbour_faces1 = set()
            neighbour_faces2 = set()
            for vi in self._faces[fi]:
                for fi2 in vertex2faces[vi]:
                    if fi2 in neighbour_faces1:
                        neighbour_faces2.add(fi2)
                    else:
                        neighbour_faces1.add(fi2)
            neighbour_faces = neighbour_faces2
        else:
            neighbour_faces = set()
            for vi in self._faces[fi]:
                neighbour_faces.update(vertex2faces[vi])

        neighbour_faces.discard(fi)
        return neighbour_faces

    def get_closest_vertex(self, ref_pos):
        """Get the vertex index closest to the given 3D point, and its distance."""
        ref_pos = np.asarray(ref_pos, np.float32)
        if ref_pos.shape != (3,):
            raise ValueError("ref_pos must be a position (3 values).")

        distances = np.linalg.norm(self._vertices - ref_pos, axis=1)
        vi = np.nanargmin(distances)
        return vi, distances[vi]

    def select_vertices_over_surface(self, ref_vertex, max_distance):
        """Given a reference vertex, select more nearby vertices (over the surface).
        Returns a dict mapping vertex indices to dicts containing {pos, color, sdist, adist}.
        """
        vertices = self._vertices
        vi0 = int(ref_vertex)
        p0 = vertices[vi0]
        # selected_vertices = {vi: dict(pos=[x1, y1, z1], color=color, sdist=0, adist=0)}
        selected_vertices = {vi0}
        vertices2check = [(vi0, 0)]
        while len(vertices2check) > 0:
            vi1, cumdist = vertices2check.pop(0)
            p1 = vertices[vi1]
            for vi2 in self.get_neighbour_vertices(vi1):
                if vi2 not in selected_vertices:
                    p2 = vertices[vi2]
                    sdist = cumdist + np.linalg.norm(p2 - p1)
                    if sdist < max_distance:
                        adist = np.linalg.norm(p2 - p0)
                        # selected_vertices[vi2] = dict(pos=[xn, yn, zn], color=color, sdist=sdist, adist=adist)
                        selected_vertices.add(vi2)
                        vertices2check.append((vi2, sdist))
        return selected_vertices


if __name__ == "__main__":
    import pygfx as gfx

    # The geometries with sharp corners like cubes and all hedrons are not closed
    # because we deliberately don't share vertices, so that the normals don't interpolate
    # and make the edges look weird.
    #
    # Some other geometries, like the sphere and torus knot, also seems open thought, let's fix that!

    def get_tetrahedron():
        """A closed tetrahedron as simple list objects, so we can easily add stuff to create corrupt meshes."""
        vertices = [
            [-1, 0, -1 / 2**0.5],
            [+1, 0, -1 / 2**0.5],
            [0, -1, 1 / 2**0.5],
            [0, +1, 1 / 2**0.5],
        ]
        faces = [
            [2, 0, 1],
            [0, 2, 3],
            [1, 0, 3],
            [2, 1, 3],
        ]
        return vertices, faces, True

    # geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
    # geo = gfx.sphere_geometry(1)
    # geo = gfx.geometries.tetrahedron_geometry()
    geo = maybe_pygfx.smooth_sphere_geometry(1)
    # m = AbstractMesh(geo.positions.data, geo.indices.data)
    m = AbstractMesh(*get_tetrahedron())

    m._check_manifold_nr1()

    # positions = np.array(
    #     [
    #         [1, 1, 1],
    #         [-1, -1, 1],
    #         [-1, 1, -1],
    #         [1, -1, -1],
    #     ],
    #     dtype=np.float32,
    # )
    #
    # indices = np.array(
    #     [
    #         [2, 0, 1],
    #         [0, 2, 3],
    #         [1, 0, 3],
    #         [2, 1, 3],
    #     ],
    #     dtype=np.int32,
    # )
    # m = AbstractMesh(positions, indices)
