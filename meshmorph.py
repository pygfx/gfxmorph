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
        self._meta = {}
        self._gl_objects = None
        self._version_count = 0

        # Delegate initialization
        self.set_data(vertices, faces, oversize_factor)

    @property
    def is_closed(self):
        """Whether the mesh is closed.

        If this tool is used in a GUI, you can use this flag to show a
        warning symbol. If this value is False, a warning with more
        info is shown in the console.
        """
        return len(self._original_unclosed_faces) == 0

    @property
    def component_count(self):
        """The number of sub-meshes (connected components) this meshs consists of."""
        return self._component_count

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
            raise ValueError("The mesh must have more nonzero faces and vertices.")

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
        self._check_closed()

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

    @property
    def edges(self):
        """Return all edges of this mesh as pairs of vertex indices

        Returns
        -------
        ndarray, [n_faces, 3, 2]
            pairs of vertex-indices specifying an edge.
            the ith edge is the edge opposite from the ith vertex of the face

        """
        array = self.faces[:, [[1, 2], [2, 0], [0, 1]]]
        array.setflags(write=False)
        return array

    def _check_closed(self):
        """Check that the mesh is closed.

        This method also autocorrects the winding of indidual faces,
        as well as of a whole sub-mesh (connected component) so that their
        volumes are positive.

        This method reports its findings by showing a warning and by setting
        a private attribute (which can e.g. be checked in tests and debugging).
        """

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
                    if len(matching_vertices) >= 2:
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
                        unclosed_case2.append((fi_check, neighbour_per_edge))

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
        if total_reversed_count > 0:
            warnings.warn(
                f"Auto-corrected the winding of {total_reversed_count} faces."
            )

        # Report on the mesh not being closed - this is a problem!
        # todo: how bad is this problem again? Should we abort or still allow e.g. morphing the mesh?
        self._original_unclosed_faces = {fi for fi, _ in unclosed_case1} | {
            fi for fi, _ in unclosed_case2
        }
        if unclosed_case1:
            lines = [f"    - {fi} ({nbe})" for fi, nbe in unclosed_case1]
            n = len(unclosed_case1)
            warnings.warn(
                f"There is a hole in the mesh at {n} faces:\n" + "\n".join(lines)
            )
        if unclosed_case2:
            lines = [f"    - {fi} ({nbe})" for fi, nbe in unclosed_case2]
            n = len(unclosed_case2)
            warnings.warn(
                f"Too many neighbour faces at {n} faces:\n" + "\n".join(lines)
            )

    def validate_internals(self):
        raise NotImplementedError()
        # todo: xx

    def get_volume(self):
        """Calculate the volume of the mesh."""
        (valid_face_indices,) = np.where(self._faces[:, 0] > 0)
        valid_faces = self._faces[valid_face_indices]
        return maybe_pylinalg.volume_of_closed_mesh(self._vertices, valid_faces)

    def split(self):
        """Return a list of Mesh objects, one for each connected component."""
        # I don't think we need this for our purpose, but this class is capable
        # of doing something like this, so it could be a nice util.
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

    def get_neighbour_faces(self, fi):
        """Get a list of face indices that neighbour the given face index."""
        faces = self._faces
        vertices = self._vertices
        vertex2faces = self._vertex2faces

        neighbour_faces = set()
        for vi in faces[fi]:
            neighbour_faces.update(vertex2faces[vi])
        neighbour_faces.discard(fi)

        # todo: is it worth converting to array?
        return neighbour_faces
        # return np.array(neighbour_faces, np.int32)

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

    # geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
    # geo = gfx.sphere_geometry(1)
    # geo = gfx.geometries.tetrahedron_geometry()
    geo = maybe_pygfx.solid_sphere_geometry(1)
    m = AbstractMesh(geo.positions.data, geo.indices.data)

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
