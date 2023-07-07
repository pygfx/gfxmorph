# ## Developer notes
#
# Our internal arrays are larger than needed - we have free slots. This
# is because we must be able to dynamically add and remove vertices and
# faces.

# We always make a copy of the given data:
# - so we control the dtype.
# - we will change the values, avoid surprises by modifying given arrays.
# - we need the first vertex to be empty. ---> not anymore
# - we may want to initialize with some extra size.
#
# Vertex indices are denoted with vi, face indices with fi.

import queue

import numpy as np

from .dynamicmesh import DynamicMesh

# from .maybe_pylinalg import ()
# from .maybe_pygfx import ()
from . import meshfuncs


# todo: better name
class AbstractMesh:
    """Representation of a mesh, with utilities to modify it.
    The idea is that this can be subclassed to hook it up in a visualization
    system (like pygfx), e.g. process updates in a granular way.
    """

    def __init__(self, vertices, faces):
        self._data = DynamicMesh()

        self._props = {}
        self._props_faces = (
            "is_edge_manifold",
            "is_closed",
            "is_oriented",
            "component_labels",
            "is_only_connected_by_edges",
            "nonmanifold_vertices",
        )
        self._props_verts = ()
        self._props_verts_and_faces = "volume", "surface"

        # Delegate initialization
        if vertices is not None or faces is not None:
            self.add_mesh(vertices, faces)

    def _check_prop(self, name):
        assert name in self._props_faces or name in self._props_verts
        if self._props.get("version_faces", 0) != self._data.version_faces:
            self._props["version_faces"] = self._data.version_faces
            for x in self._props_faces + self._props_verts_and_faces:
                self._props.pop(x, None)
        if self._props.get("version_verts", 0) != self._data.version_verts:
            self._props["version_verts"] = self._data.version_verts
            for x in self._props_verts + self._props_verts_and_faces:
                self._props.pop(x, None)
        return name in self._props

    @property
    def faces(self):
        return self._data.faces

    @property
    def vertices(self):
        return self._data.vertices

    @property
    def component_labels(self):
        """A tuple of connected components that this mesh consists of."""
        if not self._check_prop("component_labels"):
            component_labels = meshfuncs.mesh_get_component_labels(
                self._data.faces, self._data.vertex2faces
            )
            self._props["component_labels"] = component_labels
        return self._props["component_labels"]

    @property
    def is_connected(self):
        """Whether the mesh is a single connected component."""
        # Note that connectedness is defined as going via edges, not vertices.
        return self.component_labels.max() == 0

    @property
    def is_edge_manifold(self):
        """Whether the mesh is edge-manifold.

        A mesh being edge-manifold means that each edge is part of
        either 1 or 2 faces. It is one of the two condition for a mesh
        to be manifold.
        """
        if not self._check_prop("is_edge_manifold"):
            nonmanifold_edges = meshfuncs.mesh_get_non_manifold_edges(self._data.faces)
            self._props["nonmanifold_edges"] = nonmanifold_edges
        return len(self._props["nonmanifold_edges"]) == 0

    @property
    def is_vertex_manifold(self):
        """Whether the mesh is vertex-manifold.

        A mesh being vertex-manifold means that for each vertex, the
        faces incident to that vertex form a single (closed or open)
        fan. It is one of the two condition for a mesh to be manifold.

        In contrast to edge-manifoldness, a mesh being non-vertex-manifold,
        can still be closed and oriented.
        """
        if not self._check_prop("nonmanifold_vertices"):
            nonmanifold_vertices = meshfuncs.mesh_get_non_manifold_vertices(
                self._data.faces, self._data.vertex2faces
            )
            self._props["nonmanifold_vertices"] = nonmanifold_vertices
        return self.is_edge_manifold and len(self._props["nonmanifold_vertices"]) == 0

    @property
    def is_manifold(self):
        """Whether the mesh is manifold (both edge- and vertex-manifold)."""
        return self.is_edge_manifold and self.is_vertex_manifold

    @property
    def is_closed(self):
        """Whether the mesh is closed.

        A closed mesh has 2 faces incident to all its edges. This
        implies that the mesh is edge-manifold, and has no boundary
        edges.
        """
        if not self._check_prop("is_closed"):
            _, is_closed = meshfuncs.mesh_is_edge_manifold_and_closed(self._data.faces)
            self._props["is_closed"] = is_closed
        return self._props["is_closed"]

    @property
    def is_oriented(self):
        """Whether the mesh is orientable.

        The mesh being orientable means that the face orientation (i.e.
        winding) is consistent - each two neighbouring faces have the
        same orientation. This can only be true if the mesh is edge-manifold.
        """
        if not self._check_prop("is_oriented"):
            is_oriented = meshfuncs.mesh_is_oriented(self._data.faces)
            self._props["is_oriented"] = is_oriented
        return self._props["is_oriented"]

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

    def get_volume(self):
        """The volume of the mesh.

        CCW winding is assumed. If this is negative, the mesh is
        probably inside-out. If the mesh is not manifold, oriented, and closed,
        this method raises an error.
        """
        if not (self.is_manifold and self.is_oriented and self.is_closed):
            raise RuntimeError(
                "Cannot get volume of a mesh that is not manifold, oriented and closed."
            )
        return meshfuncs.mesh_get_volume(self._data.vertices, self._data.faces)

    def add_mesh(self, vertices, faces):
        """Add vertex and face data.

        The data is copied and the internal data structure.
        """
        faces = np.asarray(faces, np.int32)

        # The DynamicMesh class also does some checks, but it will
        # only check if incoming faces match any vertex, not just the
        # ones we add here, so we perform that check here.
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        vertex_index_offset = len(self._data.vertices)
        self._data.add_vertices(vertices)
        self._data.add_faces(faces + vertex_index_offset)

    # todo: questions:
    # Do we want to keep track of components?
    # Is so, how would we keep track?
    # If you add faces / vertices, would you add them to a specific component?
    #  -> but what if you add stuff making it no longer connected??
    #  -> and what if you add a face that connects two components?
    #  -> this is a more general question: when you modify the mesh, it may be (temporary) non-manifold, un-closed, etc.
    #  -> so even if you mark an added face as belonging to a specific component, we cannot really rely on it?
    # Why would we want to keep track of components? How would we use it?
    # In an UI you could show components in a list. Also e.g. highlight a specific component when selected, or make others transparent.

    # %% Repairs

    def repair_manifold(self):
        """Repair the mesh to maybe make it manifold.

        * Remove collapsed faces.
        * Remove duplicate faces.
        * Remove faces incident to edges that have more than 2 incident faces.

        The repair always produces an edge-manifold mesh but it may
        have less faces (it could even be empty) and the mesh may have
        holes where it previously attached to other parts of the mesh.
        """

        # Remove collapsed faces
        collapsed_faces = np.array(
            [len(set(f)) != len(f) for f in self._data.faces], bool
        )
        (indices,) = np.where(collapsed_faces)
        if len(indices):
            self._data.delete_faces(indices)

        # Remove duplicate faces
        while True:
            sorted_buf = np.frombuffer(np.sort(self._data.faces, axis=1), dtype="V12")
            _, index, counts = np.unique(
                sorted_buf, axis=0, return_index=True, return_counts=True
            )
            indices = index[counts > 1]
            if len(indices):
                self._data.delete_faces(indices)
            else:
                break

        # Remove non-manifold faces
        # todo: maybe the edge-info can be used to stitch the mesh back up?
        self.is_edge_manifold
        nonmanifold_edges = self._props["nonmanifold_edges"]
        indices = []
        for edge, fii in nonmanifold_edges.items():
            indices.extend(fii)
        if len(indices):
            self._data.delete_faces(indices)

    def repair_vertex_manifold(self):
        """Repair vertices that are non-manifold.

        Non-manifold vertices are vertices who's incident faces do not
        form a single (open or closed) fan. It's tricky to find such
        vertices, but it's easy to repair them, once found. The vertices
        are duplicated and assigned to the fans so that the
        vertex-manifold condition is attained.

        The repair can only fail is the mesh is not edge-manifold.
        """

        # Trigger 'nonmanifold_vertices' to be available
        self.is_vertex_manifold

        # Process all required changes
        # We update each group individually. It may be more efficient
        # to collect changes, but it'd also make the code more complex.
        # Note that we can safely do this because no vertices/faces are
        # deleted in this process, so the indices in
        # 'nonmanifold_vertices' remain valid.
        for vi, groups in self._props["nonmanifold_vertices"].items():
            assert len(groups) >= 2
            for face_indices in groups[1:]:
                # Add vertex
                self._data.add_vertices([self._data.vertices[vi]])
                vi2 = len(self._data.vertices) - 1
                # Update faces
                faces = self._data.faces[face_indices, :]
                # faces = faces if faces.base is None else faces.copy()
                faces[faces == vi] = vi2  # todo: must be disallowed!
                self._data.update_faces(face_indices, faces)

    def repair_oriented(self):
        """Repair the winding of individual faces so that it is consistent.

        Returns the number of faces that are reversed.

        The repair can only fail if the mesh is not manifold or when
        it is not orientable (i.e. a Mobius strip or Klein bottle).
        """

        # This implementation walks over the surface using a front. The
        # algorithm is similar to the one for splitting the mesh in
        # connected components, except it does more work at the deepest
        # nesting.
        #
        # It starts out from one face, and reverses the neighboring
        # faces that don't match the winding of the current face. And
        # so on. Faces that have been been processed cannot be reversed
        # again. So the fix operates as a wave that flows over the mesh,
        # with the first face defining the winding.
        #
        # A closed form solution for this problem does not exist. The skcg
        # lib uses pycosat to find the solution. The below might be slower
        # (being implemented in pure Python), but it's free of dependencies
        # and speed matters less in a repair function, I suppose.

        # Make a copy of the faces, so we can reverse them in-place. We'll later update the real faces.
        faces = self._data.faces.copy()

        reversed_faces = []
        vertex2faces = self._data.vertex2faces
        faces_to_check = set(range(len(faces)))

        while len(faces_to_check) > 0:
            # Create new front - once for each connected component in the mesh
            vi_next = faces_to_check.pop()
            front = queue.deque()
            front.append(vi_next)

            # Walk along the front
            while front:
                fi_check = front.popleft()
                vi1, vi2, vi3 = faces[fi_check]
                _, neighbours = meshfuncs.face_get_neighbours2(
                    faces, vertex2faces, fi_check
                )
                for fi in neighbours:
                    if fi in faces_to_check:
                        faces_to_check.remove(fi)
                        front.append(fi)

                        vj1, vj2, vj3 = faces[fi]
                        matching_vertices = {vj1, vj2, vj3} & {vi1, vi2, vi3}
                        if len(matching_vertices) == 2:
                            if vi3 not in matching_vertices:
                                # vi1 in matching_vertices and vi2 in matching_vertices
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
                            elif vi1 not in matching_vertices:
                                # vi2 in matching_vertices and vi3 in matching_vertices
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
                            elif vi2 not in matching_vertices:
                                # vi3 in matching_vertices and vi1 in matching_vertices
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

        if reversed_faces:
            # Get faces to update. If over half was reversed,
            # we take the other half and reverse that instead.
            if len(reversed_faces) < 0.5 * len(faces):
                new_faces = faces[reversed_faces]
            else:
                mask = np.ones(len(faces), np.int32)
                mask[reversed_faces] = 0
                reversed_faces = np.where(mask)[0]
                new_faces = faces[reversed_faces]
                tmp = new_faces[:, 2].copy()
                new_faces[:, 2] = new_faces[:, 1]
                new_faces[:, 1] = tmp
            # Update
            self._data.update_faces(reversed_faces, new_faces)

        # Reverse all the faces if this is an oriented closed manifold with a negative volume.
        if self.is_manifold and self.is_oriented and self.is_closed:
            if self.get_volume() < 0:
                new_faces = self._data.faces.copy()
                tmp = new_faces[:, 2].copy()
                new_faces[:, 2] = new_faces[:, 1]
                new_faces[:, 1] = tmp
                reversed_faces = np.arange(len(new_faces), dtype=np.int32)
                self._data.update_faces(reversed_faces, new_faces)

        return len(reversed_faces)

    def repair_closed(self):
        """Repair holes in the mesh.

        At the moment this only repairs holes of a single face, but this can be improved.
        """
        if not self.is_manifold:
            raise RuntimeError("Repairing open meshes requires them to be manifold.")

        faces = self._data.faces
        new_faces = []

        # Detect boundaries
        boundaries = meshfuncs.mesh_get_boundaries(faces)

        # Now we check all boundaries
        for boundary in boundaries:
            assert len(boundary) >= 3  # I don't think they can be smaller, right?
            if len(boundary) == 3:
                new_faces.append(boundary)
            elif len(boundary) == 4:
                new_faces.append(boundary[:3])
                new_faces.append(boundary[2:] + boundary[:1])
            else:
                pass
                # We can apply the earcut algororithm to fill larger
                # holes as well. Leaving this open for now.

        if new_faces:
            self._data.add_faces(new_faces)

    def deduplicate_vertices(self, *, atol=None):
        """Merge vertices that are the same or close together according to the given tolerance.

        Note that this method can cause the mesh to become non-manifold.
        You should probably only apply this method if you know the
        topology of the mesh, especially if you specify a tolerance.
        """

        faces = self._data.faces.copy()
        vertices = self._data.vertices

        # Collect duplicate vertices
        duplicate_map = {}
        duplicate_mask = np.zeros((len(vertices),), bool)
        for vi in range(len(vertices)):
            if not duplicate_mask[vi]:
                if atol is None:
                    mask3 = vertices == vertices[vi]
                else:
                    mask3 = np.isclose(vertices, vertices[vi], atol=atol)
                mask = mask3.sum(axis=1) == 3  # all positions of a vertex must match
                if mask.sum() > 1:
                    duplicate_mask[mask] = True
                    mask[vi] = False
                    duplicate_map[vi] = np.where(mask)[0]
        # Now we can apply them. Some heavy iterations here ...
        for vi1, vii in duplicate_map.items():
            for vi2 in vii:
                faces[faces == vi2] = vi1

        # Check what faces have been changed in our copy.
        changed = faces != self._data.faces  # Nx3
        changed_count = changed.sum(axis=1)
        (indices,) = np.where(changed_count > 0)

        if len(indices):
            # Update the faces
            self._data.update_faces(indices, faces[indices])
            # We could trace all the vertices that we eventually popped
            # this way, but let's do it the easy (and robust) way.
            self.remove_unused_vertices()

    def remove_unused_vertices(self):
        """Delete vertices that are not used by the faces.

        This is a cleanup step that is safe to apply. Though it should
        not be necessary to call this after doing processing steps -
        these should clean up after themselves (though they could use
        this method for that).
        """
        faces = self._data.faces

        vertices_mask = np.zeros((len(self._data.vertices),), bool)
        vii = np.unique(faces.flatten())
        vertices_mask[vii] = True

        vii_not_used = np.where(~vertices_mask)[0]
        self._data.delete_vertices(vii_not_used)

    def remove_small_components(self, min_faces=4):
        """Remove small connected components from the mesh."""

        # We need the mesh to be manifold to do this
        if not self.is_edge_manifold:
            self.repair_edge_manifold()
        if not self.is_vertex_manifold:
            self.repair_edge_manifold()
        assert self.is_manifold

        # Get labels and their counts
        component_labels = self.component_labels
        labels, counts = np.unique(component_labels, return_counts=True)

        # Determine what faces to remove
        faces_to_remove = []
        for label, count in zip(labels, counts):
            if count < min_faces:
                faces_to_remove.extend(np.where(component_labels == label)[0])

        # Determine what vertices to remove - important to be vertex-manifold!
        vertices_to_remove = np.unique(self._data.faces[faces_to_remove].flatten())

        # check
        for vi in vertices_to_remove:
            fii, _ = np.where(self.faces == vi)
            for fi in fii:
                assert fi in faces_to_remove

        # Apply
        if len(faces_to_remove):
            self._data.delete_faces(faces_to_remove)
            self._data.delete_vertices(vertices_to_remove)

    def split(self):
        """Return a list of Mesh objects, one for each connected component."""
        # I don't think we need this for our purpose, but this class is capable
        # of doing something like this, so it could be a nice util.
        raise NotImplementedError()

    def merge(self, other):
        raise NotImplementedError()

    # %% Walk over the surface

    def get_closest_vertex(self, ref_pos):
        """Get the vertex index closest to the given 3D point, and its distance."""
        ref_pos = np.asarray(ref_pos, np.float32)
        if ref_pos.shape != (3,):
            raise ValueError("ref_pos must be a position (3 values).")

        distances = np.linalg.norm(self.vertices - ref_pos, axis=1)
        vi = np.nanargmin(distances)
        return vi, distances[vi]

    def select_vertices_over_surface(self, ref_vertex, max_distance):
        """Given a reference vertex, select more nearby vertices (over the surface).
        Returns a dict mapping vertex indices to dicts containing {pos, color, sdist, adist}.
        """
        vertices = self.vertices
        vi0 = int(ref_vertex)
        # p0 = vertices[vi0]
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
                        # adist = np.linalg.norm(p2 - p0)
                        # selected_vertices[vi2] = dict(pos=[xn, yn, zn], color=color, sdist=sdist, adist=adist)
                        selected_vertices.add(vi2)
                        vertices2check.append((vi2, sdist))
        return selected_vertices
