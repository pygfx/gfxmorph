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

import numpy as np

from .basedynamicmesh import BaseDynamicMesh

# from .maybe_pylinalg import ()
# from .maybe_pygfx import ()
from . import meshfuncs


class DynamicMesh(BaseDynamicMesh):
    """Representation of a mesh, with utilities to modify it.

    In addition to BaseDynamicMesh, this class adds higher level logic
    to detect certain properties of the mesh, make repairs, and other
    (higher level) modifications.
    """

    def __init__(self, positions, faces):
        super().__init__()
        # Delegate initialization
        if positions is not None or faces is not None:
            self.add_mesh(positions, faces)

    @property
    def component_labels(self):
        """A tuple of connected components that this mesh consists of."""
        cache = self._cache_depending_on_faces
        key = "component_labels"
        if key not in cache:
            cache[key] = meshfuncs.mesh_get_component_labels(
                self.faces, self.vertex2faces
            )
        return cache[key]

    @property
    def component_count(self):
        """The number of components that this mesh consists of."""
        # Note that connectedness is defined as going via edges, not vertices.
        return self.component_labels.max() + 1

    @property
    def is_connected(self):
        """Whether the mesh is a single connected component."""
        return self.component_count == 1

    @property
    def is_edge_manifold(self):
        """Whether the mesh is edge-manifold.

        A mesh being edge-manifold means that each edge is part of
        either 1 or 2 faces. It is one of the two condition for a mesh
        to be manifold.
        """
        cache = self._cache_depending_on_faces
        key = "nonmanifold_edges"
        if key not in cache:
            cache[key] = meshfuncs.mesh_get_non_manifold_edges(self.faces)
        return len(cache[key]) == 0

    @property
    def is_vertex_manifold(self):
        """Whether the mesh is vertex-manifold.

        A mesh being vertex-manifold means that for each vertex, the
        faces incident to that vertex form a single (closed or open)
        fan. It is one of the two condition for a mesh to be manifold.

        In contrast to edge-manifoldness, a mesh being non-vertex-manifold,
        can still be closed and oriented.
        """
        cache = self._cache_depending_on_faces
        key = "nonmanifold_vertices"
        if key not in cache:
            cache[key] = meshfuncs.mesh_get_non_manifold_vertices(
                self.faces, self.vertex2faces
            )
        return self.is_edge_manifold and len(cache[key]) == 0

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
        cache = self._cache_depending_on_faces
        key = "is_closed"
        if key not in cache:
            _, is_closed = meshfuncs.mesh_is_edge_manifold_and_closed(self.faces)
            cache[key] = is_closed
        return cache[key]

    @property
    def is_oriented(self):
        """Whether the mesh is orientable.

        The mesh being orientable means that the face orientation (i.e.
        winding) is consistent - each two neighbouring faces have the
        same orientation. This can only be true if the mesh is edge-manifold.
        """
        cache = self._cache_depending_on_faces
        key = "is_oriented"
        if key not in cache:
            cache[key] = meshfuncs.mesh_is_oriented(self.faces)
        return cache[key]

    @property
    def edges(self):
        """All edges of this mesh as pairs of vertex indices

        Returns
        -------
        ndarray, [n_faces, 3, 2]
            pairs of vertex-indices specifying an edge.
            the ith edge is the edge opposite from the ith vertex of the face

        """
        array = self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        array.setflags(write=False)
        return array

    @property
    def metadata(self):
        """A dict with metadata about the mesh."""
        arrays = (
            self._faces_buf,
            self._positions_buf,
            self._normals_buf,
        )
        nb = sum([a.nbytes for a in arrays if a is not None])
        mem = f"{nb/2**20:0.2f} MiB" if nb > 2**20 else f"{nb/2**10:0.2f} KiB"

        return {
            "is_edge_manifold": self.is_edge_manifold,
            "is_vertex_manifold": self.is_vertex_manifold,
            "is_closed": self.is_closed,
            "is_oriented": self.is_oriented,
            "nfaces": len(self._faces),
            "nvertices": len(self._positions),
            "free_vertices": len(self._positions_buf) - len(self._positions),
            "free_faces": len(self._faces_buf) - len(self._faces),
            "approx_mem": mem,
        }

    # %%

    def get_surface_area(self):
        """Get the surface area of the mesh."""
        return meshfuncs.mesh_get_surface_area(self.positions, self.faces)

    def get_volume(self):
        """Get the volume of the mesh.

        CCW winding is assumed. If this is negative, the mesh is
        probably inside-out. If the mesh is not manifold, oriented, and closed,
        this method raises an error.
        """
        if not (self.is_manifold and self.is_oriented and self.is_closed):
            raise RuntimeError(
                "Cannot get volume of a mesh that is not manifold, oriented and closed."
            )
        return meshfuncs.mesh_get_volume(self.positions, self.faces)

    def add_mesh(self, positions, faces):
        """Add a (partial) mesh.

        The vertices and faces are appended to the end. The values of
        the faces are modified to still target the appropriate vertices
        (which may now have an offset).
        """
        faces = np.asarray(faces, np.int32)

        # The DynamicMesh class also does some checks, but it will
        # only check if incoming faces match any vertex, not just the
        # ones we add here, so we perform that check here.
        if faces.min() < 0 or faces.max() >= len(positions):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        vertex_index_offset = len(self._positions)
        self.add_vertices(positions)
        self.add_faces(faces + vertex_index_offset)

    # %% Repairs

    def repair(self, close=False):
        """Perform various repairs to the mesh.

        If close is given and True, also try to close the mesh. The
        resulting mesh is guaranteed to be manifold, but may not be
        closed. It will be oriented if the topology allows it (e.g. not
        a Klein bottle).
        """
        self.repair_manifold()
        if close:
            self.repair_touching_boundaries()
            self.repair_holes()
        self.repair_orientation()
        self.remove_unused_vertices()

    def repair_manifold(self):
        """Repair the mesh to make it manifold.

        This method includes a number of steps:

        * Remove collapsed faces.
        * Remove duplicate faces.
        * Remove faces incident to edges that have more than 2 incident faces.
        * Duplicate non-manifold vertices and assign them to the respective faces.

        The result is always a manifold mesh, but it may have less faces
        (it could even be empty) and the mesh may have holes where it
        previously attached to other parts of the mesh.

        Returns the number of deleted/updated faces.
        """
        n_updated = 0

        if self.is_manifold:
            return n_updated

        # Remove collapsed faces. A collapsed face results in the mesh
        # being either not vertex- or not edge- manifold, depending on
        # whether it is at a boundary.
        collapsed_faces = np.array([len(set(f)) != len(f) for f in self.faces], bool)
        (indices,) = np.where(collapsed_faces)
        if len(indices):
            self.delete_faces(indices)
            n_updated += len(indices)

        # Remove duplicate faces.
        sorted_buf = np.frombuffer(np.sort(self.faces, axis=1), dtype="V12")
        unique_buf, counts = np.unique(sorted_buf, axis=0, return_counts=True)
        duplicate_values = unique_buf[counts > 1]
        indices = []
        for value in duplicate_values:
            (indices_for_value,) = np.where(sorted_buf == value)
            indices.extend(indices_for_value[1:])
        if len(indices):
            self.delete_faces(indices)
            n_updated += len(indices)

        # Remove non-manifold edges.
        # Use the is_edge_manifold prop to trigger 'nonmanifold_edges' to be up to date.
        if not self.is_edge_manifold:
            nonmanifold_edges = self._cache_depending_on_faces["nonmanifold_edges"]
            indices = []
            for edge, fii in nonmanifold_edges.items():
                indices.extend(fii)
            if len(indices):
                self.delete_faces(indices)
                n_updated += len(indices)

        # Fix non-manifold vertices.
        # Non-manifold vertices are vertices who's incident faces do not
        # form a single (open or closed) fan. It's tricky to find such
        # vertices, but it's easy to repair them, once found. The vertices
        # are duplicated and assigned to the respective fans.
        # Use the is_vertex_manifold prop to trigger 'nonmanifold_edges' to be up to date.
        if not self.is_vertex_manifold:
            # We update each group individually. It may be more efficient
            # to collect changes, but it'd also make the code more complex.
            # Note that we can safely do this because no vertices/faces are
            # deleted in this process, so the indices in
            # 'nonmanifold_vertices' remain valid.
            nonmanifold_vertices = self._cache_depending_on_faces[
                "nonmanifold_vertices"
            ]
            for vi, groups in nonmanifold_vertices.items():
                assert len(groups) >= 2
                for face_indices in groups[1:]:
                    # Add vertex
                    self.add_vertices([self._positions[vi]])
                    vi2 = len(self._positions) - 1
                    # Update faces
                    faces = self.faces[face_indices, :]
                    faces[faces == vi] = vi2
                    self.update_faces(face_indices, faces)
                    n_updated += len(face_indices)

        return n_updated

    def repair_orientation(self):
        """Repair the winding of individual faces to make the mesh oriented.

        Faces that do not match the winding of their neighbours are
        flipped in a recursive algorithm. If the mesh is a close and
        oriented manifold, but it has a negative volume, all faces are
        flipped.

        The repair can only fail if the mesh is not manifold or when
        it is not orientable (i.e. a Mobius strip or Klein bottle).

        Returns the number of faces that are flipped.
        """
        n_flipped = 0

        if not self.is_oriented:
            # Try making the winding consistent
            modified_faces = meshfuncs.mesh_get_consistent_face_orientation(
                self.faces, self.vertex2faces
            )
            (indices,) = np.where(modified_faces[:, 2] != self.faces[:, 2])
            if len(indices) > 0:
                self.update_faces(indices, modified_faces[indices])
            n_flipped = len(indices)

        # Reverse all the faces if this is an oriented closed manifold with a negative volume.
        if self.is_manifold and self.is_oriented and self.is_closed:
            if self.get_volume() < 0:
                new_faces = self.faces.copy()
                tmp = new_faces[:, 2].copy()
                new_faces[:, 2] = new_faces[:, 1]
                new_faces[:, 1] = tmp
                indices = np.arange(len(new_faces), dtype=np.int32)
                self.update_faces(indices, new_faces)
                n_flipped = len(new_faces)

        return n_flipped

    def repair_touching_boundaries(self, *, atol=1e-5):
        """Repair open meshes by stitching boundary vertices that are close together.

        Vertices (on boundaries) that are the same or close together
        (according to the given tolerance) are de-duplicated, thereby
        stitching the mesh parts together. The purpose is for meshes
        that are visually closed but mathematically open, to become
        mathematically closed.

        There is no guarantee that this results in a closed mesh,
        because that depends entirely on the presence of near-touching
        boundaries. If the stitching of a group of boundaries would
        result in a non-manifold mesh, it is skipped. (So it should be
        safe to call this method.)

        Returns the number of updated faces.
        """

        if self.is_closed:
            return 0

        # Stitch, getting a copy of the faces
        faces = meshfuncs.mesh_stitch_boundaries(self.positions, self.faces, atol=atol)

        # Check what faces have been changed in our copy.
        changed = faces != self.faces  # Nx3
        changed_count = changed.sum(axis=1)
        (indices,) = np.where(changed_count > 0)

        if len(indices) == 0:
            return 0

        # Update the faces
        self.update_faces(indices, faces[indices])

        # Clean up collapsed faces
        collapsed_faces = np.array([len(set(f)) != len(f) for f in self.faces], bool)
        if np.any(collapsed_faces):
            self.delete_faces(np.where(collapsed_faces)[0])

        # Clean up any vertices that are no longer in use
        self.remove_unused_vertices()

        return len(indices)

    def repair_holes(self):
        """Repair holes in the mesh.

        Small boundaries are removed by filling these holes with new faces.

        At the moment this only repairs holes of 3 or 4 vertices (i.e.
        1  or 2 faces), but this can later be improved. So if only small
        holes are present, the result will be a closed mesh. However,
        if the mesh is not manifold, this method may not be able to
        repair all holes. Also note that e.g. the four courners of a
        rectangular surface would be connected with new faces.

        Returns the number of added faces.
        """

        if self.is_closed:
            return 0

        # Detect boundaries.
        try:
            boundaries = meshfuncs.mesh_get_boundaries(self.faces)
        except RuntimeError:
            # The mesh probably has non-manifold edges/vertices near the boundaries,
            # causing the algorithm in `mesh_get_boundaries()` to fail.
            return 0

        # Now we check all boundaries
        new_faces = []
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
            self.add_faces(new_faces)

        return len(new_faces)

    def remove_unused_vertices(self):
        """Delete vertices that are not used by the faces.

        This is a cleanup step that is safe to apply. Though it should
        not be necessary to call this after doing processing steps -
        these should clean up after themselves (though they could use
        this method for that).
        """
        faces = self.faces

        vertices_mask = np.zeros((len(self.positions),), bool)
        vii = np.unique(faces.flatten())
        vertices_mask[vii] = True

        indices = np.where(~vertices_mask)[0]
        if len(indices) > 0:
            self.delete_vertices(indices)

    def remove_small_components(self, min_faces=4):
        """Remove small connected components from the mesh."""

        # We need the mesh to be manifold to do this
        self.repair_manifold()
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
        vertices_to_remove = np.unique(self.faces[faces_to_remove].flatten())

        # check
        for vi in vertices_to_remove:
            fii, _ = np.where(self.faces == vi)
            for fi in fii:
                assert fi in faces_to_remove

        # Apply
        if len(faces_to_remove):
            self.delete_faces(faces_to_remove)
            self.delete_vertices(vertices_to_remove)

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

        distances = np.linalg.norm(self.positions - ref_pos, axis=1)
        vi = np.nanargmin(distances)
        return vi, distances[vi]

    def select_vertices_over_surface(self, ref_vertices, ref_distances, max_distance):
        """Select nearby vertices, starting from the given reference vertices.

        Walks over the surface from the reference vertices to include
        more vertices until the distance to a vertex (by walking over
        the edges) exceeds the max_distance. Each reference vertex is
        also associated with a starting distance.

        By allowing multiple reference vertices it is possible to "grab
        the mesh" on a precise point within a face, or using specific
        shapes (e.g. a line piece) to grab the mesh.

        Returns xxxx (maybe also include sdists).
        """

        # Init
        positions = self.positions
        faces = self.faces
        vertex2faces = self.vertex2faces

        # Allow singleton use
        if isinstance(ref_vertices, (int, np.int32, np.int64)):
            ref_vertices = [ref_vertices]
        if isinstance(ref_distances, (float, int, np.float32, np.float64)):
            ref_distances = [ref_distances]

        # The list of vertices to check for neighbours
        vertices2check = []
        selected_vertices = {}
        for vi, sdist in zip(ref_vertices, ref_distances):
            vertices2check.append((vi, sdist))
            selected_vertices[vi] = sdist

        # Walk over the surface
        while len(vertices2check) > 0:
            vi1, cumdist = vertices2check.pop(0)
            p1 = positions[vi1]
            for vi2 in meshfuncs.vertex_get_neighbours(faces, vertex2faces, vi1):
                p2 = positions[vi2]
                sdist = cumdist + np.linalg.norm(p2 - p1)
                if sdist < max_distance:
                    # We will have a closer look if we have not yet selected this
                    # vertex, but also if we did but found a shorter route to it.
                    # This means that we may sometimes do duplicate work. To avoid
                    # this, we'd need a binary heap to store the front.
                    if vi2 not in selected_vertices or sdist < selected_vertices[vi2]:
                        selected_vertices[vi2] = sdist
                        vertices2check.append((vi2, sdist))

        # Return as ndarray of indices
        return np.array(sorted(selected_vertices.keys()), np.int32)
