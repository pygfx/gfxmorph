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

import heapq

import numpy as np

from .basedynamicmesh import BaseDynamicMesh
from .utils import logger

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
        if len(faces) > 0:
            if faces.min() < 0 or faces.max() >= len(positions):
                raise ValueError(
                    "The faces array containes indices that are out of bounds."
                )

        vertex_index_offset = len(self._positions)
        self.add_vertices(positions)
        self.add_faces(faces + vertex_index_offset)

    # %% Low level modifications

    def resample(self, reference_distance):
        raise NotImplementedError()

    def resample_selection(self, vertex_indices, weights, reference_distance):
        """Resample the selected region of the mesh.

        Edges that are too long are split, and edges that are too short
        are resolved by removing a vertex and filling up the hole.

        Note: although this code tries to prevent the mesh from becoming
        non-manifold, it's hard to foresee all edge-cases (get it?
        *edge* cases?), and it's probably a good idea to test for this
        and cancel the change if this happens.

        """

        # todo: I've observed this algorithm to sometimes produces a non-manifold mesh for small components.

        positions = self.positions
        faces = self.faces
        vertex2faces = self.vertex2faces

        # Given the reference_distance, we must define dmin and dmax.
        # when the length of an edge is above dmax, we split it in two.
        # When the length of an edge is below dmin, we remove it.
        #
        # On first sight, the ratio between dmin and dmax should be
        # slightly higher than 2 to avoid constant resampling.
        #
        # However, splitting an edge in two also creates new edges. In
        # the below image, consider adding the bottom-center vertex (on
        # the edge between de two vertices on each side). This also
        # introduces a new edge between the top-center and bottom-center
        # vertices, which is quite short. Assuming a ratio of 1.5
        # between shortest and longest edge of an average triangle, we
        # can employ a ratio between dmin and dmax of about 3.
        #
        #       _ - X - _
        #   _ -     1     - _
        # X---------X---------X
        #
        # Note that the loosen factor (depending on weight) helps
        # prevent near-degenerate triangles that can normally can occur
        # near the boundary of the selection.

        # Note: A method that we could have used but don't, is to pop
        # an edge by merging the two vertices, and position the new
        # vertex in between the original positions. This is a relatively
        # simple approach, and less invasive than smashing a a hole in
        # the mesh by removing a vertex. Unfortunately, we also have
        # less control over this change, and it more quickly results
        # into folded faces.

        # Calculate dmin and dmax such that: dmax = dmin * dmin_dmax_ratio
        dmin_dmax_ratio = 3
        dmin = reference_distance / dmin_dmax_ratio**0.5
        dmax = reference_distance * dmin_dmax_ratio**0.5

        # Collect edges
        edges = {}  # (vi1, vi2) -> weight
        for vi1, w in zip(vertex_indices, weights):
            vi1 = int(vi1)
            for vi2 in meshfuncs.vertex_get_neighbours(faces, vertex2faces, vi1):
                key = (vi1, vi2) if vi1 < vi2 else (vi2, vi1)
                edges[key] = edges.get(key, 0) + 0.5 * w

        # Determine whether they are too short/long
        too_short = []
        too_long = []
        for vii, w in edges.items():
            d = np.linalg.norm(positions[vii[0]] - positions[vii[1]])
            loosen = 1 + 2 * min(max(w, 0), 1)
            if d > loosen * dmax:
                too_long.append((d / loosen, *vii))
            elif d < dmin / loosen:
                too_short.append((d * loosen, *vii))

        # Sort so that the most extreme cases get handled first
        too_long.sort(key=lambda x: -x[0])
        too_short.sort(key=lambda x: x[0])

        # Next, we take a few steps to determine what edges we split,
        # and what vertices we will pop. In this process, we keep track
        # of the faces that effected by the planned changes so far.
        # This is to prevent changes to clash and create a corrupt mesh.
        #
        # This means that some edges which are too short or too long
        # are not handled in this call. Though the worst cases are
        # prioritized. It is somewhat assumed that this method is called
        # in an interactive setting. If necessary, a new subset can be
        # selected and resampled.
        #
        # Another solution is to apply each change separetely, but this
        # results in relatively large undo stacks.
        affected_faces = set()

        # Determine vertices incident to edges that are too long.
        edges_to_split = []
        for d, vi1, vi2 in too_long:
            touched_faces = set(vertex2faces[vi1]) & set(vertex2faces[vi2])
            if not (touched_faces & affected_faces):
                affected_faces.update(touched_faces)
                edges_to_split.append((vi1, vi2))

        # Determine vertices that connect multiple edges that are too short.
        too_short_vertices = {}
        for d, vi1, vi2 in too_short:
            too_short_vertices[vi1] = too_short_vertices.get(vi1, 0) + 1
            too_short_vertices[vi2] = too_short_vertices.get(vi2, 0) + 1
        hot_vertices = [
            (count, vi) for vi, count in too_short_vertices.items() if count > 1
        ]
        hot_vertices.sort(reverse=True)

        # We will pop the ones that we can
        vertices_to_pop = []
        for _, vi in hot_vertices:
            touched_faces = set(vertex2faces[vi])
            if not (touched_faces & affected_faces):
                affected_faces.update(touched_faces)
                vertices_to_pop.append(vi)

        # And we also pop one vertex of the other too-short-edges, preferring
        # the ones with the most neighbours, to avoid star-like structures.
        for d, vi1, vi2 in too_short:
            if vi1 in vertices_to_pop or vi2 in vertices_to_pop:
                continue
            vii = vi1, vi2
            if len(vertex2faces[vi1]) < len(vertex2faces[vi2]):
                vii = vi2, vi1
            for vi in vii:
                touched_faces = set(vertex2faces[vi])
                if not (touched_faces & affected_faces):
                    affected_faces.update(touched_faces)
                    vertices_to_pop.append(vi)
                    break

        # Next we collect the changes, so we can apply them in one go...
        vertices_to_remove = []
        vertices_to_add = []
        faces_to_remove = []
        faces_to_add = []

        # Changes to make the mesh more detailed
        for vi1, vi2 in edges_to_split:
            new_index = len(self.positions) + len(vertices_to_add)
            a_verts, r_faces, a_faces = self._add_vertex_on_edge(vi1, vi2, new_index)
            vertices_to_add.extend(a_verts)
            faces_to_remove.extend(r_faces)
            faces_to_add.extend(a_faces)

        # Changes to make the mesh more coarse
        for vi in vertices_to_pop:
            try:
                r_verts, r_faces, a_faces = self._pop_vertex(vi)
            except RuntimeError as err:
                logger.warn(str(err))
                continue
            vertices_to_remove.extend(r_verts)
            faces_to_remove.extend(r_faces)
            faces_to_add.extend(a_faces)

        # Apply changes
        if len(vertices_to_add) > 0:
            self.add_vertices(vertices_to_add)
        self.delete_and_add_faces(faces_to_remove, faces_to_add)
        if len(vertices_to_remove) > 0:
            self.delete_vertices(vertices_to_remove)

    def add_vertex_on_edge(self, vi1, vi2):
        """Add a vertex on the give edge.

        This splits the edge between vi1 and vi2 into two edges, and
        splits the two faces that contain that edge. Results in 1
        additional vertex, 2 additional faces, 3 additional edges.
        """

        # Do the algorithmic
        vertices_to_add, faces_to_remove, faces_to_add = self._add_vertex_on_edge(
            vi1, vi2, len(self.positions)
        )

        # Apply changes
        if len(vertices_to_add) > 0:
            self.add_vertices(vertices_to_add)
        if len(faces_to_add) > 0:
            self.add_faces(faces_to_add)
        if len(faces_to_remove) > 0:
            self.delete_faces(faces_to_remove)

    def _add_vertex_on_edge(self, vi1, vi2, new_index):
        # This code:
        #
        # - place a vertex on the given edge
        # - removes the faces incident to that edge
        # - creates new faces incident to the new vertex
        #
        # X______          X______
        # |\     |         |\    /|
        # | \    |         | \  / |
        # |  \   |   -->   |  \/  |
        # |   \  |         |  /\  |
        # |    \ |         | /  \ |
        # |_____\|         |/____\|
        #        X                X

        positions = self.positions
        normals = self.normals
        faces = self.faces
        vertex2faces = self.vertex2faces

        # Get what faces to split in two
        faces1 = vertex2faces[vi1]
        faces2 = vertex2faces[vi2]
        faces2split = list(set(faces1).intersection(faces2))
        if len(faces2split) == 0:
            raise ValueError("Given vertices do not make an edge.")
        elif len(faces2split) > 2:
            raise ValueError("Cannot split edge on a non-manifold piece of the mesh.")

        # Position it right in between
        p1, p2 = positions[vi1], positions[vi2]
        new_position = 0.5 * (p1 + p2)

        # Add contribution for curvature ...
        # In practice there is very little difference, and when
        # interactively morphing and smoothing, we should perhaps not
        # not make things too fancy.
        if False:
            n1, n2 = normals[vi1], normals[vi2]
            dist = np.linalg.norm(p1 - p2)
            # Get orthogonal vector
            dir = p2 - p1
            dir /= np.linalg.norm(dir)
            ort = np.cross(n1, dir) + np.cross(n2, dir)
            ort /= np.linalg.norm(ort)
            # Get directional vectors
            dir1 = -np.cross(n1, ort)
            dir2 = +np.cross(n2, ort)
            dir1 /= np.linalg.norm(dir1)
            dir2 /= np.linalg.norm(dir2)
            # Add contribution for surface curvature
            new_position += 0.125 * dist * (dir1 + dir2)

        # Calculate new faces. Needs a bit of triage to get the winding correct.
        vi3 = new_index
        new_faces = []
        for fi in faces2split:
            face1 = faces[fi].tolist()
            if face1[0] == vi1:
                if face1[1] == vi2:
                    face1[1] = vi3
                else:
                    face1[2] = vi3
                face2 = vi2, face1[2], face1[1]
            elif face1[1] == vi1:
                if face1[0] == vi2:
                    face1[0] = vi3
                else:
                    face1[2] = vi3
                face2 = face1[2], vi2, face1[0]
            else:  # face1[2] == vi1:
                if face1[0] == vi2:
                    face1[0] = vi3
                else:
                    face1[1] = vi3
                face2 = face1[1], face1[0], vi2
            new_faces.append(face1)
            new_faces.append(face2)

        return [new_position], faces2split, new_faces

    # todo: naming things .. we already have pop_vertices!

    def pop_vertex(self, vi, delete_vertex=True):
        """Remove the given vertex.

        Removing the vertex creates a hole, which is refilled with new
        faces. Results in 1 less vertex, 2 less faces, 3 less edges.

        All faces that contain vi are removed, forming a hole. The
        boundary of this hole is formed by the (former) direct
        neightbours of vi. This boundary (i.e. polygon) is re-tessalated
        using an algorithm that makes use of the two-ears-theorem,
        prioritizing nice (not-elongated) faces.
        """

        # Do the algorithmic
        try:
            vertices_to_remove, faces_to_remove, faces_to_add = self._pop_vertex(vi)
        except RuntimeError as err:
            logger.warn(str(err))
            return

        # Apply changes
        if len(faces_to_add) > 0:
            self.add_faces(faces_to_add)
        if len(faces_to_remove) > 0:
            self.delete_faces(faces_to_remove)
        if len(vertices_to_remove) > 0:
            self.delete_vertices(vertices_to_remove)

    def _pop_vertex(self, vi):
        # This code:
        #
        # - Obtains the faces incident to the vertex to remove.
        # - We calculate how removing these faces affects the neighborhood of the component.
        # - In some cases this code will return early.
        # - If the vertex is on a boundary, we can just remove the vertex and incident faces.
        # - Otherwise, we calculate the  boundary of the hole.
        # - This boundary is tesselated. This is the hard part.

        # This method contains a few checks that raise a runtime error.
        # Some of these cases can occur in particular topologies. Others cannot
        # in theory, but that's what I thought of the aforementioned cases too,
        # so let's assume nothing. Also, if the mesh is not manifold, some of these
        # cases may be triggered.

        positions = self.positions
        faces = self.faces
        vertex2faces = self.vertex2faces

        vi_to_remove = int(vi)
        vertices_to_remove = [vi_to_remove]
        faces_to_remove = list(vertex2faces[vi_to_remove])

        # Compute the context_faces: the faces in this component that border the ones we remove
        vii = set()
        for fi in faces_to_remove:
            vii.update(faces[fi])
        nearby_faces = set()
        for vi in vii:
            nearby_faces.update(vertex2faces[vi])
        context_faces = nearby_faces - set(faces_to_remove)

        # If no faces remain in this component, we dont pop the vertex
        if not context_faces:
            return [], [], []

        # Some cases are easy ...
        if not faces_to_remove:
            # This is a standalone vertex
            return vertices_to_remove, [], []
        elif len(faces_to_remove) <= 2:
            # This vertex is on a boundary, we can just remove the faces!
            return vertices_to_remove, faces_to_remove, []

        # Otherwise, removing the faces creates a hole that we must tesselate ...

        # If we assume the mesh to be closed, having just one context
        # face means that we have a tetrahedron which we dont want to
        # reduce. This rule means that if (this component of) the mesh
        # is not closed, we cannot reduce components further than 4
        # faces, but that's probably fine (deteting this case is
        # relatively expensive).
        if len(context_faces) <= 1:
            return [], [], []

        # Collect the boundary vertices, store as a directed graph
        boundary_map = {}  # vi -> vi
        for fi in faces_to_remove:
            vi1, vi2, vi3 = faces[fi]
            if vi1 == vi_to_remove:
                boundary_map[vi2] = vi3
            elif vi2 == vi_to_remove:
                boundary_map[vi3] = vi1
            elif vi3 == vi_to_remove:
                boundary_map[vi1] = vi2
            else:
                raise RuntimeError("Unexpected face winding for contour?")

        # Make a boundary list, with consistent winding
        vi = next(iter(boundary_map))  # start anywhere (should not matter)
        boundary_list = [vi]
        while len(boundary_list) < len(boundary_map):
            vi = boundary_map.get(vi, -1)
            boundary_list.append(vi)
            if vi < 0:
                raise RuntimeError("Unexpected error while tracing the boundary.")

        # Must be circular. Note that the second case *can* indeed occur.
        if len(boundary_list) < 3:
            raise RuntimeError("Unexpected boundary too short.")
        if boundary_map[boundary_list[-1]] != boundary_list[0]:
            raise RuntimeError("Unexpected boundary not circular or has a shortcut.")

        # Get tesselated faces
        new_faces = meshfuncs.mesh_fill_hole(
            positions, faces, vertex2faces, boundary_list
        )
        if len(new_faces) != len(faces_to_remove) - 2:
            raise RuntimeError("Unexpected tesselation result.")

        return vertices_to_remove, faces_to_remove, new_faces

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

        Boundaries are removed by filling these holes with new faces.

        If the mesh is not manifold, this method may not be able to
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
        new_faces_list = []
        for boundary in boundaries:
            assert len(boundary) >= 3  # I don't think they can be smaller, right?
            new_faces = meshfuncs.mesh_fill_hole(
                self.positions, self.faces, self.vertex2faces, boundary
            )
            new_faces_list.append(new_faces)

        # Apply
        if new_faces_list:
            new_faces = np.concatenate(new_faces_list)
            self.add_faces(new_faces)
            return len(new_faces)
        else:
            return 0

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

    def select_vertices_over_surface(
        self, ref_vertices, ref_distances, max_distance, distance_measure="auto"
    ):
        """Select nearby vertices, starting from the given reference vertices.

        Walks over the surface from the reference vertices to include
        more vertices until the distance to a vertex (by walking over
        the surface) exceeds the max_distance. Each reference vertex is
        also associated with a starting distance.

        By allowing multiple reference vertices it is possible to "grab
        the mesh" on a precise point within a face, or using specific
        shapes (e.g. a line piece) to grab the mesh.

        Parameters
        ----------
        ref_vertices : int or list or ndarray
            A single vertex index, or a set of vertex indices, to start
            the selection from.
        ref_distances : float or list or ndarray
            The initial distance, or distances, corresponding to the
            reference vertices.
        max_distance : float
            The maximum (geodesic) distance that a vertex can have to
            be included in the selection.
        distance_measure : str
            The method to calculate the geodesic distance. With "edge"
            it sums the edge lengths, with "smooth1" it smooths the
            path to compensate for zig-zag patterns, with "smooth2" it
            does this smarter to avoid deviating from the surface. With
            "auto" it starts with smooth2, and falls back to "edge" halfway.
            Default "auto".

        Returns
        -------
        vertices : ndarray
            The selected vertex indices.
        distances : ndarray
            The corresponding (geodesic) distances.
        """

        # Init
        positions = self.positions
        normals = self.normals
        faces = self.faces
        vertex2faces = self.vertex2faces

        # Allow singleton use
        if isinstance(ref_vertices, (int, np.int32, np.int64)):
            ref_vertices = [ref_vertices]
        if isinstance(ref_distances, (float, int, np.float32, np.float64)):
            ref_distances = [ref_distances]

        # Select path class
        distance_measure = distance_measure or "smooth2"
        defer_to_edge_measure = False
        if distance_measure == "edge":
            MeshPath = MeshPathEdge  # noqa
        elif distance_measure == "smooth1":
            MeshPath = MeshPathSmooth1  # noqa
        elif distance_measure == "smooth2":
            MeshPath = MeshPathSmooth2  # noqa
        elif distance_measure == "auto":
            MeshPath = MeshPathSmooth2  # noqa
            defer_to_edge_measure = True
        else:
            raise ValueError(
                "The distance_measure arg must be 'edge' 'smooth1', 'smooth2' or 'auto'."
            )

        # The list of vertices to check for neighbours.
        vertices2check = []  # binary heap
        selected_vertices = {}
        for vi, dist in zip(ref_vertices, ref_distances):
            path = MeshPath(dist).add(positions[vi], normals[vi])
            heapq.heappush(vertices2check, (path, vi))

        # Walk over the surface
        while vertices2check:
            path1, vi1 = heapq.heappop(vertices2check)
            if vi1 in selected_vertices:
                continue
            selected_vertices[vi1] = path1.dist
            if defer_to_edge_measure and path1.dist > 0.5 * max_distance:
                MeshPath = MeshPathEdge  # noqa
            for vi2 in meshfuncs.vertex_get_neighbours(faces, vertex2faces, vi1):
                if vi2 in selected_vertices:
                    continue
                path2 = path1.add(positions[vi2], normals[vi2], MeshPath)
                if path2.dist < max_distance:
                    heapq.heappush(vertices2check, (path2, vi2))

        vertices = np.array(sorted(selected_vertices.keys()), np.int32)
        distances = np.array([selected_vertices[vi] for vi in vertices], "f4")
        return vertices, distances


class BaseMeshPath:
    """Base class to help calculate the geodesic distance of a path over a surface."""

    __slots__ = ["positions", "normals", "edist", "dist"]

    mode = ""

    def __init__(self, initial_dist=0.0):
        self.positions = []
        self.normals = []
        self.edist = initial_dist  # edge distance
        self.dist = initial_dist  # the processed distance

    def __lt__(self, other):
        return self.dist < other.dist

    def add(self, p, n, cls=None):
        cls = cls or self.__class__
        new = cls()
        new.positions = self.positions[-3:] + [p]
        new.normals = self.normals[-3:] + [n]
        new.edist = self.edist
        new.dist = self.dist
        d = np.linalg.norm(self.positions[-1] - p) if self.positions else 0
        new._process_new_position(d)
        return new

    def _process_new_position(self, d):
        self.edist += d


class MeshPathEdge(BaseMeshPath):
    """Calculate the geodesic distance by simply summing the lengths
    of edges that the path goes over.
    """

    __slots__ = []

    def _process_new_position(self, d):
        self.edist += d
        self.dist += d


class MeshPathSmooth1(BaseMeshPath):
    """Calculate the geodesic distance by (virtually) repositioning
    visited points over the surface. Simply by putting it in between
    the neighboring points. Simple, but it (quite literally) cuts corners.
    """

    __slots__ = []

    def _process_new_position(self, d):
        self.edist += d
        self.dist += d
        if len(self.positions) >= 3:
            p, delta_dist = self._refine_position(
                *self.positions[-3:], *self.normals[-3:]
            )
            self.positions[-2] = p
            self.dist += delta_dist

    def _refine_position(self, p1, p2, p3, n1, n2, n3):
        # Get current distance
        dist1 = np.linalg.norm(p1 - p2)
        dist3 = np.linalg.norm(p2 - p3)
        dist_before = dist1 + dist3
        if dist1 == 0 or dist3 == 0:
            return p2, 0

        # Get the point on the line between p1 and p3, but positioned relatively
        f1 = 1 - dist1 / dist_before
        f3 = 1 - dist3 / dist_before
        assert 0.999 < (f1 + f3) < 1.001
        p = f1 * p1 + f3 * p3

        # Calculate new distance
        dist_after = np.linalg.norm(p1 - p) + np.linalg.norm(p - p3)
        delta_dist = dist_after - dist_before

        return p, delta_dist


class MeshPathSmooth2(MeshPathSmooth1):
    """Calculate the geodesic distance by (virtually) repositioning
    visited points over the surface. This is still a pretty crude
    estimate of the "real" geodesic distance, but quite a bit more
    precise than simply summing the edge lenghts.
    """

    __slots__ = []

    def _refine_position(self, p1, p2, p3, n1, n2, n3):
        norm = np.linalg.norm

        # Get current distance
        dist1 = norm(p1 - p2)
        dist3 = norm(p2 - p3)
        dist_before = dist1 + dist3
        if dist1 == 0 or dist3 == 0:
            return p2, 0

        # Normalize the normal, make sure its nonzero
        n2_len = norm(n2)
        if n2_len == 0:
            return p2, 0
        n2 = n2 / n2_len

        # Get the point on the line between p1 and p3, but positioned relatively
        f1 = 1 - dist1 / dist_before
        f3 = 1 - dist3 / dist_before
        assert 0.999 < (f1 + f3) < 1.001
        p_between = f1 * p1 + f3 * p3

        # Define a plane through p2, with a normal that is a combination.
        # Just using n2 does not seem right, since we move the point towards
        # p1 and p3, it makes sense to include contributions of these normals.
        plane_pos = p2
        plane_normal = n1 + n2 + n2 + n3
        length = norm(plane_normal)
        plane_normal = n2 if length == 0 else plane_normal / length

        # Project the point on the surface. The surface is estimated using the plane avove.
        p = p_between - np.dot(plane_normal, (p_between - plane_pos)) * plane_normal

        # Calculate new distance
        dist_after = norm(p1 - p) + norm(p - p3)
        delta_dist = dist_after - dist_before

        return p, delta_dist
