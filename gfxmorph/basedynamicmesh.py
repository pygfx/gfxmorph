import traceback
import logging

import numpy as np


logger = logging.getLogger("meshmorph")

EXCEPTION_IN_ATOMIC_CODE = "Unexpected exception in code that is considered atomic!"


# todo: test the undo mechanics in unit tests (maybe as part of test_corrupt)


class MeshChangeTracker:
    """A class that is notified of changes to a BaseDynamicMesh.

    To use, this class must be subclassed, implementing (a subset of)
    its methods. The instance of the subclass can then be subscribed
    to a ``BaseDynamicMesh``object to process its changes.

    Use-case for tracking mesh changes include e.g. logging, updating
    a GPU representation of the mesh, or keeping an undo stack.
    """

    # --- API that is the same as changes to BaseDynamicMesh, except for the `old` argument

    def clear(self):
        pass

    def add_faces(self, faces):
        pass

    def pop_faces(self, n, old):
        pass

    def swap_faces(self, indices1, indices2):
        pass

    def update_faces(self, indices, faces, old):
        pass

    def add_vertices(self, positions):
        pass

    def pop_vertices(self, n, old):
        pass

    def swap_vertices(self, indices1, indices2):
        pass

    def update_vertices(self, indices, positions, old):
        pass

    # --- Extra API for e.g. GPU viz

    def set_vertex_arrays(self, positions, normals, colors):
        pass

    def set_face_array(self, array):
        pass

    def update_normals(self, indices):
        pass


class Safecall:
    """Context manager for doing calls that should not raise. If an
    exception is raised, it is caught, logged, and the context exits
    normally.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            lines = traceback.format_exception(exc_type, exc_val, exc_tb)
            msg = "Exception in update callback:\n" + "".join(lines)
            logger.error(msg)
        return True  # Yes, we handled the exception


# todo: The order of methods on this class can be better. Let's fix that in a separate pr.


class BaseDynamicMesh:
    """A mesh object that can be modified in-place.

    It manages buffers that are oversized so the vertex and face array
    can grow. When the buffer is full, a larger buffer is allocated.
    The exposed arrays are contiguous views onto these buffers.

    It also maintains a vertex2faces incidence map, and keeps the vertex
    normals up to date. It can notify other object of changes, so that
    any representation of the mesh  (e.g. a visualization on the GPU)
    can be kept in sync.
    """

    # We assume meshes with triangles (not quads).
    # Note that there are a few places where loops are unrolled, and verts_per_face is thus hardcoded.
    _verts_per_face = 3

    # In debug mode, the mesh checks its internal state after each change
    _debug_mode = False

    def __init__(self):
        # Caches that subclasses can use to cache stuff. When the
        # vertices/faces change, the respective caches are cleared.
        self._cache_depending_on_verts = {}
        self._cache_depending_on_faces = {}
        self._cache_depending_on_verts_and_faces = {}
        # A list of trackers that are notified of changes.
        self._change_trackers = []
        # Init!
        self.clear()

    @property
    def faces(self):
        """The faces of the mesh.

        This is a C-contiguous readonly ndarray. Note that the array
        may change as data is added and deleted, including faces being
        moved arround to fill holes left by deleted faces.
        """
        return self._faces_r

    @property
    def vertices(self):
        """The vertices of the mesh.

        This is a C-contiguous ndarray. Note that the array may change
        as data is added and deleted, including vertices being moved
        arround to fill holes left by deleted vertices.
        """
        # todo: vertices or positions? technically normals and colors (etc) also apply to a vertex
        return self._positions

    @property
    def vertex2faces(self):
        """Maps vertex indices to a list of face indices.

        This map can be used to e.g. traverse the mesh over its surface.

        Although technically the map is a list that can be modified in
        place, you should really not do that. Note that each element
        lists face indices in arbitrary order and may contain duplicate
        face indices.
        """
        return self._vertex2faces

    def track_changes(self, tracker, *, remove=False):
        """The given object is notified of updates of this mesh, e.g.
        to maintain a representation of the mesh. If ``remove`` is True,
        the tracker is removed instead.
        """
        if not isinstance(tracker, MeshChangeTracker):
            raise TypeError("Expected a MeshChangeTracker subclass.")
        while tracker in self._change_trackers:
            self._change_trackers.remove(tracker)
        if not remove:
            self._change_trackers.append(tracker)
            self._init_change_tracker(tracker)

    def _init_change_tracker(self, tracker):
        with Safecall():
            tracker.clear()
            tracker.set_face_array(self._faces_buf)
            tracker.set_vertex_arrays(
                self._positions_buf, self._normals_buf, self._colors_buf
            )
            if len(self._positions):
                tracker.add_vertices(self._positions)
                tracker.update_normals(np.arange(len(self._positions), dtype=np.int32))
            if len(self._faces):
                tracker.add_faces(self._faces)

    def clear(self):
        """Empty all vertices and faces."""

        initial_size = 8

        # Create the buffers that contain the data, and which are larger
        # than needed. These arrays should *only* be referenced in the
        # allocate- and deallocate- methods.
        self._faces_buf = np.zeros((initial_size, self._verts_per_face), np.int32)
        self._faces_normals_buf = np.zeros((initial_size, 3), np.float32)
        self._positions_buf = np.zeros((initial_size, 3), np.float32)
        self._normals_buf = np.zeros((initial_size, 3), np.float32)
        self._colors_buf = np.zeros((initial_size, 4), np.float32)
        # todo: Maybe face colors are more convenient?
        # todo: also, are colors better managed outside of this class?

        # We set unused positions to nan, so that code that uses the
        # full buffer does not accidentally use invalid vertex positions.
        self._positions_buf.fill(np.nan)

        # Create faces array views. The internals operate on the ._faces
        # array. We publicly expose the readonly ._faces_r array, because
        # any changes to it can corrupt the mesh (e.g. use nonexisting
        # vertices).
        self._faces = self._faces_buf[:0]
        self._faces_normals = self._faces_normals_buf[:0]
        self._faces_r = self._faces_buf[:0]
        self._faces_r.flags.writeable = False

        # The vertex array views. Not much harm can be done to these.
        self._positions = self._positions_buf[:0]
        self._normals = self._normals_buf[:0]
        self._colors = self._colors_buf[:0]

        # Notify
        for tracker in self._change_trackers:
            self._init_change_tracker(tracker)

        # Reverse map
        # This array is jagged, because the number of faces incident
        # to one vertex can potentially be big (e.g. the top and bottom
        # of a sphere sampled in lon/lat directions). We could use a
        # numpy array of lists, which has the advantage that you can
        # do `vertex2faces[multiple_indices].sum()`. However, benchmarks
        # show that this is *slower* than a simple list of lists. A
        # list of sets could also work, but is slightly slower.
        #
        # Other data structures are also possibe, e.g. one based on
        # shifts. These structures can be build faster, but using them
        # is slower due to the extra indirection.
        self._vertex2faces = []  # vi -> [fi1, fi2, ..]

    def reset(self, vertices=None, faces=None):
        """Remove all vertices and faces and add the given ones (if any)."""

        self.clear()

        if vertices is not None:
            self.add_vertices(vertices)

        if faces is not None:
            self.add_faces(faces)

    def export(self):
        """Get a copy of the array of vertices and faces.

        Note that the arrays are copied because the originals are
        modified in place when e.g. faces are removed or updated.
        """
        return self.vertices.copy(), self.faces.copy()

    def _check_internal_state(self):
        """Method to validate the integrity of the internal state. In
        practice this check should not be needed, but whole running
        tests and during dev it add an extra layer of security.
        """

        # Note: some vertices not being used is technically an ok state.
        # It is also unavoidable, because one first adds vertices and
        # then the faces to use them. But a bug in our internals could
        # make the number of unused vertices grow, so maybe we'll want
        # some sort of check for it at some point.

        vertices = self._positions
        faces = self._faces
        if len(faces) == 0:
            return
        # Check that faces match a vertex
        assert faces.min() >= 0
        assert faces.max() < len(vertices)

        # Build vertex2faces
        vertex2faces = [[] for _ in range(len(vertices))]
        for fi in range(len(faces)):
            face = faces[fi]
            vertex2faces[face[0]].append(fi)
            vertex2faces[face[1]].append(fi)
            vertex2faces[face[2]].append(fi)

        assert len(vertex2faces) == len(self._vertex2faces)
        for face1, face2 in zip(vertex2faces, self._vertex2faces):
            assert sorted(face1) == sorted(face2)

    def _allocate_faces(self, n):
        """Increase the size of the faces view to hold n free faces at
        the end. If necessary the underlying buffer is re-allocated.
        """
        n = int(n)
        assert n >= 1
        nfaces1 = len(self._faces)
        nfaces2 = nfaces1 + n
        # Re-allocate buffer?
        if nfaces2 > len(self._faces_buf):
            new_size = max(8, nfaces2)
            new_size = 2 ** int(np.ceil(np.log2(new_size)))
            self._faces_buf = np.zeros((new_size, self._verts_per_face), np.int32)
            self._faces_buf[:nfaces1] = self._faces
            self._faces_normals_buf = np.zeros((new_size, 3), np.float32)
            self._faces_normals_buf[:nfaces1] = self._faces_normals
            for tracker in self._change_trackers:
                with Safecall():
                    tracker.set_face_array(self._faces_buf)
        # Reset views
        self._faces = self._faces_buf[:nfaces2]
        self._faces_normals = self._faces_normals_buf[:nfaces2]
        self._faces_r = self._faces_buf[:nfaces2]
        self._faces_r.flags.writeable = False

    def _deallocate_faces(self, n):
        """ "Decrease the size of the faces view, discarting the n faces
        at the end. If it makes sense, the underlying buffer is re-allocated.
        """
        n = int(n)
        assert n >= 1
        nfaces = len(self._faces) - n
        # Re-allocate buffer?
        if nfaces <= 0.25 * len(self._faces_buf):
            new_size = max(8, nfaces * 2)
            new_size = 2 ** int(np.ceil(np.log2(new_size)))
            self._faces_buf = np.zeros((new_size, self._verts_per_face), np.int32)
            self._faces_buf[:nfaces] = self._faces[:nfaces]
            self._faces_normals_buf = np.zeros((new_size, 3), np.float32)
            self._faces_normals_buf[:nfaces] = self._faces_normals[:nfaces]
            for tracker in self._change_trackers:
                with Safecall():
                    tracker.set_face_array(self._faces_buf)
        else:
            # Tidy up
            self._faces_buf[nfaces:] = 0
            self._faces_normals_buf[nfaces:] = 0.0
        # Reset views
        self._faces = self._faces_buf[:nfaces]
        self._faces_normals = self._faces_normals_buf[:nfaces]
        self._faces_r = self._faces_buf[:nfaces]
        self._faces_r.flags.writeable = False

    def _allocate_vertices(self, n):
        """Increase the vertex views to hold n free vertices at the end. If
        necessary the underlying buffers are re-allocated.
        """
        n = int(n)
        assert n >= 1
        nverts1 = len(self._positions)
        nverts2 = nverts1 + n
        # Re-allocate buffer?
        if nverts2 > len(self._positions_buf):
            new_size = max(8, nverts2)
            new_size = 2 ** int(np.ceil(np.log2(new_size)))
            self._positions_buf = np.zeros((new_size, 3), np.float32)
            self._positions_buf[:nverts1] = self._positions
            self._positions_buf[nverts2:] = np.nan
            self._normals_buf = np.zeros((new_size, 3), np.float32)
            self._normals_buf[:nverts1] = self._normals
            self._colors_buf = np.zeros((new_size, 4), np.float32)
            self._colors_buf[:nverts1] = self._colors
            for tracker in self._change_trackers:
                with Safecall():
                    tracker.set_vertex_arrays(
                        self._positions_buf, self._normals_buf, self._colors_buf
                    )
        # Reset views
        self._positions = self._positions_buf[:nverts2]
        self._normals = self._normals_buf[:nverts2]
        self._colors = self._colors_buf[:nverts2]

    def _deallocate_vertices(self, n):
        """Decrease the size of the vertices views, discarting the n vertices
        at the end. If it makes sense, the underlying buffer is re-allocated.
        """
        n = int(n)
        assert n >= 1
        nverts = len(self._positions) - n
        # Re-allocate buffer?
        if nverts <= 0.25 * len(self._positions_buf):
            new_size = max(8, nverts * 2)
            new_size = 2 ** int(np.ceil(np.log2(new_size)))
            self._positions_buf = np.zeros((new_size, 3), np.float32)
            self._positions_buf[:nverts] = self._positions[:nverts]
            self._positions_buf[nverts:] = np.nan
            self._normals_buf = np.zeros((new_size, 3), np.float32)
            self._normals_buf[:nverts] = self._normals[:nverts]
            self._colors_buf = np.zeros((new_size, 4), np.float32)
            self._colors_buf[:nverts] = self._colors[:nverts]
            for tracker in self._change_trackers:
                with Safecall():
                    tracker.set_vertex_arrays(
                        self._positions_buf, self._normals_buf, self._colors_buf
                    )
        else:
            # Tidy up
            self._positions_buf[nverts:] = np.nan
            self._normals_buf[nverts:] = 0
            self._colors_buf[nverts:] = 0
        # Reset views
        self._positions = self._positions_buf[:nverts]
        self._normals = self._normals_buf[:nverts]
        self._colors = self._colors_buf[:nverts]

    def _update_face_normals(self, face_indices):
        """Update the selected face normals."""
        face_indices = np.asarray(face_indices, np.int32)
        faces = self._faces[face_indices]
        positions = self._positions

        r1 = positions[faces[:, 0], :]
        r2 = positions[faces[:, 1], :]
        r3 = positions[faces[:, 2], :]
        face_normals = np.cross((r2 - r1), (r3 - r1))  # assumes CCW
        # faces_areas = 0.5 * np.linalg.norm(face_normals, axis=1)

        self._faces_normals[face_indices] = face_normals

        # The thing with vertex normals is that they depend on all
        # incident faces, so doing a partial update is tricky.
        # * We could first undo the contribution of the selected faces. E.g.
        #   using a list of dicts: vi -> fi -> normals. Likely slow.
        # * Or we first select the vertex_indices (by flattening faces), and use
        #   vertex2faces to come up with a slightly larger set of face_indices,
        #   which we then use to update the normals for the vertex_indices (and more)
        #   and then only update vertex_indices. Also likely slow.
        # * Only update the face normals and vertex normals of connected components.
        # * Or we just update all vertex normals, but notify trackers
        #   with the indices of vertices who's normal actually changed.
        # * Note: we could implement some form of lazy computation for the
        #   normals, or an option to turn all this off if no normals are needed.
        self._update_vertex_normals()

        # Get indices of vertices who's normal changed
        vertex_mask = np.zeros((len(self._positions),), bool)
        vertex_mask[faces.flatten()] = True
        vertex_indices = np.where(vertex_mask)[0]

        # Pass on the update
        for tracker in self._change_trackers:
            with Safecall():
                tracker.update_normals(vertex_indices)

    def _update_vertex_normals(self):
        """Update all vertex normals."""
        vertex_normals = self._normals
        vertex_normals.fill(0.0)
        for i in range(3):
            np.add.at(vertex_normals, self._faces[:, i], self._faces_normals)

        norms = np.linalg.norm(vertex_normals, axis=1)

        (zeros,) = np.where(norms == 0)
        norms[zeros] = 1.0  # prevent divide-by-zero
        vertex_normals /= norms[:, np.newaxis]
        vertex_normals[zeros] = 0.0

    def _get_indices(self, indices, n, what_for):
        """Convenience function used by methods that accept an index array."""
        result = None
        typ = type(indices).__name__
        if isinstance(indices, int):
            result = [indices]
        elif isinstance(indices, list):
            result = indices
        elif isinstance(indices, np.ndarray):
            typ = (
                "ndarray:"
                + "x".join(str(x) for x in indices.shape)
                + "x"
                + indices.dtype.name
            )
            if indices.size == 0:
                result = []
            elif indices.ndim == 1 and indices.dtype.kind == "i":
                result = indices

        if result is None:
            raise TypeError(
                f"The {what_for} must be given as int, list, or 1D int array, not {typ}."
            )
        result = np.asarray(result, np.int32)
        if len(result) == 0:
            raise ValueError(f"The {what_for} must not be empty.")
        elif result.min() < 0:
            raise ValueError("Negative indices not allowed.")
        elif result.max() >= n:
            raise ValueError("Index out of bounds.")

        return result

    def swap_faces(self, face_indices1, face_indices2):
        """Swap the faces indicated by the given indices.

        This method is public, but likely not generally useful by
        itself. The ``delete_faces()`` method is a convenience
        combination of ``swap_faces()`` and ``pop_faces()``.
        """

        # Technically this can also be done with update_faces, but
        # swapping is faster and costs significantly less memory in
        # e.g. an undo stack.

        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        indices1 = self._get_indices(
            face_indices1, len(self._faces), "face indices to swap (1)"
        )
        indices2 = self._get_indices(
            face_indices2, len(self._faces), "face indices to swap (2)"
        )

        if not len(indices1) == len(indices2):
            raise ValueError("Both index arrays must have the same length.")

        # --- Apply

        try:
            # Update reverse map (unrolled loops for small performance bump)
            for fi1, fi2 in zip(indices1, indices2):
                face1 = self._faces[fi1]
                fii = vertex2faces[face1[0]]
                fii.remove(fi1)
                fii.append(fi2)
                fii = vertex2faces[face1[1]]
                fii.remove(fi1)
                fii.append(fi2)
                fii = vertex2faces[face1[2]]
                fii.remove(fi1)
                fii.append(fi2)
                face2 = self._faces[fi2]
                fii = vertex2faces[face2[0]]
                fii.remove(fi2)
                fii.append(fi1)
                fii = vertex2faces[face2[1]]
                fii.remove(fi2)
                fii.append(fi1)
                fii = vertex2faces[face2[2]]
                fii.remove(fi2)
                fii.append(fi1)

            # Swap the faces themselves
            for a in [self._faces, self._faces_normals]:
                a[indices1], a[indices2] = a[indices2], a[indices1]

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_faces = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.swap_faces(indices1, indices2)
        if self._debug_mode:
            self._check_internal_state()

    def pop_faces(self, n, _old=None):
        """Remove the last n faces from the mesh."""
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        if n <= 0:
            raise ValueError("Number of faces to pop must be larger than zero.")
        if n > len(self._faces):
            raise ValueError(
                "Number of faces to pop is larger than the total number of faces."
            )

        nfaces1 = len(self.faces)
        nfaces2 = nfaces1 - n

        # --- Apply

        old_faces = self._faces[nfaces2:].copy()

        try:
            # Update reverse map
            for fi in range(nfaces2, nfaces1):
                face = self._faces[fi]
                vertex2faces[face[0]].remove(fi)
                vertex2faces[face[1]].remove(fi)
                vertex2faces[face[2]].remove(fi)

            # Adjust the array lengths (reset views)
            self._deallocate_faces(n)

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_faces = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.pop_faces(n, old_faces)
        if self._debug_mode:
            self._check_internal_state()

    def delete_faces(self, face_indices):
        """Delete the faces indicated by the given face indices.

        The deleted faces are replaced with faces from the end of the
        array (except for deleted faces that leave no holes because
        they are already at the end).

        An error can be raised if e.g. the indices are out of bounds.
        """

        # --- Prepare / checks

        indices = self._get_indices(
            face_indices, len(self._faces), "face indices to delete"
        )
        to_delete = set(indices)

        nfaces1 = len(self._faces)
        nfaces2 = nfaces1 - len(to_delete)

        to_maybe_move = set(range(nfaces2, nfaces1))  # these are for filling the holes
        to_just_drop = to_maybe_move & to_delete  # but some of these may be at the end

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2), "Internal error"

        # --- Apply -> delegate

        # Do a move, so all faces to delete are at the end
        if len(indices1):
            self.swap_faces(indices1, indices2)

        # Pop from the end
        self.pop_faces(len(to_delete))

    def swap_vertices(self, vertex_indices1, vertex_indices2):
        """Move the vertices indicated by the given indices.

        This method is public, but likely not generally useful by
        itself. The ``delete_vertices()`` method is a convenience
        combination of ``swap_vertices()`` and ``pop_vertices()``.
        """

        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        indices1 = self._get_indices(
            vertex_indices1, len(self._positions), "vertex indices to swap (1)"
        )
        indices2 = self._get_indices(
            vertex_indices2, len(self._positions), "vertex indices to swap (2)"
        )

        if not len(indices1) == len(indices2):
            raise ValueError("Both index arrays must have the same length.")

        # --- Apply

        try:
            # Swap the vertices themselves
            for a in [self._positions, self._normals, self._colors]:
                a[indices1], a[indices2] = a[indices2], a[indices1]

            # Update the faces that refer to the moved indices
            faces = self._faces
            for vi1, vi2 in zip(indices1, indices2):
                mask1 = faces == vi1
                mask2 = faces == vi2
                faces[mask2] = vi1
                faces[mask1] = vi2

            # Update reverse map
            for vi1, vi2 in zip(indices1, indices2):
                vertex2faces[vi1], vertex2faces[vi2] = (
                    vertex2faces[vi2],
                    vertex2faces[vi1],
                )

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_verts = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.swap_vertices(indices1, indices2)
        if self._debug_mode:
            self._check_internal_state()

    def pop_vertices(self, n, _old=None):
        """Remove the last n vertices from the mesh."""
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        if n <= 0:
            raise ValueError("Number of vertices to pop must be larger than zero.")
        if n > len(self._positions):
            raise ValueError(
                "Number of vertices to pop is larger than the total number of vertices."
            )

        nverts1 = len(self._positions)
        nverts2 = nverts1 - n

        # --- Apply

        old_positions = self._positions[nverts2:].copy()

        try:
            # Drop unused vertices at the end
            self._deallocate_vertices(nverts1 - nverts2)

            # Update reverse map
            vertex2faces[nverts2:] = []

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_verts = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.pop_vertices(n, old_positions)
        if self._debug_mode:
            self._check_internal_state()

    def delete_vertices(self, vertex_indices):
        """Delete the vertices indicated by the given vertex indices.

        The deleted vertices are replaced with vertices from the end of the
        array (except for deleted vertices that leave no holes because
        they are already at the end).

        An error can be raised if e.g. the indices are out of bounds.
        """

        # Note: defragmenting when deleting vertices is somewhat
        # expensive because we also need to update the faces. From a
        # technical perspective it's fine to have unused vertices, so
        # we could just leave them as holes, which would likely be faster.
        # However, the current implementation also has advantages:
        #
        # - It works the same as for the faces.
        # - With a contiguous vertex array it is easy to check if faces are valid.
        # - No need to select vertices that are in use (e.g. for bounding boxes).
        # - Getting free slots for vertices is straightforward without
        #   the need for additional structures like a set of free vertices.
        # - The vertices and faces can at any moment be copied and be sound.

        vertex2faces = self._vertex2faces

        # --- Prepare / checcks

        indices = self._get_indices(
            vertex_indices, len(self._positions), "vertex indices to delete"
        )
        to_delete = set(indices)

        nverts1 = len(self._positions)
        nverts2 = nverts1 - len(to_delete)

        to_maybe_move = set(range(nverts2, nverts1))
        to_just_drop = to_maybe_move & to_delete

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2), "Internal error"

        # Check that none of the vertices are in use
        if any(len(vertex2faces[vi]) > 0 for vi in to_delete):
            raise ValueError("Vertex to delete is in use.")

        # --- Apply

        # Do a move, so all vertices to delete are at the end
        if len(indices1):
            self.swap_vertices(indices1, indices2)

        # Pop from the end
        self.pop_vertices(len(to_delete))

    def add_faces(self, new_faces):
        """Add the given faces to the mesh.

        The faces must reference existing vertices.
        """
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        # Check incoming array
        faces = np.asarray(new_faces, np.int32)
        if not (
            isinstance(faces, np.ndarray)
            and faces.ndim == 2
            and faces.shape[1] == self._verts_per_face
        ):
            raise TypeError("Faces must be a Nx3 array")
        # It's fine for the mesh to have zero faces, but it's likely
        # an error if the user calls this with an empty array.
        if len(faces) == 0:
            raise ValueError("Cannot add zero faces.")
        # Check sanity of the faces
        if faces.min() < 0 or faces.max() >= len(self._positions):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        n = len(faces)
        n1 = len(self._faces)
        indices = np.arange(n1, n1 + n, dtype=np.int32)

        # --- Apply

        try:
            self._allocate_faces(n)
            self._faces[n1:] = faces
            self._update_face_normals(indices)

            # Update reverse map
            for i, face in enumerate(faces):
                fi = i + n1
                vertex2faces[face[0]].append(fi)
                vertex2faces[face[1]].append(fi)
                vertex2faces[face[2]].append(fi)

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_faces = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.add_faces(faces)
        if self._debug_mode:
            self._check_internal_state()

    def add_vertices(self, new_positions):
        """Add the given vertices to the mesh."""
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        # Check incoming array
        positions = np.asarray(new_positions, np.float32)
        if not (
            isinstance(positions, np.ndarray)
            and positions.ndim == 2
            and positions.shape[1] == 3
        ):
            raise TypeError("Vertices must be a Nx3 array")
        if len(positions) == 0:
            raise ValueError("Cannot add zero vertices.")

        n = len(positions)
        n1 = len(self._positions)

        # --- Apply

        try:
            self._allocate_vertices(n)
            self._positions[n1:] = new_positions
            self._colors[n1:] = 0.7, 0.7, 0.7, 1.0

            # Update reverse map
            vertex2faces.extend([] for i in range(n))

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_verts = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.add_vertices(positions)
        if self._debug_mode:
            self._check_internal_state()

    def update_faces(self, face_indices, new_faces, _old=None):
        """Update the value of the given faces."""

        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        indices = self._get_indices(
            face_indices, len(self._faces), "face indices to update"
        )
        faces = np.asarray(new_faces, np.int32).reshape(-1, self._verts_per_face)

        if len(indices) != len(faces):
            raise ValueError("Indices and faces to update have different lengths.")

        # Check sanity of the faces
        if faces.min() < 0 or faces.max() >= len(self._positions):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        # Note: this should work too, but moves more stuff around, so its less efficient.
        # self.delete_faces(face_indices)
        # self.add_faces(faces)

        # --- Apply

        old_faces = self._faces[indices]

        try:
            # Update reverse map
            for fi, new_face in zip(indices, faces):
                old_face = self._faces[fi]
                vertex2faces[old_face[0]].remove(fi)
                vertex2faces[old_face[1]].remove(fi)
                vertex2faces[old_face[2]].remove(fi)
                vertex2faces[new_face[0]].append(fi)
                vertex2faces[new_face[1]].append(fi)
                vertex2faces[new_face[2]].append(fi)

            # Update
            self._faces[indices] = faces
            self._update_face_normals(indices)

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_faces = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.update_faces(indices, old_faces)
        if self._debug_mode:
            self._check_internal_state()

    def update_vertices(self, vertex_indices, new_positions, _old=None):
        """Update the value of the given vertices."""

        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        indices = self._get_indices(
            vertex_indices, len(self._positions), "vertex indices to update"
        )
        positions = np.asarray(new_positions, np.int32).reshape(-1, 3)

        if len(indices) != len(positions):
            raise ValueError("Indices and positions to update have different lengths.")

        # --- Apply

        old_positions = self._positions[indices]

        try:
            self._positions[indices] = positions

            # Note: if the number of changed vertices is large (say 50% or more)
            # it'd probably be more efficient to collect face_indices via a mask.
            face_indices = set()
            for vi in indices:
                face_indices.update(vertex2faces[vi])
            self._update_face_normals(list(face_indices))

        except Exception:  # pragma: no cover
            logger.warn(EXCEPTION_IN_ATOMIC_CODE)
            raise

        # --- Notify

        self._cache_depending_on_verts = {}
        self._cache_depending_on_verts_and_faces = {}
        for tracker in self._change_trackers:
            with Safecall():
                tracker.update_vertices(indices, old_positions)
        if self._debug_mode:
            self._check_internal_state()
