import numpy as np


class UnexpectedExceptionInAtomicCode(RuntimeError):
    """Raised when an error occurs within a piece of code that is supposed to be atomic."""

    pass


class DynamicMesh:
    """An object that holds mesh data that can be modified in-place.
    It has buffers that are oversized so the vertex and face array can
    grow. When the buffer is full, a larger buffer is allocated. The
    arrays are contiguous views onto the buffers. Modifications are
    managed to keep the arrays without holes.
    """

    # We assume meshes with triangles (not quads)
    _verts_per_face = 3

    # In debug mode, the mesh checks its internal state after each change
    _debug_mode = False

    def __init__(self):
        self.version_verts = 0
        self.version_faces = 0
        self.reset(initial_size=0)

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

    def reset(self, *, vertices=None, faces=None, initial_size=0):
        """Remove all vertices and faces and add the given ones (if any)."""

        # Create the buffers that contain the data, and which are larger
        # than needed. These arrays should *only* be referenced in the
        # allocate- and deallocate- methods.
        self._faces_buf = np.zeros((initial_size, self._verts_per_face), np.int32)
        self._positions_buf = np.zeros((initial_size, 3), np.float32)
        self._normals_buf = np.zeros((initial_size, 3), np.float32)
        self._colors_buf = np.zeros((initial_size, 4), np.float32)
        # todo: Maybe face colors are more convenient?

        # Create faces array views. The internals operate on the ._faces
        # array. We publicly expose the readonly ._faces_r array, because
        # any changes to it can corrupt the mesh (e.g. use nonexisting
        # vertices).
        self._faces = self._faces_buf[:0]
        self._faces_r = self._faces_buf[:0]
        self._faces_r.flags.writeable = False

        # The vertex array views. Not much harm can be done to these.
        self._positions = self._positions_buf[:0]
        self._normals = self._normals_buf[:0]
        self._colors = self._colors_buf[:0]

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

        if vertices is not None:
            self.add_vertices(vertices)

        if faces is not None:
            self.add_faces(faces)

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

        # todo: being able to do this test might be a reason to use sets instead of lists
        # assert vertex2faces == self._vertex2faces
        assert len(vertex2faces) == len(self._vertex2faces)
        for face1, face2 in zip(vertex2faces, self._vertex2faces):
            assert set(face1) == set(face2)

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
            # print(f"Re-allocating faces array to {new_size} elements.")
            self._faces_buf = np.zeros((new_size, self._verts_per_face), np.int32)
            self._faces_buf[:nfaces1] = self._faces
        # Reset views
        self._faces = self._faces_buf[:nfaces2]
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
            # print(f"De-allocating faces array to {new_size} elements.")
            self._faces_buf = np.zeros((new_size, self._verts_per_face), np.int32)
            self._faces_buf[:nfaces] = self._faces[:nfaces]
        else:
            # Tidy up
            self._faces_buf[nfaces:] = 0
        # Reset views
        self._faces = self._faces_buf[:nfaces]
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
            # print(f"Re-allocating vertex array to {new_size} elements.")
            self._positions_buf = np.zeros((new_size, 3), np.float32)
            self._positions_buf[:nverts1] = self._positions
            self._normals_buf = np.zeros((new_size, 3), np.float32)
            self._normals_buf[:nverts1] = self._normals
            self._colors_buf = np.zeros((new_size, 4), np.float32)
            self._colors_buf[:nverts1] = self._colors
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
            # print(f"Re-allocating vertex array to {new_size} elements.")
            self._positions_buf = np.zeros((new_size, 3), np.float32)
            self._positions_buf[:nverts] = self._positions[:nverts]
            self._normals_buf = np.zeros((new_size, 3), np.float32)
            self._normals_buf[:nverts] = self._normals[:nverts]
            self._colors_buf = np.zeros((new_size, 4), np.float32)
            self._colors_buf[:nverts] = self._colors[:nverts]
        else:
            # Tidy up
            self._positions_buf[nverts:] = 0
            self._normals_buf[nverts:] = 0
            self._colors_buf[nverts:] = 0
        # Reset views
        self._positions = self._positions_buf[:nverts]
        self._normals = self._normals_buf[:nverts]
        self._colors = self._colors_buf[:nverts]

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
        elif len(result) == 0:
            raise ValueError(f"The {what_for} must not be empty.")
        elif min(result) < 0:
            raise ValueError("Negative indices not allowed.")
        elif max(result) >= n:
            raise ValueError("Index out of bounds.")

        return result

    def delete_faces(self, face_indices):
        """Delete the faces indicated by the given face indices.

        The deleted faces are replaced with faces from the end of the
        array, except for deleted faces that leave no holes  because
        they are already at the end.

        An error can be raised if e.g. the indices are out of bounds.
        """
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        to_delete = set(
            self._get_indices(face_indices, len(self._faces), "face indices to delete")
        )

        nfaces1 = len(self._faces)
        nfaces2 = nfaces1 - len(to_delete)

        to_maybe_move = set(range(nfaces2, nfaces1))  # these are for filling the holes
        to_just_drop = to_maybe_move & to_delete  # but some of these may be at the end

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2), "Internal error"

        # --- Apply

        try:
            # Update reverse map
            for fi in to_delete:
                face = self._faces[fi]
                vertex2faces[face[0]].remove(fi)
                vertex2faces[face[1]].remove(fi)
                vertex2faces[face[2]].remove(fi)
            for fi1, fi2 in zip(indices1, indices2):
                face = self._faces[fi2]
                for i in range(self._verts_per_face):
                    fii = vertex2faces[face[i]]
                    fii.remove(fi2)
                    fii.append(fi1)

            # Move vertices from the end into the slots of the deleted vertices
            self._faces[indices1] = self._faces[indices2]

            # Adjust the array lengths (reset views)
            self._deallocate_faces(nfaces1 - nfaces2)

            # Bump version
            self.version_faces += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:  # pragma: no cover
            raise UnexpectedExceptionInAtomicCode(err)

    def delete_vertices(self, vertex_indices):
        # Note: defragmenting when deleting vertices is somewhat expensive
        # because we also need to update the faces. It'd likely be cheaper
        # to let the vertex buffers contain holes, and fill these up as vertices
        # are added. However, the current implementation also has advantages
        # and I like that it works the same as for the faces. Some observations:
        #
        # - With a contiguous vertex array it is easy to check if faces are valid.
        # - No nan checks anywhere.
        # - Getting free slots for vertices is straightforward without
        #   the need for additional structures like a set of free vertices.
        # - The vertices and faces can at any moment be copied and be sound. No export needed.

        vertex2faces = self._vertex2faces

        # --- Prepare / checcks

        to_delete = set(
            self._get_indices(
                vertex_indices, len(self._positions), "vertex indices to delete"
            )
        )

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

        try:
            # Move vertices from the end into the slots of the deleted vertices
            self._positions[indices1] = self._positions[indices2]
            self._normals[indices1] = self._normals[indices2]
            self._colors[indices1] = self._colors[indices2]

            # Drop unused vertices at the end
            self._deallocate_vertices(nverts1 - nverts2)

            # Update the faces that refer to the moved indices
            faces = self._faces
            for vi1, vi2 in zip(indices1, indices2):
                faces[faces == vi2] = vi1

            # Update reverse map
            for vi1, vi2 in zip(indices1, indices2):
                vertex2faces[vi1] = vertex2faces[vi2]
            vertex2faces[nverts2:] = []

            # Bump version
            self.version_verts += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:  # pragma: no cover
            raise UnexpectedExceptionInAtomicCode(err)

    def add_faces(self, new_faces):
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

        # --- Apply

        try:
            self._allocate_faces(n)
            self._faces[n1:] = faces

            # Update reverse map
            for i in range(len(faces)):
                fi = i + n1
                face = faces[i]
                vertex2faces[face[0]].append(fi)
                vertex2faces[face[1]].append(fi)
                vertex2faces[face[2]].append(fi)

            self.version_faces += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:  # pragma: no cover
            raise UnexpectedExceptionInAtomicCode(err)

    def add_vertices(self, new_positions):
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

            self.version_verts += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:  # pragma: no cover
            raise UnexpectedExceptionInAtomicCode(err)

    def update_faces(self, face_indices, new_faces):
        """Update the value of the given faces."""

        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        indices = self._get_indices(
            face_indices, len(self._faces), "face indices to update"
        )
        faces = np.asarray(new_faces, np.int32)

        if len(indices) != len(faces):
            raise ValueError("Indices and faces to update have different lengths.")

        # Check sanity of the faces
        if faces.min() < 0 or faces.max() >= len(self._positions):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        # Note: this should work to, but moves more stuff around, so its less efficient.
        # self.delete_faces(face_indices)
        # self.add_faces(faces)

        # --- Apply

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

            self.version_faces += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:  # pragma: no cover
            raise UnexpectedExceptionInAtomicCode(err)
