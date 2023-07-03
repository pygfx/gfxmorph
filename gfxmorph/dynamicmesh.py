import numpy as np

# We assume meshes with triangles (not quads)
VERTICES_PER_FACE = 3


class UnexpectedExceptionInAtomicCode(RuntimeError):
    pass


class DynamicMesh:
    """An object that holds mesh data that can be modified in-place.
    It has buffers that are oversized so the vertex and face array can
    grow. When the buffer is full, a larger buffer is allocated. The
    arrays are contiguous views onto the buffers. Modifications are
    managed to keep the arrays without holes.
    """

    def __init__(self):
        initial_size = 0

        self._debug_mode = True

        # Create the buffers
        self._faces_buf = np.zeros((initial_size, 3), np.int32)
        self._positions_buf = np.zeros((initial_size, 3), np.float32)
        self._normals_buf = np.zeros((initial_size, 3), np.float32)
        self._colors_buf = np.zeros((initial_size, 4), np.float32)
        # todo: Maybe face colors are more convenient?

        # Create array views
        self._faces = self._faces_buf[:0]
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

        self.version_verts = 0
        self.version_faces = 0

    @property
    def faces(self):
        # todo: return a readonly version
        return self._faces

    @property
    def vertices(self):
        # todo: return a readonly version
        # todo: vertices or positions? technically normals and colors (etc) also apply to a vertex
        return self._positions

    def _check_internal_state(self):
        # Some vertices not being used is technically an ok state. It
        # is also unavoidable, because one first adds vertices and then
        # the faces to use them. But a bug in our internals could make
        # the number of unused vertices grow, so maybe we'll want some
        # sort of check for it at some point.

        faces = self.faces
        if len(faces) == 0:
            return
        # Check that faces match a vertex
        assert faces.min() >= 0
        assert faces.max() < len(self.vertices)

        # Build vertex2faces
        vertex2faces = [[] for _ in range(len(self.vertices))]
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

    def _allocate_faces_buffer(self, n):
        # Sanity check
        nfaces = len(self._faces)
        assert n >= nfaces
        # Allocate new array.
        self._faces_buf = np.zeros((n, VERTICES_PER_FACE), np.int32)
        # Copy the data and reset view
        self._faces_buf[:nfaces, :] = self._faces
        self._faces = self._faces_buf[:nfaces]

    def _allocate_vertices_buffer(self, n):
        # Sanity check
        nverts = len(self._positions)
        assert n >= nverts
        # Allocate new arrays.
        self._positions_buf = np.empty((n, 3), np.float32)
        self._normals_buf = np.empty((n, 3), np.float32)
        self._colors_buf = np.empty((n, 4), np.float32)
        # Copy the data
        self._positions_buf[:nverts, :] = self._positions
        self._normals_buf[:nverts, :] = self._normals
        self._colors_buf[:nverts, :] = self._colors
        # Reset views
        self._positions = self._positions_buf[:nverts]
        self._normals = self._normals_buf[:nverts]
        self._colors = self._colors_buf[:nverts]

    def _ensure_free_faces(self, n, resize_factor=1.5):
        """Make sure that there are at least n free slots for faces.
        If not, increase the total size of the array by resize_factor.
        """
        n = int(n)
        assert n >= 1
        assert resize_factor > 1
        free_faces = len(self._faces_buf) - len(self._faces)

        if free_faces < n:
            new_size = max(
                len(self._faces) + n, int(len(self._faces_buf) * resize_factor)
            )
            # print(f"Re-allocating faces array to {new_size} elements.")
            self._allocate_faces_buffer(new_size)

    def _ensure_free_vertices(self, n, resize_factor=1.5):
        """Make sure that there are at least n free slots for vertices.
        If not, increase the total size of the array by resize_factor.
        """
        n = int(n)
        assert n >= 1
        assert resize_factor > 1
        free_vertices = len(self._positions_buf) - len(self._positions)

        # todo: also reduce slots if below a certain treshold?
        # To do this without remapping any indices (which we want to avoid, e.g. to keep undo working)
        # we can only make the array as small as the largest index in use. But we can
        # reorganize the array a bit (just a few moves) on each resample(), so that the arrays
        # stay more or less tightly packed.

        if free_vertices < n:
            new_size = max(
                len(self._positions) + n, int(len(self._positions_buf) * resize_factor)
            )
            # print(f"Re-allocating vertex array to {new_size} elements.")
            self._allocate_vertices_buffer(new_size)

    def _get_indices(self, indices, what_for):
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
            # note: we could allow deleting faces/vertices via a view, but .. maybe not?
            # elif indices.ndim == 2 and faces.shape[1] == 3 and faces.base is self._faces_buf:
            #     addr0 = self._faces_buf.__array_interface__['data'][0]
            #     addr1 = faces.__array_interface__['data'][0]

        if result is None:
            raise TypeError(
                f"The {what_for} must be given as int, list, or 1D int array, not {typ}."
            )
        elif len(result) == 0:
            raise ValueError(f"The {what_for} must not be empty.")
        elif min(indices) < 0:
            raise ValueError("Negative indices not allowed.")

        return result

    def delete_faces(self, face_indices):
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        to_delete = set(self._get_indices(face_indices, "face indices to delete"))

        nfaces1 = len(self._faces_buf)
        nfaces2 = nfaces1 - len(to_delete)

        to_maybe_move = set(range(nfaces2, nfaces1))  # these are for filling the holes
        to_just_drop = to_maybe_move & to_delete  # but some of these may be at the end

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2), "Internal error"

        if not all(0 <= fi < nfaces1 for fi in to_delete):
            raise ValueError("Face to delete is out of bounds.")

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
                for i in range(3):
                    fii = vertex2faces[face[i]]
                    fii.remove(fi2)
                    fii.append(fi1)

            # Move vertices from the end into the slots of the deleted vertices
            self._faces[indices1] = self._faces[indices2]

            # Adjust the array lengths (reset views)
            self._faces = self._faces_buf[:nfaces2]

            # Tidy up
            self._faces_buf[nfaces2:nfaces1] = 0

            # Bump version
            self.version_faces += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:
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

        to_delete = set(self._get_indices(vertex_indices, "vertex indices to delete"))

        nverts1 = len(self._positions)
        nverts2 = nverts1 - len(to_delete)

        to_maybe_move = set(range(nverts2, nverts1))
        to_just_drop = to_maybe_move & to_delete

        indices1 = list(to_delete - to_just_drop)
        indices2 = list(to_maybe_move - to_just_drop)
        assert len(indices1) == len(indices2), "Internal error"

        # Check that none of the vertices are in use
        for vi in to_delete:
            if not (0 <= vi < nverts1):
                raise ValueError("Vertex to delete is out of bounds.")
            if len(vertex2faces[vi]) > 0:
                raise ValueError("Vertex to delete is in use.")

        # --- Apply

        try:
            # Move vertices from the end into the slots of the deleted vertices
            self._positions[indices1] = self._positions[indices2]
            self._normals[indices1] = self._normals[indices2]
            self._colors[indices1] = self._colors[indices2]

            # Adjust the array lengths (reset views)
            self._positions = self._positions_buf[:nverts2]
            self._normals = self._normals_buf[:nverts2]
            self._colors = self._colors_buf[:nverts2]

            # Zero out the free slots, just to tidy up
            self._positions_buf[nverts2:nverts1] = 0
            self._normals_buf[nverts2:nverts1] = 0
            self._colors_buf[nverts2:nverts1] = 0

            # Update the faces that refer to the moved indices
            for vi1, vi2 in zip(indices1, indices2):
                self._faces[self._faces == vi2] = vi1

            # Update reverse map
            for vi1, vi2 in zip(indices1, indices2):
                vertex2faces[vi1] = vertex2faces[vi2]
            vertex2faces[nverts2:] = []

            # Bump version
            self.version_verts += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:
            raise UnexpectedExceptionInAtomicCode(err)

    def add_faces(self, new_faces):
        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        # Check incoming array
        faces = np.asarray(new_faces, np.int32)
        if not (
            isinstance(faces, np.ndarray)
            and faces.ndim == 2
            and faces.shape[1] == VERTICES_PER_FACE
        ):
            raise TypeError("Faces must be a Nx3 array")
        # We want to be able to assume that there is at least one face, and 3 valid vertices
        # (or 4 faces and 4 vertices for a closed mesh)
        if len(faces) == 0:
            raise ValueError("Cannot add zero faces.")
        # Check sanity of the faces
        if faces.min() < 0 or faces.max() >= len(self._positions):
            raise ValueError(
                "The faces array containes indices that are out of bounds."
            )

        n = len(faces)
        n1 = len(self._faces)
        n2 = n1 + n

        # --- Apply

        try:
            self._ensure_free_faces(n)
            self._faces = self._faces_buf[:n2]
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

        except Exception as err:
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
        n2 = n1 + n

        # --- Apply

        try:
            self._ensure_free_vertices(n)
            self._positions = self._positions_buf[:n2]
            self._normals = self._normals_buf[:n2]
            self._colors = self._colors_buf[:n2]

            self._positions[n1:] = new_positions
            self._normals[n1:] = 0.0
            self._colors[n1:] = 0.7, 0.7, 0.7, 1.0

            # Update reverse map
            vertex2faces.extend([] for i in range(n))

            self.version_verts += 1
            if self._debug_mode:
                self._check_internal_state()

        except Exception as err:
            raise UnexpectedExceptionInAtomicCode(err)

    def update_faces(self, face_indices, new_faces):
        """Update the value of the given faces."""

        vertex2faces = self._vertex2faces

        # --- Prepare / checks

        nfaces1 = len(self.faces)
        indices = self._get_indices(face_indices, "face indices to update")
        faces = np.asarray(new_faces, np.int32)

        if len(indices) != len(faces):
            raise ValueError("Indices and faces to update have different lengths.")
        if len(indices) == 0:
            raise ValueError("Cannot update zero faces.")

        # Check indices
        if not all(0 <= fi < nfaces1 for fi in indices):
            raise ValueError("Face to update is out of bounds.")

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

        except Exception as err:
            raise UnexpectedExceptionInAtomicCode(err)

    # def allocate_faces(self, n):
    #     """Add n new faces to the mesh. Return an nx3 array with the new faces,
    #     which is a view on the faces buffer (so one can write to it).
    #     """
    #     n = int(n)
    #     n1 = len(self._faces)
    #     n2 = n1 + n
    #
    #     self._ensure_free_faces(n)
    #     self._faces = self._faces_buf[:n2]
    #     return self._faces[n1:]
    #
    # def allocate_vertices(self, n):
    #     """Add n new vertices to the mesh. Return an nx3 array with the new positions,
    #     which is a view on the positions buffer (so one can write to it).
    #     """
    #     n = int(n)
    #     n1 = len(self._positions)
    #     n2 = n1 + n
    #
    #     self._ensure_free_vertices(n)
    #     self._positions = self._positions_buf[:n2]
    #     self._normals = self._normals_buf[:n2]
    #     self._colors = self._colors_buf[:n2]
    #     return self._positions[n1:]
