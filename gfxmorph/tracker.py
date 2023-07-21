class MeshChangeTracker:
    """Base class for tracking changes of a BaseDynamicMesh.

    To track changes, create a subclass and implementing (a subset of)
    its methods. The instance of the subclass can then be subscribed
    using ``dynamic_mesh.track_changes()`` to receive updates.

    The methods of this class are called at specific events. Together
    they (hopefully) provide an API to adequately track changes for a
    number of use-cases. Such use-cases may include e.g. logging,
    keeping an undo stack, and updating a GPU representation of the
    mesh.
    """

    def init(self, mesh):
        """Called when the tracker is subscribed to the mesh.

        This enables a tracker to properly reset as needed, or e.g.
        prevent registering to multiple meshes, etc.

        You may want to take into account that the mesh already has
        vertices and faces at the moment that the tracker is subscribed.
        """
        pass

    # API that is the same as changes to BaseDynamicMesh, except for the `old` argument

    def add_faces(self, faces):
        """Called when faces are added to the mesh.

        Same signature as ``DynamicMesh.add_faces``.
        """
        pass

    def pop_faces(self, n, old):
        """Called when faces are are removed from the mesh.

        Same signature as ``DynamicMesh.pop_faces``, except for the ``old`` arg.
        Note that calling ``delete_faces()`` on the mesh results in a
        ``swap_faces()`` and a ``pop_faces()``.
        """
        pass

    def swap_faces(self, indices1, indices2):
        """Called when the mesh swaps faces (to keep the arrays contiguous).

        Same signature as ``DynamicMesh.swap_faces``.
        """
        pass

    def update_faces(self, indices, faces, old):
        """Called when faces are are updated.

        Same signature as ``DynamicMesh.update_faces``, except for the ``old`` arg.
        """
        pass

    def add_vertices(self, positions):
        """Called when vertices are added to the mesh.

        Same signature as ``DynamicMesh.add_vertices``.
        """
        pass

    def pop_vertices(self, n, old):
        """Called when vertices are are removed from the mesh.

        Same signature as ``DynamicMesh.pop_vertices``, except for the ``old`` arg.
        Note that calling ``delete_vertices()`` on the mesh results in a
        ``swap_vertices()`` and a ``pop_vertices()``.
        """
        pass

    def swap_vertices(self, indices1, indices2):
        """Called when the mesh swaps vertices (to keep the arrays contiguous).

        Same signature as ``DynamicMesh.swap_vertices``.
        """
        pass

    def update_vertices(self, indices, positions, old):
        """Called when vertices are are updated.

        Same signature as ``DynamicMesh.update_vertices``, except for the ``old`` arg.
        """
        pass

    # Bit of additional API, leaning somewhat towards GPU usage

    def new_faces_buffer(self, mesh):
        """Called when a new buffer is allocated to store the faces.

        This happens when more memory is needed to store the faces, or
        when faces are deleted and the buffer can be made smaller.

        The mesh is passed as an argument so the tracker has full access
        to it to address this situation. The new buffer array can be
        obtained via ``mesh.faces.base``.
        """
        pass

    def new_vertices_buffer(self, mesh):
        """Called when new buffers are allocated to store the vertices.

        This happens when more memory is needed to store the vertices, or
        when vertices are deleted and the buffers can be made smaller.

        The mesh is passed as an argument so the tracker has full access
        to it to address this situation. The new buffer array can be
        obtained via ``mesh.vertices.base``.
        """
        pass

    def update_normals(self, indices):
        """Called when the given normals have changed.

        It's not enought to use ``update_vertices``, because when a vertex
        position changes, it also affects the normals of neighbouring vertices.

        For reasons of performance and simplicity, the new normals are
        not provided as an argument. If needed, store the normals buffer in
        ``new_vertices_buffer`` and then sample the values from that.
        """
        pass


class MeshLogger(MeshChangeTracker):
    """A simple logger that produces textual messages about changes to the mesh."""

    def __init__(self, print_func):
        self.print = print_func

    def add_faces(self, faces):
        self.print(f"Adding {len(faces)} faces.")

    def pop_faces(self, n, old):
        self.print(f"Removing {n} faces.")

    def update_faces(self, indices, faces, old):
        self.print(f"Updating {len(indices)} faces.")

    def add_vertices(self, positions):
        self.print(f"Adding {len(positions)} vertices.")

    def pop_vertices(self, n, old):
        self.print(f"Removing {n} vertices.")

    def update_vertices(self, indices, positions, old):
        self.print(f"Updating {len(indices)} vertices.")


class MeshUndoTracker(MeshChangeTracker):
    """A mesh change tracker functioning as an undo stack."""

    def init(self, mesh):
        self._doing = None
        self._undo = []
        self._redo = []

    def add_faces(self, faces):
        self._append(("pop_faces", len(faces)))

    def pop_faces(self, n, old):
        self._append(("add_faces", old))

    def swap_faces(self, indices1, indices2):
        self._append(("swap_faces", indices2, indices1))

    def update_faces(self, indices, faces, old):
        self._append(("update_faces", indices, old))

    def add_vertices(self, positions):
        self._append(("pop_vertices", len(positions)))

    def pop_vertices(self, n, old):
        self._append(("add_vertices", old))

    def swap_vertices(self, indices1, indices2):
        self._append(("swap_vertices", indices2, indices1))

    def update_vertices(self, indices, positions, old):
        self._append(("update_vertices", indices, old))

    def _append(self, step):
        # Incoming changes pass through here.
        # Normally they are added to the undo stack, and clear the redo stack.
        # But if we are un-doing, it should be added to the redo stack instead.
        # And if we are re-doing, it's just added to the undo stack.
        if not self._doing:
            self._undo.append(step)
            self._redo.clear()
        elif self._doing == "undo":
            self._redo.append(step)
        elif self._doing == "redo":
            self._undo.append(step)

    def get_version(self):
        """Get the current 'version' of the mesh."""
        return len(self._undo)

    def apply_version(self, dynamic_mesh, version):
        """Apply the given version. The given mesh must be the same as the mesh being tracked."""
        while self._undo and version < len(self._undo):
            self._do(dynamic_mesh, self._undo.pop(-1), "undo")
        while self._redo and version > len(self._undo):
            self._do(dynamic_mesh, self._redo.pop(-1), "redo")

    def _do(self, dynamic_mesh, step, what):
        self._doing = what
        try:
            method_name, *args = step
            f = getattr(dynamic_mesh, method_name)
            f(*args)
        finally:
            self._doing = None

    def undo(self, dynamic_mesh):
        """Undo the last change."""
        self.apply_version(dynamic_mesh, self.get_version() - 1)

    def redo(self, dynamic_mesh):
        """Redo the last undone change."""
        self.apply_version(dynamic_mesh, self.get_version() + 1)


# Commented, because we cannot have a reference to DynamicMesh because
# of circular imports. But this is all we need to create a mesh that
# replicates another. Probably a useless use-case, but it does
# illustrate the elegance of the change tracker API.
#
# class ReplicatingMesh(DynamicMesh, MeshChangeTracker):
#     pass
