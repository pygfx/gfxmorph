import numpy as np


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

    def create_faces(self, faces):
        """Called when faces are added to the mesh.

        Same signature as ``DynamicMesh.create_faces``.
        """
        pass

    def delete_last_faces(self, n, old):
        """Called when faces are are removed from the mesh.

        Same signature as ``DynamicMesh.delete_last_faces``, except for the ``old`` arg.
        Note that calling ``delete_faces()`` on the mesh results in a
        ``swap_faces()`` and a ``delete_last_faces()``.
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

    def create_vertices(self, positions):
        """Called when vertices are added to the mesh.

        Same signature as ``DynamicMesh.create_vertices``.
        """
        pass

    def delete_last_vertices(self, n, old):
        """Called when vertices are are removed from the mesh.

        Same signature as ``DynamicMesh.delete_last_vertices``, except for the ``old`` arg.
        Note that calling ``delete_vertices()`` on the mesh results in a
        ``swap_vertices()`` and a ``delete_last_vertices()``.
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

        This happens when more memory is needed to store the vertices
        (positions and normals), or when vertices are deleted and the
        buffers can be made smaller.

        The mesh is passed as an argument so the tracker has full access
        to it to address this situation. The new buffer array can be
        obtained via ``mesh.positions.base``.
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

    def create_faces(self, faces):
        self.print(f"Adding {len(faces)} faces.")

    def delete_last_faces(self, n, old):
        self.print(f"Removing {n} faces.")

    def update_faces(self, indices, faces, old):
        self.print(f"Updating {len(indices)} faces.")

    def create_vertices(self, positions):
        self.print(f"Adding {len(positions)} vertices.")

    def delete_last_vertices(self, n, old):
        self.print(f"Removing {n} vertices.")

    def update_vertices(self, indices, positions, old):
        self.print(f"Updating {len(indices)} vertices.")


class MeshUndoTracker(MeshChangeTracker):
    """A mesh change tracker functioning as an undo stack.

    To create a new version, call ``commit()``. The new version contains
    all changes since the last commit. It is recommended to make commits
    by using this object as a context manager. It will then prevent
    (unintended) commits until the context exits.
    """

    def init(self, mesh):
        self._work_in_progress = False
        self._pending = []
        self._undo = []
        self._redo = []
        self._stack_level = 0

    def __enter__(self):
        # Can re-enter, but only the first context counts
        self._stack_level += 1
        return self

    def __exit__(self, *args):
        self._stack_level -= 1
        if self._stack_level <= 0:
            self._stack_level = 0
            self.commit()

    def create_faces(self, faces):
        self._append(("delete_last_faces", len(faces)))

    def delete_last_faces(self, n, old):
        self._append(("create_faces", old))

    def swap_faces(self, indices1, indices2):
        self._append(("swap_faces", indices2, indices1))

    def update_faces(self, indices, faces, old):
        self._append(("update_faces", indices, old))

    def create_vertices(self, positions):
        self._append(("delete_last_vertices", len(positions)))

    def delete_last_vertices(self, n, old):
        self._append(("create_vertices", old))

    def swap_vertices(self, indices1, indices2):
        self._append(("swap_vertices", indices2, indices1))

    def update_vertices(self, indices, positions, old):
        self._append(("update_vertices", indices, old))

    def _append(self, step):
        # See if we can merge.
        # A common case is that the mesh is deformed interactively,
        # resulting in many updates to the same set of vertices. We can
        # easily detect this case. We can then simply drop the new
        # update, because the previous undo-step undoes it up to the
        # beginning.
        if len(self._pending) > 0:
            last_step = self._pending[-1]
            if step[0] == "update_vertices" and last_step[0] == "update_vertices":
                indices, last_indices = step[1], last_step[1]
                # If the application uses the same i32 ndarray to update the vertices,
                # we can make this test fast. Otherwise, we need to testa bit more.
                if indices is last_indices:
                    return
                elif len(indices) == len(last_indices):
                    if np.all(indices == last_indices):
                        return

        # Add to staging list
        self._pending.append(step)

    def get_version(self):
        """Get the current 'version' of the mesh.

        The version is an integer that increases with each version.
        """
        return len(self._undo)

    def has_pending_changes(self):
        """Get whether there are pending changes that can be comitted or cancelled."""
        return len(self._pending) > 0

    def commit(self):
        """Save the current state as a new version, and return the new version number.

        In other words, this commits the pending changes to the undo stack.
        If the object is currently used as a context, this does nothing.
        """
        if not (self._work_in_progress or self._stack_level > 0):
            self._undo.append(self._pending)
            self._pending = []
            self._redo.clear()
        return self.get_version()

    def commit_amend(self):
        """Add the current state to the latest version, and return the version number.

        In other words, this commits the pending changes to the undo stack,
        but instead of making a new entry, it's appended to the last entry.
        If the object is currently used as a context, this does nothing.
        """
        if not self._undo:
            return self._commit()
        if not (self._work_in_progress or self._stack_level > 0):
            self._undo[-1].extend(self._pending)
            self._pending = []
            self._redo.clear()
        return self.get_version()

    def cancel(self, dynamic_mesh):
        """Cancel any uncommited changes.

        Pending changes are lost.
        """
        if self._pending and not self._work_in_progress:
            dummy_target = []
            steps = self._pending
            self._pending = []
            self._do(dynamic_mesh, steps, dummy_target)
        else:
            self._pending = []

    def undo(self, dynamic_mesh):
        """Undo the changes of the last comitted version.

        If there are pending changes, these are cancelled first. The
        mesh is then reverted to the previous version (the version
        before the last committed version). If the undo-stack is empty
        (i.e. there are no changes to undo), this step does nothing.
        """
        self.apply_version(dynamic_mesh, self.get_version() - 1)

    def redo(self, dynamic_mesh):
        """Redo the last undone change.

        If there are pending changes, these are cancelled first. The
        mesh is then reverted to the version that was last undo. If the
        redo-stack is empty (i.e. if a new commit has been made since
        the last undo) this step does nothing.
        """
        self.apply_version(dynamic_mesh, self.get_version() + 1)

    def apply_version(self, dynamic_mesh, version):
        """Apply the given version.

        If there are pending changes, these are cancelled first. If the
        version is either the current version or out of range, this
        step does nothing. The given mesh must be the same as the mesh
        being tracked.
        """
        if self._stack_level > 0:
            raise RuntimeError(
                "Cannot undo/redo while the MeshUndoTracker is used as a context."
            )
        self.cancel(dynamic_mesh)
        while self._undo and version < len(self._undo):
            self._do(dynamic_mesh, self._undo.pop(-1), self._redo)
        while self._redo and version > len(self._undo):
            self._do(dynamic_mesh, self._redo.pop(-1), self._undo)

    def _do(self, dynamic_mesh, steps, target):
        assert len(self._pending) == 0
        self._work_in_progress = True
        try:
            for step in reversed(steps):
                method_name, *args = step
                f = getattr(dynamic_mesh, method_name)
                f(*args)
        finally:
            target.append(self._pending)
            self._pending = []
            self._work_in_progress = False


# class MeshUndoTracker(MeshChangeTracker):
#     """A mesh change tracker functioning as an undo stack."""
#
#     def init(self, mesh):
#         # self._doing = None
#         self._undo = []
#         self._redo = []
#         self._stack = None
#         self._stack_level = 0
#
#     def __enter__(self):
#         # Can re-enter, but only the first context counts
#         self._stack_level += 1
#         if self._stack is None:
#             self._stack = []
#         return self
#
#     def __exit__(self, *args):
#         self._stack_level -= 1
#         if self._stack_level <= 0:
#             self._stack_level = 0
#             if self._stack is not None:
#                 if len(self._stack):
#                     self._undo.append(self._stack)
#                     self._redo.clear()
#                 self._stack = None
#
#     def create_faces(self, faces):
#         self._append(("delete_last_faces", len(faces)))
#
#     def delete_last_faces(self, n, old):
#         self._append(("create_faces", old))
#
#     def swap_faces(self, indices1, indices2):
#         self._append(("swap_faces", indices2, indices1))
#
#     def update_faces(self, indices, faces, old):
#         self._append(("update_faces", indices, old))
#
#     def create_vertices(self, positions):
#         self._append(("delete_last_vertices", len(positions)))
#
#     def delete_last_vertices(self, n, old):
#         self._append(("create_vertices", old))
#
#     def swap_vertices(self, indices1, indices2):
#         self._append(("swap_vertices", indices2, indices1))
#
#     def update_vertices(self, indices, positions, old):
#         self._append(("update_vertices", indices, old))
#
#     def _append(self, step):
#         if self._stack is None:
#             # If there is no active stack, we simply add the one step to the undo list.
#             # create a new stack, add to that, add to undo.
#             self._undo.append([step])
#             self._redo.clear()
#
#         else:
#             # See if we can merge
#             if len(self._stack) > 0:
#                 last_step = self._stack[-1]
#                 if last_step[0] == "update_vertices" and step[0] == "update_vertices":
#                     print(last_step[1].__class__.__name__, step[1].__class__.__name__, last_step[1] is step[1])
#
#             self._stack.append(step)
#
#     def get_version(self):
#         """Get the current 'version' of the mesh."""
#         return len(self._undo)
#
#     def apply_version(self, dynamic_mesh, version):
#         """Apply the given version. The given mesh must be the same as the mesh being tracked."""
#         if self._stack is not None:
#             raise RuntimeError("Cannot undo/redo while under a context.")
#         while self._undo and version < len(self._undo):
#             self._do(dynamic_mesh, self._undo.pop(-1), self._redo)
#         while self._redo and version > len(self._undo):
#             self._do(dynamic_mesh, self._redo.pop(-1), self._undo)
#
#     def _do(self, dynamic_mesh, steps, target):
#         assert self._stack is None
#         self._stack = []
#         try:
#             for step in reversed(steps):
#                 method_name, *args = step
#                 f = getattr(dynamic_mesh, method_name)
#                 f(*args)
#         finally:
#             target.append(self._stack)
#             self._stack = None
#
#     def undo(self, dynamic_mesh):
#         """Undo the last change.
#
#         This is more of an example, because in practice one "step" from
#         the application point of view likely consists of multiple raw steps.
#         """
#         self.apply_version(dynamic_mesh, self.get_version() - 1)
#
#     def redo(self, dynamic_mesh):
#         """Redo the last undone change."""
#         self.apply_version(dynamic_mesh, self.get_version() + 1)
#
#
#     def collect(self):
#         self._stack = []
#
#     def append_last(self):
#         if self._undo:
#             self._stack = self._undo.pop()
#
#     def merge_last(self):
#         if len(self._undo) < 2:
#             return
#
#         steps1 = self._undo[-2]  # Second to last -> the new last
#         steps2 = self._undo.pop(-1)  # Last
#
#         laststep = steps1[-1]
#         # for step in steps2:
#
#         steps1.extend(steps2)
#         print(len(steps1))

# Commented, because we cannot have a reference to DynamicMesh because
# of circular imports. But this is all we need to create a mesh that
# replicates another. Probably a useless use-case, but it does
# illustrate the elegance of the change tracker API.
#
# class ReplicatingMesh(DynamicMesh, MeshChangeTracker):
#     pass
