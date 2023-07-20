"""
Example for demonstrating the dynamic mesh.

Key bindings:

* 1: add sphere mesh
* 2: add 2 spheres connected by a corrupt face
* b: remove a random face
* r: repair (note, can only repair small holes)
* m: print metadata

"""

import json

import numpy as np
import pygfx as gfx
from gfxmorph.maybe_pygfx import smooth_sphere_geometry, DynamicMeshGeometry
from gfxmorph import DynamicMesh, MeshChangeTracker
from wgpu.gui.auto import WgpuCanvas
import observ.store

# -- Functions to modify the mesh


def add_mesh():
    geo = smooth_sphere_geometry(subdivisions=1)
    vertices, faces = geo.positions.data, geo.indices.data
    vertices += np.random.normal(0, 5, size=(3,))

    m.add_mesh(vertices, faces)
    save_state()

    camera.show_object(scene)
    renderer.request_draw()


def add_broken_mesh():
    geo = smooth_sphere_geometry()
    vertices1, faces1 = geo.positions.data, geo.indices.data
    vertices1 += np.random.normal(0, 5, size=(3,))

    vertices2 = vertices1 + (2.2, 0, 0)
    faces2 = faces1 + len(vertices1)

    faces3 = [[faces2[10][0], faces2[10][1], faces1[20][0]]]

    m.add_mesh(
        np.concatenate([vertices1, vertices2]), np.concatenate([faces1, faces2, faces3])
    )

    camera.show_object(scene)
    renderer.request_draw()
    save_state()


def breakit():
    m.delete_faces(np.random.randint(0, len(m.faces)))
    save_state()


def repair():
    m.repair(True)
    save_state()


# -- Setup the undo/redo

# ### Implementing undo with diffs
#
# The idea is that for every action, the DynamicMesh gives us an object,
# by which it can be undone. When you undo a collection of actions, you
# get an object by which it can be re-done.
#
# It's very important that the actions are done in-order, and that no
# untracked actions are done in between.
#
# This approach is very efficient, as the data that is stored on the
# undo stack is quite minimal.
#
# However, undo steps do not represent "state", and will have to be
# tracked separately, I suppose.
#
# A modification could be that the undo-objects already contain the
# information to redo the operation. That would consume more memory,
# but then we may not need an external stack. Are there any advantages
# to that?


class MeshUndoTracker(MeshChangeTracker):
    def clear(self):
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
        if not self._doing:
            # Normally they are added to the undo stack, and clear the redo stack.
            self._undo.append(step)
            self._redo.clear()
        elif self._doing == "undo":
            # But if we are un-doing, it should be added to the redo stack instead.
            self._redo.append(step)
        elif self._doing == "redo":
            # And if we are re-doing, it's just added to the undo stack.
            self._undo.append(step)

    def get_version(self):
        return len(self._undo)

    def apply_version(self, version):
        while self._undo and version < len(self._undo):
            self.undo()
        while self._redo and version > len(self._undo):
            self.redo()

    def _do(self, step, what):
        self._doing = what
        try:
            method_name, *args = step
            f = getattr(m, method_name)
            f(*args)
        finally:
            self._doing = None

    def undo(self):
        if self._undo:
            self._do(self._undo.pop(-1), "undo")

    def redo(self):
        if self._redo:
            self._do(self._redo.pop(-1), "redo")


# class MeshUndoStack:
#     def __init__(self):
#         self._undo = []
#         self._redo = []
#
#     def append(self, undo_step):
#         self._undo.append(undo_step)
#         self._redo = []
#         return len(self._undo)
#
#     def apply_version(self, version):
#         while self._undo and version < len(self._undo):
#             self.undo()
#         while self._redo and version > len(self._undo):
#             self.redo()
#
#     def undo(self):
#         if self._undo:
#             undo_steps = self._undo.pop(-1)
#             undo_steps.reverse()
#             print([x[0] for x in undo_steps])
#             redo_steps = m.do(undo_steps)
#             self._redo.insert(0, redo_steps)
#
#     def redo(self):
#         if self._redo:
#             redo_steps = self._redo.pop(0)
#             redo_steps.reverse()
#             undo_steps = m.do(redo_steps)
#             self._undo.append(undo_steps)


# undo_stack = MeshUndoStack()
undo_tracker = MeshUndoTracker()


def save_state():
    # todo: if we also put the mesh stack object here, we can support hot reload.
    store.set_mesh_state(undo_tracker.get_version())
    # store.set_mesh_state(undo_stack.append(undo_step))


def restore_state(state):
    undo_tracker.apply_version(state)
    # undo_stack.apply_version(state)


# A Store object has undo functionality (in contrast to a normal state
# object obtained with observ.reactive). But it's readonly so we need to
# subclass it to make it settable.


class Store(observ.store.Store):
    """The observ store to track thet 'mesh state'."""

    @observ.store.mutation
    def set_mesh_state(self, state):
        self.state["mesh_state"] = state

    @observ.store.computed
    def mesh_state(self):
        return self.state["mesh_state"]


store = Store({"mesh_state": 0})

# The watch() function re-calls the first function whenever any
# observables that it uses change, and then passes the result to the
# second function. I guess immediate determines when this happens?
_ = observ.watch(
    lambda: store.mesh_state,
    restore_state,  # noqa: T201
    sync=True,
    immediate=True,
)

# -- Setup the mesh

# Create the dynamicmesh instance that we will work with
m = DynamicMesh(None, None)

# Create the geometry object, and set it up to receive updates from the DynamicMesh
geo = DynamicMeshGeometry()
m.track_changes(geo)
m.track_changes(undo_tracker)

# Two world objects, one for the front and one for the back (so we can clearly see holes)
mesh1 = gfx.Mesh(
    geo,
    gfx.materials.MeshPhongMaterial(color="green", flat_shading=False, side="FRONT"),
)
mesh2 = gfx.Mesh(
    geo,
    gfx.materials.MeshPhongMaterial(color="red", flat_shading=False, side="BACK"),
)


# -- Setup the viz

renderer = gfx.WgpuRenderer(WgpuCanvas())

camera = gfx.OrthographicCamera()
camera.show_object((0, 0, 0, 8))

scene = gfx.Scene()
scene.add(gfx.Background(None, gfx.BackgroundMaterial(0.4, 0.6)))
scene.add(camera.add(gfx.DirectionalLight()), gfx.AmbientLight())
scene.add(mesh1, mesh2)

# -- Make it responsive

print(__doc__)


@renderer.add_event_handler("key_down")
def on_key(e):
    if e.key == "1":
        print("Adding a sphere.")
        add_mesh()
    elif e.key == "2":
        print("Adding 2 spheres connected with a corrupt face.")
        add_broken_mesh()
    elif e.key == "h":
        print("Creating a hole.")
        breakit()
    elif e.key == "r":
        print("Repairing.")
        repair()
    elif e.key == "m":
        print("Metadata:")
        print(json.dumps(m.metadata, indent=2))
    elif e.key.lower() == "z" and ("Control" in e.modifiers or "Meta" in e.modifiers):
        if "Shift" in e.modifiers:
            store.redo()
        else:
            store.undo()


# -- Run

gfx.show(scene, camera=camera, renderer=renderer)
