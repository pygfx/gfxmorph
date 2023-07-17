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
from gfxmorph import DynamicMesh
from wgpu.gui.auto import WgpuCanvas
import observ.store

# -- Functions to modify the mesh


def add_mesh():
    geo = smooth_sphere_geometry()
    vertices, faces = geo.positions.data, geo.indices.data
    vertices += np.random.normal(0, 5, size=(3,))

    m.add_mesh(vertices, faces)

    camera.show_object(scene)
    renderer.request_draw()

    store.set_mesh_state(stack.save())


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

    store.set_mesh_state(stack.save())


def breakit():
    m.delete_faces(np.random.randint(0, len(m.faces)))

    store.set_mesh_state(stack.save())


def repair():
    m.repair(True)

    store.set_mesh_state(stack.save())


# -- Setup the undo/redo

# We use observ to store the mesh state, but this state is only a
# "version number". The real state is stored in the simple Stack class
# below. We store the full arrays of vertices and faces and reset them
# when undo/redo is invoked.
#
# We should also think whether we want to facilitate undo as part of
# the basedynamicmesh or a utility of sorts, or whether we just show
# how to do it in an example.
#
# ### Taking a full-mesh snapshot at each undo-version
#
# This is a simple and effective solution, that works well in principle,
# but can cost a lot of memory for a deep undo stack.
#
# A mesh of 100k vertices takes about 8MiB, so with 100 actions you
# reach 800MiB. Seems large but manageable, though if any of these
# number become higher, memory becoming a problem is not far off.
#
# Exporting the mesh by making copies of the data does not take a lot
# of time. However, resetting to that data costs about a quarter of a
# second for a mesh with 100k vertices. Most of that is spent on
# recreating the vertex2faces array and normals. Again manageable, but
# if the mesh gets larger this delay may become anoying.
#
# A solution to mitigate the memory problem is to offload older
# snapshots to disk, to a temporary file. The time-aspect can be
# mitigated by allowing the export to create an opaque object that
# includes some internal state (like the vertices2faces and the
# normals). This does introduce some complexity as to how to serialize
# all that data ...
#
# A more sophisticated solution would be to store objects that
# represents diffs that can be applied to travel between states of the
# mesh. This should be more memory efficient, especially in the case
# with an undo stack of many small changes, as can be expected with
# morphing.


class MeshStack:
    """Simply a stack of full vertices and faces arrays."""

    def __init__(self):
        self.index = 0
        self.stack = [(None, None)]

    def save(self):
        self.index += 1
        self.stack[self.index :] = []
        self.stack.append((m.vertices.copy(), m.faces.copy()))
        return self.index

    def apply(self, i):
        if i == self.index:
            return
        self.index = i
        vertices, faces = self.stack[self.index]
        m.reset(vertices=vertices, faces=faces)


stack = MeshStack()

# A Store object has undo functionality (in contrast to a normal state
# object obtained with observ.reactive). But it's readonly so we need to
# subclass it to make it settable.


class Store(observ.store.Store):
    """The observ store to track thet 'mesh state'."""

    @observ.store.mutation
    def set_mesh_state(self, val):
        self.state["mesh_state"] = val

    @observ.store.computed
    def mesh_state(self):
        return self.state["mesh_state"]


store = Store({"mesh_state": 0})

# The watch() function re-calls the first function whenever any
# observables that it uses change, and then passes the result to the
# second function. I guess immediate determines when this happens?
_ = observ.watch(
    lambda: store.mesh_state,
    lambda val: stack.apply(val),  # noqa: T201
    sync=True,
    immediate=True,
)

# -- Setup the mesh

# Create the dynamicmesh instance that we will work with
m = DynamicMesh(None, None)

# Create the geometry object, and set it up to receive updates from the DynamicMesh
geo = DynamicMeshGeometry()
m.track_changes(geo)

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
