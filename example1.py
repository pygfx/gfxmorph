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


# -- Functions to modify the mesh


def add_mesh():
    geo = smooth_sphere_geometry()
    vertices, faces = geo.positions.data, geo.indices.data
    vertices += np.random.normal(0, 5, size=(3,))

    m.add_mesh(vertices, faces)

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


def breakit():
    m.delete_faces(np.random.randint(0, len(m.faces)))


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
        m.repair(True)
    elif e.key == "m":
        print("Metadata:")
        print(json.dumps(m.metadata, indent=2))


# -- Run

gfx.show(scene, camera=camera, renderer=renderer)
