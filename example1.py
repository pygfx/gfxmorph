import numpy as np
import pygfx as gfx  # noqa
from gfxmorph.maybe_pygfx import smooth_sphere_geometry, DynamicMeshGeometry  # noqa
from gfxmorph import DynamicMesh, MeshChangeTracker  # noqa

# geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
# geo = gfx.sphere_geometry(1)
# geo = gfx.geometries.tetrahedron_geometry()
# geo = smooth_sphere_geometry(1)
# m = DynamicMesh(geo.positions.data, geo.indices.data)

# Pseudo code:
# m = DynamicMesh() or higher level so we have manifold props etc.
# morph = MorphLogic(m)
# geo = DynamicMeshGeometry()
# m.track_changes(geo)


m = DynamicMesh(None, None)


def add_mesh():
    geo = smooth_sphere_geometry()
    vertices, faces = geo.positions.data, geo.indices.data
    vertices += np.random.normal(0, 5, size=(3,))

    m.add_mesh(vertices, faces)

    if d:
        d.camera.show_object(d.scene)
        d.renderer.request_draw()


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

    d.camera.show_object(d.scene)
    d.renderer.request_draw()


def breakit():
    # Remove a face, use m.repair_closed() to fix!
    m._data.delete_faces(np.random.randint(0, len(m.faces)))


d = None
for i in range(0):
    add_mesh()


mesh = gfx.Mesh(
    DynamicMeshGeometry(),  # gfx.geometries.DynamicMeshGeometry
    gfx.materials.MeshPhongMaterial(color="red", flat_shading=False, side="FRONT"),
)

m.track_changes(mesh.geometry)

camera = gfx.OrthographicCamera()
camera.show_object((0, 0, 0, 8))
d = gfx.Display(camera=camera)
d.show(mesh)
