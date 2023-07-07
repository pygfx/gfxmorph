import numpy as np
import pygfx as gfx  # noqa
from gfxmorph.maybe_pygfx import smooth_sphere_geometry  # noqa
from gfxmorph import DynamicMesh, AbstractMesh  # noqa

# geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
# geo = gfx.sphere_geometry(1)
# geo = gfx.geometries.tetrahedron_geometry()
# geo = smooth_sphere_geometry(1)
# m = AbstractMesh(geo.positions.data, geo.indices.data)


m = AbstractMesh(None, None)


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
for i in range(1):
    add_mesh()


class DynamicMeshGeometry(gfx.Geometry):
    def __init__(self, abstract_mesh):
        super().__init__()
        abstract_mesh._data._update_receivers.append(self)
        self.positions = gfx.Buffer(
            abstract_mesh._data._positions_buf
        )  # todo: should not use private vars
        self.indices = gfx.Buffer(abstract_mesh._data._faces_buf)
        self.normals = gfx.Buffer(abstract_mesh._data._normals_buf)

    def set_vertex_arrays(self, positions, normals, colors):
        self.positions = gfx.Buffer(positions)
        self.normals = gfx.Buffer(normals)
        # self.colors = gfx.Buffer(colors)
        print("new vertices")

    def set_face_array(self, array):
        self.indices = gfx.Buffer(array)
        print("new faces")

    def set_n_vertices(self, n):
        pass  # no need

    def set_n_faces(self, n):
        print("number of faces", n)
        self.indices.view = 0, n
        self.normals.update_range()

    def update_positions(self, indices):
        # todo: optimize
        # We can update positions more fine-grained,
        # but the normals are actually updates in full.
        # Also, if we render with flat_shading, we don't need the normals!
        self.positions.update_range()

    def update_normals(self, indices):
        self.normals.update_range()

    def update_faces(self, indices):
        self.indices.update_range()


mesh = gfx.Mesh(
    DynamicMeshGeometry(m),  # gfx.geometries.DynamicMeshGeometry
    gfx.materials.MeshPhongMaterial(color="red", flat_shading=False, side="FRONT"),
)


camera = gfx.OrthographicCamera()
camera.show_object((0, 0, 0, 8))
d = gfx.Display(camera=camera)
d.show(mesh)
