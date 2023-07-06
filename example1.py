import numpy as np
import pygfx as gfx  # noqa
from gfxmorph.maybe_pygfx import smooth_sphere_geometry  # noqa
from gfxmorph import DynamicMesh, AbstractMesh  # noqa

# geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
# geo = gfx.sphere_geometry(1)
# geo = gfx.geometries.tetrahedron_geometry()
# geo = smooth_sphere_geometry(1)
# m = AbstractMesh(geo.positions.data, geo.indices.data)

m = DynamicMesh()


def add_mesh():
    geo = smooth_sphere_geometry()
    vertices, faces = geo.positions.data, geo.indices.data
    vertices += np.random.normal(0, 5, size=(3,))
    faces += len(m.vertices)

    m.add_vertices(vertices)
    m.add_faces(faces)

    if d:
        d.camera.show_object(d.scene)
        d.renderer.request_draw()


# d = None
# for i in range(3):
#     add_mesh()


class DynamicMeshGeometry(gfx.Geometry):
    def __init__(self, dynamicmesh):
        super().__init__()
        dynamicmesh._update_receivers.append(self)
        self.positions = gfx.Buffer(
            dynamicmesh._positions_buf
        )  # todo: should not use private vars
        self.indices = gfx.Buffer(dynamicmesh._faces_buf)

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

    def update_vertices(self, indices):
        # todo: this is horribly inefficient
        self.positions.update_range()

    def update_faces(self, indices):
        self.indices.update_range()


mesh = gfx.Mesh(
    DynamicMeshGeometry(m),  # gfx.geometries.DynamicMeshGeometry
    gfx.materials.MeshBasicMaterial(color="red"),
)


d = gfx.Display()
d.show(mesh)
