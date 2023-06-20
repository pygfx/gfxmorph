import time

import maybe_pygfx
import pygfx as gfx
import meshmorph
from meshmorph import AbstractMesh

from skcg.core.mesh import Mesh


class Timer:
    def __init__(self):
        self.measurements = []

    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self, title):
        elapsed = time.perf_counter() - self.t0
        self.add_data(title, f"{elapsed:0.3f}")

    def add_data(self, title, value):
        self.measurements.append((title, str(value)))


def iter_big_meshes():
    geo = maybe_pygfx.smooth_sphere_geometry(100, subdivisions=6)
    vertices = geo.positions.data
    faces = geo.indices.data
    yield "sphere", vertices, faces

    geo = gfx.geometries.torus_knot_geometry(100, 20, 10000, 12, stitch=True)
    vertices = geo.positions.data
    faces = geo.indices.data
    yield "knot", vertices, faces


def benchmark():
    columns = []

    for name, vertices, faces in iter_big_meshes():
        t = Timer()
        t.add_data("MESH", "")
        t.add_data("name", name)
        t.add_data("nvertices", len(vertices))
        t.add_data("nfaces", len(faces))
        t.add_data("", "")

        if True:
            t.add_data("NEW", "")
            t.tic()
            m = AbstractMesh(vertices, faces)
            t.toc("init")

            t.tic()
            # m.check_edge_manifold_and_closed()
            meshmorph.mesh_is_edge_manifold_and_closed(m.faces)
            t.toc("check e-manifold & closed")

            t.tic()
            # m.check_oriented()
            meshmorph.mesh_is_oriented(m.faces)
            t.toc("check oriented")

            t.tic()
            meshmorph.mesh_get_component_labels(m.faces, m._data._vertex2faces)
            t.toc("split components")

            t.tic()
            meshmorph.mesh_get_non_manifold_vertices(m.faces, m._data._vertex2faces)
            t.toc("check v-manifold")

            t.tic()
            v = m.get_volume()
            t.toc("volume")
            t.add_data("", v)

            t.add_data("", "")

            # t.tic()
            # i, d = m.get_closest_vertex((0, 0, 0))
            # verts = m.select_vertices_over_surface(i, 65)
            # t.toc("Select vertices")
            # t.add_data("", len(verts))

        if False:
            t.add_data("", "")
            t.add_data("--", "--")
            t.add_data("SKCG", "")

            t.tic()
            m2 = Mesh(vertices, faces)
            t.toc("init")

            t.tic()
            m2.is_manifold
            t.toc("is_manifold")

            t.tic()
            # m2.is_really_manifold
            t.toc("is_manifold full")

            t.tic()
            m2.is_closed
            t.toc("is_closed")

            t.tic()
            m2.is_oriented
            t.toc("is_oriented")

            m2 = Mesh(vertices, faces)

            t.tic()
            m2.split_connected_components()
            t.toc("Split components")

            t.tic()
            v = m2.computed_interior_volume
            t.toc("Volume")
            t.add_data("", v)

        columns.append(t.measurements)

    for row in zip(*columns):
        titles = [x[0] for x in row]
        assert len(set(titles)) == 1, "expected titles to match"
        print(titles[0].rjust(32), *[x[1].rjust(10) for x in row])


def benchmark_sphere():
    t = Timer()
    t.tic()
    maybe_pygfx.smooth_sphere_geometry(100, subdivisions=7)
    t.toc("Create smooth spere")
    print(t.measurements)


if __name__ == "__main__":
    benchmark()
    # benchmark_sphere()
