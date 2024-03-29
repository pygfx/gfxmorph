import time

import numpy as np
import pygfx as gfx
from gfxmorph.maybe_pygfx import smooth_sphere_geometry
from gfxmorph import DynamicMesh
from gfxmorph import meshfuncs
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
    geo = smooth_sphere_geometry(100, subdivisions=6)
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
            m = DynamicMesh(vertices, faces)
            t.toc("init")

            t.add_data("nbytes", m.metadata["approx_mem"])

            t.tic()
            # m.check_edge_manifold_and_closed()
            meshfuncs.mesh_is_edge_manifold_and_closed(m.faces)
            t.toc("check e-manifold & closed")

            t.tic()
            # m.check_oriented()
            meshfuncs.mesh_is_oriented(m.faces)
            t.toc("check oriented")

            t.tic()
            meshfuncs.mesh_get_component_labels(m.faces, m.vertex2faces)
            t.toc("split components")

            t.tic()
            meshfuncs.mesh_get_non_manifold_vertices(m.faces, m.vertex2faces)
            t.toc("check v-manifold")

            t.tic()
            # v = m.get_volume() -> slow because it checks for manifoldness, because a volume of a nonmanifold or nonmanifold mesh means nothing.
            v = meshfuncs.mesh_get_volume(m.positions, m.faces)
            t.toc("volume")

            t.tic()
            vertices, faces = m.export()
            t.toc("export")

            t.tic()
            m.reset(None, None)
            m.reset(vertices, faces)
            t.toc(f"reset")

            t.tic()
            m.delete_last_faces(len(m.faces))
            t.toc(f"pop faces")
            t.tic()
            m.delete_last_vertices(len(m.positions))
            t.toc(f"pop vertices")
            t.tic()
            m.create_vertices(vertices)
            t.toc(f"add vertices")
            t.tic()
            m.create_faces(faces)
            t.toc(f"add faces")

            t.tic()
            m.delete_faces(np.arange(0, len(m.faces), 2, np.int32))
            t.toc(f"delete 50% faces (swap and pop)")

            m.reset(vertices, None)
            t.tic()
            m.delete_vertices(np.arange(0, len(m.positions), 2, np.int32))
            t.toc(f"delete 50% vertices (swap and pop)")

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
    smooth_sphere_geometry(100, subdivisions=7)
    t.toc("Create smooth spere")
    print(t.measurements)


def benchmark_selecting_vertices():
    geo = smooth_sphere_geometry(100, subdivisions=5)
    vertices = geo.positions.data
    faces = geo.indices.data
    m = DynamicMesh(vertices, faces)

    for method in ["edge", "smooth1", "smooth2", "auto"]:
        t0 = time.perf_counter()
        verts, dists = m.select_vertices_over_surface(
            0, 0, 100, distance_measure=method
        )
        t1 = time.perf_counter()
        print(
            f"Selecting {len(verts)}/{len(m.positions)} vertices using {method}: {t1-t0:0.3f}"
        )


if __name__ == "__main__":
    # benchmark()
    # benchmark_sphere()
    benchmark_selecting_vertices()
