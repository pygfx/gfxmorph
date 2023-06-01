import time

import maybe_pygfx
import pygfx as gfx
from meshmorph import AbstractMesh

from skcg.core.mesh import Mesh


class Timer:
    def tic(self):
        self.t0 = time.perf_counter()

    def toc(self, title):
        elapsed = time.perf_counter() - self.t0
        print(f"{title}: {elapsed:0.3f} s")


def benchmark_select_vertices_over_surface():
    t = Timer()

    # Consider the units to be mm, so we create a 10 cm spehere, with 1 mm edges
    t.tic()
    # geo = maybe_pygfx.solid_sphere_geometry(100, 10)
    geo = gfx.geometries.torus_knot_geometry(10, 2, 160, 400, stitch=True)
    # geo = gfx.geometries.klein_bottle_geometry(stitch=True)
    # geo = gfx.mobius_strip_geometry(stitch=True)
    vertices = geo.positions.data
    faces = geo.indices.data
    t.toc("Create mesh data")

    t.tic()
    sm = Mesh(vertices, faces)
    sm.is_closed
    sm.is_oriented
    sm.is_connected
    sm.fix_orientation()
    t.toc("create skcg mesh")
    t.tic()

    m = AbstractMesh(vertices, faces)
    t.toc("Instantiate closed mesh and validate")
    print(m.metadata)
    #
    # vi, _ = m.get_closest_vertex((-100, 0, 0))
    #
    # t.tic()
    # selected = m.select_vertices_over_surface(vi, 105)
    # t.toc(f"Select {len(selected)} vertices")

    # gfx.show(gfx.Line(geo, gfx.LineMaterial(thickness=3)))
    # gfx.show(gfx.Points(geo, gfx.PointsMaterial(size=9, opacity=0.8)))
    m = gfx.Mesh(
        geo,
        gfx.MeshPhongMaterial(wireframe=1, wireframe_thickness=2, flat_shading=False),
    )
    gfx.show(m)


if __name__ == "__main__":
    benchmark_select_vertices_over_surface()
