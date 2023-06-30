import pygfx as gfx  # noqa
from gfxmorph.maybe_pygfx import smooth_sphere_geometry  # noqa
from gfxmorph import AbstractMesh

# geo = gfx.torus_knot_geometry(tubular_segments=640, radial_segments=180)
# geo = gfx.sphere_geometry(1)
# geo = gfx.geometries.tetrahedron_geometry()
geo = smooth_sphere_geometry(1)
m = AbstractMesh(geo.positions.data, geo.indices.data)
