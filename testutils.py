import maybe_pygfx
import pygfx as gfx


def get_tetrahedron():
    """A closed tetrahedron as simple list objects, so we can easily add stuff to create corrupt meshes."""
    vertices = [
        [-1, 0, -1 / 2**0.5],
        [+1, 0, -1 / 2**0.5],
        [0, -1, 1 / 2**0.5],
        [0, +1, 1 / 2**0.5],
    ]
    faces = [
        [2, 0, 1],
        [0, 2, 3],
        [1, 0, 3],
        [2, 1, 3],
    ]
    return vertices, faces, True


def get_sphere():
    geo = maybe_pygfx.smooth_sphere_geometry()
    return geo.positions.data.tolist(), geo.indices.data.tolist(), True


def get_knot():
    geo = gfx.geometries.torus_knot_geometry(
        tubular_segments=16, radial_segments=5, stitch=True
    )
    return geo.positions.data.tolist(), geo.indices.data.tolist(), True


def get_quad():
    vertices = [
        [-1, -1, 0],
        [+1, -1, 0],
        [+1, +1, 0],
        [-1, +1, 0],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
    ]
    return vertices, faces, False


def get_fan():
    vertices = [
        [0, 0, 0],
        [-1, -1, 0],
        [+1, -1, 0],
        [+1, +1, 0],
        [-1, +1, 0],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [0, 1, 4],
    ]
    return vertices, faces, False


def get_strip():
    # A strip of 3 triangles
    vertices, faces, _ = get_fan()
    faces.pop(-1)
    return vertices, faces, False


def iter_fans():
    """Produce a series of fans to test stuff on."""

    # An open fan with 10 faces
    open_fan = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [0, 5, 4],
        [0, 6, 5],
        [0, 7, 6],
        [0, 8, 7],
        [0, 9, 8],
        [0, 10, 9],
        [0, 11, 10],
    ]

    # Open fans
    for i in range(1, 11):
        yield [x for x in open_fan[:i]]

    # Closed fans
    for i in range(2, 11):
        fan = [x for x in open_fan[:i]]
        fan.append([0, 1, fan[-1][1]])
        yield fan


def iter_test_meshes():
    yield get_tetrahedron()  # 4 vertices, 4 faces
    yield get_sphere()  # 32 vertices, 60 faces
    yield get_knot()  # 80 vertices, 160 faces
    yield get_quad()  # 4 vertices, 2 faces
    yield get_fan()  # 5 vertices, 4 faces
    yield get_strip()  # 5 vertices, 3 faces


def iter_closed_meshes():
    yield get_tetrahedron()  # 4 vertices, 4 faces
    yield get_sphere()  # 32 vertices, 60 faces
    yield get_knot()  # 80 vertices, 160 faces
