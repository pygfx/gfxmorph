import numpy as np
import pylinalg as la
import pygfx as gfx

# todo: why not make all polyhedrons in gfx solid, and use flat-shading to show their sharp edges?


def solid_tetrahedon():
    """The smallest/simplest possible mesh that contains a volume.
    It is a closed orientable manifold.

    """
    vertices = np.array(
        [
            [-1, 0, -1 / 2**0.5],
            [+1, 0, -1 / 2**0.5],
            [0, -1, 1 / 2**0.5],
            [0, +1, 1 / 2**0.5],
        ],
        dtype=np.float32,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[..., None]

    faces = np.array(
        [
            [2, 0, 1],
            [0, 2, 3],
            [1, 0, 3],
            [2, 1, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


# todo: solid / closed / something else?
def solid_sphere_geometry(
    radius=1.0, max_edge_length=None, recursive_subdivisions=None
):
    """Generate a solid sphere.

    Creates a sphere that has its center in the local origin. The sphere
    consists of 60 regular triangular faces and 32 vertices (a Pentakis
    Dodecahedron). The triangles are subdivided if necessary to create
    a smoother surface.

    This geometry differs from the `sphere_geometry` in that it's
    solid/closed. This means that it consists of a single contiguous
    surface that encloses the space inside. The faces are also
    distributed evenly over the surface, which means less vertices are
    needed to create a rounder surface. The downside is that one cannot
    easily apply a 2D texture map to this geometry.

    The resulting mesh is an orientable 2-manifold without boundaries, and is regular: it consists of triangles
    of equal size, and each triangle has the same number of incident faces.

    Parameters
    ----------
    radius : float
        The radius of the sphere. Vertices are placed at this distance around
        the local origin.
    max_edge_length : float | None
        If given and not None, it is used to calculate the `recursive_subdivisions`.
        The faces will be recursively subdivided until the length of each edge
        is no more than this value (taking the given radius into account).
    recursive_subdivisions : int | None
        The number of times to recursively subdivide the faces. The total number of faces
        will be ``60 * 4 ** recursive_subdivisions``. Default 0.

    Returns
    -------
    sphere : Geometry
        A geometry object that represents a sphere.

    """

    # Dev notes:
    #
    # The idea of this function is to take a polyhedron, and subdivide it by
    # splitting each edge in two, turning each face into 4 smaller faces.
    # In the image below the triangle a-b-c is subdivided:
    #
    #     c
    #     /\
    # ca /__\ bc
    #   /\  /\
    #  /__\/__\
    # a   ab   b
    #
    # Imagine this to be one triangle on a polyhedron. One can see that
    # the new vertices are surrounded by 6 faces (3 in triangle a-b-c,
    # and 3 more in the neighbouring triangle). One can also see that
    # for the original vertices (e.g. a), the number of surrounding
    # faces does not change by the subdividing process. It will only
    # be defined by the original geometry.
    #
    # Therefore, in order to create a geometry that remains truely
    # regular after subdividing it, we need to start with a polyhedron,
    # consisting of triangles, where each vertex is shared by 6 faces.
    # As far as I know, the Pentakis Dodecahedron (a.k.a.
    # Kisdodecahedron) is the smallest polyhedron that meets these
    # criteria. E.g. subdividing a Tetrahedron will result in rings
    # around the original vertices. An Icosahedron (which has 5 faces
    # around each vertex) would produce similar artifacts.
    #
    # The values of vertices and faces was taken from:
    # Source: http://dmccooey.com/polyhedra/PentakisDodecahedron.txt

    # Check radius
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Radius must be larger than zero.")

    # Determine the number of subdivisions
    if max_edge_length is not None and recursive_subdivisions is not None:
        raise ValueError(
            "Either max_edge_length or recursive_subdivisions must be given, or none, but not both."
        )
    elif max_edge_length is not None:
        if max_edge_length <= 0:
            raise ValueError("max_edge_length must be larger than zero.")
        # Emulate the effect of the subdivision process on two neighbouring vertices on the mesh
        a = np.array([+0.35682207, 0.93417233, 0.0], np.float64)
        b = np.array([-0.35682207, 0.93417233, 0.0], np.float64)
        subdivisions = 0
        while la.vec_dist(a, b) * radius > max_edge_length:
            b = 0.5 * (a + b)
            b /= np.linalg.norm(b)
            subdivisions += 1
    elif recursive_subdivisions is not None:
        subdivisions = max(0, int(recursive_subdivisions))
    else:
        subdivisions = 0

    # Calculate number of faces and vertices
    nfaces = 60 * 4**subdivisions
    nvertices = nfaces // 2 + 2  # weird, but true

    # Allocate arrays
    vertices = np.empty((nvertices, 3), np.float32)
    faces = np.empty((nfaces, 3), np.int32)

    # These keep track while fillting the above arrays
    vertex_index = 0
    face_index = 0

    # A dict to enable faces to reuse vertices
    index_map = {}

    def add_vertex(xyz):
        nonlocal vertex_index
        vertices[vertex_index] = xyz
        vertex_index += 1

    def add_face(i1, i2, i3):
        nonlocal face_index
        faces[face_index] = i1, i2, i3
        face_index += 1

    def define_face(ia, ib, ic, div=subdivisions):
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        if div <= 0:
            # Store faces here
            add_face(ia, ib, ic)
        else:
            # Create new vertices (or reuse them)
            # new point on edge a-b
            key_ab = min(ia, ib), max(ia, ib)
            iab = index_map.get(key_ab, None)
            if iab is None:
                index_map[key_ab] = iab = vertex_index
                ab = (a + b) * 0.5
                add_vertex(ab / np.linalg.norm(ab))
            # new point on edge b-c
            key_bc = min(ib, ic), max(ib, ic)
            ibc = index_map.get(key_bc, None)
            if ibc is None:
                index_map[key_bc] = ibc = vertex_index
                bc = (b + c) * 0.5
                add_vertex(bc / np.linalg.norm(bc))
            # new point on edge c-a
            key_ca = min(ic, ia), max(ic, ia)
            ica = index_map.get(key_ca, None)
            if ica is None:
                index_map[key_ca] = ica = vertex_index
                ca = (c + a) * 0.5
                add_vertex(ca / np.linalg.norm(ca))
            # Recurse to define the faces
            define_face(ia, iab, ica, div - 1)
            define_face(ib, ibc, iab, div - 1)
            define_face(ic, ica, ibc, div - 1)
            define_face(iab, ibc, ica, div - 1)

    C0 = 0.927050983124842272306880251548  # == 3 * (5**0.5 - 1) / 4
    C1 = 1.33058699733550141141687582919  # == 9 * (9 + 5**0.5) / 76
    C2 = 2.15293498667750705708437914596  # == 9 * (7 + 5 * 5**0.5) / 76
    C3 = 2.427050983124842272306880251548  # == 3 * (1 + 5**0.5) / 4

    # Add vertices of the Pentakis Dodecahedron
    add_vertex((0.0, C0, C3))
    add_vertex((0.0, C0, -C3))
    add_vertex((0.0, -C0, C3))
    add_vertex((0.0, -C0, -C3))
    add_vertex((C3, 0.0, C0))
    add_vertex((C3, 0.0, -C0))
    add_vertex((-C3, 0.0, C0))
    add_vertex((-C3, 0.0, -C0))
    add_vertex((C0, C3, 0.0))
    add_vertex((C0, -C3, 0.0))
    add_vertex((-C0, C3, 0.0))
    add_vertex((-C0, -C3, 0.0))
    add_vertex((C1, 0.0, C2))
    add_vertex((C1, 0.0, -C2))
    add_vertex((-C1, 0.0, C2))
    add_vertex((-C1, 0.0, -C2))
    add_vertex((C2, C1, 0.0))
    add_vertex((C2, -C1, 0.0))
    add_vertex((-C2, C1, 0.0))
    add_vertex((-C2, -C1, 0.0))
    add_vertex((0.0, C2, C1))
    add_vertex((0.0, C2, -C1))
    add_vertex((0.0, -C2, C1))
    add_vertex((0.0, -C2, -C1))
    add_vertex((1.5, 1.5, 1.5))
    add_vertex((1.5, 1.5, -1.5))
    add_vertex((1.5, -1.5, 1.5))
    add_vertex((1.5, -1.5, -1.5))
    add_vertex((-1.5, 1.5, 1.5))
    add_vertex((-1.5, 1.5, -1.5))
    add_vertex((-1.5, -1.5, 1.5))
    add_vertex((-1.5, -1.5, -1.5))

    # The vertices are not on the unit sphere, they seem to not even
    # be exactly on the same sphere. So we push them to the unit sphere.
    lengths = np.linalg.norm(vertices[:vertex_index], axis=1)
    vertices[:vertex_index, 0] /= lengths
    vertices[:vertex_index, 1] /= lengths
    vertices[:vertex_index, 2] /= lengths

    # Apply the faces of the Pentakis Dodecahedron.
    # Except that these may recurse to create sub-faces.
    define_face(12, 0, 2)
    define_face(12, 2, 26)
    define_face(12, 26, 4)
    define_face(12, 4, 24)
    define_face(12, 24, 0)
    define_face(13, 3, 1)
    define_face(13, 1, 25)
    define_face(13, 25, 5)
    define_face(13, 5, 27)
    define_face(13, 27, 3)
    define_face(14, 2, 0)
    define_face(14, 0, 28)
    define_face(14, 28, 6)
    define_face(14, 6, 30)
    define_face(14, 30, 2)
    define_face(15, 1, 3)
    define_face(15, 3, 31)
    define_face(15, 31, 7)
    define_face(15, 7, 29)
    define_face(15, 29, 1)
    define_face(16, 4, 5)
    define_face(16, 5, 25)
    define_face(16, 25, 8)
    define_face(16, 8, 24)
    define_face(16, 24, 4)
    define_face(17, 5, 4)
    define_face(17, 4, 26)
    define_face(17, 26, 9)
    define_face(17, 9, 27)
    define_face(17, 27, 5)
    define_face(18, 7, 6)
    define_face(18, 6, 28)
    define_face(18, 28, 10)
    define_face(18, 10, 29)
    define_face(18, 29, 7)
    define_face(19, 6, 7)
    define_face(19, 7, 31)
    define_face(19, 31, 11)
    define_face(19, 11, 30)
    define_face(19, 30, 6)
    define_face(20, 8, 10)
    define_face(20, 10, 28)
    define_face(20, 28, 0)
    define_face(20, 0, 24)
    define_face(20, 24, 8)
    define_face(21, 10, 8)
    define_face(21, 8, 25)
    define_face(21, 25, 1)
    define_face(21, 1, 29)
    define_face(21, 29, 10)
    define_face(22, 11, 9)
    define_face(22, 9, 26)
    define_face(22, 26, 2)
    define_face(22, 2, 30)
    define_face(22, 30, 11)
    define_face(23, 9, 11)
    define_face(23, 11, 31)
    define_face(23, 31, 3)
    define_face(23, 3, 27)
    define_face(23, 27, 9)

    # Double-check that the expected numbers match the real ones
    assert nfaces == face_index
    assert nvertices == vertex_index

    # Return as a geometry
    return gfx.Geometry(positions=vertices * radius, indices=faces, normals=vertices)


if __name__ == "__main__":
    m = gfx.Mesh(
        solid_sphere_geometry(1, None, 2),
        gfx.MeshPhongMaterial(wireframe=False, wireframe_thickness=3),
    )
    m.material.side = "FRONT"

    gfx.show(m)
