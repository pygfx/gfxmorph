import numpy as np
import pylinalg as la
import pygfx as gfx

from .basedynamicmesh import MeshChangeTracker


class DynamicMeshGeometry(gfx.Geometry, MeshChangeTracker):
    """A geometry class specifically for representing dynamic meshes.

    This class also inherits from ``gfxmorph.MeshChangeTracker`` so
    that the geometry can do precise updates to the GPU buffers when
    the mesh is changed dynamically.
    """

    def init(self, mesh):
        self._nverts = len(mesh.positions)
        self._nfaces = len(mesh.faces)
        self.new_vertices_buffer(mesh)
        self.new_faces_buffer(mesh)

    def new_vertices_buffer(self, mesh):
        self.positions = gfx.Buffer(mesh.positions.base)
        self.normals = gfx.Buffer(mesh.normals.base)
        # self.colors = gfx.Buffer(colors)

    def new_faces_buffer(self, mesh):
        self.indices = gfx.Buffer(mesh.faces.base)

    def add_faces(self, faces):
        old_n = self._nfaces
        self._nfaces += len(faces)
        self.indices.update_range(old_n, self._nfaces)
        self.indices.view = 0, self._nfaces

    def pop_faces(self, n, old):
        self._nfaces -= n
        self.indices.view = 0, self._nfaces

    def swap_faces(self, indices1, indices2):
        self.indices.update_range(indices1.min(), indices1.max())
        self.indices.update_range(indices2.min(), indices2.max())

    def update_faces(self, indices, faces, old):
        self.indices.update_range(indices.min(), indices.max())

    def add_vertices(self, positions):
        old_n = self._nverts
        self._nverts += len(positions)
        self.positions.update_range(old_n, self._nverts)

    def pop_vertices(self, n, old):
        self._nverts -= n

    def swap_vertices(self, indices1, indices2):
        self.positions.update_range(indices1.min(), indices1.max())
        self.positions.update_range(indices2.min(), indices2.max())

    def update_vertices(self, indices, positions, old):
        # todo: Optimize this, both here and on the pygfx side.
        # - We can update positions more fine-grained (using chunking).
        # - Consider an API where a mask is passed (e.g. positions.update_mask()),
        #   maybe this would be more efficient than using indices?
        # - If we render with flat_shading, we don't need the normals!
        #   So if the normal-updates are a bottleneck, it could be made optional.
        self.positions.update_range(indices.min(), indices.max())

    def update_normals(self, indices):
        self.normals.update_range(indices.min(), indices.max())


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


def subdivide_faces(vertices, faces):
    r""" Subdivide a mesh.

    This function subdivides the given faces, by dividing each triangle
    into 4 new triangles, as in the image below:

             /\
            /__\
           /\  /\
          /__\/__\

    The returned arrays must be processed by the caller. This is
    intentional, because it depends on the use-case how this is best
    done. E.g. the new vertices may be re-positioned a bit, or perhaps
    the subdivision was applied on a subset of the total faces, and
    merging the result requires some special indexing.

    Returns
    -------
        new_vertices : ndarray
            The new vertices that lie in the middle of the edges of the
            original faces. It is the responsibility of the caller to
            ``row_stack`` these to the original vertices.
        new_faces : ndarray
            The new faces. These should replace the given faces.
    """

    # First collect unique edges. We will create one new vertex in
    # the middle of each edge.
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    all_edges = edges.reshape(-1, 2)
    all_edges.sort(axis=1)

    # Find unique edges. We use a trick to get a 25% performance boost
    # unique_edges, reverse_index = np.unique(all_edges, axis=0, return_inverse=True)
    all_edges_buf = np.frombuffer(all_edges, dtype="V8")
    unique_edges_buf, reverse_index = np.unique(
        all_edges_buf, axis=0, return_inverse=True
    )
    unique_edges = np.frombuffer(unique_edges_buf, dtype=np.int32).reshape(-1, 2)

    # The most tricky step in this algorithm is finding the new
    # indices (on the new faces) based on the edges on which these
    # new vertices are placed. Using a Python dict (where the keys
    # are edges, represented as tuples of 2 indices) works, but
    # makes things slow.
    #
    # The trick: we use the reverse_index produced by `np.unique`
    # to create a map that has the same shape as the faces array,
    # but the indices along axis 1 represent edges instead of
    # vertices.
    indices_to_new_vertices = np.arange(len(unique_edges), dtype=np.int32) + len(
        vertices
    )
    face_edges_to_new_indices = indices_to_new_vertices[reverse_index]
    face_edges_to_new_indices.shape = -1, 3

    # Create new vertices on the middle of the edges.
    new_vertices = 0.5 * (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]])

    # We replace each triangle with 4 new triangles, like this.
    #
    #      v2
    #      /\
    #  e2 /__\ e1
    #    /\  /\
    #   /__\/__\
    # v0   e0   v1
    #
    smaller_faces = [
        # face 1
        faces[:, 0],
        face_edges_to_new_indices[:, 0],
        face_edges_to_new_indices[:, 2],
        # face 2
        faces[:, 1],
        face_edges_to_new_indices[:, 1],
        face_edges_to_new_indices[:, 0],
        # face 3
        faces[:, 2],
        face_edges_to_new_indices[:, 2],
        face_edges_to_new_indices[:, 1],
        # face 4
        face_edges_to_new_indices[:, 0],
        face_edges_to_new_indices[:, 1],
        face_edges_to_new_indices[:, 2],
    ]
    new_faces = np.column_stack(smaller_faces).reshape(-1, 3)

    return new_vertices, new_faces


def smooth_sphere_geometry(radius=1.0, max_edge_length=None, subdivisions=None):
    """Generate a sphere consisting of homogenous triangles.

    Creates a sphere that has its center in the local origin. The sphere
    consists of 60 regular triangular faces and 32 vertices (a Pentakis
    Dodecahedron). The triangles are subdivided if necessary to create
    a smoother surface.

    This geometry differs from the `sphere_geometry` in that it's
    mathematically closed; it consists of a single contiguous surface
    that encloses the space inside. The faces are also distributed
    evenly over the surface, all edges have the same length, and each
    triangle has the same number of incident faces. This means less
    vertices are needed to create a smoother surface. The downside is
    that one cannot easily apply a 2D texture map to this geometry.

    Parameters
    ----------
    radius : float
        The radius of the sphere. Vertices are placed at this distance around
        the local origin.
    max_edge_length : float | None
        If given and not None, it is used to calculate the `subdivisions`.
        The faces will be recursively subdivided until the length of each edge
        is no more than this value (taking the given radius into account).
    subdivisions : int | None
        The number of times to recursively subdivide the faces. The total number of faces
        will be ``60 * 4 ** subdivisions``. Default 0.

    Returns
    -------
    sphere : Geometry
        A geometry object that represents a sphere. Mathematically, the
        mesh is an orientable closed manifold.

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
    if max_edge_length is not None and subdivisions is not None:
        raise ValueError(
            "Either max_edge_length or subdivisions must be given, or none, but not both."
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
    elif subdivisions is not None:
        subdivisions = max(0, int(subdivisions))
    else:
        subdivisions = 0

    # Calculate number of faces and vertices. Mostly to validate the result.
    nfaces = 60 * 4**subdivisions
    nvertices = nfaces // 2 + 2  # weird, but true

    c0 = 0.927050983124842272306880251548  # == 3 * (5**0.5 - 1) / 4
    c1 = 1.33058699733550141141687582919  # == 9 * (9 + 5**0.5) / 76
    c2 = 2.15293498667750705708437914596  # == 9 * (7 + 5 * 5**0.5) / 76
    c3 = 2.427050983124842272306880251548  # == 3 * (1 + 5**0.5) / 4

    # Add vertices of the Pentakis Dodecahedron
    vertices = np.array(
        [
            (0.0, c0, c3),
            (0.0, c0, -c3),
            (0.0, -c0, c3),
            (0.0, -c0, -c3),
            (c3, 0.0, c0),
            (c3, 0.0, -c0),
            (-c3, 0.0, c0),
            (-c3, 0.0, -c0),
            (c0, c3, 0.0),
            (c0, -c3, 0.0),
            (-c0, c3, 0.0),
            (-c0, -c3, 0.0),
            (c1, 0.0, c2),
            (c1, 0.0, -c2),
            (-c1, 0.0, c2),
            (-c1, 0.0, -c2),
            (c2, c1, 0.0),
            (c2, -c1, 0.0),
            (-c2, c1, 0.0),
            (-c2, -c1, 0.0),
            (0.0, c2, c1),
            (0.0, c2, -c1),
            (0.0, -c2, c1),
            (0.0, -c2, -c1),
            (1.5, 1.5, 1.5),
            (1.5, 1.5, -1.5),
            (1.5, -1.5, 1.5),
            (1.5, -1.5, -1.5),
            (-1.5, 1.5, 1.5),
            (-1.5, 1.5, -1.5),
            (-1.5, -1.5, 1.5),
            (-1.5, -1.5, -1.5),
        ],
        np.float32,
    )

    # The vertices are not on the unit sphere, they seem to not even
    # be exactly on the same sphere. So we push them to the unit sphere.
    lengths = np.linalg.norm(vertices, axis=1)
    vertices[:, 0] /= lengths
    vertices[:, 1] /= lengths
    vertices[:, 2] /= lengths

    # Apply the faces of the Pentakis Dodecahedron.
    # Except that these may recurse to create sub-faces.
    faces = np.array(
        [
            (12, 0, 2),
            (12, 2, 26),
            (12, 26, 4),
            (12, 4, 24),
            (12, 24, 0),
            (13, 3, 1),
            (13, 1, 25),
            (13, 25, 5),
            (13, 5, 27),
            (13, 27, 3),
            (14, 2, 0),
            (14, 0, 28),
            (14, 28, 6),
            (14, 6, 30),
            (14, 30, 2),
            (15, 1, 3),
            (15, 3, 31),
            (15, 31, 7),
            (15, 7, 29),
            (15, 29, 1),
            (16, 4, 5),
            (16, 5, 25),
            (16, 25, 8),
            (16, 8, 24),
            (16, 24, 4),
            (17, 5, 4),
            (17, 4, 26),
            (17, 26, 9),
            (17, 9, 27),
            (17, 27, 5),
            (18, 7, 6),
            (18, 6, 28),
            (18, 28, 10),
            (18, 10, 29),
            (18, 29, 7),
            (19, 6, 7),
            (19, 7, 31),
            (19, 31, 11),
            (19, 11, 30),
            (19, 30, 6),
            (20, 8, 10),
            (20, 10, 28),
            (20, 28, 0),
            (20, 0, 24),
            (20, 24, 8),
            (21, 10, 8),
            (21, 8, 25),
            (21, 25, 1),
            (21, 1, 29),
            (21, 29, 10),
            (22, 11, 9),
            (22, 9, 26),
            (22, 26, 2),
            (22, 2, 30),
            (22, 30, 11),
            (23, 9, 11),
            (23, 11, 31),
            (23, 31, 3),
            (23, 3, 27),
            (23, 27, 9),
        ],
        np.int32,
    )

    for _ in range(subdivisions):
        # Subdivide!
        new_vertices, new_faces = subdivide_faces(vertices, faces)

        # Process new vertices
        lengths = np.linalg.norm(new_vertices, axis=1)
        new_vertices[:, 0] /= lengths
        new_vertices[:, 1] /= lengths
        new_vertices[:, 2] /= lengths
        vertices = np.row_stack([vertices, new_vertices])

        # The faces are simply replaced
        faces = new_faces

    # Double-check that the expected numbers match the real ones
    assert nfaces == len(faces)
    assert nvertices == len(vertices)

    # Return as a geometry
    return gfx.Geometry(positions=vertices * radius, indices=faces, normals=vertices)


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    geo = smooth_sphere_geometry(100, None, 1)
    print(time.perf_counter() - t0)
    m = gfx.Mesh(
        geo,
        gfx.MeshPhongMaterial(wireframe=True, wireframe_thickness=3),
    )
    # m.material.side = "FRONT"

    gfx.show(m)
