import numpy as np

from pylinalg import vec_dist
from maybe_pylinalg import volume_of_triangle, volume_of_triangle, volume_of_closed_mesh
from maybe_pygfx import solid_tetrahedron


def test_volume_of_triangle():
    triangles = [
        [(0, 0, 1), (0, 1, 1), (1, 1, 1)],
        [(0, 0, 2), (0, 1, 2), (1, 1, 2)],
        [(0, 0, 1), (0, -1, 1), (1, -1, 1)],
        [(0, 0, 1), (0, -1, 1), (-1, 1, 1)],
    ]
    expected = [-1 / 6, -1 / 3, 1 / 6, -1 / 6]

    # Test individual calls
    for triangle, expect in zip(triangles, expected):
        v = volume_of_triangle(triangle[0], triangle[1], triangle[2])
        assert np.allclose(v, expect)

    # Test batch call
    triangles_array = np.array(triangles, np.float32)
    v1 = triangles_array[:, 0, :]
    v2 = triangles_array[:, 1, :]
    v3 = triangles_array[:, 2, :]
    volumes = volume_of_triangle(v1, v2, v3)
    assert np.allclose(volumes, expected)


def test_volume_of_closed_mesh():
    # Create a regular tetrahedron
    vertices, faces = solid_tetrahedron()

    edge = 1.632993117309891
    expected_volume = edge**3 / (6 * 2**0.5)

    # Make sure the tetrahedron is not inside-out
    for face in faces:
        a, b, c = vertices[face]
        assert volume_of_triangle(a, b, c) > 0

    # Measure the volume of this tetrahedron
    v = volume_of_closed_mesh(vertices, faces)
    assert np.allclose(v, expected_volume)

    # Make it twice as large (in three dimensions, so it's 4x the volume)
    vertices *= 2
    v = volume_of_closed_mesh(vertices, faces)
    assert np.allclose(v, expected_volume * 8)

    # Move it
    vertices += 10
    v = volume_of_closed_mesh(vertices, faces)
    assert np.allclose(v, expected_volume * 8)

    # Duplicate it, using faces
    faces2 = np.row_stack([faces, faces])
    v = volume_of_closed_mesh(vertices, faces2)
    assert np.allclose(v, expected_volume * 16)

    # Really duplicate it
    vertices2 = np.row_stack([vertices, vertices + 10])
    faces2 = np.row_stack([faces, faces + 4])
    v = volume_of_closed_mesh(vertices2, faces2)
    assert np.allclose(v, expected_volume * (8 + 8), atol=0.001)

    # Reduce size of the first one
    vertices2[:4] -= 10
    vertices2[:4] *= 0.5
    v = volume_of_closed_mesh(vertices2, faces2)
    assert np.allclose(v, expected_volume * (8 + 1), atol=0.001)

    # Reduce size of the second one
    vertices2[4:] -= 20
    vertices2[4:] *= 0.5
    v = volume_of_closed_mesh(vertices2, faces2)
    assert np.allclose(v, expected_volume * (1 + 1), atol=0.001)

    # Move one inside out
    faces2[:4, 1], faces2[:4, 2] = faces2[:4, 2].copy(), faces2[:4, 1].copy()
    v = volume_of_closed_mesh(vertices2, faces2)
    assert np.allclose(v, expected_volume * 0, atol=0.001)

    # Move the other inside out too, now the volume is negative
    faces2[4:, 1], faces2[4:, 2] = faces2[4:, 2].copy(), faces2[4:, 1].copy()
    v = volume_of_closed_mesh(vertices2, faces2)
    assert np.allclose(v, -expected_volume * 2, atol=0.001)


if __name__ == "__main__":
    test_volume_of_triangle()
    test_volume_of_closed_mesh()
