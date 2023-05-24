import numpy as np
import pylinalg as la

from maybe_pygfx import solid_sphere_geometry
from maybe_pylinalg import volume_of_closed_mesh


def test_solid_sphere_geometry():
    # recursive_subdivisions

    g = solid_sphere_geometry(1)
    assert g.indices.data.shape == (60, 3)
    assert g.positions.data.shape == (32, 3)

    g = solid_sphere_geometry(1, recursive_subdivisions=1)
    assert g.indices.data.shape == (60 * 4, 3)
    assert g.positions.data.shape == (60 * 4 / 2 + 2, 3)

    g = solid_sphere_geometry(1, recursive_subdivisions=2)
    assert g.indices.data.shape == (60 * 16, 3)
    assert g.positions.data.shape == (60 * 16 / 2 + 2, 3)

    # max_edge_length

    def get_edge_length(g):
        ia, ib, _ = g.indices.data[0]
        a, b = g.positions.data[[ia, ib]]
        return la.vec_dist(a, b)

    g = solid_sphere_geometry(1, max_edge_length=1)
    assert g.indices.data.shape == (60, 3)
    assert get_edge_length(g) < 1

    g = solid_sphere_geometry(0.5, max_edge_length=0.5)
    assert g.indices.data.shape == (60, 3)
    assert get_edge_length(g) < 0.5

    g = solid_sphere_geometry(1, max_edge_length=0.5)
    assert g.indices.data.shape == (60 * 4, 3)
    assert get_edge_length(g) < 0.5

    g = solid_sphere_geometry(2, max_edge_length=1)
    assert g.indices.data.shape == (60 * 4, 3)
    assert get_edge_length(g) < 1

    g = solid_sphere_geometry(2, max_edge_length=0.5)
    assert g.indices.data.shape == (60 * 16, 3)
    assert get_edge_length(g) < 0.5

    # radius, volume, and max_edge_length

    for radius in (0.4, 0.8, 1.2, 1.4):
        volume_errors = []
        for el in (0.4, 0.3, 0.2, 0.1):
            g = solid_sphere_geometry(radius, max_edge_length=el)
            assert get_edge_length(g) < el
            assert np.allclose(np.linalg.norm(g.positions.data, axis=1), radius)

            expected_v = (4 / 3) * np.pi * radius**3
            v = volume_of_closed_mesh(g.positions.data, g.indices.data)
            assert np.allclose(v, expected_v, atol=0.5 * el)
            volume_errors.append(expected_v - v)

        # Check that more subdivisions produces a better approximation os a sphere
        assert volume_errors[0] > volume_errors[-1]


if __name__ == "__main__":
    test_solid_sphere_geometry()
