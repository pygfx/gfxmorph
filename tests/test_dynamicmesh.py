import numpy as np
import pytest

from gfxmorph import maybe_pygfx
from gfxmorph import DynamicMesh
from testutils import run_tests


# todo: there are probably more relevant tests in arbiter


def test_mesh_basics():
    geo = maybe_pygfx.smooth_sphere_geometry(1, subdivisions=3)
    vertices = geo.positions.data
    faces = geo.indices.data

    # Create a silly mesh consisting of a single triangle
    triangle = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    m = DynamicMesh(triangle, [[0, 1, 2]])
    assert not m.is_closed

    # Cannot creata a null mesh
    with pytest.raises(ValueError):
        DynamicMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    with pytest.raises(ValueError):
        DynamicMesh(triangle, np.zeros((0, 3), np.int32))
    with pytest.raises(ValueError):
        DynamicMesh(np.zeros((0, 3), np.float32), [[0, 1, 2]])

    # Creat mesh and check its volume
    m = DynamicMesh(vertices, faces)
    expected_volume = (4 / 3) * np.pi
    assert 0.9 * expected_volume < m.get_volume() < expected_volume
    assert m.is_closed
    assert m.component_count == 1

    # Create a mesh with two objects
    vertices2 = np.row_stack([vertices + 10, vertices + 20])
    faces2 = np.row_stack([faces, faces + len(vertices)])

    # Check its volume
    m = DynamicMesh(vertices2, faces2)
    expected_volume = 2 * (4 / 3) * np.pi
    assert 0.9 * expected_volume < m.get_volume() < expected_volume
    assert m.is_closed
    assert m.component_count == 2

    # Now the meshes inside out
    faces2[:, 1], faces2[:, 2] = faces2[:, 2].copy(), faces2[:, 1].copy()

    # The repair mechanism will flip exactly what's needed to fix up the mesh!
    m = DynamicMesh(vertices2, faces2)
    expected_volume = 2 * (4 / 3) * np.pi
    assert m.get_volume() < 0
    m.repair()
    assert 0.9 * expected_volume < m.get_volume() < expected_volume
    assert m.is_closed
    assert m.component_count == 2


def test_mesh_selection():
    # Create a mesh
    geo = maybe_pygfx.smooth_sphere_geometry(1, subdivisions=1)
    vertices = geo.positions.data
    faces = geo.indices.data
    m = DynamicMesh(vertices, faces)

    # Select a vertex, twice
    i1, d1 = m.get_closest_vertex((-2, 0, 0))
    i2, d2 = m.get_closest_vertex((+2, 0, 0))

    assert i1 != i2
    assert d1 == 1.0
    assert d2 == 1.0

    # Select over surface
    selected1 = m.select_vertices_over_surface(i1, 0.5)
    selected2 = m.select_vertices_over_surface(i2, 0.5)

    # Since this mesh is very regular, the selected number must be the same
    assert len(selected1) == 7
    assert len(selected2) == 7

    # Select over surface, with very high distance, so that the whole mesh is selected
    selected1 = m.select_vertices_over_surface(i1, 4)
    selected2 = m.select_vertices_over_surface(i2, 4)

    assert selected1 == selected2
    assert len(selected1) == len(vertices)


if __name__ == "__main__":
    run_tests(globals())
