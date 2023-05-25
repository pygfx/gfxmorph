import numpy as np
import pygfx as gfx

import maybe_pygfx
from meshmorph import AbstractMesh


def test_abstract_mesh_basics():
    geo = maybe_pygfx.solid_sphere_geometry(1, recursive_subdivisions=3)
    vertices = geo.positions.data
    faces = geo.indices.data

    # Creat mesh and check its volume
    m = AbstractMesh(vertices, faces)
    expected_volume = (4 / 3) * np.pi
    assert 0.9 * expected_volume < m.get_volume() < expected_volume
    assert m._n_reversed_faces == 0
    assert m.is_closed
    assert m.component_count == 1

    # Create a mesh with two objects
    vertices2 = np.row_stack([vertices + 10, vertices + 20])
    faces2 = np.row_stack([faces, faces + len(vertices)])

    # Check its volume
    m = AbstractMesh(vertices2, faces2)
    expected_volume = 2 * (4 / 3) * np.pi
    assert 0.9 * expected_volume < m.get_volume() < expected_volume
    assert m._n_reversed_faces == 0
    assert m.is_closed
    assert m.component_count == 2

    # Now flip one inside out
    n = len(faces)
    faces2[:n, 1], faces2[:n, 2] = faces2[:n, 2].copy(), faces2[:n, 1].copy()

    # The auto-correction mechanism will flip exactly what's needed to fix up the mesh!
    m = AbstractMesh(vertices2, faces2)
    expected_volume = 2 * (4 / 3) * np.pi
    assert 0.9 * expected_volume < m.get_volume() < expected_volume
    assert m._n_reversed_faces == 3840  # 60*4**3
    assert m.is_closed
    assert m.component_count == 2

    # todo: also test that the right faces are set for upload

    # todo: also test individually reversed faces

    # todo: also test holes


# todo: there are probably more tests in arbiter


if __name__ == "__main__":
    test_abstract_mesh_basics()
