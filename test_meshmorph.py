import numpy as np
import pygfx as gfx
import pytest

import maybe_pygfx
import meshmorph
from meshmorph import AbstractMesh
import testutils


def test_abstract_mesh_basics():
    geo = maybe_pygfx.smooth_sphere_geometry(1, subdivisions=3)
    vertices = geo.positions.data
    faces = geo.indices.data

    # Create a silly mesh consisting of a single triangle
    triangle = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    m = AbstractMesh(triangle, [[0, 1, 2]])
    assert not m.is_closed

    # Cannot creata a null mesh
    with pytest.raises(ValueError):
        AbstractMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    with pytest.raises(ValueError):
        AbstractMesh(triangle, np.zeros((0, 3), np.int32))
    with pytest.raises(ValueError):
        AbstractMesh(np.zeros((0, 3), np.float32), [[0, 1, 2]])

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


def test_mesh_selection():
    # Create a mesh
    geo = maybe_pygfx.smooth_sphere_geometry(1, subdivisions=1)
    vertices = geo.positions.data
    faces = geo.indices.data
    m = AbstractMesh(vertices, faces)

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


def test_boundaries():
    # Test boundaries on a bunch of fans
    for faces in testutils.iter_fans():
        is_closed = len(faces) >= 3 and faces[-1][1] == 1
        boundaries = meshmorph.mesh_get_boundaries(faces)
        assert len(boundaries) == 1
        boundary = boundaries[0]
        nr_vertices_on_boundary = 3 * len(faces) - 2 * (len(faces) - 1)
        if is_closed:
            nr_vertices_on_boundary -= 2
        assert len(boundary) == nr_vertices_on_boundary

    # Next we'll work with a sphere. We create two holes in the sphere,
    # with increasing size. The primary faces to remove should be far
    # away from each-other, so that when we create larger holes, the
    # hole's won't touch, otherwise the mesh becomes non-manifold.
    _, faces_original, _ = testutils.get_sphere()
    vertex2faces = meshmorph.make_vertex2faces(faces_original)

    for n_faces_to_remove_per_hole in [1, 2, 3, 4]:
        faces = [face for face in faces_original]
        faces2pop = []
        for fi in (2, 30):
            faces2pop.append(fi)
            _, fii = meshmorph.face_get_neighbours2(faces, vertex2faces, fi)
            for _ in range(n_faces_to_remove_per_hole - 1):
                faces2pop.append(fii.pop())

        assert len(faces2pop) == len(set(faces2pop))  # no overlap between holes

        for fi in reversed(sorted(faces2pop)):
            faces.pop(fi)

        boundaries = meshmorph.mesh_get_boundaries(faces)

        nr_vertices_on_boundary = [0, 3, 4, 5, 6, 7][n_faces_to_remove_per_hole]
        assert len(boundaries) == 2
        assert len(boundaries[0]) == nr_vertices_on_boundary
        assert len(boundaries[1]) == nr_vertices_on_boundary


# todo: there are probably more tests in arbiter

# %%

if __name__ == "__main__":
    test_boundaries()
