"""
We cover most mesh funcs via test_corrupt.py and test_abstractmesh.py.
Here we have a few more tests to test some functions more directly.
"""

import numpy as np
from gfxmorph import meshfuncs
from testutils import run_tests, iter_fans, get_sphere


def test_mesh_boundaries():
    # Test boundaries on a bunch of fans
    for faces in iter_fans():
        is_closed = len(faces) >= 3 and faces[-1][1] == 1
        boundaries = meshfuncs.mesh_get_boundaries(faces)
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
    _, faces_original, _ = get_sphere()
    vertex2faces = meshfuncs.make_vertex2faces(faces_original)

    for n_faces_to_remove_per_hole in [1, 2, 3, 4]:
        faces = [face for face in faces_original]
        faces2pop = []
        for fi in (2, 30):
            faces2pop.append(fi)
            _, fii = meshfuncs.face_get_neighbours2(faces, vertex2faces, fi)
            for _ in range(n_faces_to_remove_per_hole - 1):
                faces2pop.append(fii.pop())

        assert len(faces2pop) == len(set(faces2pop))  # no overlap between holes

        for fi in reversed(sorted(faces2pop)):
            faces.pop(fi)

        boundaries = meshfuncs.mesh_get_boundaries(faces)

        nr_vertices_on_boundary = [0, 3, 4, 5, 6, 7][n_faces_to_remove_per_hole]
        assert len(boundaries) == 2
        assert len(boundaries[0]) == nr_vertices_on_boundary
        assert len(boundaries[1]) == nr_vertices_on_boundary


if __name__ == "__main__":
    run_tests(globals())
