import numpy as np
import pygfx as gfx
import pytest

import maybe_pygfx
import meshmorph


# Set to 1 to perform these tests on skcg, as a validation check
USE_SKCG = 1

if USE_SKCG:
    import skcg

    MeshClass = skcg.Mesh
else:
    MeshClass = meshmorph.MeshClass


# %% Utils


def get_tetrahedron():
    """tetrahedron as simple list objects, so we can easily add stuff to create corrupt meshes."""
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
    return vertices, faces


def get_sphere():
    geo = maybe_pygfx.solid_sphere_geometry()
    return geo.positions.data.tolist(), geo.indices.data.tolist()


def test_raw_meshes():
    # Check tetrahedron
    vertices, faces = get_tetrahedron()
    m = MeshClass(vertices, faces)
    assert m.is_manifold
    assert m.is_closed
    assert m.is_oriented

    # Check sphere
    vertices, faces = get_sphere()
    m = MeshClass(vertices, faces)
    assert m.is_manifold
    assert m.is_closed
    assert m.is_oriented


# %% Test is_manifold


def test_corrupt_mesh11():
    """Add one face to the mesh. Some edges will now have 3 faces."""

    # Note: the extra_face is actually in the tetrahedron, but in a different order

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        vertices, faces = mesh_func()

        extra_face = [1, 2, 3]
        assert faces[0] in faces
        assert extra_face not in faces
        faces.append(extra_face)

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh12():
    """Add a duplicate face to the mesh. Some edges will now have 3 faces."""

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        vertices, faces = mesh_func()

        faces.append(faces[0])

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh13():
    """Add one face to the mesh, using one extra vertex. One edge will now have 3 faces."""

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        vertices, faces = mesh_func()

        vertices.append([1, 2, 3])
        faces.append([faces[0][0], faces[0][1], len(vertices) - 1])

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh14():
    """Add one face to the mesh, using two extra vertices. There now is connectivity via a vertex."""

    # Note (AK): this case is excluded from skcg.Mesh, because its
    # implementation is very slow. I have changed my local version of
    # skcg to turn the test on.

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        vertices, faces = mesh_func()

        vertices.append([1, 2, 3])
        vertices.append([2, 3, 4])
        faces.append([faces[0][0], len(vertices) - 2, len(vertices) - 1])

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh15():
    """Collapse a face. Now the mesh is open, and one edge has 3 faces."""

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        vertices, faces = mesh_func()

        faces[2][0] = faces[2][1]

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


# %% Test is_closed


def test_corrupt_mesh21():
    """Remove a face, opening up the mesh."""

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        vertices, faces = mesh_func()

        faces.pop(2)

        m = MeshClass(vertices, faces)

        assert m.is_manifold
        assert not m.is_closed
        assert m.is_oriented


def test_corrupt_mesh22():
    """A plane is an open mesh."""

    geo = gfx.geometries.plane_geometry()
    vertices = geo.positions.data
    faces = geo.indices.data

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented


def test_corrupt_mesh23():
    """A Pygfx isohedron consists of a series of planes."""

    # Note: multiple components in these meshes.
    # Note: this secretly is also a test for pygfx.

    for isohedron in "tetrahedron", "octahedron", "icosahedron", "dodecahedron":
        geo = getattr(gfx.geometries, f"{isohedron}_geometry")()
        vertices = geo.positions.data
        faces = geo.indices.data

        m = MeshClass(vertices, faces)

        assert m.is_manifold
        assert not m.is_closed
        assert m.is_oriented


def test_corrupt_mesh24():
    """Pygfx can produce open and closed torus knots."""

    for stitch in (False, True):
        geo = gfx.geometries.torus_knot_geometry(stitch=stitch)
        vertices = geo.positions.data
        faces = geo.indices.data

        m = MeshClass(vertices, faces)

        assert m.is_manifold
        assert m.is_closed == stitch
        assert m.is_oriented


# %% Test is_oriented


def test_corrupt_mesh31():
    """Change the winding of one face."""

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        for face_idx in [0, 1, 2, -3, -2, -1]:
            vertices, faces = mesh_func()

            face = faces[face_idx]
            faces[face_idx] = face[0], face[2], face[1]

            m = MeshClass(vertices, faces)

            assert m.is_manifold
            assert m.is_closed
            assert not m.is_oriented


def test_corrupt_mesh32():
    """Change the winding of two adjacent faces."""

    mesh_funcs = [get_tetrahedron, get_sphere]

    for mesh_func in mesh_funcs:
        for face_idx1 in [0, 1, 2, -3, -2, -1]:
            vertices, faces = mesh_func()

            face1 = faces[face_idx1]
            faces[face_idx1] = face1[0], face1[2], face1[1]
            for face_idx2 in range(len(faces)):
                if face_idx2 == face_idx1:
                    continue
                face2 = faces[face_idx2]
                if face2[0] in face1:
                    break
            else:
                assert False, "WTF"
            face2 = faces[face_idx2]
            faces[face_idx2] = face2[0], face2[2], face2[1]

            m = MeshClass(vertices, faces)

            assert m.is_manifold
            assert m.is_closed
            assert not m.is_oriented


# todo: mobius
# todo: klein bottle
# todo: tests from arbiter?
# todo: tests from skcg?


# %% Allow running a script


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            func = globals().get(name)
            print(f"Running {name}")
            func()
    print("Done")
