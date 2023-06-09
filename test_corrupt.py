"""
Test on adverserial cases. The word corrupt is used very generous here,
e.g. it's also used for opem manifolds.
"""

import numpy as np
import pygfx as gfx
import pytest

import maybe_pygfx
import meshmorph

from testutils import iter_test_meshes, get_quad


# Set to 1 to perform these tests on skcg, as a validation check
USE_SKCG = 0

if USE_SKCG:
    import skcg

    MeshClass = skcg.Mesh
else:
    MeshClass = meshmorph.AbstractMesh


# todo: include is_edge_maniforld

# %% Some basic tests first


def test_raw_meshes():
    """Test that the raw meshes, unchanged, are what we expect them to be."""
    count = 0
    for vertices, faces, is_closed in iter_test_meshes():
        count += 1
        m = MeshClass(vertices, faces)
        assert m.is_manifold
        assert m.is_closed == is_closed
        assert m.is_oriented

    assert count == 6


def test_a_single_triangle():
    """A single triangle is an open oriented manifold."""

    vertices = [
        [-1, -1, 0],
        [+1, -1, 0],
        [+1, +1, 0],
    ]
    faces = [
        [0, 2, 1],
    ]

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented


def test_invalid_faces():
    """Test to check that faces must have valid indices for the vertices"""
    # Copied from skcg
    with pytest.raises(ValueError):
        MeshClass(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[0, 1, 9999]]
        )

    with pytest.raises(ValueError):
        MeshClass([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[0, 1, -1]])


# %% Test is_manifold


def test_corrupt_mesh11():
    """Add one face to the mesh. Some edges will now have 3 faces."""

    # Note: the extra_face is actually in the tetrahedron, but in a different order

    for vertices, faces, _ in iter_test_meshes():
        # Cannot do this on quad and tetrahedron
        if len(vertices) <= 4:
            continue

        extra_face = faces[1].copy()
        extra_face[1] = len(vertices) - 1
        assert not any(set(face) == set(extra_face) for face in faces)
        faces.append(extra_face)

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh12():
    """Add a duplicate face to the mesh. Some edges will now have 3 faces."""

    for vertices, faces, _ in iter_test_meshes():
        faces.append(faces[0])

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh13():
    """Add one face to the mesh, using one extra vertex. One edge will now have 3 faces."""

    for vertices, faces, _ in iter_test_meshes():
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

    for vertices, faces, _ in iter_test_meshes():
        vertices.append([1, 2, 3])
        vertices.append([2, 3, 4])
        faces.append([faces[0][0], len(vertices) - 2, len(vertices) - 1])

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        # A mesh that is edge-manifold but not full manifold can still be oriented!
        # todo: I guess it's also closed?
        assert m.is_oriented


def test_corrupt_mesh15():
    """Add one face to the mesh, using one extra vertex, connected via two vertices (no edges)."""

    # Note: similar case as the previous one.

    for vertices, faces, _ in iter_test_meshes():
        # Cannot do this on quad and tetrahedron
        if len(vertices) <= 4:
            continue

        vertices.append([1, 2, 3])
        extra_vi = 3 if len(vertices) == 6 else 10
        extra_face = [1, extra_vi, len(vertices) - 1]
        assert not any(len(set(face).intersection(extra_face)) > 1 for face in faces)
        faces.append(extra_face)

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert m.is_oriented


def test_corrupt_mesh16():
    """Collapse a face. Now the mesh is open, and one edge has 3 faces."""

    for vertices, faces, _ in iter_test_meshes():
        faces[1][1] = faces[1][0]

        m = MeshClass(vertices, faces)

        assert not m.is_manifold
        assert not m.is_closed
        assert not m.is_oriented


def test_corrupt_mesh17():
    """Collapse a face, but at the edge, so its still edge-manifold."""

    vertices, faces, _ = get_quad()
    faces[1][0] = faces[1][1]

    m = MeshClass(vertices, faces)

    assert not m.is_manifold
    assert not m.is_closed
    assert m.is_oriented


# %% Test is_closed


def test_corrupt_mesh21():
    """Remove a face, opening up the mesh."""

    for vertices, faces, _ in iter_test_meshes():
        # Does not work for quad and strip
        if len(faces) <= 3:
            continue

        faces.pop(1)

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

    for face_idx in [0, 1, 2, 11, 29, -3, -2, -1]:
        for vertices, faces, is_closed in iter_test_meshes():
            # For some face indices, the mesh has too little faces
            face_idx = face_idx if face_idx >= 0 else len(faces) + face_idx
            if face_idx >= len(faces):
                continue

            face = faces[face_idx]
            faces[face_idx] = face[0], face[2], face[1]

            m = MeshClass(vertices, faces)

            assert m.is_manifold
            assert m.is_closed == is_closed
            assert not m.is_oriented


def test_corrupt_mesh32():
    """Change the winding of two adjacent faces."""

    for face_idx1 in [0, 1, 2, 11, 29, -3, -2, -1]:
        for vertices, faces, is_closed in iter_test_meshes():
            ori_faces = [face.copy() for face in faces]

            # If we reverse all faces, it will still be oriented
            if len(faces) <= 2:
                continue

            # For some face indices, the mesh has too little faces
            face_idx1 = face_idx1 if face_idx1 >= 0 else len(faces) + face_idx1
            if face_idx1 >= len(faces):
                continue

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
            assert m.is_closed == is_closed
            assert not m.is_oriented


def test_corrupt_mesh33():
    """A Möbius strip!"""

    # This one is just a surface, really
    geo = gfx.geometries.mobius_strip_geometry(stitch=False)
    vertices = geo.positions.data
    faces = geo.indices.data

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented

    # This is the real deal
    geo = gfx.geometries.mobius_strip_geometry(stitch=True)
    vertices = geo.positions.data
    faces = geo.indices.data

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert not m.is_closed
    assert not m.is_oriented


def test_corrupt_mesh34():
    """A Klein bottle!"""

    # This one is just a surface, really
    geo = gfx.geometries.klein_bottle_geometry(stitch=False)
    vertices = geo.positions.data
    faces = geo.indices.data

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented

    # This is the real deal
    geo = gfx.geometries.klein_bottle_geometry(stitch=True)
    vertices = geo.positions.data
    faces = geo.indices.data

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert m.is_closed
    assert not m.is_oriented


# %% Allow running a script


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            func = globals().get(name)
            print(f"Running {name}")
            func()
    print("Done")
