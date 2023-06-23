"""
Test on adverserial cases. The word corrupt is used very generous here,
e.g. it's also used for opem manifolds.
"""

import numpy as np
import pygfx as gfx
import pytest

import maybe_pygfx
import meshmorph

from testutils import (
    iter_test_meshes,
    iter_closed_meshes,
    get_quad,
    get_sphere,
    iter_fans,
)


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


def test_a_null_mesh():
    m = MeshClass(None, None)

    # I'm not sure what would be correct, but  I guess with zero faces all conditions are valid?
    assert m.is_manifold
    assert m.is_closed
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


def test_non_manifold_collapse_face_1():
    """Collapse a face. Now the mesh is open, and one edge has 3 faces."""
    for vertices, faces, is_closed in iter_test_meshes():
        faces[1][1] = faces[1][0]

        m = MeshClass(vertices, faces)

        assert not m.is_edge_manifold
        assert not m.is_vertex_manifold  # cannot have sound fans with 3-face edges
        assert not m.is_closed  # cannot be closed if not edge manifold
        assert not m.is_oriented  #  cannot be oriented if not edge manifold

        # Collapsing this face the small meshes makes it non-manifold even after repair
        if len(faces) <= 4:
            continue

        m.repair_manifold()  # We can detect and remove collapsed faces!

        assert m.is_manifold
        assert not m.is_closed  # there's a hole in the mesh
        assert m.is_oriented


def test_non_manifold_collapse_face_2():
    """Collapse a face, but at the edge, so its still edge-manifold."""

    vertices, faces, _ = get_quad()
    faces[1][0] = faces[1][1]

    m = MeshClass(vertices, faces)

    assert m.is_edge_manifold
    assert not m.is_vertex_manifold
    assert not m.is_closed
    assert m.is_oriented  # the collapsed face maket is non-orientable

    m.repair_manifold()  # We can detect and remove collapsed faces!

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented


def test_non_manifold_add_face_1():
    """Add one face to the mesh. Some edges will now have 3 faces."""

    for vertices, faces, _ in iter_test_meshes():
        # Cannot do this on quad and tetrahedron
        if len(vertices) <= 4:
            continue

        extra_face = faces[1].copy()
        extra_face[1] = len(vertices) - 1
        assert not any(set(face) == set(extra_face) for face in faces)
        faces.append(extra_face)

        m = MeshClass(vertices, faces)

        assert not m.is_edge_manifold
        assert not m.is_manifold  # double-check
        assert not m.is_closed
        assert not m.is_oriented

        m.repair_manifold()

        assert m.is_manifold
        assert m.is_closed == (False if len(m.faces) > 0 else True)
        assert m.is_oriented


def test_non_manifold_add_face_2():
    """Add a duplicate face to the mesh. Some edges will now have 3 faces."""

    for vertices, faces, is_closed in iter_test_meshes():
        faces.append(faces[0])

        m = MeshClass(vertices, faces)

        assert not m.is_edge_manifold
        assert not m.is_vertex_manifold
        assert not m.is_closed
        assert not m.is_oriented

        m.repair_manifold()  # We can detect and remove duplicate faces!

        assert m.is_edge_manifold
        assert m.is_manifold
        assert m.is_closed == is_closed
        assert m.is_oriented


def test_non_manifold_add_face_3():
    """Add one face to the mesh, using one extra vertex. One edge will now have 3 faces."""

    for vertices, faces, _ in iter_test_meshes():
        vertices.append([1, 2, 3])
        faces.append([faces[0][0], faces[0][1], len(vertices) - 1])

        m = MeshClass(vertices, faces)

        assert not m.is_edge_manifold
        assert not m.is_vertex_manifold
        assert not m.is_closed
        assert not m.is_oriented

        m.repair_manifold()

        assert m.is_manifold
        assert m.is_closed == (False if len(m.faces) > 0 else True)
        assert m.is_oriented


def test_non_manifold_add_face_4():
    """Add one face to the mesh, using two extra vertices. There now is connectivity via a vertex."""

    # Note (AK): this case is excluded from skcg.Mesh, because its
    # implementation is very slow. I have changed my local version of
    # skcg to turn the test on.

    for vertices, faces, _ in iter_test_meshes():
        # Break it

        vertices.append([1, 2, 3])
        vertices.append([2, 3, 4])
        faces.append([faces[0][0], len(vertices) - 2, len(vertices) - 1])

        m = MeshClass(vertices, faces)

        assert m.is_edge_manifold
        assert not m.is_vertex_manifold
        assert not m.is_manifold
        assert not m.is_closed  # not closed because the added face is not
        assert m.is_oriented

        # Repair it

        m.repair_vertex_manifold()

        assert m.is_manifold
        assert not m.is_closed
        assert m.is_oriented


def test_non_manifold_add_face_5():
    """Add one face to the mesh, using one extra vertex, connected via two vertices (no edges)."""

    for vertices, faces, _ in iter_test_meshes():
        # Cannot do this on quad and tetrahedron
        if len(vertices) <= 4:
            continue

        # Break it

        vertices.append([1, 2, 3])
        extra_vi = 3 if len(vertices) == 6 else 10
        extra_face = [1, extra_vi, len(vertices) - 1]
        assert not any(len(set(face).intersection(extra_face)) > 1 for face in faces)
        faces.append(extra_face)

        m = MeshClass(vertices, faces)

        assert m.is_edge_manifold
        assert not m.is_vertex_manifold
        assert not m.is_manifold
        assert not m.is_closed
        assert m.is_oriented

        # Repair it

        m.repair_vertex_manifold()

        assert m.is_manifold
        assert not m.is_closed
        assert m.is_oriented


def test_non_manifold_weakly_connected_1():
    """Attach one part of the mesh to another part, using just one vertex."""

    # Note: This is a single connected component that *also* has a
    # one-vertex connection. A case that I initially overlooked.

    # Some hand-picked vertex pairs to connect
    vertex_pairs = [
        (12, 29),  # about opposite the sphere
        (0, 21),  # one vertex in between
    ]

    vertices, faces, _ = get_sphere()

    failed = []
    for index_to_start_from in range(len(faces)):
        for vi1, vi2 in vertex_pairs:
            vertices, faces, _ = get_sphere()

            # Break it

            faces = np.asarray(faces, np.int32)
            faces[faces == vi2] = vi1
            faces[0], faces[index_to_start_from] = tuple(
                faces[index_to_start_from]
            ), tuple(faces[0])

            m = MeshClass(vertices, faces)

            assert m.is_edge_manifold
            assert not m.is_vertex_manifold
            assert m.is_closed  # still closed!
            assert m.is_oriented

            # Repair it

            m.repair_vertex_manifold()

            assert m.is_manifold
            assert m.is_closed
            assert m.is_oriented


def test_non_manifold_weakly_connected_2():
    """Attach two meshes, using just one vertex."""

    # Stitch the test meshes together, some are open, some closed.

    for vertices1, faces1, closed1 in iter_test_meshes():
        for vertices2, faces2, closed2 in iter_test_meshes():
            faces1 = np.array(faces1)
            faces2 = np.array(faces2)
            offset2 = faces1.max() + 1
            faces = np.concatenate([faces1, faces2 + offset2])
            vertices = vertices1 + vertices2

            # Sanity check: adding individual meshes is manifold
            m = MeshClass(vertices, faces)
            assert m.is_edge_manifold
            assert m.is_vertex_manifold
            assert m.is_oriented
            assert m.is_closed == (closed1 and closed2)

            # Try this for a couple of different vertices
            for iter in range(4):
                # Break it (stitch together)

                i1 = np.random.randint(faces1.max() + 1)
                i2 = np.random.randint(faces2.max() + 1) + offset2
                stitched_faces = faces.copy()
                stitched_faces[faces == i2] = i1

                m = MeshClass(vertices, stitched_faces)
                assert m.is_edge_manifold
                assert not m.is_vertex_manifold
                assert m.is_oriented
                assert m.is_closed == (closed1 and closed2)

                # Repair it

                m.repair_vertex_manifold()

                assert m.is_manifold
                assert m.is_closed == (closed1 and closed2)
                assert m.is_oriented


def test_non_manifold_weakly_connected_3():
    """Attatch two fans in many different configurations, using one vertex."""

    # This tests a variety of fan topologies, and combinations
    # thereof, to help ensure that the vertex-manifold detection is sound.

    # Note that in this test we assume that the algorithm to detect
    # vertex-manifoldness only looks at the faces incident to the vertex
    # in question, and e.g. does not consider neighbours of these faces
    # (except the ones incident to the same vertex).

    for faces1 in iter_fans():
        for faces2 in iter_fans():
            faces1 = np.array(faces1)
            faces2 = np.array(faces2)
            offset2 = faces1.max() + 1
            faces = np.concatenate([faces1, faces2 + offset2])

            # Stub vertices
            vertices = np.zeros((faces.max() + 1, 3), np.float32)

            # Sanity check: adding individual meshes is manifold
            m = MeshClass(vertices, faces)
            assert m.is_edge_manifold
            assert m.is_vertex_manifold
            assert m.is_oriented
            assert not m.is_closed

            # Break it (stitch them together)

            faces[faces == offset2] = 0

            m = MeshClass(vertices, faces)
            assert m.is_edge_manifold
            assert not m.is_vertex_manifold
            assert m.is_oriented
            assert not m.is_closed  # none of these fans are closed

            # Repair it

            m.repair_vertex_manifold()

            assert m.is_manifold
            assert not m.is_closed
            assert m.is_oriented


# %% Test is_closed


def test_non_closed_remove_face():
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


def test_non_closed_planes_1():
    """A plane is an open mesh."""

    geo = gfx.geometries.plane_geometry()
    vertices = geo.positions.data
    faces = geo.indices.data

    m = MeshClass(vertices, faces)

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented


def test_non_closed_planes_2():
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


def test_non_closed_knot():
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


def test_non_oriented_change_winding_1():
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

            n_reversed = m.repair_oriented()

            assert n_reversed == 1
            assert m.is_manifold
            assert m.is_closed == is_closed
            assert m.is_oriented


def test_non_oriented_change_winding_2():
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

            n_reversed = m.repair_oriented()

            if len(faces) > 4:
                assert n_reversed == 2
            assert m.is_manifold
            assert m.is_closed == is_closed
            assert m.is_oriented


def test_non_oriented_inside_out():
    for vertices, faces, _ in iter_closed_meshes():
        # Turn the mesh inside-out
        faces = np.asarray(faces)
        tmp = faces[:, 0].copy()
        faces[:, 0] = faces[:, 1]
        faces[:, 1] = tmp

        m = MeshClass(vertices, faces)

        assert m.is_closed
        assert m.get_volume() < 0
        assert m.is_oriented  # still oriented ...

        n_reversed = m.repair_oriented()
        assert n_reversed == len(faces)  # ... but inside out

        assert m.is_closed
        assert m.get_volume() > 0  # now its all good!
        assert m.is_oriented


def test_non_oriented_mobius():
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

    m.repair_oriented()

    assert m.is_manifold
    assert not m.is_closed
    assert not m.is_oriented  # cannot repair this topology


def test_non_oriented_klein_bottle():
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

    m.repair_oriented()

    assert m.is_manifold
    assert m.is_closed
    assert not m.is_oriented  # cannot repair this topology


# %% Allow running as a script


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            func = globals().get(name)
            print(f"Running {name}")
            func()
    print("Done")
