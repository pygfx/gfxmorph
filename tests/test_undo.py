import random

import numpy as np
import pygfx as gfx
from gfxmorph import DynamicMesh, MeshUndoTracker

from testutils import run_tests


def test_undo():
    snapshots = []

    def snapshot():
        vertices, faces = m.export()
        v = undo.get_version()
        snapshots.append((v, vertices, faces))

    geo = gfx.geometries.torus_knot_geometry(stitch=False)
    vertices = geo.positions.data
    faces = geo.indices.data

    m = DynamicMesh(None, None)
    undo = MeshUndoTracker()
    m.track_changes(undo)
    snapshot()

    m.add_mesh(vertices, faces)
    snapshot()

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented

    # Stitch the mesh back up
    m.repair_touching_boundaries()
    snapshot()

    assert m.is_manifold
    assert m.is_closed
    assert m.is_oriented

    # Create some holes
    m.delete_faces([1, 123, 250, 312])
    snapshot()

    assert m.is_manifold
    assert not m.is_closed
    assert m.is_oriented

    # Close the holes
    m.repair(True)
    snapshot()

    assert m.is_manifold
    assert m.is_closed
    assert m.is_oriented

    # Also move some vertices
    ii = np.arange(10, dtype=np.int32)
    m.update_vertices(ii, m.vertices[ii] * 1.1)
    snapshot()

    assert m.is_manifold
    assert m.is_closed
    assert m.is_oriented

    # Now backtrack all the way to the empty map

    assert len(m.faces) > 0

    for v, vertices, faces in reversed(snapshots):
        undo.apply_version(m, v)
        assert np.all(m.vertices == vertices)
        assert np.all(m.faces == faces)

    assert len(m.faces) == 0

    # And redo all the steps!

    for v, vertices, faces in snapshots:
        undo.apply_version(m, v)
        assert np.all(m.vertices == vertices)
        assert np.all(m.faces == faces)

    assert len(m.faces) > 0

    # The order can be anything!

    shuffled_snapshots = snapshots.copy()
    random.shuffle(shuffled_snapshots)
    assert [x[0] for x in shuffled_snapshots] != [x[0] for x in snapshots]

    for v, vertices, faces in shuffled_snapshots:
        undo.apply_version(m, v)
        assert np.all(m.vertices == vertices)
        assert np.all(m.faces == faces)

    # We can also do a step by step by step undo
    for i in range(20):
        undo.undo(m)
    assert np.all(m.vertices == snapshots[0][1])
    assert np.all(m.faces == snapshots[0][2])

    for i in range(20):
        undo.redo(m)
    assert np.all(m.vertices == snapshots[-1][1])
    assert np.all(m.faces == snapshots[-1][2])


if __name__ == "__main__":
    run_tests(globals())
