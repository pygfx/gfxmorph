import numpy as np
from testutils import run_tests, get_sphere
from gfxmorph import DynamicMesh
import pytest


def test_dynamicmesh_init():

    vertices, faces,_ = get_sphere()

    m = DynamicMesh()

    # Mesh is empty
    assert isinstance(m.faces, np.ndarray)
    assert m.faces.dtype == np.int32 and m.faces.shape == (0, 3)
    assert isinstance(m.vertices, np.ndarray)
    assert m.vertices.dtype == np.float32 and m.vertices.shape == (0, 3)

    m.add_vertices(vertices)
    m.add_faces(faces)

    # Mesh is not empty
    assert isinstance(m.faces, np.ndarray)
    assert m.faces.dtype == np.int32 and m.faces.shape == (len(faces), 3)
    assert isinstance(m.vertices, np.ndarray)
    assert m.vertices.dtype == np.float32 and m.vertices.shape == (len(vertices), 3)

    # Cannot load faces before vertices
    m = DynamicMesh()
    with pytest.raises(ValueError):
        m.add_faces(faces)


if __name__ == "__main__":
    run_tests(globals())
