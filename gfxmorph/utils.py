import logging
import traceback

import numpy as np


logger = logging.getLogger("meshmorph")


class Safecall:
    """Context manager for doing calls that should not raise. If an exception
    is raised, it is caught, logged, and the context exits normally.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            lines = traceback.format_exception(exc_type, exc_val, exc_tb)
            logger.error("Exception in update callback:\n" + "".join(lines))
        return True  # Yes, we handled the exception


def check_indices(indices, n, what_for):
    """Check indices and convert to an in32 array (if necessary)."""
    result = None
    typ = type(indices).__name__
    if isinstance(indices, int):
        result = [indices]
    elif isinstance(indices, list):
        result = indices
    elif isinstance(indices, np.ndarray):
        typ = "ndarray:" + "x".join(str(x) for x in indices.shape)
        typ += "x" + indices.dtype.name
        if indices.size == 0:
            result = []
        elif indices.ndim == 1 and indices.dtype.kind == "i":
            result = indices

    if result is None:
        raise TypeError(
            f"The {what_for} must be given as int, list, or 1D int array, not {typ}."
        )
    result = np.asarray(result, np.int32)
    if len(result) == 0:
        raise ValueError(f"The {what_for} must not be empty.")
    elif result.min() < 0:
        raise ValueError("Negative indices not allowed.")
    elif result.max() >= n:
        raise ValueError("Index out of bounds.")

    return result


def as_immutable_array(array):
    """Return a read-only view of the given array."""
    v = array.view()
    v.setflags(write=False)
    return v


def make_vertex2faces(faces, nverts=None):
    """Create a simple map to map vertex indices to a list of face indices."""
    faces = np.asarray(faces, np.int32)
    if nverts is None:
        nverts = faces.max() + 1

    vertex2faces = [[] for _ in range(nverts)]
    for fi in range(len(faces)):
        face = faces[fi]
        vertex2faces[face[0]].append(fi)
        vertex2faces[face[1]].append(fi)
        vertex2faces[face[2]].append(fi)
    return vertex2faces
