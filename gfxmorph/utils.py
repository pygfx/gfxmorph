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


def check_indices(indices, n, what_for, *, allow_empty=False):
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
    if allow_empty and len(result) == 0:
        return result
    elif len(result) == 0:
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


class ImmutableMapOfSequences:
    """A thin readonly wrapper for for a map containing sequences, (e.g. a list of lists) ."""

    __slots__ = ["_map", "__len__", "__getitem__"]

    def __init__(self, map):
        self._map = map
        self.__len__ = map.__len__
        self.__getitem__ = lambda i: tuple(map[i])

    def __setitem__(self, index, value):
        raise ValueError("Map is readonly.")


def make_vertex2faces(faces, nverts=None):
    """Create a simple map to map vertex indices to a list of face indices."""
    # Prepare
    faces = np.asarray(faces, np.int32)
    if nverts is None:
        nverts = faces.max() + 1
    # Fill
    vertex2faces = [[] for _ in range(nverts)]
    for fi in range(len(faces)):
        face = faces[fi]
        vertex2faces[face[0]].append(fi)
        vertex2faces[face[1]].append(fi)
        vertex2faces[face[2]].append(fi)
    # Return as read-only
    vertex2faces = [tuple(fii) for fii in vertex2faces]
    return ImmutableMapOfSequences(vertex2faces)
