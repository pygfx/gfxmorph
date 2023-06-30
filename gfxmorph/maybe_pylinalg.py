import numpy as np


def volume_of_triangle(p1, p2, p3):
    """Get the volume that a triangle has, the fourth point being the origin.

    Assumes CCW winding. Negate the result for CW winded triangles.
    """
    # https://stackoverflow.com/a/1568551
    p1 = np.reshape(p1, (-1, 3))
    p2 = np.reshape(p2, (-1, 3))
    p3 = np.reshape(p3, (-1, 3))

    p1x, p1y, p1z = p1[:, 0], p1[:, 1], p1[:, 2]
    p2x, p2y, p2z = p2[:, 0], p2[:, 1], p2[:, 2]
    p3x, p3y, p3z = p3[:, 0], p3[:, 1], p3[:, 2]
    v321 = p3x * p2y * p1z
    v231 = p2x * p3y * p1z
    v312 = p3x * p1y * p2z
    v132 = p1x * p3y * p2z
    v213 = p2x * p1y * p3z
    v123 = p1x * p2y * p3z
    result = (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
    return float(result) if result.shape == (1,) else result


def volume_of_closed_mesh(vertices, faces):
    """Calculate the volume of the mesh.

    It is assumed that all faces have consisted, and that the winding
    is CCW (produces a negative volume if the winding is CW). It is
    also assumed that the mesh is closed. Unclosed volumes are leaky,
    resulting in an incorrect volume calculation.
    """

    # Note: it's possible to get a hint on whether the volume is
    # actually closed, by displacing the vertices and calculating the
    # volume again. If there is a leak in the mesh, that leak will cause
    # the second volume to be different. This depends a bit on the
    # geometry, and it's hard to tell when errors are round-off errors,
    # or may be an indication of a leak. Therefore its probably better
    # to perform a proper analysis to detect the mesh being closed.

    # Batch-calculate
    volumes = volume_of_triangle(
        vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    )
    return volumes if isinstance(volumes, float) else volumes.sum()
