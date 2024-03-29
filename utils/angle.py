import numpy as np
from numba import njit


def rotation_from_two_vectors(a, b):
    """Find the rotation matrix that aligns vector a with vector b"""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.multiply(np.dot(vx, vx), (1 - c) / (s**2))
    return r


def rotation_matrix(A, B):
    """Find the rotation matrix that aligns vector A with vector B"""

    ax = A[0]
    ay = A[1]
    az = A[2]

    bx = B[0]
    by = B[1]
    bz = B[2]

    au = A / (np.sqrt(ax * ax + ay * ay + az * az))
    bu = B / (np.sqrt(bx * bx + by * by + bz * bz))

    R = np.array(
        [
            [bu[0] * au[0], bu[0] * au[1], bu[0] * au[2]],
            [bu[1] * au[0], bu[1] * au[1], bu[1] * au[2]],
            [bu[2] * au[0], bu[2] * au[1], bu[2] * au[2]],
        ]
    )

    return R


@njit(fastmath=True, cache=True, parallel=True)
def rod_rot(a, b):  # Rodrigues's rotation formula
    """Find the rotation matrix that points vector a towards vector b"""
    v = np.cross(a, b)
    c = np.dot(a, b)
    h = (1 - c) / (1 - c**2)

    vx, vy, vz = v
    rot = [
        [c + h * vx**2, h * vx * vy - vz, h * vx * vz + vy],
        [h * vx * vy + vz, c + h * vy**2, h * vy * vz - vx],
        [h * vx * vz - vy, h * vy * vz + vx, c + h * vz**2],
    ]
    return rot
