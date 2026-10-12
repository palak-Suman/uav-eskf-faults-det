#MATHEMATICAL HELPER FUNCTIONS
import numpy as np

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Computes 3x3 skew-symmetric matrix [v_x] from a 3D vector v = [v0, v1, v2].
    Replaces vector cross products with matrix operations: (v x u) = [v_x] @ u.
    """
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]
    ])

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Computes Hamilton product of two quaternions q1 x q2.
    Format: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion q = [w, x, y, z] to a 3x3 Rotation Matrix R_Body_to_World.
    Transforms vectors from IMU Body Frame to World Navigation Frame.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

