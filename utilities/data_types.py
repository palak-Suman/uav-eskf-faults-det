#DATA STRUCTURES

from dataclasses import dataclass
import numpy as np

@dataclass
class IMUMeasurement:
    timestamp: float        # Seconds
    accel: np.ndarray       # [ax, ay, az] in m/s^2
    gyro: np.ndarray        # [wx, wy, wz] in rad/s

@dataclass
class GroundTruthState:
    timestamp: float        # Seconds
    position: np.ndarray    # [x, y, z] in meters
    orientation: np.ndarray # Quaternion [w, x, y, z]
    velocity: np.ndarray    # [vx, vy, vz] in m/s

