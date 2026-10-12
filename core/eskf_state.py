import numpy as np

class ESKFState:
    """
    15-DOF Error-State Kalman Filter State Container.
    Manages both 16D Nominal State and 15x15 Error Covariance Matrix (P).
    """
    def __init__(self):
        # --- 1. NOMINAL STATE INITIALIZATION (16-Dimensional) ---
        # Position in World Frame [x, y, z] in meters
        self.p = np.zeros(3)                 
        
        # Velocity in World Frame [vx, vy, vz] in m/s
        self.v = np.zeros(3)                 
        
        # Orientation Quaternion [w, x, y, z] mapping Body to World Frame
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) 
        
        # Accelerometer Bias [b_ax, b_ay, b_az] in m/s^2
        self.ba = np.zeros(3)                
        
        # Gyroscope Bias [b_gx, b_gy, b_gz] in rad/s
        self.bg = np.zeros(3)                

        # --- 2. ERROR STATE VECTOR INITIALIZATION (15-Dimensional) ---
        # delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]
        self.delta_x = np.zeros(15)

        # --- 3. ERROR COVARIANCE MATRIX P INITIALIZATION (15x15) ---
        self.P = np.zeros((15, 15))
        self.init_covariance()

    def init_covariance(self):
        """Initializes diagonal variance blocks of the 15x15 Covariance Matrix P."""
        sigma_p = 0.01      # Standard deviation for position: 1 cm (0.01 m)
        sigma_v = 0.01      # Standard deviation for velocity: 1 cm/s (0.01 m/s)
        sigma_theta = 0.01  # Standard deviation for orientation: ~0.57 degrees (0.01 rad)
        sigma_ba = 0.05     # Standard deviation for accel bias: 0.05 m/s^2
        sigma_bg = 0.005    # Standard deviation for gyro bias: 0.005 rad/s

        # Set 3x3 diagonal blocks using variance (sigma squared)
        self.P[0:3, 0:3]     = (sigma_p ** 2) * np.eye(3)     # Position variance
        self.P[3:6, 3:6]     = (sigma_v ** 2) * np.eye(3)     # Velocity variance
        self.P[6:9, 6:9]     = (sigma_theta ** 2) * np.eye(3) # Attitude variance
        self.P[9:12, 9:12]   = (sigma_ba ** 2) * np.eye(3)    # Accel bias variance
        self.P[12:15, 12:15] = (sigma_bg ** 2) * np.eye(3)    # Gyro bias variance
