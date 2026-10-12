import numpy as np
from core.eskf_state import ESKFState
from utilities.data_types import IMUMeasurement
from utilities.helper_math_fn import (
    skew_symmetric, 
    quaternion_multiply, 
    quaternion_to_rotation_matrix
)

def propagate_covariance(state: ESKFState, F_x: np.ndarray, dt: float,
                         accel_noise_std: float = 0.1, 
                         gyro_noise_std: float = 0.01,
                         accel_bias_std: float = 0.001, 
                         gyro_bias_std: float = 0.0001):
    """
    Propagates the 15x15 Error Covariance Matrix P forward in time.
    
    P_k = F_x * P_{k-1} * F_x^T + Q
    """
    # 1. Discrete process noise variances
    var_a = (accel_noise_std * dt) ** 2
    var_g = (gyro_noise_std * dt) ** 2
    var_ba = (accel_bias_std * dt) ** 2
    var_bg = (gyro_bias_std * dt) ** 2

    # Diagonal array for the 12 process noise sources
    Q_diag = np.concatenate([
        np.full(3, var_a),   # Accelerometer noise
        np.full(3, var_g),   # Gyroscope noise
        np.full(3, var_ba),  # Accel bias random walk
        np.full(3, var_bg)   # Gyro bias random walk
    ])

    # 2. Noise Mapping Matrix G (15x12) mapping noise into error-state delta_x
    G = np.zeros((15, 12))
    G[3:6, 0:3] = np.eye(3)    # Noise -> Velocity error
    G[6:9, 3:6] = np.eye(3)    # Noise -> Orientation error
    G[9:12, 6:9] = np.eye(3)   # Noise -> Accel bias error
    G[12:15, 9:12] = np.eye(3) # Noise -> Gyro bias error

    # Process noise covariance Q (15x15)
    Q = G @ np.diag(Q_diag) @ G.T

    # 3. Propagate error covariance matrix P
    state.P = F_x @ state.P @ F_x.T + Q

    # 4. Enforce numerical symmetry P = 0.5 * (P + P^T)
    state.P = 0.5 * (state.P + state.P.T)

def compute_error_jacobian(state: ESKFState, accel_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> np.ndarray:
    """
    Computes the 15x15 discrete error transition matrix F_x for ESKF covariance propagation.
    Linearizes system dynamics around current nominal state and IMU inputs over time interval dt.
    """
    # Initialize 15x15 identity matrix (base transition state)
    F_x = np.eye(15)
    
    # Compute current Body-to-World Rotation Matrix from nominal orientation quaternion
    R_bw = quaternion_to_rotation_matrix(state.q)
    
    # Correct raw IMU measurements by subtracting estimated sensor biases
    accel_corrected = accel_meas - state.ba
    gyro_corrected = gyro_meas - state.bg
    
    # Block 1: Position error update from velocity error (delta_p = delta_p + delta_v * dt)
    F_x[0:3, 3:6] = np.eye(3) * dt
    
    # Block 2a: Velocity error update from orientation error (delta_v += -R * [a_x] * delta_theta * dt)
    F_x[3:6, 6:9] = -R_bw @ skew_symmetric(accel_corrected) * dt
    
    # Block 2b: Velocity error update from accel bias error (delta_v += -R * delta_ba * dt)
    F_x[3:6, 9:12] = -R_bw * dt
    
    # Block 3a: Orientation error self-propagation under angular rate (delta_theta += -[w_x] * delta_theta * dt)
    F_x[6:9, 6:9] = np.eye(3) - skew_symmetric(gyro_corrected) * dt
    
    # Block 3b: Orientation error update from gyro bias error (delta_theta += -I * delta_bg * dt)
    F_x[6:9, 12:15] = -np.eye(3) * dt
    
    return F_x

def predict_imu_step(state: ESKFState, imu: IMUMeasurement, dt: float,
                     accel_noise_std: float = 0.1, 
                     gyro_noise_std: float = 0.01):
    """
    Full ESKF Propagation Step:
    1. Updates Nominal States (Position, Velocity, Orientation Quaternion).
    2. Calculates Discrete Error State Jacobian F_x.
    3. Propagates Error Covariance Matrix P.
    """
    gravity_world = np.array([0.0, 0.0, 9.81])

    # Correct raw IMU inputs using current bias estimates
    accel_corr = imu.accel - state.ba
    gyro_corr = imu.gyro - state.bg

    # --- Step 1A: NOMINAL ORIENTATION UPDATE ---
    omega_norm = np.linalg.norm(gyro_corr)
    if omega_norm > 1e-8:
        axis = gyro_corr / omega_norm
        angle = omega_norm * dt
        
        # Incremental rotation delta quaternion dq
        dq = np.array([
            np.cos(angle / 2.0),
            axis[0] * np.sin(angle / 2.0),
            axis[1] * np.sin(angle / 2.0),
            axis[2] * np.sin(angle / 2.0)
        ])
        
        # Quaternion composition q_k+1 = q_k (x) dq
        state.q = quaternion_multiply(state.q, dq)
        state.q /= np.linalg.norm(state.q)  # Normalize quaternion

    # --- Step 1B: NOMINAL POSITION & VELOCITY UPDATE ---
    R_bw = quaternion_to_rotation_matrix(state.q)
    accel_world = R_bw @ accel_corr - gravity_world

    # Kinematic integration with second-order acceleration term
    state.p += state.v * dt + 0.5 * accel_world * (dt ** 2)
    state.v += accel_world * dt

    # --- Step 2: COMPUTE DISCRETE JACOBIAN F_x ---
    F_x = compute_error_jacobian(state, imu.accel, imu.gyro, dt)

    # --- Step 3: PROPAGATE ERROR COVARIANCE P ---
    propagate_covariance(state, F_x, dt, accel_noise_std, gyro_noise_std)