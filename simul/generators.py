# SYNTHETIC DATA GENERATOR (Initial Test Data)

import numpy as np
from utilities.data_types import IMUMeasurement, GroundTruthState

def get_dummy_sensor_data():
    #Generates a single synthetic IMU measurement and Ground Truth state for testing.
    #A timestamp is a precise numerical marker representing the exact moment (in seconds or nanoseconds) a sensor sample was captured by hardware
    #IMUs sample at high speeds while camera samples are at low speeds, so their timestamps will not align perfectly.
    #Timestamps allow you to interleave these out-of-sync streams into a single chronologically ordered event timeline.
    timestamp = 0.0
    
    # Vehicle sitting still on a table: accelerometer reads Gravity (+9.81 on Z)
    imu = IMUMeasurement(
        timestamp=timestamp,
        accel=np.array([0.0, 0.0, 9.81]),  # m/s^2
        gyro=np.array([0.0, 0.0, 0.0])      # rad/s
    )

    #Ground truth represents the absolute, perfect real-world position, velocity, and orientation of the vehicle at any timestamp.
    #To estimate and track errors we need to compare our estimated state against this ground truth.
    gt = GroundTruthState(
        timestamp=timestamp,
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    return imu, gt

# 2b. SYNTHETIC TIME-SERIES GENERATOR (Continuous Stream)

def generate_synthetic_stream(duration: float = 2.0, dt: float = 0.01):
    """
    Generates a synchronized stream of IMU measurements and Ground Truth states over time.
    
    Parameters:
        duration : Total simulation time in seconds (e.g., 2.0s)
        dt       : Sampling interval in seconds (0.01s = 100 Hz sampling rate)
    """
    imu_stream = []
    gt_stream = []
    
    # 1. Generate array of timestamps: [0.00, 0.01, 0.02, ..., 1.99]
    timestamps = np.arange(0.0, duration, dt)
    
    # Gravity vector in World frame (Z-axis pointing UP)
    gravity_world = np.array([0.0, 0.0, 9.81])
    
    # True vehicle acceleration along X-axis (0.5 m/s^2)
    #Accelerometers measure proper acceleration, not coordinate acceleration
    accel_true = np.array([0.5, 0.0, 0.0])
    
    for t in timestamps:
        # --- A. GROUND TRUTH STATE ---
        # Kinematic equations for constant acceleration starting from rest
        pos_true = 0.5 * accel_true * (t ** 2)          # p = 0.5 * a * t^2
        vel_true = accel_true * t                       # v = a * t
        quat_true = np.array([1.0, 0.0, 0.0, 0.0])      # Identity quaternion (no rotation)
        
        gt_state = GroundTruthState(
            timestamp=t,
            position=pos_true,
            orientation=quat_true,
            velocity=vel_true
        )
        gt_stream.append(gt_state)
        
        # --- B. IMU MEASUREMENT ---
        # Accelerometer measures proper acceleration = true motion + gravity
        accel_measured = accel_true + gravity_world
        gyro_measured = np.array([0.0, 0.0, 0.0])       # Zero angular velocity
        
        imu_meas = IMUMeasurement(
            timestamp=t,
            accel=accel_measured,
            gyro=gyro_measured
        )
        imu_stream.append(imu_meas)
        
    return imu_stream, gt_stream
