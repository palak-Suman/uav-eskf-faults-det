from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

#numpy is used for numerical operations and array handling, which is essential for representing vectors and matrices.
#numpy does not have a built-in quaternion type, so we represent quaternions as 4-element numpy arrays [w, x, y, z].

#The state vector models a full 6-DOF (Degrees of Freedom) rigid body in 3D space.

# 1. CORE DATA STRUCTURES

#@xyz is a decorator that automatically generates special methods like __init__() and __repr__() for classes. 
# It is used here to define simple data structures for IMU measurements and ground truth states.

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

class StateVector:
    #Stores the full estimated state of the vehicle.
    def __init__(self):
        self.p = np.zeros(3)                 # Position in World frame [x, y, z]
        self.v = np.zeros(3)                 # Velocity in World frame [vx, vy, vz]
        self.q = np.array([1.0, 0, 0, 0])    # Orientation [w, x, y, z] (Identity)
        self.ba = np.zeros(3)                # Accelerometer bias [b_ax, b_ay, b_az]
        self.bg = np.zeros(3)                # Gyroscope bias [b_gx, b_gy, b_gz]

#np.array([1.0, 0, 0, 0]): Converts a standard Python list into a fixed NumPy array.
#Here, it creates a 4-element vector representing an initial identity unit quaternion (w=1,x=0,y=0,z=0), meaning zero rotational offset.
#np.ndarray: The primary data type in NumPy representing an n-dimensional array
#np.zeros(3): Instantiates a 1D vector of length 3 populated with zeros: array([0., 0., 0.])

# 2. SYNTHETIC DATA GENERATOR (Initial Test Data)

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

# 3. IMU INTEGRATOR / DEAD RECKONING ENGINE (Step 3)

    #Performs a single step of forward Euler IMU dead-reckoning integration.
    #1.Subtract gravity from the accelerometer measurement to get the net acceleration in the world frame.
    #IMU sensor is glued to the drone's chassis, its accelerometer and gyroscope measure everything relative to the drone's body not the room.
    #World Frame = Absolute / Fixed: The room or map coordinate system. It never moves or rotates.
    #Body Frame = Relative / Moving: Attached directly to the IMU sensor. It rolls, pitches, and turns with the vehicle.

    #Position and velocity updates bridge the gap between fast IMU predictions and slow camera corrections.
    #2.Update the position using the current velocity and time step.
    #3.Update the velocity using the net acceleration and time step.

def predict_imu_step(state: StateVector, imu: IMUMeasurement, dt: float):
    gravity_world = np.array([0.0, 0.0, 9.81])
    
    # 1. Convert measured proper acceleration into coordinate acceleration in World frame
    # Subtract accelerometer bias (ba) and remove World gravity (g)
    accel_world = (imu.accel - state.ba) - gravity_world
    
    # 2. Position Euler Integration: p_{k+1} = p_k + v_k * dt
    # Uses velocity at start of interval (forward Euler step)
    state.p = state.p + state.v * dt
    
    # 3. Velocity Euler Integration: v_{k+1} = v_k + a_world * dt
    state.v = state.v + accel_world * dt

# 4. STEP 4 VISUALIZATION UTILITY

def plot_error_analysis(timestamps, gt_positions, est_positions, pos_errors):
    """
    Renders 3D trajectory comparison and 2D position drift accumulation over time.
    """
    fig = plt.figure(figsize=(12, 5))

    # Plot A: Trajectory comparison
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    gt_pos = np.array(gt_positions)
    est_pos = np.array(est_positions)
    
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'g-', label='Ground Truth', linewidth=2)
    ax1.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r--', label='Euler Dead Reckoning', linewidth=1.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()

    # Plot B: Error Drift Growth over time
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(timestamps, np.array(pos_errors) * 1000, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (mm)')
    ax2.set_title('Accumulated Euler Truncation Error (Drift)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# 5. PIPELINE ENTRY POINT

if __name__ == "__main__":
    print("--- STEP 1: INITIALIZING STATE & TESTING INGESTION ---")
    
    state = StateVector()
    imu_sample, gt_sample = get_dummy_sensor_data()
    
    state.p = gt_sample.position.copy()
    state.v = gt_sample.velocity.copy()
    state.q = gt_sample.orientation.copy()
    
    print(f"\n[IMU Received at t={imu_sample.timestamp}s]:")
    print(f"  Accel : {imu_sample.accel} m/s^2")
    print(f"  Gyro  : {imu_sample.gyro} rad/s")
    
    print(f"\n[State Initialized]:")
    print(f"  Position    : {state.p}")
    print(f"  Velocity    : {state.v}")
    print(f"  Orientation : {state.q}")
    print(f"  Accel Bias  : {state.ba}")
    print(f"  Gyro Bias   : {state.bg}")
    
    print("\n Step 1 complete: Data structures and test pipeline functional!\n")

    print("--- STEP 2: MULTI-SAMPLE SYNTHETIC STREAM GENERATION ---")
    
    dt = 0.01
    imu_stream, gt_stream = generate_synthetic_stream(duration=2.0, dt=dt)
    
    print(f"Total IMU Samples Generated   : {len(imu_stream)}")
    print(f"Total GT Samples Generated    : {len(gt_stream)}")
    print(" Step 2 complete: Synthetic continuous data stream generated and verified!\n")

    print("--- STEP 3: RUNNING IMU DEAD RECKONING INTEGRATION LOOP ---")
    
    # Initialize estimated state to starting Ground Truth (t = 0.0s)
    estimated_state = StateVector()
    estimated_state.p = gt_stream[0].position.copy()
    estimated_state.v = gt_stream[0].velocity.copy()
    estimated_state.q = gt_stream[0].orientation.copy()
    
    # Logging containers for Step 4 continuous tracking
    timestamps = [gt_stream[0].timestamp]
    gt_positions = [gt_stream[0].position.copy()]
    est_positions = [estimated_state.p.copy()]
    pos_errors = [0.0]
    vel_errors = [0.0]
    
    print("Starting dead-reckoning integration over 200 samples...\n")
    
    # Loop through the continuous stream (Skipping index 0 as it's time zero initialization)
    for i in range(1, len(imu_stream)):
        imu_meas = imu_stream[i]
        gt_meas = gt_stream[i]
        
        # Perform 1 step of Euler dead reckoning
        predict_imu_step(estimated_state, imu_meas, dt)
        
        # Calculate Euclidean distances using np.linalg.norm
        #np.linalg.norm() calculates the Euclidean length (magnitude-scalar quantity) of a vector
        pos_err = np.linalg.norm(estimated_state.p - gt_meas.position)
        vel_err = np.linalg.norm(estimated_state.v - gt_meas.velocity)
        
        # Log continuous metrics for Step 4
        timestamps.append(gt_meas.timestamp)
        gt_positions.append(gt_meas.position.copy())
        est_positions.append(estimated_state.p.copy())
        pos_errors.append(pos_err)
        vel_errors.append(vel_err)
        
        # Checkpoint comparison at t = 1.0s (Index 100)
        if i == 100:
            print(f"[Checkpoint t = 1.00s (Sample 100)]:")
            print(f"  GT Position        : {gt_meas.position} m")
            print(f"  Estimated Position : {estimated_state.p} m")
            print(f"  Position Error     : {pos_err * 1000:.2f} mm")
            print(f"  Velocity Error     : {vel_err * 1000:.2f} mm/s\n")

    # Final Checkpoint at t = 1.99s (Index 199)
    gt_last = gt_stream[-1]
    final_pos_err = np.linalg.norm(estimated_state.p - gt_last.position)
    final_vel_err = np.linalg.norm(estimated_state.v - gt_last.velocity)
    
    print(f"[Final Checkpoint t = 1.99s (Sample 199)]:")
    print(f"  GT Position        : {gt_last.position} m")
    print(f"  Estimated Position : {estimated_state.p} m")
    print(f"  Position Error     : {final_pos_err * 1000:.2f} mm")
    print(f"  Velocity Error     : {final_vel_err * 1000:.2f} mm/s\n")
    
    print(" Step 3 complete: Dead reckoning loop executed successfully!\n")

    print("--- STEP 4: ERROR ANALYSIS & CONTINUOUS DRIFT TRACKING ---")
    print(f"Error Analysis Summary over {len(timestamps)} timesteps:")
    print(f"  Initial Position Error : {pos_errors[0] * 1000:.2f} mm")
    print(f"  Checkpoint t=1.00s     : {pos_errors[100] * 1000:.2f} mm")
    print(f"  Final Error t=1.99s    : {pos_errors[-1] * 1000:.2f} mm")
    print(f"  Maximum Drift Rate     : {(pos_errors[-1] / timestamps[-1]) * 1000:.2f} mm/s")
    
    print("\n Step 4 complete: Continuous error drift recorded and analyzed!")
    
    # Render 3D trajectory and error plots
    plot_error_analysis(timestamps, gt_positions, est_positions, pos_errors)

    '''RECORDS:
    Step 1: State vector representations and single-sample data structures.
    Step 2: Synthetic sensor stream generation.
    Step 3: IMU Strapdown Euler Integration & coordinate transformation principles.
    Step 4: Continuous error drift analysis and matplotlib spatial visualization.
    '''