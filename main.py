import numpy as np
import matplotlib.pyplot as plt

from core.eskf_state import ESKFState
from core.propagator import predict_imu_step
from simul.generators import generate_synthetic_stream

def main():
    # 1. GENERATE SYNTHETIC SENSOR & GROUND TRUTH DATA
    duration = 2.0  # seconds
    dt = 0.01       # 100 Hz IMU rate
    imu_stream, gt_stream = generate_synthetic_stream(duration=duration, dt=dt)

    # 2. INITIALIZE ESKF STATE
    state = ESKFState()

    # Data logging containers for evaluation
    time_history = []
    pos_est_history = []
    pos_gt_history = []
    pos_std_history = []  # 3-sigma error bounds from covariance matrix P

    # 3. PROPAGATION LOOP
    prev_timestamp = imu_stream[0].timestamp

    for imu, gt in zip(imu_stream, gt_stream):
        # Calculate step dt dynamically from sensor timestamps
        step_dt = imu.timestamp - prev_timestamp
        prev_timestamp = imu.timestamp

        # Run ESKF prediction step (Nominal state integration + Covariance propagation)
        if step_dt > 0:
            predict_imu_step(
                state=state,
                imu=imu,
                dt=step_dt,
                accel_noise_std=0.1,
                gyro_noise_std=0.01
            )

        # Log trajectory states and 3-sigma position uncertainties
        time_history.append(imu.timestamp)
        pos_est_history.append(state.p.copy())
        pos_gt_history.append(gt.position.copy())
        
        # Standard deviation sigma_p = sqrt(P[0:3, 0:3] diagonal)
        pos_std = np.sqrt(np.diag(state.P[0:3, 0:3]))
        pos_std_history.append(pos_std)

    # Convert logged lists to NumPy arrays for easier slicing
    pos_est = np.array(pos_est_history)
    pos_gt = np.array(pos_gt_history)
    pos_std = np.array(pos_std_history)

    # 4. RESULTS & PERFORMANCE VERIFICATION
    final_error = np.linalg.norm(pos_est[-1] - pos_gt[-1])
    print("===========================================")
    print("ESKF PROPAGATION TEST COMPLETE")
    print("===========================================")
    print(f"Total Simulation Time : {duration:.2f} s ({len(imu_stream)} steps)")
    print(f"Final Estimated Pos   : {pos_est[-1]} m")
    print(f"Final Ground Truth Pos: {pos_gt[-1]} m")
    print(f"Final Position Error  : {final_error:.6f} m")
    print("===========================================")

    # 5. PLOT TRAJECTORY & 3-SIGMA COVARIANCE BOUNDS
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, pos_gt[:, 0], 'k--', label='Ground Truth X (m)')
    plt.plot(time_history, pos_est[:, 0], 'b-', label='ESKF Estimate X (m)')
    
    # Plot 3-sigma confidence bounds (±3 * sigma)
    plt.fill_between(
        time_history,
        pos_est[:, 0] - 3 * pos_std[:, 0],
        pos_est[:, 0] + 3 * pos_std[:, 0],
        color='blue', alpha=0.15, label='±3σ Uncertainty'
    )

    plt.title('ESKF IMU Strapdown Propagation (X-Axis Motion)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()