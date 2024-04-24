import numpy as np
import matplotlib.pyplot as plt

def simulationTrajectory(numSteps, F, x0, Q, seed=0):
    np.random.seed(seed)
    trajectory = np.zeros((4, numSteps))
    trajectory[:, 0] = x0
    for k in range(1, numSteps):
        noise = np.random.multivariate_normal([0, 0, 0, 0], Q)
        trajectory[:, k] = F @ trajectory[:, k-1] + noise
    return trajectory

def observations_funct(H, trajectory, R, numSteps):
    observations = np.zeros((2, numSteps))
    for k in range(numSteps):
        expected_trajectory = H @ trajectory[:, k]
        observation_noise = np.random.multivariate_normal([0, 0], R)
        observations[:, k] = expected_trajectory + observation_noise
    return observations

def kalman_filter(F, x_estimate, P_estimate, Q, R, H, observation):
    x_predict = F @ x_estimate
    P_predict = F @ P_estimate @ F.T + Q
    K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
    z = observation
    x_estimate = x_predict + K @ (z - H @ x_predict)
    P_estimate = (np.eye(len(x_estimate)) - K @ H) @ P_predict
    return x_estimate, P_estimate

dt = 0.2
totalTime = 4
numSteps = int(totalTime/dt)
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
x0 = np.array([0, 3, 0, 0])
P0 = np.eye(4) * 10

q_values = [0.005, 0.5, 2]  
observation_noises = [0.005, 0.5, 1]  

plt.figure(figsize=(24, 12))
plot_number = 1

for obs_noise in observation_noises:
    for q in q_values:
        Q = np.eye(4) * q
        R = np.diag([obs_noise, obs_noise])
        trajectory = simulationTrajectory(numSteps, F, x0, Q)
        x_estimate = x0
        P_estimate = P0
        estimated_trajectory = np.zeros((4, numSteps))
        estimated_trajectory[:, 0] = x_estimate
        obstrajectory = observations_funct(H, trajectory, R, numSteps)

        for k in range(numSteps):
            x_estimate, P_estimate = kalman_filter(F, x_estimate, P_estimate, Q, R, H, obstrajectory[:, k])
            estimated_trajectory[:, k] = x_estimate

        plt.subplot(3, 3, plot_number)
        plt.plot(trajectory[0, :], trajectory[2, :], 'k-', label='True Trajectory')
        plt.scatter(obstrajectory[0, :], obstrajectory[1, :], color='red', label='Observed Trajectory', alpha=0.6)
        plt.plot(estimated_trajectory[0, :], estimated_trajectory[2, :], 'b--', label='Estimated Trajectory')
        plt.title(f'Process Noise: {q}, Observation Noise(R): {obs_noise}')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.legend()
        plot_number += 1  

plt.tight_layout()
plt.show()
