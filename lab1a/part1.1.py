import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dt = 0.2
totalTime = 4
q = 0.01
numSteps = int(totalTime/dt)
sigma_px = 0.05
sigma_py = 0.05
x0 = np.array([0, 3, 0, 0])
P0 = np.eye(4) * 10
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])

Q = q * np.array([[dt**3/3, dt**2/2, 0, 0],
                  [dt**2/2, dt, 0, 0],
                  [0, 0, dt**3/3, dt**2/2],
                  [0, 0, dt**2/2, dt]])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

R = np.array([[sigma_px**2, 0],
              [0, sigma_py**2]])


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
    P_predict = Q + F @ P_estimate @ F.T

    K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
    z = observation
    x_estimate = x_predict + K @ (z - H @ x_predict)
    P_estimate = (np.eye(4) - K @ H) @ P_predict
    return x_estimate, P_estimate, K, z

def euclidean_error_funct(point1, point2):
    euclidean_error = np.linalg.norm(point1 - point2)
    return euclidean_error

numSteps = int(totalTime / dt)
trajectory = simulationTrajectory(numSteps, F, x0, Q)
observations = observations_funct(H, trajectory, R, numSteps)

x_estimate = x0
P_estimate = P0
x_estimate_store = np.zeros_like(trajectory)
P_estimate_store = np.zeros((4, 4, numSteps))
euclidean_error_store_kalman = np.zeros(numSteps)
euclidean_error_store_observation = np.zeros(numSteps)

for k in range(numSteps):
    x_estimate, P_estimate, K, z = kalman_filter(F, x_estimate, P_estimate, Q, R, H, observations[:, k])
    euclidean_error_store_kalman[k] = euclidean_error_funct(trajectory[:2, k], x_estimate[:2])
    s = euclidean_error_funct(trajectory[:2, k], observations[:2, k])
    print(s)
    euclidean_error_store_observation[k] = s
    x_estimate_store[:, k] = x_estimate
    P_estimate_store[:, :, k] = P_estimate



fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))  

axes[0, 0].plot(trajectory[0, :], trajectory[2, :], 'b', label='True Trajectory')
axes[0, 0].plot(x_estimate_store[0, :], x_estimate_store[2, :], 'r--', label='Estimated Trajectory')
axes[0, 0].plot(observations[0, :], observations[1, :], 'g', label='Observations')
axes[0, 0].set_xlabel('Position in X')
axes[0, 0].set_ylabel('Position in Y')
axes[0, 0].legend()
axes[0, 0].set_title('True and Estimated Trajectories')

axes[0, 1].plot(observations[0, :], observations[1, :], 'r', label='Observations')
axes[0, 1].set_xlabel('Position in X')
axes[0, 1].set_ylabel('Position in Y')
axes[0, 1].set_title('Observations in XY Plane')
axes[0, 1].legend()

axes[0, 2].plot(x_estimate_store[0, :], x_estimate_store[2, :], 'r--', label='Estimated Trajectory')
axes[0, 2].set_xlabel('Position in X')
axes[0, 2].set_ylabel('Position in Y')
axes[0, 2].legend()
axes[0, 2].set_title('Estimated Trajectories in XY Plane')

axes[1, 0].plot(range(numSteps), euclidean_error_store_kalman, 'b', label='Kalman Filter Errors')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Euclidean Error')
axes[1, 0].legend()
axes[1, 0].set_title('Kalman Filter Errors Over Time')

axes[1, 1].plot(range(numSteps), euclidean_error_store_observation, 'r', label='Observation Errors')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Error')
axes[1, 1].set_title('Observation Errors Over Time')
axes[1, 1].legend()

axes[1, 2].plot(range(numSteps), euclidean_error_store_observation, 'r', label='Observation Sensor')
axes[1, 2].plot(range(numSteps), euclidean_error_store_kalman, 'b', label='Kalman Filter')
axes[1, 2].set_xlabel('Time Step')
axes[1, 2].set_ylabel('Euclidean Error')
axes[1, 2].set_title('Comparison of Errors')
axes[1, 2].legend()

plt.tight_layout()
plt.show()


q_values = np.linspace(0.005, 2, 100)  
sigma_values = np.linspace(0.005, 1, 100) 

average_kalman_errors = np.zeros((len(q_values), len(sigma_values)))

for i, q in enumerate(q_values):
    Q = q * np.array([[dt**3/3, dt**2/2, 0, 0],
                      [dt**2/2, dt, 0, 0],
                      [0, 0, dt**3/3, dt**2/2],
                      [0, 0, dt**2/2, dt]])
    trajectory = simulationTrajectory(numSteps, F, x0, Q)
    for j, sigma in enumerate(sigma_values):
        R = np.array([[sigma**2, 0],
                      [0, sigma**2]])
        observations = observations_funct(H, trajectory, R, numSteps)
        x_estimate = x0
        P_estimate = P0
        euclidean_errors = []

        for k in range(numSteps):
            x_estimate, P_estimate, K, z = kalman_filter(F, x_estimate, P_estimate, Q, R, H, observations[:, k])
            error = np.linalg.norm(trajectory[:2, k] - x_estimate[:2])  
            euclidean_errors.append(error)

        average_kalman_errors[i, j] = np.mean(euclidean_errors)

Q, Sigma = np.meshgrid(q_values, sigma_values)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Q, Sigma, average_kalman_errors.T, cmap='viridis')  

ax.set_xlabel('Process Noise (q)')
ax.set_ylabel('Measurement Noise (sigma)')
ax.set_zlabel('Average Euclidean Error')
ax.set_title('Kalman Filter Error Surface Analysis')
fig.colorbar(surf)

plt.show()

