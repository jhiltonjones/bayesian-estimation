import numpy as np
import matplotlib.pyplot as plt


dt = 0.2
totalTime = 4
q = 0.01
numSteps = int(totalTime/dt)
sigma_px = 0.1
sigma_py = 0.1
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
plt.plot(trajectory[0, :], trajectory[2, :], 'b', label='True Trajectory')
plt.plot(x_estimate_store[0, :], x_estimate_store[2, :], 'r--', label='Estimated Trajectory')
plt.plot(observations[0, :], observations[1, :], c='r', label='Observations')
plt.xlabel('Position in X')
plt.ylabel('Position in Y')
plt.legend()
plt.title('Comparison of True and Estimated Trajectories in XY Plane')
plt.grid(True)
plt.show()

plt.plot(observations[0, :], observations[1, :], c='r', label='Observations')
plt.xlabel('Position in X')
plt.ylabel('Position in Y')
plt.title('Observations in XY Plane')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(x_estimate_store[0, :], x_estimate_store[2, :], 'r--', label='Estimated Trajectory')
plt.xlabel('Position in X')
plt.ylabel('Position in Y')
plt.legend()
plt.title('Comparison of True and Estimated Trajectories in XY Plane')
plt.grid(True)
plt.show()

plt.plot(range(numSteps), euclidean_error_store_kalman, 'b', label = 'Euclidean Errors')
plt.xlabel('Time Step')
plt.ylabel('Euclidean Error')
plt.legend()
plt.title('Euclidean Errors')
plt.grid(True)
plt.show()

plt.plot(range(numSteps), euclidean_error_store_observation, 'b', label='Euclidean Errors')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Euclidean Error Over Time')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(range(numSteps), euclidean_error_store_observation, 'r', label='Observation Sensor')
plt.plot(range(numSteps), euclidean_error_store_kalman, 'b', label = 'Kalman Filter')
plt.xlabel('Time Step')
plt.ylabel('Euclidean Error')
plt.title('Euclidean Error between Kalman filter and Observation Sensor')
plt.legend()
plt.grid(True)
plt.show()

print("Observation errors",sum(euclidean_error_store_observation)/len(euclidean_error_store_observation)) 
print("Kalman filter error",sum(euclidean_error_store_kalman)/len(euclidean_error_store_kalman))
