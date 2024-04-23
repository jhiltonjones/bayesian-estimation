import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dt = 2
totalTime = 100
numSteps = int(totalTime / dt)
x0 = np.array([0, 0, 0, 0])
P0 = np.eye(4) * 1000
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

q_values = np.arange(0.01, 0.31, 0.05)
sigma_values = np.arange(1, 21, 2)

def run_simulation(q, sigma_px, sigma_py):
    Q = q * np.array([[dt**3/3, dt**2/2, 0, 0],
                      [dt**2/2, dt, 0, 0],
                      [0, 0, dt**3/3, dt**2/2],
                      [0, 0, dt**2/2, dt]])

    R = np.array([[sigma_px**2, 0],
                  [0, sigma_py**2]])

    trajectory = run_simulation(numSteps, F, x0, Q)
    observations = observations_funct(H, trajectory, R, numSteps)

    x_estimate = x0
    P_estimate = P0
    euclidean_error_store_kalman = np.zeros(numSteps)
    euclidean_error_store_observation = np.zeros(numSteps)

    for k in range(numSteps):
        x_estimate, P_estimate, K, z = kalman_filter(F, x_estimate, P_estimate, Q, R, H, observations[:, k])
        euclidean_error_store_kalman[k] = euclidean_error_funct(trajectory[:2, k], x_estimate[:2])
        euclidean_error_store_observation[k] = euclidean_error_funct(trajectory[:2, k], observations[:2, k])

    return euclidean_error_store_kalman, euclidean_error_store_observation

results = {}
for q in q_values:
    for sigma in sigma_values:
        key = f"q={q}, sigma={sigma}"
        results[key] = run_simulation(q, sigma, sigma)

results.keys()

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


for key in results:
    q_value, sigma_value = map(float, key.replace('q=', '').replace('sigma=', '').split(', '))
    Q = q_value * np.array([[dt**3/3, dt**2/2, 0, 0],
                            [dt**2/2, dt, 0, 0],
                            [0, 0, dt**3/3, dt**2/2],
                            [0, 0, dt**2/2, dt]])
    R = np.array([[sigma_value**2, 0],
                  [0, sigma_value**2]])
    trajectory = run_simulation(numSteps, F, x0, Q)
    observations = observations_funct(H, trajectory, R, numSteps)
    x_estimate = x0
    P_estimate = P0
    x_estimate_store = np.zeros_like(trajectory)
    for k in range(numSteps):
        x_estimate, P_estimate, K, z = kalman_filter(F, x_estimate, P_estimate, Q, R, H, observations[:, k])
        x_estimate_store[:, k] = x_estimate
    euclidean_error_store_kalman, euclidean_error_store_observation = results[key]



    plt.figure(figsize=(8, 12))
    plt.suptitle(f'Plots for {key}', fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(trajectory[0, :], trajectory[2, :], 'b', label='True Trajectory')
    plt.plot(x_estimate_store[0, :], x_estimate_store[2, :], 'r--', label='Estimated Trajectory')
    plt.plot(observations[0, :], observations[1, :], 'go', label='Observations', markersize=3)
    plt.xlabel('Position in X')
    plt.ylabel('Position in Y')
    plt.legend()
    plt.title('True, Estimated Trajectories and Observations in XY Plane')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(range(numSteps), euclidean_error_store_observation, 'r', label='Observation Sensor')
    plt.plot(range(numSteps), euclidean_error_store_kalman, 'b', label='Kalman Filter')
    plt.xlabel('Time Step')
    plt.ylabel('Euclidean Error')
    plt.title('Comparison of Euclidean Errors: Kalman Filter vs Observation Sensor')
    plt.legend()
    plt.grid(True)


    plt.subplot(3, 1, 3)
    plt.plot(range(numSteps), euclidean_error_store_kalman, 'b', label = 'Kalman Filter Errors')
    plt.xlabel('Time Step')
    plt.ylabel('Euclidean Error')
    plt.legend()
    plt.title('Kalman Filter Euclidean Errors Over Time')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()  
    print("Observation errors",sum(euclidean_error_store_observation)/len(euclidean_error_store_observation)) 
    print("Kalman filter error",sum(euclidean_error_store_kalman)/len(euclidean_error_store_kalman))

average_kalman_errors = np.zeros((len(q_values), len(sigma_values)))
average_observation_errors = np.zeros((len(q_values), len(sigma_values)))

for i, q in enumerate(q_values):
    for j, sigma in enumerate(sigma_values):
        kalman_error, observation_error = run_simulation(q, sigma, sigma)
        average_kalman_errors[i, j] = np.mean(kalman_error)
        average_observation_errors[i, j] = np.mean(observation_error)
Q, Sigma = np.meshgrid(q_values, sigma_values)
Q = Q.T
Sigma = Sigma.T


fig = plt.figure(figsize=(20, 20))


ax1 = fig.add_subplot(222, projection='3d')
ax1.plot_surface(Q, Sigma, average_kalman_errors, cmap='coolwarm')
ax1.set_xlabel('Q value')
ax1.set_ylabel('Sigma value')
ax1.set_zlabel('Avg Error', labelpad=20)  
ax1.set_title('Kalman Filter Error Surface')


ax2 = fig.add_subplot(221, projection='3d')
ax2.plot_surface(Q, Sigma, average_observation_errors, cmap='coolwarm')
ax2.set_xlabel('Q value')
ax2.set_ylabel('Sigma value')
ax2.set_zlabel('Avg Error', labelpad=20)  
ax2.set_title('Observation Sensor Error Surface')


fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)


fig.savefig("3d_plots.png", dpi=250, bbox_inches='tight')


plt.show()