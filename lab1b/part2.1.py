import numpy as np
import matplotlib.pyplot as plt
dt = 5  
totalTime = 100 
numSteps = int(totalTime / dt) 
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  

q = 0.1
Q = q * np.array([[dt**3/3, dt**2/2, 0, 0],
                  [dt**2/2, dt, 0, 0],
                  [0, 0, dt**3/3, dt**2/2],
                  [0, 0, dt**2/2, dt]])  

def simulationTrajectory(numSteps, F, x0, Q):
    trajectory = np.zeros((4, numSteps))
    trajectory[:, 0] = x0
    for k in range(1, numSteps):
        noise = np.random.multivariate_normal([0, 0, 0, 0], Q)
        trajectory[:, k] = F @ trajectory[:, k - 1] + noise
    return trajectory

x0 = np.array([0, 10, 0, 0])  
P0 = np.eye(4) * 1000
trajectory = simulationTrajectory(numSteps, F, x0, Q)


sensor_locations = np.array([[0, 0], [200, 0], [400, 0]])


H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

base_noise=0.1
max_noise = 4
window_size = 5
smoothed_length = numSteps - window_size + 1
num_sensors = len(sensor_locations)
def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def calculateDistance(sensor_location, true_position):
    return np.linalg.norm(sensor_location - true_position)/10

def calculateDynamicNoise(distance, base_noise, max_noise):
    if distance <1:
        return base_noise
    noise = base_noise + np.log(distance) * 0.05
    return min(noise, max_noise)


def run_kalmanFilter(F, H, Q, x0, P0, trajectory, numSteps, sensor_locations, base_noise, max_noise):
    x_estimate = np.copy(x0)
    P_estimate = np.copy(P0)
    
    all_x_estimates = np.zeros((num_sensors, 4, numSteps))
    all_P_updates = np.zeros((num_sensors, 4, 4, numSteps))
    all_mutual_info = np.zeros((num_sensors, numSteps))
    all_euclidean_errors = np.zeros((num_sensors, numSteps))
    all_P_updates = np.zeros((num_sensors, 4, 4, numSteps))
    smoothed_all_euclidean_errors = np.zeros((num_sensors, smoothed_length))
    smoothed_all_mutual_information = np.zeros((num_sensors, smoothed_length))

    
    final_x_estimate_store = np.zeros((4, numSteps))
    final_mutual_information = np.zeros(numSteps)
    best_euclidean_errors = np.zeros(numSteps)

    for k in range(numSteps):
        max_mutual_info = -np.inf
        best_sensor_idx = -1
        best_P_predict = None
        best_x_predict = None

        for sensor_idx in range(num_sensors):
            
            x_predict = F @ x_estimate
            P_predict = F @ P_estimate @ F.T + Q
            true_position = trajectory[[0, 2], k]
            distance = calculateDistance(sensor_locations[sensor_idx], true_position[:2])
            sigma_p = calculateDynamicNoise(distance, base_noise, max_noise)
            dynamic_R = np.array([[sigma_p**2, 0], [0, sigma_p**2]])*distance**2
            if np.linalg.det(dynamic_R) < 1e-10:
              dynamic_R += np.eye(2) * 1e-10
            observation_noise = np.random.multivariate_normal([0, 0], dynamic_R)
            z = true_position + observation_noise

            K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + dynamic_R)

            x_update = x_predict + K @ (z - H @ x_predict)
            P_update = (np.eye(4) - K @ H) @ P_predict

            mutual_info = 0.5 * np.log2(np.linalg.det(H @ P_predict @ H.T + dynamic_R) / np.linalg.det(dynamic_R))

            all_x_estimates[sensor_idx, :, k] = x_update
            all_P_updates[sensor_idx, :, :, k] = P_update
            all_mutual_info[sensor_idx, k] = mutual_info
            all_euclidean_errors[sensor_idx, k] = np.linalg.norm(trajectory[[0, 2], k] - x_update[[0, 2]])
            smoothed_all_euclidean_errors[sensor_idx, :] = moving_average(all_euclidean_errors[sensor_idx, :], window_size)
            smoothed_all_mutual_information[sensor_idx, :] = moving_average(all_mutual_info[sensor_idx, :], window_size)

            if mutual_info > max_mutual_info:
                max_mutual_info = mutual_info
                best_sensor_idx = sensor_idx
                best_P_predict = P_predict
                best_x_predict = x_predict



        best_sensor_distance = calculateDistance(sensor_locations[best_sensor_idx], trajectory[[0, 2], k])
        best_sigma_p = calculateDynamicNoise(best_sensor_distance, base_noise, max_noise)
        best_dynamic_R = np.array([[best_sigma_p**2, 0], [0, best_sigma_p**2]])*best_sensor_distance**2
        best_observation = trajectory[[0, 2], k] + np.random.multivariate_normal([0, 0], best_dynamic_R)
        K = best_P_predict @ H.T @ np.linalg.inv(H @ best_P_predict @ H.T + best_dynamic_R)
        x_estimate = best_x_predict + K @ (best_observation - H @ best_x_predict)
        P_estimate = (np.eye(4) - K @ H) @ best_P_predict


        final_x_estimate_store[:, k] = x_estimate
        final_mutual_information[k] = max_mutual_info
        best_euclidean_errors[k] = np.linalg.norm(trajectory[[0, 2], k] - x_estimate[[0, 2]])



    return all_x_estimates, all_mutual_info, all_euclidean_errors, final_x_estimate_store, final_mutual_information, best_euclidean_errors, smoothed_all_mutual_information, smoothed_all_euclidean_errors

all_x_estimates, all_mutual_info, all_euclidean_errors,  final_x_estimate_store, final_mutual_information, best_euclidean_errors, smoothed_all_mutual_information, smoothed_all_euclidean_errors = run_kalmanFilter(F, H, Q, x0, P0, trajectory, numSteps, sensor_locations, base_noise, max_noise)

plt.figure(figsize=(12, 6))

plt.plot(trajectory[0, :], trajectory[2, :], label='True Trajectory', marker='o')

for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    plt.plot(all_x_estimates[sensor_idx, 0, :], all_x_estimates[sensor_idx, 2, :], label=f'Sensor {sensor_idx+1} Estimated Trajectory', linestyle='dashed')

plt.plot(final_x_estimate_store[0, :], final_x_estimate_store[2, :], label='Best Sensor Estimated Trajectory', color='black', linewidth=2, linestyle='--', alpha=0.5)

plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('True vs Estimated Trajectories ')
plt.legend()
plt.show()

for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    average_euclidean_error = np.sum(all_euclidean_errors[sensor_idx, :]) / numSteps

    print(f'Average Euclidean Error for {sensor_idx} =', average_euclidean_error)
average_euclidean_error_best = np.sum(best_euclidean_errors) / numSteps
print("Average Euclidean Error for best sensor =", average_euclidean_error_best)

plt.figure(figsize=(12, 6))

plt.plot(best_euclidean_errors, final_mutual_information, label='Mutual Information and Euclidean Error of Best Sensor', color='black', linewidth=2, linestyle='--', alpha=0.5)
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('M vs E')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for sensor_idx, _ in enumerate(sensor_locations):
    plt.plot(range(numSteps), all_mutual_info[sensor_idx, :], label=f'Sensor {sensor_idx+1} Mutual Information')

plt.plot(range(numSteps), final_mutual_information, label='Best Sensor Mutual Information', color='black', linewidth=2, linestyle='--', alpha=0.5)

plt.xlabel('Time Step')
plt.ylabel('Mutual Information')
plt.title('Mutual Information over Time ')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for sensor_idx, _ in enumerate(sensor_locations):
    plt.plot(range(smoothed_length), smoothed_all_mutual_information[sensor_idx, :], label=f'Sensor {sensor_idx+1} Mutual Information')

plt.xlabel('Time Step')
plt.ylabel('Moving Average Mutual Information')
plt.title('Moving Average Mutual Information ')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    plt.plot(range(numSteps), all_euclidean_errors[sensor_idx, :], label=f'Sensor {sensor_idx+1} Euclidean Error')

plt.plot(range(numSteps), best_euclidean_errors, label='Best Sensor Euclidean Error', color='black', linewidth=2, linestyle='--', alpha=0.5)

plt.xlabel('Time Step')
plt.ylabel('Euclidean Error')
plt.title('Euclidean Error over Time ')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    plt.plot(range(smoothed_length), smoothed_all_euclidean_errors[sensor_idx, :], label=f'Sensor {sensor_idx+1} Euclidean Error')

plt.xlabel('Time Step')
plt.ylabel('Euclidean Error Moving Average')
plt.title('Moving Euclidean Error ')
plt.legend()
plt.show()


