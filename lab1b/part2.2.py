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
P0 = np.eye(4) * 10
trajectory = simulationTrajectory(numSteps, F, x0, Q)

print("Final X position:", trajectory[0, -1])


sensor_locations = np.array([[50, 0], [50, 0]])


H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

base_noise=0.1
max_noise = 4


def calculateDistance(sensor_location, true_position):
    return np.linalg.norm(sensor_location - true_position)/45

def calculateDynamicNoise(distance, base_noise, max_noise):
    if distance <1:
        return base_noise
    noise = base_noise + np.log(distance * 1.5)
    return min(noise, max_noise)
def calculateDynamicNoise2(distance, base_noise, max_noise):
    if distance <1:
        return base_noise
    noise = base_noise + np.log(distance * 10)
    return min(noise, max_noise)
def kl_divergence(mu1, sigma1, mu2, sigma2):
    d = len(mu1)
    sigma2_inv = np.linalg.inv(sigma2)
    diff_mu = mu2 - mu1
    term1 = diff_mu.T @ sigma2_inv @ diff_mu
    term2 = np.trace(sigma2_inv @ sigma1)
    term3 = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    kl_divergence_calc = 0.5 * (term1 + term2 - d + term3)
    return kl_divergence_calc
def renyi_divergence(mu1, sigma1, mu2, sigma2, alpha):
    d = len(mu1)
    alpha_comp = 1 - alpha
    diff_mu = mu2 - mu1
    sigma_alpha = alpha * sigma1 + alpha_comp * sigma2
    equation_1 = 0.5 / alpha * np.log(np.linalg.det(sigma_alpha) / (np.linalg.det(sigma1)**alpha * np.linalg.det(sigma2)**alpha_comp))
    equation_2 = 1 / (2 * alpha_comp) * diff_mu.T @ np.linalg.inv(sigma_alpha) @ diff_mu
    renyi_divergence_calc = equation_1 + equation_2
    return renyi_divergence_calc
def renyi_divergence3(mu1, sigma1, mu2, sigma2, alpha3):
  d = len(mu1)
  alpha_comp = 1 - alpha3
  diff_mu = mu2 - mu1
  sigma_alpha = alpha3 * sigma1 + alpha_comp * sigma2
  equation_1 = 0.5 / alpha3 * np.log(np.linalg.det(sigma_alpha) / (np.linalg.det(sigma1)**alpha3 * np.linalg.det(sigma2)**alpha_comp))
  equation_2 = 1 / (2 * alpha_comp) * diff_mu.T @ np.linalg.inv(sigma_alpha) @ diff_mu
  renyi_divergence_calc = equation_1 + equation_2
  return renyi_divergence_calc

def renyi_divergence2(mu1, sigma1, mu2, sigma2, alpha2):
    d = len(mu1)
    alpha_comp = 1 - alpha2
    diff_mu = mu2 - mu1
    sigma_alpha = alpha2 * sigma1 + alpha_comp * sigma2
    equation_1 = 0.5 / alpha2 * np.log(np.linalg.det(sigma_alpha) / (np.linalg.det(sigma1)**alpha2 * np.linalg.det(sigma2)**alpha_comp))
    equation_2 = 1 / (2 * alpha_comp) * diff_mu.T @ np.linalg.inv(sigma_alpha) @ diff_mu
    renyi_divergence_calc = equation_1 + equation_2
    return renyi_divergence_calc

def run_kalmanFilter(F, H, Q, x0, P0, trajectory, numSteps, sensor_locations, base_noise, max_noise):
    num_sensors = len(sensor_locations)
    x_estimate = np.copy(x0)
    P_estimate = np.copy(P0)
    all_x_estimates = np.zeros((num_sensors, 4, numSteps))
    all_P_updates = np.zeros((num_sensors, 4, 4, numSteps))
    all_mutual_info = np.zeros((num_sensors, numSteps))
    all_euclidean_errors = np.zeros((num_sensors, numSteps))
    all_shannon_entropy = np.zeros((num_sensors, numSteps))
    all_kl_divergences = np.zeros((num_sensors, num_sensors, numSteps))
    all_renyi_divergences = np.zeros((num_sensors, num_sensors, numSteps))
    all_renyi_divergences2 = np.zeros((num_sensors, num_sensors, numSteps))
    all_renyi_divergences3 = np.zeros((num_sensors, num_sensors, numSteps))
    all_P_updates = np.zeros((num_sensors, 4, 4, numSteps))

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
            if sensor_idx == 0:

              distance = calculateDistance(sensor_locations[sensor_idx], true_position[:2])
              sigma_p = calculateDynamicNoise(distance, base_noise, max_noise)
            else:
              distance = calculateDistance(sensor_locations[sensor_idx], true_position[:2])
              sigma_p = calculateDynamicNoise2(distance, base_noise, max_noise)

            dynamic_R = np.array([[sigma_p**2, 0], [0, sigma_p**2]])*distance**2
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
            shannon_entropy = 0.5 * np.log((2 * np.pi * np.e) ** 4 * np.linalg.det(P_update))
            all_shannon_entropy[sensor_idx, k] = shannon_entropy




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

    for k in range(numSteps):
        for i in range(num_sensors):
            for j in range(i+1, num_sensors):
                mu1, sigma1 = all_x_estimates[i, :, k], all_P_updates[i, :, :, k]
                mu2, sigma2 = all_x_estimates[j, :, k], all_P_updates[j, :, :, k]
                all_kl_divergences[i, j, k] = kl_divergence(mu1, sigma1, mu2, sigma2)
                all_renyi_divergences[i, j, k] = renyi_divergence(mu1, sigma1, mu2, sigma2, alpha=0.5)
                all_renyi_divergences2[i, j, k] = renyi_divergence(mu1, sigma1, mu2, sigma2, alpha=1.2)
                all_renyi_divergences3[i, j, k] = renyi_divergence(mu1, sigma1, mu2, sigma2, alpha=0.1)



    return all_x_estimates, all_mutual_info, all_euclidean_errors, all_shannon_entropy, all_kl_divergences, all_renyi_divergences, all_renyi_divergences2, all_renyi_divergences3
all_x_estimates, all_mutual_info, all_euclidean_errors, all_shannon_entropy, all_kl_divergences, all_renyi_divergences, all_renyi_divergences2, all_renyi_divergences3 = run_kalmanFilter(F, H, Q, x0, P0, trajectory, numSteps, sensor_locations, base_noise, max_noise)
plt.plot(trajectory[0, :], trajectory[2, :], label='True Trajectory', marker='o')

for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    plt.plot(all_x_estimates[sensor_idx, 0, :], all_x_estimates[sensor_idx, 2, :], label=f'Sensor {sensor_idx+1} Estimated Trajectory', linestyle='dashed')

plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('True vs Estimated Trajectories for Sensors in Same Place')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    plt.plot(range(numSteps), all_shannon_entropy[sensor_idx, :], label=f'Sensor {sensor_idx+1} Shannon Entropy')

plt.xlabel('Time Step')
plt.ylabel('Shannon Entropy')
plt.title('Shannon Entropy over Time for Observation Noise')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))


plt.figure(figsize=(12, 6))
for sensor_idx, (sensor_x, sensor_y) in enumerate(sensor_locations):
    plt.plot(range(numSteps), all_euclidean_errors[sensor_idx, :], label=f'Sensor {sensor_idx+1} Euclidean Error')

plt.xlabel('Time Step')
plt.ylabel('Euclidean Error')
plt.title('Euclidean Error over Time for Sensor\'s in the Same Position')
plt.legend()
plt.show()
