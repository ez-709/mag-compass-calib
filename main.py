import os
import numpy as np
from parser import parse_H
from math_model import RLSM, GA, compensate
from plots import plot_sphere_comparison, plot_convergence

path = os.getcwd()
data_path = os.path.join(path, 'sensors_data', 'magnetic_data.txt')

data = parse_H(data_path)

r_norm = np.mean(np.linalg.norm(data, axis=1))
data_scaled = data / r_norm

eps = 0.01

print("=== RLSM ===")
delta_H, delta_K, trace_history = RLSM(data_scaled, eps)
print(f"dH = {delta_H}")
print(f"dK = {delta_K}")

data_comp = compensate(data_scaled, delta_H, delta_K)
r_before = np.linalg.norm(data_scaled, axis=1)
r_after  = np.linalg.norm(data_comp, axis=1)
print(f"before: r = {np.mean(r_before):.4f}, std = {np.std(r_before):.4f}")
print(f"after:  r = {np.mean(r_after):.4f},  std = {np.std(r_after):.4f}")

plot_convergence(trace_history, eps)
plot_sphere_comparison(data_scaled, data_comp)

print("\n=== GA ===")
delta_H_ga, delta_K_ga, fitness_history = GA(data_scaled)
print(f"dH = {delta_H_ga}")
print(f"dK = {delta_K_ga}")

data_comp_ga = compensate(data_scaled, delta_H_ga, delta_K_ga)
r_after_ga = np.linalg.norm(data_comp_ga, axis=1)
print(f"before: r = {np.mean(r_before):.4f}, std = {np.std(r_before):.4f}")
print(f"after:  r = {np.mean(r_after_ga):.4f},  std = {np.std(r_after_ga):.4f}")

plot_convergence(fitness_history, eps)
plot_sphere_comparison(data_scaled, data_comp_ga)