import os
import numpy as np
from parser import parse_H
from math_model import RLSM, compensate
from plots import plot_sphere_comparison, plot_convergence

path = os.getcwd()
data_path = os.path.join(path, 'sensors_data', 'magnetic_data.txt')

data = parse_H(data_path)
print(f"Загружено точек: {len(data)}")

eps = 0.01
delta_H, delta_K, trace_history = RLSM(data, eps)
print(f"\nСдвиг нуля:  dH = {delta_H}")
print(f"Масштаб:     dK = {delta_K}")

data_comp = compensate(data, delta_H, delta_K)

r_before = np.sqrt(np.sum(data**2, axis=1))
r_after  = np.sqrt(np.sum(data_comp**2, axis=1))
print(f"\nДо калибровки:    среднее r = {np.mean(r_before):.4f}, СКО = {np.std(r_before):.4f}")
print(f"После калибровки: среднее r = {np.mean(r_after):.4f},  СКО = {np.std(r_after):.4f}")

plot_convergence(trace_history, eps)
plot_sphere_comparison(data, data_comp)