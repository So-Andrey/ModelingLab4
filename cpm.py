import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

def initialize_cpm(size, num_cells):

    lattice = np.zeros((size, size), dtype=int)

    for cell_id in range(1, num_cells + 1):
        x, y = np.random.randint(0, size), np.random.randint(0, size)
        lattice[x, y] = cell_id

    return lattice

def cpm_step(lattice, beta, J_matrix, target_areas, lambda_area, mu_perimeter):

    size = lattice.shape[0]

    for _ in range(size * size):
        i, j = np.random.randint(0, size), np.random.randint(0, size)
        current_cell = lattice[i, j]

        if current_cell == 0:
            continue
        
        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        ni, nj = neighbors[np.random.randint(0, 4)]
        ni, nj = ni % size, nj % size
        neighbor_cell = lattice[ni, nj]
        
        if neighbor_cell == current_cell:
            continue
        
        delta_E = 0
        delta_E += J_matrix[current_cell, neighbor_cell] - J_matrix[current_cell, current_cell]
        current_area = np.sum(lattice == current_cell)
        delta_E += lambda_area * (1 - 2 * (current_area >= target_areas[current_cell]))
        perimeter = np.sum(lattice[(i+1)%size, j] != current_cell) + \
                          np.sum(lattice[(i-1)%size, j] != current_cell) + \
                          np.sum(lattice[i, (j+1)%size] != current_cell) + \
                          np.sum(lattice[i, (j-1)%size] != current_cell)
        delta_E += mu_perimeter * perimeter
        
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            lattice[ni, nj] = current_cell
    
    return lattice

def simulate_cpm(size, num_cells, steps, beta, J_matrix, target_areas, 
                lambda_area=0.1, mu_perimeter=0.01, save_interval=10, 
                output_dir="cpm_results"):

    os.makedirs(output_dir, exist_ok=True)
    lattice = initialize_cpm(size, num_cells)
    frames = []
    cmap = colors.ListedColormap(['black'] + [np.random.rand(3) for _ in range(num_cells)])
    
    for step in range(steps + 1):
        lattice = cpm_step(lattice, beta, J_matrix, target_areas, lambda_area, mu_perimeter)
        
        if step % save_interval == 0:
            plt.figure(figsize=(6, 6))
            plt.imshow(lattice, cmap=cmap)
            plt.title(f"Step {step}")
            plt.colorbar()
            plt.savefig(f"{output_dir}/frame_{step:04d}.png")
            plt.close()
            frames.append(lattice.copy())
    
    return frames

params = {
    "size": 50,
    "num_cells": 5,
    "steps": 200,
    "beta": 1.0,
    "J_matrix": np.zeros((6, 6)),  # 5 клеток + фон
    "target_areas": {1: 100, 2: 120, 3: 80, 4: 150, 5: 90},
    "lambda_area": 0.1,
    "mu_perimeter": 0.01,
    "save_interval": 20
}
params["J_matrix"][1:, 1:] = 0.5  # Взаимодействие между клетками
params["J_matrix"][:, 0] = 1.0     # Взаимодействие с фоном

frames = simulate_cpm(**params)

print("Симуляция завершена. Результаты сохранены в папку 'cpm_results'")