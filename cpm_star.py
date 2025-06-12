import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

def initialize_cpm(size, num_cells_A, num_cells_B):
    """Инициализация решетки с клетками типов A и B"""
    lattice = np.zeros((size, size), dtype=int)
    
    # Клетки типа A (ID: 1..num_cells_A)
    for cell_id in range(1, num_cells_A + 1):
        x, y = np.random.randint(0, size), np.random.randint(0, size)
        lattice[x, y] = cell_id
    
    # Клетки типа B (ID: num_cells_A+1..num_cells_A+num_cells_B)
    for cell_id in range(num_cells_A + 1, num_cells_A + num_cells_B + 1):
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

def simulate_cpm(size, num_cells_A, num_cells_B, steps, beta, J_matrix, target_areas, 
                lambda_area=0.1, mu_perimeter=0.01, save_interval=10, 
                output_dir="cpm_graner_glazier"):
    os.makedirs(output_dir, exist_ok=True)
    lattice = initialize_cpm(size, num_cells_A, num_cells_B)
    frames = []
    
    # Цвета: черный - фон, синий - тип A, красный - тип B
    cmap = colors.ListedColormap(['black', 'blue', 'red'])
    bounds = [0, 0.5, num_cells_A + 0.5, num_cells_A + num_cells_B + 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    for step in range(steps + 1):
        lattice = cpm_step(lattice, beta, J_matrix, target_areas, lambda_area, mu_perimeter)
        
        if step % save_interval == 0:
            plt.figure(figsize=(8, 8))
            plt.imshow(lattice, cmap=cmap, norm=norm)
            plt.title(f"Step {step}")
            plt.colorbar(ticks=[0, 1, 2], label='0: фон, 1: тип A, 2: тип B')
            plt.savefig(f"{output_dir}/frame_{step:04d}.png")
            plt.close()
            frames.append(lattice.copy())
    
    return frames

# Параметры из Graner & Glazier (1992)
num_cells_A = 10  # Количество клеток типа A
num_cells_B = 10  # Количество клеток типа B
total_cells = num_cells_A + num_cells_B

# Матрица взаимодействий J (0 - фон, 1..num_cells_A - тип A, num_cells_A+1..total_cells - тип B)
J_matrix = np.zeros((total_cells + 1, total_cells + 1))

# Заполняем J_matrix согласно статье:
J_A_A = 10  # A-A взаимодействие
J_B_B = 10  # B-B взаимодействие
J_A_B = 16  # A-B взаимодействие
J_A_M = 12  # A-фон взаимодействие
J_B_M = 12  # B-фон взаимодействие

# Все клетки типа A имеют одинаковые взаимодействия
for i in range(1, num_cells_A + 1):
    for j in range(1, num_cells_A + 1):
        J_matrix[i, j] = J_A_A  # A-A
        
    for j in range(num_cells_A + 1, total_cells + 1):
        J_matrix[i, j] = J_A_B  # A-B
        J_matrix[j, i] = J_A_B  # B-A
        
    J_matrix[i, 0] = J_A_M  # A-фон
    J_matrix[0, i] = J_A_M  # фон-A

# Все клетки типа B имеют одинаковые взаимодействия
for i in range(num_cells_A + 1, total_cells + 1):
    for j in range(num_cells_A + 1, total_cells + 1):
        J_matrix[i, j] = J_B_B  # B-B
        
    J_matrix[i, 0] = J_B_M  # B-фон
    J_matrix[0, i] = J_B_M  # фон-B

params = {
    "size": 100,
    "num_cells_A": num_cells_A,
    "num_cells_B": num_cells_B,
    "steps": 1000,
    "beta": 1.0,
    "J_matrix": J_matrix,
    "target_areas": {i: 50 for i in range(1, total_cells + 1)},  # Одинаковая целевая площадь
    "lambda_area": 0.1,
    "mu_perimeter": 0.01,
    "save_interval": 50
}

frames = simulate_cpm(**params)
print("Симуляция завершена. Результаты сохранены в папку 'cpm_graner_glazier'")