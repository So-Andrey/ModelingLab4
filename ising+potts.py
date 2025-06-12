import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


# Модель Изинга
def initialize_ising(size):

    return np.random.choice([-1, 1], size=(size, size))


def ising_step(lattice, beta, J):

    N = lattice.shape[0]

    for _ in range(N * N):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        spin = lattice[i, j]
        neighbors = lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + \
                    lattice[(i-1)%N, j] + lattice[i, (j-1)%N]
        delta_E = 2 * J * spin * neighbors

        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            lattice[i, j] *= -1

    return lattice


def simulate_ising(size, steps, beta, J):

    lattice = initialize_ising(size)
    frames = [lattice.copy()]

    for _ in range(steps):
        lattice = ising_step(lattice, beta, J)
        frames.append(lattice.copy())

    return frames


# Модель Поттса
def initialize_potts(size, q):

    return np.random.randint(0, q, size=(size, size))


def potts_step(lattice, beta, J, q):

    N = lattice.shape[0]

    for _ in range(N * N):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        current = lattice[i, j]
        proposal = np.random.choice([x for x in range(q) if x != current])
        neighbors = [lattice[(i+1)%N, j], lattice[(i-1)%N, j],
                     lattice[i, (j+1)%N], lattice[i, (j-1)%N]]
        delta_E = -J * sum([int(proposal == n) - int(current == n) for n in neighbors])

        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            lattice[i, j] = proposal

    return lattice


def simulate_potts(size, steps, beta, J, q):

    lattice = initialize_potts(size, q)
    frames = [lattice.copy()]

    for _ in range(steps):
        lattice = potts_step(lattice, beta, J, q)
        frames.append(lattice.copy())

    return frames


# Сохранение GIF
def save_gif(frames, filename, cmap='gray'):

    images = []

    for frame in frames:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.axis('off')
        ax.imshow(frame, cmap=cmap, interpolation='nearest', vmin=frame.min(), vmax=frame.max())
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))  # RGBA
        images.append(image)
        plt.close(fig)

    imageio.mimsave(filename, images, fps=10)


# Параметры и запуск
os.makedirs("output", exist_ok=True)
size = 50
steps = 100

for J in [0.5, 1.0, 1.5]:
    frames = simulate_ising(size, steps, beta=0.5, J=J)
    save_gif(frames, f"output/ising_J{J}.gif", cmap='gray')

for beta in [0.2, 0.6, 1.5]:
    frames = simulate_ising(size, steps, beta=beta, J=1.0)
    save_gif(frames, f"output/ising_beta{beta}.gif", cmap='gray')

for beta in [0.2, 0.6, 1.5]:
    frames = simulate_potts(size, steps, beta=beta, J=1.0, q=3)
    save_gif(frames, f"output/potts_beta{beta}_q3.gif", cmap='nipy_spectral')

print("GIF-анимации сохранены в папке output")
