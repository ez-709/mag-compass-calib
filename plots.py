import numpy as np
import matplotlib.pyplot as plt

def plot_sphere_comparison(H_raw, H_comp):
    fig = plt.figure(figsize=(18, 12))
    
    # --- 3D графики ---
    ax1 = fig.add_subplot(241, projection='3d')
    ax1.scatter(H_raw[:, 0], H_raw[:, 1], H_raw[:, 2], s=5, c='red', alpha=0.6)
    ax1.set_title('До калибровки (3D)')
    ax1.set_xlabel('H1')
    ax1.set_ylabel('H2')
    ax1.set_zlabel('H3')
    max_range = np.max(np.abs(H_raw))
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    ax1.set_zlim(-max_range, max_range)

    ax2 = fig.add_subplot(245, projection='3d')
    ax2.scatter(H_comp[:, 0], H_comp[:, 1], H_comp[:, 2], s=5, c='blue', alpha=0.6)
    ax2.set_title('После калибровки (3D)')
    ax2.set_xlabel('H1')
    ax2.set_ylabel('H2')
    ax2.set_zlabel('H3')
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    r = np.mean(np.sqrt(np.sum(H_comp**2, axis=1)))
    ax2.plot_surface(
        r * np.outer(np.cos(u), np.sin(v)),
        r * np.outer(np.sin(u), np.sin(v)),
        r * np.outer(np.ones(np.size(u)), np.cos(v)),
        alpha=0.1, color='blue'
    )
    max_range = np.max(np.abs(H_comp))
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.set_zlim(-max_range, max_range)

    # --- Проекции ---
    projections = [
        (0, 1, 'H1', 'H2', 'XY'),
        (0, 2, 'H1', 'H3', 'XZ'),
        (1, 2, 'H2', 'H3', 'YZ'),
    ]

    for i, (xi, yi, xlabel, ylabel, title) in enumerate(projections):
        ax = fig.add_subplot(2, 4, i + 2)
        ax.scatter(H_raw[:, xi], H_raw[:, yi], s=5, c='red', alpha=0.8)
        ax.set_title(f'До — проекция {title}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.grid(True)

        ax = fig.add_subplot(2, 4, i + 6)
        ax.scatter(H_comp[:, xi], H_comp[:, yi], s=5, c='blue', alpha=0.8)
        ax.set_title(f'После — проекция {title}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_convergence(trace_history, eps):
    plt.figure(figsize=(8, 4))
    plt.plot(trace_history)
    plt.axhline(y=eps, color='red', linestyle='--', label=f'eps = {eps}')
    plt.xlabel('Итерация')
    plt.ylabel('Сумма диаг(P)')
    plt.title('Сходимость РМНК')
    plt.legend()
    plt.grid(True)
    plt.show()