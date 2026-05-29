import numpy as np
import matplotlib.pyplot as plt

def plot_sphere_comparison(H_raw, H_comp):
    fig = plt.figure(figsize=(18, 12))
    all_data = np.vstack([H_raw, H_comp])
    global_max = np.max(np.abs(all_data))
    limit = global_max * 1.05

    ax1 = fig.add_subplot(241, projection='3d')
    ax1.scatter(H_raw[:, 0], H_raw[:, 1], H_raw[:, 2], s=5, c='red', alpha=0.6)
    ax1.set_title('Before Calibration (3D)')
    ax1.set_xlabel('H1')
    ax1.set_ylabel('H2')
    ax1.set_zlabel('H3')
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax1.set_zlim(-limit, limit)
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=20, azim=45)

    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    r_raw = np.mean(np.sqrt(np.sum(H_raw**2, axis=1)))
    ax1.plot_surface(
        r_raw * np.outer(np.cos(u), np.sin(v)),
        r_raw * np.outer(np.sin(u), np.sin(v)),
        r_raw * np.outer(np.ones(np.size(u)), np.cos(v)),
        alpha=0.1, color='red'
    )

    ax2 = fig.add_subplot(245, projection='3d')
    ax2.scatter(H_comp[:, 0], H_comp[:, 1], H_comp[:, 2], s=5, c='blue', alpha=0.6)
    ax2.set_title('After Calibration (3D)')
    ax2.set_xlabel('H1')
    ax2.set_ylabel('H2')
    ax2.set_zlabel('H3')
    ax2.set_xlim(-limit, limit)
    ax2.set_ylim(-limit, limit)
    ax2.set_zlim(-limit, limit)
    ax2.set_box_aspect([1, 1, 1])
    ax2.view_init(elev=20, azim=45)

    r_comp = np.mean(np.sqrt(np.sum(H_comp**2, axis=1)))
    ax2.plot_surface(
        r_comp * np.outer(np.cos(u), np.sin(v)),
        r_comp * np.outer(np.sin(u), np.sin(v)),
        r_comp * np.outer(np.ones(np.size(u)), np.cos(v)),
        alpha=0.1, color='blue'
    )

    projections = [
        (0, 1, 'H1', 'H2', 'XY'),
        (0, 2, 'H1', 'H3', 'XZ'),
        (1, 2, 'H2', 'H3', 'YZ'),
    ]

    for i, (xi, yi, xlabel, ylabel, title) in enumerate(projections):
        ax_raw = fig.add_subplot(2, 4, i + 2)
        ax_raw.scatter(H_raw[:, xi], H_raw[:, yi], s=5, c='red', alpha=0.8)
        ax_raw.set_title(f'Before - {title} Projection')
        ax_raw.set_xlabel(xlabel)
        ax_raw.set_ylabel(ylabel)
        ax_raw.grid(True)
        ax_raw.set_xlim(-limit, limit)
        ax_raw.set_ylim(-limit, limit)
        ax_raw.set_aspect('equal', adjustable='box')

        ax_comp = fig.add_subplot(2, 4, i + 6)
        ax_comp.scatter(H_comp[:, xi], H_comp[:, yi], s=5, c='blue', alpha=0.8)
        ax_comp.set_title(f'After - {title} Projection')
        ax_comp.set_xlabel(xlabel)
        ax_comp.set_ylabel(ylabel)
        ax_comp.grid(True)
        ax_comp.set_xlim(-limit, limit)
        ax_comp.set_ylim(-limit, limit)
        ax_comp.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_convergence(trace_history, label='Sum of diag(P)', title='RLSM Convergence'):
    plt.figure(figsize=(8, 4))
    plt.plot(trace_history)
    plt.xlabel('Iteration')
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.show()