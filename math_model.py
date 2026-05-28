import numpy as np

def apply_error_model(H_ideal, delta_H, delta_K):
    H = H_ideal * (1 + delta_K) + delta_H
    return H


def compensate(H, delta_H, delta_K):
    H_comp = (H - delta_H) / (1 + delta_K)
    return H_comp


def RLSM(data, eps=0.01):

    X = np.array([0, 1, 0, 1, 0, -1], dtype=float)
    P = np.diag([1, 9, 9, 9, 9, 20]).astype(float)
    trace_history = []

    for row in data:
        H1, H2, H3 = row[0], row[1], row[2]

        Z = -H1**2
        h = np.array([-2*H1, H2**2, -2*H2, H3**2, -2*H3, 1])

        e = Z - h @ X
        X = X + (P @ h / (1 + h @ P @ h)) * e
        P = P - (P @ h.reshape(-1, 1) @ h.reshape(1, -1) @ P) / (1 + h @ P @ h)

        trace_history.append(np.sum(np.diag(P)))

        if np.sum(np.diag(P)) < eps:
            break

    C1, C2, C3, C4, C5, C6 = X
    dH1 = C1
    dH2 = C3 / C2
    dH3 = C5 / C4
    dK1 = 0.0
    dK2 = 1 / np.sqrt(abs(C2)) - 1
    dK3 = 1 / np.sqrt(abs(C4)) - 1

    return np.array([dH1, dH2, dH3]), np.array([dK1, dK2, dK3]), trace_history


def _fitness(params, data):
    dH = params[:3]
    dK = params[3:]
    H_comp = (data - dH) / (1 + dK)
    r_ref = np.mean(np.linalg.norm(data, axis=1))   
    return np.mean((np.linalg.norm(H_comp, axis=1) - r_ref) ** 2)


def GA(data, pop_size=60, n_gen=300, eps=1e-5, alpha=0.5, p_mut=0.15, sigma0=0.05, seed=42):
    rng = np.random.default_rng(seed)

    H_max = np.max(np.abs(data))
    lo = np.array([-H_max, -H_max, -H_max, -0.5, -0.5, -0.5])
    hi = np.array([ H_max,  H_max,  H_max,  0.5,  0.5,  0.5])
    rng_range = hi - lo

    pop = rng.uniform(lo, hi, size=(pop_size, 6))
    fitness_history = []

    for gen in range(n_gen):
        sigma = sigma0 * (0.01 ** (gen / n_gen))
        fits = np.array([_fitness(ind, data) for ind in pop])
        fitness_history.append(np.min(fits))

        if np.min(fits) < eps:
            break

        order = np.argsort(fits)
        new_pop = pop[order[:2]].copy()

        while len(new_pop) < pop_size:
            def tournament():
                idx = rng.choice(pop_size, size=3, replace=False)
                return pop[idx[np.argmin(fits[idx])]]

            A = tournament()
            B = tournament()

            u = rng.uniform(-alpha, 1 + alpha, size=6)
            child = A + u * (B - A)
            child = np.clip(child, lo, hi)

            mask = rng.random(6) < p_mut
            child[mask] += sigma * rng_range[mask] * rng.standard_normal(mask.sum())
            child = np.clip(child, lo, hi)

            new_pop = np.vstack([new_pop, child])

        pop = new_pop[:pop_size]

    fits = np.array([_fitness(ind, data) for ind in pop])
    best = pop[np.argmin(fits)]

    return best[:3], best[3:], fitness_history