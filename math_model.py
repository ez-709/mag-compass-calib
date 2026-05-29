import numpy as np

def RLSM(data, eps=0.1):
    X = np.array([0, 1, 0, 1, 0, -1], dtype=float)
    P = np.diag([0.5, 1.5, 1.5, 1.5, 1.5, 20.0])
    trace_history = []
    h_history = []
    K_history = []

    for row in data:
        H1, H2, H3 = row[0], row[1], row[2]
        Z = -H1**2
        h = np.array([-2*H1, H2**2, -2*H2, H3**2, -2*H3, 1])
        denom = 1 + h @ P @ h
        K = P @ h / denom
        e = Z - h @ X
        X = X + K * e
        P = P - np.outer(K, h @ P)

        trace_history.append(np.sum(np.diag(P)))
        h_history.append(h.copy())
        K_history.append(K.copy())

        if np.sum(np.diag(P)) < eps:
            break

    C1, C2, C3, C4, C5, C6 = X
    dH1 = C1
    dH2 = C3 / C2
    dH3 = C5 / C4
    val = C1**2 + C3**2 / abs(C2) + C5**2 / abs(C4) - C6
    dK1 = np.sqrt(max(val, 0.0)) - 1
    dK2 = (1 + dK1) / np.sqrt(abs(C2)) - 1
    dK3 = (1 + dK1) / np.sqrt(abs(C4)) - 1

    return (np.array([dH1, dH2, dH3]),
            np.array([dK1, dK2, dK3]),
            trace_history,
            np.array(h_history),
            np.array(K_history))

def fitness(params, data):
    dH = params[:3]
    dK = params[3:]
    H_comp = (data - dH) / (1 + dK)
    r_ref = np.mean(np.linalg.norm(data, axis=1))
    return np.mean((np.linalg.norm(H_comp, axis=1) - r_ref) ** 2)


def apply_error_model(H_ideal, delta_H, delta_K):
    H = H_ideal * (1 + delta_K) + delta_H
    return H

def compensate(H, delta_H, delta_K):
    H_comp = (H - delta_H) / (1 + delta_K)
    return H_comp

def tournament(pop, fits, rng, pop_size):
    idx = [rng.integers(0, pop_size) for _ in range(3)]
    best = idx[0]
    for i in idx[1:]:
        if fits[i] < fits[best]:
            best = i
    return pop[best]

def crossover(A, B, rng):
    child = np.zeros(6)
    for g in range(6):
        if rng.random() < 0.5:
            child[g] = A[g]
        else:
            child[g] = B[g]
    return child

def mutate(child, sigma, rng_range, lo, hi, p_mut, rng):
    for g in range(6):
        if rng.random() < p_mut:
            child[g] += sigma * rng_range[g] * rng.standard_normal()
            child[g] = np.clip(child[g], lo[g], hi[g])
    return child

def GA(data, pop_size=60, n_gen=300, p_mut=0.15, sigma0=0.05, seed=42):
    rng = np.random.default_rng(seed)
    H_max = np.max(np.abs(data))
    lo = np.array([-H_max, -H_max, -H_max, -1, -1, -1])
    hi = np.array([ H_max,  H_max,  H_max,  1,  1,  1])
    rng_range = hi - lo

    pop = []
    for _ in range(pop_size):
        ind = np.array([rng.uniform(lo[g], hi[g]) for g in range(6)])
        pop.append(ind)

    fitness_history = []

    for gen in range(n_gen):
        sigma = sigma0 * (0.01 ** (gen / n_gen))
        fits = [fitness(ind, data) for ind in pop]
        fitness_history.append(min(fits))

        order = sorted(range(pop_size), key=lambda i: fits[i])
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]

        while len(new_pop) < pop_size:
            A = tournament(pop, fits, rng, pop_size)
            B = tournament(pop, fits, rng, pop_size)
            child = crossover(A, B, rng)
            child = mutate(child, sigma, rng_range, lo, hi, p_mut, rng)
            new_pop.append(child)

        pop = new_pop

    fits = [fitness(ind, data) for ind in pop]
    best = pop[fits.index(min(fits))]
    return best[:3], best[3:], fitness_history