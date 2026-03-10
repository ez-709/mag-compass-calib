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
        P = P - (P @ h.reshape(-1,1) @ h.reshape(1,-1) @ P) / (1 + h @ P @ h)
        
        trace_history.append(np.sum(np.diag(P)))
        
        if np.sum(np.diag(P)) < eps:
            break
    
    C1, C2, C3, C4, C5, C6 = X
    dH1 = C1
    dH2 = C3 / C2
    dH3 = C5 / C4
    dK2 = 1/np.sqrt(C2) - 1
    dK3 = 1/np.sqrt(C4) - 1
    
    return np.array([dH1, dH2, dH3]), np.array([0, dK2, dK3]), trace_history