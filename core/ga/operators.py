import numpy as np


def tournament_selection(population: list[np.ndarray], fitnesses: list[float], tournament_size: int = 3,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    idxs = rng.integers(0, len(population), size=tournament_size)
    best_idx = idxs[0]
    best_fit = fitnesses[best_idx]
    for i in idxs[1:]:
        if fitnesses[i] > best_fit:
            best_idx = i
            best_fit = fitnesses[i]
    return population[best_idx].copy()


def uniform_crossover(p1: np.ndarray, p2: np.ndarray, pc: float = 0.8,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if rng.random() > pc:
        return p1.copy()
    mask = rng.random(p1.shape[0]) < 0.5
    child = np.where(mask, p1, p2)
    if not child.any():
        idx = rng.integers(0, len(child))
        child[idx] = True
    return child


def bit_flip_mutation(mask: np.ndarray, pm: float = 0.01,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    flips = rng.random(mask.shape[0]) < pm
    child = mask.copy()
    child[flips] = ~child[flips]
    if not child.any():
        idx = rng.integers(0, len(child))
        child[idx] = True
    return child
