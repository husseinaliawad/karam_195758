import numpy as np


def random_mask(n_features: int, p_init: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mask = rng.random(n_features) < p_init
    if not mask.any():
        idx = rng.integers(0, n_features)
        mask[idx] = True
    return mask


def ensure_valid_mask(mask: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if not np.any(mask):
        idx = rng.integers(0, len(mask))
        mask[idx] = True
    return mask
