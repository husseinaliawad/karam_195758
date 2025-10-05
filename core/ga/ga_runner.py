from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .chromosome import random_mask, ensure_valid_mask
from .operators import tournament_selection, uniform_crossover, bit_flip_mutation
from .fitness import compute_fitness
from .utils import get_default_estimator


def run_ga(df: pd.DataFrame,
           target_col: str,
           task_type: str,
           scoring: str | None = None,
           estimator=None,
           cv: int = 5,
           population_size: int = 30,
           generations: int = 20,
           pc: float = 0.8,
           pm: float = 0.05,
           elitism: int = 2,
           lambda_penalty: float = 0.0,
           p_init: float = 0.5,
           random_state: int | None = None,
           max_seconds: float | None = None):
    rng = np.random.default_rng(random_state)
    start_time = time.perf_counter()
    feature_names = df.drop(columns=[target_col]).columns
    n_features = len(feature_names)

    if estimator is None:
        estimator = get_default_estimator(task_type)

    if scoring is None:
        scoring = 'accuracy' if task_type == 'classification' else 'r2'

    # Initialize population
    population = [random_mask(n_features, p_init, rng) for _ in range(population_size)]
    fitnesses = [compute_fitness(mask, df, target_col, task_type, estimator, scoring, cv, lambda_penalty, rng)
                 for mask in population]

    history = []
    for gen in range(generations):
        if max_seconds is not None and (time.perf_counter() - start_time) >= max_seconds:
            break
        # Elitism: carry over top-k individuals
        elite_indices = np.argsort(fitnesses)[-elitism:]
        new_population = [population[i].copy() for i in elite_indices]

        # Generate rest of the new population
        while len(new_population) < population_size:
            if max_seconds is not None and (time.perf_counter() - start_time) >= max_seconds:
                break
            p1 = tournament_selection(population, fitnesses, rng=rng)
            p2 = tournament_selection(population, fitnesses, rng=rng)
            child = uniform_crossover(p1, p2, pc, rng)
            child = bit_flip_mutation(child, pm, rng)
            child = ensure_valid_mask(child, rng)
            new_population.append(child)

        population = new_population
        fitnesses = [compute_fitness(mask, df, target_col, task_type, estimator, scoring, cv, lambda_penalty, rng)
                     for mask in population]

        best_idx = int(np.argmax(fitnesses))
        best_fit = float(fitnesses[best_idx])
        best_mask = population[best_idx]
        history.append({
            'generation': gen,
            'best_fitness': best_fit,
            'mean_fitness': float(np.mean(fitnesses)),
            'selected_count': int(best_mask.sum()),
        })

    # Final best
    best_idx = int(np.argmax(fitnesses))
    best_mask = population[best_idx]
    best_fit = float(fitnesses[best_idx])
    selected_features = list(feature_names[best_mask])

    return {
        'best_features': selected_features,
        'best_score': best_fit,
        'selected_mask': best_mask.astype(int).tolist(),
        'n_features_selected': int(best_mask.sum()),
        'history': history,
    }
