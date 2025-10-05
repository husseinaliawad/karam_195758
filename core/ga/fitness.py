from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.base import clone
from ..data.preprocess import build_preprocessor


def _normalize_scoring(task_type: str, scoring: str, y: pd.Series) -> str:
    s = (scoring or '').lower()
    classification_metrics = {
        'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
    }
    regression_metrics = {
        'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'
    }
    if task_type == 'classification':
        if s in regression_metrics or s == '':
            s = 'accuracy'
        # handle multiclass variants
        try:
            n_classes = int(y.nunique(dropna=False))
        except Exception:
            n_classes = 0
        if s == 'f1' and n_classes > 2:
            s = 'f1_weighted'
        if s == 'roc_auc' and n_classes > 2:
            s = 'roc_auc_ovr'
    else:
        if s in classification_metrics or s == '':
            s = 'r2'
    return s


def compute_fitness(mask: np.ndarray,
                    df: pd.DataFrame,
                    target_col: str,
                    task_type: str,
                    estimator,
                    scoring: str,
                    cv: int | object,
                    lambda_penalty: float = 0.0,
                    rng=None) -> float:
    # Select features
    feature_df = df.drop(columns=[target_col])
    n_total = feature_df.shape[1]
    selected_cols = feature_df.columns[mask]
    if len(selected_cols) == 0:
        return float('-inf')

    # Drop rows with missing target
    y_all = df[target_col]
    valid_rows = y_all.notna()
    if not np.any(valid_rows):
        return float('-inf')

    X = df.loc[valid_rows, selected_cols]
    y = y_all.loc[valid_rows]
    if len(y) < 3:
        # Not enough data to perform CV
        return float('-inf')

    # Build preprocessing + estimator pipeline
    est = clone(estimator)
    preprocessor = build_preprocessor(X, est)
    pipeline = Pipeline(steps=[('prep', preprocessor), ('est', est)])

    # Build a safe CV object
    if isinstance(cv, int):
        cv_splits = max(2, int(cv))
    else:
        # default fallback
        cv_splits = 5

    if task_type == 'classification':
        vc = y.value_counts(dropna=False)
        if len(vc) < 2:
            # Not a valid classification target
            return float('-inf')
        min_class_count = int(vc.min())
        n_splits = min(cv_splits, min_class_count, len(y))
        if n_splits < 2:
            return float('-inf')
        cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = min(cv_splits, len(y))
        if n_splits < 2:
            return float('-inf')
        cv_obj = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Normalize scoring to avoid incompatible choices (e.g., f1 -> f1_weighted for multiclass)
    scoring_norm = _normalize_scoring(task_type, scoring, y)

    try:
        scores = cross_val_score(pipeline, X, y, cv=cv_obj, scoring=scoring_norm)
        # Guard against NaN/inf scores
        if not np.all(np.isfinite(scores)):
            return float('-inf')
        mean_score = float(np.mean(scores))
    except Exception:
        return float('-inf')

    penalty = lambda_penalty * (len(selected_cols) / float(n_total))
    return mean_score - penalty
