from __future__ import annotations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_cv(task_type: str, cv: int | KFold | StratifiedKFold):
    if isinstance(cv, int):
        if task_type == 'classification':
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            return KFold(n_splits=cv, shuffle=True, random_state=42)
    return cv


def get_default_estimator(task_type: str):
    if task_type == 'classification':
        # More robust for multiclass and mixed feature types
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
