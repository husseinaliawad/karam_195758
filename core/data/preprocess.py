from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


def detect_column_types(X: pd.DataFrame):
    # Heuristic: numeric dtypes treated as numeric; others as categorical
    numeric_cols = X.select_dtypes(include=["number"]) .columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame, est=None) -> ColumnTransformer:
    numeric_cols, categorical_cols = detect_column_types(X)

    numeric_pipeline = (
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True, with_std=True),
    )
    # Wrap into a pipeline-like tuple for ColumnTransformer usage
    # sklearn allows a tuple (name, transformer, columns) where transformer can be
    # a Pipeline or a single transformer. We'll build Pipelines explicitly.
    from sklearn.pipeline import Pipeline

    num_pipe = Pipeline([
        ("imputer", numeric_pipeline[0]),
        ("vt", VarianceThreshold(threshold=0.0)),  # drop constant columns
        ("scaler", numeric_pipeline[1]),
    ])

    # Choose encoding strategy based on estimator type (trees prefer ordinal to avoid huge one-hot)
    enc_is_onehot = True
    if est is not None:
        name = est.__class__.__name__.lower()
        if any(k in name for k in ("forest", "tree", "boost", "xgb", "lgbm", "catboost")):
            enc_is_onehot = False

    if enc_is_onehot:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ])
    else:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
