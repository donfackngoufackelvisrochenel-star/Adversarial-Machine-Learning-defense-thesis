"""
Model training and persistence.

Provides a registry of supported classifiers (Random Forest, XGBoost,
LightGBM), a generic training function, and utilities for saving and
loading models to/from disk.
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from configs.config import MODEL_PARAMS, RANDOM_STATE, MODELS_DIR


# ---------------------------------------------------------------------------
# Model registry — maps model names to their scikit-learn-compatible classes
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
) -> object:
    """
    Train a single model by name.

    Looks up the model class in MODEL_REGISTRY, instantiates it with
    the provided (or default) parameters, fits it, and prints performance
    on training and validation sets.

    Args:
        model_name: One of 'random_forest', 'xgboost', 'lightgbm'.
        X_train, y_train: Training data.
        X_val, y_val: Optional validation data (used for early stopping in XGBoost).
        params: Optional dict of hyperparameters. Falls back to config defaults.

    Returns:
        Fitted model object.
    """
    # Validate model name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}")

    # Get model class and parameters
    model_cls = MODEL_REGISTRY[model_name]
    if params is None:
        params = MODEL_PARAMS.get(model_name, {})

    model = model_cls(**params)

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    # Print training accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"[model] {model_name} | Train acc: {train_acc:.4f} | Time: {elapsed:.1f}s")

    # Print validation metrics if available
    if X_val is not None:
        start = time.time()
        val_pred = model.predict(X_val)
        elapsed_val = time.time() - start
        val_acc = accuracy_score(y_val, val_pred)
        val_prec = precision_score(y_val, val_pred, average="weighted", zero_division=0)
        val_rec = recall_score(y_val, val_pred, average="weighted", zero_division=0)
        val_f1 = f1_score(y_val, val_pred, average="weighted")
        print(f"[model] {model_name} | Val acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | Val time: {elapsed_val:.1f}s")

    return model


def save_model(model: object, model_name: str, tag: str = "") -> Path:
    """
    Serialise a trained model to the models/ directory using joblib.

    Args:
        model: Trained model object.
        model_name: Base name (e.g., 'random_forest').
        tag: Optional suffix (e.g., 'clean', 'adv_trained').

    Returns:
        Path to the saved .pkl file.
    """
    suffix = f"_{tag}" if tag else ""
    path = MODELS_DIR / f"{model_name}{suffix}.pkl"
    joblib.dump(model, path)
    print(f"[model] Saved: {path}")
    return path


def load_model(path: Path) -> object:
    """Load a serialised model from disk."""
    return joblib.load(path)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
) -> dict:
    """
    Train every model in the registry and save each to disk.

    Each model is trained independently so a single failure (e.g. a
    missing system library for LightGBM) does not prevent the others
    from being available in the dashboard.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Optional validation data.

    Returns:
        Dict mapping model names to trained model objects:
        {'random_forest': <RF>, 'xgboost': <XGB>, 'lightgbm': <LGBM>}
    """
    models = {}
    for name in MODEL_REGISTRY:
        print(f"\n{'='*40}\nTraining {name}...\n{'='*40}")
        try:
            model = train_model(name, X_train, y_train, X_val, y_val)
            models[name] = model
            save_model(model, name, "clean")
        except Exception as exc:
            print(f"[trainer] {name} FAILED: {exc}")
    return models
