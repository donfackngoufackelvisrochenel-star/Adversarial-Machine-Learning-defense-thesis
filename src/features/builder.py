"""
Feature engineering and selection utilities.

Provides functions to rank features by importance using a Random Forest
probe and to reduce the feature set to only the top-K most important
(or least vulnerable) features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from configs.config import RANDOM_STATE


def select_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 20,
) -> list:
    """
    Use a Random Forest to score feature importance and return the top K.

    This is used by the feature-selection defense to identify which
    features carry the most predictive signal.

    Args:
        X_train: Training features.
        y_train: Training labels.
        top_k: Number of top features to retain.

    Returns:
        List of column names for the top-K most important features.
    """
    # Train a quick Random Forest as a probe
    selector = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
    )
    selector.fit(X_train, y_train)

    # Build a DataFrame of feature names and their importance scores
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": selector.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Keep only the top-K
    top_features = importances.head(top_k)["feature"].tolist()
    print(f"[features] Selected top {top_k} features: {top_features[:5]}...")
    return top_features


def reduce_to_features(X: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Subset a DataFrame to only the given feature columns.

    Silently drops any requested features that do not exist in X.

    Args:
        X: Input DataFrame.
        features: List of column names to keep.

    Returns:
        DataFrame with only the specified columns (in order).
    """
    missing = [f for f in features if f not in X.columns]
    if missing:
        print(f"[features] Warning: missing features: {missing}")
    present = [f for f in features if f in X.columns]
    return X[present]
