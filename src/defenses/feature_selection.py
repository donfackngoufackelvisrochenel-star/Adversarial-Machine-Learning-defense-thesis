"""
Feature Selection Defense.

Strategy: Identify which features are most *vulnerable* to adversarial
perturbation (i.e., whose noise causes the largest drop in accuracy)
and remove them. Training on only the least vulnerable features reduces
the attack surface available to an adversary.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.trainer import train_model, save_model
from configs.config import RANDOM_STATE, DEFENSE_PARAMS


def _rank_feature_vulnerability(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_perturbations: int = 5,
) -> pd.DataFrame:
    """
    Rank features by how sensitive model accuracy is to their perturbation.

    For each feature:
      1. Inject Gaussian noise (scale = 10% of feature std-dev).
      2. Measure the drop in accuracy compared to the clean baseline.
      3. Repeat `n_perturbations` times and average the accuracy loss.

    A higher average loss means the feature is more vulnerable.

    Args:
        model: A trained classifier.
        X_train, y_train: Training data.
        n_perturbations: Number of noise realisations per feature.

    Returns:
        DataFrame with columns ['feature', 'vulnerability'] sorted
        from most to least vulnerable.
    """
    base_acc = model.score(X_train, y_train)
    sensitivities = []

    for col in X_train.columns:
        acc_losses = []
        # Average over multiple random noise seeds
        for _ in range(n_perturbations):
            X_pert = X_train.copy()
            noise = np.random.normal(0, 0.1 * X_train[col].std(), size=X_train.shape[0])
            X_pert[col] += noise
            pert_acc = model.score(X_pert, y_train)
            acc_losses.append(base_acc - pert_acc)
        sensitivities.append(np.mean(acc_losses))

    vuln = pd.DataFrame({"feature": X_train.columns, "vulnerability": sensitivities})
    vuln = vuln.sort_values("vulnerability", ascending=False)
    return vuln


def feature_selection_defense(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    top_k: int = None,
) -> (object, list):
    """
    Train a model on only the K least vulnerable features.

    Steps:
      1. Train a quick Random Forest probe on the full feature set.
      2. Rank features by vulnerability (accuracy loss under noise).
      3. Keep only the `top_k` least vulnerable features.
      4. Train the target model on the reduced feature set.

    Args:
        model_name: Model to train (e.g., 'xgboost').
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        top_k: Number of features to retain.

    Returns:
        (trained model, list of kept feature names)
    """
    # Get default top_k from config if not specified
    params = DEFENSE_PARAMS["feature_selection"]
    top_k = top_k or params["top_k"]

    print("[feature_selection] Ranking feature vulnerability...")
    # Use a cheap Random Forest as the probe model (capped depth for speed)
    probe = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    probe.fit(X_train, y_train)

    # Rank features on a 10% subsample (30k rows at 300k) — ranking is stable
    # with fewer rows since we only need relative ordering
    sample_size = min(30000, len(X_train))
    if sample_size < len(X_train):
        X_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
        y_sample = y_train.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        X_sample = X_train
        y_sample = y_train
    vuln = _rank_feature_vulnerability(probe, X_sample, y_sample)

    # Keep the *least* vulnerable features (tail of the sorted list)
    keep_features = vuln.tail(top_k)["feature"].tolist()
    dropped = vuln.head(len(X_train.columns) - top_k)["feature"].tolist()
    print(f"[feature_selection] Keeping {len(keep_features)} features, dropped {len(dropped)}")

    # Train the target model on the reduced feature set
    X_train_red = X_train[keep_features]
    X_val_red = X_val[keep_features] if X_val is not None else None

    model = train_model(model_name, X_train_red, y_train, X_val_red, y_val)
    save_model(model, model_name, "feat_select")
    print(f"[feature_selection] Model saved as '{model_name}_feat_select.pkl'")
    return model, keep_features
