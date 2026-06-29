"""
Fast Gradient Sign Method (FGSM) — Evasion Attack for Tabular Data.

FGSM is a single-step attack that perturbs each feature in the direction
that maximally increases the loss. For tree-based models (RF, XGBoost,
LightGBM), true gradients are not available, so we approximate the
perturbation direction using feature importances and random signs
weighted by the model's prediction confidence.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial
Examples" (ICLR 2015).
"""

import numpy as np
import pandas as pd


def fgsm_attack(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    epsilon: float = 0.1,
    feature_bounds: dict = None,
) -> pd.DataFrame:
    """
    Generate adversarial examples using FGSM.

    The attack works by:
    1. Computing the per-feature perturbation magnitude as
       epsilon * std(feature).
    2. Determining the perturbation direction using feature importances
       (for tree models) and the model's prediction confidence.
    3. Adding the perturbation to each sample.

    Args:
        model: A trained classifier. Must implement predict() and
               predict_proba().
        X: Clean input features.
        y: True labels.
        epsilon: Perturbation magnitude as a fraction of feature standard
                 deviation. Higher values produce more aggressive attacks.
        feature_bounds: Optional dict mapping column names to (min, max)
                        tuples. Perturbed values are clipped to these bounds.

    Returns:
        DataFrame of adversarial examples with the same shape as X.
    """
    # Use DataFrame throughout to preserve feature names (required by sklearn)
    X_adv = X.copy().astype(np.float64)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    # Compute per-feature epsilon = fraction of the feature's standard deviation
    eps = epsilon * X_adv.std(axis=0).values
    # Avoid division by zero for constant features
    eps = np.where(eps == 0, epsilon, eps)

    # Get model predictions and probabilities
    probs = model.predict_proba(X_adv)
    preds = model.predict(X_adv)

    # Use feature importances as a proxy for the gradient direction.
    # Tree models do not have smooth gradients, so we approximate.
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        imp = imp / (imp.sum() + 1e-10)
    else:
        imp = np.ones(X_adv.shape[1]) / X_adv.shape[1]

    # Vectorized gradient computation — no Python row loop
    n = len(X_adv)
    # Zero out the true-class probability to find the second-best class
    probs_masked = probs.copy()
    probs_masked[np.arange(n), y_arr] = 0
    second_best = np.argmax(probs_masked, axis=1)
    # Gap between second-best and true class probability
    gap = probs[np.arange(n), second_best] - probs[np.arange(n), y_arr]
    gradient_sign = np.sign(gap)[:, np.newaxis] * imp[np.newaxis, :]

    # Random direction for already-misclassified samples
    mis_mask = preds != y_arr
    n_mis = mis_mask.sum()
    if n_mis > 0:
        gradient_sign[mis_mask] = np.sign(np.random.randn(n_mis, X_adv.shape[1])) * imp[np.newaxis, :]

    # Apply perturbation in one vectorized operation
    X_adv = X_adv + eps[np.newaxis, :] * gradient_sign

    # Clip perturbed values to the specified bounds if provided
    if feature_bounds:
        for col, (lo, hi) in feature_bounds.items():
            if col in X_adv.columns:
                X_adv[col] = X_adv[col].clip(lo, hi)

    # Report attack success rate (fraction of samples where the prediction changed)
    success = (model.predict(X_adv) != y).mean()
    print(f"[fgsm] epsilon={epsilon} | Attack success rate: {success:.4f}")
    return X_adv
