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
    # Convert to float64 NumPy array for safe in-place perturbation
    X_adv = X.copy().values.astype(np.float64)
    y = y.reset_index(drop=True) if isinstance(y, pd.Series) else y

    # Compute per-feature epsilon = fraction of the feature's standard deviation
    eps = epsilon * np.std(X_adv, axis=0)
    # Avoid division by zero for constant features
    eps = np.where(eps == 0, epsilon, eps)

    # Get model predictions and probabilities
    probs = model.predict_proba(X_adv)
    preds = model.predict(X_adv)

    # Use feature importances as a proxy for the gradient direction.
    # Tree models do not have smooth gradients, so we approximate.
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        # Normalise so that importance weights sum to 1
        imp = imp / (imp.sum() + 1e-10)
    else:
        # Equal weighting if no importances available
        imp = np.ones(X_adv.shape[1]) / X_adv.shape[1]

    # Perturb each sample individually
    for i in range(len(X_adv)):
        pred_class = preds[i]
        true_class = y[i]

        # If the model already misclassifies this sample, random perturbation
        if pred_class == true_class:
            gradient_sign = np.sign(np.random.randn(X_adv.shape[1]))
        else:
            # Push the sample further in the wrong direction
            gradient_sign = -np.sign(probs[i, pred_class] - probs[i, true_class]) * imp

        # Apply perturbation
        perturbation = eps * gradient_sign
        X_adv[i] += perturbation

    # Convert back to DataFrame with original column names
    X_adv = pd.DataFrame(X_adv, columns=X.columns)

    # Clip perturbed values to the specified bounds if provided
    if feature_bounds:
        for col, (lo, hi) in feature_bounds.items():
            if col in X_adv.columns:
                X_adv[col] = X_adv[col].clip(lo, hi)

    # Report attack success rate (fraction of samples where the prediction changed)
    success = (model.predict(X_adv) != y).mean()
    print(f"[fgsm] epsilon={epsilon} | Attack success rate: {success:.4f}")
    return X_adv
