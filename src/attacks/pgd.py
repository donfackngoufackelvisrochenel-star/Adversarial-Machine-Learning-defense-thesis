"""
Projected Gradient Descent (PGD) — Iterative Evasion Attack for Tabular Data.

PGD extends FGSM by taking multiple small steps and projecting the
perturbation back onto an epsilon-ball around the original sample
after each step. This produces stronger adversarial examples at the
cost of additional computation.

Reference: Madry et al., "Towards Deep Learning Models Resistant to
Adversarial Attacks" (ICLR 2018).
"""

import numpy as np
import pandas as pd


def pgd_attack(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    num_iter: int = 10,
    feature_bounds: dict = None,
    random_start: bool = True,
) -> pd.DataFrame:
    """
    Generate adversarial examples using PGD.

    Algorithm:
    1. (Optional) Initialise each adversarial sample with a small random
       perturbation inside the epsilon-ball.
    2. For each iteration:
       a. Compute the perturbation direction (same gradient approximation
          as FGSM).
       b. Take a step of size `alpha` in that direction.
       c. Project the total perturbation back into the epsilon-ball
          (element-wise clipping).
       d. Clip to feature bounds if provided.
    3. Return the final perturbed samples.

    Args:
        model: Trained classifier.
        X: Clean input features.
        y: True labels.
        epsilon: Maximum allowed perturbation per feature (fraction of std).
        alpha: Step size per iteration.
        num_iter: Number of PGD iterations.
        feature_bounds: Optional {col: (min, max)} for clipping.
        random_start: If True, initialise with random noise inside the ball.

    Returns:
        DataFrame of adversarial examples.
    """
    # Work with float64 NumPy arrays for performance
    X_orig = X.values.astype(np.float64)
    y_arr = y.reset_index(drop=True).values if isinstance(y, pd.Series) else np.asarray(y)

    # Per-feature epsilon and step size (fraction of std-dev)
    eps = epsilon * np.std(X_orig, axis=0)
    eps = np.where(eps == 0, epsilon, eps)
    step = alpha * np.std(X_orig, axis=0)
    step = np.where(step == 0, alpha, step)

    # Random initialisation inside the epsilon-ball
    if random_start:
        X_adv = X_orig + np.random.uniform(-eps, eps, size=X_orig.shape)
    else:
        X_adv = X_orig.copy()

    # Iterative perturbation
    for iteration in range(num_iter):
        preds = np.asarray(model.predict(X_adv))

        # Approximate gradient using feature importances
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp = imp / (imp.sum() + 1e-10)
        else:
            imp = np.ones(X_adv.shape[1]) / X_adv.shape[1]

        for i in range(len(X_adv)):
            # Skip samples that are already misclassified
            if preds[i] != y_arr[i]:
                continue

            # Random perturbation direction weighted by feature importance
            gradient_sign = np.sign(np.random.randn(X_adv.shape[1]))
            perturbation = step * gradient_sign * imp
            X_adv[i] += perturbation

        # Project the total perturbation back into the epsilon-ball
        delta = X_adv - X_orig
        delta = np.clip(delta, -eps, eps)
        X_adv = X_orig + delta

        # Clip to user-specified feature bounds
        if feature_bounds:
            for col_name, (lo, hi) in feature_bounds.items():
                col_idx = X.columns.get_loc(col_name)
                X_adv[:, col_idx] = np.clip(X_adv[:, col_idx], lo, hi)

    # Convert back to DataFrame
    X_adv = pd.DataFrame(X_adv, columns=X.columns)

    # Report attack success rate
    success = (np.asarray(model.predict(X_adv)) != y_arr).mean()
    print(f"[pgd] epsilon={epsilon}, alpha={alpha}, iter={num_iter} | Attack success rate: {success:.4f}")
    return X_adv
