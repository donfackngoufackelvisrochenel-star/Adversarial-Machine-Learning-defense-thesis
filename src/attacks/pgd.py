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
    # Work with DataFrame throughout to preserve feature names
    y_arr = y.reset_index(drop=True).values if isinstance(y, pd.Series) else np.asarray(y)
    cols = X.columns

    # Per-feature epsilon and step size (fraction of std-dev) as Series
    std_vals = X.std(axis=0).values
    eps = np.where(std_vals == 0, epsilon, epsilon * std_vals)
    step = np.where(std_vals == 0, alpha, alpha * std_vals)

    # Initialise adversarial examples as DataFrame
    X_adv = X.copy().astype(np.float64).values
    X_orig = X_adv.copy()

    if random_start:
        noise = np.random.uniform(-eps, eps, size=X_adv.shape)
        X_adv = X_orig + noise

    # Precompute importance weights
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        imp = imp / (imp.sum() + 1e-10)
    else:
        imp = np.ones(X_adv.shape[1]) / X_adv.shape[1]

    n = X_adv.shape[0]

    # Iterative perturbation
    for iteration in range(num_iter):
        # Convert numpy arrays back to DataFrame for sklearn predict
        X_adv_df = pd.DataFrame(X_adv, columns=cols, index=X.index)
        preds = model.predict(X_adv_df)
        probs = model.predict_proba(X_adv_df)

        # Vectorized gradient computation
        probs_masked = probs.copy()
        probs_masked[np.arange(n), y_arr] = 0
        second_best = np.argmax(probs_masked, axis=1)
        gap = probs[np.arange(n), second_best] - probs[np.arange(n), y_arr]
        gradient_sign = np.sign(gap)[:, np.newaxis] * imp[np.newaxis, :]

        # Zero perturbation for already-misclassified samples (already skipped)
        mis_mask = preds != y_arr
        gradient_sign[mis_mask] = 0

        # Vectorized step
        X_adv = X_adv + step[np.newaxis, :] * gradient_sign

        # Project the total perturbation back into the epsilon-ball
        delta = X_adv - X_orig
        delta_clipped = np.clip(delta, -eps, eps)
        X_adv = X_orig + delta_clipped

    # Convert back to DataFrame
    X_adv = pd.DataFrame(X_adv, columns=cols, index=X.index)

    # Clip to user-specified feature bounds
    if feature_bounds:
        for col_name, (lo, hi) in feature_bounds.items():
            if col_name in X_adv.columns:
                X_adv[col_name] = X_adv[col_name].clip(lo, hi)

    # Report attack success rate
    success = (model.predict(X_adv) != y_arr).mean()
    print(f"[pgd] epsilon={epsilon}, alpha={alpha}, iter={num_iter} | Attack success rate: {success:.4f}")
    return X_adv
