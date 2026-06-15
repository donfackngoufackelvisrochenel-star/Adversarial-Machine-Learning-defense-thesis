"""
Evaluation metrics for classification and adversarial robustness.

Provides standard classification metrics (accuracy, precision, recall,
F1, ROC-AUC) plus attack-specific metrics like the attack success rate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None, prefix=""):
    """
    Compute a standard set of classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted class probabilities (optional, used for ROC-AUC).
        prefix: String to prepend to metric names (for disambiguation
                when comparing multiple models).

    Returns:
        Dict of metric_name → value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # ROC-AUC is only defined for binary classification
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            metrics["roc_auc"] = 0.0
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def attack_success_rate(y_true, y_pred_adv):
    """
    Compute the Attack Success Rate (ASR).

    ASR = fraction of samples where the model's prediction on the
    adversarial example differs from the true label.

    A higher ASR means the attack is more effective.
    """
    return (y_pred_adv != y_true).mean()


def robustness_curve(model, X_test, y_test, attack_fn, epsilons):
    """
    Compute model accuracy across a range of attack strengths.

    For each epsilon value, generates adversarial examples with the
    given attack function and records the resulting accuracy. This
    produces a robustness curve that can be plotted to compare
    different models / defenses.

    Args:
        model: Classifier to evaluate.
        X_test, y_test: Test data.
        attack_fn: Function with signature (model, X, y, epsilon) -> X_adv.
        epsilons: List of epsilon values to test.

    Returns:
        DataFrame with columns ['epsilon', 'accuracy'].
    """
    results = []
    base_acc = accuracy_score(y_test, model.predict(X_test))
    results.append({"epsilon": 0, "accuracy": base_acc})

    for eps in epsilons:
        X_adv = attack_fn(model, X_test, y_test, epsilon=eps)
        acc = accuracy_score(y_test, model.predict(X_adv))
        results.append({"epsilon": eps, "accuracy": acc})
        print(f"[robustness] epsilon={eps:.3f} | accuracy={acc:.4f}")

    return pd.DataFrame(results)


def summarize_defense_comparison(results: dict) -> pd.DataFrame:
    """
    Convert a dict of {defense_name: metrics_dict} into a summary DataFrame.

    Useful for printing a side-by-side comparison of multiple defenses.

    Args:
        results: e.g. {'None': {'accuracy': 0.99, ...},
                       'squeeze': {'accuracy': 0.97, ...}}

    Returns:
        DataFrame with one row per defense.
    """
    rows = []
    for defense_name, metrics in results.items():
        row = {"defense": defense_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
