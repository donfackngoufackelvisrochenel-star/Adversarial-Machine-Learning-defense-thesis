"""
Defense comparison and plotting.

Provides functions to compare clean vs. defended models across
multiple attack strengths, plot robustness curves, and export
summary tables to CSV.
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score
from configs.config import REPORTS_DIR


# ---------------------------------------------------------------------------
# Feature-alignment helpers
# ---------------------------------------------------------------------------
# Models trained with feature selection may expect fewer columns than X_test
# has. These helpers detect the mismatch and align the data automatically.


def _safe_predict(model, X: pd.DataFrame):
    """
    Predict with automatic feature alignment.

    If the model expects a different set of features than those present
    in X, this function pads missing columns with zeros and drops extra
    columns so that the prediction succeeds.
    """
    try:
        return model.predict(X)
    except (ValueError, KeyError) as e:
        msg = str(e).lower()
        if "feature_names" in msg or "feature" in msg:
            # Extract the list of features the model expects
            model_features = getattr(model, "feature_names_in_", None)
            if model_features is None:
                try:
                    model_features = model.get_booster().feature_names
                except Exception:
                    pass
            if model_features is not None:
                available = [f for f in model_features if f in X.columns]
                missing = [f for f in model_features if f not in X.columns]
                if missing:
                    print(f"[comparison] Adding {len(missing)} missing features with 0s")
                    for f in missing:
                        X[f] = 0
                return model.predict(X[list(model_features)])
        raise


def _safe_attack(attack_fn, model, X, y, epsilon):
    """
    Run an evasion attack with automatic feature alignment.

    Some attacks may fail if the model expects different features.
    This function aligns X to the model's expected feature set
    before running the attack.
    """
    try:
        return attack_fn(model, X, y, epsilon=epsilon)
    except Exception:
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            try:
                model_features = model.get_booster().feature_names
            except Exception:
                raise
        available = [f for f in model_features if f in X.columns]
        missing = [f for f in model_features if f not in X.columns]
        X_aligned = X[available].copy()
        for f in missing:
            X_aligned[f] = 0
        return attack_fn(model, X_aligned, y, epsilon=epsilon)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_attack_defense(
    clean_models: dict,
    defended_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    attack_fn,
    attack_name: str,
    epsilons: list = None,
) -> pd.DataFrame:
    """
    Compare the robustness of clean and defended models.

    For each epsilon:
      1. Generate adversarial examples using attack_fn.
      2. Evaluate every clean and defended model on those examples.
      3. Record the accuracy.

    Args:
        clean_models: Dict of {name: model} for undefended models.
        defended_models: Dict of {name: model} for defended models.
        X_test, y_test: Test data.
        attack_fn: Function (model, X, y, epsilon) -> X_adv.
        attack_name: Label for the attack (used in plots / filenames).
        epsilons: List of perturbation magnitudes to test.

    Returns:
        DataFrame with columns [model, defense, epsilon, accuracy].
    """
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

    rows = []

    # Evaluate clean (undefended) models
    for model_name, model in clean_models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_acc = accuracy_score(y_test, _safe_predict(model, X_test))
            for eps in epsilons:
                if eps == 0:
                    rows.append({"model": model_name, "defense": "None", "epsilon": 0, "accuracy": base_acc})
                else:
                    X_adv = _safe_attack(attack_fn, model, X_test, y_test, eps)
                    acc = accuracy_score(y_test, _safe_predict(model, X_adv))
                    rows.append({"model": model_name, "defense": "None", "epsilon": eps, "accuracy": acc})

    # Evaluate defended models
    for def_name, def_model in defended_models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for eps in epsilons:
                if eps == 0:
                    acc = accuracy_score(y_test, _safe_predict(def_model, X_test))
                    rows.append({"model": def_name, "defense": def_name, "epsilon": 0, "accuracy": acc})
                else:
                    X_adv = _safe_attack(attack_fn, def_model, X_test, y_test, eps)
                    acc = accuracy_score(y_test, _safe_predict(def_model, X_adv))
                    rows.append({"model": def_name, "defense": def_name, "epsilon": eps, "accuracy": acc})

    return pd.DataFrame(rows)


def plot_robustness_comparison(df: pd.DataFrame, attack_name: str, save: bool = True):
    """
    Plot accuracy vs. epsilon for each defense.

    Each line in the plot represents one defense strategy, showing how
    its accuracy degrades as the attack strength increases.

    Args:
        df: DataFrame from compare_attack_defense().
        attack_name: Label for the plot title and filename.
        save: If True, saves the plot to reports/robustness_{attack_name}.png.
    """
    plt.figure(figsize=(10, 6))
    for defense in df["defense"].unique():
        subset = df[df["defense"] == defense]
        plt.plot(
            subset["epsilon"],
            subset["accuracy"],
            marker="o",
            label=defense,
        )

    plt.xlabel("Epsilon (attack strength)")
    plt.ylabel("Accuracy")
    plt.title(f"Robustness Comparison: {attack_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = REPORTS_DIR / f"robustness_{attack_name}.png"
        plt.savefig(path, dpi=150)
        print(f"[comparison] Plot saved to {path}")
    plt.close()


def generate_report_table(df: pd.DataFrame, attack_name: str) -> pd.DataFrame:
    """
    Create a pivot table showing the best accuracy per defense per epsilon.

    Exports to reports/report_{attack_name}.csv.

    Args:
        df: DataFrame from compare_attack_defense().
        attack_name: Used in the CSV filename.

    Returns:
        Pivot DataFrame (defenses × epsilons).
    """
    summary = df.groupby(["defense", "epsilon"])["accuracy"].max().reset_index()
    pivot = summary.pivot(index="defense", columns="epsilon", values="accuracy")
    pivot = pivot.round(4)
    csv_path = REPORTS_DIR / f"report_{attack_name}.csv"
    pivot.to_csv(csv_path)
    print(f"[comparison] Report saved to {csv_path}")
    return pivot
