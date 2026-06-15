"""
SHAP (SHapley Additive exPlanations) model interpretability.

Uses TreeExplainer for tree-based models (RF, XGBoost, LightGBM)
and falls back to KernelExplainer for other model types. Generates
summary bar plots showing which features contribute most to model
predictions.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configs.config import REPORTS_DIR


def explain_model(model, X_sample: pd.DataFrame, model_name: str, save: bool = True):
    """
    Compute and visualise SHAP values for a given model.

    Args:
        model: Trained classifier.
        X_sample: A subset of data to explain (e.g., X_test.head(100)).
        model_name: Name used in the saved plot filename.
        save: If True, saves the summary plot to reports/.

    Returns:
        (shap_values, explainer) tuple for further analysis.
    """
    # Try TreeExplainer first (works for RF, XGBoost, LightGBM)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        print(f"[explainer] TreeExplainer used for {model_name}")
    except Exception:
        # Fall back to KernelExplainer (model-agnostic but slower)
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample)
        print(f"[explainer] KernelExplainer used for {model_name}")

    # Save the SHAP summary plot
    if save:
        # For multi-class with many classes (>2), summary_plot can't handle
        # the full values directly — it conflates class indices with feature
        # indices.  Instead, take the mean absolute SHAP across all classes
        # and plot a summary bar chart.
        # shap_values can be:
        #   - list of 2D arrays (older SHAP): len = n_classes, each (n, f)
        #   - 3D ndarray (newer SHAP):       shape (n, f, n_classes)
        n_classes = model.n_classes_ if hasattr(model, "n_classes_") else 1
        is_multi_class = (
            (isinstance(shap_values, list) and len(shap_values) > 2)
            or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] > 2)
        )
        if is_multi_class:
            if isinstance(shap_values, np.ndarray):
                # 3D array: (n_samples, n_features, n_classes) -> avg|SHAP|
                shap_avg = np.mean(np.abs(shap_values), axis=2)
            else:
                # list of 2D arrays: avg across all classes
                shap_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(shap_avg, X_sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
        path = REPORTS_DIR / f"shap_summary_{model_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explainer] SHAP summary saved to {path}")

    return shap_values, explainer


def get_feature_importance_shap(model, X_sample: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Return a DataFrame of features ranked by their mean absolute SHAP value.

    Useful for identifying which features the model relies on most.

    Args:
        model: Trained classifier.
        X_sample: Data to explain.
        model_name: Name of the model (for logging).

    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted by
        importance descending, or None if SHAP computation fails.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        return None

    # For multi-class models, shap_values is a list of arrays (one per class)
    if isinstance(shap_values, list):
        shap_values = np.abs(shap_values[1]) if len(shap_values) > 1 else np.abs(shap_values[0])
    else:
        shap_values = np.abs(shap_values)

    mean_shap = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return mean_shap
