"""
Feature Squeezing Defense.

Strategy: Reduce the precision of input features so that an adversary
cannot hide subtle perturbations. By squeezing each feature into a
coarser quantisation, small malicious changes are removed while the
overall classification remains accurate for legitimate samples.

Two squeezers are provided:
  1. Bit-depth reduction  — quantise continuous features to 2^bits levels.
  2. Median smoothing     — replace each value with the median of its
     neighbours (useful for ordered / time-series features).

Reference: Xu et al., "Feature Squeezing: Detecting Adversarial Examples
in Deep Neural Networks" (NDSS 2018).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from configs.config import DEFENSE_PARAMS


def _reduce_bit_depth(X: pd.DataFrame, bits: int = 4) -> pd.DataFrame:
    """
    Quantise each continuous feature to a given number of bits.

    For each feature column, the value range [min, max] is divided into
    2^bits equal-width bins. All values are rounded to the nearest bin
    centre. This removes low-magnitude perturbations.

    Args:
        X: Feature DataFrame.
        bits: Number of quantisation bits (4 → 16 levels).

    Returns:
        Quantised DataFrame.
    """
    squeezed = X.copy()
    for col in squeezed.columns:
        col_min, col_max = squeezed[col].min(), squeezed[col].max()
        if col_max > col_min:  # skip constant columns
            levels = 2 ** bits
            # Normalise to [0, 1], quantise, then scale back
            squeezed[col] = np.round(
                (squeezed[col] - col_min) / (col_max - col_min) * (levels - 1)
            ) / (levels - 1) * (col_max - col_min) + col_min
    return squeezed


def _median_smoothing(X: pd.DataFrame, kernel_size: int = 3) -> pd.DataFrame:
    """
    Apply a rolling median filter to each feature column.

    Each value is replaced by the median of a window of `kernel_size`
    neighbouring rows. This smooths out isolated spikes that an adversary
    might introduce.

    Note: This assumes that the rows have some meaningful order. For
    randomly shuffled data the effect is similar to adding a small amount
    of random noise.

    Args:
        X: Feature DataFrame.
        kernel_size: Window size for the rolling median.

    Returns:
        Smoothed DataFrame.
    """
    smoothed = X.copy()
    for col in smoothed.columns:
        smoothed[col] = (
            smoothed[col]
            .rolling(window=kernel_size, center=True, min_periods=1)
            .median()
        )
    return smoothed


def feature_squeezing_defense(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    squeeze_type: str = "bit_depth",
    bit_depth: int = None,
) -> (pd.DataFrame, float):
    """
    Apply feature squeezing and evaluate model accuracy on the squeezed data.

    Args:
        model: Trained classifier to evaluate.
        X_test: Test features.
        y_test: True test labels.
        squeeze_type: 'bit_depth', 'median', or 'both'.
        bit_depth: Number of quantisation bits.

    Returns:
        (squeezed X, accuracy of model on squeezed data)
    """
    params = DEFENSE_PARAMS["feature_squeezing"]
    bit_depth = bit_depth or params["bit_depth"]

    print(f"[feature_squeezing] Applying '{squeeze_type}' defense...")

    # Apply the selected squeezing technique(s)
    if squeeze_type == "bit_depth":
        X_squeezed = _reduce_bit_depth(X_test, bits=bit_depth)
    elif squeeze_type == "median":
        X_squeezed = _median_smoothing(X_test)
    elif squeeze_type == "both":
        X_squeezed = _reduce_bit_depth(X_test, bits=bit_depth)
        X_squeezed = _median_smoothing(X_squeezed)
    else:
        raise ValueError(f"Unknown squeeze_type: {squeeze_type}")

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_squeezed))
    print(f"[feature_squeezing] Post-squeeze accuracy: {acc:.4f}")
    return X_squeezed, acc
