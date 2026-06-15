"""
Data scaling and label encoding processor.

Wraps scikit-learn's StandardScaler and LabelEncoder in a single
pipeline object that can be fitted on training data and then applied
to validation / test data, ensuring consistent transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path


class DataProcessor:
    """
    Handles feature scaling (z-score) and optional label encoding.

    Usage:
        processor = DataProcessor()
        X_train_scaled, y_train_encoded = processor.fit_transform(X_train, y_train)
        X_test_scaled,  y_test_encoded  = processor.transform(X_test, y_test)
        processor.save(MODELS_DIR / "processor.pkl")
    """

    def __init__(self):
        # StandardScaler computes mean and std-dev per feature for z-score normalization
        self.scaler = StandardScaler()
        # LabelEncoder maps string class labels to consecutive integers 0..N-1
        self.label_encoder = LabelEncoder()
        # Whether fit() has been called
        self._fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        Compute scaling parameters from training data and (optionally)
        learn the label encoding.

        Args:
            X_train: Training features.
            y_train: Training labels (optional, for label encoding).
        """
        # Fit scaler on training features
        self.scaler.fit(X_train)
        # Remember feature names so transform() can reorder columns
        self._feature_names = X_train.columns.tolist()

        # Fit label encoder only if y_train contains string labels
        # Check dtype *name* to support both 'object' and 'str' (pandas >= 2.0)
        if y_train is not None and y_train.dtype.name in ("object", "str", "string", "category"):
            self.label_encoder.fit(y_train)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Apply scaling and (optionally) label encoding.

        Args:
            X: Features to scale.
            y: Labels to encode (optional).

        Returns:
            (X_scaled, y_encoded) tuple.
            If y is None, returns (X_scaled, None).
        """
        if not self._fitted:
            raise ValueError("Processor not fitted yet. Call fit() first.")

        # Ensure columns match the training order (important for prediction)
        X = X[self._feature_names].copy() if hasattr(self, '_feature_names') else X.copy()

        # Apply z-score scaling
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self._feature_names if hasattr(self, '_feature_names') else X.columns,
            index=X.index,
        )

        # Encode labels if they are still strings
        if y is not None and y.dtype.name in ("object", "str", "string", "category"):
            y_enc = self.label_encoder.transform(y)
            return X_scaled, y_enc

        return X_scaled, y

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """Convenience: fit on training data, then transform it."""
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    def save(self, path: Path):
        """Persist the fitted processor to disk with joblib."""
        joblib.dump({
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self._feature_names,
        }, path)

    def load(self, path: Path):
        """Restore a previously saved processor."""
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self._feature_names = data["feature_names"]
        self._fitted = True
