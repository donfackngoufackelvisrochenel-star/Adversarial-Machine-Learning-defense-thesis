"""
Ensemble Defense.

Strategy: Combine predictions from multiple heterogeneous models
(Random Forest, XGBoost, LightGBM) via soft or hard voting. An
adversary must fool *all* models simultaneously to change the
ensemble's output, making it inherently more robust than any
single model.

Reference: Tramer et al., "Ensemble Adversarial Training: Attacks
and Defenses" (ICLR 2018).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.models.trainer import MODEL_REGISTRY, train_model, save_model
from configs.config import MODEL_PARAMS, RANDOM_STATE, DEFENSE_PARAMS


class EnsembleDefense:
    """
    Ensemble of multiple classifiers with configurable voting.

    Usage:
        ensemble = EnsembleDefense(voting='soft')
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
    """

    def __init__(self, models: dict = None, voting: str = "soft"):
        """
        Args:
            models: Optional dict of {name: model} to initialise with.
            voting: 'soft' (average probabilities) or 'hard' (majority vote).
        """
        self.models = models or {}
        self.voting = voting
        self.classes_ = None

    def add_model(self, name: str, model: object):
        """Register a trained model in the ensemble."""
        self.models[name] = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None):
        """
        Train all three base models (RF, XGB, LGBM) from scratch.

        Only trains models that have not already been added to the ensemble.
        """
        for name in MODEL_REGISTRY:
            if name not in self.models:
                print(f"[ensemble] Training {name}...")
                model = train_model(name, X_train, y_train, X_val, y_val)
                self.models[name] = model
        # Store the unique classes for consistent probability alignment
        self.classes_ = np.unique(y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels using the chosen voting strategy.

        - 'hard': each model votes for one class; the class with the most
          votes wins.
        - 'soft': the predicted probabilities are averaged across models;
          the class with the highest average probability wins.
        """
        if self.voting == "hard":
            # Majority vote
            preds = np.column_stack([m.predict(X) for m in self.models.values()])
            return np.array([np.bincount(row).argmax() for row in preds])
        else:
            # Soft voting (average probabilities)
            probs = self._predict_proba_array(X)
            return self.classes_[probs.argmax(axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return averaged class probabilities across all models."""
        return self._predict_proba_array(X)

    def _predict_proba_array(self, X: pd.DataFrame) -> np.ndarray:
        """
        Average the probability vectors from all models.

        Aligns every model's probability output to the ensemble's full
        class list so that averaging across models is well-defined.
        """
        all_probs = []
        for m in self.models.values():
            p = m.predict_proba(X)
            full = np.zeros((len(p), len(self.classes_)))
            for i, c in enumerate(m.classes_):
                idx = np.where(self.classes_ == c)[0]
                if len(idx) == 1:
                    full[:, idx[0]] = p[:, i]
            all_probs.append(full)
        return np.mean(all_probs, axis=0)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Return accuracy on the given test set."""
        return accuracy_score(y, self.predict(X))


def ensemble_defense(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
) -> EnsembleDefense:
    """
    Create and train an ensemble defense combining RF, XGBoost, and LightGBM.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        Trained EnsembleDefense instance.
    """
    params = DEFENSE_PARAMS["ensemble"]
    ensemble = EnsembleDefense(voting=params["voting"])
    ensemble.fit(X_train, y_train, X_val, y_val)
    save_model(ensemble, "ensemble", "defense")
    print(f"[ensemble] Ensemble saved as 'ensemble_defense.pkl'")
    return ensemble
