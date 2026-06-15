"""
Real-time IoMT data streamer for live inference simulation.

Loads the processed test set from disk and streams samples
one at a time, optionally with adversarial perturbations.
Supports both polling and WebSocket-based consumption.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from configs.config import MODELS_DIR


class IoMTStreamer:
    def __init__(self, X_test, y_test, model=None, delay=0.05):
        self.X = X_test.reset_index(drop=True) if isinstance(X_test, pd.DataFrame) else X_test
        self.y = y_test.reset_index(drop=True) if isinstance(y_test, pd.Series) else y_test
        self.model = model
        self.delay = delay
        self._idx = 0
        self._n = len(self.X)
        self.attack_active = False
        self.attack_type = "fgsm"
        self.epsilon = 0.1

    @classmethod
    def from_disk(cls, model_name=None, delay=0.05):
        X = joblib.load(MODELS_DIR / "X_test.pkl")
        y = joblib.load(MODELS_DIR / "y_test.pkl")
        model = None
        if model_name:
            path = MODELS_DIR / f"{model_name}_clean.pkl"
            if path.exists():
                model = joblib.load(path)
        return cls(X, y, model=model, delay=delay)

    def next_sample(self):
        if self._n == 0:
            raise StopIteration("No test samples loaded")
        if self._idx >= self._n:
            self._idx = 0
        sample = self.X.iloc[self._idx]
        label = int(self.y.iloc[self._idx])
        self._idx += 1
        if self.delay > 0:
            time.sleep(self.delay)
        return sample, label

    def next_sample_perturbed(self):
        sample, label = self.next_sample()
        if self.attack_active and self.model is not None and self.epsilon > 0:
            df = pd.DataFrame([sample])
            if self.attack_type == "pgd":
                from src.attacks.pgd import pgd_attack
                df_adv = pgd_attack(self.model, df, pd.Series([label]), epsilon=self.epsilon)
            else:
                from src.attacks.fgsm import fgsm_attack
                df_adv = fgsm_attack(self.model, df, pd.Series([label]), epsilon=self.epsilon)
            sample = df_adv.iloc[0]
        return sample, label

    def predict_sample(self, sample):
        if self.model is None:
            return None, None
        df = pd.DataFrame([sample])
        pred = int(self.model.predict(df)[0])
        prob = float(np.max(self.model.predict_proba(df)[0]))
        return pred, prob

    def stream(self, max_samples=None, perturbed=False):
        count = 0
        while max_samples is None or count < max_samples:
            try:
                if perturbed:
                    sample, label = self.next_sample_perturbed()
                else:
                    sample, label = self.next_sample()
                pred, prob = self.predict_sample(sample)
                yield {
                    "sample_index": self._idx - 1,
                    "true_label": label,
                    "prediction": pred,
                    "confidence": prob,
                    "correct": bool(pred == label),
                    "attack_active": self.attack_active and perturbed,
                    "attack_type": self.attack_type if self.attack_active else "none",
                    "epsilon": self.epsilon if self.attack_active else 0.0,
                }
                count += 1
            except StopIteration:
                break

    @property
    def progress(self):
        return f"{self._idx} / {self._n}"

    def reset(self):
        self._idx = 0
