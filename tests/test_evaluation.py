import numpy as np
from src.evaluation.metrics import compute_metrics, attack_success_rate


def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.4, 0.6]])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in m
    assert 0.0 <= m["accuracy"] <= 1.0


def test_attack_success_rate():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    asr = attack_success_rate(y_true, y_pred)
    assert asr == 0.5
