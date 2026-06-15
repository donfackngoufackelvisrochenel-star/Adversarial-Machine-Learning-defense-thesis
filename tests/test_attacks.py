import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


def _dummy_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.randint(0, 2, 50))
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_fgsm_attack():
    model, X, y = _dummy_model()
    X_adv = fgsm_attack(model, X, y, epsilon=0.1)
    assert X_adv.shape == X.shape
    assert isinstance(X_adv, pd.DataFrame)


def test_pgd_attack():
    model, X, y = _dummy_model()
    X_adv = pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, num_iter=5)
    assert X_adv.shape == X.shape
    assert isinstance(X_adv, pd.DataFrame)
