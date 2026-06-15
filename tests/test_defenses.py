import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.defenses.feature_squeezing import feature_squeezing_defense


def test_feature_squeezing():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 3), columns=[f"f{i}" for i in range(3)])
    y = pd.Series(np.random.randint(0, 2, 20))
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    X_sqz, acc = feature_squeezing_defense(model, X, y, squeeze_type="bit_depth", bit_depth=4)
    assert X_sqz.shape == X.shape
    assert 0.0 <= acc <= 1.0
