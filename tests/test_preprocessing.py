import pandas as pd
import numpy as np
from src.preprocessing.loader import clean_data, split_data
from src.preprocessing.processor import DataProcessor


def test_clean_data():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [None, None, None, None], "c": ["x", "y", "z", "w"], "label": [0, 1, 0, 1]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
    assert "b" not in cleaned.columns


def test_split_data():
    df = pd.DataFrame({"f1": range(100), "f2": range(100), "label": [0] * 50 + [1] * 50})
    result = split_data(df)
    assert len(result) == 6
    assert result[0].shape[0] > 0
    assert result[2].shape[0] > 0


def test_processor():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])
    proc = DataProcessor()
    Xt, yt = proc.fit_transform(X, y)
    assert abs(Xt["a"].mean()) < 1e-10
    assert Xt.shape == X.shape
