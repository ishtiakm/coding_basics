from __future__ import annotations

from typing import Tuple, List, Dict
import numpy as np
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def load_iris():
    ds = sklearn_load_iris()
    X = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)
    feature_names = list(ds.feature_names)
    target_names = list(ds.target_names)
    return X, y, feature_names, target_names


def split_data(X, y, test_size: float, random_state: int):
    return train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y
    )


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }