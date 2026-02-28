# tests/test_utils.py
from __future__ import annotations

import math

import numpy as np
import pytest

from coding_basics.utils import load_iris, split_data, compute_metrics


def test_load_iris_shapes_and_names():
    X, y, feature_names, target_names = load_iris()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    # Basic Iris properties
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 4  # sepal length/width, petal length/width

    # Names
    assert len(feature_names) == 4
    assert len(target_names) == 3
    assert all(isinstance(n, str) for n in feature_names)
    assert all(isinstance(n, str) for n in target_names)


def test_split_data_preserves_counts_and_stratification():
    X, y, *_ = load_iris()

    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    # Total counts preserved
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]

    # Feature dimensions preserved
    assert X_train.shape[1] == X.shape[1]
    assert X_test.shape[1] == X.shape[1]

    # Approx size check (allow 1 sample rounding)
    expected_test = int(round(X.shape[0] * test_size))
    assert abs(X_test.shape[0] - expected_test) <= 1

    # Stratification sanity: all classes should appear in both splits for Iris
    assert set(np.unique(y)) == set(np.unique(y_train))
    assert set(np.unique(y)) == set(np.unique(y_test))


def test_compute_metrics_keys_and_ranges():
    y_true = np.array([0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 2, 1, 2])

    m = compute_metrics(y_true, y_pred)

    assert isinstance(m, dict)
    assert "accuracy" in m
    assert "macro_f1" in m

    # Values should be finite and in [0, 1]
    for k in ["accuracy", "macro_f1"]:
        assert isinstance(m[k], float)
        assert math.isfinite(m[k])
        assert 0.0 <= m[k] <= 1.0


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 2, 1, 0])

    m = compute_metrics(y_true, y_pred)

    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)