from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_model(config: Dict[str, Any]):
    model_type = config.get("model_type", "logistic_regression")

    if model_type == "logistic_regression":
        hp = config.get("logistic_regression", {})
        return LogisticRegression(
            max_iter=int(hp.get("max_iter", 200)),
            solver=str(hp.get("solver", "lbfgs")),
        )

    if model_type == "mlp":
        hp = config.get("mlp", {})
        hidden = tuple(int(x) for x in hp.get("hidden_layer_sizes", [16, 16]))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=int(hp.get("max_iter", 500)),
            random_state=int(hp.get("random_state", config.get("random_state", 42))),
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def fit(model, X_train: np.ndarray, y_train: np.ndarray):
    model.fit(X_train, y_train)
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)