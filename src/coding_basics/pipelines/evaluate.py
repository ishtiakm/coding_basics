from __future__ import annotations

from typing import Dict, Any
import yaml

from coding_basics.utils import load_iris, split_data, compute_metrics
from coding_basics.models.net import predict
from coding_basics.io import load_artifact


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    save_path = cfg.get("save_path", "data/model.pkl")

    artifact = load_artifact(save_path)
    model = artifact["model"]

    X, y, *_ = load_iris()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
    )

    y_pred = predict(model, X_test)
    return compute_metrics(y_test, y_pred)