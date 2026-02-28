from __future__ import annotations

import os
from typing import Any, Dict, List
import joblib


def save_artifact(path: str, model, feature_names: List[str], target_names: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names, "target_names": target_names},
        path
    )


def load_artifact(path: str) -> Dict[str, Any]:
    return joblib.load(path)