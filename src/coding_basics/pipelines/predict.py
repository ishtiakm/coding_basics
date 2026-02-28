from __future__ import annotations

import numpy as np
from coding_basics.models.net import predict as model_predict
from coding_basics.io import load_artifact


def run(model_path: str, sl: float, sw: float, pl: float, pw: float) -> str:
    artifact = load_artifact(model_path)
    model = artifact["model"]
    target_names = artifact["target_names"]

    x = np.array([[sl, sw, pl, pw]], dtype=float)
    pred_class = int(model_predict(model, x)[0])
    return f"Predicted class: {pred_class} ({target_names[pred_class]})"