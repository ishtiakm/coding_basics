from __future__ import annotations

import argparse

from coding_basics.pipelines import train as train_pipe
from coding_basics.pipelines import evaluate as eval_pipe
from coding_basics.pipelines import predict as pred_pipe


def main():
    parser = argparse.ArgumentParser(prog="coding_basics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", default="configs/default.yaml")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--config", default="configs/default.yaml")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", default="data/model.pkl")
    p_pred.add_argument("--sepal_length", type=float, required=True)
    p_pred.add_argument("--sepal_width", type=float, required=True)
    p_pred.add_argument("--petal_length", type=float, required=True)
    p_pred.add_argument("--petal_width", type=float, required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        metrics = train_pipe.run(args.config)
        print("Train metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.cmd == "evaluate":
        metrics = eval_pipe.run(args.config)
        print("Eval metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    elif args.cmd == "predict":
        out = pred_pipe.run(
            args.model,
            args.sepal_length, args.sepal_width,
            args.petal_length, args.petal_width
        )
        print(out)


if __name__ == "__main__":
    main()