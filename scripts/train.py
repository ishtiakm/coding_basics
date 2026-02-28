import argparse
from coding_basics.pipelines.train import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    metrics = run(args.config)
    print(metrics)

if __name__ == "__main__":
    main()