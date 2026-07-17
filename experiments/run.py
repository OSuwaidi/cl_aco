"""Single training run.

    uv run python -m experiments.run --config experiments/configs/fmnist_smoke.yaml
    uv run python -m experiments.run --config ... --set sampling.pheromone.rho=1.0
"""

import argparse
import json

from smart_batches.config import load_config
from smart_batches.trainer import Trainer

from .plotting import plot_single_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY.PATH=VALUE")
    parser.add_argument("--resume", default=None, help="checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    summary = trainer.train()
    print(json.dumps(summary, indent=2))
    print(f"curves: {plot_single_run(trainer.out_dir)}")


if __name__ == "__main__":
    main()
