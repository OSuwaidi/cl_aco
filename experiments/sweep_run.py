"""Weights & Biases sweep trial entry point.

The sweep controller starts this program via `wandb agent`; wandb.init()
attaches to the sweep-managed run and exposes the trial's hyperparameters in
wandb.config. Sweep parameter names (flat, underscore) are mapped onto the
nested config tree below, then a normal Trainer run executes and logs into the
adopted run (see RunLogger).

    uv run wandb agent <entity>/<project>/<sweep_id>
"""

import argparse

import wandb

from smart_batches.config import load_config
from smart_batches.trainer import Trainer

PARAM_MAP = {
    "ph_score": "sampling.pheromone.score",
    "ph_alpha": "sampling.pheromone.alpha",
    "ph_alpha_end": "sampling.pheromone.alpha_end",  # -1 means "no annealing"
    "ph_rho": "sampling.pheromone.rho",
    "ph_kappa": "sampling.pheromone.kappa",
    "ph_lam": "sampling.pheromone.lam",
    "ph_is": "sampling.pheromone.is_correction",
    "ph_beta_start": "sampling.pheromone.beta_start",
    "ph_beta_end": "sampling.pheromone.beta_end",
    "ph_transform": "sampling.pheromone.transform",
    "ph_replacement": "sampling.pheromone.replacement",
    "ph_update_rule": "sampling.pheromone.update_rule",
    "ph_evap": "sampling.pheromone.evap",
    "warmup_epochs": "sampling.warmup_epochs",
    "seed": "train.seed",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="base config YAML")
    args, _ = parser.parse_known_args()

    run = wandb.init()
    overrides = []
    for key, value in dict(run.config).items():
        if key not in PARAM_MAP:
            continue
        if key == "ph_alpha_end" and float(value) < 0:
            value = "null"
        overrides.append(f"{PARAM_MAP[key]}={value}")

    cfg = load_config(args.config, overrides=overrides)
    cfg.run_name = f"sweep_{run.id}"
    trainer = Trainer(cfg)
    summary = trainer.train()
    print(summary)


if __name__ == "__main__":
    main()
