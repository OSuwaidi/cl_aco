"""Suite runner: strategies x seeds grid, sequential runs, then aggregation.

    uv run python -m experiments.compare --config experiments/configs/fmnist_suite.yaml
    uv run python -m experiments.compare --config ... --aggregate-only
    uv run python -m experiments.compare --config ... --only pher_loss --seeds 0,1

Completed runs (out_dir/summary.json present) are skipped, so an interrupted
suite resumes at run granularity.
"""

import argparse
import copy
import gc
from pathlib import Path

import torch
import yaml

from smart_batches.config import load_config
from smart_batches.trainer import Trainer

from .plotting import plot_suite, write_summary


def set_dotted(tree: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def run_suite(spec_path: str, only: str | None = None,
              seeds: list[int] | None = None, aggregate_only: bool = False):
    spec = yaml.safe_load(Path(spec_path).read_text())
    suite = spec["suite"]
    seeds = seeds if seeds is not None else spec["seeds"]
    variants = spec["variants"]
    base = spec.get("base", {})
    suite_dir = Path(base.get("out_dir", f"results/{suite}"))

    if not aggregate_only:
        for variant in variants:
            if only and variant["name"] != only:
                continue
            for seed in seeds:
                run_name = f"{variant['name']}_s{seed}"
                out_dir = suite_dir / run_name
                if (out_dir / "summary.json").exists():
                    print(f"[skip] {run_name} already complete")
                    continue
                tree = copy.deepcopy(base)
                for key, value in variant.get("set", {}).items():
                    set_dotted(tree, key, value)
                set_dotted(tree, "train.seed", seed)
                set_dotted(tree, "run_name", run_name)
                set_dotted(tree, "out_dir", str(suite_dir))
                set_dotted(tree, "wandb.group", suite)
                cfg = load_config(base=tree)
                print(f"=== {suite} :: {run_name} ===")
                trainer = Trainer(cfg, out_dir=out_dir)
                trainer.train()
                del trainer
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    names = [v["name"] for v in variants]
    acc_targets = base.get("train", {}).get("acc_targets", [0.85, 0.88, 0.90])
    print(f"comparison plot: {plot_suite(suite_dir, names)}")
    summary_path = write_summary(suite_dir, names, acc_targets)
    print(f"summary table:  {summary_path}")
    print(summary_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="suite spec YAML")
    parser.add_argument("--only", default=None, help="run a single variant")
    parser.add_argument("--seeds", default=None, help="comma-separated override")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_suite(args.config, only=args.only, seeds=seeds,
              aggregate_only=args.aggregate_only)


if __name__ == "__main__":
    main()
