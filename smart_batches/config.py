"""Config dataclasses, YAML loading, and dotted-path CLI overrides."""

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

VALID_MODES = ("epoch_shuffle", "uniform_replacement", "pheromone")
VALID_SCORES = ("loss", "grad_norm_proxy", "grad_norm_exact")
VALID_TRANSFORMS = ("proportional", "rank")
VALID_IS = ("per", "none")
VALID_UPDATE_RULES = ("ema", "accumulate")


@dataclass
class PheromoneConfig:
    alpha: float = 0.6          # prioritization exponent (ACO pheromone exponent)
    alpha_end: float | None = None  # if set, anneal alpha -> alpha_end linearly over training
    lam: float = 0.05           # uniform mixing floor: p = (1-lam)*q/sum(q) + lam/N
    rho: float = 0.6            # ema rule deposit rate: tau <- (1-rho)*tau + rho*score (1.0 = hard assign)
    update_rule: str = "ema"    # ema | accumulate (classic ACO: tau <- (1-evap)*tau + score)
    evap: float = 0.0           # accumulate rule evaporation; 0 = deposits never forgotten
    kappa: float = 1e-3         # staleness decay of un-sampled tau toward running score mean
    eps: float = 1e-6
    transform: str = "proportional"  # proportional | rank
    replacement: bool = True    # False: pheromone-ordered pass over all distinct
                                # samples per cycle (pure reordering; IS weights
                                # are inert since the per-cycle objective is
                                # identical to a uniform epoch)
    tau_init: float = 1.0
    score: str = "loss"         # loss | grad_norm_proxy | grad_norm_exact
    is_correction: str = "per"  # per (PER-style importance weights) | none
    beta_start: float = 0.4
    beta_end: float = 1.0


@dataclass
class SamplingConfig:
    mode: str = "pheromone"     # epoch_shuffle | uniform_replacement | pheromone
    warmup_epochs: int = 1      # full-coverage shuffled passes that initialize tau
    pheromone: PheromoneConfig = field(default_factory=PheromoneConfig)


@dataclass
class DataConfig:
    dataset: str = "fashion_mnist"  # mnist | fashion_mnist | cifar10 | synthetic
    root: str = "data"
    label_noise: float = 0.0
    noise_seed: int = 0
    subset_train: int = 0       # 0 = full; >0 = random-k subset (seeded)
    subset_test: int = 0
    subset_seed: int = 1234


@dataclass
class ModelConfig:
    name: str = "small_cnn"     # small_cnn | resnet18
    num_classes: int = 10


@dataclass
class OptimConfig:
    name: str = "adam"          # adam | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9       # sgd only
    scheduler: str = "none"     # none | cosine (per-step over total_steps)


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 15            # epoch-equivalents: ceil(N/B) steps each
    log_every: int = 50
    probe_interval: int = 0     # gradient-variance probe cadence in steps; 0 = off
    probe_batches: int = 8
    exact_corr_interval: int = 0  # loss/proxy vs exact-grad correlation probe; 0 = off
    acc_targets: list = field(default_factory=lambda: [0.85, 0.88, 0.90])
    seed: int = 0
    device: str = "auto"


@dataclass
class WandbConfig:
    enabled: bool = True
    entity: str = "osuwaidi-khalifa-university"
    project: str = "CL-ACO"
    group: str = ""
    mode: str = "auto"          # auto | online | offline | disabled


@dataclass
class Config:
    run_name: str = ""
    out_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> "Config":
        s = self.sampling
        if s.mode not in VALID_MODES:
            raise ValueError(f"sampling.mode must be one of {VALID_MODES}, got {s.mode!r}")
        if s.pheromone.score not in VALID_SCORES:
            raise ValueError(f"sampling.pheromone.score must be one of {VALID_SCORES}")
        if s.pheromone.transform not in VALID_TRANSFORMS:
            raise ValueError(f"sampling.pheromone.transform must be one of {VALID_TRANSFORMS}")
        if s.pheromone.is_correction not in VALID_IS:
            raise ValueError(f"sampling.pheromone.is_correction must be one of {VALID_IS}")
        if s.pheromone.update_rule not in VALID_UPDATE_RULES:
            raise ValueError(f"sampling.pheromone.update_rule must be one of {VALID_UPDATE_RULES}")
        if not 0.0 < s.pheromone.rho <= 1.0:
            raise ValueError("sampling.pheromone.rho must be in (0, 1]")
        if s.pheromone.alpha_end is not None and s.pheromone.alpha_end < 0:
            raise ValueError("sampling.pheromone.alpha_end must be >= 0")
        if self.optim.scheduler not in ("none", "cosine"):
            raise ValueError("optim.scheduler must be 'none' or 'cosine'")
        if not 0.0 <= s.pheromone.lam < 1.0:
            raise ValueError("sampling.pheromone.lam must be in [0, 1)")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _build_dataclass(cls, data: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _build_dataclass(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _set_dotted(tree: dict, dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {dotted_key}: {k} is not a section")
    # yaml parsing coerces "0.5" -> float, "true" -> bool, "[1,2]" -> list, etc.
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path | None = None,
                overrides: list[str] | None = None,
                base: dict | None = None) -> Config:
    tree: dict = dict(base or {})
    if path is not None:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        tree = _merge(tree, loaded)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(tree, key.strip(), value)
    return _build_dataclass(Config, tree).validate()


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
