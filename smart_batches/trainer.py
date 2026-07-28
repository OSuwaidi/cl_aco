"""Training loop shared by all sampling strategies.

Per step: sample indices -> fetch batch -> forward -> importance-weighted loss
-> backward/step -> score samples -> deposit pheromones -> periodic eval,
probes, and logging. Warmup epochs (pheromone mode) run standard shuffled
full-coverage passes whose scores hard-initialize tau, replacing the "+inf
optimistic init" idea with honest first difficulty estimates.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data import build_data
from .loggers import RunLogger
from .metrics import probe_gradient_variance, probe_score_correlation
from .models import build_model
from .pheromones import PheromoneTable, linear_beta
from .samplers import (EpochShuffleSampler, PheromoneSampler,
                       PheromoneWORSampler, UniformReplacementSampler)
from .scoring import make_scorer
from .utils import Timer, pick_device, seed_everything

LOSS_EMA = 0.02


def _acc_pct(acc: float) -> float:
    return round(acc * 100.0, 2)


class Trainer:
    def __init__(self, cfg: Config, out_dir: str | Path | None = None,
                 quiet: bool = False):
        cfg.validate()
        self.cfg = cfg
        self.quiet = quiet
        seed_everything(cfg.train.seed)
        self.device = pick_device(cfg.train.device)

        self.train_data, self.test_data = build_data(cfg.data)
        self.train_data.to(self.device)
        self.test_data.to(self.device)
        self.n = len(self.train_data)

        img_size = self.train_data.x.shape[-1]
        self.model = build_model(cfg.model.name, self.train_data.in_channels,
                                 cfg.model.num_classes, img_size).to(self.device)
        self.opt = self._build_optimizer()

        tc = cfg.train
        self.batch_size = tc.batch_size
        self.steps_per_epoch = math.ceil(self.n / self.batch_size)
        self.total_steps = tc.epochs * self.steps_per_epoch
        self.scheduler = None
        if cfg.optim.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=self.total_steps)
        self.gen = torch.Generator().manual_seed(tc.seed)          # batch selection
        self.aug_gen = torch.Generator().manual_seed(tc.seed + 7)  # augmentation

        mode = cfg.sampling.mode
        self.is_pheromone = mode == "pheromone"
        self.table = None
        self.scorer = None
        self.warmup_sampler = None
        self.warmup_steps = 0
        if self.is_pheromone:
            ph = cfg.sampling.pheromone
            self.table = PheromoneTable(
                self.n, alpha=ph.alpha, lam=ph.lam, rho=ph.rho, kappa=ph.kappa,
                eps=ph.eps, transform=ph.transform, tau_init=ph.tau_init,
                update_rule=ph.update_rule, evap=ph.evap)
            if ph.replacement:
                self.sampler = PheromoneSampler(
                    self.table, self.batch_size, self.gen, ph.is_correction,
                    ph.beta_start, ph.beta_end, self.total_steps,
                    alpha_end=ph.alpha_end)
            else:
                self.sampler = PheromoneWORSampler(
                    self.table, self.batch_size, self.gen, self.total_steps,
                    alpha_end=ph.alpha_end)
            self.scorer = make_scorer(ph.score, self.model)
            self.warmup_steps = cfg.sampling.warmup_epochs * self.steps_per_epoch
            if self.warmup_steps > 0:
                self.warmup_sampler = EpochShuffleSampler(
                    self.n, self.batch_size, self.gen)
        elif mode == "epoch_shuffle":
            self.sampler = EpochShuffleSampler(self.n, self.batch_size, self.gen)
        elif mode == "uniform_replacement":
            self.sampler = UniformReplacementSampler(self.n, self.batch_size, self.gen)
        else:
            raise ValueError(mode)

        self.run_name = cfg.run_name or self._default_run_name()
        self.out_dir = Path(out_dir) if out_dir else Path(cfg.out_dir) / self.run_name
        self.logger = RunLogger(self.out_dir, cfg.to_dict(), cfg.wandb, self.run_name)

        self.step = 0
        self.best_acc = 0.0
        self.loss_ema = None
        self.targets_hit: dict[str, dict] = {}
        self.seen = torch.zeros(self.n, dtype=torch.bool)
        self.noisy_draws = 0
        self.total_draws = 0
        self.timer = Timer()
        self._wall_offset = 0.0

    def _default_run_name(self) -> str:
        cfg = self.cfg
        if self.is_pheromone:
            wor = "" if cfg.sampling.pheromone.replacement else "_wor"
            return f"pheromone_{cfg.sampling.pheromone.score}{wor}_s{cfg.train.seed}"
        return f"{cfg.sampling.mode}_s{cfg.train.seed}"

    def _build_optimizer(self):
        oc = self.cfg.optim
        if oc.name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=oc.lr,
                                    weight_decay=oc.weight_decay)
        if oc.name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=oc.lr,
                                   momentum=oc.momentum,
                                   weight_decay=oc.weight_decay)
        raise ValueError(oc.name)

    def _wall(self) -> float:
        return self._wall_offset + self.timer.elapsed()

    # ----------------------------------------------------------------- training
    def train(self, until_step: int | None = None) -> dict:
        """Runs to completion, or to `until_step` (checkpointing there) so an
        interrupted run can be resumed exactly via load_checkpoint."""
        tc = self.cfg.train
        track_noise = bool(self.train_data.corrupted.any())
        stop_step = self.total_steps if until_step is None else min(
            until_step, self.total_steps)
        self.model.train()

        for step in range(self.step, stop_step):
            in_warmup = self.is_pheromone and step < self.warmup_steps
            active = self.warmup_sampler if in_warmup else self.sampler
            idx, w = active.next_batch(step)

            x, y = self.train_data.fetch(idx, train=True, generator=self.aug_gen)
            logits = self.model(x)
            losses = F.cross_entropy(logits, y, reduction="none")
            loss = (w.to(self.device) * losses).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_pheromone:
                scores = self.scorer(x, y, logits, losses)
                if in_warmup:
                    self.table.set_scores(idx, scores)
                else:
                    self.sampler.update(idx, scores)

            with torch.no_grad():
                batch_loss = losses.mean().item()
                batch_acc = (logits.argmax(1) == y).float().mean().item()
            self.loss_ema = (batch_loss if self.loss_ema is None else
                             (1 - LOSS_EMA) * self.loss_ema + LOSS_EMA * batch_loss)
            self.seen[idx] = True
            if track_noise:
                self.noisy_draws += int(self.train_data.corrupted[idx].sum())
                self.total_draws += len(idx)
            self.step = step + 1

            record: dict = {}
            if self.step % tc.log_every == 0:
                record.update({
                    "train/loss": batch_loss,
                    "train/loss_ema": self.loss_ema,
                    "train/acc_batch": _acc_pct(batch_acc),
                    "time/wall_s": self._wall(),
                })
                if self.is_pheromone and not in_warmup:
                    ph = self.cfg.sampling.pheromone
                    record.update({
                        "is/w_mean": w.mean().item(),
                        "is/w_min": w.min().item(),
                        "is/beta": linear_beta(step, self.total_steps,
                                               ph.beta_start, ph.beta_end),
                    })
            if tc.probe_interval and self.step % tc.probe_interval == 0:
                record.update(probe_gradient_variance(
                    self.model, self.train_data, active, self.device,
                    m_batches=tc.probe_batches, seed=tc.seed + self.step,
                    step=step))
            if tc.exact_corr_interval and self.step % tc.exact_corr_interval == 0:
                record.update(probe_score_correlation(self.model, x, y))
            if self.step % self.steps_per_epoch == 0:
                record.update(self._eval_record(track_noise))
            if record:
                self.logger.log(self.step, record)

        if self.step < self.total_steps:  # interrupted: checkpoint, no summary
            self.save_checkpoint()
            self.logger.finish()
            return {}
        summary = self._summary()
        self.save_checkpoint()
        self.logger.finish(summary)
        return summary

    def _eval_record(self, track_noise: bool) -> dict:
        epoch = self.step // self.steps_per_epoch
        test_loss, test_acc = self.evaluate()
        record = {
            "epoch": epoch,
            "test/loss": test_loss,
            "test/acc": _acc_pct(test_acc),
            "sampler/coverage_epoch": self.seen.float().mean().item(),
            "time/wall_s": self._wall(),
        }
        self.seen.zero_()
        if self.is_pheromone:
            record.update({f"sampler/{k}": v for k, v in self.table.stats().items()})
        if track_noise and self.total_draws:
            record["sampler/noisy_frac"] = self.noisy_draws / self.total_draws
            self.noisy_draws = self.total_draws = 0

        self.best_acc = max(self.best_acc, test_acc)
        for t in self.cfg.train.acc_targets:
            key = f"{int(round(t * 100))}"
            if test_acc >= t and key not in self.targets_hit:
                self.targets_hit[key] = {"step": self.step, "wall_s": self._wall()}
        if not self.quiet:
            print(f"[{self.run_name}] epoch {epoch:3d} step {self.step:6d} "
                  f"test_acc {test_acc:.4f} test_loss {test_loss:.4f} "
                  f"wall {self._wall():7.1f}s")
        return record

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, count = 0.0, 0, 0
        for x, y in self.test_data.iter_eval():
            logits = self.model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
        self.model.train()
        return total_loss / count, correct / count

    def _summary(self) -> dict:
        test_loss, test_acc = self.evaluate()
        summary = {
            "run_name": self.run_name,
            "mode": self.cfg.sampling.mode,
            "score": (self.cfg.sampling.pheromone.score if self.is_pheromone else None),
            "seed": self.cfg.train.seed,
            "final_loss": test_loss,
            "best_acc": _acc_pct(self.best_acc),
            "total_steps": self.total_steps,
            "wall_s": self._wall(),
        }
        for key, hit in self.targets_hit.items():
            summary[f"steps_to_{key}"] = hit["step"]
            summary[f"wall_s_to_{key}"] = hit["wall_s"]
        return summary

    # -------------------------------------------------------------- persistence
    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.out_dir / "ckpt.pt"
        state = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "best_acc": self.best_acc,
            "loss_ema": self.loss_ema,
            "targets_hit": self.targets_hit,
            "seen": self.seen,
            "noisy_draws": self.noisy_draws,
            "total_draws": self.total_draws,
            "sampler": self.sampler.state_dict(),
            "scheduler": (self.scheduler.state_dict()
                          if self.scheduler is not None else None),
            "warmup_sampler": (self.warmup_sampler.state_dict()
                               if self.warmup_sampler else None),
            "gen": self.gen.get_state(),
            "aug_gen": self.aug_gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "wall_s": self._wall(),
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.step = state["step"]
        self.best_acc = state["best_acc"]
        self.loss_ema = state["loss_ema"]
        self.targets_hit = state["targets_hit"]
        self.seen = state["seen"].clone()
        self.noisy_draws = state["noisy_draws"]
        self.total_draws = state["total_draws"]
        self.sampler.load_state_dict(state["sampler"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.warmup_sampler and state["warmup_sampler"] is not None:
            self.warmup_sampler.load_state_dict(state["warmup_sampler"])
        self.gen.set_state(state["gen"])
        self.aug_gen.set_state(state["aug_gen"])
        torch.set_rng_state(state["torch_rng"])
        self._wall_offset = state["wall_s"]
        self.timer = Timer()
