import torch

from smart_batches.config import load_config
from smart_batches.trainer import Trainer


def make_cfg(tmp_path, **tree):
    base = {
        "out_dir": str(tmp_path),
        "data": {"dataset": "synthetic", "subset_train": 1024, "subset_test": 256},
        "model": {"name": "small_cnn"},
        "optim": {"name": "adam", "lr": 1e-3},
        "train": {"batch_size": 64, "epochs": 3, "log_every": 8,
                  "probe_interval": 0, "seed": 0, "device": "cpu"},
        "sampling": {"mode": "pheromone", "warmup_epochs": 1},
        "wandb": {"enabled": False},
    }

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(base, tree)
    return load_config(base=base)


def run(tmp_path, name, **tree):
    cfg = make_cfg(tmp_path, run_name=name, **tree)
    trainer = Trainer(cfg, quiet=True)
    summary = trainer.train()
    return trainer, summary


def test_all_strategies_train_and_learn(tmp_path):
    variants = {
        "shuffle": {"sampling": {"mode": "epoch_shuffle"}},
        "uniform": {"sampling": {"mode": "uniform_replacement"}},
        "pher_loss": {"sampling": {"mode": "pheromone",
                                   "pheromone": {"score": "loss"}}},
        "pher_grad": {"sampling": {"mode": "pheromone",
                                   "pheromone": {"score": "grad_norm_proxy"}}},
        "pher_wor": {"sampling": {"mode": "pheromone",
                                  "pheromone": {"replacement": False}}},
    }
    for name, tree in variants.items():
        tree.setdefault("train", {})["probe_interval"] = 16
        _, summary = run(tmp_path, name, **tree)
        assert summary["best_acc"] > 50.0, f"{name} failed to learn"
        metrics = (tmp_path / name / "metrics.jsonl").read_text().strip().splitlines()
        assert len(metrics) > 0


def test_exact_grad_scorer_mode_runs(tmp_path):
    _, summary = run(
        tmp_path, "pher_exact",
        data={"subset_train": 128, "subset_test": 64},
        train={"epochs": 1, "batch_size": 32},
        sampling={"pheromone": {"score": "grad_norm_exact"}})
    assert summary["total_steps"] == 4


def test_same_seed_reproduces_run(tmp_path):
    _, s1 = run(tmp_path, "det_a")
    _, s2 = run(tmp_path, "det_b")
    assert s1["best_acc"] == s2["best_acc"]
    assert s1["final_loss"] == s2["final_loss"]


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    # interrupt a 3-epoch run after 2 epochs, resume it -> identical to a
    # never-interrupted 3-epoch run
    interrupted = Trainer(make_cfg(tmp_path, run_name="resume_part"), quiet=True)
    interrupted.train(until_step=2 * interrupted.steps_per_epoch)
    ckpt = interrupted.out_dir / "ckpt.pt"
    assert ckpt.exists()

    resumed = Trainer(make_cfg(tmp_path, run_name="resume_cont"), quiet=True)
    resumed.load_checkpoint(ckpt)
    assert resumed.step == 2 * resumed.steps_per_epoch
    resumed.train()

    straight, _ = run(tmp_path, "resume_straight")

    assert torch.allclose(resumed.table.tau, straight.table.tau)
    for k, v in straight.model.state_dict().items():
        assert torch.allclose(resumed.model.state_dict()[k], v, atol=1e-6), k
    idx_a, _ = resumed.sampler.next_batch(resumed.step)
    idx_b, _ = straight.sampler.next_batch(straight.step)
    assert torch.equal(idx_a, idx_b)


def test_cosine_scheduler_decays_lr(tmp_path):
    cfg = make_cfg(tmp_path, run_name="sched",
                   optim={"name": "sgd", "lr": 0.1, "scheduler": "cosine"},
                   train={"epochs": 1})
    trainer = Trainer(cfg, quiet=True)
    trainer.train(until_step=trainer.total_steps // 2)
    lr_now = trainer.opt.param_groups[0]["lr"]
    assert 0 < lr_now < 0.1


def test_coverage_is_full_during_warmup(tmp_path):
    import json

    _, _ = run(tmp_path, "coverage")
    records = [json.loads(l) for l in
               (tmp_path / "coverage" / "metrics.jsonl").read_text().splitlines()]
    first_epoch = next(r for r in records if r.get("epoch") == 1)
    # warmup epoch is a full shuffled pass (tail top-up may pull a few extras)
    assert first_epoch["sampler/coverage_epoch"] >= 0.999


def test_wor_mode_has_full_coverage_every_epoch(tmp_path):
    import json

    _, summary = run(tmp_path, "wor_cov",
                     sampling={"pheromone": {"replacement": False}})
    assert summary["best_acc"] > 50.0
    records = [json.loads(l) for l in
               (tmp_path / "wor_cov" / "metrics.jsonl").read_text().splitlines()]
    coverages = [r["sampler/coverage_epoch"] for r in records
                 if "sampler/coverage_epoch" in r]
    # reordering-only strategy: every epoch-equivalent touches ~all samples
    # (1024 samples, B=64, 16 steps/epoch -> cycles align with epochs)
    assert all(c >= 0.999 for c in coverages)
