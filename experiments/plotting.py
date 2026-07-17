"""Shared matplotlib styling and suite aggregation plots.

Colors come from a CVD-validated categorical palette; each strategy keeps its
slot in fixed order across every figure (color follows the entity). Yellow and
aqua slots sit below 3:1 on the light surface, so every figure carries a legend
and each suite also writes a summary.md table view.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, MUTED, GRID, BASELINE, SURFACE = (
    "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "font.size": 10,
    })


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def series(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["step"], r[key]) for r in records if key in r]
    if not pts:
        return np.array([]), np.array([])
    steps, vals = zip(*pts)
    return np.array(steps), np.array(vals)


def plot_single_run(out_dir: Path) -> Path:
    apply_style()
    records = load_jsonl(Path(out_dir) / "metrics.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("test/acc", "Test accuracy"),
              ("test/loss", "Test loss"),
              ("train/loss_ema", "Train loss (EMA)")]
    for ax, (key, title) in zip(axes, panels):
        s, v = series(records, key)
        if len(s):
            ax.plot(s, v, color=PALETTE[0])
        ax.set_title(title)
        ax.set_xlabel("step")
    fig.tight_layout()
    path = Path(out_dir) / "curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _mean_std(runs: list[list[dict]], key: str):
    """Align eval points across seeds by record index (identical step grids)."""
    per_run = [series(r, key) for r in runs]
    per_run = [(s, v) for s, v in per_run if len(s)]
    if not per_run:
        return None
    n = min(len(s) for s, _ in per_run)
    steps = per_run[0][0][:n]
    vals = np.stack([v[:n] for _, v in per_run])
    return steps, vals.mean(0), vals.std(0)


def plot_suite(suite_dir: Path, variant_names: list[str]) -> Path:
    apply_style()
    suite_dir = Path(suite_dir)
    data = {}
    for name in variant_names:
        runs = []
        for run_dir in sorted(suite_dir.glob(f"{name}_s*")):
            metrics = run_dir / "metrics.jsonl"
            if metrics.exists():
                runs.append(load_jsonl(metrics))
        if runs:
            data[name] = runs

    panels = [
        ("test/acc", "Test accuracy vs step", None),
        ("test/loss", "Test loss vs step", None),
        ("probe/grad_var_rel", "Relative gradient variance (probe)", "log"),
        ("sampler/entropy_norm", "Selection entropy / log N", None),
        ("sampler/coverage_epoch", "Unique coverage per epoch-equiv", None),
        ("__acc_vs_wall__", "Test accuracy vs wall-clock (s)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, yscale) in zip(axes.flat, panels):
        for i, (name, runs) in enumerate(data.items()):
            color = PALETTE[i % len(PALETTE)]
            if key == "__acc_vs_wall__":
                acc = _mean_std(runs, "test/acc")
                wall = _mean_std(runs, "time/wall_s")
                if acc is None or wall is None:
                    continue
                n = min(len(acc[1]), len(wall[1]))
                ax.plot(wall[1][:n], acc[1][:n], color=color, label=name)
            else:
                agg = _mean_std(runs, key)
                if agg is None:
                    continue
                steps, mean, std = agg
                ax.plot(steps, mean, color=color, label=name)
                ax.fill_between(steps, mean - std, mean + std,
                                color=color, alpha=0.18, linewidth=0)
        ax.set_title(title)
        if yscale:
            ax.set_yscale(yscale)
        if key != "__acc_vs_wall__":
            ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = suite_dir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_summary(suite_dir: Path, variant_names: list[str],
                  acc_targets: list[float]) -> Path:
    suite_dir = Path(suite_dir)
    rows = []
    for name in variant_names:
        summaries = []
        for run_dir in sorted(suite_dir.glob(f"{name}_s*")):
            f = run_dir / "summary.json"
            if f.exists():
                summaries.append(json.loads(f.read_text()))
        if not summaries:
            continue

        def agg(key):
            vals = [s[key] for s in summaries if key in s and s[key] is not None]
            if not vals:
                return "—"
            if len(vals) < len(summaries):  # some seeds never hit the target
                return f"{np.mean(vals):.0f} ({len(vals)}/{len(summaries)})"
            return f"{np.mean(vals):.4g} ± {np.std(vals):.2g}"

        row = {"variant": name, "seeds": len(summaries),
               "final_acc": agg("final_acc"), "best_acc": agg("best_acc"),
               "wall_s": agg("wall_s")}
        for t in acc_targets:
            key = f"steps_to_{int(round(t * 100))}"
            row[key] = agg(key)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No completed runs found in {suite_dir}")
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path = suite_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path
