# Smart Batches

**ACO-inspired pheromone-guided mini-batch sampling for conditioning neural-network training.**

Every training sample carries a *pheromone* score. Mini-batches are drawn **with
replacement**, with selection probability increasing in pheromone strength. After each
optimizer step, the pheromones of the sampled items are updated from that step's
per-sample **loss** or per-sample **gradient magnitude (squared norm)** — so currently-hard
samples are revisited more often and easy ones less, adapting continuously as the model
learns. The idea combines curriculum-learning-style example weighting with the
deposit/evaporation dynamics of Ant Colony Optimization.

The hypothesis under test: adaptive batch composition yields **more informative, stable
gradient signals**, and therefore better optimization and generalization. This repo is a
research framework that implements the method plus the baselines and diagnostics needed
to measure that claim honestly.

## Method

### Pheromone table

Sample `i` holds pheromone `τᵢ ≥ 0`. Batch selection probabilities:

```
qᵢ = (τᵢ + ε)^α                     (or rank-based: qᵢ = rankᵢ^-α)
pᵢ = (1 − λ) · qᵢ / Σⱼ qⱼ  +  λ/N
```

- `α` — prioritization strength (ACO's pheromone exponent). `α = 0` is uniform;
  `α = 1` is fully proportional (the original formulation).
- `λ` — uniform mixing floor: every sample keeps probability ≥ `λ/N`, so nothing
  starves permanently.
- `transform: rank` replaces values with ranks — robust to heavy-tailed loss outliers.

Batches of size `B` are drawn from `p` **with replacement** via `torch.multinomial`.

`replacement: false` switches to a **without-replacement** variant: each batch is drawn
from the *remaining* samples of the current cycle in proportion to their current
pheromones, so one cycle serves every distinct sample exactly once — the strategy purely
**reorders** the data stream (hard samples early, adapting mid-cycle). Because a full
cycle is objective-identical to a uniform epoch, batch weights are always 1 and
`is_correction` does not apply in this mode.

### Pheromone update (deposit + evaporation)

After the optimizer step, each sampled item receives an ACO-style deposit with fresh
score `sᵢ` (duplicates within a batch are mean-reduced first):

```
τᵢ ← (1 − ρ)·τᵢ + ρ·sᵢ
```

`ρ = 1` is hard assignment ("pheromone := last loss"); `ρ < 1` is a soft/EMA assignment
that smooths augmentation and mini-batch noise — an evaporation-style forgetfulness
factor. `update_rule: accumulate` switches to the classic ACO dynamic
`τᵢ ← (1 − evap)·τᵢ + sᵢ`, where deposits pile up on repeatedly-hard samples
(`evap = 0`: the colony never forgets — the autocatalytic regime). Items **not**
sampled this step "gather dust": they drift toward the running
mean `s̄` of recently deposited scores at rate `κ`,

```
τⱼ ← τⱼ + κ·(s̄ − τⱼ)        (j not sampled)
```

so a sample that once scored low is gradually reconsidered, and stale-high scores relax
as the model improves globally.

### Scores

| `score` | What it is | Cost |
|---|---|---|
| `loss` | per-sample cross-entropy (`reduction='none'`) | free |
| `grad_norm_proxy` | **exact** squared gradient norm of the final linear layer: `‖softmax(zᵢ) − yᵢ‖² · (‖hᵢ‖² + 1)`, with `hᵢ` the penultimate activation captured by a forward hook | ~free (no extra backward) |
| `grad_norm_exact` | full per-sample squared gradient norm via `torch.func.vmap(grad(…))` | expensive; small models / validation (not supported on MPS) |

The proxy is a Katharopoulos & Fleuret-style bound on the full gradient norm and is what
`grad` mode uses in experiments.

### Bias correction

Non-uniform sampling biases the empirical-risk gradient. PER-style importance weights
correct it (on by default; `is_correction: none` gives the pure-conditioning ablation):

```
wᵢ = (N·pᵢ)^-β / maxⱼ wⱼ,     β: 0.4 → 1.0 linearly over training
training loss = mean(wᵢ · lossᵢ)
```

### Initialization ("the +inf idea")

Uniform-at-start is the degenerate case (`warmup_epochs: 0`, constant `τ_init`). The
default instead runs **one standard shuffled full pass** whose per-sample scores
hard-initialize `τ` — the same "everyone gets sampled early" effect the optimistic
`+inf` initialization aims for, but with honest first difficulty estimates.

### Defaults

| knob | default | meaning |
|---|---|---|
| `alpha` | 0.6 | prioritization strength |
| `lam` | 0.05 | uniform floor |
| `rho` | 0.6 | deposit rate (1.0 = hard assign) |
| `update_rule` | ema | or `accumulate` (classic ACO, with `evap`) |
| `replacement` | true | false = pheromone-ordered full passes (reordering only) |
| `kappa` | 1e-3 | staleness decay per step |
| `beta` | 0.4 → 1.0 | IS-weight annealing |
| `warmup_epochs` | 1 | full-coverage init pass |
| `transform` | proportional | or `rank` |

## Quickstart

```bash
uv sync                       # Python 3.13 env with torch/MPS
uv run pytest -q

# 2-minute sanity run (Fashion-MNIST subset, pheromone-loss)
uv run python -m experiments.run --config experiments/configs/fmnist_smoke.yaml

# main comparison: 4 strategies x 3 seeds on full Fashion-MNIST
uv run python -m experiments.compare --config experiments/configs/fmnist_suite.yaml

# soft-vs-hard assignment sweep (rho in {0.3, 0.6, 1.0})
uv run python -m experiments.compare --config experiments/configs/fmnist_rho_sweep.yaml
```

### CIFAR-10 on remote GPUs (CUDA)

The same code runs on CUDA unchanged (`train.device: auto` prefers CUDA). On each GPU
machine:

```bash
uv sync && uv run wandb login        # once per machine
# fixed baselines (epoch-shuffle + uniform, cosine SGD, 40 epochs):
uv run python -m experiments.compare --config experiments/configs/cifar10_baselines.yaml
# hyperparameter search over the pheromone design space (Bayes + Hyperband):
uv run wandb sweep --entity osuwaidi-khalifa-university --project CL-ACO \
    experiments/configs/cifar10_sweep.yaml          # prints the SWEEP_ID (create once)
uv run wandb agent osuwaidi-khalifa-university/CL-ACO/SWEEP_ID --count 12   # per machine
```

Sweep trials enter through `experiments/sweep_run.py`, which maps flat sweep parameters
(`ph_alpha`, `ph_score`, …) onto the nested config and trains normally; metrics log into
the sweep-managed wandb run.

Any config key can be overridden from the CLI:

```bash
uv run python -m experiments.run --config ... --set sampling.pheromone.rho=1.0 --set train.seed=3
```

### Strategies (`sampling.mode`)

- `epoch_shuffle` — the standard baseline (shuffled epochs, without replacement).
- `uniform_replacement` — uniform with replacement; isolates the effect of adaptive
  probabilities from replacement itself.
- `pheromone` — the method (choose the score with `sampling.pheromone.score`).

### Outputs

Each run writes `results/<suite>/<run>/`:
`config.json`, `metrics.jsonl` (source of truth), `summary.json`, `ckpt.pt` (full resume
state incl. τ and RNG), `curves.png`. Suites additionally get `comparison.png`
(mean ± std across seeds) and `summary.md`. Runs also log to Weights & Biases
(`osuwaidi-khalifa-university/CL-ACO`, grouped by suite); without a `wandb login` they
fall back to offline mode (`wandb sync` later to upload).

## What is measured

**Primary endpoint — performance/generalization** (ranks the strategies):
final and best test accuracy, steps/wall-clock to fixed accuracy targets, and the
train–test gap, aggregated mean ± std across seeds.

**Mechanism diagnostics** (does the method actually stabilize gradients?):

- **Gradient-variance probe** (`probe/grad_var`, `probe/grad_var_rel`): periodically
  freezes the model, draws M batches from the current policy, and measures
  `E‖g − ḡ‖²` (and normalized by `‖ḡ‖²`) across them — the variance of the gradient
  estimator each strategy actually feeds the optimizer. Dropout is disabled during the
  probe so the measured spread is attributable to batch selection.
- Selection entropy `H(p)/log N` and effective sample size `1/(N·Σpᵢ²)` — how
  concentrated sampling has become.
- `sampler/coverage_epoch` — unique-sample coverage per epoch-equivalent.
- `sampler/noisy_frac` (label-noise configs) — the share of draws that are corrupted
  samples. Loss-prioritization is expected to over-sample mislabeled data; the grad
  score, rank transform, and lower `ρ` are the mitigations to compare.

## Repository layout

```
smart_batches/           core package
  pheromones.py          PheromoneTable: probabilities, sampling, deposit/decay, IS weights
  samplers.py            epoch_shuffle / uniform_replacement / pheromone strategies
  scoring.py             loss | grad_norm_proxy | grad_norm_exact
  trainer.py             unified training loop, warmup, eval, checkpoint/resume
  data.py                tensor-backed datasets (fast arbitrary-index fetch), label noise
  metrics.py             gradient-variance probe
  models.py  config.py  loggers.py  utils.py
experiments/
  run.py                 single run          compare.py   suite grid + aggregation
  configs/               smoke, fmnist_suite, fmnist_rho_sweep, fmnist_aggressive,
                         fmnist_exact_grad, fmnist_noisy, cifar10_suite
tests/                   sampling correctness, update math, scorer exactness,
                         determinism, checkpoint-resume equivalence
```

Design note: with-replacement adaptive sampling needs arbitrary-index access whose
choice depends on the previous step, so DataLoader prefetching can't help. Datasets are
materialized as device tensors and batches gathered by indexing (CIFAR augmentation is
applied vectorized per batch at fetch time). `torch.multinomial` over N ≈ 60k per step
is negligible; a sum-tree is the upgrade path for N ≫ 1M.

## Deviations from the literal idea (and why)

| original idea | implementation | why |
|---|---|---|
| `τ = +inf` at start | 1 warmup epoch initializes τ with real scores | `+inf` breaks normalization; warmup achieves the same "sample everyone early" with honest estimates |
| `τ := loss` (assign) | `τ ← (1−ρ)τ + ρ·score`, `ρ` configurable | single-measurement scores are noisy; `ρ=1` recovers assignment exactly |
| `p ∝ τ` | `p ∝ (τ+ε)^α` mixed with `λ` uniform | `α` tempers heavy tails; `λ` floor prevents permanent starvation |
| — | staleness decay `κ` toward the running score mean | un-sampled scores go stale; "dust gathering" re-surfaces them |
| plain weighted loss | PER-style IS weights, β → 1 | keeps the optimization target aligned with the empirical risk |

## Related work

- Schaul et al., *Prioritized Experience Replay*, ICLR 2016 — α/β prioritization + IS correction.
- Loshchilov & Hutter, *Online Batch Selection for Faster Training of Neural Networks*, 2015.
- Katharopoulos & Fleuret, *Not All Samples Are Created Equal: Deep Learning with Importance Sampling*, ICML 2018 — last-layer gradient-norm bound.
- Bengio et al., *Curriculum Learning*, ICML 2009.
- Dorigo & Stützle, *Ant Colony Optimization*, 2004 — pheromone deposit/evaporation dynamics.
