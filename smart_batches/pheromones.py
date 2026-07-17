"""PheromoneTable — the core contribution.

Each training sample i carries a pheromone tau_i >= 0. Batches are drawn with
replacement from

    q_i = (tau_i + eps)^alpha            (or rank-based: q_i = rank_i^-alpha)
    p   = (1 - lam) * q / sum(q) + lam / N

After each optimization step the sampled items' pheromones receive an
ACO-style deposit with evaporation (soft assignment, EMA):

    tau_i <- (1 - rho) * tau_i + rho * score_i

where score_i is the fresh per-sample loss or gradient magnitude. Un-sampled
items slowly "gather dust": they drift toward the running mean of recently
deposited scores at rate kappa, so no sample goes permanently stale.

Non-uniform sampling biases the empirical-risk gradient; PER-style importance
weights w_i = (N * p_i)^-beta / max_j w_j (beta annealed toward 1) correct it.
"""

import torch

SCORE_MEAN_EMA = 0.05  # smoothing for the running mean of deposited scores


def linear_beta(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    frac = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + (end - start) * frac


class PheromoneTable:
    def __init__(self, n: int, alpha: float = 0.6, lam: float = 0.05,
                 rho: float = 0.6, kappa: float = 1e-3, eps: float = 1e-6,
                 transform: str = "proportional", tau_init: float = 1.0,
                 update_rule: str = "ema", evap: float = 0.0):
        self.n = n
        self.alpha = alpha
        self.lam = lam
        self.rho = rho
        self.update_rule = update_rule
        self.evap = evap
        self.kappa = kappa
        self.eps = eps
        self.transform = transform
        self.tau = torch.full((n,), float(tau_init))
        self.initialized = torch.zeros(n, dtype=torch.bool)
        self.score_mean = float(tau_init)
        self._score_mean_set = False

    # ---------------------------------------------------------------- sampling
    def probabilities(self) -> torch.Tensor:
        if self.transform == "rank":
            # rank 1 = highest tau; heavy-tail robust (PER rank-based variant)
            order = torch.argsort(self.tau, descending=True)
            ranks = torch.empty(self.n)
            ranks[order] = torch.arange(1, self.n + 1, dtype=torch.float32)
            q = ranks.pow(-self.alpha)
        else:
            q = (self.tau + self.eps).pow(self.alpha)
        p = q / q.sum()
        if self.lam > 0:
            p = (1.0 - self.lam) * p + self.lam / self.n
        return p

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        p = self.probabilities()
        idx = torch.multinomial(p, batch_size, replacement=True, generator=generator)
        return idx, p

    def is_weights(self, idx: torch.Tensor, p: torch.Tensor, beta: float) -> torch.Tensor:
        w = (self.n * p[idx]).pow(-beta)
        return w / w.max()

    # ----------------------------------------------------------------- updates
    def update(self, idx: torch.Tensor, scores: torch.Tensor) -> None:
        """Deposit fresh scores on sampled items (duplicates reduced by mean),
        then apply staleness decay to everything that was not sampled."""
        scores = scores.detach().float().cpu()
        uniq, inverse = idx.unique(return_inverse=True)
        sums = torch.zeros(len(uniq)).scatter_add_(0, inverse, scores)
        counts = torch.zeros(len(uniq)).scatter_add_(0, inverse, torch.ones_like(scores))
        mean_scores = sums / counts

        fresh = ~self.initialized[uniq]
        if self.update_rule == "accumulate":
            # classic ACO: deposits pile up on repeatedly-hard samples;
            # evap = 0 means the colony never forgets (autocatalytic regime)
            deposited = (1.0 - self.evap) * self.tau[uniq] + mean_scores
        else:
            deposited = (1.0 - self.rho) * self.tau[uniq] + self.rho * mean_scores
        # first-ever measurement replaces the arbitrary tau_init outright
        self.tau[uniq] = torch.where(fresh, mean_scores, deposited)
        self.initialized[uniq] = True
        self._update_score_mean(mean_scores.mean().item())

        if self.kappa > 0:
            stale = torch.ones(self.n, dtype=torch.bool)
            stale[uniq] = False
            self.tau[stale] += self.kappa * (self.score_mean - self.tau[stale])

    def set_scores(self, idx: torch.Tensor, scores: torch.Tensor) -> None:
        """Hard assignment used by the warmup pass to initialize tau."""
        scores = scores.detach().float().cpu()
        self.tau[idx] = scores
        self.initialized[idx] = True
        self._update_score_mean(scores.mean().item())

    def _update_score_mean(self, batch_mean: float) -> None:
        if not self._score_mean_set:
            self.score_mean = batch_mean
            self._score_mean_set = True
        else:
            self.score_mean += SCORE_MEAN_EMA * (batch_mean - self.score_mean)

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        p = self.probabilities()
        entropy = -(p * p.clamp_min(1e-12).log()).sum().item()
        ess = 1.0 / (p.pow(2).sum().item() * self.n)
        q = torch.quantile(self.tau, torch.tensor([0.5, 0.95]))
        return {
            "tau_mean": self.tau.mean().item(),
            "tau_p50": q[0].item(),
            "tau_p95": q[1].item(),
            "tau_max": self.tau.max().item(),
            "entropy_norm": entropy / torch.log(torch.tensor(float(self.n))).item(),
            "ess_norm": ess,
            "min_prob": p.min().item(),
        }

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "initialized": self.initialized,
            "score_mean": self.score_mean,
            "_score_mean_set": self._score_mean_set,
        }

    def load_state_dict(self, state: dict) -> None:
        self.tau = state["tau"].clone()
        self.initialized = state["initialized"].clone()
        self.score_mean = state["score_mean"]
        self._score_mean_set = state["_score_mean_set"]
