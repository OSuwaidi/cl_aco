"""Batch-selection strategies behind one interface.

next_batch(step) -> (idx CPU int64 (B,), weights CPU float32 (B,))
peek_batch(gen)  -> same, drawn from the *current* policy without mutating any
                    sampler state (used by the gradient-variance probe).
"""

import torch

from .pheromones import PheromoneTable, linear_beta


class BaseSampler:
    def next_batch(self, step: int):
        raise NotImplementedError

    def peek_batch(self, generator: torch.Generator, step: int | None = None):
        raise NotImplementedError

    def update(self, idx: torch.Tensor, scores: torch.Tensor) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class EpochShuffleSampler(BaseSampler):
    """Standard baseline: shuffled passes over the data, fixed batch size
    (the tail of a permutation is topped up from the next shuffle so every
    strategy sees identical batch sizes)."""

    def __init__(self, n: int, batch_size: int, generator: torch.Generator):
        self.n = n
        self.batch_size = batch_size
        self.generator = generator
        self.perm = torch.randperm(n, generator=generator)
        self.pos = 0

    def next_batch(self, step: int):
        take = []
        need = self.batch_size
        while need > 0:
            if self.pos >= self.n:
                self.perm = torch.randperm(self.n, generator=self.generator)
                self.pos = 0
            chunk = self.perm[self.pos:self.pos + need]
            take.append(chunk)
            self.pos += len(chunk)
            need -= len(chunk)
        idx = torch.cat(take)
        return idx, torch.ones(len(idx))

    def peek_batch(self, generator: torch.Generator, step: int | None = None):
        idx = torch.randperm(self.n, generator=generator)[:self.batch_size]
        return idx, torch.ones(len(idx))

    def state_dict(self) -> dict:
        return {"perm": self.perm, "pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        self.perm = state["perm"].clone()
        self.pos = state["pos"]


class UniformReplacementSampler(BaseSampler):
    """Control arm: with-replacement like the pheromone sampler, but uniform —
    isolates the effect of adaptive probabilities from replacement itself."""

    def __init__(self, n: int, batch_size: int, generator: torch.Generator):
        self.n = n
        self.batch_size = batch_size
        self.generator = generator

    def next_batch(self, step: int):
        idx = torch.randint(0, self.n, (self.batch_size,), generator=self.generator)
        return idx, torch.ones(self.batch_size)

    def peek_batch(self, generator: torch.Generator, step: int | None = None):
        idx = torch.randint(0, self.n, (self.batch_size,), generator=generator)
        return idx, torch.ones(self.batch_size)


class PheromoneWORSampler(BaseSampler):
    """Pheromone-prioritized sampling WITHOUT replacement.

    Each batch is drawn from the *remaining* samples of the current cycle in
    proportion to their current pheromones, so one cycle serves every distinct
    sample exactly once (hard samples early, easy ones late) and the ordering
    keeps adapting mid-cycle as pheromones update. The strategy reorders the
    data stream rather than reweighting it: over a full cycle the objective is
    identical to a uniform epoch, so batch weights are always 1 and
    `is_correction` does not apply. Cycle tails are topped up from a fresh
    cycle, mirroring EpochShuffleSampler."""

    def __init__(self, table: PheromoneTable, batch_size: int,
                 generator: torch.Generator, total_steps: int,
                 alpha_end: float | None = None):
        self.table = table
        self.batch_size = batch_size
        self.generator = generator
        self.total_steps = total_steps
        self.alpha_start = table.alpha
        self.alpha_end = alpha_end
        self.remaining = torch.ones(table.n, dtype=torch.bool)

    def _draw(self, k: int, mask: torch.Tensor, generator: torch.Generator):
        weights = self.table.probabilities() * mask
        return torch.multinomial(weights, k, replacement=False, generator=generator)

    def next_batch(self, step: int):
        if self.alpha_end is not None:
            self.table.alpha = linear_beta(step, self.total_steps,
                                           self.alpha_start, self.alpha_end)
        take = []
        need = self.batch_size
        while need > 0:
            available = int(self.remaining.sum())
            if available == 0:
                self.remaining.fill_(True)
                available = self.table.n
            k = min(need, available)
            idx = self._draw(k, self.remaining.float(), self.generator)
            self.remaining[idx] = False
            take.append(idx)
            need -= k
        idx = torch.cat(take)
        return idx, torch.ones(len(idx))

    def peek_batch(self, generator: torch.Generator, step: int | None = None):
        mask = (self.remaining if int(self.remaining.sum()) >= self.batch_size
                else torch.ones_like(self.remaining))
        idx = self._draw(self.batch_size, mask.float(), generator)
        return idx, torch.ones(self.batch_size)

    def update(self, idx: torch.Tensor, scores: torch.Tensor) -> None:
        self.table.update(idx, scores)

    def state_dict(self) -> dict:
        return {"table": self.table.state_dict(), "remaining": self.remaining}

    def load_state_dict(self, state: dict) -> None:
        self.table.load_state_dict(state["table"])
        self.remaining = state["remaining"].clone()


class PheromoneSampler(BaseSampler):
    def __init__(self, table: PheromoneTable, batch_size: int,
                 generator: torch.Generator, is_correction: str,
                 beta_start: float, beta_end: float, total_steps: int,
                 alpha_end: float | None = None):
        self.table = table
        self.batch_size = batch_size
        self.generator = generator
        self.is_correction = is_correction
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self.alpha_start = table.alpha
        self.alpha_end = alpha_end  # anneal aggressive -> gentle (or uniform at 0)
        self.last_p: torch.Tensor | None = None

    def _weights(self, idx: torch.Tensor, p: torch.Tensor, step: int) -> torch.Tensor:
        if self.is_correction != "per":
            return torch.ones(len(idx))
        beta = linear_beta(step, self.total_steps, self.beta_start, self.beta_end)
        return self.table.is_weights(idx, p, beta)

    def next_batch(self, step: int):
        if self.alpha_end is not None:
            self.table.alpha = linear_beta(step, self.total_steps,
                                           self.alpha_start, self.alpha_end)
        idx, p = self.table.sample(self.batch_size, self.generator)
        self.last_p = p
        return idx, self._weights(idx, p, step)

    def peek_batch(self, generator: torch.Generator, step: int | None = None):
        idx, p = self.table.sample(self.batch_size, generator)
        return idx, self._weights(idx, p, self.total_steps if step is None else step)

    def update(self, idx: torch.Tensor, scores: torch.Tensor) -> None:
        self.table.update(idx, scores)

    def state_dict(self) -> dict:
        return {"table": self.table.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.table.load_state_dict(state["table"])
