"""Mechanism diagnostics — chiefly the gradient-variance probe.

The research claim is that pheromone-guided batches yield more informative,
stable gradient signals. The probe measures exactly that: with the model
frozen, draw M batches from the *current* sampling policy, compute the
(importance-weighted) loss gradient for each without stepping, and report the
spread across batches. Dropout is disabled (eval mode) so the measured
variance is attributable to batch selection, not regularization noise.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def _flatten_grads(grads) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads]).float().cpu()


def probe_gradient_variance(model, data, sampler, device,
                            m_batches: int = 8, seed: int = 0,
                            step: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]

    grads = []
    for _ in range(m_batches):
        idx, w = sampler.peek_batch(generator, step)
        x, y = data.fetch(idx, train=False)  # no augmentation: isolate selection noise
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        loss = (w.to(device) * losses).mean()
        g = torch.autograd.grad(loss, params)
        grads.append(_flatten_grads(g))
    model.train(was_training)

    stacked = torch.stack(grads)                 # (M, D)
    g_bar = stacked.mean(0)
    var = (stacked - g_bar).pow(2).sum(1).mean().item()   # E ||g - g_bar||^2
    mean_sq = g_bar.pow(2).sum().item()
    return {
        "probe/grad_var": var,
        "probe/grad_var_rel": var / max(mean_sq, 1e-12),
        "probe/grad_norm_sq_mean": mean_sq,
    }


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    rank = lambda v: v.argsort().argsort().float()
    return _pearson(rank(a), rank(b))


def probe_score_correlation(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    """How faithfully do the cheap pheromone scores (per-sample loss, last-layer
    grad proxy) track the exact full per-sample gradient magnitude?

    Runs on an eval-mode CPU copy of the model so all three scores are computed
    on identical deterministic forwards (torch.func is unsupported on MPS)."""
    import copy

    from .scoring import ExactGradScorer, LastLayerGradScorer

    cpu_model = copy.deepcopy(model).cpu()
    cpu_model.eval()
    x, y = x.detach().cpu(), y.detach().cpu()

    proxy_scorer = LastLayerGradScorer(cpu_model)
    with torch.no_grad():
        logits = cpu_model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
    proxy = proxy_scorer(x, y, logits, losses)
    proxy_scorer.remove()
    exact = ExactGradScorer(cpu_model)(x, y, None, None)

    return {
        "corr/loss_exact_pearson": _pearson(losses, exact),
        "corr/loss_exact_spearman": _spearman(losses, exact),
        "corr/proxy_exact_pearson": _pearson(proxy, exact),
        "corr/proxy_exact_spearman": _spearman(proxy, exact),
    }
