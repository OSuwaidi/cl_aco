import torch
import torch.nn as nn
import torch.nn.functional as F

from smart_batches.models import SmallCNN
from smart_batches.scoring import ExactGradScorer, LastLayerGradScorer, make_scorer


def tiny_mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))


def per_sample_last_layer_sq_norms(model, last, x, y):
    """Reference: loop singleton backwards, sum squared grads of the last layer."""
    out = []
    for i in range(len(x)):
        model.zero_grad()
        loss = F.cross_entropy(model(x[i:i + 1]), y[i:i + 1])
        loss.backward()
        out.append((last.weight.grad.pow(2).sum() + last.bias.grad.pow(2).sum()).item())
    model.zero_grad()
    return torch.tensor(out)


def per_sample_full_sq_norms(model, x, y):
    out = []
    for i in range(len(x)):
        model.zero_grad()
        loss = F.cross_entropy(model(x[i:i + 1]), y[i:i + 1])
        loss.backward()
        out.append(sum(p.grad.pow(2).sum().item() for p in model.parameters()))
    model.zero_grad()
    return torch.tensor(out)


def test_proxy_equals_exact_last_layer_grad_norm():
    model = tiny_mlp()
    x, y = torch.randn(6, 5), torch.randint(0, 3, (6,))
    scorer = LastLayerGradScorer(model)
    logits = model(x)
    losses = F.cross_entropy(logits, y, reduction="none")
    proxy = scorer(x, y, logits, losses)
    reference = per_sample_last_layer_sq_norms(model, model[2], x, y)
    assert torch.allclose(proxy, reference, atol=1e-5)


def test_exact_scorer_matches_manual_loop():
    model = tiny_mlp()
    x, y = torch.randn(6, 5), torch.randint(0, 3, (6,))
    scorer = ExactGradScorer(model)
    exact = scorer(x, y, None, None)
    reference = per_sample_full_sq_norms(model, x, y)
    assert torch.allclose(exact, reference, atol=1e-5)


def test_proxy_lower_bounds_exact_on_small_cnn():
    torch.manual_seed(1)
    model = SmallCNN(1, 10)
    model.eval()  # consistent dropout behavior between proxy capture and exact
    x, y = torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
    proxy_scorer = LastLayerGradScorer(model)
    logits = model(x)
    losses = F.cross_entropy(logits, y, reduction="none")
    proxy = proxy_scorer(x, y, logits, losses)
    exact = ExactGradScorer(model)(x, y, logits, losses)
    # last-layer squared norm is one non-negative term of the full squared norm
    assert (proxy <= exact + 1e-4).all()
    assert (proxy > 0).all()


def test_score_correlation_probe():
    from smart_batches.metrics import probe_score_correlation

    torch.manual_seed(2)
    model = tiny_mlp()
    x, y = torch.randn(32, 5), torch.randint(0, 3, (32,))
    out = probe_score_correlation(model, x, y)
    assert set(out) == {"corr/loss_exact_pearson", "corr/loss_exact_spearman",
                        "corr/proxy_exact_pearson", "corr/proxy_exact_spearman"}
    for v in out.values():
        assert -1.0 <= v <= 1.0
    # on a 2-layer MLP the last layer dominates the gradient norm
    assert out["corr/proxy_exact_spearman"] > 0.8


def test_loss_scorer_returns_detached_losses():
    scorer = make_scorer("loss", None)
    losses = torch.tensor([0.5, 2.0], requires_grad=True)
    scores = scorer(None, None, None, losses)
    assert not scores.requires_grad
    assert torch.allclose(scores, torch.tensor([0.5, 2.0]))
