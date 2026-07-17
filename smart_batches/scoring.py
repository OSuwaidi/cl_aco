"""Per-sample difficulty scorers that feed the pheromone update.

- loss:             per-sample cross-entropy (free — already computed).
- grad_norm_proxy:  exact squared gradient norm of the final linear layer,
                    computed from the training forward pass without an extra
                    backward:  s_i = ||softmax(z_i) - y_i||^2 * (||h_i||^2 + 1)
                    where h_i is the penultimate activation (captured by a
                    forward hook) and the +1 accounts for the bias gradient.
                    Katharopoulos & Fleuret-style bound on the full grad norm.
- grad_norm_exact:  full per-sample squared gradient norm via
                    torch.func.vmap(grad(...)). Runs its own forward in eval
                    mode; intended for small models / validation of the proxy
                    (unsupported on MPS — use CPU or CUDA).

All scorers return detached, non-negative float32 CPU tensors of shape (B,).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise ValueError("Model has no nn.Linear layer for the grad-norm proxy")
    return last


class LossScorer:
    name = "loss"

    def __call__(self, x, y, logits, losses) -> torch.Tensor:
        return losses.detach().float().cpu()


class LastLayerGradScorer:
    """Assumes the last nn.Linear produces the logits fed to cross-entropy
    (true for SmallCNN and ResNet). Must be called right after the training
    forward pass — the hook stores that pass's activations."""

    name = "grad_norm_proxy"

    def __init__(self, model: nn.Module):
        self._h = None
        self._z = None
        self._handle = find_last_linear(model).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._h = inputs[0].detach()
        self._z = output.detach()

    def __call__(self, x, y, logits, losses) -> torch.Tensor:
        if self._z is None:
            raise RuntimeError("No forward pass captured before scoring")
        with torch.no_grad():
            g = F.softmax(self._z, dim=1)
            g[torch.arange(len(y), device=g.device), y] -= 1.0  # softmax(z) - onehot
            s = g.pow(2).sum(1) * (self._h.pow(2).sum(1) + 1.0)
        return s.float().cpu()

    def remove(self):
        self._handle.remove()


class ExactGradScorer:
    name = "grad_norm_exact"

    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self, x, y, logits, losses) -> torch.Tensor:
        from torch.func import functional_call, grad, vmap

        model = self.model
        was_training = model.training
        model.eval()  # deterministic scoring: no dropout, BN running stats
        try:
            params = {k: v.detach() for k, v in model.named_parameters()}
            buffers = {k: v.detach() for k, v in model.named_buffers()}

            def sample_loss(params, buffers, xi, yi):
                out = functional_call(model, (params, buffers), (xi.unsqueeze(0),))
                return F.cross_entropy(out, yi.unsqueeze(0))

            grads = vmap(grad(sample_loss), in_dims=(None, None, 0, 0))(
                params, buffers, x, y)
            sq = torch.zeros(len(y), device=x.device)
            for g in grads.values():
                sq += g.detach().flatten(1).pow(2).sum(1)
        finally:
            model.train(was_training)
        return sq.float().cpu()


def make_scorer(name: str, model: nn.Module):
    if name == "loss":
        return LossScorer()
    if name == "grad_norm_proxy":
        return LastLayerGradScorer(model)
    if name == "grad_norm_exact":
        return ExactGradScorer(model)
    raise ValueError(f"Unknown scorer {name!r}")
