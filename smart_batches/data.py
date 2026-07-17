"""Tensor-backed datasets with fast arbitrary-index batch fetch.

Adaptive with-replacement sampling needs random access to arbitrary index sets
whose choice depends on the *previous* optimization step, so DataLoader-style
sequential prefetching cannot help. Instead every dataset is materialized as
one device tensor and batches are gathered by indexing; CIFAR-style
augmentation is applied vectorized per batch at fetch time.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


@dataclass
class TensorData:
    """A split held fully in memory. `x` is normalized float32 unless
    `augment_pad` > 0, in which case it is raw uint8 normalized at fetch."""

    x: torch.Tensor                  # (N, C, H, W)
    y: torch.Tensor                  # (N,) int64
    n_classes: int
    corrupted: torch.Tensor = field(default=None)  # bool (N,) — label-noise mask
    augment_pad: int = 0             # random-crop padding; 0 disables augmentation
    flip: bool = False
    mean: torch.Tensor = field(default=None)       # (C,1,1) for uint8 path
    std: torch.Tensor = field(default=None)

    def __post_init__(self):
        if self.corrupted is None:
            self.corrupted = torch.zeros(len(self.y), dtype=torch.bool)

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def in_channels(self) -> int:
        return self.x.shape[1]

    def to(self, device: torch.device) -> "TensorData":
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self

    def fetch(self, idx: torch.Tensor, train: bool = True,
              generator: torch.Generator | None = None):
        idx = idx.to(self.x.device)
        xb, yb = self.x[idx], self.y[idx]
        if xb.dtype == torch.uint8:
            xb = xb.float().div_(255.0).sub_(self.mean).div_(self.std)
        if train and self.augment_pad > 0:
            xb = self._augment(xb, generator)
        return xb, yb

    def _augment(self, xb: torch.Tensor, generator: torch.Generator | None):
        b, c, h, w = xb.shape
        pad = self.augment_pad
        device = xb.device
        padded = F.pad(xb, (pad, pad, pad, pad))
        # random-crop offsets and flip mask are drawn on CPU for determinism,
        # then the crops are gathered in one advanced-indexing op (GPU-friendly)
        offs = torch.randint(0, 2 * pad + 1, (b, 2), generator=generator).to(device)
        bi = torch.arange(b, device=device).view(b, 1, 1, 1)
        ci = torch.arange(c, device=device).view(1, c, 1, 1)
        ri = offs[:, 0].view(b, 1, 1, 1) + torch.arange(h, device=device).view(1, 1, h, 1)
        cj = offs[:, 1].view(b, 1, 1, 1) + torch.arange(w, device=device).view(1, 1, 1, w)
        out = padded[bi, ci, ri, cj]
        if self.flip:
            flip_mask = (torch.rand(b, generator=generator) < 0.5).to(device)
            out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
        return out

    def iter_eval(self, batch_size: int = 512):
        for start in range(0, len(self), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(self)))
            yield self.fetch(idx, train=False)


def _subset(x: torch.Tensor, y: torch.Tensor, k: int, seed: int):
    if k <= 0 or k >= len(y):
        return x, y
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    keep = perm[:k]
    return x[keep], y[keep]


def _apply_label_noise(y: torch.Tensor, fraction: float, n_classes: int, seed: int):
    """Flip `fraction` of labels to a different uniformly-random class."""
    corrupted = torch.zeros(len(y), dtype=torch.bool)
    if fraction <= 0:
        return y, corrupted
    gen = torch.Generator().manual_seed(seed)
    n_flip = int(round(fraction * len(y)))
    flip_idx = torch.randperm(len(y), generator=gen)[:n_flip]
    # shift by 1..n_classes-1 guarantees the new label differs from the old one
    shift = torch.randint(1, n_classes, (n_flip,), generator=gen)
    y = y.clone()
    y[flip_idx] = (y[flip_idx] + shift) % n_classes
    corrupted[flip_idx] = True
    return y, corrupted


def _synthetic(n: int, means_seed: int, sample_seed: int, n_classes: int = 10):
    """Gaussian class-mean blobs shaped like 1x28x28 images — learnable, no
    download. Class means depend only on means_seed so train/test splits share
    the same underlying task."""
    means = torch.randn(n_classes, 1, 28, 28,
                        generator=torch.Generator().manual_seed(means_seed))
    gen = torch.Generator().manual_seed(sample_seed)
    y = torch.randint(0, n_classes, (n,), generator=gen)
    x = means[y] + 0.5 * torch.randn(n, 1, 28, 28, generator=gen)
    return x, y


def build_data(cfg) -> tuple[TensorData, TensorData]:
    """Returns (train, test) TensorData on CPU; caller moves them to device."""
    name = cfg.dataset
    if name == "synthetic":
        n_train = cfg.subset_train or 2048
        n_test = cfg.subset_test or 512
        xtr, ytr = _synthetic(n_train, cfg.subset_seed, cfg.subset_seed + 1)
        xte, yte = _synthetic(n_test, cfg.subset_seed, cfg.subset_seed + 2)
        n_classes = 10
        train = TensorData(xtr, ytr, n_classes)
        test = TensorData(xte, yte, n_classes)
    elif name in ("mnist", "fashion_mnist"):
        import torchvision

        cls = {"mnist": torchvision.datasets.MNIST,
               "fashion_mnist": torchvision.datasets.FashionMNIST}[name]
        mean, std = STATS[name]
        splits = []
        for is_train in (True, False):
            ds = cls(cfg.root, train=is_train, download=True)
            x = ds.data.unsqueeze(1).float().div_(255.0).sub_(mean[0]).div_(std[0])
            splits.append((x, ds.targets.long()))
        (xtr, ytr), (xte, yte) = splits
        xtr, ytr = _subset(xtr, ytr, cfg.subset_train, cfg.subset_seed)
        xte, yte = _subset(xte, yte, cfg.subset_test, cfg.subset_seed + 1)
        train = TensorData(xtr, ytr, 10)
        test = TensorData(xte, yte, 10)
    elif name == "cifar10":
        import torchvision

        mean, std = STATS[name]
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        splits = []
        for is_train in (True, False):
            ds = torchvision.datasets.CIFAR10(cfg.root, train=is_train, download=True)
            x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
            splits.append((x, torch.tensor(ds.targets, dtype=torch.long)))
        (xtr, ytr), (xte, yte) = splits
        xtr, ytr = _subset(xtr, ytr, cfg.subset_train, cfg.subset_seed)
        xte, yte = _subset(xte, yte, cfg.subset_test, cfg.subset_seed + 1)
        train = TensorData(xtr, ytr, 10, augment_pad=4, flip=True, mean=mean_t, std=std_t)
        test = TensorData(xte, yte, 10, mean=mean_t, std=std_t)
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    train.y, train.corrupted = _apply_label_noise(
        train.y, cfg.label_noise, train.n_classes, cfg.noise_seed)
    return train, test
