import torch

from smart_batches.config import DataConfig, load_config
from smart_batches.data import build_data


def synthetic_cfg(**kwargs) -> DataConfig:
    defaults = dict(dataset="synthetic", subset_train=512, subset_test=128)
    defaults.update(kwargs)
    return DataConfig(**defaults)


def test_synthetic_shapes_and_index_stability():
    train, test = build_data(synthetic_cfg())
    assert train.x.shape == (512, 1, 28, 28)
    assert len(test) == 128
    idx = torch.tensor([3, 100, 3])
    x1, y1 = train.fetch(idx)
    x2, y2 = train.fetch(idx)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(x1[0], x1[2])  # duplicate index -> identical sample


def test_build_is_deterministic():
    a, _ = build_data(synthetic_cfg())
    b, _ = build_data(synthetic_cfg())
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)


def test_label_noise_flips_exact_fraction_to_different_class():
    clean, _ = build_data(synthetic_cfg())
    noisy, _ = build_data(synthetic_cfg(label_noise=0.25, noise_seed=3))
    assert int(noisy.corrupted.sum()) == round(0.25 * 512)
    assert (noisy.y[noisy.corrupted] != clean.y[noisy.corrupted]).all()
    assert (noisy.y[~noisy.corrupted] == clean.y[~noisy.corrupted]).all()


def test_vectorized_augmentation_matches_naive_loop():
    import torch.nn.functional as F

    from smart_batches.data import TensorData

    torch.manual_seed(0)
    x = torch.rand(16, 3, 32, 32)
    y = torch.zeros(16, dtype=torch.long)
    data = TensorData(x, y, 10, augment_pad=4, flip=True)

    gen_a = torch.Generator().manual_seed(42)
    out = data._augment(x.clone(), gen_a)

    # naive reference with an identically-seeded generator
    gen_b = torch.Generator().manual_seed(42)
    pad = 4
    padded = F.pad(x, (pad,) * 4)
    offs = torch.randint(0, 2 * pad + 1, (16, 2), generator=gen_b)
    ref = torch.stack([padded[k, :, offs[k, 0]:offs[k, 0] + 32,
                              offs[k, 1]:offs[k, 1] + 32] for k in range(16)])
    flip_mask = torch.rand(16, generator=gen_b) < 0.5
    ref[flip_mask] = torch.flip(ref[flip_mask], dims=[3])

    assert torch.equal(out, ref)


def test_config_overrides_and_unknown_key_rejection():
    cfg = load_config(base={"data": {"dataset": "synthetic"}},
                      overrides=["train.batch_size=32",
                                 "sampling.pheromone.rho=1.0"])
    assert cfg.train.batch_size == 32
    assert cfg.sampling.pheromone.rho == 1.0

    try:
        load_config(base={"train": {"batch_sizze": 32}})
        raise AssertionError("expected KeyError for unknown config key")
    except KeyError:
        pass
