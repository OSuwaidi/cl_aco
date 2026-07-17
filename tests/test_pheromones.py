import torch

from smart_batches.pheromones import PheromoneTable, linear_beta


def test_initial_distribution_is_uniform():
    table = PheromoneTable(100, lam=0.0)
    p = table.probabilities()
    assert torch.allclose(p, torch.full((100,), 0.01), atol=1e-7)


def test_alpha_zero_is_uniform_regardless_of_tau():
    table = PheromoneTable(50, alpha=0.0, lam=0.0)
    table.tau = torch.rand(50) * 10
    p = table.probabilities()
    assert torch.allclose(p, torch.full((50,), 1 / 50), atol=1e-6)


def test_probabilities_proportional_to_tau_alpha():
    table = PheromoneTable(4, alpha=1.0, lam=0.0, eps=0.0)
    table.tau = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = table.probabilities()
    assert torch.allclose(p, table.tau / table.tau.sum(), atol=1e-7)

    table.alpha = 0.5
    q = table.tau.sqrt()
    assert torch.allclose(table.probabilities(), q / q.sum(), atol=1e-7)


def test_empirical_sampling_frequencies_match_p():
    torch.manual_seed(0)
    table = PheromoneTable(50)
    table.tau = torch.rand(50) + 0.1
    p = table.probabilities()
    gen = torch.Generator().manual_seed(123)
    idx, _ = table.sample(200_000, generator=gen)
    freq = torch.bincount(idx, minlength=50).float() / 200_000
    assert (freq - p).abs().max().item() < 0.01


def test_lambda_floor_bounds_min_probability():
    table = PheromoneTable(100, lam=0.1)
    table.tau = torch.zeros(100)
    table.tau[0] = 1e6  # extreme concentration
    p = table.probabilities()
    assert p.min().item() >= 0.1 / 100 - 1e-9
    assert abs(p.sum().item() - 1.0) < 1e-5


def test_rank_transform_is_monotonic_in_tau():
    table = PheromoneTable(10, transform="rank", lam=0.0)
    table.tau = torch.arange(10, dtype=torch.float32)
    p = table.probabilities()
    assert (p[1:] > p[:-1]).all()  # higher tau -> higher probability


def test_update_ema_math_and_duplicate_mean():
    table = PheromoneTable(5, rho=0.5, kappa=0.0)
    table.tau = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    table.initialized.fill_(True)
    idx = torch.tensor([0, 1, 1])
    scores = torch.tensor([2.0, 4.0, 6.0])  # sample 1 duplicated: mean = 5
    table.update(idx, scores)
    assert torch.allclose(table.tau[:2], torch.tensor([1.5, 3.5]))
    assert torch.allclose(table.tau[2:], torch.tensor([3.0, 4.0, 5.0]))


def test_first_measurement_replaces_tau_init():
    table = PheromoneTable(3, rho=0.5, kappa=0.0, tau_init=1.0)
    table.update(torch.tensor([0]), torch.tensor([10.0]))
    assert table.tau[0].item() == 10.0  # not EMA-blended with the arbitrary init
    assert table.initialized[0] and not table.initialized[1]


def test_kappa_pulls_unsampled_toward_score_mean():
    table = PheromoneTable(3, rho=1.0, kappa=0.1, tau_init=1.0)
    table.initialized.fill_(True)
    table.update(torch.tensor([0]), torch.tensor([10.0]))
    # score_mean becomes 10 on first deposit; others drift toward it
    expected = 1.0 + 0.1 * (10.0 - 1.0)
    assert torch.allclose(table.tau[1:], torch.tensor([expected, expected]))
    assert table.tau[0].item() == 10.0


def test_accumulate_rule_piles_up_deposits():
    table = PheromoneTable(3, update_rule="accumulate", evap=0.0, kappa=0.0)
    table.update(torch.tensor([0]), torch.tensor([2.0]))  # fresh: tau = 2
    table.update(torch.tensor([0]), torch.tensor([3.0]))  # accumulate: 2 + 3
    table.update(torch.tensor([0]), torch.tensor([1.0]))  # 5 + 1
    assert table.tau[0].item() == 6.0
    assert table.tau[1].item() == 1.0  # untouched (tau_init)


def test_accumulate_rule_with_evaporation():
    table = PheromoneTable(2, update_rule="accumulate", evap=0.5, kappa=0.0)
    table.initialized.fill_(True)
    table.tau = torch.tensor([4.0, 4.0])
    table.update(torch.tensor([0]), torch.tensor([1.0]))
    assert table.tau[0].item() == 0.5 * 4.0 + 1.0
    assert table.tau[1].item() == 4.0


def test_is_weights_formula_and_normalization():
    table = PheromoneTable(4, lam=0.0, alpha=1.0, eps=0.0)
    table.tau = torch.tensor([1.0, 1.0, 2.0, 4.0])
    p = table.probabilities()
    idx = torch.tensor([0, 2, 3])
    beta = 0.5
    w = table.is_weights(idx, p, beta)
    expected = (4 * p[idx]).pow(-beta)
    expected = expected / expected.max()
    assert torch.allclose(w, expected)
    assert w.max().item() == 1.0
    # rarer samples get larger corrective weight
    assert w[0] > w[1] > w[2]


def test_state_dict_roundtrip():
    table = PheromoneTable(10)
    table.update(torch.arange(5), torch.rand(5))
    clone = PheromoneTable(10)
    clone.load_state_dict(table.state_dict())
    assert torch.equal(clone.tau, table.tau)
    assert torch.equal(clone.initialized, table.initialized)
    assert clone.score_mean == table.score_mean


def test_wor_sampler_covers_every_sample_exactly_once_per_cycle():
    from smart_batches.samplers import PheromoneWORSampler

    table = PheromoneTable(64)
    table.tau = torch.rand(64) + 0.1
    gen = torch.Generator().manual_seed(0)
    sampler = PheromoneWORSampler(table, 16, gen, total_steps=100)
    # two full cycles (8 batches of 16 over n=64): each sample drawn exactly twice
    draws = torch.cat([sampler.next_batch(s)[0] for s in range(8)])
    assert torch.equal(torch.bincount(draws, minlength=64),
                       torch.full((64,), 2, dtype=torch.long))
    # weights are always 1: reordering needs no importance correction
    _, w = sampler.next_batch(8)
    assert torch.equal(w, torch.ones(16))


def test_wor_sampler_serves_high_pheromone_samples_first():
    from smart_batches.samplers import PheromoneWORSampler

    table = PheromoneTable(100, alpha=1.0, lam=0.0)
    table.tau = torch.full((100,), 1e-3)
    table.tau[7] = 1e6  # overwhelmingly hard sample
    hits = 0
    for trial in range(20):
        gen = torch.Generator().manual_seed(trial)
        sampler = PheromoneWORSampler(table, 10, gen, total_steps=100)
        first_batch, _ = sampler.next_batch(0)
        hits += int(7 in first_batch.tolist())
    assert hits >= 19  # dominant pheromone -> served in the first batch


def test_wor_sampler_state_roundtrip():
    from smart_batches.samplers import PheromoneWORSampler

    table = PheromoneTable(32)
    gen = torch.Generator().manual_seed(0)
    sampler = PheromoneWORSampler(table, 8, gen, total_steps=10)
    sampler.next_batch(0)
    state = sampler.state_dict()

    clone = PheromoneWORSampler(PheromoneTable(32), 8, gen, total_steps=10)
    clone.load_state_dict(state)
    assert torch.equal(clone.remaining, sampler.remaining)
    assert int(clone.remaining.sum()) == 24


def test_alpha_annealing_in_sampler():
    from smart_batches.samplers import PheromoneSampler

    table = PheromoneTable(20, alpha=1.0)
    gen = torch.Generator().manual_seed(0)
    sampler = PheromoneSampler(table, 4, gen, "none", 0.4, 1.0,
                               total_steps=11, alpha_end=0.0)
    sampler.next_batch(0)
    assert table.alpha == 1.0
    sampler.next_batch(5)
    assert abs(table.alpha - 0.5) < 1e-6
    sampler.next_batch(10)
    assert table.alpha == 0.0  # fully uniform by the end


def test_linear_beta_schedule():
    assert linear_beta(0, 100, 0.4, 1.0) == 0.4
    assert linear_beta(99, 100, 0.4, 1.0) == 1.0
    mid = linear_beta(50, 101, 0.4, 1.0)
    assert abs(mid - 0.7) < 1e-6
