"""Checks of the swarm. Run with ``python test_pso.py`` or ``pytest``."""
import copy

import torch
from torch import nn

from pso import Swarm


def regression(n=40, seed=0):
    """A small convex problem: fit y = A x with a linear layer."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.3
    return nn.Linear(3, 1), nn.MSELoss(), x, y


def test_one_iteration_is_the_update_rule():
    model, loss_fn, x, y = regression()
    swarm = Swarm(model, loss_fn, n_particles=7, w=0.6, c1=1.3, c2=1.7, seed=3)
    swarm.fit(x, y, iterations=0)  # evaluates the starting positions
    X, V, P, G = swarm.X.clone(), swarm.V.clone(), swarm.P.clone(), swarm.G.clone()
    gen = copy.deepcopy(swarm.gen)  # the next random numbers of the swarm
    r1 = torch.rand(7, swarm.dim, generator=gen)
    r2 = torch.rand(7, swarm.dim, generator=gen)

    swarm.fit(x, y, iterations=1)

    V_expected = 0.6 * V + 1.3 * r1 * (P - X) + 1.7 * r2 * (G - X)
    assert torch.allclose(swarm.V, V_expected)
    assert torch.allclose(swarm.X, X + V_expected)


def test_random_numbers_are_different_for_every_weight():
    model, loss_fn, x, y = regression()
    swarm = Swarm(model, loss_fn, n_particles=5, w=0.0, c1=0.0, c2=1.0, seed=1)
    swarm.fit(x, y, iterations=0)
    X, G = swarm.X.clone(), swarm.G.clone()
    best = int(swarm.pf.argmin())
    swarm.fit(x, y, iterations=1)
    i = 1 if best == 0 else 0  # a particle that is not the best one, so that G - X is not zero
    r2 = swarm.V[i] / (G - X[i])  # with w = 0 and c1 = 0: V = r2 (G - X)
    assert r2.min() >= 0 and r2.max() <= 1
    assert r2.std() > 0.1  # a single scalar for all the weights would make all the entries equal


def test_scalar_random_numbers_option():
    model, loss_fn, x, y = regression()
    swarm = Swarm(model, loss_fn, n_particles=5, w=0.0, c1=0.0, c2=1.0, per_weight=False, seed=1)
    swarm.fit(x, y, iterations=0)
    X, G = swarm.X.clone(), swarm.G.clone()
    best = int(swarm.pf.argmin())
    swarm.fit(x, y, iterations=1)
    i = 1 if best == 0 else 0
    r2 = swarm.V[i] / (G - X[i])
    assert r2.std() < 1e-5  # one number for all the weights of the particle


def test_best_loss_never_increases_and_matches_the_best_model():
    model, loss_fn, x, y = regression()
    swarm = Swarm(model, loss_fn, n_particles=20, seed=0)
    history = swarm.fit(x, y, iterations=150)
    assert all(b <= a + 1e-12 for a, b in zip(history, history[1:]))
    assert history[-1] < 0.1 * history[0]
    with torch.no_grad():
        assert abs(loss_fn(swarm.best_model()(x), y).item() - swarm.best_loss) < 1e-6


def test_seed_reproducibility_and_global_state_untouched():
    model, loss_fn, x, y = regression()
    state = torch.get_rng_state()
    a = Swarm(model, loss_fn, n_particles=10, seed=5)
    a.fit(x, y, iterations=30)
    assert torch.equal(torch.get_rng_state(), state)  # the swarm does not use the global generator
    b = Swarm(model, loss_fn, n_particles=10, seed=5)
    b.fit(x, y, iterations=30)
    c = Swarm(model, loss_fn, n_particles=10, seed=6)
    c.fit(x, y, iterations=30)
    assert a.history == b.history and a.history != c.history


def test_the_given_model_is_not_modified_and_is_particle_zero():
    model, loss_fn, x, y = regression()
    before = [p.clone() for p in model.parameters()]
    swarm = Swarm(model, loss_fn, n_particles=6, seed=0)
    start = swarm.X[0].clone()
    swarm.fit(x, y, iterations=20)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    assert torch.equal(torch.cat([p.detach().flatten() for p in before]), start)
    assert not torch.equal(swarm.X[1], start)  # the others start from other weights


def test_vmax_clips_the_velocity():
    model, loss_fn, x, y = regression()
    swarm = Swarm(model, loss_fn, n_particles=6, vmax=0.05, seed=0)
    swarm.fit(x, y, iterations=20)
    assert swarm.V.abs().max() <= 0.05 + 1e-7


def test_diagnostics_and_callback():
    model, loss_fn, x, y = regression()
    swarm = Swarm(model, loss_fn, n_particles=10, seed=0)
    calls = []
    swarm.fit(x, y, iterations=25, callback=lambda s: calls.append(s.best_loss))
    assert len(calls) == 26 and calls == swarm.history  # at the start and after every iteration
    for key, values in swarm.stats.items():
        assert len(values) == 26, key
    st = swarm.stats
    assert all(a <= b <= c for a, b, c in zip(st["loss_min"], st["loss_median"], st["loss_max"]))
    assert all(d > 0 for d in st["diversity"]) and all(0 <= f <= 1 for f in st["improved"][1:])
    assert st["loss_min"][-1] >= swarm.best_loss - 1e-12  # the best position is at least as good as any current one


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok  ", name)
