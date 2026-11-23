"""Figures of the swarm on the Iris example, in the style of the blog.

    python plots.py                  # all the figures (about 15 minutes)
    python plots.py convergence      # best training loss of the swarm against the iteration, for a few seeds
    python plots.py health           # diversity, velocity, losses of the particles: the default w, and w = 1
    python plots.py map              # final loss and stability of the swarm over the plane (c, w)
    python plots.py generalization   # training and test loss of the best model, and the size of its weights
    python plots.py stats            # 20 seeds, median and band, and swarm against Adam at equal wall time
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn

import example
from pso import Swarm
from style import ACCENT, INK, PAPER, new_figure, style_axes, style_legend

W0, C0 = 0.7298, 1.49618  # the defaults of the swarm


def data():
    iris = load_iris()
    x_train, x_test, l_train, l_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0,
                                                        stratify=iris.target)
    x_train, x_test = (torch.tensor(a, dtype=torch.float32) for a in (x_train, x_test))
    y_train, y_test = (nn.functional.one_hot(torch.tensor(a), 3).float() for a in (l_train, l_test))
    return x_train, y_train, x_test, y_test


def run_swarm(x, y, seed, iterations=example.ITERATIONS, n=example.N_PARTICLES, w=W0, c1=C0, c2=C0, callback=None):
    torch.manual_seed(seed)
    swarm = Swarm(example.build_model(), example.loss_fn, n_particles=n, w=w, c1=c1, c2=c2, seed=seed)
    swarm.fit(x, y, iterations=iterations, callback=callback)
    return swarm


def legend(ax, loc="upper right", **kw):
    style_legend(ax.legend(loc=loc, fancybox=False, framealpha=1, edgecolor=INK, facecolor=PAPER, borderpad=0.8,
                           fontsize=8, **kw))


def save(fig, name):
    fig.savefig(name, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", name)


# ----------------------------------------------------------------------------------------- convergence
def convergence(seeds=range(5)):
    x_train, y_train, x_test, y_test = data()
    model_b = example.backpropagation(x_train, y_train)
    adam_loss = example.evaluate(model_b, x_train, y_train)[0]
    fig = new_figure()
    ax = fig.add_subplot(111)
    style_axes(ax)
    for k, seed in enumerate(seeds):
        h = run_swarm(x_train, y_train, seed).history
        label = {0: "SWARM, SEED 0", 1: "SWARM, OTHER SEEDS"}.get(k)
        ax.plot(range(len(h)), h, color=ACCENT if k == 0 else INK, lw=2.2 if k == 0 else 1.2,
                alpha=1.0 if k == 0 else 0.55, zorder=4 if k == 0 else 3, label=label)
    ax.axhline(adam_loss, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=2, label="ADAM, BEST OF 5 RUNS")
    ax.set_yscale("log")
    ax.set_xlabel("ITERATION", fontweight="bold", labelpad=8)
    ax.set_ylabel("BEST TRAINING LOSS (MSE)", fontweight="bold", labelpad=8)
    ax.set_title(f"[ PSO ON IRIS ]  {example.N_PARTICLES} PARTICLES  4-4-4-3 NETWORK", fontweight="bold",
                 loc="left", pad=12, fontsize=10)
    legend(ax)
    fig.tight_layout()
    save(fig, "convergence_iris.png")


# ----------------------------------------------------------------------------------------- 1. health
def health(seed=0):
    """Diversity, velocity, current losses and fraction of improving particles, for the default w and for w = 1."""
    x_train, y_train, _, _ = data()
    runs = {"w = 0.73  (DEFAULT)": (ACCENT, run_swarm(x_train, y_train, seed)),
            "w = 1.0": (INK, run_swarm(x_train, y_train, seed, w=1.0))}
    fig = new_figure((11, 7.4))
    axes = fig.subplots(2, 2)
    panels = [("diversity", "DIVERSITY: RMS SPREAD OF THE WEIGHTS", True),
              ("velocity", "VELOCITY: RMS OF THE COMPONENTS", True),
              (None, "LOSS OF THE PARTICLES", True),
              ("improved", "FRACTION OF PARTICLES THAT IMPROVE (10-ITERATION MEAN)", False)]
    for ax, (key, title, logy) in zip(axes.ravel(), panels):
        style_axes(ax)
        for name, (color, sw) in runs.items():
            st = sw.stats
            if key == "improved":
                v = np.array(st["improved"][1:])
                v = np.convolve(v, np.ones(10) / 10, mode="valid")
                ax.plot(np.arange(10, 10 + len(v)), v, color=color, lw=1.8, label=name)
            elif key is None:
                ax.plot(sw.history, color=color, lw=1.8, ls=(0, (4, 3)), label=name + ", BEST")
                ax.plot(st["loss_median"], color=color, lw=1.8, label=name + ", MEDIAN")
                ax.plot(st["loss_max"], color=color, lw=1.0, ls=(0, (1, 2)), label=name + ", WORST")
            else:
                ax.plot(st[key], color=color, lw=1.8, label=name)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold", loc="left", pad=8, fontsize=8)
        ax.set_xlabel("ITERATION", fontweight="bold", labelpad=4, fontsize=8)
    axes[1, 0].set_ylim(1e-3, 1.0)  # room for the legend below the curves
    legend(axes[0, 0], loc="upper left")
    legend(axes[1, 0], loc="lower left", ncol=2)
    fig.suptitle(f"[ HEALTH OF THE SWARM ]  {example.N_PARTICLES} PARTICLES  c1 = c2 = {C0}  SEED {seed}",
                 x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "health_iris.png")


# ----------------------------------------------------------------------------------------- 2. map
def stagnation_rate(w, c):
    """Growth rate per step of the second moment of a particle that stagnates (p = g = 0, c1 = c2 = c).

    In one dimension the update is (x, v) -> A (x, v) with A = [[1 - phi, w], [-phi, w]] and phi = c (r1 + r2),
    where r1 and r2 are uniform in [0, 1]. Then E[phi] = c and E[phi^2] = 7 c^2 / 6, and the second moments
    (E x^2, E xv, E v^2) obey a linear recursion with the 3 x 3 matrix below. The rate is half the logarithm of
    its largest eigenvalue: negative, the swarm contracts around its attractors, positive, it spreads.
    """
    rate = np.zeros((len(w), len(c)))
    for i, wi in enumerate(w):
        for j, cj in enumerate(c):
            e1, e2 = cj, 7 / 6 * cj ** 2
            M = np.array([[1 - 2 * e1 + e2, 2 * wi * (1 - e1), wi ** 2],
                          [e2 - e1, wi * (1 - 2 * e1), wi ** 2],
                          [e2, -2 * wi * e1, wi ** 2]])
            rate[i, j] = 0.5 * np.log(np.abs(np.linalg.eigvals(M)).max())
    return rate


def second_order_boundary(w):
    """c1 = c2 = c on the stability boundary of the second moment: c1 + c2 = 24 (1 - w^2) / (7 - 5 w) (Poli 2009); I checked it against the exact recursion of ``stagnation_rate``."""
    return 12 * (1 - w ** 2) / (7 - 5 * w)


def stability_map(seeds=(0, 1), iterations=100, n=15):
    x_train, y_train, _, _ = data()
    ws_nn = np.round(np.arange(0.0, 1.21, 0.1), 2)
    cs_nn = np.arange(0.25, 2.51, 0.25)
    final = np.zeros((len(ws_nn), len(cs_nn)))
    for i, w in enumerate(ws_nn):
        for j, c in enumerate(cs_nn):
            final[i, j] = np.mean([np.log10(run_swarm(x_train, y_train, s, iterations, n, w, c, c).best_loss)
                                   for s in seeds])
        print(f"map: w = {w:.1f} done", flush=True)
    ws, cs = np.arange(0.0, 1.201, 0.05), np.arange(0.25, 2.501, 0.05)
    rate = stagnation_rate(ws, cs)

    fig = new_figure((11, 5.2))
    ax1, ax2 = fig.subplots(1, 2)
    wline = np.linspace(0, 1, 200)
    for ax in (ax1, ax2):
        style_axes(ax, grid=False)
        ax.set_xlabel(r"ATTRACTION  $c_1 = c_2 = c$", fontweight="bold", labelpad=6, fontsize=9)
        ax.set_ylabel(r"INERTIA  $w$", fontweight="bold", labelpad=6, fontsize=9)
        ax.plot(second_order_boundary(wline), wline, color=INK, lw=1.6, ls=(0, (4, 3)), zorder=5,
                label="BOUNDARY, 24(1-w^2)/(7-5w)")
        ax.plot([C0], [W0], marker="X", ms=12, mfc=ACCENT, mec=INK, mew=1.8, ls="none", zorder=6, label="DEFAULT")
    cmap = LinearSegmentedColormap.from_list("rate", [INK, PAPER, ACCENT])
    lim = np.abs(rate).max()
    m1 = ax1.pcolormesh(cs, ws, rate, cmap=cmap, norm=TwoSlopeNorm(0, -lim / 2, lim / 2), shading="nearest")
    ax1.contour(cs, ws, rate, levels=[0], colors=[INK], linewidths=2.2, zorder=4)
    ax1.set_title("STAGNATING PARTICLES: GROWTH RATE OF THE SECOND MOMENT", fontweight="bold", loc="left", pad=8,
                  fontsize=8)
    fig.colorbar(m1, ax=ax1, pad=0.02).ax.tick_params(labelsize=7)
    dc, dw = cs_nn[1] - cs_nn[0], ws_nn[1] - ws_nn[0]
    m2 = ax2.pcolormesh(np.append(cs_nn - dc / 2, cs_nn[-1] + dc / 2), np.append(ws_nn - dw / 2, ws_nn[-1] + dw / 2),
                        final, cmap=LinearSegmentedColormap.from_list("loss", [PAPER, INK]), shading="flat")
    ax2.set_title("IRIS: log10 OF THE BEST LOSS  (LIGHT = LOWER)", fontweight="bold",
                  loc="left", pad=8, fontsize=8)
    fig.colorbar(m2, ax=ax2, pad=0.02).ax.tick_params(labelsize=7)
    for ax in (ax1, ax2):
        ax.set_xlim(cs[0], cs[-1])
        ax.set_ylim(ws[0], ws[-1])
    legend(ax1, loc="upper right")
    fig.tight_layout()
    save(fig, "stability_map_iris.png")
    return ws, cs, rate, ws_nn, cs_nn, final


# ----------------------------------------------------------------------------------------- 3. generalization
def generalization(seeds=range(5)):
    x_train, y_train, x_test, y_test = data()
    curves = {}
    for seed in seeds:
        rec = {"train": [], "test": [], "test_acc": [], "weight": []}

        def callback(sw):
            m = sw.best_model()
            rec["train"].append(example.evaluate(m, x_train, y_train)[0])
            t = example.evaluate(m, x_test, y_test)
            rec["test"].append(t[0])
            rec["test_acc"].append(t[1])
            rec["weight"].append(sw.stats["max_abs_weight"][-1])

        run_swarm(x_train, y_train, seed, callback=callback)
        curves[seed] = rec
        print(f"generalization: seed {seed}: final train {rec['train'][-1]:.4f} test {rec['test'][-1]:.4f} "
              f"acc {rec['test_acc'][-1]:.3f} max|w| {rec['weight'][-1]:.1f}; "
              f"min test loss {min(rec['test']):.4f} at iteration {int(np.argmin(rec['test']))}", flush=True)

    fig = new_figure((12, 4.6))
    axes = fig.subplots(1, 3)
    panels = [("train", "TRAINING LOSS OF THE BEST MODEL", True), ("test", "TEST LOSS OF THE BEST MODEL", True),
              ("weight", "LARGEST WEIGHT OF THE BEST MODEL", True)]
    for ax, (key, title, logy) in zip(axes, panels):
        style_axes(ax)
        for seed in seeds:
            hot = seed == 1
            ax.plot(curves[seed][key], color=ACCENT if hot else INK, lw=2.4 if hot else 1.1,
                    alpha=1.0 if hot else 0.5, zorder=4 if hot else 3,
                    label={1: "SEED 1", 0: "OTHER SEEDS"}.get(seed))
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold", loc="left", pad=8, fontsize=8)
        ax.set_xlabel("ITERATION", fontweight="bold", labelpad=4, fontsize=8)
    legend(axes[0], loc="lower left")
    fig.suptitle(f"[ GENERALIZATION ]  {example.N_PARTICLES} PARTICLES", x=0.02, ha="left", fontweight="bold",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "generalization_iris.png")
    return curves


# ----------------------------------------------------------------------------------------- 4. statistics
def adam_run(x, y, seed, epochs=example.EPOCHS):
    """One run of Adam. Returns the times (training only), the training loss after every epoch, and the model."""
    torch.manual_seed(seed)
    model = example.build_model()
    opt = torch.optim.Adam(model.parameters(), lr=example.LEARNING_RATE)
    t, times, losses = 0.0, [0.0], [example.evaluate(model, x, y)[0]]
    for _ in range(epochs):
        t0 = time.perf_counter()
        order = torch.randperm(len(x))
        for i in range(0, len(x), example.BATCH_SIZE):
            idx = order[i:i + example.BATCH_SIZE]
            opt.zero_grad()
            example.loss_fn(model(x[idx]), y[idx]).backward()
            opt.step()
        t += time.perf_counter() - t0
        times.append(t)
        losses.append(example.evaluate(model, x, y)[0])
    return np.array(times), np.array(losses), model


def statistics(n_seeds=20):
    x_train, y_train, x_test, y_test = data()
    pso_t, pso_l, pso_m = [], [], []
    for seed in range(n_seeds):
        stamps = []
        sw = run_swarm(x_train, y_train, seed, callback=lambda s: stamps.append(time.perf_counter()))
        pso_t.append(np.array(stamps) - stamps[0])
        pso_l.append(np.array(sw.history))
        pso_m.append(sw.best_model())
        print(f"stats: swarm {seed + 1}/{n_seeds}", flush=True)
    adam_t, adam_l, adam_m = [], [], []
    for seed in range(n_seeds):
        t, l, m = adam_run(x_train, y_train, seed)
        adam_t.append(t)
        adam_l.append(np.minimum.accumulate(l))  # best loss so far, like the swarm
        adam_m.append(m)
        print(f"stats: adam {seed + 1}/{n_seeds}", flush=True)

    t_max = min(max(t[-1] for t in pso_t), max(t[-1] for t in adam_t))
    grid = np.linspace(0, t_max, 200)

    def on_grid(times, losses):
        return np.array([np.interp(grid, t, l) for t, l in zip(times, losses)])

    fig = new_figure((12, 5))
    ax1, ax2 = fig.subplots(1, 2)
    style_axes(ax1)
    style_axes(ax2)
    its = np.arange(len(pso_l[0]))
    L = np.array(pso_l)
    ax1.fill_between(its, np.percentile(L, 10, 0), np.percentile(L, 90, 0), color=ACCENT, alpha=0.3, lw=0)
    ax1.plot(its, np.median(L, 0), color=ACCENT, lw=2.4, label=f"MEDIAN OF {n_seeds} SEEDS, 10-90% BAND")
    ax1.set_yscale("log")
    ax1.set_xlabel("ITERATION", fontweight="bold", labelpad=6, fontsize=9)
    ax1.set_ylabel("BEST TRAINING LOSS (MSE)", fontweight="bold", labelpad=6, fontsize=9)
    ax1.set_title("SWARM", fontweight="bold", loc="left", pad=8, fontsize=9)
    legend(ax1)
    for name, color, T, Ls in (("SWARM", ACCENT, pso_t, pso_l), ("ADAM", INK, adam_t, adam_l)):
        A = on_grid(T, Ls)
        ax2.fill_between(grid, np.percentile(A, 10, 0), np.percentile(A, 90, 0), color=color, alpha=0.25, lw=0)
        ax2.plot(grid, np.median(A, 0), color=color, lw=2.2, label=name + ": MEDIAN, 10-90% BAND")
    ax2.set_yscale("log")
    ax2.set_xlabel("TRAINING TIME (SECONDS, ONE CPU PROCESS)", fontweight="bold", labelpad=6, fontsize=9)
    ax2.set_title("SWARM AGAINST ADAM, AT EQUAL TIME", fontweight="bold", loc="left", pad=8, fontsize=9)
    legend(ax2)
    fig.suptitle(f"[ STATISTICS ]  {example.N_PARTICLES} PARTICLES, {example.EPOCHS} EPOCHS OF ADAM, LR "
                 f"{example.LEARNING_RATE}", x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "statistics_iris.png")

    def final(models):
        r = np.array([example.evaluate(m, x_train, y_train) + example.evaluate(m, x_test, y_test) for m in models])
        return np.median(r, 0), np.percentile(r, 10, 0), np.percentile(r, 90, 0)

    print("final (median [10%, 90%]) over", n_seeds, "seeds: train loss, train acc, test loss, test acc")
    for name, ms in (("swarm", pso_m), ("adam", adam_m)):
        med, lo, hi = final(ms)
        print(f"  {name:6s}", "  ".join(f"{a:.4f} [{b:.4f}, {c:.4f}]" for a, b, c in zip(med, lo, hi)))
    print(f"  time per run: swarm {np.mean([t[-1] for t in pso_t]):.1f} s, adam {np.mean([t[-1] for t in adam_t]):.1f} s")
    # the loss reached at the end of the shorter run, at equal time
    A_p, A_a = on_grid(pso_t, pso_l)[:, -1], on_grid(adam_t, adam_l)[:, -1]
    print(f"  at t = {t_max:.1f} s: swarm median loss {np.median(A_p):.4f}, adam {np.median(A_a):.4f}")
    # time to reach a loss of 0.01
    for name, T, Ls in (("swarm", pso_t, pso_l), ("adam", adam_t, adam_l)):
        hit = [t[np.argmax(l < 0.01)] if (l < 0.01).any() else np.inf for t, l in zip(T, Ls)]
        print(f"  {name}: median time to reach a training loss of 0.01: {np.median(hit):.1f} s "
              f"({int(np.isfinite(hit).sum())}/{n_seeds} runs reach it)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", default="all",
                        choices=["all", "convergence", "health", "map", "generalization", "stats"])
    what = parser.parse_args().what
    if what in ("all", "convergence"):
        convergence()
    if what in ("all", "health"):
        health()
    if what in ("all", "generalization"):
        generalization()
    if what in ("all", "stats"):
        statistics()
    if what in ("all", "map"):
        stability_map()
