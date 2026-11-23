"""One random number per weight, or one per particle? 20 seeds on the Iris example.

    python ablation.py
"""
import numpy as np

import example
from plots import data, run_swarm
from pso import Swarm


def run(x, y, seed, per_weight):
    import torch
    torch.manual_seed(seed)
    swarm = Swarm(example.build_model(), example.loss_fn, n_particles=example.N_PARTICLES, per_weight=per_weight,
                  seed=seed)
    swarm.fit(x, y, iterations=example.ITERATIONS)
    return swarm


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = data()
    print("20 seeds, median [10%, 90%]: best training loss, test loss, test accuracy, diversity at the end")
    for name, per_weight in (("one random number per weight", True), ("one random number per particle", False)):
        rows = []
        for seed in range(20):
            sw = run(x_train, y_train, seed, per_weight)
            m = sw.best_model()
            rows.append([sw.best_loss, *example.evaluate(m, x_test, y_test), sw.stats["diversity"][-1]])
        r = np.array(rows)
        med, lo, hi = np.median(r, 0), np.percentile(r, 10, 0), np.percentile(r, 90, 0)
        print(f"  {name:32s}", "  ".join(f"{a:.4f} [{b:.4f}, {c:.4f}]" for a, b, c in zip(med, lo, hi)))
