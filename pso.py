"""Particle swarm optimization of the weights of a PyTorch model. No gradients are used.

A particle is a point theta in R^D, where D is the number of parameters of the model: its position is one
complete set of weights. Every particle remembers the best position p it has visited, and the swarm remembers the
best position g that any particle has visited. One iteration moves all the particles:

    v <- w v + c1 r1 (p - theta) + c2 r2 (g - theta)
    theta <- theta + v

with r1 and r2 vectors of D independent random numbers, uniform in [0, 1], drawn anew at every iteration and for
every particle (the product is element-wise, one random number per weight).

The default parameters are w = 0.7298 and c1 = c2 = 1.49618: the constriction coefficient of Clerc and Kennedy.

    J. Kennedy, R. Eberhart. Particle swarm optimization. Proc. IEEE ICNN 1995, vol. 4, pp. 1942-1948.
    Y. Shi, R. Eberhart. A modified particle swarm optimizer. Proc. IEEE ICEC 1998, pp. 69-73.   (the weight w)
    M. Clerc, J. Kennedy. The particle swarm: explosion, stability, and convergence in a multidimensional
        complex space. IEEE Trans. Evol. Comput. 6(1):58-73, 2002.                                (w, c1, c2)
    D. Bratton, J. Kennedy. Defining a standard for particle swarm optimization. IEEE SIS 2007, pp. 120-127.
"""
import copy

import torch
from torch.nn.utils import parameters_to_vector


class Swarm:
    def __init__(self, model, loss_fn, n_particles=30, w=0.7298, c1=1.49618, c2=1.49618,
                 init_velocity=0.1, vmax=None, per_weight=True, seed=0):
        """
        model          the network to train. It is not modified: its weights are the starting point of the
                       first particle, the other particles start from new random weights (the default
                       initialization of the layers). ``best_model`` returns the trained copy.
        loss_fn        function (output, target) -> scalar tensor. This is the quantity that is minimized.
        n_particles    size of the swarm.
        w, c1, c2      inertia of the velocity, pull towards the best position of the particle, pull towards the
                       best position of the swarm.
        init_velocity  the initial velocities are uniform in [-init_velocity, init_velocity].
        vmax           if given, every component of the velocity is clipped to [-vmax, vmax] (as in the first
                       versions of the algorithm, where it prevented the divergence of the swarm).
        per_weight     if True (the algorithm of the paper) r1 and r2 have one random number for every weight; if
                       False a single random number is used for all the weights of a particle (for comparison).
        seed           seed of the random numbers of the swarm (initial positions, velocities, r1, r2).
        """
        if n_particles < 2:
            raise ValueError("a swarm needs at least 2 particles")
        self.model = copy.deepcopy(model)
        self.loss_fn = loss_fn
        self.n, self.w, self.c1, self.c2, self.vmax = n_particles, w, c1, c2, vmax
        self.per_weight = per_weight
        self.gen = torch.Generator().manual_seed(seed)

        theta0 = parameters_to_vector(self.model.parameters()).detach().clone()
        self.dim = theta0.numel()

        # positions: the weights of the given model, and n - 1 new random initializations of the same network
        self.X = torch.empty(self.n, self.dim)
        self.X[0] = theta0
        with torch.random.fork_rng(devices=[]):  # the global random state of torch is left untouched
            torch.manual_seed(seed)
            for i in range(1, self.n):
                for m in self.model.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                self.X[i] = parameters_to_vector(self.model.parameters()).detach()
        self._load(self.model, theta0)

        self.V = (2 * torch.rand(self.n, self.dim, generator=self.gen) - 1) * init_velocity

        self.P = None  # best position of every particle
        self.pf = None  # and its loss
        self.G = None  # best position of the swarm
        self.gf = float("inf")  # and its loss
        self.history = []  # best loss of the swarm: at the start, and after every iteration
        # diagnostics, one number at the start and one after every iteration:
        #   diversity       spread of the particles around their centroid (rms over particles and weights)
        #   velocity        rms of the velocity components
        #   loss_min/median/max   current loss of the particles (not their best loss)
        #   improved        fraction of the particles that found a new best position in the iteration
        #   max_abs_weight  largest weight of the best position of the swarm
        self.stats = {k: [] for k in ("diversity", "velocity", "loss_min", "loss_median", "loss_max", "improved",
                                      "max_abs_weight")}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load(model, theta):
        """Copy the vector theta into the parameters of the model."""
        i = 0
        with torch.no_grad():
            for p in model.parameters():
                k = p.numel()
                p.copy_(theta[i:i + k].view_as(p))
                i += k

    def _fitness(self, theta, x, y):
        self._load(self.model, theta)
        with torch.no_grad():
            return self.loss_fn(self.model(x), y).item()

    def _evaluate_all(self, x, y):
        return torch.tensor([self._fitness(theta, x, y) for theta in self.X])

    def _record(self, f, improved):
        s = self.stats
        s["diversity"].append(float((self.X - self.X.mean(0)).pow(2).mean().sqrt()))
        s["velocity"].append(float(self.V.pow(2).mean().sqrt()))
        s["loss_min"].append(float(f.min()))
        s["loss_median"].append(float(f.median()))
        s["loss_max"].append(float(f.max()))
        s["improved"].append(improved)
        s["max_abs_weight"].append(float(self.G.abs().max()))

    # ------------------------------------------------------------------ the algorithm
    def fit(self, x, y, iterations=100, verbose=0, callback=None):
        """Run the swarm for a number of iterations and return the history of the best loss.

        The loss of every particle is computed on the whole data set (x, y), so that all the numbers that are
        compared (the loss now, the best loss of the particle, the best loss of the swarm) refer to the same
        function. Calling ``fit`` again continues the run, with the same data.

        callback  if given, it is called as ``callback(swarm)`` at the start and after every iteration, for
                  example to measure something on a test set with ``swarm.best_model()``.
        """
        if self.P is None:  # first call: the starting positions are the first best positions
            f = self._evaluate_all(x, y)
            self.P, self.pf = self.X.clone(), f
            b = int(f.argmin())
            self.G, self.gf = self.P[b].clone(), float(f[b])
            self.history.append(self.gf)
            self._record(f, float("nan"))
            if callback is not None:
                callback(self)

        for it in range(1, iterations + 1):
            shape = (self.n, self.dim) if self.per_weight else (self.n, 1)
            r1 = torch.rand(shape, generator=self.gen)
            r2 = torch.rand(shape, generator=self.gen)
            self.V = self.w * self.V + self.c1 * r1 * (self.P - self.X) + self.c2 * r2 * (self.G - self.X)
            if self.vmax is not None:
                self.V.clamp_(-self.vmax, self.vmax)
            self.X = self.X + self.V

            f = self._evaluate_all(x, y)
            better = f < self.pf  # particles that found a new best position
            self.P[better] = self.X[better]
            self.pf[better] = f[better]
            b = int(self.pf.argmin())
            if float(self.pf[b]) < self.gf:
                self.G, self.gf = self.P[b].clone(), float(self.pf[b])
            self.history.append(self.gf)
            self._record(f, float(better.float().mean()))
            if callback is not None:
                callback(self)

            if verbose and it % verbose == 0:
                print(f"iteration {it:5d}   best loss {self.gf:.5f}")
        return self.history

    # ------------------------------------------------------------------ result
    @property
    def best_loss(self):
        return self.gf

    def best_model(self):
        """A copy of the model with the best weights found by the swarm."""
        if self.G is None:
            raise RuntimeError("call fit first")
        model = copy.deepcopy(self.model)
        self._load(model, self.G)
        return model
