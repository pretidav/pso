"""Train a small network on the Iris data set with a particle swarm, and compare with backpropagation (Adam).

    python example.py

The network is small: 4 -> 4 -> 4 -> 3, sigmoid, sigmoid, softmax, with the mean squared
error between the softmax output and the one-hot label as the loss. The backpropagation baseline is the best of a
few independent runs, chosen by the loss on the training set.
"""
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn

from pso import Swarm

N_PARTICLES = 30  # particles of the swarm
ITERATIONS = 200  # iterations of the swarm
N_BACKPROP_RUNS = 5  # independent runs of backpropagation, the best one is kept
EPOCHS = 200  # epochs of every run of backpropagation
BATCH_SIZE = 8
LEARNING_RATE = 0.01

loss_fn = nn.MSELoss()


def build_model():
    return nn.Sequential(
        nn.Linear(4, 4), nn.Sigmoid(),
        nn.Linear(4, 4), nn.Sigmoid(),
        nn.Linear(4, 3), nn.Softmax(dim=1),
    )


def evaluate(model, x, y):
    """Loss and accuracy of the model. y is one-hot."""
    with torch.no_grad():
        out = model(x)
        return loss_fn(out, y).item(), (out.argmax(1) == y.argmax(1)).float().mean().item()


def backpropagation(x_train, y_train):
    best, best_loss = None, float("inf")
    for run in range(N_BACKPROP_RUNS):
        torch.manual_seed(run)
        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for _ in range(EPOCHS):
            order = torch.randperm(len(x_train))
            for i in range(0, len(x_train), BATCH_SIZE):
                idx = order[i:i + BATCH_SIZE]
                optimizer.zero_grad()
                loss_fn(model(x_train[idx]), y_train[idx]).backward()
                optimizer.step()
        train_loss, _ = evaluate(model, x_train, y_train)
        if train_loss < best_loss:
            best, best_loss = model, train_loss
    return best


if __name__ == "__main__":
    iris = load_iris()
    x_train, x_test, labels_train, labels_test = train_test_split(
        iris.data, iris.target, test_size=0.5, random_state=0, stratify=iris.target)
    x_train, x_test = (torch.tensor(a, dtype=torch.float32) for a in (x_train, x_test))
    y_train, y_test = (nn.functional.one_hot(torch.tensor(a), 3).float() for a in (labels_train, labels_test))

    # backpropagation
    model_b = backpropagation(x_train, y_train)
    print("Backprop  train loss {:.4f} acc {:.3f}   test loss {:.4f} acc {:.3f}".format(
        *evaluate(model_b, x_train, y_train), *evaluate(model_b, x_test, y_test)))

    # particle swarm
    torch.manual_seed(0)
    swarm = Swarm(build_model(), loss_fn, n_particles=N_PARTICLES, seed=0)
    swarm.fit(x_train, y_train, iterations=ITERATIONS, verbose=20)
    model_p = swarm.best_model()
    print("PSO       train loss {:.4f} acc {:.3f}   test loss {:.4f} acc {:.3f}".format(
        *evaluate(model_p, x_train, y_train), *evaluate(model_p, x_test, y_test)))
