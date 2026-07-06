"""
A small torch MLP used as the attacker's surrogate for the logit surface.

The linear regressor of the baseline logit attack underfits the gate's confidence
(nonlinear in the query through the transformer blocks). This surrogate has the
capacity to model that nonlinearity, and -- being a torch module -- exposes an
*exact* input gradient via autograd, which the Jacobian active-learning step uses
to walk queries onto the decision boundary of a nonlinear surrogate.

Inputs and targets are standardized (the logit scale and the per-feature query
scale differ), so training is well-conditioned; ``predict`` and ``value_and_grad``
undo the standardization and report in the original logit / query space.
"""

import numpy as np
import torch
import torch.nn as nn


class TorchMLPRegressor:
    def __init__(self, hidden=(64,), epochs=400, lr=1e-2, weight_decay=1e-3, seed=0):
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self._net = None
        self._x_mu = self._x_sd = None
        self._y_mu = self._y_sd = None

    def _build(self, dim):
        layers, d = [], dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-8
        self._y_mu, self._y_sd = float(y.mean()), float(y.std()) + 1e-8

        Xt = torch.tensor((X - self._x_mu) / self._x_sd)
        yt = torch.tensor((y - self._y_mu) / self._y_sd).view(-1, 1)

        self._net = self._build(X.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss_fn(self._net(Xt), yt).backward()
            opt.step()
        self._net.eval()
        return self

    def _logit_std_in(self, Xt):
        """Forward on already-standardized input; returns logit in original scale."""
        return self._net(Xt).squeeze(-1) * self._y_sd + self._y_mu

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.tensor((X - self._x_mu) / self._x_sd)
        with torch.no_grad():
            return self._logit_std_in(Xt).numpy()

    def value_and_grad(self, X):
        """Return (logit, d logit / d x) at each row of ``X``, in original space."""
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.tensor((X - self._x_mu) / self._x_sd, requires_grad=True)
        v = self._logit_std_in(Xt)
        grad_std, = torch.autograd.grad(v.sum(), Xt)   # rows decouple under sum
        # chain rule from standardized input back to raw x: d/dx = d/dXn * (1/sd)
        grad_x = grad_std.numpy() / self._x_sd
        return v.detach().numpy(), grad_x
