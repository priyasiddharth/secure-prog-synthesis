"""
Activation-function corpus (CSI-NN mechanism 2: activations have input-dependent
control flow / timing). Each model applies one activation elementwise to the
SECRET weight buffer, so the secret is the activation's input. Two baited secret
classes per activation probe the value regimes where a scalar libm implementation
branches (sign for relu; overflow/saturation for the transcendentals). The
differential question: does torch.compile/Inductor change that data-dependence
(e.g. lower a branchy libm path into a branchless vectorized polynomial)?

Consumed by measured_run.py (build/example_input/names) and run_activations.py
(secret_classes).
"""
import numpy as np
import torch
import torch.nn.functional as F

DIM = 512


def _softmax(w):
    return torch.softmax(w, dim=1)


_ACTS = {
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "gelu": F.gelu,
    "softmax": _softmax,
}


class ActModel(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))
        self._fn = fn

    def forward(self, x):
        return self._fn(self.weight)


def build(name):
    return ActModel(_ACTS[name])


def example_input(name):
    return torch.randn(1, DIM)


def names():
    return list(_ACTS)


def secret_classes(name):
    """Return ((labelA, arrA), (labelB, arrB)) — same shape/dtype, baited regimes."""
    shp = (DIM, DIM)
    if name == "relu":
        return (("neg", np.full(shp, -1.0, np.float32)),
                ("pos", np.full(shp, 1.0, np.float32)))
    # transcendentals: moderate vs large-magnitude (overflow/saturation paths in
    # scalar libm: expf overflows ~|x|>88, sigmoid/tanh saturate, gelu erf tail).
    return (("small", np.full(shp, 0.5, np.float32)),
            ("large", np.full(shp, 100.0, np.float32)))
