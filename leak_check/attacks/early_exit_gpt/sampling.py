"""
Query-selection geometry, shared by the classification and regression attacks.

Both attacks are active learners: sample at random until they have a linear
surrogate, then concentrate queries near that surrogate's current boundary,
where a query is most informative about the boundary's location.
"""

import numpy as np


def random_probes(rng, n, dim, scale):
    """``n`` isotropic Gaussian query points -- the cold-start distribution."""
    return rng.normal(scale=scale, size=(n, dim))


def boundary_probes(rng, w, b, n, dim, scale, jitter):
    """``n`` points near the linear boundary ``w . x + b = 0``.

    Each is a random draw projected onto the hyperplane, plus small jitter so the
    batch straddles the boundary rather than sitting exactly on it.
    """
    ww = float(w @ w)
    batch = np.empty((n, dim))
    for i in range(n):
        x0 = rng.normal(scale=scale, size=dim)
        step = -(w * (w @ x0 + b)) / ww
        batch[i] = x0 + step + rng.normal(scale=jitter, size=dim)
    return batch


def gradient_boundary_probes(rng, value_and_grad, n, dim, scale, jitter):
    """``n`` points near the boundary of a *nonlinear* surrogate.

    The linear version above knows the boundary in closed form; here we only have
    a local linearization. ``value_and_grad(X) -> (values, grads)`` gives the
    surrogate's predicted logit and its input gradient at each row, and one Gauss-
    Newton step ``x - v * g / (g.g)`` lands near the zero crossing (exact for a
    linear surrogate, one step of a root-find for a nonlinear one). This is the
    Jacobian-based query augmentation of arXiv:1907.00713.
    """
    x0 = random_probes(rng, n, dim, scale)
    values, grads = value_and_grad(x0)
    gg = np.einsum("ij,ij->i", grads, grads) + 1e-12
    step = -grads * (values / gg)[:, None]
    return x0 + step + rng.normal(scale=jitter, size=(n, dim))
