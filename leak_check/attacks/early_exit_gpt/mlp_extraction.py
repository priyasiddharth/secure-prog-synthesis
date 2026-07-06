"""
Nonlinear logit extraction: MLP surrogate + Jacobian active learning.

The stronger of the two logit attacks, following arXiv:1907.00713's improved
learning attack: a surrogate of matching capacity (a small MLP, not a linear map)
regresses the confidence surface, and queries are steered onto the surrogate's
current boundary using its exact input gradient (autograd). The linear logit
attack (`logit_extraction.py`) is the baseline this is meant to beat when the
logit surface is genuinely nonlinear and the query budget is sufficient.

Depends only on an ``Observe`` callable returning the scalar logit.
"""

import numpy as np

from ..base import AttackResult, ExtractionAttack, Observe
from .logit_extraction import _SignRegressor
from .mlp_surrogate import TorchMLPRegressor
from .sampling import gradient_boundary_probes, random_probes


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class MlpLogitExtractionAttack(ExtractionAttack):
    def __init__(self, config):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)

    def _new_surrogate(self):
        return TorchMLPRegressor(
            hidden=self.cfg.mlp_hidden, epochs=self.cfg.mlp_epochs,
            lr=self.cfg.mlp_lr, weight_decay=self.cfg.mlp_weight_decay,
            seed=self.cfg.seed)

    def run(self, observe: Observe, dim: int, budget: int) -> AttackResult:
        per_round = max(10, budget // self.cfg.n_rounds)
        X_all, y_all = [], []
        mlp = None
        history = []

        for r in range(self.cfg.n_rounds):
            for x in self._propose(mlp, per_round, dim):
                X_all.append(x)
                y_all.append(observe(x))       # continuous logit
            X, y = np.array(X_all), np.array(y_all)
            mlp = self._new_surrogate().fit(X, y)
            history.append({"round": r, "queries": len(X),
                            "train_r2": _r2(y, mlp.predict(X))})

        return AttackResult(surrogate=_SignRegressor(mlp),
                            queries_used=len(X_all), history=history)

    def _propose(self, mlp, n, dim):
        if mlp is None:
            return random_probes(self._rng, n, dim, self.cfg.probe_scale)
        return gradient_boundary_probes(
            self._rng, mlp.value_and_grad, n, dim,
            self.cfg.probe_scale, self.cfg.refine_jitter)
