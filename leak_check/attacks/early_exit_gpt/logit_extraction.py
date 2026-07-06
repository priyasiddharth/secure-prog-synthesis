"""
Logit-based extraction (Jagielski et al., "High-Fidelity Extraction of Neural
Network Models", arXiv:1907.00713).

Threat model, stronger than the label-only side channel: the API returns the
gate's *confidence* -- the raw logit ``gate(h)`` -- not just the exit decision.
Each query then carries the signed distance to the boundary rather than one bit,
so the attacker regresses the logit surface instead of classifying labels. The
paper's finding is that this reaches higher fidelity in far fewer queries; here
it also sidesteps the label channel's worst region (points near the boundary,
where a label is noisy but a logit is small and precise).

The recovered surrogate exposes both the model's decision (sign of the regressed
logit) and its confidence (the regressed logit itself), so it can be scored on
accuracy *and* on fidelity to the logit surface.

Depends only on an ``Observe`` callable returning the scalar logit.
"""

import numpy as np

from ..base import AttackResult, ExtractionAttack, Observe
from .sampling import boundary_probes, random_probes


class _SignRegressor:
    """Adapts a fitted regressor to the surrogate interface: ``decision_function``
    returns the predicted logit, ``predict`` its sign as the class (exit iff the
    predicted logit is positive, matching the gate)."""

    def __init__(self, regressor):
        self.regressor = regressor

    def decision_function(self, X):
        return self.regressor.predict(X)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class RegressionExtractionAttack(ExtractionAttack):
    def __init__(self, config):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)

    def run(self, observe: Observe, dim: int, budget: int) -> AttackResult:
        per_round = max(10, budget // self.cfg.n_rounds)
        X_all, y_all = [], []
        regressor = None
        history = []

        for r in range(self.cfg.n_rounds):
            for x in self._propose(regressor, per_round, dim):
                X_all.append(x)
                y_all.append(observe(x))       # the continuous logit, not a label

            X, y = np.array(X_all), np.array(y_all)
            regressor = self.cfg.regressor_factory().fit(X, y)   # fits from round 0
            history.append({"round": r, "queries": len(X),
                            "train_r2": float(regressor.score(X, y))})

        return AttackResult(surrogate=_SignRegressor(regressor),
                            queries_used=len(X_all), history=history)

    def _propose(self, regressor, n, dim):
        if regressor is None:
            return random_probes(self._rng, n, dim, self.cfg.probe_scale)
        # LinearRegression exposes the hyperplane as coef_ / intercept_ (scalar);
        # the boundary is where the predicted logit crosses zero.
        return boundary_probes(self._rng, regressor.coef_, regressor.intercept_,
                               n, dim, self.cfg.probe_scale, self.cfg.refine_jitter)
