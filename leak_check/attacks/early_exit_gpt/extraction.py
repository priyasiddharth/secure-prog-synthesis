"""
Label-only extraction: steal the gate's decision boundary from binary labels.

The classic black-box setting -- the attacker sees only a discrete decision per
query (here, "did the model early-exit?", obtained through the timing side
channel). Active learning: round 0 probes at random; each later round fits a
linear classifier to everything labelled so far, then spends its queries near the
classifier's current boundary. A linear gate converges in roughly (dim)
informative queries.

Depends only on an ``Observe`` callable that returns the label as a scalar; it
never sees the GPT, the timing, or torch.
"""

import numpy as np

from ..base import AttackResult, ExtractionAttack, Observe
from .sampling import boundary_probes, random_probes


class SurrogateExtractionAttack(ExtractionAttack):
    def __init__(self, config):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)

    def run(self, observe: Observe, dim: int, budget: int) -> AttackResult:
        per_round = max(10, budget // self.cfg.n_rounds)
        X_all, y_all = [], []
        surrogate = None
        history = []

        for r in range(self.cfg.n_rounds):
            for x in self._propose(surrogate, per_round, dim):
                X_all.append(x)
                y_all.append(int(observe(x)))

            X, y = np.array(X_all), np.array(y_all)
            if len(np.unique(y)) < 2:
                continue  # can't fit a classifier until both classes appear
            surrogate = self.cfg.surrogate_factory().fit(X, y)
            history.append({"round": r, "queries": len(X),
                            "train_acc": float(surrogate.score(X, y))})

        if surrogate is None:
            raise RuntimeError(
                "every query returned the same label; the oracle never separated "
                "the classes, so no boundary could be recovered")
        return AttackResult(surrogate=surrogate, queries_used=len(X_all),
                            history=history)

    def _propose(self, surrogate, n, dim):
        if surrogate is None:
            return random_probes(self._rng, n, dim, self.cfg.probe_scale)
        # LogisticRegression exposes the hyperplane as coef_[0] / intercept_[0].
        return boundary_probes(self._rng, surrogate.coef_[0], surrogate.intercept_[0],
                               n, dim, self.cfg.probe_scale, self.cfg.refine_jitter)
