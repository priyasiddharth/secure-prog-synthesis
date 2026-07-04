"""
The extraction algorithm: steal the gate's decision boundary from labels alone.

Active learning. Round 0 probes the input space at random. Each later round
fits a linear surrogate to everything labelled so far, then spends its queries
*near the surrogate's current boundary* -- where a label is most informative --
by projecting a random point onto the hyperplane and adding jitter. The gate is
linear, so a linear surrogate converges to it in roughly (dim) informative
queries, which is why the budget is set as a small multiple of the parameter
count.

Depends only on ``base.Labeler``; it never sees the GPT, the timing, or torch.
"""

import numpy as np

from ..base import AttackResult, ExtractionAttack, Labeler


class SurrogateExtractionAttack(ExtractionAttack):
    def __init__(self, config):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)

    def run(self, labeler: Labeler, dim: int, budget: int) -> AttackResult:
        per_round = max(10, budget // self.cfg.n_rounds)
        X_all, y_all = [], []
        surrogate = None
        history = []

        for r in range(self.cfg.n_rounds):
            for x in self._propose(surrogate, per_round, dim):
                X_all.append(x)
                y_all.append(labeler.label(x))

            X, y = np.array(X_all), np.array(y_all)
            if len(np.unique(y)) < 2:
                continue  # can't fit until both classes have been seen
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
        """Return ``n`` new query points."""
        rng = self._rng
        if surrogate is None:
            return rng.normal(scale=self.cfg.probe_scale, size=(n, dim))

        w = surrogate.coef_[0]
        b = surrogate.intercept_[0]
        ww = float(w @ w)
        batch = np.empty((n, dim))
        for i in range(n):
            x0 = rng.normal(scale=self.cfg.probe_scale, size=dim)
            step = -(w * (w @ x0 + b)) / ww          # project onto {w·x + b = 0}
            batch[i] = x0 + step + rng.normal(scale=self.cfg.refine_jitter, size=dim)
        return batch
