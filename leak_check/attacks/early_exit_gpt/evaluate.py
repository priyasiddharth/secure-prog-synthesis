"""
Scoring: how well does what the attacker learned match the real gate?

Both metrics compare against ``enclave.true_decision`` (ground truth), which only
the harness may call.

  oracle_alignment     agreement between the labeler's noisy labels and the true
                       gate. An upper bound on what any attack can learn through
                       this oracle -- caps how good the surrogate can get.
  surrogate_alignment  agreement between the stolen surrogate and the true gate:
                       the actual attack success rate (accuracy, in the paper's
                       terms).
  logit_fidelity       R^2 of the surrogate's predicted logit against the true
                       gate logit: how well the confidence surface itself was
                       reconstructed (fidelity), not just its sign. Only defined
                       for a surrogate that exposes decision_function (the logit
                       attack).
"""

import numpy as np


def _ground_truth(enclave, X):
    return np.array([enclave.true_decision(x) for x in X])


def oracle_alignment(labeler, enclave, X):
    labels = np.array([labeler.label(x) for x in X])
    return float((labels == _ground_truth(enclave, X)).mean())


def surrogate_alignment(result, enclave, X):
    return float((result.predict(X) == _ground_truth(enclave, X)).mean())


def logit_fidelity(result, logit_oracle, X):
    """R^2 between the surrogate's predicted logit and the true gate logit."""
    surrogate = result.surrogate
    if not hasattr(surrogate, "decision_function"):
        raise TypeError("logit_fidelity needs a surrogate exposing decision_function "
                        "(the logit attack); the label attack has no logit estimate")
    true = np.array([logit_oracle.query(x) for x in X])
    pred = surrogate.decision_function(X)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
