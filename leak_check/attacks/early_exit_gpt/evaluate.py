"""
Scoring: how well does what the attacker learned match the real gate?

Both metrics compare against ``enclave.true_decision`` (ground truth), which only
the harness may call.

  oracle_alignment     agreement between the labeler's noisy labels and the true
                       gate. An upper bound on what any attack can learn through
                       this oracle -- caps how good the surrogate can get.
  surrogate_alignment  agreement between the stolen surrogate and the true gate:
                       the actual attack success rate.
"""

import numpy as np


def _ground_truth(enclave, X):
    return np.array([enclave.true_decision(x) for x in X])


def oracle_alignment(labeler, enclave, X):
    labels = np.array([labeler.label(x) for x in X])
    return float((labels == _ground_truth(enclave, X)).mean())


def surrogate_alignment(result, enclave, X):
    return float((result.predict(X) == _ground_truth(enclave, X)).mean())
