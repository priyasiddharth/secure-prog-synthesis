"""
The attacker's view of the enclave, and how a raw observable becomes a label.

Three oracles implement the ``base.Oracle`` protocol over the same enclave; they
differ in threat model and in how much they reveal per query:

  TimingOracle      the realistic side channel -- median wall-clock latency of the
                    forward pass. Noisy, but exactly what a remote attacker
                    measures through an API. Reveals ~one bit (fast vs. slow).
  LayerCountOracle  a noise-free control -- the number of blocks executed, read
                    off directly. Not available to a real attacker; it isolates
                    the attack algorithm from timing noise so a failure can be
                    attributed (algorithm vs. measurement). Mirrors the detector
                    side's "branchless control". Also ~one bit.
  LogitOracle       a stronger, direct threat model -- the API returns the gate's
                    confidence (the raw logit ``gate(h)``), not just the decision.
                    Reveals the signed distance to the boundary, which is what the
                    logit-based extraction attack (arXiv:1907.00713) exploits.

ThresholdLabeler turns a (one-bit) observable into the binary early-exit label by
calibrating a midpoint between a known-fast and known-slow probe. The LogitOracle
needs no labeler -- its zero crossing is the boundary.
"""

import numpy as np


class TimingOracle:
    """Observable = median forward-pass latency over ``reps`` calls (seconds)."""

    def __init__(self, enclave, reps=1):
        self._enclave = enclave
        self.reps = reps

    def query(self, x: np.ndarray) -> float:
        samples = [self._enclave.forward(x, measure=True)[2] for _ in range(self.reps)]
        return float(np.median(samples))


class LayerCountOracle:
    """Observable = number of transformer blocks executed (noise-free control)."""

    def __init__(self, enclave):
        self._enclave = enclave

    def query(self, x: np.ndarray) -> float:
        return float(self._enclave.forward(x, measure=False)[1])


class LogitOracle:
    """Observable = the secret gate's raw logit ``gate(h)`` (confidence).

    A stronger threat model than the timing/layer-count channels: the API hands
    back the continuous decision value instead of one bit. The gate exits iff the
    logit is positive, so the boundary is the zero crossing -- no calibration
    needed. This is the signal the logit-based extraction attack regresses.
    """

    def __init__(self, enclave):
        self._enclave = enclave

    def query(self, x: np.ndarray) -> float:
        return self._enclave.gate_logit(x)


class ThresholdLabeler:
    """Labels a query 1 (early exit / fast path) or 0 by comparing the oracle
    observable to a calibrated midpoint.

    The midpoint is set from two extreme probes (a uniformly-negative and a
    uniformly-positive fill). Which extreme is the fast one depends on the secret
    gate's sign, but the label "below the midpoint = fast path" is well-defined
    either way, so calibration needs no knowledge of the secret.
    """

    def __init__(self, oracle, dim, lo_fill=-6.0, hi_fill=6.0, calib_reps=5):
        self._oracle = oracle
        self.dim = dim
        self.lo_fill = lo_fill
        self.hi_fill = hi_fill
        self.calib_reps = calib_reps
        self.threshold = None

    def calibrate(self):
        lo = np.median([self._oracle.query(np.full(self.dim, self.lo_fill))
                        for _ in range(self.calib_reps)])
        hi = np.median([self._oracle.query(np.full(self.dim, self.hi_fill))
                        for _ in range(self.calib_reps)])
        self.threshold = (lo + hi) / 2
        return self.threshold

    def label(self, x: np.ndarray) -> int:
        if self.threshold is None:
            raise RuntimeError("calibrate() must be called before label()")
        return 1 if self._oracle.query(x) < self.threshold else 0
