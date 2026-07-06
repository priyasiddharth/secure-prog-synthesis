"""
One place for every knob, so the attack modules stay free of magic numbers.

Two presets:
  small  a tiny GPT whose per-layer cost is amplified (synthetic_delay) so the
         timing leak clears CPU noise -- fast, for smoke tests and CI.
  full   the notebook's 124M-parameter GPT-2 with real (un-amplified) latency.
"""

from dataclasses import dataclass, field
from typing import Callable

from sklearn.linear_model import LinearRegression, LogisticRegression

from .model import GPTConfig


def _default_surrogate():
    # The attacker's model class for the label-only attack. Injected (not
    # hard-coded in the attack) so a different hypothesis class can be swapped in
    # without touching the algorithm.
    return LogisticRegression(max_iter=2000)


def _default_regressor():
    # The hypothesis class for the logit-based attack: fit the confidence surface.
    return LinearRegression()


@dataclass
class AttackConfig:
    gpt: GPTConfig

    # Target
    exit_after_layer: int = 2
    context_len: int = 8
    synthetic_delay_per_layer: float = 0.0

    # Attack loop
    query_budget_multiplier: int = 5     # budget = this * (# secret params)
    n_rounds: int = 15
    probe_scale: float = 3.0             # std of the random query distribution
    refine_jitter: float = 0.5          # noise added to boundary-projected probes

    # Calibration / oracle
    calib_lo: float = -6.0
    calib_hi: float = 6.0
    calib_reps: int = 5
    oracle_reps: int = 1

    seed: int = 1
    surrogate_factory: Callable = field(default=_default_surrogate)   # label attack
    regressor_factory: Callable = field(default=_default_regressor)   # logit attack (linear)

    # MLP logit attack (nonlinear surrogate + Jacobian active learning)
    mlp_hidden: tuple = (64,)
    mlp_epochs: int = 400
    mlp_lr: float = 1e-2
    mlp_weight_decay: float = 1e-3

    @classmethod
    def preset(cls, mode: str) -> "AttackConfig":
        if mode == "small":
            gpt = GPTConfig(block_size=16, vocab_size=100, n_layer=6,
                            n_head=4, n_embd=32, bias=True)
            # per-layer cost on a 32-dim model is sub-microsecond; amplify so the
            # 2-vs-6 layer difference is measurable above wall-clock noise.
            return cls(gpt=gpt, exit_after_layer=2, synthetic_delay_per_layer=2e-4)
        if mode == "full":
            gpt = GPTConfig(block_size=256)   # 124M GPT-2, 12 layers
            return cls(gpt=gpt, exit_after_layer=2, synthetic_delay_per_layer=0.0)
        raise ValueError(f"unknown mode {mode!r} (expected 'small' or 'full')")
