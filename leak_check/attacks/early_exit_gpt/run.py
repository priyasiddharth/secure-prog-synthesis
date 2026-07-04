"""
Driver: assemble the target, oracle, labeler and attack, then report.

    python -m attacks.early_exit_gpt.run --mode small
    python -m attacks.early_exit_gpt.run --mode small --oracle layercount
    python -m attacks.early_exit_gpt.run --mode full

Run from the ``leak_check/`` directory.
"""

import argparse

import numpy as np
import torch

from .config import AttackConfig
from .enclave import EarlyExitEnclave
from .evaluate import oracle_alignment, surrogate_alignment
from .extraction import SurrogateExtractionAttack
from .oracle import LayerCountOracle, ThresholdLabeler, TimingOracle

_ORACLES = {"timing": lambda e, c: TimingOracle(e, reps=c.oracle_reps),
            "layercount": lambda e, c: LayerCountOracle(e)}


def build(config, oracle_kind="timing"):
    """Wire target -> oracle -> labeler for the given config."""
    enclave = EarlyExitEnclave(
        config.gpt, config.exit_after_layer, context_len=config.context_len,
        synthetic_delay_per_layer=config.synthetic_delay_per_layer, seed=config.seed)
    if oracle_kind not in _ORACLES:
        raise ValueError(f"unknown oracle {oracle_kind!r} (expected {list(_ORACLES)})")
    oracle = _ORACLES[oracle_kind](enclave, config)
    labeler = ThresholdLabeler(
        oracle, dim=enclave.query_dim, lo_fill=config.calib_lo,
        hi_fill=config.calib_hi, calib_reps=config.calib_reps)
    return enclave, oracle, labeler


def run_attack(config, oracle_kind="timing", n_test=300, verbose=True):
    torch.set_num_threads(1)
    enclave, oracle, labeler = build(config, oracle_kind)
    labeler.calibrate()

    budget = config.query_budget_multiplier * enclave.secret_dim
    result = SurrogateExtractionAttack(config).run(
        labeler, dim=enclave.query_dim, budget=budget)

    rng = np.random.default_rng(config.seed + 1)
    X_test = rng.normal(scale=config.probe_scale, size=(n_test, enclave.query_dim))
    metrics = {
        "budget": budget,
        "queries_used": result.queries_used,
        "secret_dim": enclave.secret_dim,
        "oracle_alignment": oracle_alignment(labeler, enclave, X_test),
        "surrogate_alignment": surrogate_alignment(result, enclave, X_test),
    }
    if verbose:
        _report(config, oracle_kind, metrics)
    return result, metrics


def _report(config, oracle_kind, m):
    g = config.gpt
    print(f"target   : GPT n_layer={g.n_layer} n_embd={g.n_embd}, "
          f"early exit after layer {config.exit_after_layer}")
    print(f"secret   : early-exit gate, {m['secret_dim']} parameters")
    print(f"oracle   : {oracle_kind}")
    print(f"queries  : {m['queries_used']} "
          f"({m['queries_used'] / m['secret_dim']:.1f}x gate parameters)")
    print(f"oracle alignment    (labels vs true gate) : "
          f"{m['oracle_alignment'] * 100:.1f}%")
    print(f"surrogate alignment (stolen vs true gate) : "
          f"{m['surrogate_alignment'] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["small", "full"], default="small")
    ap.add_argument("--oracle", choices=["timing", "layercount"], default="timing")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    config = AttackConfig.preset(args.mode)
    if args.seed is not None:
        config.seed = args.seed
    run_attack(config, oracle_kind=args.oracle)


if __name__ == "__main__":
    main()
