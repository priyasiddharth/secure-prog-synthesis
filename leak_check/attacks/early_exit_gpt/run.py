"""
Driver: assemble the target, oracle and attack, then report.

    python -m attacks.early_exit_gpt.run --mode small                       # label attack, timing channel
    python -m attacks.early_exit_gpt.run --mode small --oracle layercount   # label attack, noise-free control
    python -m attacks.early_exit_gpt.run --mode small --attack logit        # logit (confidence) attack
    python -m attacks.early_exit_gpt.run --mode full  --attack logit        # 124M GPT-2 (slow)

``--attack label`` learns from one bit per query (did it early-exit?), read
through ``--oracle`` (timing = realistic side channel, layercount = noise-free
control). ``--attack logit`` uses the stronger threat model of arXiv:1907.00713:
the API returns the gate's confidence, which the attacker regresses.

Run from the ``leak_check/`` directory.
"""

import argparse

import numpy as np
import torch

from .config import AttackConfig
from .enclave import EarlyExitEnclave
from .evaluate import logit_fidelity, oracle_alignment, surrogate_alignment
from .extraction import SurrogateExtractionAttack
from .logit_extraction import RegressionExtractionAttack
from .mlp_extraction import MlpLogitExtractionAttack
from .oracle import LayerCountOracle, LogitOracle, ThresholdLabeler, TimingOracle

# logit-channel attacks: attack name -> algorithm constructed from the config.
_LOGIT_ATTACKS = {"logit": RegressionExtractionAttack,
                  "mlp": MlpLogitExtractionAttack}

_LABEL_ORACLES = {"timing": lambda e, c: TimingOracle(e, reps=c.oracle_reps),
                  "layercount": lambda e, c: LayerCountOracle(e)}


def build_enclave(config):
    return EarlyExitEnclave(
        config.gpt, config.exit_after_layer, context_len=config.context_len,
        synthetic_delay_per_layer=config.synthetic_delay_per_layer, seed=config.seed)


def run_attack(config, attack="label", oracle_kind="timing", n_test=300, verbose=True):
    """Run one attack; return ``(result, metrics)``."""
    torch.set_num_threads(1)
    enclave = build_enclave(config)
    budget = config.query_budget_multiplier * enclave.secret_dim
    rng = np.random.default_rng(config.seed + 1)
    X_test = rng.normal(scale=config.probe_scale, size=(n_test, enclave.query_dim))

    metrics = {"attack": attack, "budget": budget,
               "secret_dim": enclave.secret_dim}

    if attack == "label":
        oracle = _LABEL_ORACLES[oracle_kind](enclave, config)
        labeler = ThresholdLabeler(
            oracle, dim=enclave.query_dim, lo_fill=config.calib_lo,
            hi_fill=config.calib_hi, calib_reps=config.calib_reps)
        labeler.calibrate()
        result = SurrogateExtractionAttack(config).run(
            labeler.label, dim=enclave.query_dim, budget=budget)
        metrics["channel"] = oracle_kind
        metrics["oracle_alignment"] = oracle_alignment(labeler, enclave, X_test)
    elif attack in _LOGIT_ATTACKS:
        oracle = LogitOracle(enclave)
        result = _LOGIT_ATTACKS[attack](config).run(
            oracle.query, dim=enclave.query_dim, budget=budget)
        metrics["channel"] = "logit"
        metrics["logit_fidelity"] = logit_fidelity(result, oracle, X_test)
    else:
        raise ValueError(
            f"unknown attack {attack!r} (expected 'label', 'logit', or 'mlp')")

    metrics["queries_used"] = result.queries_used
    metrics["surrogate_alignment"] = surrogate_alignment(result, enclave, X_test)
    if verbose:
        _report(config, metrics)
    return result, metrics


def _report(config, m):
    g = config.gpt
    print(f"target   : GPT n_layer={g.n_layer} n_embd={g.n_embd}, "
          f"early exit after layer {config.exit_after_layer}")
    print(f"secret   : early-exit gate, {m['secret_dim']} parameters")
    print(f"attack   : {m['attack']} (channel: {m['channel']})")
    print(f"queries  : {m['queries_used']} "
          f"({m['queries_used'] / m['secret_dim']:.1f}x gate parameters)")
    if "oracle_alignment" in m:
        print(f"oracle alignment    (labels vs true gate) : "
              f"{m['oracle_alignment'] * 100:.1f}%")
    if "logit_fidelity" in m:
        print(f"logit fidelity      (R^2 vs true logit)   : {m['logit_fidelity']:.3f}")
    print(f"surrogate alignment (stolen vs true gate) : "
          f"{m['surrogate_alignment'] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["small", "full"], default="small")
    ap.add_argument("--attack", choices=["label", "logit", "mlp"], default="label")
    ap.add_argument("--oracle", choices=["timing", "layercount"], default="timing",
                    help="label channel; ignored for --attack logit")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    config = AttackConfig.preset(args.mode)
    if args.seed is not None:
        config.seed = args.seed
    run_attack(config, attack=args.attack, oracle_kind=args.oracle)


if __name__ == "__main__":
    main()
