"""
Smoke self-check, in the harness's "earn trust before you judge it" spirit.

Runs both attacks on the small preset over NOISE-FREE oracles (layer count for
the label attack, gate confidence for the logit attack) so the assertions test
the attack logic, not the machine's timing jitter. Checks that each recovers the
gate and that the logit attack is at least as accurate as the label attack on the
same underlying signal (the arXiv:1907.00713 thesis).

    python -m attacks.early_exit_gpt.check

Exits 0 on success, nonzero on failure.
"""

import sys

from .config import AttackConfig
from .run import run_attack

LABEL_FLOOR = 0.85       # label-attack stolen-vs-true agreement
LOGIT_FLOOR = 0.95       # logit-attack agreement (richer signal -> higher)
FIDELITY_FLOOR = 0.90    # logit R^2 (small model's gate is near-linear in x)


def main():
    config = AttackConfig.preset("small")
    _, lab = run_attack(config, attack="label", oracle_kind="layercount", verbose=False)
    _, log = run_attack(config, attack="logit", verbose=False)
    _, mlp = run_attack(config, attack="mlp", verbose=False)

    checks = [
        ("noise-free label matches the true gate",
         lab["oracle_alignment"] > 0.99),
        (f"label attack recovers the gate (>= {LABEL_FLOOR:.0%})",
         lab["surrogate_alignment"] >= LABEL_FLOOR),
        (f"linear logit attack recovers the gate (>= {LOGIT_FLOOR:.0%})",
         log["surrogate_alignment"] >= LOGIT_FLOOR),
        (f"linear logit surface reconstructed (R^2 >= {FIDELITY_FLOOR})",
         log["logit_fidelity"] >= FIDELITY_FLOOR),
        ("logit attack >= label attack accuracy (confidence helps)",
         log["surrogate_alignment"] >= lab["surrogate_alignment"]),
        # The MLP method must run and recover the gate; we do NOT assert it beats
        # the linear surrogate -- on this near-linear target it does not, and the
        # honest result is reported rather than enforced.
        (f"mlp logit attack recovers the gate (>= {LOGIT_FLOOR:.0%})",
         mlp["surrogate_alignment"] >= LOGIT_FLOOR),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print(f"  (label acc={lab['surrogate_alignment']:.3f}  "
          f"lin acc={log['surrogate_alignment']:.3f} R^2={log['logit_fidelity']:.3f}  "
          f"mlp acc={mlp['surrogate_alignment']:.3f} R^2={mlp['logit_fidelity']:.3f})")

    if not ok:
        print("self-check FAILED")
        return 1
    print("self-check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
