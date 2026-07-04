"""
Smoke self-check, in the harness's "earn trust before you judge it" spirit.

Runs the small preset end-to-end on the *noise-free* LayerCountOracle -- which
removes timing noise so the assertion tests the attack logic, not the machine's
jitter -- and asserts the recovered surrogate matches the true gate.

    python -m attacks.early_exit_gpt.check

Exits 0 on success, nonzero on failure.
"""

import sys

from .config import AttackConfig
from .run import run_attack

SURROGATE_FLOOR = 0.85   # stolen-vs-true agreement on held-out queries


def main():
    config = AttackConfig.preset("small")
    _, m = run_attack(config, oracle_kind="layercount", verbose=False)

    checks = [
        ("noise-free oracle matches the true gate",
         m["oracle_alignment"] > 0.99),
        (f"surrogate recovers the gate (>= {SURROGATE_FLOOR:.0%})",
         m["surrogate_alignment"] >= SURROGATE_FLOOR),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {label}  "
              f"(oracle={m['oracle_alignment']:.3f}, "
              f"surrogate={m['surrogate_alignment']:.3f})")
        ok = ok and passed

    if not ok:
        print("self-check FAILED")
        return 1
    print("self-check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
