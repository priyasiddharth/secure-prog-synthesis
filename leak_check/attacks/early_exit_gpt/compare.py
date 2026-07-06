"""
Three extraction attacks at matched query budgets:

  label   one bit per query (did it exit?), read through the NOISE-FREE
          layer-count oracle so the contrast isolates the signal, not timing noise
  logit   the confidence channel (arXiv:1907.00713), linear regressor
  mlp     the confidence channel with a nonlinear (MLP) surrogate + Jacobian
          active learning -- the paper's stronger method

    python -m attacks.early_exit_gpt.compare              # small preset (fast)

The confidence attacks beat the label attack (a continuous logit carries more
than one bit). Whether the *nonlinear* surrogate beats the *linear* one depends
on the target: on this early-exit gate the logit is close to linear in the query
(the residual stream carries the input through), so once the budget exceeds the
input dimension the linear regressor already reaches R^2 ~ 1 and the MLP adds
little. On the full model the effect is clearer -- see README (the linear fit is
under-determined, not the surface nonlinear, below ~dim queries).

Columns: accuracy = stolen-vs-true decision agreement; R^2 = fidelity of the
recovered confidence surface (logit attacks only).
"""

from dataclasses import replace

from .config import AttackConfig
from .run import run_attack

BUDGET_MULTIPLIERS = (1, 2, 3, 5)   # x (gate parameters)


def main(mode="small"):
    base = AttackConfig.preset(mode)
    print(f"target: {mode} preset (dim={base.gpt.n_embd}) | "
          f"label channel: layercount (noise-free) | logit channel: gate confidence\n")
    header = (f"{'budget':>7} {'queries':>8} | {'label acc':>9} | "
              f"{'lin acc':>8} {'lin R^2':>8} | {'mlp acc':>8} {'mlp R^2':>8}")
    print(header)
    print("-" * len(header))
    for mult in BUDGET_MULTIPLIERS:
        cfg = replace(base, query_budget_multiplier=mult)
        _, lab = run_attack(cfg, attack="label", oracle_kind="layercount", verbose=False)
        _, lin = run_attack(cfg, attack="logit", verbose=False)
        _, mlp = run_attack(cfg, attack="mlp", verbose=False)
        print(f"{mult:>6}x {lab['queries_used']:>8} | "
              f"{lab['surrogate_alignment'] * 100:>8.1f}% | "
              f"{lin['surrogate_alignment'] * 100:>7.1f}% {lin['logit_fidelity']:>8.3f} | "
              f"{mlp['surrogate_alignment'] * 100:>7.1f}% {mlp['logit_fidelity']:>8.3f}")


if __name__ == "__main__":
    main()
