"""
The differential non-interference criterion.

For one model we measure two secret classes (zero vs random) under two builds
(eager vs compiled), on two deterministic channels (callgrind Ir/Bc and memcheck
taint). A channel "distinguishes" the classes for a build iff:
    - callgrind: the instruction OR branch count differs between zero and random,
      or
    - memcheck : the secret produced a control-flow/address dependence.

Verdict:
    distinguishable under BOTH builds        -> "authored"   (source's fault)
    distinguishable ONLY after compilation   -> "compiler-introduced"
    distinguishable ONLY in eager            -> "compiler-removed"
    distinguishable under NEITHER            -> "oblivious"
"""

import instruments as I


def _measure_build(model, zero, rand, compile):
    cg_zero = I.callgrind_count(model, zero, compile=compile)
    cg_rand = I.callgrind_count(model, rand, compile=compile)
    ir_diff = cg_rand["Ir"] - cg_zero["Ir"]
    bc_diff = cg_rand["Bc"] - cg_zero["Bc"]
    # Taint only needs the random secret; a dependence there is a dependence.
    taint = I.memcheck_taint(model, rand, compile=compile)
    distinguishable = (ir_diff != 0) or (bc_diff != 0) or taint["leak"]
    return {
        "cg_zero": cg_zero, "cg_rand": cg_rand,
        "ir_diff": ir_diff, "bc_diff": bc_diff,
        "taint": taint, "distinguishable": distinguishable,
    }


def analyze(model, zero, rand):
    eager = _measure_build(model, zero, rand, compile=False)
    comp = _measure_build(model, zero, rand, compile=True)
    de, dc = eager["distinguishable"], comp["distinguishable"]
    if de and dc:
        verdict = "authored"
    elif dc and not de:
        verdict = "compiler-introduced"
    elif de and not dc:
        verdict = "compiler-removed"
    else:
        verdict = "oblivious"
    return {"model": model, "eager": eager, "compiled": comp, "verdict": verdict}
