"""
Activation-function differential sweep (CSI-NN mechanism 2).

For each activation, generate its two baited secret classes, then reuse the
differential non-interference criterion (noninterference.analyze) to decide, on
the deterministic channels (callgrind Ir/Bc + memcheck taint), whether execution
distinguishes the classes under eager vs compiled — and thus whether Inductor
introduced / removed / preserved a data-dependent activation path.

    python run_activations.py                     # focused default set
    python run_activations.py relu exp gelu ...   # subset
"""
import os
import sys

import corpus_activations as CA
import noninterference as NI

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.join(HERE, "secrets", "act")

# Focused default: sign-branch, a transcendental, a compound, and a reduction.
DEFAULT = ["relu", "exp", "gelu", "softmax"]


def gen(name):
    os.makedirs(SEC, exist_ok=True)
    (la, a), (lb, b) = CA.secret_classes(name)
    pa = os.path.join(SEC, f"{name}_{la}.npy")
    pb = os.path.join(SEC, f"{name}_{lb}.npy")
    import numpy as np
    np.save(pa, a); np.save(pb, b)
    return la, pa, lb, pb


def main():
    acts = sys.argv[1:] or DEFAULT
    print("Activation differential sweep (eager vs Inductor) | channels: "
          "callgrind Ir/Bc + memcheck taint\n")
    rows = []
    for act in acts:
        la, pa, lb, pb = gen(act)
        print("=" * 74)
        print(f"MODEL: {act}   classes: {la} vs {lb}")
        print("=" * 74)
        res = NI.analyze(act, pa, pb)  # class A = pa (=zero slot), B = pb (=rand slot)
        for build in ("eager", "compiled"):
            b = res[build]
            chan = []
            if b["ir_diff"]: chan.append(f"dIr={b['ir_diff']:+d}")
            if b["bc_diff"]: chan.append(f"dBc={b['bc_diff']:+d}")
            if b["taint"]["leak"]: chan.append("taint")
            verd = "DISTINGUISHABLE" if b["distinguishable"] else "oblivious"
            print(f"  {build:<9}: {verd:<16} {' '.join(chan) or 'clean'}")
        print(f"  VERDICT: {res['verdict'].upper()}\n")
        rows.append((act, res["verdict"],
                     res["eager"]["distinguishable"],
                     res["compiled"]["distinguishable"]))

    print("=" * 74)
    print("SUMMARY  (eager? / compiled?  ->  verdict)")
    print("=" * 74)
    for act, verdict, de, dc in rows:
        print(f"  {act:<10} eager={'L' if de else '.'}  compiled={'L' if dc else '.'}"
              f"   -> {verdict}")


if __name__ == "__main__":
    main()
