"""
Adversarial probe for the empty quadrant: a leak the *compiler* can move, that
the deterministic channels (Ir/taint) CANNOT see, only timing can.

Subnormal (denormal) IEEE-754 floats are handled ~10-100x slower by the FPU than
normal floats, with the SAME instruction stream (no branch, same Ir, clean
taint). Whether that slowdown appears depends on flush-to-zero / denormals-are-
zero (FTZ/DAZ) MXCSR state, which a compiler can set (e.g. -ffast-math) or leave.

So this is the honest stress test the calibration corpus omitted:

  * Both secret classes drive the SAME branchless matmul (x @ w) -> deterministic
    channels read dIr=0, taint clean in every build (they are blind here).
  * Class A: tiny-but-NORMAL weights (~1e-20). Class B: DENORMAL weights (~1e-40).
    Same magnitude regime, so the only difference is the float *representation*.
  * We measure wall-clock for eager vs compiled and ask the non-interference
    question on the TIMING channel:
        eager clean, compiled leaky  -> compiler INTRODUCED a timing leak
        eager leaky, compiled clean  -> compiler REMOVED it (set FTZ)
        leaky in both / neither       -> not the compiler's doing
"""

import time
import numpy as np
import torch
from scipy import stats

torch.manual_seed(0)
torch.set_num_threads(1)

DIM = 4096
ITERS = 400
WARMUP = 30

# Class A: tiny but NORMAL float32 (smallest normal ~1.18e-38, so 1e-20 is normal).
# Class B: DENORMAL float32 (below 1.18e-38, e.g. ~1e-40 -> subnormal).
NORMAL_VAL = 1e-20
DENORMAL_VAL = 1e-40


def make_weight(kind):
    rng = np.random.default_rng(0)
    signs = rng.integers(0, 2, size=(DIM, DIM)).astype(np.float32) * 2 - 1
    base = NORMAL_VAL if kind == "normal" else DENORMAL_VAL
    w = (signs * base).astype(np.float32)
    # sanity: confirm the denormal class is actually subnormal
    return w


def is_subnormal(w):
    a = np.abs(w)
    return np.mean((a > 0) & (a < np.finfo(np.float32).tiny))


class Branchless(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))

    def forward(self, x):
        return torch.matmul(x, self.weight)


def set_w(model, w):
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(w))


def interleaved(fn, x, model, wA, wB):
    for w in (wA, wB):
        set_w(model, w)
        for _ in range(WARMUP):
            fn(x)
    tA = np.empty(ITERS)
    tB = np.empty(ITERS)
    for i in range(ITERS):
        set_w(model, wA)
        t0 = time.perf_counter(); fn(x); tA[i] = time.perf_counter() - t0
        set_w(model, wB)
        t0 = time.perf_counter(); fn(x); tB[i] = time.perf_counter() - t0
    return tA, tB


def auc(tA, tB):
    u, p = stats.mannwhitneyu(tB, tA, alternative="two-sided")
    a = u / (len(tA) * len(tB))
    return a, abs(a - 0.5) * 2, p


def report(tag, tA, tB):
    a, strength, p = auc(tA, tB)
    print(f"  {tag}")
    print(f"    normal(A)   median {np.median(tA)*1e6:9.1f} us")
    print(f"    denormal(B) median {np.median(tB)*1e6:9.1f} us   "
          f"({np.median(tB)/np.median(tA):.1f}x slower)")
    print(f"    attacker AUC={a:.3f}  leakage strength={strength:.3f}  p={p:.1e}"
          f"  -> {'LEAKS' if strength > 0.2 else 'clean'}")
    return strength


def main():
    wA, wB = make_weight("normal"), make_weight("denormal")
    print(f"DIM={DIM} iters={ITERS} | class A=normal(~{NORMAL_VAL:.0e}) "
          f"B=denormal(~{DENORMAL_VAL:.0e})")
    print(f"subnormal fraction: A={is_subnormal(wA):.2f}  B={is_subnormal(wB):.2f}\n")
    x = torch.randn(1, DIM)

    m = Branchless()
    s_e = report("EAGER (backend=aten)", *interleaved(lambda i: m(i), x, m, wA, wB))

    mc = Branchless()
    comp = torch.compile(mc, fullgraph=True)
    s_c = report("COMPILED (backend=inductor)",
                 *interleaved(lambda i: comp(i), x, mc, wA, wB))

    print("\nDeterministic channels here would read dIr=0 / taint clean in BOTH "
          "builds (same instructions).")
    el = s_e > 0.2
    cl = s_c > 0.2
    print(f"TIMING non-interference: eager={'LEAK' if el else 'clean'} "
          f"compiled={'LEAK' if cl else 'clean'}")
    if not el and cl:
        print("=> COMPILER-INTRODUCED timing leak (invisible to Ir/taint).")
    elif el and not cl:
        print("=> COMPILER-REMOVED (compiled path flushes denormals, e.g. FTZ).")
    elif el and cl:
        print("=> Denormal timing leak present in BOTH -> hardware effect, not the "
              "compiler's doing.")
    else:
        print("=> No timing leak either build (FTZ likely on everywhere).")


if __name__ == "__main__":
    main()
