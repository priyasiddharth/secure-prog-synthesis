"""
Worker: measure the normal-vs-denormal timing leak on x@w under one backend
config. Run in its own process so that flush-to-zero MXCSR state set by loading a
-ffast-math kernel cannot contaminate the other conditions.

    python _denorm_worker.py {eager|compiled|compiled_fastmath}

Prints one line:
    RESULT mode=.. normal_us=.. denormal_us=.. slowdown=.. auc=.. strength=..
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from scipy import stats

torch.manual_seed(0)
torch.set_num_threads(1)

DIM = 4096
ITERS = 300
WARMUP = 25
NORMAL_VAL = 1e-20     # tiny but NORMAL float32
DENORMAL_VAL = 1e-40   # subnormal float32


def mkw(kind):
    rng = np.random.default_rng(0)
    signs = rng.integers(0, 2, size=(DIM, DIM)).astype(np.float32) * 2 - 1
    return (signs * (NORMAL_VAL if kind == "normal" else DENORMAL_VAL)).astype(np.float32)


class Branchless(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))

    def forward(self, x):
        return torch.matmul(x, self.weight)


def set_w(m, w):
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w))


def main():
    mode = sys.argv[1]
    if mode == "compiled_fastmath":
        import torch._inductor.config as ic
        ic.cpp.enable_unsafe_math_opt_flag = True          # -ffast-math family
        try:
            ic.cpp.enable_floating_point_contract_flag = "fast"
        except Exception:
            pass

    m = Branchless()
    fn = torch.compile(m, fullgraph=True) if mode.startswith("compiled") else m

    wN, wD = mkw("normal"), mkw("denormal")
    x = torch.randn(1, DIM)
    for w in (wN, wD):
        set_w(m, w)
        for _ in range(WARMUP):
            fn(x)

    tN, tD = np.empty(ITERS), np.empty(ITERS)
    for i in range(ITERS):
        set_w(m, wN); t0 = time.perf_counter(); fn(x); tN[i] = time.perf_counter() - t0
        set_w(m, wD); t0 = time.perf_counter(); fn(x); tD[i] = time.perf_counter() - t0

    u, p = stats.mannwhitneyu(tD, tN, alternative="two-sided")
    auc = u / (ITERS * ITERS)
    strength = abs(auc - 0.5) * 2
    print(f"RESULT mode={mode} normal_us={np.median(tN)*1e6:.1f} "
          f"denormal_us={np.median(tD)*1e6:.1f} "
          f"slowdown={np.median(tD)/np.median(tN):.2f} "
          f"auc={auc:.3f} strength={strength:.3f}")


if __name__ == "__main__":
    main()
