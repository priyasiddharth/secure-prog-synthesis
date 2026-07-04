"""
Toolchain leakage sweep: compile the same C kernels with {gcc, clang} x flag
sweep, and for each build run the differential non-interference test on the
deterministic channels (callgrind Ir/Bc/Dw + memcheck taint).

The point: the SAME source flips between leaks/oblivious depending on
(compiler, flags) -- so a clean toolchain does not imply "compilation is safe".

    python run_ct.py                 # full sweep
    python run_ct.py relu select_ct  # subset of kernels
"""
import os
import sys
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import instruments as I

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
SEC = os.path.join(HERE, "secrets")

MM_N, EL_N = 256, 4096

COMPILERS = ["gcc", "clang"]
FLAGS = [
    ("O0", ["-O0"]),
    ("O2", ["-O2"]),
    ("O3", ["-O3"]),
    ("O3-native", ["-O3", "-march=native"]),
    ("Ofast-native", ["-Ofast", "-march=native"]),
]

# Two secret classes per kernel: identical size, differ only in protected values.
def gen_secrets():
    os.makedirs(SEC, exist_ok=True)
    rng = np.random.default_rng(0)
    S = {
        "matmul": (np.zeros((MM_N, MM_N), np.float32),
                   rng.standard_normal((MM_N, MM_N), dtype=np.float32)),
        "relu": (np.full(EL_N, -1.0, np.float32),      # all take else-branch
                 np.full(EL_N, 1.0, np.float32)),       # all take if-branch
        "select_branch": (np.zeros(EL_N, np.uint8), np.ones(EL_N, np.uint8)),
        "select_ct": (np.zeros(EL_N, np.uint8), np.ones(EL_N, np.uint8)),
        "memeq": (_memeq_A(), _memeq_B()),
    }
    paths = {}
    for k, (a, b) in S.items():
        pa = os.path.join(SEC, f"{k}_A.bin")
        pb = os.path.join(SEC, f"{k}_B.bin")
        a.tofile(pa); b.tofile(pb)
        paths[k] = (pa, pb)
    return paths


def _memeq_A():                      # guess is 0xAA; A mismatches at byte 0 -> exit fast
    v = np.zeros(EL_N, np.uint8); v[0] = 0x00; return v


def _memeq_B():                      # B matches until the last byte -> full scan
    v = np.full(EL_N, 0xAA, np.uint8); v[-1] = 0x00; return v


def compile_all(kernels):
    os.makedirs(BUILD, exist_ok=True)
    bins = {}
    for cc in COMPILERS:
        for fname, flags in FLAGS:
            out = os.path.join(BUILD, f"harness_{cc}_{fname}")
            cmd = [cc, *flags, "-o", out,
                   os.path.join(HERE, "harness.c"), os.path.join(HERE, "kernels.c")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [compile FAIL] {cc} {fname}: {r.stderr.strip()[:200]}")
                continue
            bins[(cc, fname)] = out
    return bins


def analyze(binpath, kernel, pa, pb):
    ca = I.callgrind_count_cmd([binpath, kernel, pa, "count"], cache_sim=True)
    cb = I.callgrind_count_cmd([binpath, kernel, pb, "count"], cache_sim=True)
    taint = I.memcheck_taint_cmd([binpath, kernel, pb, "taint"])
    dIr, dBc, dDw = cb["Ir"] - ca["Ir"], cb["Bc"] - ca["Bc"], cb["Dw"] - ca["Dw"]
    leak = (dIr != 0) or (dBc != 0) or taint["leak"]
    return dict(dIr=dIr, dBc=dBc, dDw=dDw, taint=taint["leak"], leak=leak,
                Dw_A=ca["Dw"], Dw_B=cb["Dw"])


def main():
    kernels = sys.argv[1:] or ["matmul", "relu", "select_branch", "select_ct", "memeq"]
    paths = gen_secrets()
    print("Compiling ...")
    bins = compile_all(kernels)
    configs = [(cc, f) for cc in COMPILERS for f, _ in FLAGS if (cc, f) in bins]

    # Matrix: rows = kernels, cols = (compiler, flags); L = leaks, . = oblivious
    header = "  ".join(f"{cc[:1]}:{f}" for cc, f in configs)
    print(f"\nLEAK MATRIX (L = distinguishable, . = oblivious)\n")
    print(f"{'kernel':<15} {header}")
    details = []
    for k in kernels:
        pa, pb = paths[k]
        cells = []
        for cc, f in configs:
            r = analyze(bins[(cc, f)], k, pa, pb)
            cells.append(("L" if r["leak"] else ".", r, (cc, f)))
        row = "  ".join(f"{sym:>{max(3,len(cc[:1])+len(f)+2)}}"
                        for (sym, _, _), (cc, f) in zip(cells, configs))
        print(f"{k:<15} {row}")
        for sym, r, (cc, f) in cells:
            details.append((k, cc, f, r))

    print("\nDETAIL (dIr/dBc/dDw across classes; taint = control-flow dependence)")
    for k, cc, f, r in details:
        tag = "LEAK" if r["leak"] else "obliv"
        chan = []
        if r["dIr"]: chan.append(f"dIr={r['dIr']:+d}")
        if r["dBc"]: chan.append(f"dBc={r['dBc']:+d}")
        if r["taint"]: chan.append("taint")
        print(f"  {k:<15} {cc:<5} {f:<14} {tag:<6} {' '.join(chan) or 'clean':<24}"
              f"  Dw={r['Dw_A']}")


if __name__ == "__main__":
    main()
