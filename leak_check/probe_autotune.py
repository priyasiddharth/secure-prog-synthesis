"""
Probe 1: can max-autotune leak the secret through COMPILE-TIME kernel selection?

For each secret class (zero, random) present at compile time, force a fresh
autotune, capture the generated C++ (TORCH_LOGS=output_code), then time the chosen
kernel on an identical neutral w0. Decisive channel = the generated code diff:

    identical normalized codegen -> kernel choice is value-INDEPENDENT (no leak)
    different codegen            -> the secret steered kernel selection (leak)

Timing on w0 is a corroborating channel.
"""
import os, re, subprocess, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 512


def normalize(code):
    """Strip run-to-run noise (temp paths, cache hashes, kernel-name suffixes,
    hex addresses) so two structurally-identical kernels compare equal."""
    out = []
    # Strip the TORCH_LOGS line prefix "V0703 HH:MM:SS.mmm PID <path>] [..] [__output_code] "
    # (timestamp + PID vary every run and would make every line spuriously differ).
    prefix = re.compile(r"^V\d+ [\d:.]+ \d+ [^\]]*\] \[[^\]]*\] \[__output_code\] ?")
    for line in code.splitlines():
        line = prefix.sub("", line)
        if any(s in line for s in ("/tmp", "codecache", "async_compile",
                                   "AsyncCompile", ".so", "torch version",
                                   "# Topologically", "from torch")):
            continue
        line = re.sub(r"c[a-z0-9]{20,}", "HASH", line)          # cache hashes
        line = re.sub(r"(cpp_fused\w*?_)[0-9a-f]+", r"\1X", line)  # kernel ids
        line = re.sub(r"0x[0-9a-f]+", "0xADDR", line)
        line = re.sub(r"kernel_\w+", "kernel_X", line)
        if line.strip():
            out.append(line.rstrip())
    return "\n".join(out)


def run(secret, w0_path, out_npy):
    env = dict(os.environ, TORCH_LOGS="output_code", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "_autotune_worker.py", secret, w0_path, out_npy],
                       cwd=HERE, capture_output=True, text=True, env=env, timeout=1800)
    median = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            median = float(line.split("median_us=")[1])
    # output_code is emitted on stderr by the logging system
    code = normalize(p.stderr)
    if median is None:
        raise RuntimeError(f"{secret}: no RESULT\nSTDERR tail:\n{p.stderr[-1000:]}")
    return median, code


def main():
    w0_path = os.path.join(HERE, "secrets", "w0.npy")
    os.makedirs(os.path.dirname(w0_path), exist_ok=True)
    np.save(w0_path, np.random.default_rng(7).standard_normal((DIM, DIM), dtype=np.float32))

    print("Probe 1 — max-autotune compile-time kernel selection vs secret\n")
    med_z, code_z = run("zero", w0_path, os.path.join(HERE, "secrets", "at_zero.npy"))
    med_r, code_r = run("random", w0_path, os.path.join(HERE, "secrets", "at_random.npy"))

    same_code = code_z == code_r
    print(f"  zero   : compiled kernel median-on-w0 = {med_z:.2f} us")
    print(f"  random : compiled kernel median-on-w0 = {med_r:.2f} us")
    print(f"  generated code identical (normalized): {same_code}")

    tz = np.load(os.path.join(HERE, "secrets", "at_zero.npy"))
    tr = np.load(os.path.join(HERE, "secrets", "at_random.npy"))
    u, p = stats.mannwhitneyu(tr, tz, alternative="two-sided")
    auc = u / (len(tz) * len(tr))
    print(f"  timing-on-w0 AUC(zero-compile vs random-compile) = {auc:.3f} "
          f"(0.5 = same kernel)")

    print()
    if same_code:
        print("=> NO compile-time leak: max-autotune selected the SAME kernel "
              "regardless of the secret present at compile time (value-independent).")
    else:
        print("=> POSSIBLE compile-time leak: the secret changed the generated kernel. "
              "Writing diffs to secrets/at_code_{zero,random}.txt for inspection.")
        with open(os.path.join(HERE, "secrets", "at_code_zero.txt"), "w") as f:
            f.write(code_z)
        with open(os.path.join(HERE, "secrets", "at_code_random.txt"), "w") as f:
            f.write(code_r)


if __name__ == "__main__":
    main()
