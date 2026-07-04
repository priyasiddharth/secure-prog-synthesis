"""
Probe 2: does the compiler MOVE the denormal timing leak via fast-math / FTZ?

Runs the normal-vs-denormal timing test under three backends, each in a fresh
process (so FTZ MXCSR state can't leak between them):
    eager, compiled (default Inductor), compiled_fastmath (Inductor + -ffast-math)

Non-interference reading on the timing channel:
    eager leaks, compiled_fastmath clean -> compiler REMOVED the leak (set FTZ)
    eager clean, compiled leaks          -> compiler INTRODUCED it
    same in all                          -> not the compiler's doing
"""
import re
import subprocess
import sys

HERE = __import__("os").path.dirname(__file__)
MODES = ["eager", "compiled", "compiled_fastmath"]


def run(mode):
    p = subprocess.run([sys.executable, "_denorm_worker.py", mode],
                       cwd=HERE, capture_output=True, text=True, timeout=1200)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return dict(re.findall(r"(\w+)=([^\s]+)", line))
    raise RuntimeError(f"{mode} produced no RESULT:\n{p.stdout}\n{p.stderr[-800:]}")


def main():
    print("Probe 2 — fast-math / FTZ on the denormal leak (branchless x@w)\n")
    rows = {m: run(m) for m in MODES}
    print(f"{'mode':<20}{'normal us':>12}{'denormal us':>14}{'slowdown':>10}"
          f"{'AUC':>8}{'strength':>10}  verdict")
    leaks = {}
    for m in MODES:
        r = rows[m]
        strength = float(r["strength"])
        leaks[m] = strength > 0.2
        print(f"{m:<20}{float(r['normal_us']):>12.1f}{float(r['denormal_us']):>14.1f}"
              f"{float(r['slowdown']):>10.2f}{float(r['auc']):>8.3f}{strength:>10.3f}"
              f"  {'LEAKS' if leaks[m] else 'clean'}")

    print()
    if leaks["eager"] and not leaks["compiled_fastmath"]:
        print("=> COMPILER-REMOVED: -ffast-math flushes denormals (FTZ), erasing a "
              "leak present in eager.")
    elif not leaks["eager"] and leaks["compiled"]:
        print("=> COMPILER-INTRODUCED timing leak.")
    elif leaks["eager"] and leaks["compiled"] and leaks["compiled_fastmath"]:
        print("=> Leak in every build -> hardware effect; default Inductor does NOT "
              "set FTZ, and even -ffast-math did not here.")
    else:
        print("=> Mixed/none; see table.")


if __name__ == "__main__":
    main()
