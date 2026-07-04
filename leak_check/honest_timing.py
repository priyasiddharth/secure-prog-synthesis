"""
Honest side-channel harness for the "torch.compile leaks secret-weight entropy
via latency" claim.

The original demo asserted its conclusion from two noisy mean() values and an
arbitrary threshold. It also conflated two very different causes of a timing
difference:

  (1) An AUTHORED algorithmic short-circuit ("if the weights are all zero, skip
      the matmul"). This is a data-dependent branch the *model author* wrote. It
      is observable in plain eager Python -- no compiler required.

  (2) A COMPILER-INTRODUCED signature: the hypothesis that torch.compile /
      Inductor lowers a value-INDEPENDENT operation into value-DEPENDENT machine
      code, thereby manufacturing a leak that did not exist in eager mode.

To tell these apart we sweep a 2 x 2 x 2 matrix:

    Model   : Branchless (always x @ w)   vs   Branched (skip matmul via torch.cond)
    Backend : Eager                       vs   torch.compile (Inductor)
    Secret  : Zero weights                vs   Random weights

Reasoning:
  * Branchless is the CONTROL. A dense matmul does the same FLOPs regardless of
    values. If it is data-oblivious in eager AND stays oblivious after compile,
    the compiler is not manufacturing a leak on its own.
  * Branched uses torch.cond (NOT a Python `if`), so it actually compiles with
    fullgraph=True. If it leaks in BOTH eager and compiled, the leak is authored,
    not compiler-introduced.

Leakage metric: attacker AUC -- the probability that a single latency sample lets
an attacker classify zero vs random weights (equivalently the normalized
Mann-Whitney U statistic). 0.5 = indistinguishable, 1.0 = perfectly leaking.
This answers "can an attacker actually tell?", unlike a raw delta in seconds.
"""

import time
import torch
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Pin the environment to cut variance (single thread, fixed seed, CPU).
# ---------------------------------------------------------------------------
torch.manual_seed(0)
torch.set_num_threads(1)
device = torch.device("cpu")

DIM = 4096
WARMUP = 50       # calls discarded per condition (compile + cache warm)
ITERS = 800       # measured calls per condition


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class BranchlessLinear(torch.nn.Module):
    """Control: always performs the matmul, no data-dependent control flow."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(DIM, DIM, device=device))

    def forward(self, x):
        return torch.matmul(x, self.weight)


class BranchedLinear(torch.nn.Module):
    """
    Authored short-circuit expressed with torch.cond so that it is actually
    traceable/compilable (a plain Python `if` on tensor values is not).
    torch.cond executes only the taken branch at runtime, in both eager and
    compiled modes -- so the algorithmic leak is preserved either way.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(DIM, DIM, device=device))

    def forward(self, x):
        def skip(x, w):
            return torch.zeros(x.shape[0], w.shape[1], device=device)

        def compute(x, w):
            return torch.matmul(x, w)

        pred = torch.all(self.weight == 0)
        return torch.cond(pred, skip, compute, (x, self.weight))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def set_secret(model, kind):
    with torch.no_grad():
        if kind == "zero":
            model.weight.zero_()
        else:
            model.weight.normal_()


def benchmark(fn, x, model, secret, iters=ITERS, warmup=WARMUP):
    """Return an array of per-call wall-clock times (seconds)."""
    set_secret(model, secret)
    for _ in range(warmup):
        fn(x)
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        fn(x)
        times[i] = time.perf_counter() - t0
    return times


def attacker_auc(t_zero, t_rand):
    """
    AUC = P(a random-weight sample is slower than a zero-weight sample), the
    normalized Mann-Whitney U. 0.5 => indistinguishable. We report the
    two-sided leakage strength |AUC - 0.5| * 2 in the verdict.
    """
    u, p = stats.mannwhitneyu(t_rand, t_zero, alternative="two-sided")
    auc = u / (len(t_rand) * len(t_zero))
    return auc, p


def summarize(name, t_zero, t_rand):
    med_z, med_r = np.median(t_zero) * 1e6, np.median(t_rand) * 1e6  # microseconds
    iqr_z = (np.percentile(t_zero, 75) - np.percentile(t_zero, 25)) * 1e6
    iqr_r = (np.percentile(t_rand, 75) - np.percentile(t_rand, 25)) * 1e6
    auc, p = attacker_auc(t_zero, t_rand)
    strength = abs(auc - 0.5) * 2
    print(f"  {name}")
    print(f"    zero   : median {med_z:8.1f} us   IQR {iqr_z:7.1f} us")
    print(f"    random : median {med_r:8.1f} us   IQR {iqr_r:7.1f} us")
    print(f"    attacker AUC = {auc:.3f}  (leakage strength {strength:.3f})   "
          f"MWU p = {p:.1e}")
    return auc, strength, p


# ---------------------------------------------------------------------------
# Interleaved trials: alternate zero/random so thermal/frequency drift over the
# run cannot masquerade as a value-dependent signal.
# ---------------------------------------------------------------------------
def interleaved_benchmark(fn, x, model, iters=ITERS, warmup=WARMUP):
    # warm both secrets
    for secret in ("random", "zero"):
        set_secret(model, secret)
        for _ in range(warmup):
            fn(x)
    t_zero = np.empty(iters)
    t_rand = np.empty(iters)
    for i in range(iters):
        # random half
        set_secret(model, "random")
        t0 = time.perf_counter(); fn(x); t_rand[i] = time.perf_counter() - t0
        # zero half
        set_secret(model, "zero")
        t0 = time.perf_counter(); fn(x); t_zero[i] = time.perf_counter() - t0
    return t_zero, t_rand


def run_case(title, model_ctor):
    print("=" * 72)
    print(title)
    print("=" * 72)
    x = torch.randn(1, DIM, device=device)

    # --- Eager ---
    model = model_ctor()
    tz, tr = interleaved_benchmark(lambda inp: model(inp), x, model)
    auc_e, str_e, p_e = summarize("Eager (backend=aten)", tz, tr)

    # --- Compiled (Inductor) ---
    model_c = model_ctor()
    compiled = torch.compile(model_c, fullgraph=True)
    tz, tr = interleaved_benchmark(lambda inp: compiled(inp), x, model_c)
    auc_c, str_c, p_c = summarize("Compiled (backend=inductor)", tz, tr)

    print()
    return dict(eager=str_e, compiled=str_c)


def verdict(name, res, threshold=0.2):
    """threshold on leakage strength |AUC-0.5|*2 for calling something a leak."""
    e_leak = res["eager"] > threshold
    c_leak = res["compiled"] > threshold
    print(f"[{name}]")
    print(f"    eager    leakage strength = {res['eager']:.3f}  -> "
          f"{'LEAKS' if e_leak else 'no detectable leak'}")
    print(f"    compiled leakage strength = {res['compiled']:.3f}  -> "
          f"{'LEAKS' if c_leak else 'no detectable leak'}")
    if e_leak and c_leak:
        print("    => Leak present in BOTH -> AUTHORED (algorithmic), not "
              "compiler-introduced.")
    elif (not e_leak) and c_leak:
        print("    => Leak appears ONLY after compilation -> COMPILER-INTRODUCED.")
    elif e_leak and not c_leak:
        print("    => Leak in eager, gone after compile -> compiler removed it.")
    else:
        print("    => No detectable value-dependent timing in either mode.")
    print()


if __name__ == "__main__":
    print("threads =", torch.get_num_threads(), "| iters =", ITERS,
          "| warmup =", WARMUP, "| dim =", DIM)
    print()
    res_branchless = run_case(
        "CONTROL: Branchless (always x @ w, no data-dependent control flow)",
        BranchlessLinear)
    res_branched = run_case(
        "AUTHORED BRANCH: skip matmul when weights==0, via torch.cond",
        BranchedLinear)

    print("=" * 72)
    print("VERDICTS  (leakage strength = |attacker_AUC - 0.5| * 2; "
          "0 = oblivious, 1 = perfect leak)")
    print("=" * 72)
    verdict("Branchless / control", res_branchless)
    verdict("Authored branch (torch.cond)", res_branched)

    print("Interpretation guide:")
    print("  * Large-N Mann-Whitney p-values will often be < 0.05 even when the")
    print("    effect is negligible -- trust the AUC/leakage strength, not p.")
    print("  * A null result means 'not detectable with this harness', not")
    print("    'provably constant-time'. CPU wall-clock timing has a noise floor.")
