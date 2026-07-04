# Methodology: is the *compiler* introducing an information leak?

Durable reference. This describes the method, not any particular run's numbers.

## The question, precisely

A program leaks a secret through a side channel when some **observable execution
property** depends on secret values. We want to attribute such a leak to the right
cause:

- **Authored / algorithmic leak** — the source code itself contains secret-dependent
  control flow (e.g. "skip the matmul when the weights are all zero"). Present in the
  reference semantics; the compiler merely preserves it.
- **Compiler-introduced leak** — the source computation is data-oblivious, but a
  compiler optimization lowers it into secret-dependent machine code, manufacturing a
  channel that did not exist before.

Only the second is the compiler's fault. Distinguishing them is the whole game.

### Leakage is a property of a CONFIG POINT, not of "compilation"
A verdict of "no compiler-introduced leak" is scoped to one point in
`(source op, compiler, backend, version, flags, target ISA, libraries)`. It does NOT
generalize to other toolchains. Empirically (see `../empirical/results.md`, toolchain
axis): the same C source flips between leaking and oblivious across `-O0…-Ofast`, and
**gcc and clang disagree at identical flags** (a per-element select branch that clang
vectorizes to a blend but gcc leaves as a data-dependent branch). Corollary: to certify
a deployment, measure the *exact* toolchain+flags it ships with; and treat the toolchain
as an experimental **axis**, not a constant. Optimization level also changes the
*physical* attack surface independently of digital leaks — higher `-O` keeps
intermediates in registers, collapsing the memory-store (`Dw`) surface a power/EM
(CSI-NN) attacker targets (~500× on matmul, `-O0`→`-O2`).

## Core principle: differential deterministic non-interference

A leak is **compiler-introduced** iff a computation that is *oblivious under the
reference build* becomes *secret-dependent after compilation*. Formalize as a
2×2 differential test over an observation channel:

- **Builds**: reference (eager / interpreter / `-O0`) vs compiled (Inductor / `-O2`).
- **Secret classes A, B** (see below).

Let `div(build)` = distinguishability of A vs B under the chosen channel.

| `div(reference)` | `div(compiled)` | Verdict |
|---|---|---|
| ≈ 0 | > 0 | **Compiler-introduced leak** |
| > 0 | > 0 | Authored / algorithmic (not the compiler) |
| ≈ 0 | ≈ 0 | Oblivious (no leak at this channel's resolution) |
| > 0 | ≈ 0 | Compiler *removed* an authored leak |

## Why two secret classes A and B

A single measurement is meaningless in isolation — an instruction count of "50,000"
says nothing about secret dependence. A leak only manifests as a **difference in the
observable between two secrets** that a constant-time program must treat identically
(same shape, same dtype, differing only in the protected bytes).

- Same observable for A and B → execution didn't depend on the secret → oblivious.
- Different observable → execution branched on secret values → that difference *is*
  the leak.

Choose A, B as **extremes** (all-zeros vs random), not two random draws, for
sensitivity: special values like all-zeros are exactly what trip the risky
optimizations below (constant-folding, sparse/skip fast paths, short-circuit
reductions). If any input can provoke a data-dependent path, a zero-vs-random contrast
is the most likely to expose it.

Two classes are also what make the *differential* verdict expressible: "indistinguishable
under eager but distinguishable after compile" needs a distinguishability measure,
which needs two points.

## Observation channels (strongest → weakest)

The channel is the execution property we read off to tell A from B.

1. **Taint tracking (gold standard).** Mark the secret bytes UNDEFINED (Valgrind
   `VALGRIND_MAKE_MEM_UNDEFINED`) and run under memcheck. It reports *"Conditional jump
   depends on uninitialised value"* / *"use of uninitialised value in address"* at the
   exact instruction whenever a secret influences **control flow or memory addressing**
   — the definitional side channel. Zero reports ⇒ that path is provably oblivious;
   any report localizes the leaking instruction. Needs no magnitude difference at all.

2. **Deterministic instruction / branch counts.** callgrind `Ir` (retired
   instructions) and `Bc`/`Bi` (conditional/indirect branches). Exactly reproducible,
   so a **single** run per class is decisive: `Ir(A) == Ir(B)` byte-for-byte ⇒
   oblivious at the instruction level; any difference is a real secret-dependent path.

3. **Wall-clock latency (weakest).** The dudect/TVLA-style statistical test — Welch
   t-test or the Mann-Whitney **attacker-AUC** (0.5 = indistinguishable, 1.0 = perfect
   leak). Real and attacker-observable through an API, but noisy: **detection only,
   never proof**. A null result means "not detected with this harness", not "constant
   time".

Prefer channels 1–2. `perf` hardware counters are an optional cross-check but are not
required — callgrind gives the same signal deterministically, in userspace, without
elevated privileges.

### The hierarchy is only partial — channels cover DIFFERENT leaks (learned the hard way)
"Strongest → weakest" applies **only to control-flow / instruction-count / addressing**
leaks. It is FALSE for **data-dependent cycle-count** leaks, where the instruction
stream is byte-identical but the *cycles per instruction* depend on the secret:
subnormal/denormal floats (~25× slower, measured), variable-latency instructions
(division, sqrt), cache/TLB/port contention. For those, channels 1–2 read `dIr=0` /
taint-clean — **provably "oblivious" and completely wrong** — while channel 3 (timing)
shows a perfect leak (denormal probe: AUC=1.000, invisible to Ir/taint). Lesson: the
channels are **complementary, not a ladder**. A clean Tier-1/2 result bounds only
*digital* obliviousness; it says nothing about *analog* (timing) leakage. Always run the
timing tier too, even when the deterministic channels are clean.

## Key enabling technique: isolate the generated kernel

Running Valgrind over a whole Python+torch process is slow and noisy (CPython branches
constantly). Two ways to isolate just the code under test:

- **Gate the instruments to the forward pass.** A tiny C shim (`vgshim.c`, loaded via
  `ctypes`) exposes the Valgrind client requests so the Python process can, from inside
  itself, toggle callgrind instrumentation (`cg_start`/`cg_stop`) and mark the secret
  buffer. All compile/warmup/allocation happens *outside* the measured region, so the
  channel sees only the kernel. This is what `leak_check/measured_run.py` does.
- **Emit a standalone artifact.** AOTInductor (`torch._inductor.aoti_compile_and_package`)
  compiles a model to a self-contained `.so`; drive it from a small C harness. Generalizes
  to any compiler: compile the secret-processing code to a standalone binary at the
  target optimization level, then apply channels 1–2 to it and to a `-O0` reference.

## Catalog of risky optimizations to target

What actually turns oblivious code into secret-dependent execution — design A/B classes
to trip these:

- Constant-folding / dead-code elimination on special values (all-zeros → skip work).
- Sparse and denormal (subnormal-float) fast paths.
- Value-dependent kernel autotuning / dispatch (a different kernel chosen by values).
- Guard specialization & recompilation keyed on secret-derived shapes.
- Short-circuit reductions (`any` / `all`).
- `select` / `where` lowered to a branch instead of a vectorized select.

## Controls & pitfalls

- **Same shape/dtype across A and B** — differ only in protected bytes. Otherwise you
  measure the difference in *setup* (e.g. `normal_()` does RNG work, `zero_()` doesn't),
  not the kernel. Generate both secrets identically and load them the same way.
- **Warm outside the measured region** — exclude compilation, autotuning, and guard
  overhead; measure steady state.
- **Data-dependent *value* is not a leak; data-dependent *execution* is.** A branch-free
  `where`/`select` produces a secret-dependent output through oblivious execution — the
  method must (and does) report it as no-leak. This is the `where_select` control.
- **Trust effect size, not p-values** on the timing channel: at large N a Mann-Whitney
  p can be < 0.05 while the AUC is ≈ 0.5 (negligible). Decide on AUC.
- **Determinism hygiene** for the counting channel: single-thread BLAS/OMP,
  `PYTHONHASHSEED=0`, warm the compile cache so measured runs load it rather than
  invoking `g++`.
- **Re-define derived outputs** after a tainted forward, so process teardown doesn't
  raise spurious memcheck reports from the deliberately-undefined bytes.

## Calibration (the harness must earn trust before it judges)

- Positive control: an explicit secret-dependent branch **must** raise a taint report
  and a nonzero count difference.
- Negative control: a branch-free constant-time version **must not**.
- The `leak_check/corpus.py` trio encodes expected verdicts: `branchless` → oblivious,
  `cond_skip` → authored (leaks in both builds), `where_select` → oblivious.
