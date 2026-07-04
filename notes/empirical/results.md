# Empirical results

Concrete measurements. These are machine- and version-specific (see
`../reference/environment.md`) and change on re-run.

## Activation-function corpus (CSI-NN mechanism 2)

Each model applies one activation elementwise to the secret weight; two baited input
classes probe libm's value-regime branches (relu: neg vs pos; transcendentals: small
`0.5` vs large `100.0`, i.e. overflow/saturation). Differential eager-vs-Inductor on the
deterministic channels. Code: `leak_check/corpus_activations.py`, `run_activations.py`.

| activation | eager | compiled | verdict | channels used |
|---|---|---|---|---|
| relu | oblivious | oblivious | **OBLIVIOUS** | Ir/Bc + taint |
| exp | **DISTINGUISHABLE** dIr=+35,061,760, dBc=+3,416,064, taint | oblivious, clean | **COMPILER-REMOVED** | Ir/Bc + taint |
| gelu | oblivious | oblivious | oblivious (count only) | Ir/Bc |
| softmax | oblivious | **dIr=-573, dBc=-93** | *UNVERIFIED lead* | Ir/Bc |

### Findings
1. **relu is already branchless in PyTorch** — unlike the naive C `relu` (which branches
   at `-O0`), the eager TensorIterator kernel is a vectorized `max`, so no sign leak in
   either build. Library kernels can be constant-time where hand code isn't.
2. **exp → COMPILER-REMOVED (the headline, fully confirmed on both channels).** Eager
   scalar `libm expf` branches hard on magnitude (35M-instruction difference between
   small and overflow inputs, taint fires) — exactly CSI-NN mechanism 2's premise.
   Inductor lowers it to a **branchless vectorized polynomial** → compiled is
   instruction-identical and taint-clean across classes. Compilation *erased* the
   activation-timing side channel that CSI NN exploits. This is the richest confirmation
   yet of the where_select/relu "compiler removes value-dependent control flow" pattern.
3. **gelu oblivious** on the count channel in both builds (taint not run in the fast pass).

### ⚠ softmax: an UNVERIFIED lead, NOT a finding
softmax is **oblivious in eager but shows a small `dIr=-573, dBc=-93` in compiled** —
the shape of a *compiler-introduced* leak (the empty quadrant). BUT: (a) it is tiny;
(b) both secret classes are uniform, so `x - max(x) = 0` for both — semantically they
should be identical, which makes a real value-dependent path surprising; (c) the taint
channel was NOT run for softmax; (d) we have already been burned once by an unverified
"compiler-introduced" signal (the max-autotune false positive from a normalizer bug).
**Do not record softmax as compiler-introduced until verified.** Required checks before
any claim: re-run the compiled softmax pair to confirm the count difference is
deterministic (callgrind is deterministic, so a stable ≠0 is real; a varying one is a
harness artifact); run the taint channel; inspect the generated softmax kernel for a
value-dependent branch (e.g. a denormal/relative-precision path in the vectorized exp).

### Method note (channel coverage of this pass)
relu/exp used both channels (full run); gelu/softmax used **callgrind only** (fast pass,
after the full memcheck run was killed at ~8 h — `torch.compile` under memcheck is
pathologically slow, ~hours/model). So gelu/softmax "oblivious" means
instruction-count-oblivious; a taint-only control-flow leak (equal count, different
branch direction — cf. the C `select_branch`) would be missed and must be checked before
finalizing.

## Toolchain axis: the SAME source leaks or not depending on (compiler, flags)

The headline experiment answering "could a clean toolchain differ from a leaky one?"
C testbed in `leak_check/ctbench/` (kernels.c, harness.c, run_ct.py): hand-written
kernels compiled by **gcc 13.3.0** and **clang 18.1.3** across a flag sweep, each build
run through the deterministic channels (callgrind Ir/Bc/Dw + memcheck taint). A cell
"leaks" if execution is distinguishable between two secret classes (dIr/dBc≠0 or taint).

Leak matrix (L = distinguishable, `.` = oblivious):

| kernel | g:O0 | g:O2 | g:O3 | g:O3-nat | g:Ofast | c:O0 | c:O2 | c:O3 | c:O3-nat | c:Ofast |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul (control) | . | . | . | . | . | . | . | . | . | . |
| relu | **L** | **L** | . | . | . | **L** | . | . | . | . |
| select_branch | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **L** | **.** | **.** |
| select_ct | . | . | . | . | . | . | . | . | . | . |
| memeq (early-exit) | L | L | L | L | L | L | L | L | L | L |

### Findings (this is the answer: YES, leakage is a toolchain property)
1. **The flip.** `relu` (a branch on sign) LEAKS at `-O0` (taint: the conditional jump
   depends on the secret) but goes oblivious once the compiler vectorizes it to a
   branchless `max` — gcc at `-O3`, clang already at `-O2`. Same source, opposite verdict.
2. **Cross-compiler divergence at identical flags — the money shot.** `select_branch`
   at `-O3 -march=native`: **gcc still LEAKS** (keeps a per-element data-dependent
   branch; taint fires) while **clang is oblivious** (lowers it to a vectorized blend).
   Two "reasonable" toolchains, same source, same flags → different security. Also `relu`
   at `-O2`: gcc leaks, clang clean.
3. **Constant-time source survives.** `select_ct` (branch-free `b ^ ((a^b) & -mask)`) is
   oblivious in *every* build — no compiler/flag reintroduced a branch here. The idiom
   held (the feared "compiler breaks constant-time" case did not fire in this set, but
   the harness is exactly what would catch it if it did).
4. **Irreducible source leak.** `memeq` (early-exit compare) leaks everywhere — dIr and
   dBc differ by class at every build; an algorithmic leak no optimizer removes.
5. **Bonus — optimization shrinks the CSI-NN memory-store attack surface (channel C2).**
   `matmul` is instruction-oblivious at every level, but its data-write count `Dw`
   collapses from **132,104 (`-O0`) to ~258 (`-O2+`)** — a ~500× reduction. At `-O0` the
   accumulator spills to memory every iteration (many storable `x·w` intermediates for a
   CPA/power probe); at `-O2+` it stays in a register. So higher optimization *reduces*
   the physical (CSI-NN mechanism-1) attack surface even where it changes no digital leak.
   Same pattern in relu/select (Dw ~8200 → ~514).

### Bottom line
Our Inductor result ("no compiler-introduced leak") is a statement about **one config
point**. Swept across toolchains, the same source flips between leaking and oblivious,
and **gcc and clang disagree at identical flags** (`select_branch -O3-native`). Leakage
is a property of `(source, compiler, flags, ISA, libs)`, not of "compilation." A verified
clean toolchain does not certify a different one.

## What has been measured: the timing (wall-clock) channel

From `leak_check/honest_timing.py` (full output in `leak_check/honest_timing.run.out`).
Single thread, 800 iterations, interleaved A/B, DIM=4096. Leakage strength =
`|attacker_AUC - 0.5| * 2` (0 = oblivious, 1 = perfect leak).

| Model | Build | zero median | random median | Attacker AUC | Leakage strength | Verdict |
|---|---|---|---|---|---|---|
| Branchless (`x@w`) | Eager | 5667 µs | 5621 µs | 0.473 | **0.054** | no leak |
| Branchless | Inductor | 5921 µs | 5967 µs | 0.546 | **0.093** | no leak |
| `torch.cond` skip | Eager | 93 814 µs | 109 948 µs | 0.901 | **0.802** | LEAKS |
| `torch.cond` skip | Inductor | 7268 µs | 13 071 µs | 0.999 | **0.998** | LEAKS |

### Findings
1. **The compiler did not manufacture a leak.** The branchless control stays
   effectively oblivious after Inductor (strength 0.093). Note its Mann-Whitney
   p = 1.3e-3 ("significant") while the effect is negligible — the large-N p-value trap;
   decide on AUC, not p.
2. **The `torch.cond` leak is authored, not compiler-introduced.** It leaks in *both*
   eager (0.802) and compiled (0.998). A signal present before compilation cannot have
   been created by compilation.
3. **Compilation denoised a pre-existing authored leak.** Inductor cut `torch.cond`
   framework overhead ~13× (94 ms → 7 ms on the zero path) and, by stripping that noise,
   made the authored channel *easier* to observe (AUC 0.901 → 0.999). "Delta got bigger
   after compiling" is a denoising effect, not a new channel.

## Relevant behavioral fact (not a timing measurement)

`torch.compile(..., fullgraph=True)` **rejects** a plain Python `if` on tensor values
(`torch._dynamo.exc.Unsupported: Data-dependent branching`). To get a compilable
data-dependent branch at all you must use `torch.cond`. So the original demo's premise —
that Inductor silently lowers `if torch.all(weight==0)` into a value-dependent kernel —
does not even reach the compiler; Dynamo refuses it up front.

## The deterministic channels: MEASURED

Valgrind 3.22.0. Full sweep via `python leak_check/run_all.py`
(raw output in `leak_check/run_all.out`). Instruments: callgrind (`Ir` instructions,
`Bc` conditional branches, gated to the forward pass) + memcheck taint. `dIr`/`dBc` =
count(random) − count(zero); exactly 0 ⇒ provably oblivious at that granularity.

| Model | Build | Ir(zero) | Ir(random) | dIr | dBc | taint | distinguishable? |
|---|---|---|---|---|---|---|---|
| branchless | eager | 148,524 | 148,524 | **+0** | +0 | clean | no |
| branchless | compiled | 314,332 | 314,332 | **+0** | +0 | clean | no |
| cond_skip | eager | 6,632,618 | 6,446,487 | −186,131 | +2,799 | **TAINT** | yes |
| cond_skip | compiled | 485,366 | 564,624 | +79,258 | +4,144 | **TAINT** | yes |
| where_select | eager | 299,075 | 299,339 | +264 | +0 | **TAINT** | yes |
| where_select | compiled | 455,881 | 455,881 | **+0** | +0 | clean | no |

### Verdicts
- **branchless → OBLIVIOUS** (matches prediction). Bit-exact identical instruction and
  branch counts across secrets in *both* eager and compiled, taint clean. This is the
  decisive cell: **Inductor did not manufacture a leak**, proven deterministically
  (not merely "undetected" as timing would give). Note compiled runs *more*
  instructions than eager (314k vs 148k) at this tiny size — Inductor wrapper overhead
  — but that's secret-independent, so `dIr` is still exactly 0.
- **cond_skip → AUTHORED** (matches prediction). Distinguishable in both builds:
  taint fires on the `all(weight==0)` predicate and `dIr`/`dBc` are large. A leak
  present in eager cannot have been created by the compiler. (Eager `dIr` is *negative*
  — the zero/skip path actually executes *more* instructions than the matmul path,
  because eager `torch.cond` bookkeeping dwarfs this small BLAS call; the sign is
  irrelevant, only `≠ 0` matters.)
- **where_select → COMPILER-REMOVED** (prediction was "oblivious" — the run overturned
  it, see below).

### The surprise: Inductor *removed* a side channel (where_select)
Prediction was oblivious; reality is more interesting. In **eager**, `torch.where` over
the secret-derived mask leaks: taint fires (a per-element conditional select on the
mask) and there's even a tiny `dIr = +264`. In **compiled**, `dIr = 0`, `dBc = 0`,
taint clean — Inductor lowered the `where` into a **branchless vectorized blend**,
eliminating the control-flow dependence on the secret.

This is the exact opposite of the original demo's thesis: not only does the compiler
not introduce a leak, here it **erased** one that existed in eager. The `[DIFF]` flag
in the summary is not a harness failure — it's the harness catching a real effect the
prediction missed (the channel-disagreement case flagged pre-run: count says one thing,
taint another, and compilation resolves it).

The exact eager memcheck report (from `leak_check/where_select_taint.txt`):

    Conditional jump or move depends on uninitialised value(s)

i.e. the elementwise `where` kernel executes a per-element conditional branch on the
secret-derived mask; Inductor lowers the same `where` to a branchless vector blend.

### Cross-check against the timing tier
The deterministic channels agree with (and sharpen) the earlier wall-clock results:
branchless oblivious, cond_skip an authored leak in both builds. The deterministic
tiers additionally resolve what timing could not: exact obliviousness for branchless,
and the eager-only `where_select` control-flow dependence that timing's noise floor
would never have surfaced.

### Headline
Across this corpus, **`torch.compile`/Inductor introduced zero leaks** and in one case
removed one. Every "leak" observed is authored data-dependent control flow in the
source, present in eager. The apparatus is what would catch a genuine
compiler-introduced leak (eager-clean, compiled-distinguishable) — none occurred here.

## Adversarial probe + a blind spot in the deterministic channels (denormals)

The calibration corpus (branchless / cond_skip / where_select) validates the *detector*
but never populates the one quadrant that would vindicate the original thesis —
**eager-clean → compiled-distinguishable**. `leak_check/denormal_probe.py` is the first
real candidate aimed at it: the SAME branchless `x@w`, with two secret classes that
differ only in float *representation* — tiny-normal (~1e-20) vs subnormal (~1e-40).

Measured (DIM=4096, interleaved, 400 iters):

| Build | normal median | denormal median | slowdown | AUC | timing verdict |
|---|---|---|---|---|---|
| eager | 7,125 µs | 178,939 µs | **25.1×** | 1.000 | LEAKS |
| compiled | 7,416 µs | 180,065 µs | **24.3×** | 1.000 | LEAKS |

Two findings:

1. **Still not compiler-introduced.** The 25× subnormal slowdown is present *identically*
   in both builds → a **microarchitectural hardware effect** (the FPU handles subnormals
   ~25× slower), not something Inductor added. Neither build sets flush-to-zero (FTZ/DAZ),
   so both leak equally. (A compiler *could* move this by setting FTZ via `-ffast-math`;
   this one doesn't, in either direction.)

2. **The deterministic channels are BLIND to it — a genuine limitation.** The instruction
   stream is byte-identical for normal vs subnormal inputs, so `dIr = 0`, `dBc = 0`, and
   taint is clean in every build — yet the timing channel shows a *perfect* leak
   (AUC = 1.000; one sample distinguishes the secret). The earlier claim "deterministic
   channels dominate timing" holds only for **control-flow / instruction-count / addressing**
   leaks. It is FALSE for **data-dependent cycle-count** leaks (subnormals, cache/port
   contention, variable-latency instructions like division). For those, timing (Tier 3)
   is the *only* instrument that sees the leak. Use all tiers; none subsumes the others.

## Two more adversarial probes aimed straight at the empty quadrant

### Probe 1 — `max-autotune` compile-time kernel selection (`probe_autotune.py`)
Hypothesis: `mode="max-autotune"` benchmarks candidate GEMM kernels at compile time
and could pick a different kernel depending on the secret weights present then —
baking the secret into the generated code. Test: force a fresh autotune per secret
(`force_disable_caches`), dump the generated C++, and diff it; corroborate by timing
each compiled artifact on an identical neutral `w0`.

Result: **NO compile-time leak.** Both secrets produce byte-identical generated code —
both select `extern_kernels.mm` (the ATEN library kernel) for this `1×512 @ 512×512`
shape. Kernel choice is value-independent (Inductor autotunes with synthetic sample
inputs and caches by shape, not by parameter values).

> **Near-miss / methodology lesson.** The *first* run of this probe reported
> "POSSIBLE compile-time leak: generated code differs" with a timing AUC of 0.891 — a
> false positive. Cause: the code-diff normalizer failed to strip the `TORCH_LOGS` line
> prefix (`V0703 HH:MM:SS.mmm PID …`), so every line differed only by *timestamp/PID*,
> not content. Inspecting the actual diff showed identical `extern_kernels.mm` in both.
> After fixing the normalizer the probe self-reports "identical / NO leak." Two takeaways:
> (1) the **timing-on-w0 AUC is an unreliable channel** — it read 0.891 then 0.684 on
> re-run, pure cross-process drift, while the code-diff is decisive; (2) I nearly
> reported a compiler-introduced leak that was my own measurement bug — the value of
> looking at the raw artifact before believing a summary statistic.

### Probe 2 — `-ffast-math` / FTZ on the denormal leak (`probe_fastmath.py`)
Hypothesis: fast-math codegen sets flush-to-zero (FTZ/DAZ), which would erase the 25×
denormal leak in the compiled build (compiler-*removed*). Test: normal-vs-denormal
timing under eager, compiled (default), and compiled + `cpp.enable_unsafe_math_opt_flag`.

| build | slowdown | leak? |
|---|---|---|
| eager | 26.9× | LEAKS |
| compiled (default) | 24.6× | LEAKS |
| compiled + fast-math | 25.5× | LEAKS |

Result: **leak in all three** → the compiler does not move it in either direction here.
Default Inductor does not set FTZ, and the fast-math flag did not flush denormals in
this build either. (Caveat: worth confirming the flag actually reached `g++`; but the
empirical outcome — symmetric 25× across builds — makes the leak hardware, not the
compiler's doing, regardless.)

### Status of the "compiler-introduced" quadrant
Across **5 kernels/configs** (branchless, cond_skip, where_select, denormal,
max-autotune) plus the fast-math variant, the eager-clean → compiled-distinguishable
quadrant has **never fired**. Evidence, not proof — but a consistent picture with a
structural explanation: optimizing compilers *remove* value-dependent branches
(where_select's vectorized blend) rather than add them, and their value-sensitive
decision points (autotune) are deliberately fed synthetic inputs. The one real,
dramatic leak found (denormals, 25×) is hardware and symmetric across builds — and is
invisible to the deterministic channels. Net: this compiler introduced no leak in any
probe tried; the risks that *are* real are authored control flow and microarchitecture,
neither of which is "the compiler introducing a leak."
