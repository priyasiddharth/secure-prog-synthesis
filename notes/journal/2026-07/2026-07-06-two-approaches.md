# Two approaches to the compiler-leak question

[OBS 2026-07-06] The project currently holds two distinct approaches to
deciding whether a *compiler* (rather than the source program) is responsible
for a secret-dependent execution path. They answer different questions and
should not be conflated. Recording the split so a future session can decide
which one the work is chartered against.

## Approach A — small composable primitives (differential non-interference)

Take two secret weight classes, `w1` and `w2`, that a constant-time program
must treat identically (same shape/dtype, differing only in the protected
bytes). Push each through a *small* computation that is data-oblivious under
the reference build. If the two are indistinguishable before compilation but
distinguishable after, attribute the difference to the compiler — the "empty
quadrant" (eager-clean → compiled-distinguishable) of the 2×2 differential
test in [`../../durable/methodology.md`](../../durable/methodology.md).

The primitives are the elementwise building blocks that compose into networks
(relu, exp, gelu, softmax) plus matmul, and a set of hand-written C kernels
run across a gcc/clang flag sweep. This is what `leak_check/` implements today.

**State:** the empty quadrant has never fired across the corpus at this config
point — the compiler *removed* a value-dependent branch (where_select →
vectorized blend) rather than adding one; the real leaks found are authored
control flow and microarchitectural (denormals, ~25×, invisible to the
deterministic channels). See [`../../empirical/results.md`](../../empirical/results.md).
Reading: this approach is the rigorous audit of the original thesis, and it
returns a well-supported **negative** on the toolchain measured.

## Approach B — complex network directly (end-to-end threat model)

Build an executable threat model on a full network and mount a timing/semantic
attack end-to-end: an attacker observes only response latency and recovers a
secret (e.g. an early-exit gate) via query-driven extraction. This is seeded by
`notebooks/Compilers.ipynb` (a timing attack on nano-GPT).

**State:** the current instance runs on a randomly-initialized toy network with
an *authored* early-exit gate, so it demonstrates attack mechanics but not a
compiler-introduced effect on a realistic model. To be a legitimate deliverable
for this line it needs a real model and a real extraction pipeline.

## Why the distinction matters

The two approaches answer different questions:

1. *Does the compiler introduce a leak?* — Approach A answers it (no, at the
   measured config point). The residual signal that is real lives in config
   points (gcc/clang divergence at identical flags), microarchitecture
   (denormals), and authored control flow — none of which is "compilation
   manufacturing a channel."
2. *Can a semantic/timing attack recover a secret from a real model?* —
   Approach B targets this. Approach A's audit harness does not directly serve
   it, because it is a defensive differential audit, not an offensive attack
   pipeline.

[OPEN] Which question is the project's deliverable is undecided. If (1),
Approach A is on-target and largely complete; further hardening will keep
returning negatives on cooperative toolchains. If (2), a real model is required
and the audit harness is a supporting instrument rather than the product.
