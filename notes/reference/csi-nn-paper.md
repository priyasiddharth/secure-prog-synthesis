# Reference: CSI NN (weight-stealing via side channels)

**Batina, Bhasin, Jap, Picek — "CSI Neural Network: Using Side-channels to Recover
Your Artificial Neural Network Information"** — arXiv:1810.09076 (2018); later
USENIX Security 2019. https://arxiv.org/abs/1810.09076

## What it does
Reverse-engineers a neural network — architecture AND actual weight values — from
**passive power / electromagnetic** side channels on an **ARM Cortex-M3**
microcontroller (also uses reaction-time/timing).

## Three mechanisms
1. **Weight recovery — Correlation Power Analysis (CPA).** Target `m = x·w`; vary
   known input `x`; correlate measured **power** against the **Hamming weight of `m`
   as it is stored to memory**, over weight hypotheses (Pearson ρ). Recovers IEEE-754
   weights byte-by-byte (sign→exponent→mantissa), ~0.01 precision. *Analog channel.*
2. **Activation-function identification — timing.** relu ≈ 6 µs, sigmoid/tanh ≈
   50–220 µs, softmax ≈ 700–880 µs; exp/div introduce **input-dependent conditional
   branching** → distinct timing signatures. *Software-observable channel.*
3. **Layer/neuron count — SPA.** Multiply vs activation have different trace
   signatures; count them. Layer boundaries found by which CPA hypothesis (input vs
   intermediate) correlates.

## Why it matters for `leak_check/` (the connection)
- **It is published proof of our blind spot.** Mechanism 1 steals weights from a
  **data-oblivious** multiply — the exact code our deterministic channels
  (`dIr=0`, taint-clean) certify as "oblivious." The leak is in the Hamming-weight
  **power** of an instruction our instruments see as constant. So a clean Tier-1/2
  verdict means "no *digital* leak," NOT "no leak"; CSI NN operates below our floor.
  Stronger than the denormal case — needs no special values at all. See
  `../durable/methodology.md` (channel-coverage caveat) and `../empirical/results.md`.
- **Mechanism 2 is directly in our reach** — input-value-dependent activation control
  flow is exactly what Ir/Bc/taint/timing detect. Motivates an **activation-function
  corpus** (relu/sigmoid/tanh/exp/softmax/gelu) run eager-vs-Inductor.
- **Mechanism 1's leakage point maps onto our taint instrument.** CSI NN leaks at the
  **store of `m=x·w`**; memcheck taint localizes precisely secret-derived memory ops.
  → a digital **CPA-attack-surface proxy**: count secret-tainted memory stores, and
  test whether Inductor **fusion** (keeping intermediates in registers vs spilling to
  DRAM) **reduces** that surface.

## Proposed follow-ups (see results.md when run)
① activation-function differential corpus; ② tainted-store counting as CPA-surface
proxy; ③ (heavier) Hamming-weight CPA leakage simulation à la ELMO.

## Caveats
Power/EM on a bare-metal MCU with a probe is a richer, more invasive channel than our
software-only timing/instruction-count on a Linux desktop; we cannot reproduce it.
A null in our harness does not imply CSI NN would fail — it sees what ours cannot.
