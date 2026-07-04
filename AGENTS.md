# AGENTS.md — working constitution for this repository

Rules for any agent writing reports (`notes/`) or code here. This is a
security-measurement project; its credibility rests on claims being no stronger
than the evidence. Keep to these.

## 1. Evidence discipline

- Every quantitative claim names the script and the exact command that produced
  it, so a reader can re-run it. No number without a source.
- Label each claim as one of **measured**, **hypothesized**, or **unverified
  lead**. Do not let a hypothesis graduate to a finding without a run behind it.
- "Not detected" is not "proven absent." A null timing result means the harness
  did not see a leak with this setup — never that the code is constant-time. Say
  the weaker thing.
- Before trusting a summary statistic, look at the raw artifact (generated code,
  the callgrind diff, the memcheck log). Report near-misses and your own
  measurement bugs; they are part of the record.

## 2. Scope every verdict to a config point

A result holds for one point in `(source, compiler, backend, version, flags,
target ISA, libraries)` — and, for attacks, one `(model, oracle, machine)`. It
does not generalize to other toolchains or machines. State the config point; do
not imply universality.

## 3. Tone

- Sober and technical. State what was measured and what it means. Let the
  evidence carry the weight.
- No hype or self-narration: avoid "money shot", "the headline", "the surprise",
  "CRITICAL SECURITY ALERT", "the richest confirmation yet", and similar. No
  unearned superlatives. Use **bold** sparingly, for genuine key terms only.
- Prefer plain declaratives over exclamations. A reader should not be able to
  tell whether the result pleased you.

## 4. Reproducibility

- Include the runnable command in the write-up.
- Pin determinism: fixed seed, single-thread BLAS/OMP, warm caches outside the
  measured region. Note the machine/versions in `notes/reference/`.

## 5. Code

- SOLID and additive. Do not rewrite a colleague's working module to add a feature; extend alongside it.
- No module-level mutable state standing in for objects. Encapsulate a target, its measurement, and the algorithm as separate units.
- Depend on abstractions. New attacks/targets should slot into existing
  interfaces (see `leak_check/attacks/base.py`) without editing unrelated code.
- Calibrate a tool before you trust its verdict: a positive control must fire and
  a negative control must stay silent (cf. `leak_check/corpus.py`).
