# attacks/ — offensive side-channel attacks

The detector harness in the parent directory decides *whether* a secret-dependent
execution path exists and *who* introduced it (source vs. compiler). This package
is the complementary offensive half: given a leak that does exist, reconstruct the
secret through it.

## Where this fits the detector taxonomy

The detectors classify a leak as **authored** (the source contains secret-dependent
control flow) or **compiler-introduced**. An *early exit* — skip the rest of the
network when a gate says the input is "easy" — is an authored, data-dependent
branch, the same class as `cond_skip` / `memeq` in the detector corpus, and it
leaks in every build. This package demonstrates the consequence of such a leak:
an attacker turns it into full recovery of the secret gate.

## `early_exit_gpt` — stealing a secret early-exit gate

A GPT (nanoGPT / GPT-2 architecture) runs the first `exit_after_layer` blocks,
then a **secret linear gate** reads the hidden state and decides whether to skip
the remaining blocks. Skipped blocks are work not done, so the forward-pass
latency depends on the gate's decision — that is the leak. An attacker who can
only time the forward pass recovers the gate's decision boundary.

### Attack shape

1. **Oracle.** The attacker sees one scalar per query: the forward-pass latency
   (`TimingOracle`). A `ThresholdLabeler` calibrates a fast/slow midpoint from two
   extreme probes and turns latency into a binary "did it exit?" label.
2. **Active-learning extraction.** `SurrogateExtractionAttack` fits a logistic
   regression to the labels, then spends each round's queries near the surrogate's
   current boundary (the most informative region). A linear gate is recovered in a
   small multiple of its parameter count.
3. **Scoring.** `oracle_alignment` (labels vs. the true gate) upper-bounds what
   any attack can learn through the channel; `surrogate_alignment` (stolen model
   vs. true gate) is the attack's success rate.

### Running

From the `leak_check/` directory:

    python -m attacks.early_exit_gpt.run --mode small                 # fast (amplified tiny model)
    python -m attacks.early_exit_gpt.run --mode small --oracle layercount   # noise-free control
    python -m attacks.early_exit_gpt.run --mode full                  # 124M GPT-2, real latency (slow)
    python -m attacks.early_exit_gpt.check                            # smoke self-check

`--oracle layercount` reads the executed block count directly instead of timing
it. A real attacker cannot do this; it is a control that removes measurement noise
so a result can be attributed to the algorithm rather than the machine — the
offensive analogue of the detector side's branchless control.

Dependencies: `pip install -r attacks/requirements.txt`.

## Architecture

The attack is written against the interfaces in [`base.py`](base.py), not against
the GPT, so a new target or algorithm drops in without editing the other side:

| Concern            | Interface (`base.py`)     | `early_exit_gpt` implementation        |
|--------------------|---------------------------|----------------------------------------|
| the target model   | —                         | `model.py` (nanoGPT)                   |
| what leaks         | `Oracle.query`            | `oracle.py` (`TimingOracle`, `LayerCountOracle`) |
| observable → label | `Labeler.label`           | `oracle.py` (`ThresholdLabeler`)       |
| the attack         | `ExtractionAttack.run`    | `extraction.py`                        |
| the secret holder  | —                         | `enclave.py` (`EarlyExitEnclave`)      |
| scoring            | —                         | `evaluate.py`                          |
| knobs / presets    | —                         | `config.py`                            |
| wiring + report    | —                         | `run.py`                               |

To add a new extraction attack, subclass `ExtractionAttack`. To attack a different
target, expose it through an `Oracle`. Neither change touches the other modules.
