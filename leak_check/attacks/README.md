# attacks/ — offensive side-channel attacks

The detector harness in the parent directory decides *whether* a secret-dependent
path exists and *who* caused it. This package is the other half: given a leak,
reconstruct the secret through it, and show the impact.

## `early_exit_gpt` — steal a secret early-exit gate

A GPT runs `exit_after_layer` blocks, then a **secret linear gate** decides whether
to skip the rest. Skipped blocks change latency, so the gate's decision leaks — an
*authored* branch (detector class `cond_skip`/`memeq`), leaking in every build. The
attacker recovers the gate's decision boundary from queries alone; with it, the
gate's decisions can be replayed, evaded, or manipulated.

### Threat models

| | Channel (`Oracle`) | Per query | Attack | Surrogate |
|---|---|---|---|---|
| **A** | `TimingOracle` (latency), via `ThresholdLabeler` | 1 bit (exit?) | `SurrogateExtractionAttack` | logistic |
| **B** | `LogitOracle` (gate confidence, arXiv:1907.00713) | continuous logit | `RegressionExtractionAttack` / `MlpLogitExtractionAttack` | linear / torch-MLP |

All share one boundary-focused active loop (`sampling.py`): probe randomly, fit a
surrogate, then concentrate queries on its current boundary. B additionally offers
the paper's stronger method — an MLP surrogate whose exact input gradient (autograd)
drives Jacobian query augmentation. `LayerCountOracle` is a noise-free label control
(blocks executed, not timed) to separate algorithm error from timing noise.

### Results (measured, full 124M model, iid queries, held-out R² of the logit)

| queries | linear R² | linear acc | MLP R² | MLP acc |
|---|---|---|---|---|
| 400  | 0.635 | 81.0% | 0.515 | 80.5% |
| 800  | 0.996 | 98.5% | 0.796 | 85.5% |
| 1600 | 1.000 | 100 % | 0.973 | 96.5% |

- **Confidence beats one bit:** the logit attacks hit ≈100% accuracy vs the label
  attack's ≈97% at equal budget (`compare`).
- **A linear surrogate suffices; the MLP does not help here.** The early-exit logit
  is ~linear in the query (the residual stream carries `x` through), so a linear fit
  reaches R² ≈ 1 once queries exceed the query dimension. The MLP is correct and is
  the right tool for a *nonlinear* confidence surface, but adds only variance here.

### Run (from `leak_check/`)

    python -m attacks.early_exit_gpt.run --mode small                       # A: timing (amplified tiny model)
    python -m attacks.early_exit_gpt.run --mode small --oracle layercount   # A: noise-free control
    python -m attacks.early_exit_gpt.run --mode small --attack logit|mlp    # B: linear | MLP surrogate
    python -m attacks.early_exit_gpt.run --mode full  --attack logit        # 124M GPT-2 (slow)
    python -m attacks.early_exit_gpt.compare                                # label vs linear vs MLP
    python -m attacks.early_exit_gpt.check                                  # smoke self-check

Dependencies: `pip install -r attacks/requirements.txt`.

## Architecture

An attack learns from `(x, observe(x))` pairs, where `observe: x → float` is any
scalar query (`Labeler.label` or `Oracle.query`). It depends only on the `base.py`
interfaces, never on the GPT — so a new algorithm subclasses `ExtractionAttack`, a
new target implements `Oracle`, and neither touches the other.

| Concern | Files |
|---|---|
| target / secret holder | `model.py`, `enclave.py` (`EarlyExitEnclave`) |
| channels | `oracle.py` (`TimingOracle`, `LayerCountOracle`, `LogitOracle`, `ThresholdLabeler`) |
| attacks | `extraction.py`, `logit_extraction.py`, `mlp_extraction.py` |
| surrogate / sampling | `mlp_surrogate.py`, `sampling.py` |
| scoring / config / drivers | `evaluate.py`, `config.py`, `run.py`, `compare.py`, `check.py` |
