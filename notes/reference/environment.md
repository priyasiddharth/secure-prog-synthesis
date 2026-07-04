# Reference: environment & tooling

Pointers and versions. Changes when the machine or toolchain changes.

## Platform
- Host working dir: `/home/siddharth/sps`
- Python venv: `/home/siddharth/compiler_env` (Python 3.12)
- CPU: 8 cores (`nproc` = 8); harness pins to 1 thread for determinism.

## Key library versions
- `torch` 2.12.1+cu130 (CPU path used; `device = cpu`)
- `numpy` 2.5.0
- `scipy` (installed by user for the Mann-Whitney AUC tier)
- AOTInductor: `torch._inductor.aoti_compile_and_package` present.
- Inductor code dump: `TORCH_LOGS=output_code` (env, always available).

## Instrument availability (the constraint that shaped the design)
| Tool | Status | Role |
|------|--------|------|
| `g++` | present (`/usr/bin/g++`) | build `vgshim.so`; Inductor CPU codegen |
| `objdump` | present | inspect generated `.so` |
| `valgrind` (memcheck/cachegrind/callgrind) | **INSTALLED — 3.22.0** | deterministic count + taint channels |
| `perf` | MISSING | optional hardware-counter cross-check only |

### Valgrind (installed)
`valgrind-3.22.0` installed by the user; headers present at
`/usr/include/valgrind/{valgrind,memcheck,callgrind}.h`. Build + run:

    g++ -shared -fPIC -O0 leak_check/vgshim.c -o leak_check/vgshim.so
    python leak_check/run_all.py   # run_all rebuilds the shim automatically

`perf` is intentionally not required — callgrind gives the same signal
deterministically in userspace without elevated privileges.

### Gotchas discovered while wiring it up
- `vgshim.c` must wrap its API in `extern "C"` — `run_all.py` builds it with
  `g++`, which otherwise mangles the symbols and `ctypes` can't find them.
- callgrind is gated with `--instr-atstart=no` + `CALLGRIND_START/STOP_INSTRUMENTATION`,
  so the counted region is the **`totals:`** line of the callgrind output, NOT
  `summary:` (which tracks *collection*, left off, and reads 0).
- memcheck cost is dominated by importing torch under instrumentation:
  **~8 min per run** at DIM=512. callgrind legs are ~70 s each. A full corpus
  sweep (3 models × {eager,compiled} × {2 callgrind + 1 memcheck}) is ~50-70 min.

## File map
- `leak_check/timing_leak.py` — the original (flawed) demo, kept for continuity.
- `leak_check/honest_timing.py` — the corrected timing-channel harness.
- `leak_check/honest_timing.run.out` — its recorded output (source of `empirical/results.md`).
- `leak_check/vgshim.c` — Valgrind client-request bridge (taint + callgrind gating).
- `leak_check/corpus.py` — models + secret classes + expected verdicts.
- `leak_check/measured_run.py` — one fixed-code-path forward pass under an instrument.
- `leak_check/instruments.py` — callgrind counter + memcheck taint runner/parsers.
- `leak_check/noninterference.py` — the differential criterion + verdict.
- `leak_check/run_all.py` — driver; builds shim, generates secrets, prints table.
- `leak_check/secrets/{zero,random}.npy` — the two secret-class buffers.
- `leak_check/run_all.out` / `.err` — latest deterministic-channel run output.
- `leak_check/denormal_probe.py` — timing probe: subnormal-float leak (blind spot).
- `leak_check/probe_fastmath.py` + `_denorm_worker.py` — Probe 2 (FTZ / fast-math).
- `leak_check/probe_autotune.py` + `_autotune_worker.py` — Probe 1 (max-autotune codegen).
- `leak_check/probe_{fastmath,autotune}.out` — their outputs.
- `leak_check/ctbench/{kernels,harness}.c` — toolchain C testbed (gcc/clang flag sweep).
- `leak_check/ctbench/run_ct.py` — sweep driver + leak matrix; output `run_ct.out`.
- `leak_check/instruments.py` `*_cmd` runners — instrument an arbitrary binary (adds Dw).
