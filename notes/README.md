# Notes: detecting compiler-introduced information leaks

Documentation for the `leak_check/` harness — a general, compiler-agnostic method
(demonstrated on `torch.compile`/Inductor) for deciding whether a *compiler*, rather
than the source program, introduces a secret-dependent execution path.

Notes are split by how stable the content is:

| Dir | Holds | Changes when |
|-----|-------|--------------|
| [`durable/`](durable/) | The method: principles, criteria, instrument tiers, pitfalls | The methodology itself is revised |
| [`empirical/`](empirical/) | Concrete measurements we obtained | Re-run on different code / machine / library version |
| [`reference/`](reference/) | Environment, tool availability, versions | Machine or toolchain changes |

Start with [`durable/methodology.md`](durable/methodology.md).

## Index
- `durable/methodology.md` — the full method: differential non-interference, the
  three observation channels, the two-secret-class design, isolate-the-kernel,
  the risky-optimization catalog, and pitfalls.
- `empirical/results.md` — what has actually been measured so far (timing tier),
  plus the deterministic tiers (Valgrind installed; sweep running via `run_all.py`).
- `reference/environment.md` — machine, versions, installed/missing instruments.
- `reference/csi-nn-paper.md` — CSI NN weight-stealing paper (arXiv:1810.09076): how
  it maps onto this harness (proves the deterministic-channel blind spot; motivates the
  activation-function corpus and the tainted-store CPA-surface proxy).
