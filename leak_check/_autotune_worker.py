"""
Worker for Probe 1. Compile a branchless x@w under max-autotune with a specific
SECRET present at compile time, forcing a fresh (uncached) autotune so the secret
could in principle influence kernel selection. Then neutralize the data to a fixed
w0 and time the chosen kernel, and let TORCH_LOGS=output_code dump the generated
C++ to stderr (captured by the driver).

    TORCH_LOGS=output_code python _autotune_worker.py {zero|random} w0.npy out.npy

If Inductor autotunes with synthetic sample inputs and caches by shape (not by
value), the generated code and the w0-timing will be identical across secrets.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

torch.manual_seed(0)
torch.set_num_threads(1)

DIM = 512
ITERS = 200
WARMUP = 20


class Branchless(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))

    def forward(self, x):
        return torch.matmul(x, self.weight)


def main():
    secret, w0_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    import torch._inductor.config as ic
    ic.force_disable_caches = True          # force a real, fresh autotune
    ic.max_autotune = True

    m = Branchless()
    with torch.no_grad():
        if secret == "zero":
            m.weight.zero_()
        else:
            m.weight.copy_(torch.from_numpy(
                np.random.default_rng(0).standard_normal((DIM, DIM), dtype=np.float32)))

    x = torch.randn(1, DIM)
    fn = torch.compile(m, mode="max-autotune", fullgraph=True)
    # Compile + autotune happen here, with `secret` sitting in m.weight.
    for _ in range(3):
        fn(x)

    # Neutralize the data: identical w0 for both secrets, so any timing/codegen
    # difference is attributable to the COMPILE-TIME secret, not the runtime data.
    w0 = np.load(w0_path)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w0))
    for _ in range(WARMUP):
        fn(x)

    t = np.empty(ITERS)
    for i in range(ITERS):
        t0 = time.perf_counter(); fn(x); t[i] = time.perf_counter() - t0
    np.save(out_path, t)
    print(f"RESULT secret={secret} median_us={np.median(t)*1e6:.2f}")


if __name__ == "__main__":
    main()
