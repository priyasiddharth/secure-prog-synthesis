"""
One measured forward pass with a *fixed, secret-independent code path*.

The whole point: everything except the contents of model.weight is identical
across invocations, so any difference an instrument sees is attributable to the
secret bytes. All compilation, warmup, and allocation happen OUTSIDE the
instrumented region (between cg_start/cg_stop, or before the taint mark).

Usage:
  python measured_run.py <model> <secret.npy> [--compile] [--taint]

Instruments wrap this script:
  * callgrind counts instructions between cg_start()/cg_stop()
  * memcheck reports any control-flow/address dependence on the tainted weights
"""

import os
import sys
import ctypes
import argparse

# Determinism knobs BEFORE importing torch.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.manual_seed(0)
torch.set_num_threads(1)

import corpus
import corpus_activations


def _resolve(name):
    """Dispatch a model name to whichever corpus module defines it."""
    return corpus_activations if name in corpus_activations.names() else corpus


def load_shim(path):
    shim = ctypes.CDLL(path)
    shim.vg_make_undefined.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    shim.vg_make_defined.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    shim.vg_marker.argtypes = [ctypes.c_char_p]
    shim.vg_running.restype = ctypes.c_int
    return shim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=corpus.names() + corpus_activations.names())
    ap.add_argument("secret", help="path to float32 .npy of shape [DIM, DIM]")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--taint", action="store_true")
    ap.add_argument("--shim", default=os.path.join(os.path.dirname(__file__), "vgshim.so"))
    args = ap.parse_args()

    shim = load_shim(args.shim)

    src = _resolve(args.model)
    w = np.load(args.secret).astype(np.float32)
    model = src.build(args.model)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(w))
    x = src.example_input(args.model)

    fn = model
    if args.compile:
        fn = torch.compile(model, fullgraph=True)

    # --- Warm EVERYTHING outside the measured region: compile, autotune,
    #     allocator caches, dispatch caches. Two passes for good measure. ---
    with torch.no_grad():
        for _ in range(2):
            _ = fn(x)

    nbytes = model.weight.numel() * model.weight.element_size()
    wptr = model.weight.data_ptr()

    if args.taint:
        # Mark the secret bytes UNDEFINED. Their real values stay in memory; only
        # memcheck's definedness shadow flips, so any branch/address computed from
        # them is reported at the exact leaking instruction.
        shim.vg_marker(b"### LEAKCHECK taint region begin ###")
        shim.vg_make_undefined(ctypes.c_void_p(wptr), ctypes.c_size_t(nbytes))
        with torch.no_grad():
            out = fn(x)
        # Re-define the derived output so our own teardown (dealloc, exit) can't
        # raise spurious reports.
        shim.vg_make_defined(ctypes.c_void_p(out.data_ptr()),
                             ctypes.c_size_t(out.numel() * out.element_size()))
        shim.vg_make_defined(ctypes.c_void_p(wptr), ctypes.c_size_t(nbytes))
        shim.vg_marker(b"### LEAKCHECK taint region end ###")
    else:
        # Counting mode: gate callgrind to just the forward pass.
        shim.cg_start()
        with torch.no_grad():
            out = fn(x)
            sink = float(out.reshape(-1)[0])  # force materialization
        shim.cg_stop()
        # keep `sink` observable so nothing is optimized away
        if sink == 123456789.0:
            print("unreachable", sink)


if __name__ == "__main__":
    main()
