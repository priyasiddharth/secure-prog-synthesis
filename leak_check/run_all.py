"""
Top-level driver: generate the two secret buffers, run the differential
non-interference analysis over the whole corpus, print a table, and check each
verdict against the method's own prediction (corpus.EXPECTED).

    python run_all.py                # whole corpus
    python run_all.py cond_skip ...  # subset
"""

import os
import sys
import ctypes
import subprocess

import numpy as np

import corpus
import noninterference as NI

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET_DIR = os.path.join(HERE, "secrets")


def ensure_shim():
    so = os.path.join(HERE, "vgshim.so")
    src = os.path.join(HERE, "vgshim.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(["g++", "-shared", "-fPIC", "-O0", src, "-o", so],
                       check=True)
    shim = ctypes.CDLL(so)
    shim.vg_running.restype = ctypes.c_int
    if shim.vg_running() != 0:
        raise RuntimeError("vgshim reports it is under Valgrind at import time?!")
    return so


def ensure_secrets():
    os.makedirs(SECRET_DIR, exist_ok=True)
    zero = os.path.join(SECRET_DIR, "zero.npy")
    rand = os.path.join(SECRET_DIR, "random.npy")
    d = corpus.DIM
    if not os.path.exists(zero):
        np.save(zero, np.zeros((d, d), dtype=np.float32))
    if not os.path.exists(rand):
        rng = np.random.default_rng(0)
        np.save(rand, rng.standard_normal((d, d), dtype=np.float32))
    return zero, rand


def fmt_build(b):
    t = "TAINT" if b["taint"]["leak"] else "clean"
    return (f"Ir_zero={b['cg_zero']['Ir']:>12,}  Ir_rand={b['cg_rand']['Ir']:>12,}  "
            f"dIr={b['ir_diff']:>+12,}  dBc={b['bc_diff']:>+10,}  taint={t}  "
            f"-> {'DISTINGUISHABLE' if b['distinguishable'] else 'oblivious'}")


def main():
    ensure_shim()
    zero, rand = ensure_secrets()
    models = sys.argv[1:] or corpus.names()

    print(f"corpus DIM={corpus.DIM} | secrets: zero vs random | "
          f"instruments: callgrind(Ir,Bc) + memcheck(taint)\n")

    rows = []
    for m in models:
        print("=" * 78)
        print(f"MODEL: {m}   (predicted: {corpus.EXPECTED.get(m, '?')})")
        print("=" * 78)
        res = NI.analyze(m, zero, rand)
        print(f"  eager    : {fmt_build(res['eager'])}")
        print(f"  compiled : {fmt_build(res['compiled'])}")
        exp = corpus.EXPECTED.get(m)
        ok = "  [MATCHES PREDICTION]" if exp == res["verdict"] else \
             f"  [!! predicted {exp} !!]" if exp else ""
        print(f"  VERDICT  : {res['verdict'].upper()}{ok}\n")
        rows.append((m, res["verdict"], exp))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for m, v, exp in rows:
        flag = "ok " if v == exp else "DIFF"
        print(f"  [{flag}] {m:<14} verdict={v:<22} predicted={exp}")


if __name__ == "__main__":
    main()
