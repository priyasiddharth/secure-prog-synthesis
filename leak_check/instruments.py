"""
Deterministic instruments that wrap measured_run.py.

  callgrind_count(model, secret, compile)  -> {"Ir":int, "Bc":int, "Bi":int}
      Retired-instruction and conditional-branch counts for the forward pass
      only (instrumentation is gated by cg_start/cg_stop in measured_run.py).
      Deterministic: identical code+data give identical counts, so a difference
      across two secrets is a real secret-dependent execution path -- no
      statistics required.

  memcheck_taint(model, secret, compile)   -> {"leak":bool, "reports":[str]}
      Marks the weight bytes UNDEFINED and reports any branch or memory address
      that depends on them (the definitional side-channel check).
"""

import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 900  # memcheck is slow; be generous


def _env():
    e = dict(os.environ)
    e.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             PYTHONUNBUFFERED="1")
    return e


def _cmd(model, secret, compile, taint):
    c = ["python", "measured_run.py", model, secret]
    if compile:
        c.append("--compile")
    if taint:
        c.append("--taint")
    return c


def callgrind_count(model, secret, compile=False):
    with tempfile.NamedTemporaryFile(suffix=".cg", delete=False) as f:
        out = f.name
    try:
        vg = ["valgrind", "--tool=callgrind", "--instr-atstart=no",
              "--branch-sim=yes", "--cache-sim=no",
              f"--callgrind-out-file={out}", "--quiet"]
        subprocess.run(vg + _cmd(model, secret, compile, taint=False),
                       cwd=HERE, env=_env(), timeout=TIMEOUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return _parse_callgrind(out)
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def _parse_callgrind(path):
    # We gate *instrumentation* (--instr-atstart=no + CALLGRIND_START/STOP), so
    # the fully-counted region is the "totals:" line. "summary:" tracks
    # *collection*, which we leave off and which therefore reads 0.
    events, totals = None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("events:"):
                events = line.split(":", 1)[1].split()
            elif line.startswith("totals:"):
                totals = [int(x) for x in line.split(":", 1)[1].split()]
    if not events or not totals:
        raise RuntimeError(f"could not parse callgrind output {path}")
    m = dict(zip(events, totals))
    return {"Ir": m.get("Ir", 0), "Bc": m.get("Bc", 0), "Bi": m.get("Bi", 0),
            "Dr": m.get("Dr", 0), "Dw": m.get("Dw", 0)}


# ---------------------------------------------------------------------------
# Generalized runners: instrument an ARBITRARY command (e.g. a compiled C
# binary) instead of `python measured_run.py`. The C harness gates callgrind
# with CALLGRIND_START/STOP and prints the same taint-region markers, so the
# existing parsers apply unchanged. `cache_sim=True` adds Dr/Dw (the CSI-NN
# memory-store attack surface).
# ---------------------------------------------------------------------------
def callgrind_count_cmd(cmd, cwd=HERE, env=None, cache_sim=False):
    with tempfile.NamedTemporaryFile(suffix=".cg", delete=False) as f:
        out = f.name
    try:
        vg = ["valgrind", "--tool=callgrind", "--instr-atstart=no",
              "--branch-sim=yes",
              "--cache-sim=yes" if cache_sim else "--cache-sim=no",
              f"--callgrind-out-file={out}", "--quiet"]
        subprocess.run(vg + list(cmd), cwd=cwd, env=env or _env(), timeout=TIMEOUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return _parse_callgrind(out)
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def memcheck_taint_cmd(cmd, cwd=HERE, env=None):
    with tempfile.NamedTemporaryFile(suffix=".mc", delete=False) as f:
        log = f.name
    try:
        vg = ["valgrind", "--tool=memcheck", "--track-origins=yes",
              "--error-exitcode=0", f"--log-file={log}"]
        subprocess.run(vg + list(cmd), cwd=cwd, env=env or _env(), timeout=TIMEOUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return _parse_memcheck(log)
    finally:
        try:
            os.unlink(log)
        except OSError:
            pass


_LEAK_PAT = re.compile(r"depends on uninitialised value", re.I)
_BEGIN = "taint region begin"
_END = "taint region end"


def memcheck_taint(model, secret, compile=False):
    with tempfile.NamedTemporaryFile(suffix=".mc", delete=False) as f:
        log = f.name
    try:
        vg = ["valgrind", "--tool=memcheck", "--track-origins=yes",
              "--error-exitcode=0", f"--log-file={log}"]
        subprocess.run(vg + _cmd(model, secret, compile, taint=True),
                       cwd=HERE, env=_env(), timeout=TIMEOUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return _parse_memcheck(log)
    finally:
        try:
            os.unlink(log)
        except OSError:
            pass


def _parse_memcheck(path):
    """Count leak reports that occur strictly inside the marked taint region."""
    in_region = False
    reports = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if _BEGIN in line:
                in_region = True
                continue
            if _END in line:
                in_region = False
                continue
            if in_region and _LEAK_PAT.search(line):
                reports.append(line.strip())
    return {"leak": len(reports) > 0, "reports": reports}
