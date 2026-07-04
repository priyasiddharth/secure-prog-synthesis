"""
Corpus of operators to probe, plus the two secret classes each is measured
under. Every model exposes a `.weight` Parameter of shape [DIM, DIM]; the secret
lives entirely in that buffer so the harness can mark exactly those bytes.

The trio is chosen to exercise all three verdicts:

  branchless   : always x @ w. Data-OBLIVIOUS execution -> expect NO leak in any
                 build. This is the control that proves the compiler does not
                 manufacture a leak on its own.

  cond_skip    : "skip the matmul when the weights are all zero", expressed with
                 torch.cond so it actually compiles. An AUTHORED data-dependent
                 branch -> expect a leak in BOTH eager and compiled (proving the
                 leak is the author's, not the compiler's).

  where_select : out = where(w > 0, x@w, -x@w). The RESULT depends on secret
                 values, but the execution is a branch-free vectorized select ->
                 expect NO leak. This is the crucial negative control: it shows
                 the method flags data-dependent *execution*, not merely
                 data-dependent *output*.
"""

import torch

DIM = 512  # smaller than the timing harness: Valgrind is ~20-50x slow.


class Branchless(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))

    def forward(self, x):
        return torch.matmul(x, self.weight)


class CondSkip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))

    def forward(self, x):
        def skip(x, w):
            return torch.zeros(x.shape[0], w.shape[1])

        def compute(x, w):
            return torch.matmul(x, w)

        pred = torch.all(self.weight == 0)
        return torch.cond(pred, skip, compute, (x, self.weight))


class WhereSelect(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(DIM, DIM))

    def forward(self, x):
        pos = torch.matmul(x, self.weight)
        neg = -pos
        col_sign = (self.weight.sum(dim=0) > 0)  # [DIM] secret-derived mask
        return torch.where(col_sign, pos, neg)


_MODELS = {
    "branchless": Branchless,
    "cond_skip": CondSkip,
    "where_select": WhereSelect,
}

# What each verdict *should* be, so run_all can report pass/fail against the
# method's own predictions (calibration).
EXPECTED = {
    "branchless": "oblivious",
    "cond_skip": "authored",       # leaks in both builds
    "where_select": "oblivious",   # data-dependent value, oblivious execution
}


def build(name):
    return _MODELS[name]()


def example_input(name):
    return torch.randn(1, DIM)


def names():
    return list(_MODELS.keys())
