"""
The target: a GPT with a secret early-exit gate.

After ``exit_after_layer`` blocks, a secret linear gate looks at the last hidden
state and decides whether the sequence may skip the remaining blocks (an "easy"
input needs less compute -- the standard early-exit / cascade optimization).
Because a skipped block is real work not done, the number of blocks executed --
and therefore the forward-pass latency -- depends on the gate's decision. That
dependence is the leak.

The gate weights are the secret the attacker wants. This class encapsulates them
(and the model, context and configuration that the notebook kept in module
globals). The only *legitimate* attacker view is the latency of ``forward``;
``true_decision`` exposes the ground-truth label and exists solely so the harness
can score how well an attack did -- an attack must never call it.
"""

import time

import numpy as np
import torch

from .model import GPT


class EarlyExitEnclave:
    def __init__(self, cfg, exit_after_layer, context_len=8,
                 synthetic_delay_per_layer=0.0, seed=1, verbose=False):
        if not 0 < exit_after_layer < cfg.n_layer:
            raise ValueError(
                f"exit_after_layer must be in (0, {cfg.n_layer}), got {exit_after_layer}"
            )
        torch.manual_seed(seed)
        self.cfg = cfg
        self.exit_after_layer = exit_after_layer
        self.context_len = context_len
        self.synthetic_delay_per_layer = synthetic_delay_per_layer

        self.model = GPT(cfg, verbose=verbose)
        self.model.eval()
        # THE SECRET: a linear gate over the hidden state. Its weight+bias are
        # what the attacker reconstructs; nothing outside this object reads them.
        self.exit_gate = torch.nn.Linear(cfg.n_embd, 1)
        # A fixed public context; only the final token embedding is the query.
        self.context_ids = torch.randint(0, cfg.vocab_size, (1, context_len - 1))

    @property
    def secret_dim(self):
        """Number of scalar parameters in the gate (weight + bias)."""
        return self.cfg.n_embd + 1

    @property
    def query_dim(self):
        """Dimensionality of an attacker query (the last-token embedding)."""
        return self.cfg.n_embd

    @torch.no_grad()
    def _build_embedding(self, x_last_np):
        T = self.context_len
        ctx_emb = (self.model.transformer.wte(self.context_ids)
                   + self.model.transformer.wpe(torch.arange(T - 1).unsqueeze(0)))
        pos_last = torch.tensor([[T - 1]])
        x_last = torch.tensor(x_last_np, dtype=torch.float32).view(1, 1, -1)
        last_emb = x_last + self.model.transformer.wpe(pos_last)
        return torch.cat([ctx_emb, last_emb], dim=1)

    @torch.no_grad()
    def _run_blocks(self, x, start, end):
        for blk in self.model.transformer.h[start:end]:
            x = blk(x)
        return x

    @torch.no_grad()
    def forward(self, x_last_np, measure=False):
        """Run the enclave. Returns ``(will_exit, n_layers_run, latency)``.

        ``latency`` is ``None`` unless ``measure`` is set. With a nonzero
        ``synthetic_delay_per_layer`` the per-layer cost is amplified so the leak
        is observable above CPU noise on tiny models (the real cost difference on
        a full-size model needs no amplification).
        """
        x = self._build_embedding(x_last_np)
        t0 = time.perf_counter() if measure else None

        x = self._run_blocks(x, 0, self.exit_after_layer)
        h_last = x[0, -1, :]
        will_exit = bool((self.exit_gate(h_last) > 0).item())
        if not will_exit:
            x = self._run_blocks(x, self.exit_after_layer, self.cfg.n_layer)

        n_layers_run = self.exit_after_layer if will_exit else self.cfg.n_layer
        latency = None
        if measure:
            latency = time.perf_counter() - t0
            latency += n_layers_run * self.synthetic_delay_per_layer
        return will_exit, n_layers_run, latency

    @torch.no_grad()
    def gate_logit(self, x_last_np):
        """Raw secret-gate logit ``gate(h)`` at the exit point (the confidence a
        logit-returning API would expose). Sign > 0 means early exit."""
        x = self._build_embedding(x_last_np)
        x = self._run_blocks(x, 0, self.exit_after_layer)
        return float(self.exit_gate(x[0, -1, :]).item())

    @torch.no_grad()
    def true_decision(self, x_last_np):
        """Ground-truth gate label (1 = early exit). Evaluation only -- not part
        of the attacker's view."""
        will_exit, _, _ = self.forward(x_last_np, measure=False)
        return int(will_exit)
