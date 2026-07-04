"""
Shared abstractions for extraction attacks.

The point of this module is the seam between *what leaks* and *how it is
exploited*. An attack is written against these interfaces only, so a new target
(a different model, a different observable) or a new algorithm can be added
without editing the other side.

  Oracle           the black-box view of a target: input vector -> scalar
                   observable (latency, a layer count, ...). The ONLY surface an
                   attacker is allowed to touch.
  Labeler          collapses an oracle observable into the discrete class the
                   attacker actually learns.
  ExtractionAttack an algorithm that learns a surrogate of the target's secret
                   decision boundary from label queries alone.
  AttackResult     the surrogate plus accounting (how many queries it cost).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Oracle(Protocol):
    """Black-box target: one input vector -> one scalar observable.

    Whatever physically leaks is exposed here and nowhere else. An attack that
    depends only on ``query`` cannot accidentally reach into the target's
    internals, which is what keeps the attack reusable across targets.
    """

    def query(self, x: np.ndarray) -> float:
        ...


@runtime_checkable
class Labeler(Protocol):
    """Maps an oracle observable to the discrete class an attacker learns."""

    def label(self, x: np.ndarray) -> int:
        ...


@dataclass
class AttackResult:
    """Outcome of an extraction run.

    ``surrogate`` is any object exposing ``predict(X) -> array`` (e.g. a fitted
    scikit-learn estimator). ``history`` holds per-round diagnostics for reports.
    """

    surrogate: Any
    queries_used: int
    history: list = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.surrogate.predict(X)


class ExtractionAttack(ABC):
    """A secret-extraction attack: learn a surrogate of a target's decision
    boundary using only label queries.

    Depends on ``Labeler`` (which depends on ``Oracle``), never on the target
    itself -- the inversion of control that lets one attack run against any
    target satisfying the interface.
    """

    @abstractmethod
    def run(self, labeler: Labeler, dim: int, budget: int) -> AttackResult:
        """Spend at most ``budget`` label queries recovering a ``dim``-dimensional
        decision boundary."""
        ...
