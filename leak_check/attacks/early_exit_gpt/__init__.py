"""
Extraction attack against a GPT with a secret early-exit gate.

See ``run.py`` for the entry point and ``../README.md`` for the write-up.
"""

from .config import AttackConfig
from .enclave import EarlyExitEnclave
from .oracle import LayerCountOracle, ThresholdLabeler, TimingOracle
from .extraction import SurrogateExtractionAttack

__all__ = [
    "AttackConfig",
    "EarlyExitEnclave",
    "TimingOracle",
    "LayerCountOracle",
    "ThresholdLabeler",
    "SurrogateExtractionAttack",
]
