"""
Extraction attack against a GPT with a secret early-exit gate.

See ``run.py`` for the entry point and ``../README.md`` for the write-up.
"""

from .config import AttackConfig
from .enclave import EarlyExitEnclave
from .oracle import LayerCountOracle, LogitOracle, ThresholdLabeler, TimingOracle
from .extraction import SurrogateExtractionAttack
from .logit_extraction import RegressionExtractionAttack
from .mlp_extraction import MlpLogitExtractionAttack
from .mlp_surrogate import TorchMLPRegressor

__all__ = [
    "AttackConfig",
    "EarlyExitEnclave",
    "TimingOracle",
    "LayerCountOracle",
    "LogitOracle",
    "ThresholdLabeler",
    "SurrogateExtractionAttack",
    "RegressionExtractionAttack",
    "MlpLogitExtractionAttack",
    "TorchMLPRegressor",
]
