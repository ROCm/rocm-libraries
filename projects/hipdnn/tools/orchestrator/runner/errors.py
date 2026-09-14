# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Error types. ConfigError means nothing was launched; StepError means something ran."""
from __future__ import annotations


class OrchestratorError(Exception):
    """Base for every failure this tool raises deliberately."""


class ConfigError(OrchestratorError):
    """Invalid declarative input. Raised before any process is launched."""


class RefError(ConfigError):
    """A ${...} reference names something that does not exist."""


class StepError(OrchestratorError):
    """A step ran and did not satisfy its contract (exit code, outputs, asserts)."""


class LoopExhausted(OrchestratorError):
    """A loop reached max_iterations without satisfying its `until` condition."""
