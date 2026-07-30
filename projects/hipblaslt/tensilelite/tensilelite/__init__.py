# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public package boundary for the ROCm-coupled TensileLite generator."""

from importlib.metadata import version as _distribution_version
from os import path

from ._runtime import RuntimeInfo, validate_runtime


# This is the compatibility version written to generated logic/configuration
# files. It is intentionally independent from the ROCm-tagged wheel version.
GENERATOR_VERSION = "5.0.0"

__version__ = _distribution_version("tensilelite")
RUNTIME: RuntimeInfo = validate_runtime(__version__)
TENSILELITE_CLIENT_PATH = RUNTIME.client

# Retained until the canonical CLI replaces the legacy TensileGetPath entry
# point in the next logical change.
ROOT_PATH: str = path.dirname(__file__)


def PrintTensileRoot():
    print(ROOT_PATH, end="")

__all__ = [
    "GENERATOR_VERSION",
    "RUNTIME",
    "TENSILELITE_CLIENT_PATH",
    "ROOT_PATH",
    "PrintTensileRoot",
    "__version__",
]
