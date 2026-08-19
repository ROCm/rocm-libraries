# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public package boundary for the ROCm-coupled TensileLite generator."""

from importlib.metadata import version

from . import _runtime


# This is the compatibility version written to generated logic/configuration
# files. It is intentionally independent from the ROCm-tagged wheel version.
GENERATOR_VERSION = "5.0.0"

__version__ = version("tensilelite")

__all__ = [
    "GENERATOR_VERSION",
    "__version__",
]

_runtime.initialize()
