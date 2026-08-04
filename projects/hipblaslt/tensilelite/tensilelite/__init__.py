# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public package boundary for the ROCm-coupled TensileLite generator."""

from importlib.metadata import version as _distribution_version

from . import _runtime


# This is the compatibility version written to generated logic/configuration
# files. It is intentionally independent from the ROCm-tagged wheel version.
GENERATOR_VERSION = "5.0.0"

__version__ = _distribution_version("tensilelite")
_runtime.initialize(__version__)

__all__ = [
    "GENERATOR_VERSION",
    "__version__",
]
