# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public package boundary for the ROCm-coupled TensileLite generator."""

from importlib.metadata import version as _distribution_version

from ._runtime import RuntimeInfo, validate_runtime


# This is the compatibility version written to generated logic/configuration
# files. It is intentionally independent from the ROCm-tagged wheel version.
GENERATOR_VERSION = "5.0.0"

__version__ = _distribution_version("tensilelite")
RUNTIME: RuntimeInfo = validate_runtime(__version__)
TENSILELITE_CLIENT_PATH = RUNTIME.client

__all__ = [
    "GENERATOR_VERSION",
    "RUNTIME",
    "TENSILELITE_CLIENT_PATH",
    "__version__",
]
