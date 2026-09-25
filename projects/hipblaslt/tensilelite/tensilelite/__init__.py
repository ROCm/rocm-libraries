# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public package boundary for the ROCm-coupled TensileLite generator."""

from _tensilelite_client_binding import current_installation

from . import _runtime


# This is the compatibility version written to generated logic/configuration
# files. It is intentionally independent from the ROCm-tagged wheel version.
GENERATOR_VERSION = "5.0.0"

__version__ = current_installation().version

__all__ = [
    "GENERATOR_VERSION",
    "__version__",
]

_runtime.initialize()
