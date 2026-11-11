# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public package boundary for the ROCm-coupled TensileLite generator."""

from _tensilelite_client_binding import current_installation

from . import _runtime


# This is the compatibility version written to generated logic/configuration
# files. It is intentionally independent from the ROCm-tagged wheel version.
GENERATOR_VERSION = "5.0.0"

# The version of the package that owns this import, not of any other
# tensilelite metadata on sys.path.
__version__ = current_installation().version

__all__ = [
    "GENERATOR_VERSION",
    "__version__",
]

_runtime.initialize()
