# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deprecated alias for the compiled extension; import from ``origami`` instead."""

import warnings

from ._pyorigami import *  # noqa: F401,F403

warnings.warn(
    "origami.origami is deprecated; import names from the origami package directly",
    DeprecationWarning,
    stacklevel=2,
)
