# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-exports ``kernels.common.conv_direct_grouped`` (streaming row-by-row direct grouped convolution).

Backward-compatible re-export for callers still importing from the old
``rocke.instances.common`` location. New code should import directly from
``kernels.common.conv_direct_grouped`` instead.
"""

from kernels.common.conv_direct_grouped import *  # noqa: F401,F403
