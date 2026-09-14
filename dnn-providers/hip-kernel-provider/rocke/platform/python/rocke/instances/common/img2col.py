# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-exports ``kernels.common.img2col`` (image-to-column (im2col) kernel instance).

Backward-compatible re-export for callers still importing from the old
``rocke.instances.common`` location. New code should import directly from
``kernels.common.img2col`` instead.
"""

from kernels.common.img2col import *  # noqa: F401,F403
