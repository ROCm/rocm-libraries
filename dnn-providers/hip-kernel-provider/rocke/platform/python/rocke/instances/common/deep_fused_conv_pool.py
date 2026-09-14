# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-exports ``kernels.common.deep_fused_conv_pool`` (family-agnostic deep-fused conv + maxpool prototype).

Backward-compatible re-export for callers still importing from the old
``rocke.instances.common`` location. New code should import directly from
``kernels.common.deep_fused_conv_pool`` instead.
"""

from kernels.common.deep_fused_conv_pool import *  # noqa: F401,F403
