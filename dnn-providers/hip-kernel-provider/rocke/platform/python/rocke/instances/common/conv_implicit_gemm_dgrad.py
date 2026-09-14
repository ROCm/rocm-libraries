# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-exports ``kernels.common.conv_implicit_gemm_dgrad`` (implicit-GEMM backward-data convolution).

Backward-compatible re-export for callers still importing from the old
``rocke.instances.common`` location. New code should import directly from
``kernels.common.conv_implicit_gemm_dgrad`` instead.
"""

from kernels.common.conv_implicit_gemm_dgrad import *  # noqa: F401,F403
