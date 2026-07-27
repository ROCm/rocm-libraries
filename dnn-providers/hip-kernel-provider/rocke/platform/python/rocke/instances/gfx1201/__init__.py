# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1201 (RDNA4, wave32, WMMA) arch-specific instance builders (hybrid layout).

Put a kernel here only when its *algorithm* genuinely exploits a gfx1201-only
capability in a way that changes the kernel structure versus the shared
``instances/common/`` version. Shared, arch-polymorphic kernels (e.g. the
deep-fused conv/pool body, driven by the resolved ``MmaOp``) live in
``instances/common/``; this module only pins the WMMA geometry and re-exports.
"""

from .deep_fused_conv_pool import (  # noqa: F401
    FusedConvPoolProblem,
    Gfx1201DeepFusedConvPoolSpec,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    deep_fused_conv_pool_signature,
    is_valid_spec as is_valid_deep_fused_conv_pool_spec,
)
