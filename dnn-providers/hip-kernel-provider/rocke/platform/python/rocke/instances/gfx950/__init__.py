# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 (CDNA4 / MI350-MI355) arch-specific instance builders (hybrid layout).

Put a kernel here only when its *algorithm* genuinely exploits a gfx950-only
capability (e.g. ``ds_read_*_tr_*`` transpose-LDS, MX fp4/fp6 MFMA, the wider
``16x16x32`` / ``32x32x16`` atoms) in a way that changes the kernel structure
versus the shared ``instances/common/`` version. Shared, arch-polymorphic
kernels belong in ``instances/common/``.

See ``dsl_docs/architecture/multi_arch_data_layout.md``.
"""

from .deep_fused_conv_pool import (  # noqa: F401
    FusedConvPoolProblem,
    Gfx950DeepFusedConvPoolSpec,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    deep_fused_conv_pool_signature,
    is_valid_spec as is_valid_deep_fused_conv_pool_spec,
)
