# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 (CDNA3 / MI300) arch-specific instance builders (hybrid layout).

Put a kernel here only when its *algorithm* genuinely differs on gfx942 from the
shared ``instances/common/`` version (e.g. a K-loop or staging strategy that
cannot use a gfx950-only primitive such as ``ds_read_*_tr_*``). Shared,
arch-polymorphic kernels belong in ``instances/common/``.

Adding/improving a gfx942 variant here must not edit another arch's files or a
shared ``common/`` builder. See
``dsl_docs/architecture/multi_arch_data_layout.md``.
"""

from .attention_tiled_2d import (  # noqa: F401
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
    supports_tiled_2d,
)

from .attention_tiled_3d import (  # noqa: F401
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    supports_tiled_3d,
)
