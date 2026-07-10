# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA/MHA kernel definitions (migrated from ``rocke.instances``).

Relative imports below resolve within this ``kernels`` package (``common/`` and
``gfx*/`` mirror the former ``instances`` substructure). Platform primitives are
imported absolutely as ``rocke.*`` from the per-module bodies.
"""

from .common.attention_unified import (  # noqa: F401
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    attention_3d_workspace_nbytes,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
    run_unified_attention_torch,
    supports_native_unified_attention,
    supports_native_unified_attention_tiled,
    supports_native_unified_attention_3d_tiled,
)

# Tiled-2D attention is arch-divergent (gfx950 wide-K/transpose-read vs gfx942
# narrow-atom/strided-V). Route the public re-exports through the arch-aware
# ``_tiled_2d_impl(arch)`` seam instead of binding the gfx950 module directly,
# so no caller resolves the gfx950 builder/gate unconditionally for a gfx942
# request. ``UnifiedAttention2DTiledSpec`` is re-exported from the gfx950 module
# as the default spec shape (the gfx942 spec is a structural superset that only
# adds flag-rejection in ``__post_init__``); arch-specific spec resolution goes
# through ``_tiled_2d_impl(arch)``.
from .gfx950.attention_tiled_2d import (  # noqa: F401
    UnifiedAttention2DTiledSpec,
)


def build_unified_attention_2d_tiled(spec, *, arch: str = "gfx950"):
    """Arch-aware wrapper: dispatch the tiled-2D builder on ``arch``.

    Routes through ``kernels/common/attention_unified._tiled_2d_impl`` so a
    gfx942 request builds the gfx942 narrow-atom variant and a gfx950 request
    (the default) builds the gfx950 wide-K variant -- never the wrong one.
    """
    from .common.attention_unified import _tiled_2d_impl

    _, _build, _ = _tiled_2d_impl(arch)
    return _build(spec, arch=arch)


def supports_tiled_2d(*, arch: str = "gfx950", **kwargs):
    """Arch-aware wrapper: dispatch the tiled-2D gate on ``arch``."""
    from .common.attention_unified import _tiled_2d_impl

    _, _, _supports = _tiled_2d_impl(arch)
    return _supports(arch=arch, **kwargs)


from .gfx950.attention_tiled_3d import (  # noqa: F401
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    supports_tiled_3d,
)

# Full FMHA / Sage / sparse attention public surface, re-exported at the package
# top level to preserve the API that ``rocke.instances`` exposed pre-carve.
from .common._fmha_common import (  # noqa: F401
    FmhaCommonSpec,
    FmhaMaskMode,
    FmhaShape,
    validate_common_spec as validate_fmha_common_spec,
)
from .common.fmha_varlen import (  # noqa: F401
    FmhaFwdVarlenSpec,
    build_fmha_fwd_varlen,
    fmha_fwd_varlen_grid,
    fmha_fwd_varlen_signature,
    is_valid_spec as is_valid_fmha_fwd_varlen_spec,
)
from .common.fmha_appendkv import (  # noqa: F401
    FmhaAppendKvSpec,
    build_fmha_fwd_appendkv,
    fmha_appendkv_grid,
    fmha_appendkv_signature,
    is_valid_spec as is_valid_fmha_appendkv_spec,
)
from .common.fmha_paged_prefill import (  # noqa: F401
    FmhaFwdPagedPrefillSpec,
    build_fmha_fwd_paged_prefill,
    fmha_fwd_paged_prefill_grid,
    fmha_fwd_paged_prefill_signature,
    is_valid_spec as is_valid_fmha_fwd_paged_prefill_spec,
)
from .common.fmha_splitkv_decode import (  # noqa: F401
    FmhaFwdSplitKvDecodeSpec,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
    fmha_fwd_splitkv_decode_reduce_grid,
    fmha_fwd_splitkv_decode_reduce_signature,
    fmha_fwd_splitkv_decode_segment_grid,
    fmha_fwd_splitkv_decode_segment_signature,
    is_valid_spec as is_valid_fmha_fwd_splitkv_decode_spec,
)
from .common.fmha_head_grouping import (  # noqa: F401
    FmhaFwdHeadGroupingSpec,
    build_fmha_fwd_head_grouping,
    fmha_fwd_head_grouping_grid,
    fmha_fwd_head_grouping_signature,
    is_valid_spec as is_valid_fmha_fwd_head_grouping_spec,
)
from .common.fmha_bwd import (  # noqa: F401
    FmhaBwdSpec,
    build_fmha_bwd,
    fmha_bwd_grid,
    fmha_bwd_signature,
    is_valid_spec as is_valid_fmha_bwd_spec,
)
from .common.fmha_fwd_fp8 import (  # noqa: F401
    FmhaFwdFp8Spec,
    build_fmha_fwd_fp8,
    fmha_fwd_fp8_grid,
    fmha_fwd_fp8_signature,
    is_valid_spec as is_valid_fmha_fwd_fp8_spec,
)
from .common.fmha_mfma import (  # noqa: F401
    FmhaMfmaSpec,
    build_fmha_fwd_mfma,
    fmha_fwd_mfma_grid,
    fmha_fwd_mfma_signature,
    is_valid_spec as is_valid_fmha_mfma_spec,
)
from .common.sage_attention import (  # noqa: F401
    SageAttentionSpec,
    SageQuantMode,
    build_sage_attention,
    is_valid_spec as is_valid_sage_attention_spec,
    sage_attention_grid,
    sage_attention_signature,
)
from .common.sparse_attention import (  # noqa: F401
    JengaSparseSpec,
    VsaSparseSpec,
    build_jenga_sparse_attention,
    build_vsa_sparse_attention,
    is_valid_jenga_spec,
    is_valid_vsa_spec,
    jenga_sparse_attention_grid,
    jenga_sparse_attention_signature,
    vsa_sparse_attention_grid,
    vsa_sparse_attention_signature,
)
