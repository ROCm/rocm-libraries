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
)
from .common.fmha_varlen import (  # noqa: F401
    FmhaFwdVarlenSpec,
    build_fmha_fwd_varlen,
)
from .common.fmha_appendkv import (  # noqa: F401
    FmhaAppendKvSpec,
    build_fmha_fwd_appendkv,
)
from .common.fmha_paged_prefill import (  # noqa: F401
    FmhaFwdPagedPrefillSpec,
    build_fmha_fwd_paged_prefill,
)
from .common.fmha_splitkv_decode import (  # noqa: F401
    FmhaFwdSplitKvDecodeSpec,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
)
from .common.fmha_head_grouping import (  # noqa: F401
    FmhaFwdHeadGroupingSpec,
    build_fmha_fwd_head_grouping,
)
from .common.fmha_bwd import (  # noqa: F401
    FmhaBwdSpec,
    build_fmha_bwd,
)
from .common.fmha_fwd_fp8 import (  # noqa: F401
    FmhaFwdFp8Spec,
    build_fmha_fwd_fp8,
)
from .common.fmha_mfma import (  # noqa: F401
    FmhaMfmaSpec,
    build_fmha_fwd_mfma,
)
from .common.sage_attention import (  # noqa: F401
    SageAttentionSpec,
    SageQuantMode,
    build_sage_attention,
)
from .common.sparse_attention import (  # noqa: F401
    JengaSparseSpec,
    VsaSparseSpec,
    build_jenga_sparse_attention,
    build_vsa_sparse_attention,
)
