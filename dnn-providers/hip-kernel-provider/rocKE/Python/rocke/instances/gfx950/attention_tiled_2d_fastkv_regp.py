# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Experimental 2D attention kernel: fast paged-KV descriptor + register P.

This module intentionally leaves ``attention_tiled_2d.py`` untouched.  It
builds a copied experimental entry point by wrapping the current tiled R4 spec
and forcing the builder down the register-P allocation path, which removes the
otherwise-unused ``P_lds`` slab for the transposed 32x32 R4 dataflow.

The actual QK/softmax/PV math stays in the production tiled builder so this
experiment measures the targeted residency/resource change rather than a second
hand-maintained fork of the full kernel body.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...core.ir import KernelDef
from .attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
    supports_tiled_2d,
)


class _FastKvRegisterPProxy:
    """Spec proxy that skips the unused P_lds allocation on transposed R4."""

    def __init__(self, spec: UnifiedAttention2DTiledSpec) -> None:
        self._spec = spec

    def __getattr__(self, name: str) -> Any:
        return getattr(self._spec, name)

    @property
    def use_register_pv(self) -> bool:
        # In the transposed 32x32 path the PV consumer reads PT32_n directly
        # from registers.  Setting this through the proxy avoids allocating
        # P_lds without relaxing the production spec validator.
        return True

    def kernel_name(self) -> str:
        return f"{self._spec.kernel_name()}_fastkv_regp"


def make_fastkv_register_p_spec(
    spec: UnifiedAttention2DTiledSpec,
    *,
    scalar_state: bool = False,
    mask_once: bool = False,
    mask_limit: bool | None = None,
    half_local_pv: bool = False,
    skip_legacy_qreg: bool = False,
) -> UnifiedAttention2DTiledSpec:
    """Return a targeted fastKV + register-P R4 spec.

    The experiment is scoped to the dominant bf16 prefill-2D trace shape family:
    ``d64_b32_h64kv8`` with ``T=64`` and ``num_warps=4``.  By default this keeps
    the R4 softmax/PV flags unchanged except for fast paged-KV and the
    register-P residency proxy, so it can be compared directly against R4.
    Callers can opt into the current s1/mask/HLPV/skip-qreg stack when they want
    to measure the broader combo policy.
    """

    use_mask_limit = (
        scalar_state
        and mask_once
        and spec.sliding_window == 0
        and not spec.has_softcap
        and not spec.use_alibi
        and not spec.use_qq_bias
        if mask_limit is None
        else mask_limit
    )
    return replace(
        spec,
        num_warps=4,
        waves_per_eu=2,
        tile_size=2 * spec.block_size,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=scalar_state,
        use_transposed_mask_once=mask_once,
        use_transposed_half_local_pv=half_local_pv,
        use_mfma32_skip_legacy_qreg=skip_legacy_qreg,
        use_transposed_mask_limit=use_mask_limit,
        use_fast_paged_kv_desc=True,
        use_agpr_alloc_zero=False,
        use_register_pv=False,
    )


def supports_fastkv_register_p_2d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_fp8: bool,
    q_dtype,
    num_query_heads: int,
    num_kv_heads: int,
    tile_size: int | None = None,
    arch: str = "gfx950",
) -> tuple[bool, str]:
    """Check support for the experimental fastKV + register-P kernel."""

    ok, reason = supports_tiled_2d(
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        num_queries_per_kv=num_queries_per_kv,
        use_alibi=use_alibi,
        use_qq_bias=use_qq_bias,
        use_fp8=use_fp8,
        q_dtype=q_dtype,
        num_warps=4,
        kv_storage_dtype=None,
        tile_size=tile_size if tile_size is not None else 2 * block_size,
        arch=arch,
    )
    if not ok:
        return ok, reason
    if not (
        dtype == "bf16"
        and head_size == 64
        and block_size == 32
        and num_query_heads == 64
        and num_kv_heads == 8
    ):
        return (
            False,
            "fastKV register-P experiment is restricted to bf16 d64_b32_h64kv8",
        )
    return True, "supported"


def build_unified_attention_2d_fastkv_register_p(
    spec: UnifiedAttention2DTiledSpec,
    *,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the experimental fastKV + register-P 2D attention kernel.

    Thin wrapper over :func:`build_unified_attention_2d_tiled`; the ``arch``
    target (gfx950-only, see that builder) is threaded straight through, so
    a gfx942 request fails with the same clean structured error.
    """

    if not spec.use_fast_paged_kv_desc:
        raise ValueError("fastKV register-P experiment requires use_fast_paged_kv_desc")
    if not (spec.use_mfma_32x32 and spec.use_transposed_qk_32x32):
        raise ValueError("fastKV register-P experiment requires transposed R4")
    if spec.kv_storage_dtype is not None:
        raise ValueError("fastKV register-P experiment does not support FP8 KV cache")
    return build_unified_attention_2d_tiled(_FastKvRegisterPProxy(spec), arch=arch)
