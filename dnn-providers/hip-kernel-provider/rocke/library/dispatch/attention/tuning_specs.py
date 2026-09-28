# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Policy-free kernel spec construction for attention tuning candidates.

This module is shared by gfx942/gfx950 dispatcher candidates.  It owns no
selection heuristic: geometry and codegen knobs are explicit, while semantic
fields come from the operator problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _tiled_2d_impl,
    _tiled_3d_impl,
)


TILE_POLICIES: Tuple[str, ...] = (
    "half",
    "1x",
    "2x",
    "4x",
    "8x",
    "64",
    "128",
)
NUM_WARPS: Tuple[int, ...] = (1, 2, 4, 8)
BLOCK_M_PER_WARP: Tuple[int, ...] = (16, 32)
WAVES_PER_EU: Tuple[Optional[int], ...] = (None, 1, 2, 3, 4)
NUM_SEGMENTS: Tuple[int, ...] = (8, 16, 32, 64, 128)


def validate_explicit_fp8_encoding(*, arch: str, use_fp8: bool, fp8_fnuz: bool) -> None:
    """Reject an explicit spec whose declared FP8 bytes mismatch the target."""
    if not use_fp8:
        if fp8_fnuz:
            raise ValueError("fp8_fnuz=True requires use_fp8=True")
        return

    from kernels.common.fmha_fwd_fp8 import _FNUZ_FP8_TARGET_FAMILIES
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    target_is_fnuz = target.target_family in _FNUZ_FP8_TARGET_FAMILIES
    if bool(fp8_fnuz) != target_is_fnuz:
        expected = "FNUZ" if target_is_fnuz else "OCP"
        declared = "FNUZ" if fp8_fnuz else "OCP"
        raise ValueError(
            f"explicit FP8 tuning spec declares {declared} bytes, but {arch} "
            f"requires {expected}"
        )


@dataclass(frozen=True)
class ExplicitAttention2DConfig:
    """A complete, non-heuristic 2D geometry plus codegen setting."""

    num_warps: int
    block_m_per_warp: int
    tile_policy: str
    waves_per_eu: Optional[int] = None
    compile_backend: str = "llvm"
    builder_kind: str = "tiled"
    knobs: Tuple[Tuple[str, object], ...] = ()

    def knob_dict(self) -> dict[str, object]:
        return dict(self.knobs)


@dataclass(frozen=True)
class ExplicitAttention3DConfig:
    """A complete, non-heuristic 3D segment/reduce setting."""

    num_segments: int
    tile_policy: str = "1x"
    waves_per_eu: Optional[int] = None
    knobs: Tuple[Tuple[str, object], ...] = ()

    def knob_dict(self) -> dict[str, object]:
        return dict(self.knobs)


def resolve_tile_policy(block_size: int, policy: str) -> int:
    """Resolve a finite dispatcher tile token without a production heuristic."""
    token = str(policy).strip().lower()
    if token == "half":
        if int(block_size) < 32:
            raise ValueError("half-block tile requires block_size >= 32")
        return int(block_size) // 2
    if token.endswith("x") and token[:-1].isdigit():
        mult = int(token[:-1])
        if mult not in (1, 2, 4, 8):
            raise ValueError(f"unsupported tile multiplier {policy!r}")
        return int(block_size) * mult
    if token in ("64", "128"):
        tile = int(token)
        if tile % int(block_size):
            raise ValueError(
                f"tile {tile} is not a multiple of block_size={block_size}"
            )
        return tile
    raise ValueError(f"tile_policy must be one of {TILE_POLICIES}, got {policy!r}")


def _semantic_fields(problem: UnifiedAttentionProblem) -> dict:
    kv_storage_dtype = "fp8e4m3" if problem.use_fp8 else None
    elem_bytes = 1 if problem.use_fp8 else 2
    block_stride = (
        int(problem.block_size)
        * int(problem.num_kv_heads)
        * int(problem.head_size)
        * elem_bytes
    )
    use_i64 = (
        int(problem.num_kv_blocks) > 0
        and int(problem.num_kv_blocks) * block_stride > 0x8000_0000
    )
    return {
        "head_size": int(problem.head_size),
        "block_size": int(problem.block_size),
        "num_query_heads": int(problem.num_query_heads),
        "num_kv_heads": int(problem.num_kv_heads),
        "dtype": str(problem.dtype),
        "use_sinks": bool(problem.use_sinks),
        "sliding_window": int(problem.sliding_window),
        "has_softcap": bool(problem.softcap > 0),
        "use_alibi": bool(problem.use_alibi),
        "use_qq_bias": bool(problem.use_qq_bias),
        "num_seqs": int(problem.num_seqs),
        "kv_storage_dtype": kv_storage_dtype,
        "use_i64_kv_addr": use_i64,
    }


_SEMANTIC_FIELDS = frozenset(
    {
        "head_size",
        "block_size",
        "num_query_heads",
        "num_kv_heads",
        "dtype",
        "use_sinks",
        "sliding_window",
        "has_softcap",
        "use_alibi",
        "use_qq_bias",
        "num_seqs",
        "kv_storage_dtype",
        "use_i64_kv_addr",
    }
)


def _checked_knobs(knobs: Mapping[str, object]) -> dict[str, object]:
    overlap = _SEMANTIC_FIELDS.intersection(knobs)
    if overlap:
        raise ValueError(
            "tuning knobs cannot override semantic fields: "
            + ", ".join(sorted(overlap))
        )
    return dict(knobs)


def make_explicit_attention_2d_spec(
    problem: UnifiedAttentionProblem,
    config: ExplicitAttention2DConfig,
    *,
    arch: str,
):
    """Construct one explicit 2D kernel spec and reject unsupported geometry."""
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(
            f"explicit 2D attention tuning requires gfx942/gfx950, got {arch}"
        )
    validate_explicit_fp8_encoding(
        arch=arch, use_fp8=problem.use_fp8, fp8_fnuz=problem.fp8_fnuz
    )
    if config.num_warps not in NUM_WARPS:
        raise ValueError(f"num_warps must be one of {NUM_WARPS}")
    if config.block_m_per_warp not in BLOCK_M_PER_WARP:
        raise ValueError(f"block_m_per_warp must be one of {BLOCK_M_PER_WARP}")
    if config.waves_per_eu not in WAVES_PER_EU:
        raise ValueError(f"waves_per_eu must be one of {WAVES_PER_EU}")
    if config.compile_backend not in ("llvm", "hipcc"):
        raise ValueError("compile_backend must be 'llvm' or 'hipcc'")
    if arch == "gfx942" and config.compile_backend != "llvm":
        raise ValueError(
            "gfx942 explicit attention supports only the validated llvm backend"
        )

    spec_type, _, supports = _tiled_2d_impl(arch)
    tile_size = resolve_tile_policy(problem.block_size, config.tile_policy)
    knobs = _checked_knobs(config.knob_dict())
    spec = spec_type(
        **_semantic_fields(problem),
        num_warps=int(config.num_warps),
        block_m_per_warp=int(config.block_m_per_warp),
        tile_size=int(tile_size),
        waves_per_eu=config.waves_per_eu,
        **knobs,
    )

    support_kwargs = dict(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
        num_warps=spec.num_warps,
        block_m_per_warp=spec.block_m_per_warp,
        kv_storage_dtype=spec.kv_storage_dtype,
        tile_size=spec.tile_size,
        arch=arch,
        use_mfma_32x32x8=bool(getattr(spec, "use_mfma_32x32x8", False)),
        use_transposed_qk_32x32=bool(spec.use_transposed_qk_32x32),
        use_k_single_buffer=bool(spec.use_k_single_buffer),
        use_conflict_free_v_store=bool(
            getattr(spec, "use_conflict_free_v_store", False)
        ),
        use_k_sliced_ring=bool(getattr(spec, "use_k_sliced_ring", False)),
        use_d256_fast=config.builder_kind == "gfx942_4warp_gqa",
    )
    if arch == "gfx942":
        support_kwargs.update(
            ring_depth=int(getattr(spec, "ring_depth", 3)),
            k_slice_hd=int(getattr(spec, "k_slice_hd", 32)),
        )
    ok, why = supports(**support_kwargs)
    if not ok:
        raise ValueError(f"unsupported explicit 2D attention config: {why}")
    return spec


def make_explicit_attention_3d_specs(
    problem: UnifiedAttentionProblem,
    config: ExplicitAttention3DConfig,
    *,
    arch: str,
):
    """Construct explicit segment and reduce specs without segment heuristics."""
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(
            f"explicit 3D attention tuning requires gfx942/gfx950, got {arch}"
        )
    validate_explicit_fp8_encoding(
        arch=arch, use_fp8=problem.use_fp8, fp8_fnuz=problem.fp8_fnuz
    )
    if config.num_segments not in NUM_SEGMENTS:
        raise ValueError(f"num_segments must be one of {NUM_SEGMENTS}")
    if config.waves_per_eu not in WAVES_PER_EU:
        raise ValueError(f"waves_per_eu must be one of {WAVES_PER_EU}")

    tile = resolve_tile_policy(problem.block_size, config.tile_policy)
    if arch == "gfx950" and tile != int(problem.block_size):
        raise ValueError("gfx950 3D implements only tile_size == block_size")
    if arch == "gfx942" and tile not in (
        int(problem.block_size),
        max(1, int(problem.block_size) // 2),
    ):
        raise ValueError("gfx942 3D implements only full- or half-block tiles")

    spec_type, reduce_type, _, _, supports = _tiled_3d_impl(arch)
    knobs = _checked_knobs(config.knob_dict())
    if arch == "gfx950" and any(
        bool(knobs.get(name, False))
        for name in ("use_invariant_hoist", "use_wide_kv_load")
    ):
        raise ValueError("gfx950 3D does not implement hoist/wide-KV tuning knobs")
    segment = spec_type(
        **_semantic_fields(problem),
        num_segments=int(config.num_segments),
        waves_per_eu=config.waves_per_eu,
        tile_size_override=(None if tile == int(problem.block_size) else int(tile)),
        **knobs,
    )
    ok, why = supports(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
        kv_storage_dtype=segment.kv_storage_dtype,
        arch=arch,
    )
    if not ok:
        raise ValueError(f"unsupported explicit 3D attention config: {why}")
    reduce = reduce_type(
        head_size=problem.head_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        num_segments=int(config.num_segments),
        waves_per_eu=config.waves_per_eu,
    )
    return segment, reduce


def build_explicit_attention_2d(spec, *, arch: str):
    """Emit IR for an explicit 2D spec."""
    _, build, _ = _tiled_2d_impl(arch)
    return build(spec, arch=arch)


def build_explicit_attention_3d(spec, reduce_spec, *, arch: str):
    """Emit the explicit split-KV segment and reduce IR pair."""
    _, _, build_segment, build_reduce, _ = _tiled_3d_impl(arch)
    return (
        build_segment(spec, arch=arch),
        build_reduce(reduce_spec, arch=arch),
    )
