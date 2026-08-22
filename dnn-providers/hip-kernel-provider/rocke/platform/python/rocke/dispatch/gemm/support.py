# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMM support predicates driven by hardware and tile configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.arch import ArchTarget
from ...instances.common.gemm_universal import UniversalGemmSpec
from .common import GemmRequest, normalize_dtype


@dataclass(frozen=True)
class GemmSupportQuery:
    """Explicit support query for one GEMM candidate configuration.

    This mirrors the dispatcher/tile_engine split between operation signature and
    algorithm knobs. The fields here are the knobs that determine whether a
    kernel can be built on a target architecture, independent of one particular
    runtime shape.
    """

    arch: str
    cta_tile: Tuple[int, int, int]
    warp_shape: Tuple[int, int, int]
    warp_tile: Tuple[int, int, int]
    dtype_a: str
    dtype_b: str
    dtype_c: str
    dtype_acc: str
    layout: str
    pipeline: str
    scheduler: str
    epilogue: str
    wave_size: int
    block_size: int
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False
    preshuffle_b: bool = False
    direct_to_lds: bool = False
    dtl_prefetch: bool = False
    active_tile_skip: bool = False
    chiplet_swizzle: bool = False


def support_query_from_universal_spec(
    spec: UniversalGemmSpec, *, arch: str
) -> GemmSupportQuery:
    t = spec.tile
    tr = spec.trait
    d = spec.data
    return GemmSupportQuery(
        arch=arch,
        cta_tile=(t.tile_m, t.tile_n, t.tile_k),
        warp_shape=(t.warp_m, t.warp_n, t.warp_k),
        warp_tile=(t.warp_tile_m, t.warp_tile_n, t.warp_tile_k),
        dtype_a=d.dtype_a,
        dtype_b=d.dtype_b,
        dtype_c=d.dtype_c,
        dtype_acc=d.dtype_acc,
        layout=d.layout,
        pipeline=tr.pipeline,
        scheduler=tr.scheduler,
        epilogue=tr.epilogue,
        wave_size=spec.wave_size,
        block_size=spec.block_size,
        pad_m=tr.pad_m,
        pad_n=tr.pad_n,
        pad_k=tr.pad_k,
        persistent=tr.persistent,
        preshuffle_b=tr.preshuffle_b,
        direct_to_lds=tr.direct_to_lds,
        dtl_prefetch=tr.dtl_prefetch,
        active_tile_skip=tr.active_tile_skip,
        chiplet_swizzle=tr.chiplet_swizzle,
    )


def _mma_family(target: ArchTarget) -> str:
    return "wmma" if target.wave_size == 32 else "mma"


_DTYPE_BYTES = {
    "fp16": 2,
    "f16": 2,
    "bf16": 2,
    "fp8": 1,
    "f8": 1,
    "bf8": 1,
    "fp8e4m3": 1,
    "bf8e5m2": 1,
    "i8": 1,
    "fp32": 4,
    "f32": 4,
    "i32": 4,
}


def _dtype_bytes(dtype: str) -> float:
    d = normalize_dtype(dtype)
    try:
        return _DTYPE_BYTES[d]
    except KeyError as exc:
        raise ValueError(f"unknown GEMM dtype {dtype!r} for LDS sizing") from exc


# Per-pipeline LDS budget, mirrored from the canonical codegen validator
# (``dispatcher/codegen/arch_filter.py`` ``LDS_CAPACITY_LIMITS``). compv4 and
# preshuffle pipelines reserve part of LDS, so their usable budget is smaller
# than the hardware capacity.
_LDS_CAPACITY_LIMITS = {"compv4": 32768, "preshufflev2": 32768, "default": 65536}


def _lds_bytes(q: GemmSupportQuery) -> float:
    # Mirror dispatcher/codegen/arch_filter.py::_validate_lds_capacity: a single
    # A-tile + B-tile staged in LDS. The canonical validator (final source of
    # truth) intentionally does NOT add double-buffer or C-shuffle bytes here,
    # so we don't either -- the previous DSL estimate was strictly more gating.
    tm, tn, tk = q.cta_tile
    matrix_a = tm * tk * _dtype_bytes(q.dtype_a)
    matrix_b = tn * tk * _dtype_bytes(q.dtype_b)
    return matrix_a + matrix_b


def _lds_capacity(q: GemmSupportQuery) -> int:
    return _LDS_CAPACITY_LIMITS.get(q.pipeline, _LDS_CAPACITY_LIMITS["default"])


def gemm_config_supported(q: GemmSupportQuery) -> Tuple[bool, str]:
    """Return whether a GEMM candidate configuration is viable on an arch.

    This predicate is a function of CTA tile, warp shape, warp tile/MMA atom,
    dtype, algorithmic traits, and hardware facts from
    :class:`rocke.core.arch.ArchTarget`.
    """
    try:
        target = ArchTarget.from_gfx(q.arch)
    except KeyError as e:
        return False, str(e)

    if q.layout.upper() != "RCR":
        return False, f"unsupported GEMM layout {q.layout!r}"
    if q.dtype_a != q.dtype_b or q.dtype_a != q.dtype_c:
        return False, (
            "phase-1 UniversalGemm requires homogeneous A/B/C dtypes; "
            f"got A={q.dtype_a}, B={q.dtype_b}, C={q.dtype_c}"
        )
    if normalize_dtype(q.dtype_acc) != "fp32":
        return (
            False,
            f"phase-1 UniversalGemm requires fp32 accumulation, got {q.dtype_acc!r}",
        )
    family = _mma_family(target)
    atom_m, atom_n, atom_k = q.warp_tile
    dtype = normalize_dtype(q.dtype_a)
    if not target.supports_dtype_combo(dtype, dtype, "fp32", family=family):
        return False, f"unsupported GEMM dtype {dtype!r} on {q.arch}"
    if not target.mma.has_shape(
        family=family,
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=atom_m,
        n=atom_n,
        k=atom_k,
    ):
        return (
            False,
            f"unsupported {dtype} {family} warp_tile {q.warp_tile} on {q.arch}",
        )

    if q.wave_size != target.wave_size:
        return (
            False,
            f"spec wave_size {q.wave_size} != {q.arch} wave_size {target.wave_size}",
        )

    tm, tn, tk = q.cta_tile
    wm, wn, wk = q.warp_shape
    if tm % (wm * atom_m):
        return False, "tile_m not divisible by warp_m * warp_tile_m"
    if tn % (wn * atom_n):
        return False, "tile_n not divisible by warp_n * warp_tile_n"
    if tk % (wk * atom_k):
        return False, "tile_k not divisible by warp_k * warp_tile_k"

    expected_bs = wm * wn * wk * q.wave_size
    if expected_bs != q.block_size:
        return False, (
            f"block_size {q.block_size} != warp_m*warp_n*warp_k*wave_size = {expected_bs}"
        )
    if q.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {q.block_size} > {target.max_threads_per_block} hardware cap on {q.arch}"
        )

    if family == "wmma":
        if q.warp_tile != (16, 16, 16):
            return False, f"WMMA path supports only 16x16x16, got {q.warp_tile}"
        if q.pipeline != "mem":
            return (
                False,
                f"WMMA path supports only the 'mem' pipeline, got {q.pipeline!r}",
            )
        if q.epilogue != "default":
            return (
                False,
                f"WMMA path supports only the 'default' epilogue, got {q.epilogue!r}",
            )
        for flag, label in (
            (q.preshuffle_b, "preshuffle_b"),
            (q.direct_to_lds, "direct_to_lds"),
            (q.dtl_prefetch, "dtl_prefetch"),
            (q.active_tile_skip, "active_tile_skip"),
            (q.chiplet_swizzle, "chiplet_swizzle"),
        ):
            if flag:
                return False, f"WMMA path does not support {label} on {q.arch}"

    lds_used = _lds_bytes(q)
    lds_cap = min(_lds_capacity(q), target.lds_capacity_bytes)
    if lds_used > lds_cap:
        return (
            False,
            f"LDS budget {lds_used} > {lds_cap} cap on {q.arch}",
        )

    if tm * tk < q.block_size or tn * tk < q.block_size:
        return False, "block too small for one element/thread/phase"

    return True, "ok"


def request_shape_supported(
    req: GemmRequest, spec: UniversalGemmSpec
) -> Tuple[bool, str]:
    """Return whether ``req.M/N/K`` satisfy the spec's padding/granularity."""
    t = spec.tile
    tr = spec.trait
    checks = (
        ("M", req.M, t.tile_m, tr.pad_m),
        ("N", req.N, t.tile_n, tr.pad_n),
        ("K", req.K, t.tile_k, tr.pad_k),
    )
    for name, dim, tile, padded in checks:
        if not padded and dim % tile:
            return (
                False,
                f"{name}={dim} is not divisible by tile_{name.lower()}={tile} "
                f"and pad_{name.lower()} is disabled",
            )
    return True, "ok"
