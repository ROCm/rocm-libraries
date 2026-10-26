# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMM support predicates driven by hardware and tile configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.arch import ArchTarget
from ...instances.common.gemm_universal import (
    DTL_BUFFER_NUM_RECORDS,
    UniversalGemmSpec,
    is_valid_spec,
    launch_shape_error,
    universal_lds_bytes,
)
from ...instances.common.gemm_mn_operand import GEMM_LAYOUTS
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
    # LDS-plan inputs (what the body really allocates; see
    # :func:`rocke.instances.common.gemm_universal.universal_lds_bytes`).
    ab_double_buffer: bool | None = None
    cshuffle_no_alias: bool = False
    lds_k_pad: int = 0
    interwave_setprio: bool = False
    # alias-scoped DTL ring stages (LDS = stages x AB; 0 = the ping-pong)
    dtl_stages: int = 0
    # register-staged prefetch depth (0 = off); such bodies are bounded by
    # their real LDS allocation only
    vgpr_prefetch: int = 0


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
        ab_double_buffer=tr.ab_double_buffer,
        cshuffle_no_alias=tr.cshuffle_no_alias,
        lds_k_pad=tr.lds_k_pad,
        interwave_setprio=tr.interwave_setprio,
        dtl_stages=tr.dtl_stages,
        vgpr_prefetch=int(getattr(tr, "vgpr_prefetch", 0) or 0),
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


def _dtype_bytes(dtype: str) -> int:
    d = normalize_dtype(dtype)
    try:
        return _DTYPE_BYTES[d]
    except KeyError as exc:
        raise ValueError(f"unknown GEMM dtype {dtype!r} for LDS sizing") from exc


# Per-pipeline LDS budget, mirrored from the canonical codegen validator
# (``dispatcher/codegen/arch_filter.py`` ``LDS_CAPACITY_LIMITS``): one A + B
# tile against a fixed per-pipeline budget. It still bounds the non-DTL bodies
# (on top of the real allocation checked below), so their sweep space is the
# one the CK-mirrored rule admits; the direct-to-LDS and register-staged
# (``vgpr_prefetch``) bodies are bounded by the target's real capacity only (a 256x256 DTL tile needs more than these caps).
_LDS_CAPACITY_LIMITS = {"compv4": 32768, "preshufflev2": 32768, "default": 65536}


def _ck_lds_estimate(q: GemmSupportQuery) -> int:
    tm, tn, tk = q.cta_tile
    return tm * tk * _dtype_bytes(q.dtype_a) + tn * tk * _dtype_bytes(q.dtype_b)


def _lds_bytes(q: GemmSupportQuery, lds_cap: int) -> Tuple[int, int, int]:
    """LDS bytes the universal body allocates for ``q``: ``(total, ab, c)``.

    Single source with ``is_valid_spec`` (:func:`universal_lds_bytes`): the AB
    staging doubled when ping-ponged (``dtl_prefetch`` / compv4 double buffer),
    plus the cshuffle C tile aliased onto it (``max``) unless
    ``cshuffle_no_alias``. CK's codegen-validator estimate (one A + B tile
    against a fixed per-pipeline budget, :data:`_LDS_CAPACITY_LIMITS`)
    under-counts the ping-pong, so the real allocation is always checked too.
    Operands and C are sized by their dtype (1-, 2- and 4-byte queries are
    sized, not rejected).
    """
    tm, tn, tk = q.cta_tile
    return universal_lds_bytes(
        tm,
        tn,
        tk,
        pipeline=q.pipeline,
        epilogue=q.epilogue,
        direct_to_lds=q.direct_to_lds,
        dtl_prefetch=q.dtl_prefetch,
        ab_double_buffer=q.ab_double_buffer,
        cshuffle_no_alias=q.cshuffle_no_alias,
        lds_k_pad=q.lds_k_pad,
        lds_cap=lds_cap,
        elem_bytes_a=_dtype_bytes(q.dtype_a),
        elem_bytes_b=_dtype_bytes(q.dtype_b),
        elem_bytes_c=_dtype_bytes(q.dtype_c),
        dtl_stages=q.dtl_stages,
    )


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

    if q.layout.upper() not in GEMM_LAYOUTS:
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

    # ``persistent`` is accepted (a no-op rename on the universal body, see
    # ``gemm_universal``); ``scheduler`` must be a known value; the opt-in
    # MFMA-cluster priority bookends exist only on the MFMA body.
    if q.scheduler not in ("intrawave", "interwave"):
        return False, f"unknown scheduler {q.scheduler!r}"
    if q.interwave_setprio and family != "mma":
        return False, f"interwave_setprio is MFMA-only on {q.arch}"

    # LDS budget: the bytes the body really allocates against the target's
    # per-WG capacity; non-DTL bodies also keep the CK-mirrored per-pipeline
    # estimate (see _LDS_CAPACITY_LIMITS).
    lds_cap = target.lds_capacity_bytes
    if not q.direct_to_lds and not q.vgpr_prefetch:
        ck_cap = min(
            _LDS_CAPACITY_LIMITS.get(q.pipeline, _LDS_CAPACITY_LIMITS["default"]),
            lds_cap,
        )
        ck_used = _ck_lds_estimate(q)
        if ck_used > ck_cap:
            return False, f"LDS budget {ck_used} > {ck_cap} cap on {q.arch}"
    try:
        lds_used, ab_bytes, c_bytes = _lds_bytes(q, lds_cap)
    except ValueError as e:
        return False, str(e)
    if lds_used > lds_cap:
        return False, (
            f"LDS budget {lds_used} > {lds_cap} cap (AB={ab_bytes}, C={c_bytes}) "
            f"on {q.arch}"
        )

    if tm * tk < q.block_size or tn * tk < q.block_size:
        return False, "block too small for one element/thread/phase"

    return True, "ok"


def universal_spec_supported(spec: UniversalGemmSpec, *, arch: str) -> Tuple[bool, str]:
    """Dispatcher support for one universal-GEMM spec on ``arch``.

    The hardware/tile predicate (:func:`gemm_config_supported`) followed by
    the builder's own validity gate (:func:`is_valid_spec`), so a candidate is
    never admitted for a spec the builder would refuse (or that the gate knows
    to be wrong, e.g. the DTL full-pass and swizzle write-granularity rules).
    One source of truth: the gate's rules are not re-implemented here.
    """
    ok, why = gemm_config_supported(support_query_from_universal_spec(spec, arch=arch))
    if not ok:
        return ok, why
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        return False, f"is_valid_spec: {why}"
    return True, "ok"


# Offsets the universal body computes in i32 (element GEP index for the
# global_load / global_store paths, byte offset for the DTL buffer loads). The
# largest element offset of an ``[R, K]`` operand is ``R * K - 1``, so up to
# ``2**31`` elements are addressable.
_I32_MAX_ELEMS = 1 << 31


def operand_range_supported(
    req: GemmRequest, spec: UniversalGemmSpec, *, elem_bytes: int = 2
) -> Tuple[bool, str]:
    """Whether the body's 32-bit address arithmetic covers ``req``'s operands.

    * Every path indexes A ``[M, K]``, B ``[N, K]`` and C ``[M, N]`` with an
      i32 element offset, so each needs at most ``2**31`` elements.
    * The direct-to-LDS (DTL) load path addresses A and B through a buffer
      resource with ``num_records = DTL_BUFFER_NUM_RECORDS`` (just under
      2 GiB) and an i32 BYTE offset. A load past ``num_records`` returns 0
      with no fault, so an operand of 2 GiB or more would compute silently
      wrong results (e.g. ``M=65536, K=16384`` f16). DTL candidates therefore
      require ``rows * K * elem_bytes <= num_records`` for A and B.
    """
    M, N, K = int(req.M), int(req.N), int(req.K)
    for nm, elems in (("A", M * K), ("B", N * K), ("C", M * N)):
        if elems > _I32_MAX_ELEMS:
            return False, (
                f"operand {nm} has {elems} elements; the universal body's i32 "
                f"element offsets cover at most 2**31"
            )
    if spec.trait.direct_to_lds:
        for nm, rows in (("A", M), ("B", N)):
            nbytes = rows * K * elem_bytes
            if nbytes > DTL_BUFFER_NUM_RECORDS:
                return False, (
                    f"direct_to_lds: operand {nm} is {nbytes} bytes > the DTL "
                    f"buffer range {DTL_BUFFER_NUM_RECORDS:#x} (loads past it "
                    "return 0 silently)"
                )
    return True, "ok"


def request_shape_supported(
    req: GemmRequest, spec: UniversalGemmSpec
) -> Tuple[bool, str]:
    """Return whether ``req.M/N/K`` satisfy the spec's padding/granularity and
    the body's 32-bit address range (:func:`operand_range_supported`)."""
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
    # the body's own launch rules (ragged-row bounds, the DTL K tail, split-K
    # slices, grid limits): one source with ``check_launch_shape``
    err = launch_shape_error(spec, int(req.M), int(req.N), int(req.K))
    if err is not None:
        return False, err
    # the VGPR-staged loader reads the padded rows of a
    # K-contiguous RCR operand past the operand unless kc_row_guard is set
    # (non-RCR layouts always guard); pad_row_clamp and
    # vgpr_prefetch (tight buffer range) bound those rows too
    if (
        not tr.direct_to_lds
        and spec.data.layout.upper() == "RCR"
        and not tr.kc_row_guard
        and not getattr(tr, "pad_row_clamp", False)
        and not getattr(tr, "vgpr_prefetch", 0)
    ):
        for name, dim, tile in (("M", req.M, t.tile_m), ("N", req.N, t.tile_n)):
            if dim % tile:
                return False, (
                    f"{name}={dim} is not divisible by tile_{name.lower()}={tile}: the "
                    "unguarded VGPR-staged loader would read past the operand "
                    "(kc_row_guard)"
                )
    return operand_range_supported(
        req, spec, elem_bytes=_dtype_bytes(spec.data.dtype_a)
    )
