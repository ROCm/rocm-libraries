# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared surface for the ``MatMulNBits`` gfx1151 instance family.

DSL counterpart of ONNX Runtime's ``MatMulNBits`` (fp16 activations, packed
int4 weights, group-of-32 scales). The kernel computes::

    C[M, N] = A[M, K] @ dequant(B[N, K], scales[N, K / 32])^T

with ``A`` / ``C`` fp16 row-major, ``B`` packed two signed int4 per byte, and
one scale per ``(n, k // group_size)`` group.

The three planned specializations (``large_n`` / ``skinny_n`` / ``decode_gemv``;
see ``dsl_docs/architecture/matmul_nbits_plan.md``) share their operand layout,
validity rules, signature, grid, and the 64-row outer-loop driver. Those live
here so the per-family bodies only add what differs. The public dispatcher and
the per-family validity extras live in
:mod:`rocke.instances.common.matmul_nbits`.

This module is Milestone 1: spec, validator, signature, grid, and the
64-row outer-loop helper. No kernel body yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Tuple

from ...helpers.spec import (
    SignatureBuilder,
    WarpTileBlockSizeMixin,
    ceil_div_grid,
    derive_block_size,
    kernel_name_join,
)
from .gemm_universal import TileSpec

if TYPE_CHECKING:  # numpy is a host-test/reference dependency, not a build dep.
    import numpy as np

__all__ = [
    "FAMILIES",
    "MatMulNBitsFamily",
    "MatMulNBitsSpec",
    "SUPPORTED_ARCHES",
    "V1_ARCH",
    "V1_GROUP_SIZE",
    "dequant_i4_weights",
    "matmul_nbits_grid",
    "matmul_nbits_outer_tiles",
    "matmul_nbits_reference",
    "matmul_nbits_signature",
    "pack_i4_weights_for_matmul_nbits",
    "validate_common_spec",
]

MatMulNBitsFamily = Literal["large_n", "skinny_n", "decode_gemv"]
FAMILIES: Tuple[str, ...] = ("large_n", "skinny_n", "decode_gemv")

# v1 default arch is gfx1151; the gfx1201 (RDNA4) port shares the same body via
# arch-specific WMMA fragment params. Both are wave32 WMMA + group-of-32
# signed-symmetric int4 (see plan).
V1_ARCH = "gfx1151"
SUPPORTED_ARCHES = frozenset({"gfx1151", "gfx1201"})
V1_GROUP_SIZE = 32

# WMMA atom the dequant-to-fp16 path feeds (gfx1151 wave32). Resolved from the
# target MMA catalog so the validator stays target-neutral in structure.
_WMMA_C_DTYPE = "fp32"
_WMMA_AB_DTYPE = "fp16"


def _scale_wire_dtype(scale_dtype: str) -> str:
    """Canonicalise the scale storage dtype to a manifest wire dtype."""
    if scale_dtype in ("fp16", "f16"):
        return "f16"
    if scale_dtype in ("fp32", "f32"):
        return "f32"
    raise ValueError(f"scale_dtype must be fp16 / fp32, got {scale_dtype!r}")


@dataclass(frozen=True)
class MatMulNBitsSpec(WarpTileBlockSizeMixin):
    """One ``MatMulNBits`` specialization.

    ``N`` / ``K`` are compile-time specialization fields; ``M`` (= ``seq_len``)
    is a runtime kernel argument. ``tile`` reuses the GEMM-family
    :class:`~rocke.instances.common.gemm_universal.TileSpec` so naming, grid,
    occupancy, and block-size derivation follow the same path as
    ``UniversalGemmSpec`` and friends. ``block_size`` defaults to ``0`` and is
    derived from ``warp_m * warp_n * warp_k * wave_size`` by
    :class:`WarpTileBlockSizeMixin`.
    """

    name: str
    N: int
    K: int
    tile: TileSpec
    group_size: int = V1_GROUP_SIZE
    seq_len_tile: int = 64
    wave_size: int = 32
    block_size: int = 0
    scale_dtype: str = "fp16"
    zero_points: bool = False
    packing: str = "row_k_contiguous"
    family: str = "large_n"
    optimized: bool = False

    def __post_init__(self) -> None:
        self._init_block_size()

    def kernel_name(self) -> str:
        t = self.tile
        return kernel_name_join(
            self.name,
            self.family,
            _WMMA_AB_DTYPE,
            f"N{self.N}K{self.K}",
            f"g{self.group_size}",
            f"t{t.tile_m}x{t.tile_n}x{t.tile_k}",
            f"w{t.warp_m}x{t.warp_n}x{t.warp_k}",
            f"wt{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}",
            f"s{_scale_wire_dtype(self.scale_dtype)}",
            flags={"zp": self.zero_points},
        )


def validate_common_spec(
    spec: MatMulNBitsSpec, arch: str = V1_ARCH
) -> Tuple[bool, str]:
    """Family-agnostic validity gate for ``spec`` on ``arch``.

    Returns ``(ok, reason)``. Covers the v1 contract (gfx1151-only,
    group-of-32, signed-symmetric int4, simple row-K-contiguous packing), the
    WMMA warp-tile atom (resolved from the target MMA catalog), geometry
    divisibility, and the block-size / thread-cap budget. Per-family extra
    checks live in :func:`rocke.instances.common.matmul_nbits.is_valid_spec`.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    # v1 contract gates.
    if arch not in SUPPORTED_ARCHES:
        return False, (
            f"matmul_nbits v1 supports {sorted(SUPPORTED_ARCHES)} only (got {arch!r})"
        )
    if spec.family not in FAMILIES:
        return False, f"unknown family {spec.family!r}; expected one of {FAMILIES}"
    if spec.group_size != V1_GROUP_SIZE:
        return False, (
            f"matmul_nbits v1 fixes group_size={V1_GROUP_SIZE} (got {spec.group_size})"
        )
    if spec.zero_points:
        return False, (
            "matmul_nbits v1 is signed-symmetric int4 only; zero_points not "
            "yet supported"
        )
    if spec.packing != "row_k_contiguous":
        return False, (
            f"matmul_nbits v1 ships packing='row_k_contiguous' only "
            f"(got {spec.packing!r})"
        )
    try:
        _scale_wire_dtype(spec.scale_dtype)
    except ValueError as e:
        return False, str(e)

    # Problem-size sanity.
    if spec.N <= 0 or spec.K <= 0:
        return False, f"N / K must be positive (got N={spec.N}, K={spec.K})"
    if spec.K % spec.group_size:
        return False, (
            f"K ({spec.K}) must be divisible by group_size ({spec.group_size})"
        )

    # Wave geometry must match the target.
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    # The decode-GEMV family is a scalar (no-WMMA) body: one thread per output
    # column, M=1. It does not feed a WMMA atom and is not bound by warp-tile
    # geometry, so skip the atom lookup and tile-divisibility gates below; the
    # only structural requirement is its block_size budget (checked further on).
    if spec.family != "decode_gemv":
        # WMMA atom must exist in the target catalog for fp16 in / fp32 acc. The
        # dequantised int4 weights are fed as fp16 fragments, so the atom dtype
        # is fp16 regardless of the on-disk int4 storage.
        family = "wmma" if target.wave_size == 32 else "mma"
        t = spec.tile
        atom = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
        if not target.supports_dtype_combo(
            _WMMA_AB_DTYPE, _WMMA_AB_DTYPE, _WMMA_C_DTYPE, family=family
        ):
            return False, f"unsupported matmul_nbits dtype fp16 on {arch}"
        if (
            target.mma.op_for_shape(
                family=family,
                a_dtype=_WMMA_AB_DTYPE,
                b_dtype=_WMMA_AB_DTYPE,
                c_dtype=_WMMA_C_DTYPE,
                m=t.warp_tile_m,
                n=t.warp_tile_n,
                k=t.warp_tile_k,
            )
            is None
        ):
            return False, f"unsupported fp16 warp_tile {atom} on {arch}"

        # Geometry divisibility.
        if t.tile_m % (t.warp_m * t.warp_tile_m):
            return False, "tile_m not divisible by warp_m * warp_tile_m"
        if t.tile_n % (t.warp_n * t.warp_tile_n):
            return False, "tile_n not divisible by warp_n * warp_tile_n"
        if t.tile_k % t.warp_tile_k:
            return False, "tile_k not divisible by warp_tile_k"
        if spec.N % t.tile_n:
            return False, f"N ({spec.N}) must be divisible by tile_n ({t.tile_n})"
        if spec.K % t.tile_k:
            return False, f"K ({spec.K}) must be divisible by tile_k ({t.tile_k})"

    # block_size = warp_m * warp_n * warp_k * wave_size, matching the canonical
    # derivation in helpers/spec.py::derive_block_size (the source of the spec's
    # own block_size). Omitting warp_k here would falsely reject any warp_k != 1.
    expected_bs = derive_block_size(spec.tile, spec.wave_size)
    if expected_bs != spec.block_size:
        return False, (
            f"block_size {spec.block_size} != warp_m*warp_n*warp_k*wave_size = "
            f"{expected_bs}"
        )
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > {target.max_threads_per_block} "
            f"(hardware cap) on {arch}"
        )

    if spec.seq_len_tile <= 0:
        return False, f"seq_len_tile must be positive (got {spec.seq_len_tile})"

    return True, "ok"


def matmul_nbits_signature(spec: MatMulNBitsSpec) -> List[dict]:
    """Manifest signature: A (fp16), packed B (int4-in-i8), Scales, C (fp16),
    and the runtime ``M``. ``N`` / ``K`` are baked in as compile-time
    constants, so only ``M`` is passed at launch.
    """
    return (
        SignatureBuilder()
        .ptr("A", "f16")
        .ptr("B", "i8")  # two signed int4 packed per byte
        .ptr("Scales", _scale_wire_dtype(spec.scale_dtype))
        .ptr("C", "f16")
        .scalar("M", "i32")
        .build()
    )


def matmul_nbits_grid(M: int, spec: MatMulNBitsSpec) -> Tuple[int, int, int]:
    """Launch grid for a single hot-loop invocation over ``M`` rows:
    ``(ceil(N / tile_n), ceil(M / tile_m), 1)``.

    For dynamic ``seq_len`` the host drives :func:`matmul_nbits_outer_tiles`
    and launches once per 64-row outer tile (or passes the full ``M`` here when
    the body owns the outer loop).
    """
    t = spec.tile
    return ceil_div_grid((spec.N, t.tile_n), (M, t.tile_m))


def matmul_nbits_outer_tiles(
    seq_len: int, spec: MatMulNBitsSpec
) -> List[Tuple[int, int]]:
    """Split a dynamic ``seq_len`` into ``seq_len_tile``-row outer tiles.

    Returns a list of ``(m_outer, m_tile)`` pairs where ``m_tile`` is
    ``seq_len_tile`` for every full tile and the remainder for a final partial
    tile. In the common ``seq_len % seq_len_tile == 0`` case every tile is full
    (tail-free). Callers detect a tail via ``tiles[-1][1] != spec.seq_len_tile``.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative (got {seq_len})")
    st = spec.seq_len_tile
    return [(m, min(st, seq_len - m)) for m in range(0, seq_len, st)]


# ---------------------------------------------------------------------
# Host-side packer + reference (Milestone 2)
# ---------------------------------------------------------------------
#
# The v1 ``row_k_contiguous`` layout is deliberately the simplest thing that
# verifies (see ``dsl_docs/architecture/matmul_nbits_plan.md`` "Packing And
# Layout"):
#
#   * B is logical signed int4 ``[N, K]`` in ``[-8, 7]``;
#   * two values pack per byte, K contiguous inside each N row:
#       byte (n, j) low nibble  = B[n, 2*j]
#       byte (n, j) high nibble = B[n, 2*j + 1]
#     giving packed ``uint8 [N, K // 2]``;
#   * scales are one fp16/fp32 value per ``(n, k // group_size)`` group:
#     ``[N, K // group_size]``.
#
# These run on the host (numpy) for test/reference only; the build path never
# imports numpy. They mirror the in-kernel unpack convention in
# :func:`rocke.helpers.i4_dequant.unpack_i4_byte_to_pair_f32` (low nibble
# first, sign-extended to ``[-8, 7]``).


def pack_i4_weights_for_matmul_nbits(
    weights: "np.ndarray", spec: MatMulNBitsSpec
) -> "np.ndarray":
    """Pack a logical signed-int4 ``[N, K]`` weight matrix into the v1
    ``row_k_contiguous`` layout: ``uint8 [N, K // 2]``, two signed int4 per
    byte (low nibble = even ``k``, high nibble = odd ``k``).

    ``weights`` must be integer-valued in ``[-8, 7]`` with shape matching
    ``(spec.N, spec.K)``. Returns a C-contiguous ``uint8`` array.
    """
    import numpy as np

    w = np.asarray(weights)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2-D [N, K], got shape {w.shape}")
    n, k = w.shape
    if (n, k) != (spec.N, spec.K):
        raise ValueError(f"weights shape {(n, k)} != spec (N, K) {(spec.N, spec.K)}")
    if k % 2:
        raise ValueError(f"K ({k}) must be even to pack two int4 per byte")
    if not np.issubdtype(w.dtype, np.integer):
        if not np.all(w == np.round(w)):
            raise ValueError("weights must be integer-valued signed int4")
    wi = w.astype(np.int64)
    if wi.min() < -8 or wi.max() > 7:
        raise ValueError(
            f"weights out of signed-int4 range [-8, 7] (min={wi.min()}, max={wi.max()})"
        )
    low = wi[:, 0::2] & 0x0F
    high = wi[:, 1::2] & 0x0F
    return (low | (high << 4)).astype(np.uint8)


def dequant_i4_weights(
    packed: "np.ndarray", scales: "np.ndarray", spec: MatMulNBitsSpec
) -> "np.ndarray":
    """Reconstruct fp32 ``[N, K]`` weights from the packed ``uint8 [N, K // 2]``
    layout and per-group ``scales [N, K // group_size]``.

    Inverse of :func:`pack_i4_weights_for_matmul_nbits` followed by the
    forward per-group scale multiply. Nibbles are sign-extended to ``[-8, 7]``
    to match the in-kernel unpack.
    """
    import numpy as np

    p = np.asarray(packed, dtype=np.uint8)
    if p.shape != (spec.N, spec.K // 2):
        raise ValueError(f"packed shape {p.shape} != expected {(spec.N, spec.K // 2)}")
    g = spec.group_size
    s = np.asarray(scales, dtype=np.float32)
    if s.shape != (spec.N, spec.K // g):
        raise ValueError(f"scales shape {s.shape} != expected {(spec.N, spec.K // g)}")

    low = (p & 0x0F).astype(np.int32)
    high = ((p >> 4) & 0x0F).astype(np.int32)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    w = np.empty((spec.N, spec.K), dtype=np.float32)
    w[:, 0::2] = low
    w[:, 1::2] = high
    return w * np.repeat(s, g, axis=1)


def matmul_nbits_reference(
    a: "np.ndarray",
    packed: "np.ndarray",
    scales: "np.ndarray",
    spec: MatMulNBitsSpec,
) -> "np.ndarray":
    """Host reference: ``C[M, N] = A[M, K] @ dequant(B, scales)^T``.

    ``a`` is ``[M, K]`` (fp16/fp32); ``packed`` / ``scales`` are the v1 layout.
    The matmul accumulates in fp32 and returns fp32 ``[M, N]``; callers compare
    against the fp16 kernel output within tolerance.
    """
    import numpy as np

    af = np.asarray(a, dtype=np.float32)
    if af.ndim != 2 or af.shape[1] != spec.K:
        raise ValueError(f"A shape {af.shape} incompatible with K={spec.K}")
    bf = dequant_i4_weights(packed, scales, spec)
    return af @ bf.T
