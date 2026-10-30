# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX (microscaling) GEMM kernel (CK Tile ``42_mx_gemm`` parity).

The OCP MX format: every 32 consecutive K elements of a row of A (a column of
B) share one 8-bit E8M0 scale, an unsigned exponent biased by 127::

 A_real[m, k] = A[m, k] * 2^(AScale[k // 32, m] - 127)
 B_real[k, n] = B[k, n] * 2^(BScale[k // 32, n] - 127)
 C[m, n] = sum_k A_real[m, k] * B_real[k, n] (f32)

Scale code 255 is NaN. Both scale tensors are K-major: ``AScale`` is
``(K / 32, M)`` and ``BScale`` ``(K / 32, N)``, so the codes of one scale group
for consecutive rows (columns) are consecutive bytes. The mantissas are fp8
(E4M3) or bf8 (E5M2) in the encoding the target's matrix cores read (FNUZ on
gfx942, OCP on gfx950).

Numerics: every output is within the runbook GEMM tolerance of the exact
result, with IEEE special values (a NaN scale code or a NaN input element gives
NaN). An output is finite as long as every scaled group term and every running
sum of them, in K order, stays within the f32 range: the scale factors never
overflow on their own. Once one of those overflows, the output may be Inf or
NaN, as in any GEMM that accumulates in f32. ``dsl_docs/instances/gemm.md``
(MX GEMM) states the contract and why each step keeps it.

Work split: one wave computes a 32x32 output tile; a workgroup is
``block_tile_m / 32 x block_tile_n / 32`` waves side by side: the waves of one
tile row read the same A rows, and those of one tile column the same B rows, at
about the same time, so that they can share those reads in the vector L1 cache.
Operands are buffer loads: each lane's offset stays in a
register for the whole K loop and the K step is a scalar offset.

How the scales are applied depends on the target:

* A target with the E8M0-scaled MFMA (gfx950) applies them in the
  instruction: one ``32x32x64`` scaled MFMA per two scale groups, each lane
  passing the scale byte of the row (column) and group it loads. That
  instruction scales every product before it sums them, so with very large
  scale codes a product can overflow although the sum would not. Each lane
  tracks the largest code it passed; if any lane of the wave saw one above
  :func:`rocke.helpers.mx_scale.largest_safe_mx_scale_code`, the wave computes
  its tile again the second way and stores that instead. A K that is not a
  multiple of 64 leaves one group, which is added the second way.
* Any target (gfx942): per scale group two ``32x32x16`` fp8 / bf8 MFMAs give
  the group's unscaled dot product, and
  :func:`rocke.helpers.mx_scale.add_mx_scaled_dot` adds it to the accumulator
  times ``2^(ea + eb)`` with the A scale of each output's row and the B scale
  of its column. The group is added one group late, so the scale arithmetic
  never waits on a load issued in the same group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple

from ...core.arch import ArchTarget, MmaOp
from ...core.arch.mfma_scale import matrix_layout
from ...core.ir import (
    F32,
    I8,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
    VectorType,
)
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.mx_scale import (
    BF8_E5M2_FNUZ,
    BF8_E5M2_OCP,
    FP8_E4M3_FNUZ,
    FP8_E4M3_OCP,
    MxElementFormat,
    add_mx_scaled_dot,
    largest_safe_mx_scale_code,
)
from ...helpers.quant import quant_ir_type
from ...helpers.spec import SignatureBuilder, kernel_name_join

# sched_barrier mask: every instruction class may be scheduled across the
# barrier except VMEM (0x10) and VMEM read (0x20).
_SCALE_LOADS_FIRST = 0x7CF
# The output tile of one wave, and of its MFMAs.
_WAVE_TILE = 32
_WAVE_SIZE = 64
# K elements per scaled MFMA (two scale groups).
_SCALED_MFMA_K = 64
# A scale codes per load: four consecutive rows of one group (one dword).
_A_SCALE_ROWS_PER_LOAD = 4
_I32_MAX = 0x7FFFFFFF

MxMantissaDType = Literal["fp8e4m3", "bf8e5m2"]

# The fp8 / bf8 encoding the MFMA of each target reads: the FNUZ encodings on
# gfx942, the OCP ones on gfx950.
_MFMA_ELEMENT_FORMATS = {
    "gfx942": {"fp8e4m3": FP8_E4M3_FNUZ, "bf8e5m2": BF8_E5M2_FNUZ},
    "gfx950": {"fp8e4m3": FP8_E4M3_OCP, "bf8e5m2": BF8_E5M2_OCP},
}


def mx_gemm_element_format(mantissa_dtype: str, arch: str) -> MxElementFormat:
    """The element format ``arch``'s MFMA reads for ``mantissa_dtype``: the
    FNUZ encodings on gfx942, the OCP ones on gfx950. Its largest value bounds
    the products the scaled MFMA sums (:func:`largest_safe_mx_scale_code`)."""
    formats = _MFMA_ELEMENT_FORMATS.get(arch, {})
    if mantissa_dtype not in formats:
        raise ValueError(
            f"mx_gemm: no {mantissa_dtype} element format known for {arch}; "
            f"the MX GEMM reads fp8e4m3 / bf8e5m2 on gfx942 and gfx950"
        )
    return formats[mantissa_dtype]


@dataclass(frozen=True, kw_only=True)
class MxGemmSpec:
    """One concrete MX GEMM kernel configuration.

    ``block_tile_m`` x ``block_tile_n`` is the output tile of one workgroup,
    both multiples of 32: one wave per 32x32 tile inside it, at most the
    target's workgroup size in threads. It has no default: which tile is
    fastest depends on the shape and the target, so the caller chooses it
    (a tuning knob). M and N need not be multiples of the tile; waves past the
    edge of C do nothing and stores past it are dropped. M must be a multiple
    of 4 (the A scale codes of four consecutive rows are read as one dword).
    Every field is keyword-only, so a positional argument cannot land in the
    workgroup tile.
    """

    M: int
    N: int
    K: int
    block_tile_m: int
    block_tile_n: int
    mantissa_dtype: MxMantissaDType = "fp8e4m3"
    group_k: int = 32  # MX spec: shared-exponent block size
    name: str = "rocke_mx_gemm"

    @property
    def group_atom(self) -> MfmaAtom:
        """The fp8 / bf8 ``32x32x16`` MFMA that sums a scale group's unscaled
        products, two per group."""
        return (
            MfmaAtom.fp8_32x32x16()
            if self.mantissa_dtype == "fp8e4m3"
            else MfmaAtom.bf8_32x32x16()
        )

    @property
    def waves(self) -> int:
        """Waves per workgroup, one per 32x32 tile of the block tile."""
        return (self.block_tile_m // _WAVE_TILE) * (self.block_tile_n // _WAVE_TILE)

    @property
    def block_size(self) -> int:
        return _WAVE_SIZE * self.waves

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.mantissa_dtype,
            f"gk{self.group_k}",
            f"t{self.block_tile_m}x{self.block_tile_n}",
        )


def _scaled_mfma(spec: MxGemmSpec, arch: str) -> MmaOp | None:
    """The target's E8M0-scaled ``32x32x64`` MFMA for the spec's mantissas
    (a catalog :class:`MmaOp`), or None when the target has none (gfx942)."""
    return ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="mfma_scaled",
        a_dtype=spec.mantissa_dtype,
        b_dtype=spec.mantissa_dtype,
        c_dtype="fp32",
        scales=("e8m0", "e8m0", spec.group_k),
        m=_WAVE_TILE,
        n=_WAVE_TILE,
        k=_SCALED_MFMA_K,
    )


def is_valid_spec(spec: MxGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for an MX-GEMM spec on ``arch``.

    The fp8 / bf8 ``32x32x16`` MFMA ships on every CDNA target with the fp8
    MFMA family (gfx942 / gfx950); the fp4 / fp6 mantissa families are not
    supported yet.
    """
    for name in ("block_tile_m", "block_tile_n"):
        tile = getattr(spec, name)
        if tile <= 0 or tile % _WAVE_TILE:
            return (
                False,
                f"{name} must be a positive multiple of {_WAVE_TILE}, got {tile}",
            )
    ok, reason, _target = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        return False, reason
    if spec.mantissa_dtype not in ("fp8e4m3", "bf8e5m2"):
        return False, (
            f"unsupported mantissa_dtype {spec.mantissa_dtype!r}; "
            "the MX GEMM reads fp8e4m3 / bf8e5m2 (fp4 / fp6 are not supported yet)"
        )
    if spec.group_k != 32:
        return False, (
            f"MX spec requires group_k = 32 (shared exponent per 32 "
            f"mantissa elements); got group_k={spec.group_k}"
        )
    if spec.M <= 0 or spec.N <= 0 or spec.K <= 0:
        return False, f"M, N and K must be positive, got {spec.M}, {spec.N}, {spec.K}"
    if spec.K % spec.group_k:
        return False, f"K ({spec.K}) must be divisible by group_k ({spec.group_k})"
    if spec.M % _A_SCALE_ROWS_PER_LOAD:
        return False, (
            f"M ({spec.M}) must be a multiple of {_A_SCALE_ROWS_PER_LOAD}: a lane reads "
            f"the A scale codes of {_A_SCALE_ROWS_PER_LOAD} consecutive rows as one dword"
        )
    for tensor, offset in _largest_byte_offsets(spec).items():
        if offset > _I32_MAX:
            return False, (
                f"the largest byte offset into {tensor} is {offset}, expected at most "
                f"{_I32_MAX}: the kernel computes every buffer offset in i32"
            )
    return True, "ok"


def _largest_byte_offsets(spec: MxGemmSpec) -> dict[str, int]:
    """The largest byte offset the kernel computes into each tensor.

    A wave whose tile starts inside C computes the offsets of its whole 32x32
    tile, so the rows and columns past the edge count too, up to the next
    multiple of 32 (their loads and stores land past the end of the buffer and
    are dropped). The AScale bound covers the last dword of A codes a lane
    reads.
    """
    groups = spec.K // spec.group_k
    padded_rows = -(-spec.M // _WAVE_TILE) * _WAVE_TILE
    padded_cols = -(-spec.N // _WAVE_TILE) * _WAVE_TILE
    return {
        "A": padded_rows * spec.K - 1,
        "B": (spec.K - 1) * spec.N + padded_cols - 1,
        "AScale": (groups - 1) * spec.M + padded_rows - 1,
        "BScale": (groups - 1) * spec.N + padded_cols - 1,
        "C": 4 * ((padded_rows - 1) * spec.N + padded_cols) - 1,
    }


class _WaveOrigin(NamedTuple):
    """A lane's index in its wave and the first row and column of the wave's
    32x32 output tile."""

    lane: Value
    row_origin: Value
    col_origin: Value


@dataclass(frozen=True)
class _WaveTile:
    """What every part of the kernel reads to compute one wave's output tile:
    its origin (:class:`_WaveOrigin`), the IR type of the mantissas and the
    buffer descriptors of the five tensors."""

    lane: Value
    row_origin: Value
    col_origin: Value
    mantissa_type: Type
    a_rsrc: Value
    b_rsrc: Value
    a_scale_rsrc: Value
    b_scale_rsrc: Value
    c_rsrc: Value


def _wave_origin(b: IRBuilder, spec: MxGemmSpec) -> _WaveOrigin:
    """The wave's 32x32 output tile origin and its lane.

    Waves are numbered row-major inside the workgroup. The wave index is the
    same in every lane, so it is read once (readfirstlane) and the tile origin
    is a scalar.
    """
    waves_n = spec.block_tile_n // _WAVE_TILE
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(_WAVE_SIZE))
    wave = b.readfirstlane(b.div(tid, b.const_i32(_WAVE_SIZE)))
    row_origin = b.add(
        b.mul(b.block_id_y(), b.const_i32(spec.block_tile_m)),
        b.mul(b.div(wave, b.const_i32(waves_n)), b.const_i32(_WAVE_TILE)),
    )
    col_origin = b.add(
        b.mul(b.block_id_x(), b.const_i32(spec.block_tile_n)),
        b.mul(b.mod(wave, b.const_i32(waves_n)), b.const_i32(_WAVE_TILE)),
    )
    return _WaveOrigin(lane, row_origin, col_origin)


def _byte(
    b: IRBuilder, rsrc: Value, voffset: Value, soffset: Value, dtype: Type
) -> Value:
    """One 1-byte element through a buffer descriptor."""
    return b.vec_extract(b.buffer_load_vN(rsrc, voffset, soffset, dtype, 1), 0)


def _sum_scaled_groups(
    b: IRBuilder,
    spec: MxGemmSpec,
    tile: _WaveTile,
    acc: Value,
    first_group: int,
    end_group: int,
    tag: str,
) -> Value:
    """``acc`` plus the scaled dot products of scale groups
    ``first_group .. end_group - 1``. ``tag`` names the loop's values (one
    kernel can hold this loop twice).

    Per group: two ``32x32x16`` MFMAs give the unscaled dot product of each of
    the lane's 16 outputs, and :func:`add_mx_scaled_dot` adds it times
    ``2^(ea + eb)``, ``ea`` from the A scale of the output's row and ``eb`` from
    the B scale of the lane's column. A group is added one iteration late: the
    first iteration adds a zero dot product with scale codes 0, a term of +0
    (``ldexp(+0, -254)``), and the last group is added after the loop.
    """
    atom = spec.group_atom
    lanes = decode_mfma_lanes(b, atom, tile.lane)
    a_row = b.add(tile.row_origin, lanes.m_in_atom)
    b_col = b.add(tile.col_origin, lanes.n_in_atom)
    k_lane = b.mul(lanes.k_blk, b.const_i32(atom.a_per_lane))
    a_offset = b.add(b.mul(a_row, b.const_i32(spec.K)), k_lane)
    b_offsets = [
        b.add(b.mul(b.add(k_lane, b.const_i32(j)), b.const_i32(spec.N)), b_col)
        for j in range(atom.b_per_lane)
    ]
    # The A scale codes of the lane's outputs, one 4-byte load per 4 outputs:
    # in the 32x32 accumulator, slots 4q .. 4q + 3 of a lane are four
    # consecutive rows of one column (MfmaAtom.lane_to_output, the rows the
    # store uses), and AScale holds a group's codes for consecutive rows in
    # consecutive bytes.
    a_scale_offsets = [
        b.add(tile.row_origin, atom.lane_to_output(b, tile.lane, q)[0])
        for q in range(0, atom.c_per_lane, _A_SCALE_ROWS_PER_LOAD)
    ]
    atoms_per_group = spec.group_k // atom.k
    zero_code = b.trunc(b.const_i32(0), I8)

    loop = b.scf_for_iter(
        b.const_i32(first_group),
        b.const_i32(end_group),
        b.const_i32(1),
        [
            (f"{tag}_acc", acc),
            (f"{tag}_prev_dot", atom.zero_acc(b)),
            *((f"{tag}_prev_a_code{i}", zero_code) for i in range(atom.c_per_lane)),
            (f"{tag}_prev_b_code", zero_code),
        ],
        iv_name=f"{tag}_group",
    )
    with loop as (group, (acc_in, prev_dot, *prev_codes)):
        group_codes = b.mul(group, b.const_i32(spec.M))
        a_codes = []
        for offset in a_scale_offsets:
            rows = b.buffer_load_vN(
                tile.a_scale_rsrc, offset, group_codes, I8, _A_SCALE_ROWS_PER_LOAD
            )
            a_codes += [b.vec_extract(rows, r) for r in range(_A_SCALE_ROWS_PER_LOAD)]
        b_code = _byte(
            b, tile.b_scale_rsrc, b_col, b.mul(group, b.const_i32(spec.N)), I8
        )
        # Issue the scale-byte loads before the operand loads: no global load
        # may be scheduled across this point (every other instruction may).
        b.sched_barrier(_SCALE_LOADS_FIRST)
        operands = []
        for atom_in_group in range(atoms_per_group):
            k_tile = b.add(
                b.mul(group, b.const_i32(spec.group_k)),
                b.const_i32(atom_in_group * atom.k),
            )
            a_operand = b.buffer_load_vN(
                tile.a_rsrc, a_offset, k_tile, tile.mantissa_type, atom.a_per_lane
            )
            row_step = b.mul(k_tile, b.const_i32(spec.N))
            b_operand = b.vec_pack(
                [
                    _byte(b, tile.b_rsrc, offset, row_step, tile.mantissa_type)
                    for offset in b_offsets
                ],
                tile.mantissa_type,
            )
            operands.append((a_operand, b_operand))
        acc_out = add_mx_scaled_dot(
            b, acc_in, prev_dot, prev_codes[:-1], [prev_codes[-1]] * atom.c_per_lane
        )
        dot = atom.zero_acc(b)
        for a_operand, b_operand in operands:
            dot = atom.emit(b, a_operand, b_operand, dot)
        b.scf_yield(acc_out, dot, *a_codes, b_code)
    acc_last, last_dot, *last_codes = loop.results
    return add_mx_scaled_dot(
        b, acc_last, last_dot, last_codes[:-1], [last_codes[-1]] * atom.c_per_lane
    )


def _scaled_mfma_steps(
    b: IRBuilder, spec: MxGemmSpec, tile: _WaveTile, mma: MmaOp
) -> tuple[Value, Value]:
    """The accumulator after ``K // 64`` scaled MFMAs, and the largest scale
    code each lane passed to them.

    The scaled MFMA reads, per lane, the A row ``lane % 32`` and the B column
    ``lane % 32``, K chunks interleaved between the two lane halves
    (:func:`rocke.core.arch.mfma_scale.matrix_layout`), and one scale byte
    each: the A scale of its row and the B scale of its column for group
    ``lane // 32`` of the step.
    """
    layout = matrix_layout(spec.mantissa_dtype)
    chunk = layout.chunk_elements
    chunk_stride = chunk * layout.lane_groups  # K distance between a lane's chunks
    row = b.mod(tile.lane, b.const_i32(layout.lanes_per_group))
    lane_group = b.div(tile.lane, b.const_i32(layout.lanes_per_group))
    a_row = b.add(tile.row_origin, row)
    b_col = b.add(tile.col_origin, row)
    k_lane = b.mul(lane_group, b.const_i32(chunk))
    a_offset = b.add(b.mul(a_row, b.const_i32(spec.K)), k_lane)
    b_offsets = [
        b.add(b.mul(b.add(k_lane, b.const_i32(j)), b.const_i32(spec.N)), b_col)
        for j in range(chunk)
    ]
    a_scale_offset = b.add(b.mul(lane_group, b.const_i32(spec.M)), a_row)
    b_scale_offset = b.add(b.mul(lane_group, b.const_i32(spec.N)), b_col)
    groups_per_step = _SCALED_MFMA_K // spec.group_k
    words = VectorType(I32, 8)
    zero = b.const_i32(0)
    loop = b.scf_for_iter(
        zero,
        b.const_i32(spec.K // _SCALED_MFMA_K),
        b.const_i32(1),
        [("scaled_acc", b.zero_vec_f32(mma.c_frag_len)), ("largest_code", zero)],
        iv_name="step",
    )
    with loop as (step, (acc, largest_code)):
        k_step = b.mul(step, b.const_i32(_SCALED_MFMA_K))
        group_step = b.mul(step, b.const_i32(groups_per_step))
        a_scale = b.zext(
            _byte(
                b,
                tile.a_scale_rsrc,
                a_scale_offset,
                b.mul(group_step, b.const_i32(spec.M)),
                I8,
            ),
            I32,
        )
        b_scale = b.zext(
            _byte(
                b,
                tile.b_scale_rsrc,
                b_scale_offset,
                b.mul(group_step, b.const_i32(spec.N)),
                I8,
            ),
            I32,
        )
        a_chunks, b_elements = [], []
        for c in range(layout.chunks_per_lane):
            k_chunk = b.add(k_step, b.const_i32(c * chunk_stride))
            a_chunks.append(
                b.buffer_load_vN(
                    tile.a_rsrc, a_offset, k_chunk, tile.mantissa_type, chunk
                )
            )
            row_step = b.mul(k_chunk, b.const_i32(spec.N))
            b_elements += [
                _byte(b, tile.b_rsrc, offset, row_step, tile.mantissa_type)
                for offset in b_offsets
            ]
        a_operand = b.bitcast(b.vec_concat(*a_chunks), words)
        b_operand = b.bitcast(b.vec_pack(b_elements, tile.mantissa_type), words)
        largest_code = b.smax(largest_code, b.smax(a_scale, b_scale))
        b.scf_yield(
            b.mma(mma, a_operand, b_operand, acc, a_scale, b_scale), largest_code
        )
    return loop.results


def _store_tile(b: IRBuilder, spec: MxGemmSpec, tile: _WaveTile, acc: Value) -> None:
    """Store the lane's 16 outputs (the rows and column of
    :meth:`MfmaAtom.lane_to_output`) through a buffer descriptor over C: a row
    past M lands past the end of C and is dropped; when N is not a multiple of
    32, a column past N is moved past the end too."""
    atom = spec.group_atom
    past_end = b.const_i32(4 * spec.M * spec.N) if spec.N % _WAVE_TILE else None
    zero = b.const_i32(0)
    for i in range(atom.c_per_lane):
        row_in_tile, col_in_tile = atom.lane_to_output(b, tile.lane, i)
        col = b.add(tile.col_origin, col_in_tile)
        element = b.add(
            b.mul(b.add(tile.row_origin, row_in_tile), b.const_i32(spec.N)), col
        )
        offset = b.mul(element, b.const_i32(4))
        if spec.N % _WAVE_TILE:
            offset = b.select(b.cmp_lt(col, b.const_i32(spec.N)), offset, past_end)
        b.buffer_store_vN(
            tile.c_rsrc, offset, zero, b.vec_pack([b.vec_extract(acc, i)], F32)
        )


def build_mx_gemm(spec: MxGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one MX GEMM instance.

    Kernel signature::

    (A: ptr<fp8 | bf8>, # (M, K) row-major mantissas
    AScale: ptr<i8>, # (K / 32, M) row-major E8M0 codes
    B: ptr<fp8 | bf8>, # (K, N) row-major mantissas
    BScale: ptr<i8>, # (K / 32, N) row-major E8M0 codes
    C: ptr<f32>, # (M, N) row-major f32 output
    M: i32, N: i32, K: i32)

    Grid: :func:`mx_gemm_grid`; block: ``spec.block_size`` threads.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid mx_gemm spec for {arch}: {why}")
    validate_mfma_atom_in_catalog(spec.group_atom, arch, where="mx_gemm")
    mma = _scaled_mfma(spec, arch)

    mantissa_type = quant_ir_type(spec.mantissa_dtype)
    group_count = spec.K // spec.group_k
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    a_param = b.param("A", PtrType(mantissa_type, "global"), readonly=True, align=16)
    a_scale_param = b.param("AScale", PtrType(I8, "global"), readonly=True, align=1)
    b_param = b.param("B", PtrType(mantissa_type, "global"), readonly=True, align=16)
    b_scale_param = b.param("BScale", PtrType(I8, "global"), readonly=True, align=1)
    c_param = b.param("C", PtrType(F32, "global"), writeonly=True, align=4)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)

    origin = _wave_origin(b, spec)

    def tile_body() -> None:
        tile = _WaveTile(
            *origin,
            mantissa_type=mantissa_type,
            a_rsrc=b.buffer_rsrc(a_param, b.const_i32(spec.M * spec.K)),
            b_rsrc=b.buffer_rsrc(b_param, b.const_i32(spec.K * spec.N)),
            a_scale_rsrc=b.buffer_rsrc(
                a_scale_param, b.const_i32(group_count * spec.M)
            ),
            b_scale_rsrc=b.buffer_rsrc(
                b_scale_param, b.const_i32(group_count * spec.N)
            ),
            c_rsrc=b.buffer_rsrc(c_param, b.const_i32(4 * spec.M * spec.N)),
        )
        zero_acc = spec.group_atom.zero_acc(b)
        if mma is None:
            _store_tile(
                b,
                spec,
                tile,
                _sum_scaled_groups(b, spec, tile, zero_acc, 0, group_count, "all"),
            )
            return
        acc, largest_code = _scaled_mfma_steps(b, spec, tile, mma)
        scaled_groups = (spec.K // _SCALED_MFMA_K) * (_SCALED_MFMA_K // spec.group_k)
        if scaled_groups < group_count:
            acc = _sum_scaled_groups(
                b, spec, tile, acc, scaled_groups, group_count, "tail"
            )
        _store_tile(b, spec, tile, acc)
        # Codes above the safe bound: compute the tile again, scaling each
        # group's dot product, and store that instead. This path is rare (a
        # wave saw a code above the bound), so the common path keeps one store
        # and no select; a lane's two stores to one address land in program
        # order.
        element = mx_gemm_element_format(spec.mantissa_dtype, arch)
        safe_code = largest_safe_mx_scale_code(element, element, _SCALED_MFMA_K)
        unsafe = b.wave_any(
            b.select(
                b.cmp_gt(largest_code, b.const_i32(safe_code)),
                b.const_i32(1),
                b.const_i32(0),
            )
        )
        with b.scf_if(b.cmp_ne(unsafe, b.const_i32(0))):
            _store_tile(
                b,
                spec,
                tile,
                _sum_scaled_groups(b, spec, tile, zero_acc, 0, group_count, "again"),
            )

    # A wave whose tile starts past the edge of C does nothing.
    if spec.M % spec.block_tile_m or spec.N % spec.block_tile_n:
        inside = b.land(
            b.cmp_lt(origin.row_origin, b.const_i32(spec.M)),
            b.cmp_lt(origin.col_origin, b.const_i32(spec.N)),
        )
        with b.scf_if(inside):
            tile_body()
    else:
        tile_body()
    b.ret()
    return b.kernel


def mx_gemm_grid(spec: MxGemmSpec) -> Tuple[int, int, int]:
    n_tiles = (spec.N + spec.block_tile_n - 1) // spec.block_tile_n
    m_tiles = (spec.M + spec.block_tile_m - 1) // spec.block_tile_m
    return (n_tiles, m_tiles, 1)


def mx_gemm_signature(spec: MxGemmSpec):
    return (
        SignatureBuilder()
        .ptr("A", spec.mantissa_dtype)
        .ptr("AScale", "i8")
        .ptr("B", spec.mantissa_dtype)
        .ptr("BScale", "i8")
        .ptr("C", "f32")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .build()
    )


__all__ = [
    "MxGemmSpec",
    "MxMantissaDType",
    "build_mx_gemm",
    "is_valid_spec",
    "mx_gemm_element_format",
    "mx_gemm_grid",
    "mx_gemm_signature",
]
