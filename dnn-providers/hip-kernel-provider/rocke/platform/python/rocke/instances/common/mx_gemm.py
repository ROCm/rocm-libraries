# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX (microscaling) GEMM kernel (CK Tile ``42_mx_gemm`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/42_mx_gemm``. The MX
GEMM uses the OCP MX-spec shared-exponent format: each 32-element
mantissa block carries one 8-bit biased E8M0 scale (bias 127), so the per-
element math is::

 A_mantissa[i] = fp8_or_fp6_or_fp4 value
 A_scale[i // 32] = 8-bit unsigned biased exponent (E8M0)
 A_real[i] = A_mantissa[i] * 2^(A_scale[i // 32] - 127)

Same for B. The kernel does an A * B^T contraction in f32 (the MX
intrinsic family ``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4``
applies the scale in-instruction; v1 applies it explicitly after each
group's MFMA with :func:`rocke.helpers.mx_scale.fma_with_mx_scale_pair`
so the inner GEMM stays interpretable). Each group's update of an output is
the exact value rounded once to f32, bit for bit; the MX GEMM section of
``dsl_docs/instances/gemm.md`` says why.

What v1 ships:

* **fp8 mantissa** for both A and B (``mantissa_dtype="fp8e4m3"``);
 the fp6 and fp4 mantissa families are reachable through the same
 spec surface once the matching unpack helpers land in
 :mod:`rocke.helpers.i4_dequant` (the fp4 / fp6 unpackers are a
 natural extension of the i4 nibble-extract pattern).
* **Block size 32 for the shared exponent** (the MX-spec default);
 ``group_k`` is the only knob (default 32).
* **f32 output**; the caller can post-convert to bf16/f16 via
 :func:`rocke.helpers.io.store_scalar_from_f32` or a follow-on
 finalize kernel.

Scalar inner v1 (per output ``(m, n)``):

 acc = 0
 for k in 0..K:
 a_mantissa = load A[m, k]; b_mantissa = load B[k, n]
 a_scale = decode(A_scale[m, k // 32])
 b_scale = decode(B_scale[k // 32, n]) # transposed for column-major B
 a = cvt_<mantissa_dtype>_to_f32(a_mantissa) * a_scale
 b = cvt_<mantissa_dtype>_to_f32(b_mantissa) * b_scale
 acc += a * b
 C[m, n] = acc

The MFMA-based body ( follow-on) replaces the per-element loop
with one ``mfma_f32_16x16x32_fp8(...)`` call per MX block plus the
block-scale multiply post-MFMA, against the same spec + helper
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I8, I32, IRBuilder, KernelDef, PtrType
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    store_acc_to_global,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.mx_scale import (
    BF8_E5M2_FNUZ,
    BF8_E5M2_OCP,
    FP8_E4M3_FNUZ,
    FP8_E4M3_OCP,
    MxElementFormat,
    exact_mx_scale_split,
    fma_with_mx_scale_pair,
)
from ...helpers.quant import quant_ir_type
from ...helpers.spec import SignatureBuilder, kernel_name_join

# sched_barrier mask: every instruction class may be scheduled across the
# barrier except VMEM (0x10) and VMEM read (0x20).
_SCALE_LOADS_FIRST = 0x7CF
# The E8M0 scale code of 2^0 (the code is the exponent biased by 127).
_E8M0_CODE_FOR_ONE = 127


MxMantissaDType = Literal["fp8e4m3", "bf8e5m2"]

# The fp8 / bf8 encoding the MFMA of each target reads: the FNUZ encodings on
# gfx942, the OCP ones on gfx950.
_MFMA_ELEMENT_FORMATS = {
    "gfx942": {"fp8e4m3": FP8_E4M3_FNUZ, "bf8e5m2": BF8_E5M2_FNUZ},
    "gfx950": {"fp8e4m3": FP8_E4M3_OCP, "bf8e5m2": BF8_E5M2_OCP},
}


def mx_gemm_element_format(mantissa_dtype: str, arch: str) -> MxElementFormat:
    """The element format ``arch``'s MFMA reads for ``mantissa_dtype``.

    The MX GEMM derives its scale split from it
    (:func:`rocke.helpers.mx_scale.exact_mx_scale_split`), so the split
    depends on the target: gfx942 reads the FNUZ encodings, gfx950 the OCP
    ones.
    """
    formats = _MFMA_ELEMENT_FORMATS.get(arch, {})
    if mantissa_dtype not in formats:
        raise ValueError(
            f"mx_gemm: no {mantissa_dtype} element format known for {arch}; "
            f"the MX GEMM reads fp8e4m3 / bf8e5m2 on gfx942 and gfx950"
        )
    return formats[mantissa_dtype]


@dataclass(frozen=True)
class MxGemmSpec:
    """One concrete MX GEMM kernel configuration."""

    M: int
    N: int
    K: int
    mantissa_dtype: MxMantissaDType = "fp8e4m3"
    group_k: int = 32  # MX spec: shared-exponent block size
    block_tile_m: int = 16
    block_tile_n: int = 16
    name: str = "rocke_mx_gemm"
    # P77: row-aware scale correctness mode. The historical
    # ``per_input_row=True`` (default) loads ``AScale[m_in_atom +
    # m_tile_base, kg]`` per lane and applies it to the lane's
    # ``c_per_lane=4`` *output* cells. For ``k_blk > 0`` lanes
    # (16-63), ``m_in_atom`` does not equal the output-row index so
    # the scale doesn't correspond to any output cell — masked by
    # parity tests using uniform per-row scales but mathematically
    # wrong on real workloads. ``per_input_row=False`` switches to
    # per-output-row scale broadcast: lane-broadcast via
    # :func:`rocke.helpers.mx_scale.load_and_decode_mx_scales_wave`
    # so each lane sees the scale matching its actual output row.
    # Defaults to True for backwards compatibility; flip to False
    # on real per-row varying-scale workloads.
    per_input_row: bool = True

    @property
    def atom(self) -> MfmaAtom:
        """fp8 / bf8 MFMA atom selected from ``mantissa_dtype``."""
        if (self.block_tile_m, self.block_tile_n) != (16, 16):
            raise ValueError(
                f"mx_gemm MFMA path supports 16x16 tiles only "
                f"(got {self.block_tile_m}x{self.block_tile_n})"
            )
        return (
            MfmaAtom.fp8_16x16x32()
            if self.mantissa_dtype == "fp8e4m3"
            else MfmaAtom.bf8_16x16x32()
        )

    @property
    def block_size(self) -> int:
        # MFMA path: one wave64 warp per CTA.
        return 64

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.mantissa_dtype,
            f"gk{self.group_k}",
            f"t{self.block_tile_m}x{self.block_tile_n}",
        )


def is_valid_spec(spec: MxGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for an MX-GEMM spec on ``arch``.

    Architecture facts (LDS capacity, max threads/block) are sourced from
    :class:`rocke.core.arch.ArchTarget` so the resource caps track the
    target instead of being hardcoded. The MX v1 body uses the fp8/bf8
    ``16x16x32`` MFMA atom, which is available on every CDNA gfx9 target
    that ships the fp8 MFMA family (gfx942 / gfx950); we reject the fp4 /
    fp6 mantissa families that genuinely require the gfx950-only MX MFMA
    intrinsics.
    """
    ok, reason, _target = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        return False, reason

    if spec.mantissa_dtype not in ("fp8e4m3", "bf8e5m2"):
        return False, (
            f"unsupported mantissa_dtype {spec.mantissa_dtype!r}; "
            "v1 ships fp8e4m3 / bf8e5m2 only (fp4 / fp6 are v2)"
        )
    if spec.group_k != 32:
        return False, (
            f"MX spec requires group_k = 32 (shared exponent per 32 "
            f"mantissa elements); got group_k={spec.group_k}"
        )
    if spec.K % spec.group_k:
        return False, f"K ({spec.K}) must be divisible by group_k ({spec.group_k})"
    if spec.M % spec.block_tile_m or spec.N % spec.block_tile_n:
        return False, "M / N must divide their tile sizes (v1 no partial tiles)"
    if (spec.block_tile_m, spec.block_tile_n) != (16, 16):
        return False, (
            f"mx_gemm MFMA path supports 16x16 tiles only "
            f"(got {spec.block_tile_m}x{spec.block_tile_n})"
        )
    return True, "ok"


def build_mx_gemm(spec: MxGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one MX GEMM instance (v1 scalar inner).

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU and is
    validated via :func:`is_valid_spec`. The fp8 / bf8 ``16x16x32`` MFMA
    atom this kernel uses ships on both targets; the scale split's cap
    depends on the element encoding the target's MFMA reads
    (:func:`mx_gemm_element_format`). The fp4 / fp6 mantissa families that
    genuinely require the gfx950-only MX MFMA intrinsics are rejected by
    :func:`is_valid_spec` (and the catalog guard) before any IR is built.

    Kernel signature::

    (A: ptr<fp8 | bf8>, # (M, K) mantissa
    AScale: ptr<i8>, # (M, K / 32) E8M0 shared exponent
    B: ptr<fp8 | bf8>, # (K, N) mantissa
    BScale: ptr<i8>, # (K / 32, N) E8M0 shared exponent
    C: ptr<f32>, # (M, N) f32 output
    M: i32, N: i32, K: i32)

    Grid: ``(ceil(N / block_tile_n), ceil(M / block_tile_m), 1)``.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid mx_gemm spec for {arch}: {why}")
    validate_mfma_atom_in_catalog(spec.atom, arch, where="mx_gemm")

    mantissa_ty = quant_ir_type(spec.mantissa_dtype)
    BS = spec.block_size
    # The split that makes each group's scale update exact, derived from the
    # bounds of a group's dot product in the elements the MFMA reads.
    element_format = mx_gemm_element_format(spec.mantissa_dtype, arch)
    scale_split = exact_mx_scale_split(element_format, element_format, spec.group_k)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(mantissa_ty, "global"), readonly=True, align=16)
    AScale = b.param("AScale", PtrType(I8, "global"), readonly=True, align=1)
    Bp = b.param("B", PtrType(mantissa_ty, "global"), readonly=True, align=16)
    BScale = b.param("BScale", PtrType(I8, "global"), readonly=True, align=1)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=4)
    _M = b.param("M", I32)  # noqa: F841 -- ABI
    _N = b.param("N", I32)  # noqa: F841 -- ABI
    _K = b.param("K", I32)  # noqa: F841 -- ABI

    lane = b.thread_id_x()
    bid_n = b.block_id_x()
    bid_m = b.block_id_y()
    atom = spec.atom
    m_tile_base = b.mul(bid_m, b.const_i32(spec.block_tile_m))
    n_tile_base = b.mul(bid_n, b.const_i32(spec.block_tile_n))

    lane_decode = decode_mfma_lanes(b, atom, lane)
    if spec.group_k % atom.k != 0:
        raise ValueError(
            f"MX group_k ({spec.group_k}) must be a multiple of atom.k "
            f"({atom.k}) so the per-group scale apply aligns with whole "
            f"MFMA invocations"
        )
    k_scale_count = spec.K // spec.group_k
    # ``atoms_per_group`` MFMA invocations cover one MX shared-exponent
    # group. With the MX-spec default (group_k=32, atom.k=32) this is
    # exactly 1, so the per-group MFMA work is a *single* atom -- emitting
    # it with a runtime ``scf.for`` of trip-count 1 (the prior structure)
    # paid a loop preamble, an iter-arg phi, and a yield per group for no
    # iteration count. We Python-unroll the group's atoms instead: the
    # count is a compile-time constant, the MFMA chain is fully visible to
    # the scheduler, and the trip-1 loop disappears.
    atoms_per_group = spec.group_k // atom.k

    # Loop-invariant scale-address bases. The DSL pass pipeline does not
    # run LICM (it never moves loads and only constant-folds/CSEs pure
    # ops), so the per-lane (m_row, n_col) coordinates and the A-scale row
    # stride must be hoisted out of the K-group loop by hand -- otherwise
    # they are recomputed every group. Inside the loop only the
    # ``kg``-dependent term is added.
    m_global_row = b.add(m_tile_base, lane_decode.m_in_atom)
    n_global_col = b.add(n_tile_base, lane_decode.n_in_atom)
    a_scale_row_base = b.mul(m_global_row, b.const_i32(k_scale_count))

    # Single-level K iteration over MX shared-exponent groups (one E8M0
    # byte per group). Each group's ``atoms_per_group`` MFMA invocations
    # are Python-unrolled into a fresh per-group acc, which is then scaled
    # by ``2^(ea + eb)`` and folded into the outer accumulator with one
    # fused ``v_pk_fma_f32`` per lane element. The MX
    # ``mfma_f32_16x16x128_f8f6f4`` intrinsic does the scale
    # in-instruction; the v1 chain replicates it explicitly so parity
    # isn't blocked on the in-instruction intrinsic.
    #
    # The fold runs one group late: group ``kg`` issues its scale, A and B
    # loads, folds group ``kg - 1`` (carried in ``prev_group`` with its two
    # scale codes) while those loads are in flight, then runs its own MFMA.
    # The scale math thus never waits on a load issued in the same group.
    # The first iteration folds a zero group with scale code 127 (2^0) into
    # the +0 accumulator, which leaves it +0, and the last group is folded
    # after the loop: the same updates in the same order as folding each
    # group straight after its MFMA, so the output is bit for bit the same.
    unit_scale_code = b.trunc(b.const_i32(_E8M0_CODE_FOR_ONE), I8)
    outer = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(k_scale_count),
        b.const_i32(1),
        [
            ("oacc", atom.zero_acc(b)),
            ("prev_group", atom.zero_acc(b)),
            ("prev_a_code", unit_scale_code),
            ("prev_b_code", unit_scale_code),
        ],
        iv_name="kg",
    )
    with outer as (kg, (outer_acc, prev_group, prev_a_code, prev_b_code)):
        # Per-group E8M0 scale byte loads. Only the ``kg``-dependent
        # term is added here; the per-lane (row, col) bases are hoisted.
        a_scale_off = b.add(a_scale_row_base, kg)
        b_scale_off = b.add(
            b.mul(kg, b.const_i32(spec.N)),
            n_global_col,
        )
        a_scale_code = b.global_load_i8(AScale, a_scale_off)
        b_scale_code = b.global_load_i8(BScale, b_scale_off)
        # Issue the two scale-byte loads before the A and B loads: no global
        # load may be scheduled across this point (every other instruction
        # may).
        b.sched_barrier(_SCALE_LOADS_FIRST)

        k_group_base = b.mul(kg, b.const_i32(spec.group_k))

        # The group's A and B operands, one pair per MFMA atom. The loads
        # stay in the shared helpers (row-major-contiguous A, col-strided B).
        operands = []
        for kt_local in range(atoms_per_group):
            k_tile_base = b.add(
                k_group_base,
                b.const_i32(kt_local * atom.k),
            )
            a_vec = load_a_row_major_contiguous(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=k_tile_base,
                K=spec.K,
            )
            b_vec = load_b_col_strided_scalars(
                b,
                B=Bp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=spec.N,
            )
            operands.append((a_vec, b_vec))

        # Fold the previous group while this group's loads are in flight.
        # The helper adds the two scale exponents as integers: the product
        # of the two decoded scales can overflow or flush to zero in f32
        # even when the result is an ordinary f32.
        new_outer = fma_with_mx_scale_pair(
            b, prev_group, prev_a_code, prev_b_code, outer_acc, split=scale_split
        )

        # Python-unrolled per-group MFMA chain on a fresh f32 acc: the
        # count is a compile-time constant, so no trip-1 ``scf.for``.
        group_acc = atom.zero_acc(b)
        for a_vec, b_vec in operands:
            group_acc = atom.emit(b, a_vec, b_vec, group_acc)
        b.scf_yield(new_outer, group_acc, a_scale_code, b_scale_code)

    # Fold the last group.
    outer_acc, last_group, last_a_code, last_b_code = outer.results
    acc_final = fma_with_mx_scale_pair(
        b, last_group, last_a_code, last_b_code, outer_acc, split=scale_split
    )

    # Output store via the shared MFMA epilogue helper: each lane
    # writes its ``c_per_lane`` cells to global via the atom's
    # ``lane_to_output`` mapping. f32 out (no cast), no atomic add.
    # The per-cell stores stay scalar because the 16x16 atom places a
    # lane's 4 outputs at the same column across 4 consecutive rows
    # (stride = N), which is not vectorisable without an LDS shuffle.
    store_acc_to_global(
        b,
        C=C,
        atom=atom,
        lane_decode=lane_decode,
        m_tile_base=m_tile_base,
        n_tile_base=n_tile_base,
        acc=acc_final,
        N=spec.N,
        out_dtype="f32",
    )
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
