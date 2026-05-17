# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Block-scaled GEMM kernel (CK Tile ``38_block_scale_gemm`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/38_block_scale_gemm``.
The kernel computes::

 C[m, n] = sum_k_block(
 a_scale[m_block, k_block] * b_scale[k_block, n_block]
 * sum_inner_k(A[m, k] * B[k, n])
 )

where ``A`` and ``B`` carry mantissa-only quantised values
(``fp8e4m3`` / ``bf8e5m2`` / packed i4) and the per-block scales are
``f32`` (one per ``(m_block, k_block)`` for A-quant, ``(k_block,
n_block)`` for B-quant, both for AB-quant). The variant matrix from
CK Tile's example (29 distinct preconfigured kernels) is captured here
by a single :class:`BlockScaleGemmSpec` whose fields select among:

* ``quant_mode`` -- ``"aquant"`` / ``"bquant"`` / ``"abquant"``: which
 side of the matmul carries the quantised mantissa.
* ``mantissa_dtype`` -- ``"fp8e4m3"`` / ``"bf8e5m2"`` / ``"i4_fp8"`` /
 ``"i4_bf8"``. The ``i4_*`` variants take packed-i4 weights and
 dequant on the fly to the trailing fp8 / bf8 mantissa before the
 MFMA path consumes them.
* ``preshuffle_b`` -- when True, expects B to be in the preshuffled
 tile-major layout described in :mod:`ck_dsl.helpers.preshuffle`.
* ``group_size_mnk`` -- ``(group_m, group_n, group_k)``; the
 per-block scale spans ``(group_m, group_n, group_k)`` elements of
 the original matrix. CK Tile defaults: ``(1, 1, 128)`` for the
 ``grouped`` variants, ``(1, 32, 128)`` / ``(1, 64, 128)`` for the
 wider-channel groups.

v1 ships a **scalar-inner** kernel: one thread per output element,
per-K-iter accumulation in f32, post-MFMA-style scale application.
This intentionally mirrors :mod:`ck_dsl.instances.streamk_gemm` -- the
 *infrastructure* (FP8 MFMA atoms, i4 unpack helpers, MX scale
helpers, preshuffle layout) ships end-to-end and is tested standalone;
the MFMA-based kernel that consumes those atoms in a tile-staged
inner loop lands as a focused follow-on against this stable spec.

The four CK Tile FP8 / BF8 / i4 / MX variant families
(``38_block_scale_gemm``'s ``aquant``, ``bquant``, ``abquant``,
``preshuffleb``, ``preshufflequant``, ``mxbf16``, ``i4`` etc.) all
resolve to a single :func:`build_block_scale_gemm` call with the
appropriate spec fields set. The shipped FP8 MFMA atoms in
:mod:`ck_dsl.helpers.atoms` and the dequant / MX helpers in
:mod:`ck_dsl.helpers.i4_dequant` / :mod:`ck_dsl.helpers.mx_scale` are
the pieces a v2 MFMA-based body composes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.atoms import MfmaAtom
from ..helpers.io import io_ir_type
from ..helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
)
from ..helpers.quant import quant_ir_type
from ..helpers.spec import SignatureBuilder, kernel_name_join


QuantMode = Literal["aquant", "bquant", "abquant"]
MantissaDType = Literal["fp8e4m3", "bf8e5m2", "i4_fp8", "i4_bf8"]


@dataclass(frozen=True)
class BlockScaleGemmSpec:
    """One concrete block-scaled GEMM kernel configuration.

    ``M`` / ``N`` / ``K`` and the ``group_size_mnk`` tuple are
    compile-time constants so the v1 scalar inner can statically
    derive the per-block scale indices. The group sizes default to
    ``(1, 1, 128)`` which matches CK Tile's ``--group_size 1x1x128``
    default.
    """

    M: int
    N: int
    K: int
    quant_mode: QuantMode = "bquant"
    mantissa_dtype: MantissaDType = "fp8e4m3"
    preshuffle_b: bool = False
    group_size_mnk: Tuple[int, int, int] = (1, 1, 128)
    block_tile_m: int = 16
    block_tile_n: int = 16
    name: str = "ck_dsl_block_scale_gemm"

    @property
    def atom(self) -> MfmaAtom:
        """MFMA atom selected from the (block_tile_m, block_tile_n,
        mantissa_dtype) triple. The 16x16 atom is the workhorse for
        fp8 / bf8 block-scaled GEMM; the 32x32 atom is a v2 hoist for
        wider output tiles.
        """
        if (self.block_tile_m, self.block_tile_n) != (16, 16):
            raise ValueError(
                f"block_scale_gemm MFMA path supports 16x16 tiles only "
                f"(got {self.block_tile_m}x{self.block_tile_n})"
            )
        if self.mantissa_dtype == "fp8e4m3":
            return MfmaAtom.fp8_16x16x32()
        if self.mantissa_dtype == "bf8e5m2":
            return MfmaAtom.bf8_16x16x32()
        # i4 variants dequant on the load path; the MFMA still uses
        # the fp8 atom on the dequantised inputs.
        if self.mantissa_dtype == "i4_fp8":
            return MfmaAtom.fp8_16x16x32()
        if self.mantissa_dtype == "i4_bf8":
            return MfmaAtom.bf8_16x16x32()
        raise ValueError(f"no atom for mantissa {self.mantissa_dtype!r}")

    @property
    def block_size(self) -> int:
        # MFMA path: one wave64 warp per CTA -- the MFMA atom is
        # per-wave and each output tile is one atom invocation chain.
        return 64

    def kernel_name(self) -> str:
        gm, gn, gk = self.group_size_mnk
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.quant_mode,
            self.mantissa_dtype,
            f"g{gm}x{gn}x{gk}",
            f"t{self.block_tile_m}x{self.block_tile_n}",
            flags={"psb": self.preshuffle_b},
        )


def is_valid_spec(spec: BlockScaleGemmSpec) -> Tuple[bool, str]:
    if spec.quant_mode not in ("aquant", "bquant", "abquant"):
        return False, f"unsupported quant_mode {spec.quant_mode!r}"
    if spec.quant_mode != "abquant":
        return False, (
            "MFMA block_scale_gemm currently ships quant_mode='abquant' only; "
            f"got {spec.quant_mode!r}"
        )
    if spec.mantissa_dtype not in ("fp8e4m3", "bf8e5m2", "i4_fp8", "i4_bf8"):
        return False, f"unsupported mantissa_dtype {spec.mantissa_dtype!r}"
    if spec.mantissa_dtype not in ("fp8e4m3", "bf8e5m2"):
        return False, (
            "MFMA block_scale_gemm currently ships fp8e4m3 / bf8e5m2 "
            f"mantissas only; got {spec.mantissa_dtype!r}"
        )
    if spec.preshuffle_b:
        return False, (
            "preshuffle_b=True requires the MFMA-based kernel body "
            "( follow-on); v1 ships the scalar inner only"
        )
    if spec.block_size > 1024:
        return False, (f"block_size {spec.block_size} > 1024 hardware cap")
    if any(g <= 0 for g in spec.group_size_mnk):
        return False, f"group_size_mnk must be positive, got {spec.group_size_mnk}"
    gk = spec.group_size_mnk[2]
    if spec.K % gk:
        return False, f"K ({spec.K}) must be divisible by group_k ({gk})"
    if spec.M % spec.block_tile_m or spec.N % spec.block_tile_n:
        return False, (
            "M / N must be divisible by their tile sizes "
            "(v1 doesn't handle partial tiles)"
        )
    return True, "ok"


def _mantissa_storage_dtype(spec: BlockScaleGemmSpec) -> str:
    """Map the mantissa enum to the storage dtype string the
    pointer parameter uses. i4 variants store as packed i8 bytes.
    """
    if spec.mantissa_dtype in ("fp8e4m3", "bf8e5m2"):
        return spec.mantissa_dtype
    # i4_fp8 / i4_bf8: packed two-per-byte storage.
    return "i8"


def build_block_scale_gemm(spec: BlockScaleGemmSpec) -> KernelDef:
    """Build the IR for one block-scaled GEMM instance (v1 scalar inner).

    Kernel signature (B-quant default; A-quant / AB-quant rotate the
    scale pointers symmetrically)::

    (A: ptr<f16 | fp8 | bf8>, # A storage dtype
    B: ptr<fp8 | bf8 | i8>, # mantissa storage (i8 for i4 packed)
    B_scale: ptr<f32, global>, # per-block scale, shape (Ks, Ns)
    C: ptr<f32, global>, # f32 output
    M: i32, N: i32, K: i32)

    Grid: ``(ceil(N / block_tile_n), ceil(M / block_tile_m), 1)``.

    v1 algorithm (per output element ``(m, n)``):

    1. ``acc = 0`` (f32).
    2. For each k in ``[0, K)``:
    ``a = cast(A[m, k]) to f32; b = cast(dequant(B[k, n])) to f32``
    ``b_scaled = b * B_scale[k // group_k, n // group_n]``
    ``acc += a * b_scaled``
    3. ``C[m, n] = acc``

    This is the **correctness oracle** for the FP8 / BF8 / i4 quant
    pipelines; the MFMA-based body ( follow-on) consumes the
    same spec + helper surface and trades the per-element scalar
    loop for the
    :func:`ck_dsl.helpers.mfma_atom("fp8", 16, 16, 32).emit` chain.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid block_scale_gemm spec: {why}")

    mantissa_store = _mantissa_storage_dtype(spec)
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    # When A is quantised, A's storage is the mantissa dtype; when B is
    # quantised, A is fp16. Possible combinations:
    # aquant : A = mantissa, B = fp16
    # bquant : A = fp16, B = mantissa
    # abquant : A = mantissa, B = mantissa (both quantised)
    a_store = mantissa_store if spec.quant_mode in ("aquant", "abquant") else "f16"
    b_store = mantissa_store if spec.quant_mode in ("bquant", "abquant") else "f16"

    A = b.param(
        "A",
        PtrType(_storage_ir_type(a_store), "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    Bp = b.param(
        "B",
        PtrType(_storage_ir_type(b_store), "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    # Scale pointers: per quant_mode.
    if spec.quant_mode in ("aquant", "abquant"):
        AScale = b.param("AScale", PtrType(F32, "global"), readonly=True, align=4)
    if spec.quant_mode in ("bquant", "abquant"):
        BScale = b.param("BScale", PtrType(F32, "global"), readonly=True, align=4)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=4)
    M = b.param("M", I32)  # noqa: F841 -- ABI; equals spec.M
    N = b.param("N", I32)  # noqa: F841 -- ABI; equals spec.N
    Kp = b.param("K", I32)  # noqa: F841 -- ABI

    lane = b.thread_id_x()
    bid_n = b.block_id_x()
    bid_m = b.block_id_y()
    atom = spec.atom
    m_tile_base = b.mul(bid_m, b.const_i32(spec.block_tile_m))
    n_tile_base = b.mul(bid_n, b.const_i32(spec.block_tile_n))

    if spec.quant_mode != "abquant":
        raise NotImplementedError(
            "MFMA block_scale_gemm v1 ships abquant only; aquant/bquant "
            "variants are a follow-on (same MFMA inner, asymmetric scale "
            "apply)."
        )
    if a_store != b_store or a_store not in ("fp8e4m3", "bf8e5m2"):
        raise NotImplementedError(
            f"MFMA path needs A and B in fp8e4m3 or bf8e5m2 "
            f"(got A={a_store}, B={b_store})"
        )

    lane_decode = decode_mfma_lanes(b, atom, lane)
    gm, gn, gk = spec.group_size_mnk
    if gk % atom.k != 0:
        raise ValueError(
            f"group_k ({gk}) must be a multiple of atom.k ({atom.k}) "
            f"so the per-group scale apply aligns with a whole number of "
            f"MFMA invocations"
        )

    n_scale_count_b = (spec.N + gn - 1) // gn
    k_scale_count_a = (spec.K + gk - 1) // gk

    def _load_a(b, kt):
        return load_a_row_major_contiguous(
            b,
            A=A,
            atom=atom,
            lane_decode=lane_decode,
            m_tile_base=m_tile_base,
            k_tile_base=b.mul(kt, b.const_i32(atom.k)),
            K=spec.K,
        )

    def _load_b(b, kt):
        return load_b_col_strided_scalars(
            b,
            B=Bp,
            atom=atom,
            lane_decode=lane_decode,
            n_tile_base=n_tile_base,
            k_tile_base=b.mul(kt, b.const_i32(atom.k)),
            N=spec.N,
        )

    # Per-group MFMA pipeline: run ``atoms_per_group`` MFMAs into a
    # zero-initialised group accumulator, multiply by ``a_scale *
    # b_scale``, fold into the outer accumulator. Mathematically
    # equivalent to scaling per element (sum_k a*sa * b*sb = sa*sb *
    # sum_k a*b) because the scales are constant within a group.
    num_groups = spec.K // gk
    c_atom_k = b.const_i32(atom.k)

    outer = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(num_groups),
        b.const_i32(1),
        [("acc", atom.zero_acc(b))],
        iv_name="kg",
    )
    with outer as (kg, (outer_acc,)):
        # Load per-group scales (one f32 each).
        a_scale_off = b.add(
            b.mul(
                b.div(b.add(m_tile_base, lane_decode.m_in_atom), b.const_i32(gm)),
                b.const_i32(k_scale_count_a),
            ),
            kg,
        )
        b_scale_off = b.add(
            b.mul(kg, b.const_i32(n_scale_count_b)),
            b.div(b.add(n_tile_base, lane_decode.n_in_atom), b.const_i32(gn)),
        )
        a_scale_v = b.global_load_f32(AScale, a_scale_off)
        b_scale_v = b.global_load_f32(BScale, b_scale_off)
        ab_scale = b.fmul(a_scale_v, b_scale_v)

        # Run atoms_per_group MFMA invocations on a fresh acc; their K
        # range is [kg * gk, (kg+1) * gk).
        k_group_base = b.mul(kg, b.const_i32(gk))

        def _load_a_in_group(b, kt_local, base=k_group_base):
            return load_a_row_major_contiguous(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=b.add(base, b.mul(kt_local, c_atom_k)),
                K=spec.K,
            )

        def _load_b_in_group(b, kt_local, base=k_group_base):
            return load_b_col_strided_scalars(
                b,
                B=Bp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=b.add(base, b.mul(kt_local, c_atom_k)),
                N=spec.N,
            )

        group_acc = mfma_k_loop(
            b,
            K=gk,
            atom=atom,
            load_a=_load_a_in_group,
            load_b=_load_b_in_group,
            iv_name="kk",
            acc_name="gacc",
        )

        # Vector fmul + fadd: scale-and-accumulate per-lane vec_f32.
        scaled = b.zero_vec_f32(atom.c_per_lane)
        for i in range(atom.c_per_lane):
            v = b.vec_extract(group_acc, i)
            old = b.vec_extract(outer_acc, i)
            new = b.fadd(old, b.fmul(v, ab_scale))
            scaled = b.vec_insert(scaled, new, i)
        b.scf_yield(scaled)

    acc_final = outer.results[0]

    # Output store: one lane writes c_per_lane cells via the atom's
    # lane→output mapping.
    for i in range(atom.c_per_lane):
        row_in, col_in = atom.lane_to_output(b, lane, i)
        row = b.add(m_tile_base, row_in)
        col = b.add(n_tile_base, col_in)
        addr = b.add(b.mul(row, b.const_i32(spec.N)), col)
        b.global_store(C, addr, b.vec_extract(acc_final, i), align=4)
    b.ret()
    return b.kernel


def _storage_ir_type(store: str):
    """Map a storage dtype string (``"f16"``, ``"fp8e4m3"``,
    ``"bf8e5m2"``, ``"i8"``) to the IR :class:`Type` for the pointer
    parameter. Packed-i4 weights live in i8 byte storage.
    """
    from ..core.ir import I8

    if store == "f16":
        return io_ir_type("f16")
    if store == "i8":
        return I8
    return quant_ir_type(store)


def block_scale_gemm_grid(spec: BlockScaleGemmSpec) -> Tuple[int, int, int]:
    n_tiles = (spec.N + spec.block_tile_n - 1) // spec.block_tile_n
    m_tiles = (spec.M + spec.block_tile_m - 1) // spec.block_tile_m
    return (n_tiles, m_tiles, 1)


def block_scale_gemm_signature(spec: BlockScaleGemmSpec):
    a_dtype = (
        "f16"
        if spec.quant_mode == "bquant"
        else ("i8" if spec.mantissa_dtype.startswith("i4_") else spec.mantissa_dtype)
    )
    b_dtype = (
        "f16"
        if spec.quant_mode == "aquant"
        else ("i8" if spec.mantissa_dtype.startswith("i4_") else spec.mantissa_dtype)
    )
    sb = SignatureBuilder().ptr("A", a_dtype).ptr("B", b_dtype)
    if spec.quant_mode in ("aquant", "abquant"):
        sb.ptr("AScale", "f32")
    if spec.quant_mode in ("bquant", "abquant"):
        sb.ptr("BScale", "f32")
    sb.ptr("C", "f32").scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
    return sb.build()


__all__ = [
    "BlockScaleGemmSpec",
    "MantissaDType",
    "QuantMode",
    "block_scale_gemm_grid",
    "block_scale_gemm_signature",
    "build_block_scale_gemm",
    "is_valid_spec",
]
