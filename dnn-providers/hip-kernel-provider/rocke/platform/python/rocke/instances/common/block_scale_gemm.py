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
 tile-major layout described in :mod:`rocke.helpers.preshuffle`.
* ``group_size_mnk`` -- ``(group_m, group_n, group_k)``; the
 per-block scale spans ``(group_m, group_n, group_k)`` elements of
 the original matrix. CK Tile defaults: ``(1, 1, 128)`` for the
 ``grouped`` variants, ``(1, 32, 128)`` / ``(1, 64, 128)`` for the
 wider-channel groups. Setting it to ``(M, N, K)`` collapses to one
 scale per tensor, which is the granularity ``torch._scaled_mm``
 passes for vLLM's dynamic fp8 linear layers.
* ``layout`` -- ``"RRR"`` (CK Tile's B as row-major ``(K, N)``) or
 ``"RCR"`` (B as row-major ``(N, K)``, i.e. a framework weight matrix
 stored ``[out_features, in_features]``). RCR makes B's K axis
 contiguous, so a lane loads its operand in one vector load.
* ``dtype_c`` -- ``"f32"`` / ``"bf16"`` / ``"f16"`` output encoding.
* ``k_split_waves`` -- how many wave64 warps share one output tile, each
 covering ``1 / k_split_waves`` of the K extent, with the partials summed
 through LDS before the epilogue. This is a pure occupancy knob: it does not
 change the result, only how much of a tile's K reduction is exposed as one
 wave's serial dependency chain. It exists because a skinny decode GEMM does
 not have enough output tiles to fill the device -- the router gate of a
 30B-class MoE model is 16 tiles against 32 CUs, so at one wave per tile half
 the CUs idle and the occupied ones have no second wave to run during a
 memory stall.

v1 ships a **scalar-inner** kernel: one thread per output element,
per-K-iter accumulation in f32, post-MFMA-style scale application.
This intentionally mirrors :mod:`rocke.instances.common.streamk_gemm` -- the
 *infrastructure* (FP8 MFMA atoms, i4 unpack helpers, MX scale
helpers, preshuffle layout) ships end-to-end and is tested standalone;
the MFMA-based kernel that consumes those atoms in a tile-staged
inner loop lands as a focused follow-on against this stable spec.

The four CK Tile FP8 / BF8 / i4 / MX variant families
(``38_block_scale_gemm``'s ``aquant``, ``bquant``, ``abquant``,
``preshuffleb``, ``preshufflequant``, ``mxbf16``, ``i4`` etc.) all
resolve to a single :func:`build_block_scale_gemm` call with the
appropriate spec fields set. The shipped FP8 MFMA atoms in
:mod:`rocke.helpers.atoms` and the dequant / MX helpers in
:mod:`rocke.helpers.i4_dequant` / :mod:`rocke.helpers.mx_scale` are
the pieces a v2 MFMA-based body composes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.atoms import MfmaAtom
from ...helpers.io import io_ir_type
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    load_b_row_major_contiguous,
    mfma_k_loop,
    store_acc_to_global,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.quant import quant_ir_type
from ...helpers.spec import SignatureBuilder, kernel_name_join
from ...helpers.tensor_view import make_lds_view


QuantMode = Literal["aquant", "bquant", "abquant"]
MantissaDType = Literal["fp8e4m3", "bf8e5m2", "i4_fp8", "i4_bf8"]
# "RRR": A (M, K), B (K, N), C (M, N), all row-major -- the CK Tile example's
# layout. "RCR": B instead row-major (N, K) (equivalently column-major (K, N)),
# which is how a framework stores a weight matrix and what ``torch._scaled_mm``
# requires of its second operand.
GemmLayout = Literal["RRR", "RCR"]
OutDType = Literal["f32", "bf16", "f16"]


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
    layout: GemmLayout = "RRR"
    dtype_c: OutDType = "f32"
    k_split_waves: int = 1
    name: str = "rocke_block_scale_gemm"
    # P77 sibling: per-output-row scale broadcast, same shape as
    # :class:`rocke.instances.common.mx_gemm.MxGemmSpec.per_input_row`.
    # Defaults to True for backwards compat; flip on per-row
    # varying-scale workloads.
    per_input_row: bool = True

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
        # The MFMA atom is per-wave and one output tile is one atom
        # invocation chain, so a CTA is ``k_split_waves`` wave64 warps: one
        # tile's worth of work, cut into that many K slices.
        return 64 * self.k_split_waves

    @property
    def lds_bytes(self) -> int:
        """LDS for the cross-wave partial reduction, zero when unsplit.

        Each of the ``k_split_waves`` waves parks its ``c_per_lane`` f32
        accumulators for all 64 lanes; wave 0 sums them and runs the epilogue.
        """
        if self.k_split_waves == 1:
            return 0
        return self.k_split_waves * 64 * self.atom.c_per_lane * 4

    def kernel_name(self) -> str:
        gm, gn, gk = self.group_size_mnk
        parts = [
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.quant_mode,
            self.mantissa_dtype,
            f"g{gm}x{gn}x{gk}",
            f"t{self.block_tile_m}x{self.block_tile_n}",
        ]
        # Non-default layout / output encoding only. The RRR f32 kernels keep
        # the names their parity fixtures and compile caches already hold.
        if self.layout != "RRR":
            parts.append(self.layout.lower())
        if self.dtype_c != "f32":
            parts.append(self.dtype_c)
        if self.k_split_waves != 1:
            parts.append(f"w{self.k_split_waves}")
        return kernel_name_join(*parts, flags={"psb": self.preshuffle_b})


def is_valid_spec(spec: BlockScaleGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    The v1 MFMA body uses the fp8 / bf8 ``16x16x32`` MFMA atom. That
    intrinsic is selectable on both gfx942 and gfx950 (the FP8 MFMA
    family ships from gfx940 onward), so the kernel is arch-neutral:
    ``arch`` (``"gfx942"`` / ``"gfx950"``) is validated against
    :class:`rocke.core.arch.ArchTarget` (unknown gfx names rejected) and
    the per-WG thread cap is taken from the target, but the emitted MFMA
    is the same on both. The fp8 atom is not gated through the MMA
    catalog ``has_shape`` check because the per-arch catalog tracks the
    f16/bf16 GEMM warp-tile shapes rather than this fp8 block-scale atom.
    """
    ok, reason, target = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        return False, reason
    if target.wave_size == 32:
        return False, (
            "block_scale_gemm is currently an MFMA low-bit path; "
            f"{arch} exposes WMMA for expert GEMMs. WMMA block-scale "
            "expert GEMMs are not implemented yet, so use BF16 expert "
            "GEMMs or target gfx942/gfx950 for FP8/BF8 block-scale."
        )
    if spec.quant_mode not in ("aquant", "bquant", "abquant"):
        return False, f"unsupported quant_mode {spec.quant_mode!r}"
    if spec.quant_mode != "abquant":
        return False, (
            "MFMA block_scale_gemm currently ships quant_mode='abquant' only; "
            f"got {spec.quant_mode!r}"
        )
    if spec.layout not in ("RRR", "RCR"):
        return False, f"unsupported layout {spec.layout!r}; expected RRR or RCR"
    if spec.dtype_c not in ("f32", "bf16", "f16"):
        return False, f"unsupported dtype_c {spec.dtype_c!r}"
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
    if spec.k_split_waves < 1:
        return False, f"k_split_waves must be >= 1, got {spec.k_split_waves}"
    if not target.fits_lds(spec.lds_bytes):
        return False, (
            f"k_split_waves={spec.k_split_waves} needs {spec.lds_bytes} B of LDS "
            f"for the cross-wave reduction, over the {target.lds_capacity_bytes} B "
            f"{arch} cap"
        )
    if any(g <= 0 for g in spec.group_size_mnk):
        return False, f"group_size_mnk must be positive, got {spec.group_size_mnk}"
    gk = spec.group_size_mnk[2]
    if spec.K % gk:
        return False, f"K ({spec.K}) must be divisible by group_k ({gk})"
    # Each wave takes an equal slice of every scale group, and a slice must be a
    # whole number of MFMA atoms. Splitting within a group rather than across
    # groups is what keeps the knob usable at per-tensor granularity, where
    # there is only one group to begin with.
    if gk % (spec.k_split_waves * spec.atom.k):
        return False, (
            f"group_k ({gk}) must be divisible by k_split_waves "
            f"({spec.k_split_waves}) x atom.k ({spec.atom.k}) so every wave "
            f"gets a whole number of MFMA atoms"
        )
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


def build_block_scale_gemm(spec: BlockScaleGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one block-scaled GEMM instance (v1 scalar inner).

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU and is
    validated via :func:`is_valid_spec`. The fp8 / bf8 ``16x16x32`` MFMA
    atom this kernel uses is selectable on both targets, so the emitted
    IR is arch-neutral.

    Kernel signature (B-quant default; A-quant / AB-quant rotate the
    scale pointers symmetrically)::

    (A: ptr<f16 | fp8 | bf8>, # A storage dtype
    B: ptr<fp8 | bf8 | i8>, # mantissa storage (i8 for i4 packed)
    B_scale: ptr<f32, global>, # per-block scale, shape (Ks, Ns)
    C: ptr<f32 | bf16 | f16>, # ``spec.dtype_c`` output
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
    :func:`rocke.helpers.mfma_atom("fp8", 16, 16, 32).emit` chain.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid block_scale_gemm spec for {arch}: {why}")
    validate_mfma_atom_in_catalog(spec.atom, arch, where="block_scale_gemm")

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
    c_ir = F32 if spec.dtype_c == "f32" else io_ir_type(spec.dtype_c)
    C = b.param(
        "C",
        PtrType(c_ir, "global"),
        writeonly=True,
        align=4 if spec.dtype_c == "f32" else 2,
    )
    M = b.param("M", I32)  # noqa: F841 -- ABI; equals spec.M
    N = b.param("N", I32)  # noqa: F841 -- ABI; equals spec.N
    Kp = b.param("K", I32)  # noqa: F841 -- ABI

    W = spec.k_split_waves
    tid = b.thread_id_x()
    # With one wave per CTA the thread id *is* the lane, and emitting the
    # mask/shift anyway would perturb the IR of every kernel that predates
    # this knob.
    lane = tid if W == 1 else b.mod(tid, b.const_i32(64))
    wave = None if W == 1 else b.div(tid, b.const_i32(64))
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
        # range is [kg * gk, (kg+1) * gk), or this wave's equal slice of it
        # when the tile's K reduction is split across waves.
        k_group_base = b.mul(kg, b.const_i32(gk))
        gk_per_wave = gk // W
        if W > 1:
            k_group_base = b.add(
                k_group_base, b.mul(wave, b.const_i32(gk_per_wave))
            )

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
            k_tile_base = b.add(base, b.mul(kt_local, c_atom_k))
            if spec.layout == "RCR":
                return load_b_row_major_contiguous(
                    b,
                    B=Bp,
                    atom=atom,
                    lane_decode=lane_decode,
                    n_tile_base=n_tile_base,
                    k_tile_base=k_tile_base,
                    K=spec.K,
                )
            return load_b_col_strided_scalars(
                b,
                B=Bp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=spec.N,
            )

        group_acc = mfma_k_loop(
            b,
            K=gk_per_wave,
            atom=atom,
            load_a=_load_a_in_group,
            load_b=_load_b_in_group,
            iv_name="kk",
            acc_name="gacc",
        )

        # Tile-level scale-and-accumulate the group's MFMA result.
        # ``group_acc`` and ``outer_acc`` are both ``<c_per_lane x f32>``
        # per-lane vectors; broadcast the scalar ``ab_scale`` to that
        # width and emit one vector fmul + one vector fadd, which lower
        # to packed ``v_pk_mul_f32`` / ``v_pk_add_f32`` (or a fused
        # ``v_pk_fma_f32``) on AMDGPU instead of the c_per_lane scalar
        # fmul/fadd chain the prior implementation produced. The
        # ab_scale broadcast is correct under the spec's current
        # uniform-scale assumption (group_m=group_n=1 collapses to per-
        # row/col scales that map 1:1 to lane outputs).
        ab_scale_vec = b.vector_splat(ab_scale, atom.c_per_lane)
        scaled_group = b.vector_mul(group_acc, ab_scale_vec)
        new_outer = b.vector_add(outer_acc, scaled_group)
        b.scf_yield(new_outer)

    acc_final = outer.results[0]

    def _epilogue(acc: "object") -> None:
        # Each lane writes its ``c_per_lane`` cells to global via the atom's
        # ``lane_to_output`` mapping. The per-cell stores stay scalar because
        # the 16x16 atom places a lane's 4 outputs at the same column across 4
        # consecutive rows (stride = N), which is not vectorisable without an
        # LDS shuffle.
        store_acc_to_global(
            b,
            C=C,
            atom=atom,
            lane_decode=lane_decode,
            m_tile_base=m_tile_base,
            n_tile_base=n_tile_base,
            acc=acc,
            N=spec.N,
            out_dtype=spec.dtype_c,
        )

    if W == 1:
        _epilogue(acc_final)
    else:
        # Lane L of every wave holds the partials for the *same* output cells,
        # so the reduction is a straight elementwise sum down the wave axis --
        # no shuffle, and each lane's whole accumulator moves as one vec4.
        cpl = atom.c_per_lane
        lds = make_lds_view(
            b, dtype=F32, shape=(W * 64, cpl), name_hint="ksplit_partials"
        ).base
        c_zero = b.const_i32(0)
        row = b.add(b.mul(wave, b.const_i32(64)), lane)
        b.smem_store_vN_f32(lds, [row, c_zero], acc_final, cpl)
        b.sync()
        with b.scf_if(b.cmp_eq(wave, c_zero)):
            total = b.smem_load_vN_f32(lds, lane, c_zero, n=cpl)
            for w in range(1, W):
                partial = b.smem_load_vN_f32(
                    lds, b.add(b.const_i32(w * 64), lane), c_zero, n=cpl
                )
                total = b.vector_add(total, partial)
            _epilogue(total)

    b.ret()
    return b.kernel


def _storage_ir_type(store: str):
    """Map a storage dtype string (``"f16"``, ``"fp8e4m3"``,
    ``"bf8e5m2"``, ``"i8"``) to the IR :class:`Type` for the pointer
    parameter. Packed-i4 weights live in i8 byte storage.
    """
    from ...core.ir import I8

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
    sb.ptr("C", spec.dtype_c).scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
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
