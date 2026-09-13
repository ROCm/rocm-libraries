# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Audit HIP debug lowering parity against production LLVM lowering.

This harness deliberately separates three levels of confidence:

1. Build a representative instance ``KernelDef``.
2. Require production LLVM lowering to succeed, then require HIP debug
   lowering to produce source for the same ``KernelDef``.
3. Optionally compile the generated HIP source to HSACO with ``hipcc
   --genco``. This catches source-level gaps that a string lowerer cannot
   see: missing prologue declarations, unsupported clang builtins, invalid
   vector types, and malformed C++.

The production path remains LLVM -> COMGR. HIP lowering is still a debug /
inspection / extraction path, so performance parity should only be claimed
after correctness and timing against the same input contract. The
``--bench-smoke`` mode currently performs that check for one cheap kernel
(``elementwise.add``) and is intended as the seed for broader per-instance
runtime parity.

Usage::

    python -m rocke.examples.common.hip_lowering_parity
    python -m rocke.examples.common.hip_lowering_parity --compile-hip --case gemm
    PYTHONPATH=python python \\
        python/rocke/examples/hip_lowering_parity.py --bench-smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from rocke.core.arch import ArchTarget  # noqa: E402
from rocke.core.lower_hip import lower_kernel_to_hip  # noqa: E402
from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from rocke.instances import (
    AddRmsnorm2DRdquantSpec,
    BatchedContractionSpec,
    BatchedGemmSpec,
    BatchedTranspose2DSpec,
    BlockScaleGemmSpec,
    ConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConvProblem,
    ElementwiseSpec,
    FlatMMSpec,
    FusedMoeSpec,
    GemmMultiAbdSpec,
    GemmMultiDSpec,
    Img2ColSpec,
    ImplicitGemmConvSpec,
    LayerNorm2DSpec,
    MfmaGemmSpec,
    MoeSmoothQuantSpec,
    MoeSortingSpec,
    MxGemmSpec,
    PermuteSpec,
    Pooling2DSpec,
    PoolingProblem,
    RMSNorm2DSpec,
    Reduce2DSpec,
    SmoothQuantSpec,
    StreamKGemmSpec,
    TileSpec,
    TopkSoftmaxSpec,
    TraitSpec,
    Transpose2DSpec,
    UniversalGemmSpec,
    build_add_rmsnorm2d_rdquant,
    build_batched_contraction,
    build_batched_gemm,
    build_batched_transpose2d,
    build_block_scale_gemm,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_elementwise,
    build_flatmm,
    build_gemm_multi_abd,
    build_gemm_multi_d,
    build_img2col,
    build_implicit_gemm_conv,
    build_layernorm2d,
    build_mfma_gemm,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_smoothquant,
    build_moe_sort_histogram,
    build_moe_sort_scan,
    build_moe_sort_scatter,
    build_moe_topk_weighted_reduce,
    build_mx_gemm,
    build_permute,
    build_pooling2d,
    build_reduce2d,
    build_rmsnorm2d,
    build_smoothquant,
    build_streamk_gemm,
    build_topk_softmax,
    build_transpose2d,
    build_universal_gemm,
)


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    build: Callable[[], object]
    # ``expect`` distinguishes an op that should lower cleanly ("ok") from one
    # whose HIP lowering is *intentionally* rejected on this target ("reject")
    # -- e.g. a WMMA atom on a CDNA/MFMA arch. For a "reject" case a
    # ``NotImplementedError`` from the HIP lowerer is the PASS condition; a
    # clean lowering is the FAILURE (it would mean the arch gate silently
    # dropped and the op mis-compiled). This is the guard the requirement asks
    # for against accidental handler drift.
    expect: str = "ok"
    # Optional per-case arch override. The arch-gated micro-kernels (WMMA needs
    # gfx11/gfx12, the async-DMA ops need gfx1250) pin the arch where they are
    # valid rather than inheriting the harness-wide ``--arch``.
    arch: Optional[str] = None


@dataclass
class AuditResult:
    name: str
    group: str
    expect: str = "ok"
    arch: str = ""
    llvm_ok: bool = False
    hip_ok: bool = False
    hip_compile_ok: Optional[bool] = None
    # Set True when ``expect == "reject"`` and the HIP lowerer raised the
    # intended ``NotImplementedError`` (the documented arch gate fired).
    rejected_ok: bool = False
    hip_chars: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        if self.expect == "reject":
            # An intentional-rejection case passes iff the HIP lowerer refused
            # the op with the documented NotImplementedError. (Some arch gates
            # -- e.g. WMMA on a CDNA target -- are symmetric: the LLVM lowerer
            # refuses it too, so we do NOT require ``llvm_ok`` here. The point
            # is that HIP never silently emits a mis-compiling handler.)
            return self.rejected_ok
        if not (self.llvm_ok and self.hip_ok):
            return False
        return self.hip_compile_ok is not False


def _base_tile() -> TileSpec:
    return TileSpec(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )


def _base_trait() -> TraitSpec:
    return TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="cshuffle")


def _base_gemm(name: str = "hip_audit_gemm") -> UniversalGemmSpec:
    return UniversalGemmSpec(name=name, tile=_base_tile(), trait=_base_trait())


def _conv_problem() -> ConvProblem:
    return ConvProblem(
        N=1,
        Hi=8,
        Wi=8,
        C=16,
        K=16,
        Y=3,
        X=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )


def _micro_cases() -> List[Case]:
    """Single-op micro-kernels for the ops added to close the HIP-lowering gap.

    Each kernel wires exactly one new op (plus the loads/stores needed to make
    it a valid ``__global__``) so a failure points unambiguously at that op's
    handler rather than at a large fused kernel. The arch-gated ops pin the
    arch where they are valid; two ``expect="reject"`` cases assert the WMMA
    arch gate fires on a CDNA target instead of silently mis-compiling.

    Imports are local so the harness's module import (used by the broad instance
    audit) does not depend on the low-level IR builder API.
    """
    from rocke.core.ir import (  # noqa: PLC0415
        BF16,
        F32,
        FP8E4M3,
        BF8E5M2,
        I32,
        IRBuilder,
        PtrType,
        VectorType,
    )

    def _vector_max() -> object:
        b = IRBuilder("micro_vector_max")
        A = b.param("A", PtrType(F32, "global"))
        B = b.param("B", PtrType(F32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        tid = b.thread_id_x()
        b.global_store_vN(
            C,
            tid,
            b.vector_max(
                b.global_load_vN(A, tid, F32, 4), b.global_load_vN(B, tid, F32, 4)
            ),
            4,
        )
        b.ret()
        return b.kernel

    def _cvt_scalef32(kind: str) -> object:
        elem = FP8E4M3 if kind == "fp8" else BF8E5M2
        b = IRBuilder(f"micro_cvt_scalef32_{kind}")
        A = b.param("A", PtrType(elem, "global"))
        S = b.param("S", PtrType(F32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        tid = b.thread_id_x()
        v = b.global_load_vN(A, tid, elem, 4)
        scale = b.global_load_f32(S, tid)
        r = (
            b.cvt_scalef32_pk_f32_fp8x4(v, scale)
            if kind == "fp8"
            else b.cvt_scalef32_pk_f32_bf8x4(v, scale)
        )
        b.global_store_vN(C, tid, r, 4)
        b.ret()
        return b.kernel

    def _wmma_bf16(name: str, width: int, gfx12: bool) -> object:
        b = IRBuilder(name)
        A = b.param("A", PtrType(BF16, "global"))
        B = b.param("B", PtrType(BF16, "global"))
        C = b.param("C", PtrType(F32, "global"))
        O = b.param("O", PtrType(F32, "global"))
        tid = b.thread_id_x()
        a = b.global_load_vN(A, tid, BF16, width)
        bb = b.global_load_vN(B, tid, BF16, width)
        c = b.global_load_vN(C, tid, F32, 8)
        r = (
            b.wmma_gfx12_f32_16x16x16_bf16(a, bb, c)
            if gfx12
            else b.wmma_f32_16x16x16_bf16(a, bb, c)
        )
        b.global_store_vN(O, tid, r, 8)
        b.ret()
        return b.kernel

    def _mfma_hero() -> object:
        # Dense unscaled fp8 MFMA -- HIP path REJECTS (expect="reject"). There
        # is no dedicated dense fp8 builtin; the only wide-K fp8 hardware op is
        # the combined ``f8f6f4`` form, exposed through the *scaled* builtin
        # (see ``_mfma_scale_f8f6f4``). Rather than fold that to an unscaled op,
        # the HIP lowerer declines and directs authors to
        # ``mfma_scale_f32_16x16x128_f8f6f4``. The LLVM path is the faithful one.
        b = IRBuilder("micro_mfma_f32_16x16x128_fp8")
        A = b.param("A", PtrType(I32, "global"))
        B = b.param("B", PtrType(I32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        O = b.param("O", PtrType(F32, "global"))
        tid = b.thread_id_x()
        a = b.bitcast(b.global_load_vN(A, tid, I32, 8), VectorType(FP8E4M3, 32))
        bb = b.bitcast(b.global_load_vN(B, tid, I32, 8), VectorType(FP8E4M3, 32))
        c = b.global_load_vN(C, tid, F32, 4)
        b.global_store_vN(O, tid, b.mma("mfma_f32_16x16x128_fp8", a, bb, c), 4)
        b.ret()
        return b.kernel

    def _s_wait_asynccnt() -> object:
        b = IRBuilder("micro_s_wait_asynccnt")
        A = b.param("A", PtrType(F32, "global"))
        O = b.param("O", PtrType(F32, "global"))
        tid = b.thread_id_x()
        v = b.global_load_f32(A, tid)
        b.s_wait_asynccnt(0)
        b.global_store_vN(O, tid, v, 1)
        b.ret()
        return b.kernel

    def _global_load_lds() -> object:
        b = IRBuilder("micro_global_load_lds")
        A = b.param("A", PtrType(F32, "global"))
        smem = b.smem_alloc(F32, [256], "lds")
        base = b.smem_addr_of(smem)
        b.global_load_lds(A, b.const_i32(0), base, 16)
        b.ret()
        return b.kernel

    def _global_load_async_to_lds() -> object:
        b = IRBuilder("micro_global_load_async_to_lds")
        A = b.param("A", PtrType(F32, "global"))
        tid = b.thread_id_x()
        smem = b.smem_alloc(F32, [256], "lds")
        b.global_load_async_to_lds(A, tid, smem, [tid], width_bytes=16)
        b.s_wait_asynccnt(0)
        b.ret()
        return b.kernel

    def _ds_read_tr_b8() -> object:
        # ``ds_read_b64_tr_b8`` transpose-read of an 8-bit LDS tile (gfx950).
        # Bitcast the ``<8 x fp8e4m3>`` result to ``<2 x i32>`` and store that,
        # so the case exercises only the transpose-read path (not the fp8
        # global-store path).
        from rocke.core.ir import (  # noqa: PLC0415
            FP8E4M3,
            I32,
            PtrType,
            VectorType,
        )

        b = IRBuilder("micro_ds_read_tr_b8")
        O = b.param("O", PtrType(I32, "global"))
        tid = b.thread_id_x()
        smem = b.smem_alloc(FP8E4M3, [256], "lds")
        raw = b.ds_read_tr_b8(smem, tid, dtype=FP8E4M3)
        packed = b.vec_bitcast(raw, VectorType(I32, 2))
        b.global_store_vN(O, tid, packed, 2)
        b.ret()
        return b.kernel

    def _inline_asm_mov() -> object:
        # Minimal single-output / single-input inline asm (``v_mov_b32``).
        # Exercises the output-lvalue + input binding of the general
        # ``asm volatile`` lowering on any AMDGPU target (no arch gate).
        b = IRBuilder("micro_inline_asm_mov")
        A = b.param("A", PtrType(I32, "global"))
        O = b.param("O", PtrType(I32, "global"))
        tid = b.thread_id_x()
        x = b.global_load(A, tid, I32)
        r = b.inline_asm("v_mov_b32 $0, $1", "=v,v", [x], result_type=I32)
        b.global_store(O, tid, r)
        b.ret()
        return b.kernel

    def _inline_asm_mfma_agpr() -> object:
        # The real register-pinning use case: unscaled
        # ``v_mfma_f32_16x16x128_f8f6f4`` with AGPR srcA/srcB and a tied VGPR
        # accumulator (constraints ``=v,0,a,a``). Validates the tied-operand,
        # vector-operand, and AGPR-constraint translation together. gfx950-only
        # (the f8f6f4 hero atom does not exist on gfx942).
        from rocke.helpers.asm import mfma_f8f6f4_agpr  # noqa: PLC0415

        b = IRBuilder("micro_inline_asm_mfma_agpr")
        A = b.param("A", PtrType(I32, "global"))
        B = b.param("B", PtrType(I32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        tid = b.thread_id_x()
        a = b.global_load_vN(A, tid, I32, 8)
        bb = b.global_load_vN(B, tid, I32, 8)
        acc = b.global_load_vN(C, tid, F32, 4)
        out = mfma_f8f6f4_agpr(b, a, bb, acc)
        b.global_store_vN(C, tid, out, 4)
        b.ret()
        return b.kernel

    def _mfma_scale_f8f6f4() -> object:
        # P15 MX MFMA *scaled* (``v_mfma_scale_f32_16x16x128_f8f6f4``). Both A/B
        # arrive as 32-byte mantissa vectors (loaded as ``<8 x i32>``) plus two
        # live i32 E8M0 scale bytes. The HIP path lowers this through the real
        # ``__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`` builtin -- the
        # only wide-K f8f6f4 hardware op, and the one the dense fp4/fp6/fp8
        # rejects point authors to -- so it is a faithful lowering, not a
        # passthrough stub. gfx950-only.
        b = IRBuilder("micro_mfma_scale_f8f6f4")
        A = b.param("A", PtrType(I32, "global"))
        B = b.param("B", PtrType(I32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        S = b.param("S", PtrType(I32, "global"))
        tid = b.thread_id_x()
        a = b.global_load_vN(A, tid, I32, 8)
        bb = b.global_load_vN(B, tid, I32, 8)
        acc = b.global_load_vN(C, tid, F32, 4)
        a_scale = b.global_load(S, tid, I32)
        b_scale = b.global_load(S, tid, I32)
        out = b.mfma_scale_f32_16x16x128_f8f6f4(a, bb, acc, a_scale, b_scale)
        b.global_store_vN(C, tid, out, 4)
        b.ret()
        return b.kernel

    def _mfma_fp4() -> object:
        # P52 dense fp4 MFMA -- HIP path REJECTS (expect="reject"). There is no
        # dedicated dense fp4 builtin; the only wide-K fp4 hardware op is the
        # combined ``f8f6f4`` form, exposed through the *scaled* builtin (see
        # ``_mfma_scale_f8f6f4``). Rather than emit a type-specific wrapper that
        # would diverge from that op, the HIP lowerer declines and directs
        # authors to ``mfma_scale_f32_16x16x128_f8f6f4``. The LLVM path is the
        # faithful one. A/B built as the 8-byte packs the LLVM path consumes.
        b = IRBuilder("micro_mfma_fp4")
        A = b.param("A", PtrType(I32, "global"))
        B = b.param("B", PtrType(I32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        tid = b.thread_id_x()
        a = b.global_load_vN(A, tid, I32, 2)
        bb = b.global_load_vN(B, tid, I32, 2)
        acc = b.global_load_vN(C, tid, F32, 4)
        out = b.mfma_f32_16x16x128_fp4(a, bb, acc)
        b.global_store_vN(C, tid, out, 4)
        b.ret()
        return b.kernel

    def _mfma_fp6() -> object:
        # P52 dense fp6 MFMA -- HIP path REJECTS (expect="reject"). There is no
        # dedicated dense fp6 builtin, and the K=96 shape is not a valid MFMA
        # shape on gfx950 or gfx1250; the only fp6 hardware op is the combined
        # K=128 ``f8f6f4`` form, exposed through the *scaled* builtin (see
        # ``_mfma_scale_f8f6f4``). Rather than emit a silently-different-K op,
        # the HIP lowerer declines and directs authors to
        # ``mfma_scale_f32_16x16x128_f8f6f4``. The LLVM path is the faithful one.
        # A/B are built as the 12-byte ``<3 x i32>`` packs the LLVM path
        # consumes (via <2 x i32> ++ scalar).
        def _i32x3(ptr: object) -> object:
            lo = b.global_load_vN(ptr, tid, I32, 2)  # <2 x i32>
            hi = b.vec_bitcast(b.global_load(ptr, tid, I32), VectorType(I32, 1))
            return b.vec_concat(lo, hi)  # <3 x i32>

        b = IRBuilder("micro_mfma_fp6")
        A = b.param("A", PtrType(I32, "global"))
        B = b.param("B", PtrType(I32, "global"))
        C = b.param("C", PtrType(F32, "global"))
        tid = b.thread_id_x()
        a = _i32x3(A)
        bb = _i32x3(B)
        acc = b.global_load_vN(C, tid, F32, 4)
        out = b.mfma_f32_16x16x96_fp6(a, bb, acc)
        b.global_store_vN(C, tid, out, 4)
        b.ret()
        return b.kernel

    def _register_p_from_qk_c() -> object:
        # P13 register permutation: cast the first 8 of 16 QK-MFMA f32 cells to
        # a ``<8 x f16>`` PV-A vector. Uses the f16 target specifically to
        # exercise the ``(fp16)`` scalar-cast spelling in the HIP shim (the raw
        # IR name ``f16`` has no C typedef in the prologue).
        from rocke.core.ir import F16  # noqa: PLC0415

        b = IRBuilder("micro_register_p_from_qk_c")
        A = b.param("A", PtrType(F32, "global"))
        O = b.param("O", PtrType(F16, "global"))
        tid = b.thread_id_x()
        # QK accumulator is <16 x f32>; global_load_vN caps f32 at width 8, so
        # load two halves and concat into the 16-wide fragment.
        qk = b.vec_concat(
            b.global_load_vN(A, tid, F32, 8), b.global_load_vN(A, tid, F32, 8)
        )
        pa = b.register_p_from_qk_c(qk, F16)
        b.global_store_vN(O, tid, pa, 8)
        b.ret()
        return b.kernel

    def _cooperative_global_store() -> object:
        # P14 cooperative global store: per-lane ``<4 x f32>`` values scattered
        # to per-lane ``<4 x i32>`` addresses. Faithful in both engines
        # (per-lane scatter store).
        b = IRBuilder("micro_cooperative_global_store")
        O = b.param("O", PtrType(F32, "global"))
        A = b.param("A", PtrType(I32, "global"))
        V = b.param("V", PtrType(F32, "global"))
        tid = b.thread_id_x()
        addrs = b.global_load_vN(A, tid, I32, 4)
        values = b.global_load_vN(V, tid, F32, 4)
        b.cooperative_global_store(O, addrs, values)
        b.ret()
        return b.kernel

    def _smem_store_distributed() -> object:
        # P42 distributed LDS publish: write a per-lane ``<4 x f32>`` slice into
        # consecutive LDS slots. Faithful in both engines (per-element store).
        b = IRBuilder("micro_smem_store_distributed")
        V = b.param("V", PtrType(F32, "global"))
        tid = b.thread_id_x()
        smem = b.smem_alloc(F32, [256], "lds")
        values = b.global_load_vN(V, tid, F32, 4)
        b.smem_store_distributed(smem, {}, values)
        b.ret()
        return b.kernel

    return [
        # -- P0: available on the default CDNA target --
        Case("micro.vector_max", "micro", _vector_max),
        Case("micro.inline_asm_mov", "micro", _inline_asm_mov),
        Case(
            "micro.inline_asm_mfma_agpr",
            "micro",
            _inline_asm_mfma_agpr,
            arch="gfx950",
        ),
        Case("micro.cvt_scalef32_fp8", "micro", lambda: _cvt_scalef32("fp8")),
        Case("micro.cvt_scalef32_bf8", "micro", lambda: _cvt_scalef32("bf8")),
        Case(
            "micro.mfma_f32_16x16x128_fp8",
            "micro",
            _mfma_hero,
            expect="reject",
            arch="gfx950",
        ),
        Case("micro.global_load_lds", "micro", _global_load_lds, arch="gfx950"),
        Case("micro.ds_read_tr_b8", "micro", _ds_read_tr_b8, arch="gfx950"),
        # -- shim-parity micro-cases (the 6 reviewed HIP shims) --
        Case("micro.mfma_scale_f8f6f4", "micro", _mfma_scale_f8f6f4, arch="gfx950"),
        Case("micro.mfma_fp4", "micro", _mfma_fp4, expect="reject", arch="gfx950"),
        Case("micro.mfma_fp6", "micro", _mfma_fp6, expect="reject", arch="gfx950"),
        Case(
            "micro.register_p_from_qk_c",
            "micro",
            _register_p_from_qk_c,
            arch="gfx950",
        ),
        Case(
            "micro.cooperative_global_store",
            "micro",
            _cooperative_global_store,
            arch="gfx950",
        ),
        Case(
            "micro.smem_store_distributed",
            "micro",
            _smem_store_distributed,
            arch="gfx950",
        ),
        # -- P1: arch-gated ops pinned to a supporting arch --
        Case(
            "micro.wmma_bf16",
            "micro",
            lambda: _wmma_bf16("micro_wmma_bf16", 16, False),
            arch="gfx1151",
        ),
        Case(
            "micro.wmma_gfx12_bf16",
            "micro",
            lambda: _wmma_bf16("micro_wmma_gfx12_bf16", 8, True),
            arch="gfx1201",
        ),
        Case("micro.s_wait_asynccnt", "micro", _s_wait_asynccnt, arch="gfx1250"),
        Case(
            "micro.global_load_async_to_lds",
            "micro",
            _global_load_async_to_lds,
            arch="gfx1250",
        ),
        # -- arch-gate assertions: WMMA must be REJECTED on a CDNA target --
        Case(
            "micro.wmma_bf16.reject_cdna",
            "micro",
            lambda: _wmma_bf16("micro_wmma_bf16_rej", 16, False),
            expect="reject",
            arch="gfx950",
        ),
        Case(
            "micro.wmma_gfx12_bf16.reject_cdna",
            "micro",
            lambda: _wmma_bf16("micro_wmma_gfx12_bf16_rej", 8, True),
            expect="reject",
            arch="gfx950",
        ),
        # ds_read_*_tr_* is gfx950-class; gfx942 has no such instruction.
        Case(
            "micro.ds_read_tr_b8.reject_cdna",
            "micro",
            _ds_read_tr_b8,
            expect="reject",
            arch="gfx942",
        ),
    ]


def make_cases(*, arch: str = "gfx950") -> List[Case]:
    base = _base_gemm()
    base_tile = _base_tile()
    base_trait = _base_trait()
    convp = _conv_problem()

    # Thread the running arch's wavefront width into the wave-size-sensitive
    # specs (the BlockReduce2d XOR-butterfly + cross-warp fold in reduce2d).
    # Defaulting to wave_size=64 would emit a 6-stage butterfly with a mask-32
    # ds_bpermute across lanes that don't exist on a wave32 (RDNA3/3.5/4)
    # target AND miscount num_warps (block_size//64 = 4 vs the real 8), so on
    # gfx1151/gfx1201 the cross-warp stage drops half the waves' partials --
    # this is the reduce2d sum/max drift the multi-arch run hit. Mirrors the
    # arch-threaded spec construction in rocke/examples/common/ck_tile_parity.py.
    red_wave = ArchTarget.from_gfx(arch).wave_size

    cases: List[Case] = [
        Case(
            "elementwise.add",
            "small",
            lambda: build_elementwise(ElementwiseSpec(op="add")),
        ),
        Case(
            "elementwise.silu",
            "small",
            lambda: build_elementwise(ElementwiseSpec(op="silu")),
        ),
        Case(
            "elementwise.gelu_tanh",
            "small",
            lambda: build_elementwise(ElementwiseSpec(op="gelu_tanh")),
        ),
        Case(
            "layernorm2d",
            "small",
            lambda: build_layernorm2d(
                LayerNorm2DSpec(n_per_block=2048, block_size=256, vec=8)
            ),
        ),
        Case(
            "rmsnorm2d",
            "small",
            lambda: build_rmsnorm2d(
                RMSNorm2DSpec(n_per_block=2048, block_size=256, vec=8)
            ),
        ),
        Case(
            "reduce.sum",
            "small",
            lambda: build_reduce2d(
                Reduce2DSpec(
                    n_per_block=2048,
                    op="sum",
                    block_size=256,
                    vec=8,
                    wave_size=red_wave,
                )
            ),
        ),
        Case(
            "reduce.max",
            "small",
            lambda: build_reduce2d(
                Reduce2DSpec(
                    n_per_block=2048,
                    op="max",
                    block_size=256,
                    vec=8,
                    wave_size=red_wave,
                )
            ),
        ),
        Case(
            "transpose2d",
            "small",
            lambda: build_transpose2d(Transpose2DSpec(tile_m=16, tile_n=16, vec=4)),
        ),
        Case(
            "batched_transpose2d",
            "small",
            lambda: build_batched_transpose2d(
                BatchedTranspose2DSpec(tile_m=16, tile_n=16, vec=4)
            ),
        ),
        Case(
            "img2col",
            "small",
            lambda: build_img2col(
                Img2ColSpec(problem=convp, block_tile_m=8, block_tile_k=64)
            ),
        ),
        Case(
            "pooling.max",
            "small",
            lambda: build_pooling2d(
                Pooling2DSpec(
                    problem=PoolingProblem(N=1, H=8, W=8, C=16, Y=2, X=2), op="max"
                )
            ),
        ),
        Case(
            "permute.2d",
            "small",
            lambda: build_permute(PermuteSpec(x_shape=(16, 32), perm=(1, 0))),
        ),
        Case(
            "smoothquant.i8",
            "quant",
            lambda: build_smoothquant(
                SmoothQuantSpec(n_per_block=2048, out_dtype="i8", block_size=256, vec=8)
            ),
        ),
        Case(
            "smoothquant.fp8",
            "quant",
            lambda: build_smoothquant(
                SmoothQuantSpec(
                    n_per_block=2048, out_dtype="fp8e4m3", block_size=256, vec=8
                )
            ),
        ),
        Case(
            "moe_smoothquant",
            "quant",
            lambda: build_moe_smoothquant(
                MoeSmoothQuantSpec(
                    n_per_block=2048,
                    topk=2,
                    experts=4,
                    out_dtype="i8",
                    block_size=256,
                    vec=8,
                )
            ),
        ),
        Case(
            "add_rmsnorm2d_rdquant",
            "quant",
            lambda: build_add_rmsnorm2d_rdquant(
                AddRmsnorm2DRdquantSpec(
                    n_per_block=2048, out_dtype="i8", block_size=256, vec=8
                )
            ),
        ),
        Case(
            "topk_softmax",
            "small",
            lambda: build_topk_softmax(
                TopkSoftmaxSpec(n_per_row=64, k=4, dtype="f16", out_dtype="f16")
            ),
        ),
        Case(
            "moe_sort_histogram",
            "moe",
            lambda: build_moe_sort_histogram(
                MoeSortingSpec(tokens=32, experts=4, topk=2)
            ),
        ),
        Case(
            "moe_sort_scan",
            "moe",
            lambda: build_moe_sort_scan(MoeSortingSpec(tokens=32, experts=4, topk=2)),
        ),
        Case(
            "moe_sort_scatter",
            "moe",
            lambda: build_moe_sort_scatter(
                MoeSortingSpec(tokens=32, experts=4, topk=2)
            ),
        ),
        Case(
            "moe_gather",
            "moe",
            lambda: build_moe_gather(
                FusedMoeSpec(
                    tokens=32,
                    experts=4,
                    topk=2,
                    hidden=128,
                    intermediate=256,
                    block_size=64,
                    vec=4,
                )
            ),
        ),
        Case(
            "moe_silu_mul",
            "moe",
            lambda: build_moe_silu_mul(
                FusedMoeSpec(
                    tokens=32,
                    experts=4,
                    topk=2,
                    hidden=128,
                    intermediate=256,
                    block_size=64,
                    vec=4,
                )
            ),
        ),
        Case(
            "moe_topk_weighted_reduce",
            "moe",
            lambda: build_moe_topk_weighted_reduce(
                FusedMoeSpec(
                    tokens=32,
                    experts=4,
                    topk=2,
                    hidden=128,
                    intermediate=256,
                    block_size=64,
                    vec=4,
                )
            ),
        ),
        Case("universal_gemm", "gemm", lambda: build_universal_gemm(base)),
        Case(
            "batched_gemm",
            "gemm",
            lambda: build_batched_gemm(
                BatchedGemmSpec(
                    name="hip_audit_batched_gemm", tile=base_tile, trait=base_trait
                )
            ),
        ),
        Case(
            "flatmm",
            "gemm",
            lambda: build_flatmm(FlatMMSpec(tile=base_tile, trait=base_trait)),
        ),
        Case(
            "batched_contraction",
            "gemm",
            lambda: build_batched_contraction(
                BatchedContractionSpec(
                    tile=base_tile, trait=base_trait, batch_shape=(2, 3)
                )
            ),
        ),
        Case(
            "gemm_multi_d",
            "gemm",
            lambda: build_gemm_multi_d(
                GemmMultiDSpec(base=base, d_operands=(("D0", "add"),))
            ),
        ),
        Case(
            "gemm_multi_abd",
            "gemm",
            lambda: build_gemm_multi_abd(
                GemmMultiAbdSpec(base=base, d_operands=(("D0", "add"),))
            ),
        ),
        Case(
            "mfma_gemm", "gemm", lambda: build_mfma_gemm(MfmaGemmSpec(M=64, N=64, K=32))
        ),
        Case(
            "block_scale_gemm",
            "gemm",
            lambda: build_block_scale_gemm(
                BlockScaleGemmSpec(
                    M=16, N=16, K=128, quant_mode="abquant", mantissa_dtype="fp8e4m3"
                )
            ),
        ),
        Case(
            "mx_gemm",
            "gemm",
            lambda: build_mx_gemm(
                MxGemmSpec(M=16, N=16, K=32, mantissa_dtype="fp8e4m3")
            ),
        ),
        Case(
            "streamk_gemm",
            "gemm",
            lambda: build_streamk_gemm(
                StreamKGemmSpec(
                    M=64, N=64, K=32, tile_m=16, tile_n=16, tile_k=16, num_cus=4
                )
            ),
        ),
        Case(
            "implicit_gemm_conv",
            "conv",
            lambda: build_implicit_gemm_conv(
                ImplicitGemmConvSpec(
                    problem=convp,
                    name="hip_audit_conv",
                    tile_m=64,
                    tile_n=64,
                    tile_k=64,
                    warp_m=2,
                    warp_n=2,
                    warp_tile_m=32,
                    warp_tile_n=32,
                    warp_tile_k=16,
                    pipeline="mem",
                    epilogue="cshuffle",
                )
            ),
        ),
        Case(
            "direct_conv_16c",
            "conv",
            lambda: build_direct_conv_16c(
                DirectConv16cSpec(
                    DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16)
                )
            ),
        ),
        Case(
            "direct_conv_4c",
            "conv",
            lambda: build_direct_conv_4c(
                DirectConv4cSpec(
                    DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4)
                )
            ),
        ),
    ]

    for dtype in ("f16", "bf16"):
        for op in (
            "copy",
            "neg",
            "abs",
            "relu",
            "quick_gelu",
            "swish",
            "tanh",
            "sigmoid",
            "exp2",
            "sub",
            "mul",
            "max",
            "min",
            "swiglu",
            "geglu",
        ):
            cases.append(
                Case(
                    f"elementwise.{dtype}.{op}",
                    "small",
                    lambda op=op, dtype=dtype: build_elementwise(
                        ElementwiseSpec(op=op, dtype=dtype)
                    ),
                )
            )

    for dtype in ("f16", "bf16"):
        cases.extend(
            [
                Case(
                    f"layernorm2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_layernorm2d(
                        LayerNorm2DSpec(
                            n_per_block=2048, dtype=dtype, block_size=256, vec=8
                        )
                    ),
                ),
                Case(
                    f"rmsnorm2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_rmsnorm2d(
                        RMSNorm2DSpec(
                            n_per_block=2048, dtype=dtype, block_size=256, vec=8
                        )
                    ),
                ),
                Case(
                    f"transpose2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_transpose2d(
                        Transpose2DSpec(tile_m=16, tile_n=16, vec=4, dtype=dtype)
                    ),
                ),
                Case(
                    f"batched_transpose2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_batched_transpose2d(
                        BatchedTranspose2DSpec(tile_m=16, tile_n=16, vec=4, dtype=dtype)
                    ),
                ),
            ]
        )

    for op in ("min", "mean", "prod"):
        cases.append(
            Case(
                f"reduce.{op}",
                "small",
                lambda op=op: build_reduce2d(
                    Reduce2DSpec(
                        n_per_block=2048,
                        op=op,
                        block_size=256,
                        vec=8,
                        wave_size=red_wave,
                    )
                ),
            )
        )
    cases.append(
        Case(
            "reduce.bf16.sum",
            "small",
            lambda: build_reduce2d(
                Reduce2DSpec(
                    n_per_block=2048,
                    op="sum",
                    dtype="bf16",
                    block_size=256,
                    vec=8,
                    wave_size=red_wave,
                )
            ),
        )
    )

    for op in ("avg", "sum"):
        cases.append(
            Case(
                f"pooling.{op}",
                "small",
                lambda op=op: build_pooling2d(
                    Pooling2DSpec(
                        problem=PoolingProblem(N=1, H=8, W=8, C=16, Y=2, X=2),
                        op=op,
                    )
                ),
            )
        )
    cases.append(
        Case(
            "permute.3d.bf16",
            "small",
            lambda: build_permute(
                PermuteSpec(x_shape=(4, 8, 16), perm=(2, 0, 1), dtype="bf16")
            ),
        )
    )

    for out_dtype in ("bf8e5m2",):
        cases.extend(
            [
                Case(
                    f"smoothquant.{out_dtype}",
                    "quant",
                    lambda out_dtype=out_dtype: build_smoothquant(
                        SmoothQuantSpec(
                            n_per_block=2048, out_dtype=out_dtype, block_size=256, vec=8
                        )
                    ),
                ),
                Case(
                    f"moe_smoothquant.{out_dtype}",
                    "quant",
                    lambda out_dtype=out_dtype: build_moe_smoothquant(
                        MoeSmoothQuantSpec(
                            n_per_block=2048,
                            topk=2,
                            experts=4,
                            out_dtype=out_dtype,
                            block_size=256,
                            vec=8,
                        )
                    ),
                ),
                Case(
                    f"add_rmsnorm2d_rdquant.{out_dtype}",
                    "quant",
                    lambda out_dtype=out_dtype: build_add_rmsnorm2d_rdquant(
                        AddRmsnorm2DRdquantSpec(
                            n_per_block=2048, out_dtype=out_dtype, block_size=256, vec=8
                        )
                    ),
                ),
            ]
        )

    for dtype, out_dtype in (("bf16", "bf16"), ("f32", "f32"), ("f16", "f32")):
        cases.append(
            Case(
                f"topk_softmax.{dtype}.{out_dtype}",
                "small",
                lambda dtype=dtype, out_dtype=out_dtype: build_topk_softmax(
                    TopkSoftmaxSpec(
                        n_per_row=64,
                        k=4,
                        dtype=dtype,
                        out_dtype=out_dtype,
                        block_size=64,
                    )
                ),
            )
        )

    for pipeline in ("compv3", "compv4"):
        cases.append(
            Case(
                f"universal_gemm.{pipeline}",
                "gemm",
                lambda pipeline=pipeline: build_universal_gemm(
                    UniversalGemmSpec(
                        name=f"hip_audit_gemm_{pipeline}",
                        tile=base_tile,
                        trait=TraitSpec(
                            pipeline=pipeline,
                            scheduler="intrawave",
                            epilogue="cshuffle",
                        ),
                    )
                ),
            )
        )
    cases.append(
        Case(
            "universal_gemm.default_epilogue",
            "gemm",
            lambda: build_universal_gemm(
                UniversalGemmSpec(
                    name="hip_audit_gemm_default_epilogue",
                    tile=base_tile,
                    trait=TraitSpec(
                        pipeline="mem", scheduler="intrawave", epilogue="default"
                    ),
                )
            ),
        )
    )
    cases.extend(
        [
            Case(
                "block_scale_gemm.bf8",
                "gemm",
                lambda: build_block_scale_gemm(
                    BlockScaleGemmSpec(
                        M=16,
                        N=16,
                        K=128,
                        quant_mode="abquant",
                        mantissa_dtype="bf8e5m2",
                    )
                ),
            ),
            Case(
                "mx_gemm.bf8",
                "gemm",
                lambda: build_mx_gemm(
                    MxGemmSpec(M=16, N=16, K=32, mantissa_dtype="bf8e5m2")
                ),
            ),
        ]
    )

    # Single-op micro-kernels for the ops added to close the HIP-lowering gap
    # (P0 cvt-scalef32 / vector.max, P1 arch-gated WMMA / async-DMA / fp8 hero).
    cases.extend(_micro_cases())

    return cases


def _selected(cases: Iterable[Case], needle: str) -> List[Case]:
    if needle == "all":
        return list(cases)
    return [c for c in cases if needle in c.name or needle == c.group]


def _compile_hip_source(
    source: str,
    *,
    name: str,
    arch: str,
    output_dir: Path,
    timeout_s: int,
) -> None:
    src = output_dir / f"{name.replace('.', '_')}.hip"
    hsaco = output_dir / f"{name.replace('.', '_')}.hsaco"
    src.write_text(source, encoding="utf-8")
    cmd = [
        "hipcc",
        f"--offload-arch={arch}",
        "--genco",
        str(src),
        "-o",
        str(hsaco),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        msg = "\n".join(proc.stdout.splitlines()[:40])
        raise RuntimeError(f"hipcc failed with rc={proc.returncode}\n{msg}")


def audit_cases(
    cases: Iterable[Case],
    *,
    compile_hip: bool,
    arch: str,
    emit_dir: Optional[Path],
    compile_timeout_s: int,
) -> List[AuditResult]:
    if emit_dir is None:
        tmp = tempfile.TemporaryDirectory()
        output_dir = Path(tmp.name)
    else:
        tmp = None
        output_dir = emit_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    results: List[AuditResult] = []
    try:
        for case in cases:
            # Per-case arch override: arch-gated micro-kernels pin the arch
            # where they are valid (WMMA -> gfx11/gfx12, async DMA -> gfx1250).
            case_arch = case.arch or arch
            result = AuditResult(
                name=case.name,
                group=case.group,
                expect=case.expect,
                arch=case_arch,
            )
            if case.expect == "reject":
                # Rejection cases assert the HIP arch gate fires. Attempt LLVM
                # independently (it may also refuse a symmetric arch gate; that
                # is fine and recorded), then require HIP to raise the
                # documented NotImplementedError.
                try:
                    kernel = case.build()
                except Exception as exc:  # noqa: BLE001
                    result.error = f"build: {type(exc).__name__}: {exc}"
                    results.append(result)
                    continue
                try:
                    lower_kernel_to_llvm(kernel, arch=case_arch)
                    result.llvm_ok = True
                except Exception:  # noqa: BLE001 - symmetric gate; not required
                    pass
                try:
                    lower_kernel_to_hip(kernel, arch=case_arch)
                    result.error = "HIP lowered an op expected to be rejected"
                except NotImplementedError as nie:
                    result.rejected_ok = True
                    result.error = f"NotImplementedError: {nie}"
                except Exception as exc:  # noqa: BLE001
                    result.error = f"{type(exc).__name__}: {exc}"
                results.append(result)
                continue
            try:
                kernel = case.build()
                lower_kernel_to_llvm(kernel, arch=case_arch)
                result.llvm_ok = True
                # Thread arch through the HIP lowering too: the HIP path is
                # arch-aware (s_waitcnt encoding, MFMA-vs-WMMA builtin family,
                # ds_read_*_tr_* availability). Without this the harness emits
                # gfx950 (default) MFMA source and then compiles it with
                # ``--offload-arch={arch}``, which silently mis-tests a non-CDNA
                # target (e.g. MFMA builtins on the WMMA-only gfx1151).
                hip_source = lower_kernel_to_hip(kernel, arch=case_arch)
                result.hip_ok = True
                result.hip_chars = len(hip_source)
                if compile_hip:
                    _compile_hip_source(
                        hip_source,
                        name=case.name,
                        arch=case_arch,
                        output_dir=output_dir,
                        timeout_s=compile_timeout_s,
                    )
                    result.hip_compile_ok = True
            except (
                Exception
            ) as exc:  # noqa: BLE001 - audit should classify all failures
                result.error = f"{type(exc).__name__}: {exc}"
                if compile_hip and result.hip_ok and result.hip_compile_ok is None:
                    result.hip_compile_ok = False
            results.append(result)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return results


def bench_elementwise_add(*, arch: str) -> None:
    # This script imports the full instance package for audit coverage, and
    # some instance modules import `runtime.launcher` at module import time.
    # Torch-backed launches need torch to initialize HIP first, so run the
    # smoke benchmark in a fresh child process with a minimal import order.
    code = f"""
import subprocess, sys, tempfile
from pathlib import Path
import torch
torch.cuda.init()
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise, elementwise_grid, elementwise_signature
from rocke.instances.common.mfma_gemm import MfmaGemmSpec, build_mfma_gemm, mfma_gemm_grid, mfma_gemm_signature
from rocke.instances.common.reduce import Reduce2DSpec, build_reduce2d, reduce2d_grid, reduce2d_signature
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release, time_launches

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)

    def make_launchers(name, kernel, signature):
        llvm_artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--{arch}")
        llvm_launcher = KernelLauncher(
            hsaco=llvm_artifact.hsaco,
            kernel_name=kernel.name,
            signature=signature,
        )
        stem = name.replace(".", "_")
        src = td_path / f"{{stem}}.hip"
        hsaco = td_path / f"{{stem}}.hsaco"
        src.write_text(lower_kernel_to_hip(kernel, arch="{arch}"), encoding="utf-8")
        proc = subprocess.run(
            ["hipcc", "--offload-arch={arch}", "--genco", str(src), "-o", str(hsaco)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            raise SystemExit(proc.returncode)
        hip_launcher = KernelLauncher(
            hsaco=hsaco.read_bytes(),
            kernel_name=kernel.name,
            signature=signature,
        )
        return llvm_launcher, hip_launcher

    def run_pair(name, llvm_launcher, hip_launcher, args_llvm, args_hip, cfg, out_llvm, out_hip, *, iters=50):
        llvm_launcher(args_llvm, config=cfg)
        hip_launcher(args_hip, config=cfg)
        synchronize_and_release()
        max_abs = float((out_llvm.float() - out_hip.float()).abs().max().item())
        llvm_ms = time_launches(lambda: llvm_launcher(args_llvm, config=cfg), warmup=5, iters=iters)
        hip_ms = time_launches(lambda: hip_launcher(args_hip, config=cfg), warmup=5, iters=iters)
        ratio = hip_ms / llvm_ms if llvm_ms else float("inf")
        print(
            f"BENCH {{name}} correct={{max_abs == 0.0}} max_abs={{max_abs:.3g}} "
            f"llvm_ms={{llvm_ms:.6f}} hip_ms={{hip_ms:.6f}} hip_over_llvm={{ratio:.3f}}"
        )

    # Memory/vector path smoke.
    ew_spec = ElementwiseSpec(op="add")
    ew_kernel = build_elementwise(ew_spec)
    ew_llvm, ew_hip = make_launchers("elementwise.add", ew_kernel, elementwise_signature(ew_spec))
    n = 1 << 20
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    c_llvm = torch.empty_like(a)
    c_hip = torch.empty_like(a)
    ew_cfg = LaunchConfig(grid=elementwise_grid(n, ew_spec), block=(ew_spec.block_size, 1, 1))
    run_pair(
        "elementwise.add",
        ew_llvm,
        ew_hip,
        {{"A": a, "B": b, "C": c_llvm, "N": n}},
        {{"A": a, "B": b, "C": c_hip, "N": n}},
        ew_cfg,
        c_llvm,
        c_hip,
    )

    # LDS reduction path smoke. wave_size threaded from the target arch so the
    # BlockReduce2d butterfly/cross-warp fold matches the hardware wavefront
    # (wave32 on RDNA, wave64 on CDNA) instead of the default 64.
    from rocke.core.arch import ArchTarget
    red_spec = Reduce2DSpec(
        n_per_block=2048, op="sum", block_size=256, vec=8,
        wave_size=ArchTarget.from_gfx("{arch}").wave_size,
    )
    red_kernel = build_reduce2d(red_spec)
    red_llvm, red_hip = make_launchers("reduce.sum", red_kernel, reduce2d_signature(red_spec))
    m = 128
    x = torch.randn(m, red_spec.n_per_block, device="cuda", dtype=torch.float16)
    y_llvm = torch.empty(m, device="cuda", dtype=torch.float16)
    y_hip = torch.empty(m, device="cuda", dtype=torch.float16)
    red_cfg = LaunchConfig(grid=reduce2d_grid(m, red_spec), block=(red_spec.block_size, 1, 1))
    run_pair(
        "reduce.sum",
        red_llvm,
        red_hip,
        {{"X": x, "Y": y_llvm, "M": m, "N": red_spec.n_per_block}},
        {{"X": x, "Y": y_hip, "M": m, "N": red_spec.n_per_block}},
        red_cfg,
        y_llvm,
        y_hip,
    )

    # MFMA path smoke.
    gemm_spec = MfmaGemmSpec(M=64, N=64, K=32)
    gemm_kernel = build_mfma_gemm(gemm_spec)
    gemm_llvm, gemm_hip = make_launchers("mfma_gemm", gemm_kernel, mfma_gemm_signature(gemm_spec))
    ga = torch.randn(gemm_spec.M, gemm_spec.K, device="cuda", dtype=torch.float16)
    gb = torch.randn(gemm_spec.K, gemm_spec.N, device="cuda", dtype=torch.float16)
    gc_llvm = torch.empty(gemm_spec.M, gemm_spec.N, device="cuda", dtype=torch.float16)
    gc_hip = torch.empty_like(gc_llvm)
    gemm_cfg = LaunchConfig(grid=mfma_gemm_grid(gemm_spec), block=(gemm_spec.block_size, 1, 1))
    run_pair(
        "mfma_gemm",
        gemm_llvm,
        gemm_hip,
        {{"A": ga, "B": gb, "C": gc_llvm, "M": gemm_spec.M, "N": gemm_spec.N, "K": gemm_spec.K}},
        {{"A": ga, "B": gb, "C": gc_hip, "M": gemm_spec.M, "N": gemm_spec.N, "K": gemm_spec.K}},
        gemm_cfg,
        gc_llvm,
        gc_hip,
        iters=30,
    )
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"bench smoke failed with rc={proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="substring or group filter")
    parser.add_argument(
        "--compile-hip", action="store_true", help="compile HIP source to HSACO"
    )
    parser.add_argument("--compile-timeout-s", type=int, default=180)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--emit-dir", type=Path, default=None)
    parser.add_argument("--bench-smoke", action="store_true")
    args = parser.parse_args()

    cases = _selected(make_cases(arch=args.arch), args.case)
    if not cases:
        print(f"no cases matched {args.case!r}")
        return 2

    results = audit_cases(
        cases,
        compile_hip=args.compile_hip,
        arch=args.arch,
        emit_dir=args.emit_dir,
        compile_timeout_s=args.compile_timeout_s,
    )
    for r in results:
        compile_status = ""
        if args.compile_hip and r.expect != "reject":
            compile_status = f" hipcc={'OK' if r.hip_compile_ok else 'FAIL'}"
        status = "OK" if r.ok else "FAIL"
        if r.expect == "reject":
            # For a rejection case, "hip" means "was the op refused as intended".
            hip_col = "REJECTED" if r.rejected_ok else "LOWERED!"
        else:
            hip_col = "OK" if r.hip_ok else "FAIL"
        print(
            f"{status:4} {r.group:9} {r.name:34} arch={r.arch:8} "
            f"llvm={'OK' if r.llvm_ok else 'FAIL'} hip={hip_col} "
            f"chars={r.hip_chars}{compile_status}"
        )
        if r.error:
            print(f"     {r.error}")

    failures = [r for r in results if not r.ok]
    print(
        f"SUMMARY total={len(results)} ok={len(results) - len(failures)} fail={len(failures)}"
    )

    if args.bench_smoke:
        bench_elementwise_add(arch=args.arch)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
