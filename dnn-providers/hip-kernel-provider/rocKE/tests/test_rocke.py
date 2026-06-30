# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the rocke Python authoring layer.

Tests are organised mirror-of-layer:

  - `TestCoreIR`       : `rocke.core.ir` - IR construction smoke tests
  - `TestLowering`     : `rocke.core.lower_llvm` - LLVM IR text shape
  - `TestTransforms`   : `rocke.helpers.transforms` - coord-transform DAG
  - `TestHelpers`      : `rocke.helpers` - atoms, geometry, loads,
                          epilogues
  - `TestInstances`    : `rocke.instances` - end-to-end build smoke
                          tests for the parametric kernels

These run in-process (no subprocess, no GPU); they test the static
IR/lowering pipeline only. End-to-end runtime tests live in
`test_rocke_examples.py`.
"""

from __future__ import annotations

import unittest

from rocke import (
    F16,
    I32,
    I64,
    IRBuilder,
    PtrType,
    TensorDescriptor,
    VariantReport,
    analyze_llvm_ir,
    compare_variant_reports,
    compile_kernel,
    embed,
    lower_kernel_to_llvm,
    mfma_atom,
    optimize_kernel,
    parse_isa,
    parse_resources,
    pad,
    summarize_runs,
    unmerge,
    select_2d_config,
    select_3d_config,
    use_2d_kernel,
)
from rocke.helpers import (
    AsyncTileLoader,
    CoalescedTileLoader,
    KernelArtifact,
    LdsLayout,
    SchedulePolicy,
    SoftwarePipeline,
    WarpGrid,
    make_gemm_manifest,
)
from rocke.helpers.compile import _comgr_options_for_kernel
from rocke.instances import (
    ConvProblem,
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    ImplicitGemmConvSpec,
    TileSpec,
    TraitSpec,
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    UniversalGemmSpec,
    attention_3d_workspace_nbytes,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
    build_direct_conv_4c,
    build_direct_conv_16c,
    build_implicit_gemm_conv,
    build_universal_gemm,
    supports_native_unified_attention,
)


# ---------------------------------------------------------------------
# Core IR
# ---------------------------------------------------------------------


class TestCoreIR(unittest.TestCase):
    def test_ir_builder_basic_kernel(self):
        b = IRBuilder("simple")
        A = b.param("A", PtrType(F16, "global"))
        M = b.param("M", I32)
        tid = b.thread_id_x()
        off = b.add(tid, M)
        b.global_load_f16(A, off)
        self.assertEqual(b.kernel.name, "simple")
        # KernelDef has a single body Region with a flat ops list.
        self.assertGreater(len(b.kernel.body.ops), 0)

    def test_lower_llvm_emits_amdgpu_target_triple(self):
        b = IRBuilder("smoke")
        b.param("A", PtrType(F16, "global"))
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn('target triple = "amdgcn-amd-amdhsa"', ll)
        self.assertIn("define amdgpu_kernel void @smoke", ll)

    def test_lower_llvm_emits_agpr_alloc_kernel_attr(self):
        b = IRBuilder("agpr_attr_smoke")
        b.kernel.attrs["agpr_alloc"] = (0, 0)
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn('"amdgpu-agpr-alloc"="0,0"', ll)

    def test_zero_agpr_alloc_requests_mfma_vgpr_form(self):
        b = IRBuilder("agpr_codegen_option_smoke")
        b.kernel.attrs["agpr_alloc"] = (0, 0)
        self.assertEqual(
            _comgr_options_for_kernel(b.kernel),
            ["-O3", "-mllvm", "-amdgpu-mfma-vgpr-form"],
        )

    def test_static_for_unrolls_without_runtime_loop(self):
        b = IRBuilder("static_for_smoke")
        b.static_for(0, 3, body=lambda i: b.const_i32(i))
        self.assertEqual(len(b.kernel.body.ops), 3)
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertNotIn("for.header", ll)

    def test_ssa_value_cannot_be_used_as_python_bool(self):
        b = IRBuilder("bool_guard_smoke")
        v = b.const_i32(1)
        with self.assertRaises(TypeError):
            if v:  # noqa: SIM108 - this is the guardrail being tested
                pass

    def test_static_if_requires_host_bool_and_scf_if_lowers(self):
        b = IRBuilder("branch_smoke")
        b.static_if(True, lambda: b.const_i32(1))
        v = b.const_i32(1)
        with self.assertRaises(TypeError):
            b.static_if(v, lambda: b.const_i32(2))
        cond = b.cmp_eq(v, b.const_i32(1))
        with b.scf_if(cond):
            b.const_i32(3)
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("br i1", ll)
        self.assertIn("if.then", ll)

    def test_optimize_kernel_cse_keeps_side_effects(self):
        b = IRBuilder("opt_smoke")
        A = b.param("A", PtrType(F16, "global"))
        c2 = b.const_i32(2)
        c3 = b.const_i32(3)
        b.add(c2, c3)
        a2 = b.add(c2, c3)
        b.global_load_f16(A, a2)
        stats = optimize_kernel(b.kernel)
        self.assertGreaterEqual(stats.common_subexpressions, 1)
        self.assertTrue(
            any(op.name == "memref.global_load" for op in b.kernel.body.ops)
        )

    def test_param_metadata_lowers_to_llvm_arg_attrs(self):
        b = IRBuilder("metadata_smoke")
        b.param(
            "A",
            PtrType(F16, "global"),
            noalias=True,
            readonly=True,
            align=16,
            dereferenceable=128,
        )
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn(
            "ptr addrspace(1) noalias readonly nocapture align 16 dereferenceable(128) %A",
            ll,
        )

    def test_s_waitcnt_encodes_extended_vmcnt_without_wrapping(self):
        b = IRBuilder("waitcnt_extended")
        # gfx950 uses the gfx9/gfx10 s_waitcnt layout: vmcnt is six bits
        # split across low bits [3:0] and high bits [15:14]. This must not
        # wrap vmcnt=16 to vmcnt(0), or the 2D attention kernel's partial
        # wait becomes a full VMEM drain. lgkmcnt=16 is out of range on
        # gfx950 and should clamp to 15 rather than wrap to 0.
        b.s_waitcnt(vmcnt=16, lgkmcnt=16)
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("call void @llvm.amdgcn.s.waitcnt(i32 20336)", ll)


# ---------------------------------------------------------------------
# Coord-transform DAG
# ---------------------------------------------------------------------


class TestTransforms(unittest.TestCase):
    def test_conv_offset_surface(self):
        N, Hi, Wi, C = 8, 56, 56, 64
        R, S = 3, 3
        Ho, Wo = 56, 56
        desc = TensorDescriptor.naive(
            "A_nhwc",
            lengths=[N, Hi, Wi, C],
            dtype=F16,
            coord_names=["n", "hi", "wi", "c"],
        ).transform(
            unmerge("m", into=["n", "ho", "wo"], dims=[N, Ho, Wo]),
            embed(["ho", "r"], "hi", strides=[1, 1], offset=-1, lo=0, hi=Hi),
            embed(["wo", "s"], "wi", strides=[1, 1], offset=-1, lo=0, hi=Wi),
            unmerge("k", into=["r", "s", "c"], dims=[R, S, C]),
            pad("r", lo=0, hi=R),
            pad("s", lo=0, hi=S),
        )
        self.assertEqual(set(desc.upper_names), {"m", "k"})

        b = IRBuilder("check_transform_dag")
        A = b.param("A", PtrType(F16, "global"))
        m = b.param("m", I32)
        k = b.param("k", I32)
        off, valid = desc.offset(b, m=m, k=k)
        safe = b.select(valid, off, b.const_i32(0))
        b.global_load_f16(A, safe)
        ll = lower_kernel_to_llvm(b.kernel)
        # The conv address arithmetic must contain at least one sdiv (for
        # `m -> (n, ho, wo)` unmerge), srem (the same), and an icmp slt
        # (the pad's bounds check).
        self.assertIn("sdiv i32 %m", ll)
        self.assertIn("srem i32 %m", ll)
        self.assertIn("icmp slt i32", ll)


# ---------------------------------------------------------------------
# Analysis / benchmark utilities
# ---------------------------------------------------------------------


class TestAnalysisAndBenchmark(unittest.TestCase):
    def test_analyze_llvm_ir_counts_async_and_mfma(self):
        llvm = """
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32, i32, i32, i32, i32)
define amdgpu_kernel void @k() {
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %r, ptr addrspace(3) %p, i32 16, i32 %v, i32 0, i32 0, i32 0)
  %acc = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a, <8 x half> %b, <16 x float> %c, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 3840)
  call void @llvm.amdgcn.s.barrier()
  ret void
}
"""
        stats = analyze_llvm_ir(llvm)
        self.assertEqual(stats.async_buffer_load_lds_calls, 1)
        self.assertEqual(stats.raw_buffer_load_calls, 0)
        self.assertEqual(stats.mfma_32x32x16, 1)
        self.assertEqual(stats.waitcnts, 1)
        self.assertEqual(stats.barriers, 1)

    def test_parse_isa_and_resources(self):
        asm = """
000000000000: v_mfma_f32_32x32x16_f16 a[0:15], v[0:7], v[8:15], a[0:15]
000000000002: v_mfma_f32_32x32x16_f16 v[32:47], v[0:7], v[8:15], v[32:47]
000000000003: v_accvgpr_read_b32 v0, a0
000000000004: v_accvgpr_write_b32 a0, v0
000000000004: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 offen
000000000008: buffer_load_lds_dwordx4 off, s[0:3], 0 offen
00000000000c: ds_read_b128 v[0:3], v0
000000000010: ds_write_b128 v0, v[0:3]
000000000014: s_waitcnt vmcnt(0)
000000000018: s_barrier
.amdhsa_next_free_vgpr 76
.amdhsa_next_free_sgpr 48
.amdhsa_group_segment_fixed_size 32768
"""
        isa = parse_isa(asm)
        res = parse_resources(asm)
        self.assertEqual(isa.mfma, 2)
        self.assertEqual(isa.mfma_dest_agpr, 1)
        self.assertEqual(isa.mfma_dest_vgpr, 1)
        self.assertEqual(isa.mfma_c_agpr, 1)
        self.assertEqual(isa.mfma_c_vgpr, 1)
        self.assertEqual(isa.accvgpr_read, 1)
        self.assertEqual(isa.accvgpr_write, 1)
        self.assertEqual(isa.buffer_load, 1)
        self.assertEqual(isa.buffer_load_lds, 1)
        self.assertEqual(isa.ds_read, 1)
        self.assertEqual(isa.ds_write, 1)
        self.assertEqual(isa.s_waitcnt, 1)
        self.assertEqual(isa.s_barrier, 1)
        self.assertEqual(res.vgpr_count, 76)
        self.assertEqual(res.sgpr_count, 48)
        self.assertEqual(res.lds_bytes, 32768)

    def test_benchmark_summary_discards_first_and_reports_spread(self):
        s = summarize_runs(
            ms=[1.0, 0.5, 0.25],
            tflops=[100.0, 200.0, 400.0],
            gbps=[10.0, 20.0, 40.0],
            discard_first=True,
        )
        self.assertEqual(s.attempts, 2)
        self.assertEqual(s.median_tflops, 300.0)
        self.assertEqual(s.min_tflops, 200.0)
        self.assertEqual(s.max_tflops, 400.0)
        self.assertGreater(s.spread_pct, 0.0)

    def test_variant_report_summary_row(self):
        b = IRBuilder("report_smoke")
        artifact = KernelArtifact(
            kernel=b.kernel,
            ir_text="",
            llvm_text="""
define amdgpu_kernel void @k() {
  %acc = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %a, <8 x half> %b, <16 x float> %c, i32 0, i32 0, i32 0)
  ret void
}
""",
            hsaco=b"",
            timings={"total": 1.0},
        )
        bench = summarize_runs(ms=[1.0], tflops=[123.0], gbps=[4.0])
        report = VariantReport.from_artifact(
            name="r",
            spec={"kind": "unit"},
            artifact=artifact,
            benchmark=bench,
        )
        rows = compare_variant_reports([report])
        self.assertEqual(rows[0]["median_tflops"], 123.0)
        self.assertEqual(rows[0]["llvm_mfma"], 1)


# ---------------------------------------------------------------------
# Attention dispatch matrix (shared helper for TestHelpers)
# ---------------------------------------------------------------------


def _attention_problem_matrix():
    """Curated ``(label, UnifiedAttentionProblem)`` list for the per-arch
    attention dispatch/spec drift net.

    The matrix is *curated*, not a full cartesian product: it spans dtype
    (fp16/bf16/fp8), head_size {64,128,256}, block_size {16,32,64},
    num_queries_per_kv {1,4,8,16}, the four regimes (decode / short prefill /
    long prefill / long-context decode), num_seqs {1, >=2} and the
    sliding-window / softcap / alibi / qq_bias / sinks toggles -- arranged so
    that BOTH the gfx942 fp16-flash branch (fp16 long-prefill D128 & D64, no
    SW/softcap, num_seqs>=2) and the bf16 transposed "combo" branch
    (bf16 / HD=64 / BS=32 / num_queries_per_kv=8 / long-prefill / multi-batch)
    of ``_tiled_spec_from_problem`` are reached, plus the decode path that
    routes to the 3D split-KV builder.

    Every config keeps ``num_query_heads % num_kv_heads == 0`` and a
    ``total_q`` / ``max_seqlen_q`` pairing consistent with its regime. The
    consumers below assert no *signature drift* (no ``TypeError`` from a
    per-arch impl whose kwargs/fields fell out of sync) -- they deliberately do
    NOT assert per-shape accept/reject verdicts, which legitimately differ by
    arch (e.g. HD=256 is unsupported on the gfx942 2D path, the gfx942 flash
    branch only fires on gfx942).
    """
    cfgs = []

    def add(label, **kw):
        base = dict(
            total_q=0,
            num_seqs=1,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=2048,
            dtype="fp16",
        )
        base.update(kw)
        if base["total_q"] == 0:
            base["total_q"] = base["num_seqs"] * base["max_seqlen_q"]
        cfgs.append((label, UnifiedAttentionProblem(**base)))

    # --- decode (max_seqlen_q=1) -> routes to the 3D split-KV builder ---
    # GQA fan-outs 1/4/8/16, all head sizes, all block sizes, fp16/bf16.
    for hd in (64, 128, 256):
        for bs in (16, 32, 64):
            add(
                f"decode_fp16_d{hd}_b{bs}",
                head_size=hd,
                block_size=bs,
                dtype="fp16",
                max_seqlen_q=1,
                num_seqs=2,
                num_query_heads=8,
                num_kv_heads=1,
                max_seqlen_k=4096,
            )
    add(
        "decode_bf16_d128_mha",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=4,
        num_query_heads=8,
        num_kv_heads=8,  # MHA: num_queries_per_kv == 1
        max_seqlen_k=4096,
    )
    add(
        "decode_bf16_d64_gqa4",
        head_size=64,
        block_size=16,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=3,
        num_query_heads=16,
        num_kv_heads=4,  # num_queries_per_kv == 4
        max_seqlen_k=4096,
    )
    add(
        "decode_fp16_d128_gqa16",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=1,  # num_queries_per_kv == 16
        max_seqlen_k=4096,
    )
    # long-context decode (large KV, still q==1) -> stresses 3D segment count.
    add(
        "decode_longctx_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=8192,
    )
    add(
        "decode_longctx_bf16_d64",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=8192,
    )
    # decode toggles: sliding window / softcap / sinks.
    add(
        "decode_sw_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        sliding_window=128,
    )
    add(
        "decode_softcap_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        softcap=30.0,
    )
    add(
        "decode_sinks_bf16_d128",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        use_sinks=True,
    )
    # decode fp8 K/V cache (use_fp8 + q_dtype set; routes to 3D).
    add(
        "decode_fp8_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        use_fp8=True,
        q_dtype="fp8e4m3",
    )
    add(
        "decode_fp8_d64_b32",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        use_fp8=True,
        q_dtype="fp8e4m3",
    )
    add(
        "decode_n1_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )

    # --- short prefill (max_seqlen_q ~128) ---
    for hd in (64, 128):
        add(
            f"short_prefill_fp16_d{hd}",
            head_size=hd,
            dtype="fp16",
            max_seqlen_q=128,
            num_seqs=2,
            num_query_heads=8,
            num_kv_heads=1,
            max_seqlen_k=1024,
        )
    add(
        "short_prefill_bf16_d128_gqa4",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=128,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=4,
        max_seqlen_k=1024,
    )
    add(
        "short_prefill_sw_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=200,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=1024,
        sliding_window=128,
    )
    add(
        "short_prefill_bf16_d64_b32_gqa8",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=128,
        num_seqs=2,
        num_query_heads=64,
        num_kv_heads=8,
        max_seqlen_k=1024,
    )
    add(
        "short_prefill_fp16_d256_b64",
        head_size=256,
        block_size=64,
        dtype="fp16",
        max_seqlen_q=128,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=1024,
    )
    add(
        "short_prefill_n1_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=128,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=1024,
    )

    # --- long prefill (max_seqlen_q ~2048) ---
    # gfx942 fp16 flash branch: D128 (any BS) and D64 (BS=64), no SW/softcap,
    # num_seqs>=2. (On gfx950 these are the plain wide-K path -- still must
    # build cleanly via the default branch.)
    add(
        "long_prefill_flash_fp16_d128",
        head_size=128,
        block_size=16,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_flash_fp16_d128_b32",
        head_size=128,
        block_size=32,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_flash_fp16_d64_b64",
        head_size=64,
        block_size=64,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    # gfx942 D64 fp16 flash with a paged block_size of 16/32 (e.g. a vLLM-style
    # 16-token KV cache). The flash regime needs T in {64,128}; before the
    # _select_2d_tile_size fix these yielded T=block_size and the spec validator
    # rejected the build on the selected-2D path. Pin both (red->green).
    add(
        "long_prefill_flash_fp16_d64_b16",
        head_size=64,
        block_size=16,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_flash_fp16_d64_b32",
        head_size=64,
        block_size=32,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    # bf16 transposed "combo" branch cohort: HD=64, BS=32, NQH=64/NKV=8
    # (num_queries_per_kv=8), long prefill, multi-batch. With/without sinks.
    add(
        "long_prefill_combo_bf16",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=64,
        num_kv_heads=8,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_combo_bf16_sinks",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=64,
        num_kv_heads=8,
        max_seqlen_k=2048,
        use_sinks=True,
    )
    # combo cohort by GQA-8 *ratio* but NOT the 64/8 absolute head count, e.g.
    # a tensor-parallel-sharded GQA-8 model (16/2). _enable_combo_2d fires on
    # the ratio; the gfx950 use_fast_paged_kv_desc validator wants absolute
    # 64/8, so this shape used to crash the gfx950 selected-2D path until the
    # spec builder gated the fast descriptor to 64/8.
    add(
        "long_prefill_combo_bf16_tp_sharded_16x2",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=2,
        max_seqlen_k=2048,
    )
    # plain default long prefill (single-seq and multi-batch), fp16/bf16.
    add(
        "long_prefill_default_fp16_d128_n1",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_default_bf16_d128_n4",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=4,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_default_fp16_d256",
        head_size=256,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    # long-prefill toggles: sliding window, softcap, alibi, qq_bias.
    add(
        "long_prefill_sw_bf16_d128",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        sliding_window=128,
    )
    add(
        "long_prefill_softcap_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        softcap=30.0,
    )
    add(
        "long_prefill_alibi_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_alibi=True,
    )
    add(
        "long_prefill_qqbias_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_qq_bias=True,
    )
    # fp8 K/V long prefill (multi-batch) -- exercises the fp8 plumbing on the
    # prefill side as well as decode.
    add(
        "long_prefill_fp8_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_fp8=True,
        q_dtype="fp8e4m3",
    )
    add(
        "long_prefill_fp8_sw_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_fp8=True,
        q_dtype="fp8e4m3",
        sliding_window=128,
    )

    return cfgs


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    def test_mfma_atom_catalog_lane_layouts(self):
        b = IRBuilder("check_atoms")
        lane = b.const_i32(33)

        a16 = mfma_atom("f16", 16, 16, 32)
        self.assertEqual((a16.a_per_lane, a16.b_per_lane, a16.c_per_lane), (8, 8, 4))
        r16, c16 = a16.lane_to_output(b, lane, 2)
        self.assertEqual(r16.type.name, "i32")
        self.assertEqual(c16.type.name, "i32")

        a32 = mfma_atom("f16", 32, 32, 16)
        self.assertEqual((a32.a_per_lane, a32.b_per_lane, a32.c_per_lane), (8, 8, 16))
        r32, c32 = a32.lane_to_output(b, lane, 9)
        self.assertEqual(r32.type.name, "i32")
        self.assertEqual(c32.type.name, "i32")

        a4 = mfma_atom("f16", 4, 4, 4)
        self.assertEqual((a4.a_per_lane, a4.b_per_lane, a4.c_per_lane), (4, 4, 4))

    def test_wmma_atom_rdna_support(self):
        """RDNA WMMA atom: wave32 metadata, layout delegated to the MmaOp SSOT,
        a single packed (non-f32) fragment load, and a wave32 WarpGrid path."""
        from rocke.core.arch import ArchTarget
        from rocke.helpers import (
            WmmaAtom,
            load_wmma_fragment,
            make_global_view,
            store_wmma_acc,
            wmma_atom,
        )

        atom = WmmaAtom.f16_16x16x16()
        self.assertEqual(atom.family, "wmma")
        self.assertEqual(atom.wave_size, 32)
        self.assertEqual(
            (atom.a_per_lane, atom.b_per_lane, atom.c_per_lane), (16, 16, 8)
        )
        self.assertEqual(wmma_atom("f16", 16, 16, 16), atom)

        # The atom's lane layout is the SAME object the arch-target exposes
        # (single source of truth — no duplicated lane math).
        op = ArchTarget.from_gfx("gfx1151").mma.by_op_id("wmma_f32_16x16x16_f16")
        self.assertIsNotNone(op)
        self.assertEqual(atom.a_layout().frag_len, op.a_layout().frag_len)
        self.assertEqual(atom.c_layout().wave_size, op.c_layout().wave_size)

        # emit routes through b.mma -> a <8 x f32> accumulator result.
        b = IRBuilder("wmma_emit")
        af = b.zero_vec_f32(16)
        bf = b.zero_vec_f32(16)
        acc = atom.zero_acc(b)
        res = atom.emit(b, af, bf, acc)
        self.assertEqual(res.type.count, 8)
        self.assertEqual(res.type.elem.name, "f32")

        # Fragment load is ONE packed <16 x half> global_load_vN (align 32),
        # NOT an f32-promoted load_tile — the issue-bound parity guarantee.
        b2 = IRBuilder("wmma_frag")
        q = b2.param("Q", PtrType(F16, "global"))
        sq = b2.param("sq", I32)
        view = make_global_view(q, shape=(256, 128), strides=(sq, 1), dtype=F16)
        lane = b2.mod(b2.thread_id_x(), b2.const_i32(32))
        win = view.tile(lengths=(16, 16), origin=(b2.const_i32(0), b2.const_i32(0)))
        frag = load_wmma_fragment(b2, win, atom, lane, role="a", k_offset=32)
        self.assertEqual(frag.type.count, 16)
        self.assertEqual(frag.type.elem.name, "f16")
        loads = [o for o in self._kernel_ops(b2.kernel) if "global_load_vN" in o.name]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0].attrs["vec"], 16)
        self.assertEqual(loads[0].attrs["align"], 32)

        # store_wmma_acc scatters c_per_lane scalars via the c_layout map.
        b3 = IRBuilder("wmma_store")
        o = b3.param("O", PtrType(F16, "global"))
        so = b3.param("so", I32)
        oview = make_global_view(o, shape=(256, 128), strides=(so, 1), dtype=F16)
        lane3 = b3.mod(b3.thread_id_x(), b3.const_i32(32))
        owin = oview.tile(lengths=(16, 16), origin=(b3.const_i32(0), b3.const_i32(0)))
        store_wmma_acc(b3, owin, atom, lane3, b3.zero_vec_f32(8), col_offset=16)
        stores = [o for o in self._kernel_ops(b3.kernel) if "global_store" in o.name]
        self.assertEqual(len(stores), 8)

        # WarpGrid takes the WMMA atom (wave32) via from_atom and from_wmma.
        g = WarpGrid.from_wmma(
            atom, tile_m=16, tile_n=16, tile_k=16, warp_m=1, warp_n=1
        )
        self.assertEqual(g.wave_size, 32)
        self.assertEqual(g.block_size, 32)
        self.assertEqual(
            WarpGrid.from_atom(
                atom, tile_m=16, tile_n=16, tile_k=16, warp_m=1, warp_n=1
            ).wave_size,
            32,
        )

    def test_wmma_tensor_tile_api(self):
        """WmmaTensor + load_wmma_tile/wmma_mma/store_wmma_tile are 1:1 packed
        wrappers: the tile API must emit the SAME single packed load / single
        mma / single vector-mul as the raw fragment path (issue-bound parity)."""
        from rocke.helpers import (
            WmmaAtom,
            WmmaTensor,
            load_wmma_tile,
            make_global_view,
            store_wmma_tile,
            wmma_mma,
        )

        atom = WmmaAtom.f16_16x16x16()

        # load_wmma_tile -> a WmmaTensor whose packed value is one <16 x half>
        # global_load_vN (NOT an f32-promoted load), identical to the raw path.
        b = IRBuilder("wmma_tile_load")
        q = b.param("Q", PtrType(F16, "global"))
        sq = b.param("sq", I32)
        view = make_global_view(q, shape=(256, 128), strides=(sq, 1), dtype=F16)
        lane = b.mod(b.thread_id_x(), b.const_i32(32))
        win = view.tile(lengths=(16, 16), origin=(b.const_i32(0), b.const_i32(0)))
        qt = load_wmma_tile(b, win, atom, lane, role="a", k_offset=32)
        self.assertIsInstance(qt, WmmaTensor)
        self.assertEqual(qt.role, "a")
        self.assertEqual(qt.value.type.count, 16)
        self.assertEqual(qt.value.type.elem.name, "f16")
        loads = [o for o in self._kernel_ops(b.kernel) if "global_load_vN" in o.name]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0].attrs["vec"], 16)

        # wmma_mma -> one b.mma; the result tile is a <8 x f32> accumulator.
        b2 = IRBuilder("wmma_tile_mma")
        at = WmmaTensor(atom, "a", b2.zero_vec_f32(16), "gfx1151")
        bt = WmmaTensor(atom, "b", b2.zero_vec_f32(16), "gfx1151")
        acc = WmmaTensor.zero_acc(b2, atom)
        out = wmma_mma(b2, at, bt, acc)
        self.assertEqual(out.role, "c")
        self.assertEqual(out.value.type.count, 8)
        self.assertEqual(out.value.type.elem.name, "f32")
        mmas = [o for o in self._kernel_ops(b2.kernel) if o.name == "tile.mma"]
        self.assertEqual(len(mmas), 1)

        # scale -> exactly one vector mul (the online-softmax rescale).
        b3 = IRBuilder("wmma_tile_scale")
        acc3 = WmmaTensor.zero_acc(b3, atom)
        alpha = b3.zero_vec_f32(8)
        acc3.scale(b3, alpha)
        muls = [o for o in self._kernel_ops(b3.kernel) if o.name == "vector.mul"]
        self.assertEqual(len(muls), 1)

        # store_wmma_tile -> c_per_lane scalar stores via the c_layout map.
        b4 = IRBuilder("wmma_tile_store")
        o = b4.param("O", PtrType(F16, "global"))
        so = b4.param("so", I32)
        oview = make_global_view(o, shape=(256, 128), strides=(so, 1), dtype=F16)
        lane4 = b4.mod(b4.thread_id_x(), b4.const_i32(32))
        owin = oview.tile(lengths=(16, 16), origin=(b4.const_i32(0), b4.const_i32(0)))
        store_wmma_tile(b4, owin, WmmaTensor.zero_acc(b4, atom), lane4, col_offset=16)
        stores = [o for o in self._kernel_ops(b4.kernel) if "global_store" in o.name]
        self.assertEqual(len(stores), 8)

    @staticmethod
    def _kernel_ops(kernel):
        """Flatten all ops in a kernel (best-effort across body/region attrs)."""
        out = []
        region = getattr(kernel, "body", None) or getattr(kernel, "region", None)
        blocks = getattr(region, "blocks", [region]) if region is not None else []
        for blk in blocks:
            out.extend(getattr(blk, "ops", []))
        return out

    def test_warp_grid_binding_emits_decomp(self):
        b = IRBuilder("check_grid")
        grid = WarpGrid(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        )
        self.assertFalse(grid.is_bound)
        self.assertEqual(grid.block_size, 256)
        bound = grid.bind(b)
        self.assertTrue(bound.is_bound)
        self.assertEqual(bound.tid.type.name, "i32")
        self.assertEqual(bound.lane.type.name, "i32")
        self.assertEqual(bound.warp_m_idx.type.name, "i32")

    def test_coalesced_loader_choose_vec(self):
        # 128x32 tile = 4096 halves, 256 threads: vec=8 -> 512 vecs ≥ 256 ✓
        self.assertEqual(
            CoalescedTileLoader.choose_vec(tile_rows=128, tile_cols=32, block_size=256),
            8,
        )
        # 64x64 with 256 threads: 4096 halves, vec=8 -> 512 > 256 ✓
        self.assertEqual(
            CoalescedTileLoader.choose_vec(tile_rows=64, tile_cols=64, block_size=256),
            8,
        )
        # 32x16 with 256 threads = 512 halves, vec=8 -> 64 < 256: fail.
        # vec=4: 128 < 256: fail. vec=2: 256 == 256 ✓.
        self.assertEqual(
            CoalescedTileLoader.choose_vec(tile_rows=32, tile_cols=16, block_size=256),
            2,
        )

    def test_async_loader_choose_dwords(self):
        # 128 halves wide => has to be multiple of 8 halves (dwords=4):
        # tile_rows=64, tile_cols=128, threads=256: chunks = 64*128/8 = 1024 ≥ 256 ✓
        self.assertEqual(
            AsyncTileLoader.choose_dwords(tile_rows=64, tile_cols=128, block_size=256),
            4,
        )

    def test_lds_layout_async_guardrails(self):
        LdsLayout.packed_async(64).validate_for_async()
        with self.assertRaises(ValueError):
            LdsLayout.padded_k(64, 8).validate_for_async()
        with self.assertRaises(ValueError):
            LdsLayout(logical_cols=64, swizzle="xor").validate_for_async()

    def test_schedule_policy_emits_expected_hints(self):
        b = IRBuilder("sched_smoke")
        policy = SchedulePolicy.for_pipeline("compv4")
        policy.emit_after_mfma_step(b, ds_read_count=2, mfma_count=4)
        self.assertEqual(
            [op.name for op in b.kernel.body.ops],
            ["tile.sched_group_barrier", "tile.sched_group_barrier"],
        )

    def test_software_pipeline_static_ping_pong(self):
        b = IRBuilder("pipeline_smoke")
        seen = []
        pipe = SoftwarePipeline(num_iters=3, double_buffer=True, wait_vmcnt=True)
        out = pipe.run_ping_pong(
            b,
            buffers=[("A0", "B0"), ("A1", "B1")],
            initial_state=0,
            issue_load=lambda it, buf: seen.append(("load", it, buf)),
            compute=lambda it, buf, state: state + it + (0 if buf[0] == "A0" else 10),
        )
        self.assertEqual(out, 13)
        self.assertEqual(seen[0], ("load", 0, ("A0", "B0")))
        self.assertTrue(any(op.name == "tile.s_waitcnt" for op in b.kernel.body.ops))

    def test_make_gemm_manifest_round_trip(self):
        spec = UniversalGemmSpec(
            name="m_test",
            tile=TileSpec(
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="default"),
        )
        kernel = build_universal_gemm(spec)
        artifact = compile_kernel(kernel, capture_ir_text=False)
        manifest = make_gemm_manifest(
            artifact=artifact,
            block_m=spec.tile.tile_m,
            block_n=spec.tile.tile_n,
            block_k=spec.tile.tile_k,
            threads_per_block=spec.block_size,
        )
        self.assertEqual(manifest["kind"], "gemm_fp16")
        self.assertEqual(manifest["kernel_name"], artifact.kernel_name)
        self.assertGreater(manifest["hsaco_bytes"], 0)
        # The schema-id is stable.
        self.assertEqual(manifest["schema"], "ck.dsl.example.manifest/v1")

    def test_attention_config_selectors_match_aiter_rules(self):
        cfg = select_2d_config(
            block_size=64,
            head_size=128,
            sliding_window=0,
            all_decode=True,
            max_seqlen_q=1,
            max_seqlen_k=512,
            num_queries_per_kv=4,
            num_2d_prgms=32,
        )
        self.assertEqual(cfg.TILE_SIZE, 64)
        self.assertEqual(cfg.BLOCK_M, 16)
        self.assertEqual(cfg.BLOCK_Q, 4)
        self.assertTrue(
            use_2d_kernel(
                head_size=128,
                sliding_window=0,
                all_decode=True,
                max_seqlen_q=1,
                max_seqlen_k=512,
                target_num_prgms=480,
                num_2d_prgms=32,
            )
        )
        attn_cfg, reduce_cfg = select_3d_config(
            head_size=128,
            block_size=64,
            element_size=2,
            max_seqlen_k=2048,
            target_num_prgms=480,
            num_2d_prgms=32,
        )
        self.assertGreaterEqual(attn_cfg.NUM_SEGMENTS_PER_SEQ, 8)
        self.assertIn(reduce_cfg.num_warps, (1, 2))

    def test_unified_attention_support_gate_is_explicit(self):
        p = UnifiedAttentionProblem(
            total_q=128,
            num_seqs=3,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            max_seqlen_q=129,
            max_seqlen_k=2011,
            dtype="fp16",
        )
        ok, reason = supports_native_unified_attention(p)
        self.assertTrue(ok)
        self.assertIn("supported", reason)

    def test_unified_attention_scalar_kernels_compile(self):
        p = UnifiedAttentionProblem(
            total_q=3,
            num_seqs=1,
            num_query_heads=4,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            max_seqlen_q=3,
            max_seqlen_k=16,
            dtype="fp16",
        )
        kernels = [
            build_unified_attention_2d(UnifiedAttention2DSpec(p)),
            build_unified_attention_3d(UnifiedAttention3DSpec(p, num_segments=8)),
            build_unified_attention_reduce(
                UnifiedAttentionReduceSpec(p, num_segments=8)
            ),
        ]
        for k in kernels:
            ll = lower_kernel_to_llvm(k)
            self.assertIn("define amdgpu_kernel void", ll)
            self.assertIn("@llvm.exp2.f32", ll)

    def test_unified_attention_2d_tiled_kernel_compiles(self):
        """The production tiled kernel emits async DMA + MFMA + ds_bpermute."""
        from rocke.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        spec = UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
        )
        k = build_unified_attention_2d_tiled(spec)
        ll = lower_kernel_to_llvm(k)
        # Async DMA for K/V should be emitted.
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
        # MFMA atoms for QK (16x16x32) and PV (16x16x16 since T=16 < 32).
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll)
        # Cross-lane softmax reduction. The 16-lane intra-row-group
        # butterfly lowers to ``ds_swizzle`` SWAP mode rather than
        # ``ds_bpermute`` for the row-group masks ≤ 16. (Larger butterfly
        # stages — e.g. cross-half xor 32 in the 32x32 path — still use
        # bpermute, but they're not present in this default decode spec.)
        self.assertIn("@llvm.amdgcn.ds.swizzle", ll)
        # NaN-guard select on neg_inf row max.
        self.assertIn("0xFFF0000000000000", ll)
        # `qq_bias_stride_0` is the very last kernel param.
        self.assertIn("i32 %qq_bias_stride_0", ll)

    def test_unified_attention_2d_tiled_half_local_pv_compiles(self):
        """The R4 half-local PV variant emits 32x32 MFMA with its suffixes."""
        from rocke.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        spec = UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=True,
            sliding_window=0,
            has_softcap=False,
            num_seqs=284,
            num_warps=4,
            waves_per_eu=2,
            tile_size=64,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=True,
            use_transposed_half_local_pv=True,
        )
        k = build_unified_attention_2d_tiled(spec)
        ll = lower_kernel_to_llvm(k)
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.bf16", ll)
        self.assertIn("_s1_", k.name)
        self.assertIn("_mask1_", k.name)
        self.assertIn("_hlpv", k.name)
        self.assertNotIn('"amdgpu-agpr-alloc"', ll)

        agpr_spec = UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=True,
            sliding_window=0,
            has_softcap=False,
            num_seqs=284,
            num_warps=4,
            waves_per_eu=2,
            tile_size=64,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=True,
            use_transposed_half_local_pv=True,
            use_agpr_alloc_zero=True,
        )
        agpr_k = build_unified_attention_2d_tiled(agpr_spec)
        agpr_ll = lower_kernel_to_llvm(agpr_k)
        self.assertIn("_agpr0", agpr_k.name)
        self.assertIn('"amdgpu-agpr-alloc"="0,0"', agpr_ll)

    def test_unified_attention_2d_tiled_alibi_qq_bias(self):
        """ALiBi/QQ-bias variants emit sitofp + masked global load with clamp."""
        from rocke.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        # Both ALiBi and QQ-bias on.
        spec = UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            use_alibi=True,
            use_qq_bias=True,
        )
        k = build_unified_attention_2d_tiled(spec)
        ll = lower_kernel_to_llvm(k)
        # ALiBi adds a position->f32 conversion (sitofp), since col_abs and
        # context_len are i32 and slope * (col-ctx) needs f32 arithmetic.
        self.assertIn("sitofp i32", ll)
        # Both biases must use the OOB-safe `masked_global_load` clamp: a
        # `select` feeding into the GEP that selects the index, before the
        # actual `global_load_dword`. The lowered IR contains a `select i1`
        # picking between the real index and a 0 constant, then a `load
        # float, ptr ...` for the bias element.
        self.assertIn("select i1", ll)
        # QQ-bias kernel name suffix.
        self.assertIn("_qqb", ll)
        # ALiBi kernel name suffix.
        self.assertIn("_alibi", ll)

    def test_unified_attention_3d_tiled_kernel_compiles(self):
        from rocke.instances import (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
        )

        seg = build_unified_attention_3d_tiled(
            UnifiedAttention3DTiledSpec(
                head_size=128,
                block_size=16,
                num_query_heads=16,
                num_kv_heads=2,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_segments=128,
                num_seqs=4,
            )
        )
        seg_ll = lower_kernel_to_llvm(seg)
        # Segment kernel must use the async DMA + transpose-read PV operand
        # path and emit MFMA atoms.
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", seg_ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", seg_ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", seg_ll)
        # Workspace writes for per-segment m / l / acc.
        self.assertIn("segm_output_ptr", seg_ll)
        self.assertIn("segm_max_ptr", seg_ll)
        self.assertIn("segm_expsum_ptr", seg_ll)
        red = build_unified_attention_reduce_tiled(
            UnifiedAttentionReduceTiledSpec(
                head_size=128,
                num_query_heads=16,
                num_kv_heads=2,
                dtype="fp16",
                num_segments=128,
            )
        )
        red_ll = lower_kernel_to_llvm(red)
        # Reduce must compute exp2-weighted segment combine and use NaN-safe
        # factor (`-inf - overall_max -> 0`).
        self.assertIn("@llvm.exp2.f32", red_ll)
        self.assertIn("fcmp ogt", red_ll)

    def test_unified_attention_3d_tiled_alibi_qq_bias(self):
        """ALiBi/QQ-bias on the 3D segment kernel emit the same primitives."""
        from rocke.instances import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        seg = build_unified_attention_3d_tiled(
            UnifiedAttention3DTiledSpec(
                head_size=128,
                block_size=16,
                num_query_heads=16,
                num_kv_heads=2,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_segments=128,
                num_seqs=4,
                use_alibi=True,
                use_qq_bias=True,
            )
        )
        ll = lower_kernel_to_llvm(seg)
        self.assertIn("sitofp i32", ll)
        self.assertIn("select i1", ll)
        # `qq_bias_stride_0` is the last kernel param.
        self.assertIn("i32 %qq_bias_stride_0", ll)
        # Both ALiBi and QQ-bias kernel-name suffixes show up.
        self.assertIn("_alibi", ll)
        self.assertIn("_qqb", ll)

    def test_attention_3d_workspace_size_matches_shapes(self):
        p = UnifiedAttentionProblem(
            total_q=3,
            num_seqs=2,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=2,
            max_seqlen_k=4096,
            dtype="fp16",
        )
        # AITER's 3D selector chooses 128 segments for this shape.
        # segm_output: 3 * 16 * 128 * 128 f32
        # segm_max/expsum: 2 * (3 * 16 * 128) f32
        expected = (3 * 16 * 128 * 128 + 2 * 3 * 16 * 128) * 4
        self.assertEqual(attention_3d_workspace_nbytes(p), expected)

    def test_fp8_cvt_intrinsic_lowering(self):
        """fp8e4m3->f32 conversion lowers to llvm.amdgcn.cvt.f32.fp8."""
        from rocke.core.ir import (
            IRBuilder,
            PtrType,
            F32,
            FP8E4M3,
        )

        b = IRBuilder("fp8_cvt_smoke")
        out_p = b.param("out", PtrType(F32, "global"), align=4)
        src_p = b.param("src", PtrType(FP8E4M3, "global"), align=1)
        tid = b.thread_id_x()
        v8 = b.global_load_fp8e4m3(src_p, tid, align=1)
        v32 = b.cvt_fp8_to_f32(v8)
        b.global_store(out_p, tid, v32, align=4)
        b.ret()
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("@llvm.amdgcn.cvt.f32.fp8", ll)
        # The conversion uses a zext-to-i32 + lane-0 intrinsic call.
        self.assertIn("zext i8", ll)

    def test_tiled_2d_support_gate_rejects_unsupported(self):
        from rocke.instances import supports_tiled_2d

        base = dict(
            head_size=128,
            block_size=16,
            dtype="fp16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
        )
        ok_fp16, _ = supports_tiled_2d(**base)
        self.assertTrue(ok_fp16)
        # head_size in {64, 128, 256}, block_size in {16, 32, 64}, dtype=bf16,
        # alibi, qq_bias all supported.
        for accept in [
            dict(head_size=256),
            dict(head_size=64),
            dict(block_size=32),
            dict(block_size=64),
            dict(dtype="bf16"),
            dict(use_alibi=True),
            dict(use_qq_bias=True),
        ]:
            kwargs = dict(base)
            kwargs.update(accept)
            ok, reason = supports_tiled_2d(**kwargs)
            self.assertTrue(ok, msg=f"expected accept for {accept}, got: {reason}")
        # FP8, unsupported head_size, and unsupported block_size still gated.
        for override in [
            dict(head_size=72),
            dict(block_size=24),
            dict(use_fp8=True),
        ]:
            kwargs = dict(base)
            kwargs.update(override)
            ok, reason = supports_tiled_2d(**kwargs)
            self.assertFalse(ok, msg=f"expected reject for {override}, got: {reason}")
            self.assertTrue(reason)

    def test_tiled_2d_dispatch_gate_accepts_block_m_per_warp_per_arch(self):
        """Regression: the shared dispatch entry
        ``supports_native_unified_attention_tiled`` forwards
        ``block_m_per_warp`` to the per-arch ``supports_tiled_2d`` gate. Every
        routed arch's gate must accept that kwarg. gfx950 previously raised
        ``TypeError`` here (its gate signature lacked the parameter), which
        broke the gfx950 SDPA dispatch path for ``backend in {tiled, auto}``.
        """
        from unittest import mock
        from rocke.instances import supports_native_unified_attention_tiled
        import rocke.instances.common.attention_unified as au

        p = UnifiedAttentionProblem(
            total_q=128,
            num_seqs=3,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=129,
            max_seqlen_k=2011,
            dtype="fp16",
        )
        # Pin the routed arch so the test is deterministic on any host (the
        # default fallback is gfx950, which is exactly the broken path).
        for arch in ("gfx950", "gfx942"):
            with mock.patch.object(au, "_resolve_attention_arch", return_value=arch):
                # Must not raise (the regression was a TypeError on the kwarg).
                ok, reason = supports_native_unified_attention_tiled(p)
                self.assertIsInstance(ok, bool)
                self.assertIsInstance(reason, str)
                self.assertTrue(
                    ok, msg=f"{arch}: D128 fp16 GQA should be supported, got: {reason}"
                )

    def test_attention_dispatch_matrix_no_signature_drift_per_arch(self):
        """Broad per-arch drift net over the curated attention problem matrix.

        For every ``(arch, config)`` pair this drives the production dispatch
        surface exactly as the runtime selector would -- ``select_path``, the
        three ``supports_native_*`` gates, and (when a gate accepts) the matching
        arch spec builder -- and asserts there is no *signature drift*: each gate
        returns a ``(bool, str)`` tuple and each spec builder returns an instance
        of the arch's own spec class. This is the single test that would have
        caught BOTH regressions that motivated this matrix: the gfx950
        ``supports_tiled_2d`` missing-kwarg ``TypeError`` and the gfx950
        ``UnifiedAttention3DTiledSpec`` missing-field ``TypeError``.

        Deliberately NOT asserted: per-shape accept/reject verdicts (those
        legitimately differ by arch, e.g. HD=256 is unsupported on the gfx942 2D
        path).

        On the **selected** path (``select_path()``), gate-True must imply the
        matching spec builder constructs WITHOUT raising at all -- that is the
        production call path, so a ``ValueError`` there is a real crash (this is
        how the gfx942 combo-on-bf16 bug was caught: the arch-agnostic flag
        choosers handed the gfx942 2D spec gfx950-only knobs; now arch-gated to
        gfx950). On the **non-selected** builder we still exercise the call but
        tolerate a ``ValueError`` (arch-legitimate rejection of a path the
        runtime would not pick) while still failing on a ``TypeError`` (the
        signature-drift class).
        """
        from unittest import mock
        import rocke.instances.common.attention_unified as au
        from rocke.instances import (
            supports_native_unified_attention,
            supports_native_unified_attention_tiled,
            supports_native_unified_attention_3d_tiled,
        )

        matrix = _attention_problem_matrix()
        self.assertTrue(matrix, "attention problem matrix is empty")
        for arch in ("gfx942", "gfx950"):
            spec_2d_cls = au._tiled_2d_impl(arch)[0]
            spec_3d_cls = au._tiled_3d_impl(arch)[0]
            for label, p in matrix:
                with self.subTest(arch=arch, cfg=label):
                    with mock.patch.object(
                        au, "_resolve_attention_arch", return_value=arch
                    ):
                        path = p.select_path()
                        self.assertIn(path, ("2d", "3d"))
                        for gate in (
                            supports_native_unified_attention,
                            supports_native_unified_attention_tiled,
                            supports_native_unified_attention_3d_tiled,
                        ):
                            ok, reason = gate(p)
                            self.assertIsInstance(ok, bool)
                            self.assertIsInstance(reason, str)
                        ok_2d, _ = supports_native_unified_attention_tiled(p)
                        ok_3d, _ = supports_native_unified_attention_3d_tiled(p)
                        # Selected production path: gate-True => builder must
                        # construct the arch spec with NO exception.
                        if path == "2d" and ok_2d:
                            spec = au._tiled_spec_from_problem(p)
                            self.assertIsInstance(spec, spec_2d_cls)
                        if path == "3d" and ok_3d:
                            spec3 = au._tiled_3d_spec_from_problem(p)
                            self.assertIsInstance(spec3, spec_3d_cls)
                        # Non-selected builder: exercise for cross-path drift.
                        # Tolerate an arch-legitimate ValueError; a TypeError is
                        # the missing-kwarg/field signature drift we guard.
                        if path != "2d" and ok_2d:
                            try:
                                spec = au._tiled_spec_from_problem(p)
                            except ValueError:
                                pass
                            else:
                                self.assertIsInstance(spec, spec_2d_cls)
                        if path != "3d" and ok_3d:
                            try:
                                spec3 = au._tiled_3d_spec_from_problem(p)
                            except ValueError:
                                pass
                            else:
                                self.assertIsInstance(spec3, spec_3d_cls)

    def test_gfx942_l4_num_warps_matches_flash_selector(self):
        """The gfx942 D128 fp16 flash/L4 branch in ``_select_2d_num_warps`` must
        agree with ``_select_gfx942_flash_num_warps`` (what the flash kernel is
        actually built and launched with). The branch is not on the live grid
        path today (every flash site reads the flash selector directly), but
        ``num_warps`` is not part of the JitCache key, so a silent disagreement
        here is a latent wrong-CTA-count trap for any future caller. Pin them
        equal.
        """
        from unittest import mock
        import rocke.instances.common.attention_unified as au

        p = UnifiedAttentionProblem(
            total_q=4096,
            num_seqs=2,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=2048,
            max_seqlen_k=2048,
            dtype="fp16",
        )
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx942"):
            self.assertTrue(
                au._enable_gfx942_l4(p), "shape must be in the gfx942 L4 flash region"
            )
            self.assertEqual(
                au._select_2d_num_warps(p),
                au._select_gfx942_flash_num_warps(p),
            )

    def test_tiled_3d_dispatch_gate_accepts_kwargs_per_arch(self):
        """Regression: the shared dispatch entry
        ``supports_native_unified_attention_3d_tiled`` forwards its kwargs to the
        per-arch ``supports_tiled_3d`` gate, and the auto selector routes decode
        (``max_seqlen_q == 1``) to this 3D split-KV path. Every routed arch's
        gate must accept the forwarded kwargs without raising. This guards the
        gfx950 3D spec-kwarg regression (the ``UnifiedAttention3DTiledSpec``
        missing-field break that took out production decode), mirroring the 2D
        dispatch test one path over.
        """
        from unittest import mock
        import rocke.instances.common.attention_unified as au
        from rocke.instances import supports_native_unified_attention_3d_tiled

        p = UnifiedAttentionProblem(
            total_q=4,
            num_seqs=4,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,  # decode -> routes to the 3D split-KV builder
            max_seqlen_k=4096,
            dtype="fp16",
        )
        for arch in ("gfx950", "gfx942"):
            with mock.patch.object(au, "_resolve_attention_arch", return_value=arch):
                # Must not raise (the regression was a TypeError on the kwarg).
                ok, reason = supports_native_unified_attention_3d_tiled(p)
                self.assertIsInstance(ok, bool)
                self.assertIsInstance(reason, str)
                self.assertTrue(
                    ok,
                    msg=f"{arch}: D128 fp16 GQA decode should route to a "
                    f"supported 3D kernel, got: {reason}",
                )

    def test_tiled_3d_spec_builder_constructs_per_arch(self):
        """Focused guard on the 3D spec builder that broke: a decode problem must
        construct the arch's ``UnifiedAttention3DTiledSpec`` (signature parity)
        on both arches, and the gfx942-only 3D knobs must be inert on gfx950 --
        ``_gfx942_3d_tile_size_override`` is ``None`` and the two
        ``_enable_gfx942_3d_*`` toggles are ``False`` -- pinning the
        ignored-field contract that lets the shared builder pass those kwargs
        unconditionally.
        """
        from unittest import mock
        import rocke.instances.common.attention_unified as au

        p = UnifiedAttentionProblem(
            total_q=4,
            num_seqs=4,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=4096,
            dtype="fp16",
        )
        for arch in ("gfx942", "gfx950"):
            with mock.patch.object(au, "_resolve_attention_arch", return_value=arch):
                spec = au._tiled_3d_spec_from_problem(p)
                self.assertIsInstance(spec, au._tiled_3d_impl(arch)[0])
        # The gfx942-only 3D knobs are inert on gfx950 (ignored-field contract).
        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx950"):
            self.assertIsNone(au._gfx942_3d_tile_size_override(p))
            self.assertFalse(au._enable_gfx942_3d_invariant_hoist(p))
            self.assertFalse(au._enable_gfx942_3d_wide_kv_load(p))

    def test_tiled_2d_spec_builder_constructs_per_arch_all_branches(self):
        """Drive ``_tiled_spec_from_problem`` through its three branches and
        assert each constructs the arch's 2D spec without signature drift:

        * gfx942 fp16 long-prefill D128 & D64 (no SW/softcap, num_seqs>=2) ->
          the gfx942 fp16-flash branch (~13 flash flags),
        * bf16 multi-batch long-prefill HD64/BS32/GQA8 -> the bf16 transposed
          "combo" branch (gfx950, where the combo flags are valid),
        * a plain fp16 shape -> the default branch.

        The flash branch only fires on gfx942 and the combo branch only builds
        cleanly on gfx950; this asserts construction + correct arch spec type for
        whichever branch fires on the relevant arch, pinning the largest
        (~25-flag) silent surface.
        """
        from unittest import mock
        import rocke.instances.common.attention_unified as au

        def problem(**kw):
            base = dict(
                total_q=0,
                num_seqs=2,
                num_query_heads=8,
                num_kv_heads=1,
                head_size=128,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="fp16",
            )
            base.update(kw)
            if base["total_q"] == 0:
                base["total_q"] = base["num_seqs"] * base["max_seqlen_q"]
            return UnifiedAttentionProblem(**base)

        # (label, problem, arches to drive). Flash -> gfx942; combo -> gfx950;
        # default -> both.
        cases = [
            (
                "flash_d128",
                problem(head_size=128, block_size=16, dtype="fp16"),
                ("gfx942",),
            ),
            (
                "flash_d64",
                problem(head_size=64, block_size=64, dtype="fp16"),
                ("gfx942",),
            ),
            (
                "combo_bf16",
                problem(
                    head_size=64,
                    block_size=32,
                    dtype="bf16",
                    num_query_heads=64,
                    num_kv_heads=8,
                ),
                ("gfx950",),
            ),
            (
                "default_fp16",
                problem(head_size=128, dtype="fp16", num_seqs=1),
                ("gfx942", "gfx950"),
            ),
        ]
        for label, p, arches in cases:
            for arch in arches:
                with self.subTest(cfg=label, arch=arch):
                    with mock.patch.object(
                        au, "_resolve_attention_arch", return_value=arch
                    ):
                        spec = au._tiled_spec_from_problem(p)
                        self.assertIsInstance(spec, au._tiled_2d_impl(arch)[0])

    def test_tiled_3d_support_gate_rejects_unsupported(self):
        """Mirror of ``test_tiled_2d_support_gate_rejects_unsupported`` for the
        per-arch ``supports_tiled_3d`` gate. Both arches share the same
        accept/reject contract, so the cases are driven for each arch via the
        ``arch=`` kwarg.
        """
        from rocke.instances import supports_tiled_3d

        base = dict(
            head_size=128,
            block_size=16,
            dtype="fp16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
        )
        for arch in ("gfx942", "gfx950"):
            ok_fp16, _ = supports_tiled_3d(arch=arch, **base)
            self.assertTrue(ok_fp16, msg=f"{arch}: base fp16 D128 should accept")
            # head_size {64,128,256}, block_size {16,32,64}, bf16, and every
            # num_queries_per_kv that divides BLOCK_M=16 are supported.
            for accept in [
                dict(head_size=256),
                dict(head_size=64),
                dict(block_size=32),
                dict(block_size=64),
                dict(dtype="bf16"),
                dict(num_queries_per_kv=1),
                dict(num_queries_per_kv=4),
                dict(num_queries_per_kv=16),
                dict(use_alibi=True),
                dict(use_qq_bias=True),
            ]:
                kwargs = dict(base)
                kwargs.update(accept)
                ok, reason = supports_tiled_3d(arch=arch, **kwargs)
                self.assertTrue(
                    ok, msg=f"{arch}: expected accept for {accept}, got: {reason}"
                )
            # Bad head_size, bad block_size, fp8-without-kv_storage, and an
            # unsupported dtype are gated on both arches.
            for override in [
                dict(head_size=72),
                dict(block_size=24),
                dict(use_fp8=True),
                dict(dtype="fp8"),
            ]:
                kwargs = dict(base)
                kwargs.update(override)
                ok, reason = supports_tiled_3d(arch=arch, **kwargs)
                self.assertFalse(
                    ok, msg=f"{arch}: expected reject for {override}, got: {reason}"
                )
                self.assertTrue(reason)

    def test_unified_attention_3d_tiled_kernel_compiles_gfx942(self):
        """gfx942 analogue of ``test_unified_attention_3d_tiled_kernel_compiles``
        (which exercises the default gfx950 arch only). Builds the gfx942 3D
        split-KV segment kernel and asserts the gfx942-specific narrow primitives
        it ACTUALLY emits -- the 16x16x16 MFMA atom and the 1-DWORD async
        global->LDS DMA (``raw.ptr.buffer.load.lds``) -- and that it does NOT emit
        the gfx950 wide 16x16x32 MFMA. Pure codegen, no GPU.
        """
        from unittest import mock
        import rocke.instances.common.attention_unified as au

        with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx942"):
            (
                UnifiedAttention3DTiledSpec,
                UnifiedAttentionReduceTiledSpec,
                build_unified_attention_3d_tiled,
                build_unified_attention_reduce_tiled,
                _supports_tiled_3d,
            ) = au._tiled_3d_impl("gfx942")
            seg = build_unified_attention_3d_tiled(
                UnifiedAttention3DTiledSpec(
                    head_size=128,
                    block_size=16,
                    num_query_heads=16,
                    num_kv_heads=2,
                    dtype="fp16",
                    use_sinks=False,
                    sliding_window=0,
                    has_softcap=False,
                    num_segments=128,
                    num_seqs=4,
                )
            )
            seg_ll = lower_kernel_to_llvm(seg)
            # The arch-dispatched reduce kernel must also build on gfx942.
            red = build_unified_attention_reduce_tiled(
                UnifiedAttentionReduceTiledSpec(
                    head_size=128,
                    num_query_heads=16,
                    num_kv_heads=2,
                    dtype="fp16",
                    num_segments=128,
                )
            )
            red_ll = lower_kernel_to_llvm(red)
        # gfx942 narrow 3D path: 16x16x16 MFMA + 1-DWORD async DMA KV feed.
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", seg_ll)
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", seg_ll)
        # Must NOT use the gfx950 wide-K 16x16x32 MFMA.
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", seg_ll)
        # Workspace writes for per-segment m / l / acc.
        self.assertIn("segm_output_ptr", seg_ll)
        self.assertIn("segm_max_ptr", seg_ll)
        self.assertIn("segm_expsum_ptr", seg_ll)
        # Reduce kernel: exp2-weighted segment combine + NaN-safe factor.
        self.assertIn("@llvm.exp2.f32", red_ll)
        self.assertIn("fcmp ogt", red_ll)


# ---------------------------------------------------------------------
# Instances (end-to-end build smoke)
# ---------------------------------------------------------------------


class TestInstances(unittest.TestCase):
    def test_universal_gemm_compv4_cshuffle_builds(self):
        spec = UniversalGemmSpec(
            name="uni_smoke",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        )
        kernel = build_universal_gemm(spec)
        ll = lower_kernel_to_llvm(kernel)
        # 256 threads in a block, max_workgroup_size attribute set.
        self.assertIn("define amdgpu_kernel void", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)

    def test_implicit_gemm_conv_builds(self):
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
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
        kernel = build_implicit_gemm_conv(spec)
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)
        # The buffer rsrc DW3 flag-word must be 0x00027000, not 0 — the
        # critical correctness fix from the bake-off debugging session.
        self.assertIn("159744", ll)  # 0x27000 = 159744

    def test_implicit_gemm_async_rejects_padded_lds(self):
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
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
            async_dma=True,
            lds_k_pad=8,
        )
        with self.assertRaises(ValueError):
            build_implicit_gemm_conv(spec)

    def test_implicit_gemm_async_emits_pingpong_artifacts(self):
        """Async-DMA conv must emit the ping-pong scaffolding.

        We sanity-check that the async-DMA path produces:
          * ``raw_ptr_buffer_load_lds`` intrinsics (the actual async DMA).
          * ``s_setprio`` bookends from the interwave scheduler (the
            high-prio / low-prio pair around each MFMA group).
          * ``s_waitcnt`` with a partial ``vmcnt`` (overlap of the
            next iter's load with this iter's compute), encoded as
            anything other than the all-zero drain.
          * The ``s_barrier`` count is at least 2 per K-iter (one
            ``sync_lds_only`` before each issue + one after the wait),
            confirming the pong-hazard guard is in place.
        """
        prob = ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
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
            async_dma=True,
        )
        kernel = build_implicit_gemm_conv(spec)
        ll = lower_kernel_to_llvm(kernel)
        # 1) The async DMA intrinsic itself is emitted.
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
        # 2) Interwave ping-pong: setprio bookends. We expect *both*
        # the high (1) and low (0) prio settings from the interwave
        # SchedulePolicy.
        self.assertIn("@llvm.amdgcn.s.setprio(i16 1)", ll)
        self.assertIn("@llvm.amdgcn.s.setprio(i16 0)", ll)
        # 3) At least one barrier per K-iter for the ABA-hazard guard
        # (K_iters = 576 / 64 = 9 in this shape, but counting exactly
        # is brittle; require >= 8 barriers as a lower bound).
        barrier_count = ll.count("@llvm.amdgcn.s.barrier")
        self.assertGreaterEqual(
            barrier_count,
            8,
            msg=f"expected >= 8 s_barriers for ping-pong, found {barrier_count}",
        )

    def test_direct_conv_16c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=True)
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # K32-folded hot loop emits ONLY the wide 16x16x32 MFMA: S=0/1 fold
        # into one wide atom and the S=2 residual is promoted to a SECOND
        # wide atom (zero-padded upper K) so both atoms on the accumulator
        # are the same width. Mixing a 16x16x16 residual into the same
        # accumulator triggered a cross-width MFMA accumulator hazard that
        # both comgr and hipcc miscompiled at the H-edges; the all-wide fold
        # is bit-correct.
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll)
        # The unfolded (gfx942-capable) path still uses only 16x16x16.
        spec_nf = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=False)
        ll_nf = lower_kernel_to_llvm(build_direct_conv_16c(spec_nf))
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll_nf)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll_nf)

    def test_direct_conv_4c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv4cSpec(problem=prob, block_q=8, block_groups=16)
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # 4x4x4 atom emits one MFMA per (r, s) tile (9 per output row).
        self.assertIn("@llvm.amdgcn.mfma.f32.4x4x4f16", ll)


# ---------------------------------------------------------------------
# Non-GEMM CK Tile counterparts (elementwise / norm / reduce / transpose).
# ---------------------------------------------------------------------


class TestElementwiseInstance(unittest.TestCase):
    def test_unary_builds(self):
        from rocke.instances import ElementwiseSpec, build_elementwise

        for op in ("copy", "neg", "abs", "relu", "exp2", "silu", "gelu_tanh"):
            kernel = build_elementwise(ElementwiseSpec(op=op))
            ll = lower_kernel_to_llvm(kernel)
            self.assertIn("define amdgpu_kernel void", ll)

    def test_binary_builds(self):
        from rocke.instances import ElementwiseSpec, build_elementwise

        for op in ("add", "sub", "mul", "max", "min"):
            kernel = build_elementwise(ElementwiseSpec(op=op))
            ll = lower_kernel_to_llvm(kernel)
            self.assertIn("define amdgpu_kernel void", ll)

    def test_bf16_path_builds(self):
        from rocke.instances import ElementwiseSpec, build_elementwise

        kernel = build_elementwise(ElementwiseSpec(op="add", dtype="bf16"))
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("bfloat", ll)

    def test_rejects_unknown_op(self):
        from rocke.instances import ElementwiseSpec, build_elementwise

        with self.assertRaises(ValueError):
            build_elementwise(ElementwiseSpec(op="bogus"))


class TestLayerNormInstance(unittest.TestCase):
    def test_builds_with_save_mean(self):
        from rocke.instances import LayerNorm2DSpec, build_layernorm2d

        spec = LayerNorm2DSpec(n_per_block=4096, save_mean_invstd=True)
        kernel = build_layernorm2d(spec)
        ll = lower_kernel_to_llvm(kernel)
        # The reduction uses LDS tree (s_barrier) and the rsqrt intrinsic.
        self.assertIn("@llvm.amdgcn.s.barrier", ll)
        self.assertIn("@llvm.amdgcn.rsq.f32", ll)

    def test_rejects_unaligned_n(self):
        from rocke.instances import LayerNorm2DSpec, build_layernorm2d

        spec = LayerNorm2DSpec(n_per_block=3072, block_size=256, vec=8)
        with self.assertRaises(ValueError):
            build_layernorm2d(spec)


class TestRMSNormInstance(unittest.TestCase):
    def test_builds(self):
        from rocke.instances import RMSNorm2DSpec, build_rmsnorm2d

        kernel = build_rmsnorm2d(RMSNorm2DSpec(n_per_block=4096))
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("@llvm.amdgcn.rsq.f32", ll)


class TestReduceInstance(unittest.TestCase):
    def test_sum_max_mean_builds(self):
        from rocke.instances import Reduce2DSpec, build_reduce2d

        for op in ("sum", "max", "mean"):
            kernel = build_reduce2d(Reduce2DSpec(n_per_block=4096, op=op))
            ll = lower_kernel_to_llvm(kernel)
            self.assertIn("define amdgpu_kernel void", ll)


class TestTransposeInstance(unittest.TestCase):
    def test_builds(self):
        from rocke.instances import Transpose2DSpec, build_transpose2d

        kernel = build_transpose2d(Transpose2DSpec())
        ll = lower_kernel_to_llvm(kernel)
        # Transpose uses a global memory load->LDS->global store and a
        # workgroup barrier between the two phases.
        self.assertIn("@llvm.amdgcn.s.barrier", ll)
        self.assertIn("addrspace(3)", ll)

    def test_rejects_oversize_tile(self):
        from rocke.instances import Transpose2DSpec, build_transpose2d

        # 128x128/2 = 8192 threads > 1024 cap.
        with self.assertRaises(ValueError):
            build_transpose2d(Transpose2DSpec(tile_m=128, tile_n=128, vec=2))


class TestBatchedGemmInstance(unittest.TestCase):
    def test_builds_with_strides(self):
        from rocke.instances import BatchedGemmSpec, build_batched_gemm

        spec = BatchedGemmSpec(
            name="bgemm_smoke",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
        )
        kernel = build_batched_gemm(spec)
        ll = lower_kernel_to_llvm(kernel)
        # The batched form pulls in `block_id_z` for the batch index.
        self.assertIn("@llvm.amdgcn.workgroup.id.z", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)


# ---------------------------------------------------------------------
# CK Tile-inspired helpers (tensor_view / io / reduction / spec).
# ---------------------------------------------------------------------


class TestTensorViewHelper(unittest.TestCase):
    def test_packed_descriptor_strides(self):
        from rocke.helpers import (
            TensorDescriptor,
            make_naive_tensor_descriptor_packed,
        )

        d = make_naive_tensor_descriptor_packed((32, 128, 64), F16)
        self.assertEqual(d.strides, (128 * 64, 64, 1))
        self.assertEqual(d.numel(), 32 * 128 * 64)
        d2 = TensorDescriptor.with_strides((4, 16), (32, 1), F16)
        self.assertEqual(d2.strides, (32, 1))

    def test_runtime_stride_offsets_via_value(self):
        from rocke.helpers import TensorDescriptor

        # The point of the runtime-stride branch: stride entry can be an
        # SSA Value (e.g. for a runtime W). We exercise it through the
        # offset builder and verify the emitted IR contains the
        # expected mul instruction.
        builder = IRBuilder("rt_stride_smoke")
        W = builder.param("W", I32)
        d = TensorDescriptor.with_strides((1, 1), (W, 1), F16)
        # The offset call is the thing under test; we don't read the
        # returned Value because we inspect the lowered IR instead.
        d.offset(builder, [builder.const_i32(3), builder.const_i32(2)])
        ll = lower_kernel_to_llvm(builder.kernel)
        self.assertIn("mul nsw i32", ll)

    def test_tile_window_origin_arithmetic(self):
        from rocke.helpers import make_global_view, make_lds_view

        builder = IRBuilder("tw_origin_smoke")
        builder.kernel.attrs["max_workgroup_size"] = 64
        X = builder.param("X", PtrType(F16, "global"), align=16)
        Y = builder.param("Y", PtrType(F16, "global"), align=16)
        H = builder.param("H", I32)  # noqa: F841 - simulating runtime stride
        x_view = make_global_view(X, shape=(8, 16), dtype=F16)
        y_view = make_global_view(Y, shape=(8, 16), dtype=F16)
        lds_view = make_lds_view(builder, dtype=F16, shape=(8, 16), name_hint="t")
        h0 = builder.const_i32(0)
        w0 = builder.const_i32(0)
        x_tile = x_view.tile(lengths=(8, 16), origin=(h0, w0))
        y_tile = y_view.tile(lengths=(8, 16), origin=(h0, w0))
        lds_tile = lds_view.tile(lengths=(8, 16), origin=(h0, w0))
        # round-trip: load -> LDS -> store
        v = x_tile.load_vec(builder, builder.const_i32(0), builder.const_i32(0), n=8)
        lds_tile.store_vec(
            builder, builder.const_i32(0), builder.const_i32(0), value=v, n=8
        )
        builder.sync()
        v2 = lds_tile.load_vec(builder, builder.const_i32(0), builder.const_i32(0), n=8)
        y_tile.store_vec(
            builder, builder.const_i32(0), builder.const_i32(0), value=v2, n=8
        )
        ll = lower_kernel_to_llvm(builder.kernel)
        # Expect LDS round-trip via addrspace(3) + a barrier.
        self.assertIn("addrspace(3)", ll)
        self.assertIn("@llvm.amdgcn.s.barrier", ll)


class TestIOHelper(unittest.TestCase):
    def test_dtype_string_aliases(self):
        from rocke.helpers import io_ir_type
        from rocke.core.ir import BF16

        self.assertIs(io_ir_type("f16"), F16)
        self.assertIs(io_ir_type("fp16"), F16)
        self.assertIs(io_ir_type("bf16"), BF16)
        with self.assertRaises(ValueError):
            io_ir_type("fp32")


class TestReductionHelper(unittest.TestCase):
    def test_block_lds_reduce_emits_barrier_chain(self):
        from rocke.helpers import block_lds_reduce, make_lds_view

        b = IRBuilder("red_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        tid = b.thread_id_x()
        lds = make_lds_view(b, dtype=F32_ir(), shape=(64,), name_hint="r").base
        total = block_lds_reduce(
            b, b.const_f32(1.0), lds, tid, block_size=64, combine="sum"
        )
        b.global_store(b.param("Y", PtrType(F32_ir(), "global")), b.const_i32(0), total)
        ll = lower_kernel_to_llvm(b.kernel)
        # Six barrier stages for block_size=64 (log2(64)=6).
        self.assertGreaterEqual(ll.count("@llvm.amdgcn.s.barrier"), 6)


def F32_ir():
    from rocke.core.ir import F32

    return F32


class TestTensorCoordinate(unittest.TestCase):
    def test_make_and_move_emit_offset_chain(self):
        from rocke.helpers import (
            make_naive_tensor_view_packed,
            make_tensor_coordinate,
            move_tensor_coordinate,
        )

        b = IRBuilder("coord_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        view = make_naive_tensor_view_packed(X, shape=(64, 128), dtype=F16)
        c0 = make_tensor_coordinate(b, view.desc, [b.const_i32(2), b.const_i32(3)])
        # The cache should be populated eagerly.
        self.assertTrue(c0.has_cached_offset)
        # Move by (1, 0) -> incremental update emits one fresh add.
        c1 = move_tensor_coordinate(b, c0, [b.const_i32(1), b.const_i32(0)])
        self.assertTrue(c1.has_cached_offset)
        # Index has been bumped.
        self.assertNotEqual(c0.index, c1.index)
        # Lower the kernel to make sure the chain is well-formed.
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("define amdgpu_kernel void", ll)

    def test_move_rank_mismatch_raises(self):
        from rocke.helpers import (
            make_naive_tensor_view_packed,
            make_tensor_coordinate,
            move_tensor_coordinate,
        )

        b = IRBuilder("coord_rank")
        X = b.param("X", PtrType(F16, "global"), align=16)
        view = make_naive_tensor_view_packed(X, shape=(4, 4), dtype=F16)
        c = make_tensor_coordinate(b, view.desc, [b.const_i32(0), b.const_i32(0)])
        with self.assertRaises(ValueError):
            move_tensor_coordinate(b, c, [b.const_i32(1)])


class TestWindowLoadStoreMethods(unittest.TestCase):
    def test_window_load_returns_distributed_tensor(self):
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_naive_tensor_view_packed,
            make_static_tile_distribution,
            make_tile_window,
        )

        enc = TileDistributionEncoding(
            Hs=((1, 64, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1, 1),
            Ys2RHs_minor=(0, 2),
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist)

        b = IRBuilder("win_load_method")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        view = make_naive_tensor_view_packed(X, shape=(512,), dtype=F16)
        tile = make_tile_window(view, lengths=(512,), origin=(b.const_i32(0),))
        tid = b.thread_id_x()

        dt = tile.load(b, distribution=dist, ps=[[tid]], traits=traits)
        self.assertEqual(dt.num_elements, dist.num_elements_per_thread)
        self.assertEqual(dt.distribution, dist)

    def test_window_store_emits_global_store(self):
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_naive_tensor_view_packed,
            make_static_distributed_tensor,
            make_static_tile_distribution,
            make_tile_window,
        )

        enc = TileDistributionEncoding(
            Hs=((1, 64, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1, 1),
            Ys2RHs_minor=(0, 2),
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist)

        b = IRBuilder("win_store_method")
        b.kernel.attrs["max_workgroup_size"] = 64
        Y = b.param("Y", PtrType(F16, "global"), align=16)
        view = make_naive_tensor_view_packed(Y, shape=(512,), dtype=F16)
        tile = make_tile_window(view, lengths=(512,), origin=(b.const_i32(0),))
        tid = b.thread_id_x()

        dt = make_static_distributed_tensor(dist, dtype=F16)
        dt.fill(b.const_f32(1.0))
        tile.store(b, dt, ps=[[tid]], traits=traits)

        ll = lower_kernel_to_llvm(b.kernel)
        # Confirm vectorised global store was emitted.
        self.assertIn("store <8 x half>", ll)


class TestBufferView(unittest.TestCase):
    def test_make_buffer_view_emits_buffer_rsrc(self):
        from rocke.helpers import (
            make_buffer_resource,
            make_buffer_view,
            make_tile_window,
        )

        b = IRBuilder("buf_view_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        N_bytes = b.param("N_bytes", I32)
        rsrc = make_buffer_resource(b, X, num_bytes=N_bytes)
        view = make_buffer_view(rsrc, shape=(0,), dtype=F16)
        self.assertEqual(view.addr_space, "buffer")
        self.assertIs(view.buffer, rsrc)
        tile = make_tile_window(view, lengths=(0,), origin=(b.const_i32(0),))
        tile.load_vec(b, b.const_i32(0), n=4)
        ll = lower_kernel_to_llvm(b.kernel)
        # The buffer rsrc construction emits a call to make.buffer.rsrc.
        self.assertIn("llvm.amdgcn.make.buffer.rsrc", ll)
        # And the load emits a raw_ptr_buffer_load intrinsic.
        self.assertIn("raw.ptr.buffer.load", ll)

    def test_buffer_view_rejects_buffer_accessor_for_global(self):
        from rocke.helpers import TensorView, TensorDescriptor

        b = IRBuilder("buf_view_misuse")
        X = b.param("X", PtrType(F16, "global"), align=16)
        view = TensorView(
            base=X, desc=TensorDescriptor.packed((4,), F16), addr_space="global"
        )
        with self.assertRaises(TypeError):
            _ = view.buffer

    def test_buffer_view_scalar_load(self):
        from rocke.helpers import (
            make_buffer_resource,
            make_buffer_view,
            make_tile_window,
        )

        b = IRBuilder("buf_view_scalar")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        N_bytes = b.param("N_bytes", I32)
        rsrc = make_buffer_resource(b, X, num_bytes=N_bytes)
        view = make_buffer_view(rsrc, shape=(0,), dtype=F16)
        tile = make_tile_window(view, lengths=(0,), origin=(b.const_i32(0),))
        tile.load_scalar(b, b.const_i32(0))
        ll = lower_kernel_to_llvm(b.kernel)
        # raw_ptr_buffer_load.u16 is the scalar half buffer load.
        self.assertIn("raw.ptr.buffer.load", ll)


class TestLlvmFlavorPolymorphism(unittest.TestCase):
    """The LLVM lowering must work against both LLVM 20 (ROCm 7.0/7.1)
    and LLVM 21+ (ROCm 7.2; ships LLVM 22) toolchains. The two
    affected intrinsic families are:

    1. ``llvm.amdgcn.make.buffer.rsrc``: renamed to ``...p8.p1`` and
       widened ``num_records`` from i32 to i64 in LLVM 21+ (PR #126828).
    2. ``llvm.amdgcn.mfma.f32.*.{fp8,bf8}.{fp8,bf8}``: A/B operand types
       collapsed from ``<2 x i32>`` to scalar ``i64`` in LLVM 21+.
    """

    def _buffer_rsrc_kernel(self):
        from rocke.helpers import (
            make_buffer_resource,
            make_buffer_view,
            make_tile_window,
        )

        b = IRBuilder("flavor_buf_rsrc")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        N_bytes = b.param("N_bytes", I32)
        rsrc = make_buffer_resource(b, X, num_bytes=N_bytes)
        view = make_buffer_view(rsrc, shape=(0,), dtype=F16)
        tile = make_tile_window(view, lengths=(0,), origin=(b.const_i32(0),))
        tile.load_vec(b, b.const_i32(0), n=4)
        return b.kernel

    def _fp8_mfma_kernel(self):
        from rocke.instances import BlockScaleGemmSpec
        from rocke.instances.common.block_scale_gemm import build_block_scale_gemm

        return build_block_scale_gemm(
            BlockScaleGemmSpec(
                M=32,
                N=32,
                K=64,
                block_tile_m=16,
                block_tile_n=16,
                quant_mode="abquant",
                mantissa_dtype="fp8e4m3",
                group_size_mnk=(1, 1, 64),
            )
        )

    def test_buffer_rsrc_llvm20_emits_p1_with_i32_num_records(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM20

        ll = lower_kernel_to_llvm(
            self._buffer_rsrc_kernel(), llvm_flavor=LLVM_FLAVOR_LLVM20
        )
        self.assertIn(
            "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
            "ptr addrspace(1) nocapture readnone, i16, i32, i32)",
            ll,
        )
        self.assertIn("i32 %N_bytes, i32 159744)", ll)
        self.assertNotIn("make.buffer.rsrc.p8.p1", ll)

    def test_buffer_rsrc_llvm22_emits_p8_p1_with_i64_num_records(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM22

        ll = lower_kernel_to_llvm(
            self._buffer_rsrc_kernel(), llvm_flavor=LLVM_FLAVOR_LLVM22
        )
        self.assertIn(
            "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
            "ptr addrspace(1) nocapture readnone, i16, i64, i32)",
            ll,
        )
        self.assertIn("zext i32 %N_bytes to i64", ll)
        # The legacy ``.p1`` declare must NOT appear; mixing would
        # produce conflicting declarations in the same module.
        self.assertNotIn("@llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1)", ll)

    def test_buffer_rsrc_llvm22_passes_i64_num_bytes_without_zext(self):
        """``i64`` ``num_bytes`` must reach the intrinsic directly --
        buffers larger than 4 GiB rely on it; a silent zext-then-trunc
        would land KV-cache tail reads at zero."""
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM22

        b = IRBuilder("flavor_buf_rsrc_i64")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        N_bytes = b.param("N_bytes", I64)
        b.buffer_rsrc(X, N_bytes)
        ll = lower_kernel_to_llvm(b.kernel, llvm_flavor=LLVM_FLAVOR_LLVM22)
        self.assertNotIn("zext", ll)
        self.assertIn("i64 %N_bytes, i32 159744)", ll)

    def test_mfma_fp8_llvm20_uses_v2i32_operands(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM20

        ll = lower_kernel_to_llvm(
            self._fp8_mfma_kernel(), llvm_flavor=LLVM_FLAVOR_LLVM20
        )
        self.assertIn(
            "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
            "<2 x i32>, <2 x i32>, <4 x float>, "
            "i32 immarg, i32 immarg, i32 immarg)",
            ll,
        )
        self.assertIn("bitcast <8 x i8>", ll)
        self.assertIn("to <2 x i32>", ll)
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64", ll)

    def test_mfma_fp8_llvm22_uses_i64_operands(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM22

        ll = lower_kernel_to_llvm(
            self._fp8_mfma_kernel(), llvm_flavor=LLVM_FLAVOR_LLVM22
        )
        self.assertIn(
            "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
            "i64, i64, <4 x float>, "
            "i32 immarg, i32 immarg, i32 immarg)",
            ll,
        )
        self.assertIn("bitcast <8 x i8>", ll)
        self.assertIn(
            " = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 ",
            ll,
        )
        self.assertNotIn("mfma.f32.16x16x32.fp8.fp8(<2 x i32>", ll)

    def test_unknown_flavor_raises(self):
        with self.assertRaises(ValueError):
            lower_kernel_to_llvm(
                self._buffer_rsrc_kernel(), llvm_flavor="not-a-real-flavor"
            )


class TestTransformsBridge(unittest.TestCase):
    def test_view_from_transforms_descriptor(self):
        from rocke.helpers import (
            make_tile_window,
            view_from_transforms_descriptor,
        )
        from rocke.helpers.transforms import TensorDescriptor as RichTensorDescriptor

        b = IRBuilder("bridge_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        rich = RichTensorDescriptor.naive("A", lengths=[16, 16])
        view = view_from_transforms_descriptor(X, rich)
        self.assertEqual(view.addr_space, "global")
        self.assertEqual(view.dtype.name, "f16")
        # The wrapped view should still expose tile() and load_vec().
        window = make_tile_window(
            view, lengths=(1, 1), origin=(b.const_i32(0), b.const_i32(0))
        )
        v = window.load_vec(b, b.const_i32(2), b.const_i32(0), n=4)
        self.assertEqual(v.type.elem.name, "f16")


class TestTileDistribution(unittest.TestCase):
    def _encoding(self):
        from rocke.helpers import TileDistributionEncoding

        return TileDistributionEncoding(
            Hs=((4, 256, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1, 1),
            Ys2RHs_minor=(0, 2),
        )

    def test_encoding_lengths(self):
        from rocke.helpers import make_static_tile_distribution

        dist = make_static_tile_distribution(self._encoding())
        self.assertEqual(dist.X_lengths, (8192,))
        self.assertEqual(dist.Y_lengths, (4, 8))
        self.assertEqual(dist.num_elements_per_thread, 32)

    def test_encoding_rejects_duplicate_h_bucket(self):
        from rocke.helpers import TileDistributionEncoding

        with self.assertRaises(ValueError):
            TileDistributionEncoding(
                Hs=((4, 8),),
                # Both P and Y point at the same (1, 0) bucket -> overlap.
                Ps2RHs_major=((1,),),
                Ps2RHs_minor=((0,),),
                Ys2RHs_major=(1,),
                Ys2RHs_minor=(0,),
            )

    def test_encoding_rejects_uncovered_h_bucket(self):
        from rocke.helpers import TileDistributionEncoding

        with self.assertRaises(ValueError):
            TileDistributionEncoding(
                Hs=((4, 8, 2),),  # 3 H levels
                Ps2RHs_major=((1,),),  # only one P contributor
                Ps2RHs_minor=((1,),),
                Ys2RHs_major=(1,),  # only one Y contributor
                Ys2RHs_minor=(0,),
                # Level 2 is uncovered -> X is not reconstructable.
            )

    def test_encoding_accepts_R_with_full_coverage(self):
        """Rs != () is now supported; an R bucket needs a P or Y."""
        from rocke.helpers import (
            TileDistributionEncoding,
            make_static_tile_distribution,
        )

        enc = TileDistributionEncoding(
            Rs=(2,),
            Hs=((4, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((0,),),
            Ys2RHs_major=(1, 0),
            Ys2RHs_minor=(1, 0),
        )
        self.assertTrue(enc.has_replication)
        dist = make_static_tile_distribution(enc)
        # Y_lengths picks up R length for the R-mapped Y.
        self.assertEqual(dist.Y_lengths, (8, 2))
        self.assertEqual(dist.num_elements_per_thread, 16)

    def test_encoding_rejects_uncovered_R(self):
        from rocke.helpers import TileDistributionEncoding

        with self.assertRaises(ValueError):
            TileDistributionEncoding(
                Rs=(2,),
                Hs=((4,),),
                Ps2RHs_major=((1,),),
                Ps2RHs_minor=((0,),),
                Ys2RHs_major=(),
                Ys2RHs_minor=(),
            )


class TestStaticDistributedTensor(unittest.TestCase):
    def test_set_get_fill_sweep(self):
        from rocke.helpers import (
            TileDistributionEncoding,
            make_static_distributed_tensor,
            make_static_tile_distribution,
        )

        enc = TileDistributionEncoding(
            Hs=((2, 4),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((0,),),
            Ys2RHs_major=(1,),
            Ys2RHs_minor=(1,),
        )
        dist = make_static_tile_distribution(enc)
        b = IRBuilder("dt_smoke")
        dt = make_static_distributed_tensor(dist, dtype=F16)
        self.assertEqual(dt.num_elements, 4)
        dt.fill(b.const_f32(7.0))
        self.assertEqual(dt.get((0,)), dt.get((1,)))
        # sweep can mutate via returned value.
        dt.sweep(lambda y, _v: b.const_f32(float(y[0])))
        # Now slot y=2 should hold 2.0.
        self.assertNotEqual(dt.get((0,)), dt.get((2,)))


class TestLoadStoreTraits(unittest.TestCase):
    def test_vector_width_picks_innermost_y(self):
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_static_tile_distribution,
        )

        enc = TileDistributionEncoding(
            Hs=((4, 256, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1, 1),
            Ys2RHs_minor=(0, 2),
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist, max_vec=8)
        self.assertEqual(traits.vector_dim_y, 1)
        self.assertEqual(traits.scalar_per_vector, 8)
        # Y_lengths = (4, 8). Non-vector axis is dim 0 with length 4.
        self.assertEqual(traits.num_access, 4)
        bases = list(traits.iterate_accesses(snake=False))
        # Vector dim is pinned at 0 in every emitted base.
        self.assertTrue(all(y[1] == 0 for y in bases))
        self.assertEqual([y[0] for y in bases], [0, 1, 2, 3])

    def test_vector_width_cap(self):
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_static_tile_distribution,
        )

        enc = TileDistributionEncoding(
            Hs=((2, 16),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((0,),),
            Ys2RHs_major=(1,),
            Ys2RHs_minor=(1,),
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist, max_vec=4)
        self.assertEqual(traits.scalar_per_vector, 4)  # capped from 16

    def test_smart_picker_finds_non_innermost_stride1_y(self):
        """Picker should not blindly pick the last Y; if Y0 has stride 1
        and Y1 doesn't, vector_dim_y must be 0."""
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_static_tile_distribution,
        )

        # Y0 -> innermost level of X0 (stride 1, len 4)
        # Y1 -> outermost level of X0 (stride 256*4 = 1024)
        enc = TileDistributionEncoding(
            Hs=((8, 256, 4),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1, 1),
            Ys2RHs_minor=(2, 0),
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist)
        self.assertEqual(traits.vector_dim_y, 0)
        self.assertEqual(traits.scalar_per_vector, 4)

    def test_smart_picker_prefers_largest_stride1_y(self):
        """When multiple Ys have stride 1, pick the one with the largest length."""
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_static_tile_distribution,
        )

        # Both Ys mapped to innermost levels of their respective X dims.
        # Y0 has length 8, Y1 has length 16; picker should choose Y1.
        enc = TileDistributionEncoding(
            Hs=((4, 8), (4, 16)),
            Ps2RHs_major=((1, 2),),
            Ps2RHs_minor=((0, 0),),
            Ys2RHs_major=(1, 2),
            Ys2RHs_minor=(1, 1),
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist)
        self.assertEqual(traits.vector_dim_y, 1)
        self.assertEqual(traits.scalar_per_vector, 8)  # 16 capped to max_vec=8

    def test_smart_picker_falls_back_to_scalar(self):
        """If no Y has stride 1 (degenerate encoding), pick scalar path."""
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_static_tile_distribution,
        )

        # Only one Y, and it maps to a non-innermost level.
        enc = TileDistributionEncoding(
            Hs=((4, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1,),
            Ys2RHs_minor=(0,),  # outer level (stride 8)
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist)
        self.assertEqual(traits.scalar_per_vector, 1)

    def test_multi_axis_snake_adjacency(self):
        """The full multi-axis snake guarantees |diff|=1 between
        consecutive emitted access bases."""
        from rocke.helpers import (
            TileDistributionEncoding,
            make_load_store_traits,
            make_static_tile_distribution,
        )

        # Three non-vector axes (Y1, Y2, Y3); vector dim Y0.
        enc = TileDistributionEncoding(
            Hs=((4, 3, 2, 256, 4),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((3,),),  # P -> level 3 (256 lanes)
            Ys2RHs_major=(1, 1, 1, 1),
            Ys2RHs_minor=(4, 2, 1, 0),  # Y0 inner (vec=4); Y1 len 2; Y2 len 3; Y3 len 4
        )
        dist = make_static_tile_distribution(enc)
        traits = make_load_store_traits(dist)
        self.assertEqual(traits.vector_dim_y, 0)
        order = list(traits.iterate_accesses(snake=True))
        # Verify adjacency: each step differs by 1 in exactly one
        # non-vector axis.
        for a, b in zip(order, order[1:]):
            non_vec = [
                (i, abs(ai - bi))
                for i, (ai, bi) in enumerate(zip(a, b))
                if i != traits.vector_dim_y
            ]
            total_diff = sum(d for _, d in non_vec)
            self.assertEqual(
                total_diff, 1, msg=f"snake adjacency broken between {a} and {b}"
            )


class TestSweepHelper(unittest.TestCase):
    def test_sweep_row_chunks_invokes_body_per_chunk(self):
        from rocke.helpers import (
            make_naive_tensor_view_packed,
            make_tile_window,
            sweep_row_chunks,
        )

        b = IRBuilder("sweep_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        N = 256
        view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=F16)
        tile = make_tile_window(
            view, lengths=(1, N), origin=(b.const_i32(0), b.const_i32(0))
        )
        tid = b.thread_id_x()
        seen = []

        def body(n_off, x_scalars):
            seen.append((n_off, x_scalars))

        res = sweep_row_chunks(
            b,
            tile,
            tid=tid,
            block_size=64,
            vec=8,
            elems_per_thread=16,
            body=body,
            cache=True,
        )
        # 16 elems/thread / 8 vec = 2 chunks; cache holds 16 f32 scalars.
        self.assertEqual(res.chunks_per_thread, 2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(res.cached), 16)
        # Each scalar in `cached` is a fresh SSA Value.
        self.assertEqual(len({id(v) for v in res.cached}), 16)

    def test_pass2_row_chunks_writes_expected_vec_count(self):
        from rocke.helpers import (
            make_naive_tensor_view_packed,
            make_tile_window,
            pass2_row_chunks,
        )

        b = IRBuilder("pass2_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        Y = b.param("Y", PtrType(F16, "global"), align=16)
        view = make_naive_tensor_view_packed(Y, shape=(1, 256), dtype=F16)
        tile = make_tile_window(
            view, lengths=(1, 256), origin=(b.const_i32(0), b.const_i32(0))
        )
        tid = b.thread_id_x()
        call_count = [0]

        def body(_n_off, _k, _x_scalars):
            call_count[0] += 1
            return [b.const_f32(0.0)] * 8

        pass2_row_chunks(
            b,
            tile,
            tid=tid,
            block_size=64,
            vec=8,
            elems_per_thread=16,
            body=body,
        )
        self.assertEqual(call_count[0], 2)

    def test_pass2_body_must_return_vec_length(self):
        from rocke.helpers import (
            make_naive_tensor_view_packed,
            make_tile_window,
            pass2_row_chunks,
        )

        b = IRBuilder("pass2_arity_smoke")
        Y = b.param("Y", PtrType(F16, "global"), align=16)
        view = make_naive_tensor_view_packed(Y, shape=(1, 256), dtype=F16)
        tile = make_tile_window(
            view, lengths=(1, 256), origin=(b.const_i32(0), b.const_i32(0))
        )
        tid = b.thread_id_x()

        with self.assertRaises(ValueError):
            pass2_row_chunks(
                b,
                tile,
                tid=tid,
                block_size=64,
                vec=8,
                elems_per_thread=16,
                body=lambda _n, _k, _x: [b.const_f32(0.0)],  # wrong arity
            )


class TestSpecHelper(unittest.TestCase):
    def test_validate_io_dtype_block_vec(self):
        from rocke.helpers import IOSpecRule, validate_io

        ok, _ = validate_io(IOSpecRule("f16", 256, 8, n_per_block=4096))
        self.assertTrue(ok)
        bad, why = validate_io(IOSpecRule("f16", 256, 8, n_per_block=4099))
        self.assertFalse(bad)
        self.assertIn("divisible", why)
        bad, _ = validate_io(IOSpecRule("fp32", 256, 8))
        self.assertFalse(bad)
        bad, _ = validate_io(IOSpecRule("f16", 96, 8))
        self.assertFalse(bad)
        bad, _ = validate_io(IOSpecRule("f16", 256, 3))
        self.assertFalse(bad)

    def test_kernel_name_join_flags_and_drops_empty(self):
        from rocke.helpers import kernel_name_join

        self.assertEqual(
            kernel_name_join(
                "rocke_op", "f16", "", "v8", flags={"smv": True, "n": False}
            ),
            "rocke_op_f16_v8_smv",
        )

    def test_signature_builder_chaining(self):
        from rocke.helpers import SignatureBuilder

        sig = (
            SignatureBuilder()
            .ptr("X", "f16")
            .ptr("Gamma", "bf16")
            .scalar("M", "i32")
            .scalar("eps", "f32")
            .build()
        )
        self.assertEqual(sig[0]["type"], "ptr<f16, global>")
        self.assertEqual(sig[1]["type"], "ptr<bf16, global>")
        self.assertEqual(sig[3]["type"], "f32")

    def test_ceil_div_grid_shapes(self):
        from rocke.helpers import ceil_div_grid

        self.assertEqual(ceil_div_grid((100, 16), (128, 32)), (7, 4, 1))
        self.assertEqual(ceil_div_grid((100, 16), (128, 32), (4, 1)), (7, 4, 4))
        self.assertEqual(ceil_div_grid((512, 1)), (512, 1, 1))


class TestGroupedGemmInstance(unittest.TestCase):
    def test_builds(self):
        from rocke.instances import GroupedGemmSpec, build_grouped_gemm

        spec = GroupedGemmSpec(
            name="ggemm_smoke",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
        )
        kernel = build_grouped_gemm(spec)
        ll = lower_kernel_to_llvm(kernel)
        # The (current) per-group launcher uses the non-batched kernel
        # so there's no block_id_z dependency.
        self.assertIn("define amdgpu_kernel void", ll)

    def test_bf16_dtype_drives_signature_and_lowering(self):
        from rocke.instances import GroupedGemmSpec, build_grouped_gemm
        from rocke.instances.common.grouped_gemm import grouped_gemm_signature

        spec = GroupedGemmSpec(
            name="ggemm_bf16_smoke",
            tile=TileSpec(
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(
                pipeline="mem", epilogue="cshuffle", pad_m=True, pad_n=True
            ),
            dtype="bf16",
        )
        sig = grouped_gemm_signature(spec)
        self.assertEqual(sig[0]["type"], "ptr<bf16, global>")
        self.assertEqual(sig[1]["type"], "ptr<bf16, global>")
        self.assertEqual(sig[2]["type"], "ptr<bf16, global>")

        ll = lower_kernel_to_llvm(build_grouped_gemm(spec))
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.bf16", ll)
        self.assertIn("load <8 x bfloat>", ll)
        self.assertIn("store <4 x bfloat>", ll)


class TestCdnaPrimitives(unittest.TestCase):
    """Coverage for the AMDGPU/CDNA-specific primitives added in
    :mod:`rocke.helpers.grid`, :mod:`rocke.helpers.schedule`,
    :mod:`rocke.helpers.layouts`, plus the new IR primitives
    ``pin_sgpr`` / ``to_sgpr_u32`` / ``wave_all`` / ``sync_half_block``.
    """

    # ---- XOR-based LDS swizzles (canonical CK Tile shared-tile layouts) ----
    def test_xor_swizzle_matches_canonical_st_16x32(self):
        from rocke.helpers.layouts import LdsLayout

        def reference(off: int) -> int:
            return off ^ (((off % 1024) >> 9) << 5)

        layout = LdsLayout.xor_swizzled(tile_rows=16, tile_cols=32, elem_bytes=2)
        for off in range(0, 16 * 32 * 2, 7):
            self.assertEqual(
                layout.apply_swizzle_bytes(off),
                reference(off),
                msg=f"swizzle mismatch at off={off}",
            )

    def test_xor_swizzle_matches_canonical_st_32x32_two_stage(self):
        from rocke.helpers.layouts import LdsLayout

        def reference(off: int) -> int:
            return off ^ (((off % 1024) >> 9) << 5) ^ (((off % 2048) >> 10) << 4)

        layout = LdsLayout.xor_swizzled(tile_rows=32, tile_cols=32, elem_bytes=2)
        for off in range(0, 32 * 32 * 2, 13):
            self.assertEqual(layout.apply_swizzle_bytes(off), reference(off))

    def test_xor_swizzle_unknown_shape_raises(self):
        from rocke.helpers.layouts import LdsLayout

        with self.assertRaises(ValueError):
            LdsLayout.xor_swizzled(tile_rows=24, tile_cols=24, elem_bytes=2)

    def test_xor_swizzle_validate_requires_stages(self):
        from rocke.helpers.layouts import LdsLayout

        # Manually constructing with swizzle='xor' but no stages should fail.
        bad = LdsLayout(logical_cols=32, k_pad=0, swizzle="xor", swizzle_stages=())
        with self.assertRaises(ValueError):
            bad.validate()

    # ---- chiplet_transform_chunked (XCD round-robin reverse) ----
    def test_chiplet_python_reference_matches_closed_form(self):
        from rocke.helpers.grid import python_chiplet_transform_chunked

        # Closed-form chiplet remap: WGs in the first
        # (num_xcds * chunk_size) block round-robin onto XCDs in chunks
        # of `chunk_size` consecutive WGs per XCD.
        def reference(wgid, num_wgs, num_xcds, chunk_size):
            block = num_xcds * chunk_size
            limit = (num_wgs // block) * block
            if wgid >= limit:
                return wgid
            xcd = wgid % num_xcds
            local_pid = wgid // num_xcds
            chunk_idx = local_pid // chunk_size
            pos_in_chunk = local_pid % chunk_size
            return chunk_idx * block + xcd * chunk_size + pos_in_chunk

        for wgid in range(0, 2048):
            self.assertEqual(
                python_chiplet_transform_chunked(
                    wgid, num_wgs=2048, num_xcds=8, chunk_size=64
                ),
                reference(wgid, 2048, 8, 64),
            )

    def test_chiplet_first_8_wgs_collapse_to_one_xcd(self):
        # After remap, WGs 0..7 should all land on the SAME XCD (XCD 0).
        from rocke.helpers.grid import python_chiplet_transform_chunked

        xcds = set()
        for wgid in range(8):
            remapped = python_chiplet_transform_chunked(
                wgid, num_wgs=2048, num_xcds=8, chunk_size=64
            )
            xcds.add(remapped % 8)
        self.assertEqual(
            xcds,
            {0},
            msg="chiplet remap failed: first 8 WGs should all land on XCD 0",
        )

    def test_chiplet_tail_falls_through_unchanged(self):
        # WGs past the last complete (num_xcds * chunk_size) block are
        # passed through unchanged (tail-handling contract).
        from rocke.helpers.grid import python_chiplet_transform_chunked

        num_wgs = 8 * 64 + 7  # one full block plus 7 tail WGs
        for wgid in range(8 * 64, num_wgs):
            self.assertEqual(
                python_chiplet_transform_chunked(
                    wgid, num_wgs=num_wgs, num_xcds=8, chunk_size=64
                ),
                wgid,
            )

    def test_super_tile_swizzle_walks_column_first_inside_group(self):
        from rocke.helpers.grid import python_super_tile_swizzle

        # Group of WGM=4 rows x num_pid_n=4 cols = 16 WGs.
        # WGs 0..3 should walk pid_m = 0..3 at pid_n = 0.
        wgm = 4
        npm = 8
        npn = 4
        for wgid in range(4):
            pid_m, pid_n = python_super_tile_swizzle(
                wgid, num_pid_m=npm, num_pid_n=npn, wgm=wgm
            )
            self.assertEqual((pid_m, pid_n), (wgid, 0))
        # WGs 4..7 should walk pid_m = 0..3 at pid_n = 1.
        for k in range(4):
            pid_m, pid_n = python_super_tile_swizzle(
                4 + k, num_pid_m=npm, num_pid_n=npn, wgm=wgm
            )
            self.assertEqual((pid_m, pid_n), (k, 1))

    # ---- SchedulePolicy modes / sched_barrier helpers ----
    def test_schedule_policy_interwave_emits_setprio_bookends(self):
        from rocke.helpers.schedule import SchedulePolicy
        from rocke.core.ir import IRBuilder, PtrType, F16
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("interwave_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.param("X", PtrType(F16, "global"))
        pol = SchedulePolicy.for_pipeline("interwave")
        pol.emit_prologue(b)
        pol.emit_compute_prologue(b)
        # Fake "compute": just emit a sched_barrier(0)
        b.sched_barrier(0)
        pol.emit_compute_epilogue(b)
        ll = lower_kernel_to_llvm(b.kernel)
        # Both s_setprio(1) and s_setprio(0) must appear.
        self.assertIn("@llvm.amdgcn.s.setprio(i16 1)", ll)
        self.assertIn("@llvm.amdgcn.s.setprio(i16 0)", ll)

    def test_schedule_policy_intrawave_emits_no_setprio_bookends(self):
        from rocke.helpers.schedule import SchedulePolicy
        from rocke.core.ir import IRBuilder, PtrType, F16
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("intrawave_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.param("X", PtrType(F16, "global"))
        pol = SchedulePolicy.for_pipeline("intrawave")
        pol.emit_compute_prologue(b)
        b.sched_barrier(0)
        pol.emit_compute_epilogue(b)
        ll = lower_kernel_to_llvm(b.kernel)
        # Intrawave mode must NOT emit the per-compute setprio bookends
        # (the prologue setprio is a separate hook).
        self.assertNotIn("@llvm.amdgcn.s.setprio", ll)

    def test_schedule_policy_emits_mfma_valu_pair_hints(self):
        from rocke.helpers.schedule import SchedulePolicy
        from rocke.core.ir import IRBuilder, PtrType, F16
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("pairs_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.param("X", PtrType(F16, "global"))
        pol = SchedulePolicy.for_pipeline("compv4")  # emit_hints=True
        pol.emit_mfma_valu_pairs(b, pairs=3, valu_per_pair=2)
        ll = lower_kernel_to_llvm(b.kernel)
        # MFMA=0x008 = i32 8 ; VALU=0x002 = i32 2.
        # `sched.group.barrier(MFMA, 1, 0)` and `sched.group.barrier(VALU, 2, 0)`.
        self.assertEqual(
            ll.count("@llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)"),
            3,
        )
        self.assertEqual(
            ll.count("@llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)"),
            3,
        )

    # ---- pin_sgpr / to_sgpr_u32 ----
    def test_to_sgpr_u32_emits_readfirstlane_and_pin(self):
        from rocke.core.ir import IRBuilder, PtrType, F16, I32
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("sgpr_pin")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.param("X", PtrType(F16, "global"))
        v = b.mul(b.const_i32(7), b.const_i32(11))
        # `to_sgpr_u32` returns the SGPR-pinned value; we don't need to
        # bind it (the test asserts on the lowered IR). The call itself
        # is what triggers the readfirstlane + asm emission.
        b.to_sgpr_u32(v)
        b.param("Out", PtrType(I32, "global"))
        ll = lower_kernel_to_llvm(b.kernel)
        # readfirstlane + matched-constraint SGPR-pin asm.
        self.assertIn("@llvm.amdgcn.readfirstlane.i32", ll)
        # Constraint "=s,0" ties output 0 (SGPR class) to input 0 — the
        # LLVM IR translation of HIP's `asm volatile("" : "+s"(x))`.
        self.assertIn('asm "", "=s,0"(i32', ll)
        # Must NOT be `sideeffect` -- that confuses divergence analysis
        # and silently breaks downstream uniform-value selection.
        self.assertNotIn("asm sideeffect", ll)

    # ---- wave_all / wave_any ----
    def test_wave_all_lowers_to_ballot_eq_minus_one(self):
        from rocke.core.ir import IRBuilder, PtrType, I32
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("wave_all_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        Out = b.param("Out", PtrType(I32, "global"))
        all_t = b.wave_all(b.const_i32(1))
        b.global_store(Out, b.const_i32(0), all_t)
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("@llvm.amdgcn.ballot.i64", ll)
        # Compares ballot to -1 (i.e. all wave64 lanes voted true).
        self.assertIn("icmp eq i64", ll)

    # ---- coherency hints ----
    def test_async_buffer_load_lds_addr_propagates_aux(self):
        from rocke.core.ir import (
            IRBuilder,
            PtrType,
            F16,
            I32,
            CACHE_STREAM,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("coh_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"))
        N = b.param("N_bytes", I32)
        rsrc = b.buffer_rsrc(X, N)
        lds = b.smem_alloc(F16, [64, 8], name_hint="stage")
        lds_base = b.smem_addr_of(lds)
        b.async_buffer_load_lds_addr(
            rsrc,
            lds_base,
            b.const_i32(0),
            b.const_i32(0),
            dwords=4,
            coherency=CACHE_STREAM,
        )
        ll = lower_kernel_to_llvm(b.kernel)
        # Last arg should be `i32 2` (CACHE_STREAM = 2).
        self.assertIn("i32 0, i32 2)", ll)

    def test_coherency_rejects_invalid_aux(self):
        from rocke.core.ir import IRBuilder, PtrType, F16, I32

        b = IRBuilder("coh_bad")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"))
        N = b.param("N_bytes", I32)
        rsrc = b.buffer_rsrc(X, N)
        lds = b.smem_alloc(F16, [64, 8], name_hint="stage")
        lds_base = b.smem_addr_of(lds)
        with self.assertRaises(ValueError):
            b.async_buffer_load_lds_addr(
                rsrc,
                lds_base,
                b.const_i32(0),
                b.const_i32(0),
                dwords=4,
                coherency=7,
            )

    # ---- N-buffer SoftwarePipeline ----
    def test_software_pipeline_quad_buffer_emits_three_prologue_loads(self):
        from rocke.helpers.pipeline import SoftwarePipeline
        from rocke.core.ir import IRBuilder, PtrType, F16

        b = IRBuilder("quadbuf_smoke")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.param("X", PtrType(F16, "global"))

        load_count = [0]
        compute_count = [0]

        def issue_load(it, buf):
            load_count[0] += 1

        def compute(it, buf, state):
            compute_count[0] += 1
            return state

        pipe = SoftwarePipeline(
            num_iters=10,
            num_buffers=4,
            wait_vmcnt=False,
            sync_after_wait=False,
            sync_before_issue=False,
        )
        pipe.run_ping_pong(
            b,
            buffers=[("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")],
            initial_state=None,
            issue_load=issue_load,
            compute=compute,
        )
        # Prologue issues (num_buffers - 1) = 3 loads; steady-state issues
        # `num_iters - (num_buffers - 1)` more = 7. Total = 10.
        self.assertEqual(load_count[0], 10)
        self.assertEqual(compute_count[0], 10)

    def test_software_pipeline_legacy_double_buffer_still_works(self):
        from rocke.helpers.pipeline import SoftwarePipeline
        from rocke.core.ir import IRBuilder, PtrType, F16

        b = IRBuilder("legacy_dbuf")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.param("X", PtrType(F16, "global"))

        cnt = [0]

        def issue_load(it, buf):
            cnt[0] += 1

        def compute(it, buf, state):
            return state

        pipe = SoftwarePipeline(num_iters=5, double_buffer=True)
        pipe.run_ping_pong(
            b,
            buffers=[("a", "b"), ("c", "d")],
            initial_state=None,
            issue_load=issue_load,
            compute=compute,
        )
        self.assertEqual(cnt[0], 5)

    # ---- waves_per_eu launch bound ----
    def test_waves_per_eu_attribute_emitted(self):
        from rocke.core.ir import IRBuilder, PtrType, F16
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("occu_smoke")
        b.kernel.attrs["max_workgroup_size"] = 256
        b.kernel.attrs["waves_per_eu"] = 2
        b.param("X", PtrType(F16, "global"))
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)

    def test_waves_per_eu_tuple_range(self):
        from rocke.core.ir import IRBuilder, PtrType, F16
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("occu_range")
        b.kernel.attrs["max_workgroup_size"] = 256
        b.kernel.attrs["waves_per_eu"] = (2, 4)
        b.param("X", PtrType(F16, "global"))
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn('"amdgpu-waves-per-eu"="2,4"', ll)

    # ---- Instance integration tests ----
    def test_universal_gemm_chiplet_swizzle_emits_remap_math(self):
        """The opt-in ``trait.chiplet_swizzle`` must emit the
        chiplet round-robin reverse plus the WGM super-tile remap
        inside the kernel body.
        """
        from rocke.instances import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            build_universal_gemm,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = UniversalGemmSpec(
            name="ugemm_swz_smoke",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4",
                epilogue="cshuffle",
                chiplet_swizzle=True,
                chiplet_wgm=8,
            ),
        )
        ll = lower_kernel_to_llvm(build_universal_gemm(spec))
        # Chiplet remap: the closed-form formula generates `sdiv` /
        # `srem` by the chunk_size and num_xcds constants.
        self.assertIn("sdiv i32", ll)
        self.assertIn("srem i32", ll)
        # The chiplet+supertile preamble constants for chunk_size=64
        # and (num_xcds * chunk_size)=8*64=512.
        self.assertIn(", 8\n", ll)  # `mod` by num_xcds = 8
        self.assertIn(", 512\n", ll)  # `div` by num_xcds*chunk_size = 512
        # And ``block_id_x`` / ``block_id_y`` are both read (the swizzle
        # flattens the 2D grid first).
        self.assertIn("@llvm.amdgcn.workgroup.id.x", ll)
        self.assertIn("@llvm.amdgcn.workgroup.id.y", ll)

    def test_universal_gemm_waves_per_eu_attr(self):
        from rocke.instances import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            build_universal_gemm,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = UniversalGemmSpec(
            name="ugemm_wpe",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4",
                epilogue="cshuffle",
                waves_per_eu=2,
            ),
        )
        ll = lower_kernel_to_llvm(build_universal_gemm(spec))
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)

    def test_implicit_gemm_conv_chiplet_swizzle_compiles(self):
        from rocke.instances import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        prob = ConvProblem(
            N=4,
            Hi=14,
            Wi=14,
            C=128,
            K=128,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
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
            chiplet_swizzle=True,
            chiplet_wgm=4,
            waves_per_eu=2,
        )
        ll = lower_kernel_to_llvm(build_implicit_gemm_conv(spec))
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", ll)
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)
        # The chiplet swizzle takes both blockIdx.x and blockIdx.y.
        self.assertIn("@llvm.amdgcn.workgroup.id.x", ll)
        self.assertIn("@llvm.amdgcn.workgroup.id.y", ll)

    def test_implicit_gemm_conv_async_uses_sgpr_pinned_lds_base(self):
        """The async-DMA conv must hoist the per-wave LDS base into
        an SGPR via ``to_sgpr_u32`` (``readfirstlane`` + SGPR-pin asm).
        """
        from rocke.instances import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        prob = ConvProblem(
            N=8,
            Hi=56,
            Wi=56,
            C=64,
            K=64,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )
        spec = ImplicitGemmConvSpec(
            problem=prob,
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
            async_dma=True,
        )
        ll = lower_kernel_to_llvm(build_implicit_gemm_conv(spec))
        # readfirstlane for wave-uniform LDS offset.
        self.assertIn("@llvm.amdgcn.readfirstlane.i32", ll)
        # SGPR-pin asm constraint.
        self.assertIn('asm "", "=s,0"(i32', ll)
        # CACHE_STREAM coherency on the async loads (aux i32 = 2).
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)

    def test_attention_tiled_2d_waves_per_eu(self):
        from rocke.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            waves_per_eu=2,
        )
        ll = lower_kernel_to_llvm(build_unified_attention_2d_tiled(spec))
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)

    def test_universal_gemm_bf16_spec_drives_ir(self):
        """Universal GEMM should use DataSpec dtype fields in params,
        LDS allocation, global loads/stores and MFMA dispatch.
        """
        from rocke.instances import (
            DataSpec,
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            build_universal_gemm,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = UniversalGemmSpec(
            name="bf16_gemm_smoke",
            tile=TileSpec(
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="cshuffle"),
            data=DataSpec(
                dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32"
            ),
        )
        ll = lower_kernel_to_llvm(build_universal_gemm(spec))
        self.assertIn("ptr addrspace(1) noalias readonly nocapture align 16 %A", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.bf16", ll)
        self.assertIn("load <8 x bfloat>", ll)
        self.assertIn("store <8 x bfloat>", ll)

    def test_universal_gemm_rejects_unsupported_bf16_atom(self):
        from rocke.instances import (
            DataSpec,
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            build_universal_gemm,
        )

        spec = UniversalGemmSpec(
            name="bf16_bad_atom",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                # 32x32x16 / 32x32x8 bf16 MFMA were added for gfx950 (#8348),
                # so warp_tile_k=16 now BUILDS. warp_tile_k=32 has no bf16
                # 32x32x32 atom in the gfx950 catalog and is still rejected.
                warp_tile_k=32,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            data=DataSpec(
                dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32"
            ),
        )
        with self.assertRaisesRegex(ValueError, "unsupported bf16 warp_tile"):
            build_universal_gemm(spec)


class TestFusionPlanner(unittest.TestCase):
    """CPU-only coverage for the graph-capture fusion planner."""

    def test_explain_matmul_bias_relu_scale(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B, bias):
            return torch.relu((torch.matmul(A, B) + bias) * 0.5)

        info = explain_fn(fn)
        self.assertTrue(info["matched"])
        self.assertEqual(info["a_arg_name"], "A")
        self.assertEqual(info["b_arg_name"], "B")
        self.assertEqual(info["bias_arg_name"], "bias")
        # Normalized post-GEMM execution order: bias, scale, relu.
        self.assertEqual(
            [s.split("_", 1)[0] for s in info["epilogue_ops"]],
            ["bias", "scale0.5", "relu"],
        )

    def test_explain_matmul_only(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B):
            return torch.matmul(A, B)

        info = explain_fn(fn)
        self.assertTrue(info["matched"])
        self.assertEqual(info["bias_arg_name"], None)
        self.assertEqual(info["epilogue_ops"], [])

    def test_explain_unsupported_graph(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.sin(A)

        info = explain_fn(fn)
        self.assertFalse(info["matched"])
        self.assertIn("no registered", info["reason"])

    def test_compile_fn_exposes_plan_without_launching(self):
        import torch
        from rocke.helpers import compile_fn

        def fn(A, B, bias):
            return torch.relu(torch.matmul(A, B) + bias)

        compiled = compile_fn(fn)
        self.assertEqual(compiled.match["pattern"], "matmul_bias_relu")
        self.assertEqual(compiled.match["bias_arg_name"], "bias")

    def test_dtype_to_ir_accepts_torch_aliases(self):
        import torch
        from rocke.core.ir import BF16, F16, F32
        from rocke.helpers import dtype_to_ir

        self.assertEqual(dtype_to_ir(torch.float16), F16)
        self.assertEqual(dtype_to_ir(torch.bfloat16), BF16)
        self.assertEqual(dtype_to_ir(torch.float32), F32)

    def test_bf16_fusion_configs_use_supported_bf16_atoms(self):
        from rocke.core.ir import BF16
        from rocke.helpers import fuse_matmul_bias_relu
        from rocke.helpers.fuse import _make_gemm_configs

        cfgs = _make_gemm_configs(epilogue=fuse_matmul_bias_relu(dtype=BF16))
        self.assertGreater(len(cfgs), 0)
        for cfg in cfgs:
            self.assertEqual(cfg.spec.data.dtype_a, "bf16")
            self.assertEqual(cfg.spec.data.dtype_b, "bf16")
            self.assertEqual(cfg.spec.data.dtype_c, "bf16")
            self.assertEqual(
                (cfg.spec.tile.warp_tile_m, cfg.spec.tile.warp_tile_n),
                (16, 16),
            )


class TestFusionSolverScaffold(unittest.TestCase):
    def _toy_graph(self):
        from rocke.helpers import FusionOp, FusionTensor, build_graph

        tensors = [
            FusionTensor("A", (128, 64), "fp16", is_input=True),
            FusionTensor("B", (64, 128), "fp16", is_input=True),
            FusionTensor("bias", (128,), "fp16", is_input=True, layout="broadcast"),
            FusionTensor("mm", (128, 128), "fp16"),
            FusionTensor("add", (128, 128), "fp16"),
            FusionTensor("out", (128, 128), "fp16", is_output=True),
        ]
        ops = [
            FusionOp("mm0", "matmul", ("A", "B"), ("mm",)),
            FusionOp("add0", "add", ("mm", "bias"), ("add",)),
            FusionOp("relu0", "relu", ("add",), ("out",)),
        ]
        return build_graph(
            tensors=tensors, ops=ops, inputs=("A", "B", "bias"), outputs=("out",)
        )

    def test_fusion_graph_hash_is_stable_and_links_users(self):
        g1 = self._toy_graph()
        g2 = self._toy_graph()
        self.assertEqual(g1.graph_hash(), g2.graph_hash())
        self.assertEqual(g1.tensors["mm"].producer, "mm0")
        self.assertEqual(g1.tensors["mm"].users, ("add0",))

    def test_legalizer_accepts_toy_graph(self):
        from rocke.helpers import FusionLegalizer

        result = FusionLegalizer().check_graph(self._toy_graph())
        self.assertTrue(result.ok, result.reasons)

    def test_legalizer_rejects_bad_matmul_shape(self):
        from rocke.helpers import FusionLegalizer, FusionOp, FusionTensor, build_graph

        g = build_graph(
            tensors=[
                FusionTensor("A", (8, 7), "fp16", is_input=True),
                FusionTensor("B", (9, 8), "fp16", is_input=True),
                FusionTensor("C", (8, 8), "fp16", is_output=True),
            ],
            ops=[FusionOp("mm0", "matmul", ("A", "B"), ("C",))],
            inputs=("A", "B"),
            outputs=("C",),
        )
        result = FusionLegalizer().check_graph(g)
        self.assertFalse(result.ok)
        self.assertIn("K mismatch", result.reasons[0])

    def test_greedy_scheduler_fuses_gemm_epilogue(self):
        from rocke.helpers import GreedyFusionScheduler

        plan = GreedyFusionScheduler().schedule(self._toy_graph())
        self.assertEqual(len(plan.regions), 1)
        self.assertEqual(plan.regions[0].kind, "gemm_epilogue")
        self.assertEqual(plan.regions[0].op_names, ("mm0", "add0", "relu0"))
        self.assertIn("fused", plan.explanation[0])

    def test_workspace_planner_allocates_cross_region_intermediate(self):
        from rocke.helpers import (
            FusionOp,
            FusionRegion,
            FusionTensor,
            WorkspacePlanner,
            build_graph,
        )

        graph = build_graph(
            tensors=[
                FusionTensor("A", (16,), "fp16", is_input=True),
                FusionTensor("tmp", (16,), "fp16"),
                FusionTensor("out", (16,), "fp16", is_output=True),
            ],
            ops=[
                FusionOp("op0", "relu", ("A",), ("tmp",)),
                FusionOp("op1", "add", ("tmp", "A"), ("out",)),
            ],
            inputs=("A",),
            outputs=("out",),
        )
        from rocke.helpers.fusion_ir import FusionPlan

        allocs = WorkspacePlanner().plan(
            FusionPlan(
                graph=graph,
                regions=(
                    FusionRegion("r0", "elementwise", ("op0",), ("A",), ("tmp",)),
                    FusionRegion("r1", "elementwise", ("op1",), ("tmp", "A"), ("out",)),
                ),
            )
        )
        self.assertEqual(len(allocs), 1)
        self.assertEqual(allocs[0].tensor_name, "tmp")
        self.assertEqual(allocs[0].first_region, 0)
        self.assertEqual(allocs[0].last_region, 1)

    def test_default_lowering_registry(self):
        from rocke.helpers import default_lowering_registry

        reg = default_lowering_registry()
        self.assertIsNotNone(reg.get("gemm_epilogue"))
        with self.assertRaises(KeyError):
            reg.require("does_not_exist")

    def test_make_autotune_key_includes_graph_and_arch(self):
        from rocke.helpers import make_autotune_key

        key = make_autotune_key(
            graph_hash="abc",
            shape=(128, 128, 64),
            dtype="fp16",
            layout="RCR",
            arch="gfx950",
            compiler="comgr-7",
            lowerer="gemm_epilogue",
            spec_hash="spec1",
        )
        self.assertEqual(
            key,
            (
                "abc",
                (128, 128, 64),
                "fp16",
                "RCR",
                "gfx950",
                "comgr-7",
                "gemm_epilogue",
                "spec1",
            ),
        )


class TestAutotuner(unittest.TestCase):
    """Coverage for ``helpers/autotune.py``: cache, winner pick, errors."""

    def test_autotune_sweep_picks_minimum(self):
        from rocke.helpers import AutotuneConfig, autotune_sweep

        configs = [
            AutotuneConfig(spec=object(), name=f"c{i}", extra={}) for i in range(5)
        ]
        # Synthetic "ms" = the index + 1, so c0 should win.
        timings = {f"c{i}": float(i + 1) for i in range(5)}

        def bench(cfg):
            return timings[cfg.name]

        winner, results = autotune_sweep(configs, bench_fn=bench)
        self.assertEqual(winner.name, "c0")
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.is_ok for r in results))

    def test_autotune_sweep_excludes_errored_configs(self):
        from rocke.helpers import AutotuneConfig, autotune_sweep

        configs = [
            AutotuneConfig(spec=object(), name="good_fast", extra={}),
            AutotuneConfig(spec=object(), name="errored", extra={}),
            AutotuneConfig(spec=object(), name="good_slow", extra={}),
        ]

        def bench(cfg):
            if cfg.name == "errored":
                raise RuntimeError("oom")
            return 5.0 if cfg.name == "good_fast" else 50.0

        winner, results = autotune_sweep(configs, bench_fn=bench)
        self.assertEqual(winner.name, "good_fast")
        # Errored config recorded but excluded from winner pick.
        self.assertEqual(len(results), 3)
        err_row = next(r for r in results if r.config_name == "errored")
        self.assertFalse(err_row.is_ok)
        self.assertIn("oom", err_row.error)

    def test_autotune_sweep_all_errored_raises(self):
        from rocke.helpers import AutotuneConfig, autotune_sweep

        configs = [
            AutotuneConfig(spec=object(), name="a", extra={}),
            AutotuneConfig(spec=object(), name="b", extra={}),
        ]

        def bench(cfg):
            raise RuntimeError("nope")

        with self.assertRaises(RuntimeError):
            autotune_sweep(configs, bench_fn=bench)

    def test_autotuner_caches_winner_and_skips_resweep(self):
        import tempfile
        import os as _os
        from rocke.helpers import Autotuner, AutotuneConfig

        configs = [
            AutotuneConfig(spec=object(), name=f"c{i}", extra={}) for i in range(3)
        ]
        ms_table = {"c0": 30.0, "c1": 10.0, "c2": 20.0}
        bench_calls = []
        launch_calls = []

        def bench(cfg, **_):
            bench_calls.append(cfg.name)
            return ms_table[cfg.name]

        def launch(cfg, **_):
            launch_calls.append(cfg.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            tuner = Autotuner(
                configs=configs,
                key_fn=lambda *, shape, **_: (shape,),
                bench_fn=bench,
                launch_fn=launch,
                cache_path=cache_path,
                verbose=False,
            )
            # First call -- sweeps all 3 configs.
            tuner(shape=(1024, 1024, 1024))
            self.assertEqual(set(bench_calls), {"c0", "c1", "c2"})
            self.assertEqual(launch_calls, ["c1"])  # c1 is the winner

            bench_calls.clear()
            launch_calls.clear()

            # Second call same shape -- cache hit, no benchmark runs.
            tuner(shape=(1024, 1024, 1024))
            self.assertEqual(bench_calls, [])  # didn't re-sweep
            self.assertEqual(launch_calls, ["c1"])

            # Different shape -- sweeps again.
            tuner(shape=(2048, 2048, 2048))
            self.assertEqual(set(bench_calls), {"c0", "c1", "c2"})

            # Persistence: rebuild Autotuner pointing at same cache
            # file; warm lookup should NOT require any sweep.
            tuner2 = Autotuner(
                configs=configs,
                key_fn=lambda *, shape, **_: (shape,),
                bench_fn=lambda *_, **__: 9999.0,  # would lose if it ran
                launch_fn=launch,
                cache_path=cache_path,
                verbose=False,
            )
            launch_calls.clear()
            tuner2(shape=(1024, 1024, 1024))
            self.assertEqual(launch_calls, ["c1"])  # persisted winner
        finally:
            if _os.path.exists(cache_path):
                _os.unlink(cache_path)

    def test_autotuner_select_returns_winner_without_launch(self):
        from rocke.helpers import Autotuner, AutotuneConfig

        configs = [
            AutotuneConfig(spec=object(), name="slow", extra={}),
            AutotuneConfig(spec=object(), name="fast", extra={}),
        ]
        launch_calls = []
        tuner = Autotuner(
            configs=configs,
            key_fn=lambda **_: (),
            bench_fn=lambda cfg, **_: 1.0 if cfg.name == "fast" else 100.0,
            launch_fn=lambda cfg, **_: launch_calls.append(cfg.name),
            cache_path=None,
            verbose=False,
        )
        winner = tuner.select(foo=1)
        self.assertEqual(winner.name, "fast")
        # `.select` must NOT dispatch a launch.
        self.assertEqual(launch_calls, [])

    def test_autotuner_rejects_duplicate_config_names(self):
        from rocke.helpers import Autotuner, AutotuneConfig

        with self.assertRaises(ValueError):
            Autotuner(
                configs=[
                    AutotuneConfig(spec=object(), name="dup"),
                    AutotuneConfig(spec=object(), name="dup"),
                ],
                key_fn=lambda **_: (),
                bench_fn=lambda *a, **k: 1.0,
                launch_fn=lambda *a, **k: None,
            )

    def test_spec_replace_with_dataclass(self):
        from rocke.helpers import spec_replace
        from rocke.instances import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
        )

        base = UniversalGemmSpec(
            name="base",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        )
        renamed = spec_replace(base, name="renamed")
        self.assertEqual(renamed.name, "renamed")
        self.assertEqual(renamed.tile.tile_m, 128)
        # Original untouched (frozen dataclass).
        self.assertEqual(base.name, "base")


# ---------------------------------------------------------------------
# Extended fusion coverage: epilogue ops, pattern matchers, lowerers,
# workspace materialization, and the validation harness.
# ---------------------------------------------------------------------


class TestExpandedEpilogueOps(unittest.TestCase):
    """Each new EpilogueOp must build legal IR and ``tag()`` cleanly."""

    def _build_epilogue_kernel(self, op):
        """Helper: emit a tiny IR kernel that exercises ``op.apply_element``.

        The kernel does not run; it just confirms the op composes
        with :class:`IRBuilder` and that ``lower_kernel_to_llvm``
        accepts the resulting graph. That's enough to catch
        IR-shape regressions (mismatched types, missing methods)
        without needing a GPU.

        Uses :class:`FusedEpilogue` as the single entry point for
        param declaration so each op's pointer-param is declared
        exactly once (calling ``op.declare_params(b)`` directly here
        would double-declare for residual ops).
        """
        from rocke.core.ir import F16, I32, IRBuilder, PtrType
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.helpers.fuse import FusedEpilogue, dtype_to_ir

        b = IRBuilder(f"epilogue_{op.tag()}")
        b.kernel.attrs["max_workgroup_size"] = 64
        in_p = b.param(
            "X", PtrType(F16, "global"), noalias=True, readonly=True, align=2
        )
        out_p = b.param(
            "Y", PtrType(F16, "global"), noalias=True, writeonly=True, align=2
        )
        m = b.param("M", I32)
        # FusedEpilogue calls op.declare_params once, then records the
        # runtime ``N`` for residual ops to use as their M-stride.
        ep_dtype = dtype_to_ir(op.dtype) if hasattr(op, "dtype") else F16
        fe = FusedEpilogue(ops=(op,), dtype=ep_dtype)
        fe.declare_params(b)
        fe.record_runtime(b, N=m)
        idx = b.thread_id_x()
        x = b.global_load_f16(in_p, idx)
        y = op.apply_element(
            b, x, m=idx, n=idx, elem_idx=0, params=dict(fe._live_params)
        )
        b.global_store(out_p, idx, y, align=2)
        ll = lower_kernel_to_llvm(b.kernel)
        return b.kernel, ll

    def test_gelu_builds_and_lowers(self):
        from rocke.helpers import GELU

        _, ll = self._build_epilogue_kernel(GELU())
        self.assertIn("amdgpu", ll)

    def test_silu_builds_and_lowers(self):
        from rocke.helpers import SiLU

        _, ll = self._build_epilogue_kernel(SiLU())
        self.assertIn("amdgpu", ll)

    def test_clamp_builds_with_lo_hi(self):
        from rocke.helpers import Clamp

        op = Clamp(lo=0.0, hi=6.0)
        kernel, ll = self._build_epilogue_kernel(op)
        self.assertIn("clamp0_6_f16", kernel.name)
        self.assertIn("amdgpu", ll)

    def test_cast_passthrough_same_dtype(self):
        from rocke.core.ir import F16
        from rocke.helpers import Cast

        # Sanity: cast f16->f16 should not change the streaming value
        # nor introduce extra cast ops. We just verify the kernel builds
        # so the same-dtype short-circuit is exercised.
        op = Cast(src_dtype=F16, dst_dtype=F16)
        _, ll = self._build_epilogue_kernel(op)
        self.assertIn("amdgpu", ll)

    def test_residual_add_declares_param_and_lowers(self):
        from rocke.helpers import ResidualAdd

        op = ResidualAdd(param_name="residual_0")
        kernel, ll = self._build_epilogue_kernel(op)
        self.assertIn("resadd_f16", kernel.name)
        # The residual pointer must show up as a kernel param.
        self.assertIn("residual_0", ll)

    def test_residual_mul_declares_param_and_lowers(self):
        from rocke.helpers import ResidualMul

        op = ResidualMul(param_name="residual_mul_0")
        kernel, ll = self._build_epilogue_kernel(op)
        self.assertIn("resmul_f16", kernel.name)
        self.assertIn("residual_mul_0", ll)


class TestExpandedPatternMatchers(unittest.TestCase):
    """Verify the new patterns in ``_PATTERN_TABLE`` match expected fns."""

    def test_explain_matmul_gelu(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B):
            return torch.nn.functional.gelu(torch.matmul(A, B))

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["epilogue_ops"], ["gelu_f16"])
        self.assertEqual(info["bias_arg_name"], None)

    def test_explain_matmul_bias_silu(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B, bias):
            return torch.nn.functional.silu(torch.matmul(A, B) + bias)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(
            [s.split("_", 1)[0] for s in info["epilogue_ops"]],
            ["bias", "silu"],
        )
        self.assertEqual(info["bias_arg_name"], "bias")

    def test_explain_matmul_with_residual(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B, bias, residual):
            return torch.matmul(A, B) + bias + residual

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        # The matcher walks output -> matmul; the LAST add (closest to
        # the output) is consumed first. Either of {bias, residual} can
        # end up in ``bias_arg_name``; what matters is that exactly one
        # of them is the residual and we recognised both placeholders.
        ph_set = {info["bias_arg_name"]}.union(set(info["residual_arg_names"]))
        self.assertEqual(ph_set, {"bias", "residual"})
        self.assertEqual(len(info["residual_arg_names"]), 1)

    def test_explain_pointwise_chain(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.relu(torch.nn.functional.gelu(A))

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertTrue(info["pattern"].startswith("pointwise_"))
        self.assertEqual(info["a_arg_name"], "A")

    def test_explain_pointwise_binary_chain(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B):
            return torch.relu(A + B)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["pattern"], "pointwise_add_relu")
        self.assertEqual(info["a_arg_name"], "A")
        self.assertEqual(info["b_arg_name"], "B")
        self.assertEqual(info["extra_attrs"]["op_kind"], "add")
        self.assertEqual(info["extra_attrs"]["unary_chain"], ("relu",))

    def test_explain_matmul_scale_clamp(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A, B):
            return torch.clamp(torch.matmul(A, B) * 0.25, min=-1.0, max=1.0)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(
            [s.split("_", 1)[0] for s in info["epilogue_ops"]],
            ["scale0.25", "clamp-1"],
        )

    def test_explain_rowwise_reduction_sum(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.sum(A, dim=-1)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["pattern"], "reduce_sum")
        self.assertEqual(info["extra_attrs"]["reduce_op"], "sum")

    def test_explain_rowwise_reduction_mean(self):
        import torch
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.mean(A, dim=-1)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["pattern"], "reduce_mean")
        self.assertEqual(info["extra_attrs"]["reduce_op"], "mean")


class TestLoweringRegistryBuild(unittest.TestCase):
    """End-to-end ``can_lower`` + ``candidates`` + ``build`` smoke tests.

    These tests cover the path from a normalized fusion graph all the
    way through HSACO build for each concrete lowerer. They do NOT
    launch the kernels (no GPU); the goal is to confirm the lowerers
    wire up a real launcher object on every supported region kind.
    """

    def _toy_gemm_graph(self, with_epilogue=True):
        from rocke.helpers import FusionOp, FusionTensor, build_graph

        tensors = [
            FusionTensor("A", (128, 64), "fp16", is_input=True),
            FusionTensor("B", (64, 128), "fp16", is_input=True),
            FusionTensor("bias", (128,), "fp16", is_input=True, layout="broadcast"),
            FusionTensor("mm", (128, 128), "fp16"),
        ]
        ops = [FusionOp("mm0", "matmul", ("A", "B"), ("mm",))]
        if with_epilogue:
            tensors.extend(
                [
                    FusionTensor("add", (128, 128), "fp16"),
                    FusionTensor("out", (128, 128), "fp16", is_output=True),
                ]
            )
            ops.append(FusionOp("add0", "add", ("mm", "bias"), ("add",)))
            ops.append(FusionOp("relu0", "relu", ("add",), ("out",)))
            inputs = ("A", "B", "bias")
            outputs = ("out",)
        else:
            tensors.append(FusionTensor("mm", (128, 128), "fp16", is_output=True))
            inputs = ("A", "B")
            outputs = ("mm",)
        return build_graph(tensors=tensors, ops=ops, inputs=inputs, outputs=outputs)

    def test_gemm_epilogue_can_lower_and_emits_candidates(self):
        from rocke.helpers import GemmEpilogueLowerer, GreedyFusionScheduler

        graph = self._toy_gemm_graph(with_epilogue=True)
        plan = GreedyFusionScheduler().schedule(graph)
        region = plan.regions[0]
        lowerer = GemmEpilogueLowerer()
        result = lowerer.can_lower(graph, region)
        self.assertTrue(result.ok, result.reasons)
        cfgs = lowerer.candidates(graph, region)
        self.assertGreater(len(cfgs), 0)
        # Every candidate must keep the fused epilogue attached.
        for cfg in cfgs:
            self.assertTrue(hasattr(cfg.spec, "_fused_epilogue"))

    def test_gemm_epilogue_build_emits_kernel_launcher(self):
        from rocke.helpers import GemmEpilogueLowerer, GreedyFusionScheduler

        graph = self._toy_gemm_graph(with_epilogue=True)
        plan = GreedyFusionScheduler().schedule(graph)
        region = plan.regions[0]
        lowerer = GemmEpilogueLowerer()
        cfgs = lowerer.candidates(graph, region)
        built = lowerer.build(cfgs[0])
        # Smoke: launcher exists with the right kernel name; block size
        # matches the warp grid; bias plumbed through.
        self.assertIsNotNone(built.launcher)
        self.assertGreater(built.block_size, 0)
        self.assertEqual(built.extra.get("bias"), "bias")

    def test_elementwise_lowerer_round_trips(self):
        from rocke.helpers import (
            ElementwiseLowerer,
            FusionOp,
            FusionTensor,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionRegion

        graph = build_graph(
            tensors=[
                FusionTensor("A", (1024,), "fp16", is_input=True),
                FusionTensor("out", (1024,), "fp16", is_output=True),
            ],
            ops=[FusionOp("r0", "relu", ("A",), ("out",))],
            inputs=("A",),
            outputs=("out",),
        )
        region = FusionRegion(
            name="r0",
            kind="elementwise",
            op_names=("r0",),
            inputs=("A",),
            outputs=("out",),
            lowerer="elementwise",
        )
        lowerer = ElementwiseLowerer()
        self.assertTrue(lowerer.can_lower(graph, region).ok)
        cfgs = lowerer.candidates(graph, region)
        self.assertGreater(len(cfgs), 0)
        built = lowerer.build(cfgs[0])
        self.assertEqual(built.spec.op, "relu")
        self.assertEqual(built.spec.dtype, "f16")

    def test_reduction_lowerer_round_trips(self):
        from rocke.helpers import (
            FusionOp,
            FusionTensor,
            ReductionLowerer,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionRegion

        # ``Reduce2DSpec`` requires ``n_per_block`` divisible by
        # ``block_size * vec``. Pick 4096 (= 256*16 = 512*8) so every
        # candidate the lowerer emits is buildable.
        graph = build_graph(
            tensors=[
                FusionTensor("A", (64, 4096), "fp16", is_input=True),
                FusionTensor("out", (64,), "fp16", is_output=True),
            ],
            ops=[FusionOp("s0", "sum", ("A",), ("out",))],
            inputs=("A",),
            outputs=("out",),
        )
        region = FusionRegion(
            name="r0",
            kind="rowwise_reduction",
            op_names=("s0",),
            inputs=("A",),
            outputs=("out",),
            attrs={"reduce_op": "sum"},
            lowerer="rowwise_reduction",
        )
        lowerer = ReductionLowerer()
        self.assertTrue(lowerer.can_lower(graph, region).ok)
        cfgs = lowerer.candidates(graph, region)
        self.assertGreater(len(cfgs), 0)
        # Iterate candidates until we find a buildable one. The
        # candidate sweep emits a few (block_size, vec) combos; some
        # are illegal under the spec's divisibility constraint and we
        # expect the lowerer to round-trip with at least one.
        from rocke.instances.common.reduce import is_valid_spec

        buildable = [c for c in cfgs if is_valid_spec(c.spec)[0]]
        self.assertGreater(len(buildable), 0)
        built = lowerer.build(buildable[0])
        self.assertEqual(built.spec.op, "sum")
        self.assertEqual(built.spec.n_per_block, 4096)

    def test_gemm_epilogue_launch_args_include_bias_and_residual(self):
        from rocke.helpers import (
            BuiltRegion,
            FusionOp,
            FusionTensor,
            GemmEpilogueLowerer,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionRegion

        class FakeTensor:
            def __init__(self, shape):
                self.shape = tuple(shape)

            def numel(self):
                n = 1
                for d in self.shape:
                    n *= d
                return n

        graph = build_graph(
            tensors=[
                FusionTensor("A", (128, 64), "fp16", is_input=True),
                FusionTensor("B", (64, 128), "fp16", is_input=True),
                FusionTensor("bias", (128,), "fp16", is_input=True, layout="broadcast"),
                FusionTensor("residual", (128, 128), "fp16", is_input=True),
                FusionTensor("mm", (128, 128), "fp16"),
                FusionTensor("add_bias", (128, 128), "fp16"),
                FusionTensor("out", (128, 128), "fp16", is_output=True),
            ],
            ops=[
                FusionOp("mm0", "matmul", ("A", "B"), ("mm",)),
                FusionOp("add_bias0", "add", ("mm", "bias"), ("add_bias",)),
                FusionOp("add_res0", "add", ("add_bias", "residual"), ("out",)),
            ],
            inputs=("A", "B", "bias", "residual"),
            outputs=("out",),
        )
        region = FusionRegion(
            name="r0",
            kind="gemm_epilogue",
            op_names=("mm0", "add_bias0", "add_res0"),
            inputs=("A", "B", "bias", "residual"),
            outputs=("out",),
            lowerer="gemm_epilogue",
        )
        lowerer = GemmEpilogueLowerer()
        cfg = lowerer.candidates(graph, region)[0]
        built = BuiltRegion(
            launcher=None,
            spec=cfg.spec,
            block_size=cfg.spec.block_size,
            extra=cfg.extra,
        )
        runtime_args = {
            "A": FakeTensor((128, 64)),
            "B": FakeTensor((64, 128)),
            "bias": FakeTensor((128,)),
            "residual": FakeTensor((128, 128)),
            "out": FakeTensor((128, 128)),
        }
        args, grid = lowerer.launch_args(graph, region, runtime_args, built)
        self.assertEqual(args["M"], 128)
        self.assertEqual(args["N"], 128)
        self.assertEqual(args["K"], 64)
        self.assertIs(args["bias"], runtime_args["bias"])
        self.assertIs(args["residual_0"], runtime_args["residual"])
        # The grid is derived from the first candidate's tile, which the
        # autotuner selects against the launch device's MMA catalog (WMMA on
        # gfx11 wave32 picks a 32x32 first tile; a CDNA/no-GPU box picks a
        # larger one). Assert the grid matches the chosen tile rather than
        # hardcoding a single-tile (1,1,1) expectation.
        tile = cfg.spec.tile
        expected_grid = (
            (128 + tile.tile_n - 1) // tile.tile_n,
            (128 + tile.tile_m - 1) // tile.tile_m,
            1,
        )
        self.assertEqual(grid, expected_grid)

    def test_elementwise_lowerer_launch_args_binary(self):
        from rocke.helpers import (
            BuiltRegion,
            ElementwiseLowerer,
            FusionOp,
            FusionTensor,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionRegion
        from rocke.instances.common.elementwise import ElementwiseSpec

        class FakeTensor:
            shape = (1024,)

            def numel(self):
                return 1024

        graph = build_graph(
            tensors=[
                FusionTensor("A", (1024,), "fp16", is_input=True),
                FusionTensor("B", (1024,), "fp16", is_input=True),
                FusionTensor("out", (1024,), "fp16", is_output=True),
            ],
            ops=[FusionOp("add0", "add", ("A", "B"), ("out",))],
            inputs=("A", "B"),
            outputs=("out",),
        )
        region = FusionRegion(
            name="r0",
            kind="elementwise",
            op_names=("add0",),
            inputs=("A", "B"),
            outputs=("out",),
            lowerer="elementwise",
        )
        spec = ElementwiseSpec(op="add", dtype="f16", block_size=256, vec=8)
        built = BuiltRegion(launcher=None, spec=spec, block_size=256)
        runtime_args = {"A": FakeTensor(), "B": FakeTensor(), "out": FakeTensor()}
        args, grid = ElementwiseLowerer().launch_args(
            graph, region, runtime_args, built
        )
        self.assertEqual(args["N"], 1024)
        self.assertIs(args["A"], runtime_args["A"])
        self.assertIs(args["B"], runtime_args["B"])
        self.assertIs(args["C"], runtime_args["out"])
        self.assertEqual(grid, (1, 1, 1))

    def test_reduction_lowerer_rejects_non_2d_input(self):
        from rocke.helpers import (
            FusionOp,
            FusionTensor,
            ReductionLowerer,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionRegion

        graph = build_graph(
            tensors=[
                FusionTensor("A", (4096,), "fp16", is_input=True),
                FusionTensor("out", (), "fp16", is_output=True),
            ],
            ops=[FusionOp("sum0", "sum", ("A",), ("out",))],
            inputs=("A",),
            outputs=("out",),
        )
        region = FusionRegion(
            name="r0",
            kind="rowwise_reduction",
            op_names=("sum0",),
            inputs=("A",),
            outputs=("out",),
            lowerer="rowwise_reduction",
        )
        result = ReductionLowerer().can_lower(graph, region)
        self.assertFalse(result.ok)
        self.assertIn("input must be 2D", result.reasons[0])


class TestWorkspaceMaterialize(unittest.TestCase):
    """``materialize_plan`` should colour slots and allocate via the pool."""

    def _two_region_plan(self):
        from rocke.helpers import (
            FusionOp,
            FusionTensor,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionPlan, FusionRegion

        graph = build_graph(
            tensors=[
                FusionTensor("A", (16,), "fp16", is_input=True),
                FusionTensor("tmp", (16,), "fp16"),
                FusionTensor("out", (16,), "fp16", is_output=True),
            ],
            ops=[
                FusionOp("op0", "relu", ("A",), ("tmp",)),
                FusionOp("op1", "add", ("tmp", "A"), ("out",)),
            ],
            inputs=("A",),
            outputs=("out",),
        )
        return FusionPlan(
            graph=graph,
            regions=(
                FusionRegion("r0", "elementwise", ("op0",), ("A",), ("tmp",)),
                FusionRegion("r1", "elementwise", ("op1",), ("tmp", "A"), ("out",)),
            ),
        )

    def test_planner_colours_disjoint_lifetimes_into_one_slot(self):
        from rocke.helpers import (
            FusionOp,
            FusionTensor,
            WorkspacePlanner,
            build_graph,
        )
        from rocke.helpers.fusion_ir import FusionPlan, FusionRegion

        # Three regions: tmp1 in r0..r1, tmp2 in r2..r3. No overlap;
        # both should land on the same physical slot.
        graph = build_graph(
            tensors=[
                FusionTensor("A", (16,), "fp16", is_input=True),
                FusionTensor("tmp1", (16,), "fp16"),
                FusionTensor("tmp2", (16,), "fp16"),
                FusionTensor("out", (16,), "fp16", is_output=True),
            ],
            ops=[
                FusionOp("p0", "relu", ("A",), ("tmp1",)),
                FusionOp("p1", "add", ("tmp1", "A"), ("tmp2",)),  # consumes tmp1
                FusionOp("p2", "relu", ("tmp2",), ("out",)),  # produces out
            ],
            inputs=("A",),
            outputs=("out",),
        )
        plan = FusionPlan(
            graph=graph,
            regions=(
                FusionRegion("r0", "elementwise", ("p0",), ("A",), ("tmp1",)),
                FusionRegion("r1", "elementwise", ("p1",), ("tmp1", "A"), ("tmp2",)),
                FusionRegion("r2", "elementwise", ("p2",), ("tmp2",), ("out",)),
            ),
        )
        allocs = WorkspacePlanner().plan(plan)
        # ``tmp1`` lives r0..r1, ``tmp2`` lives r1..r2 -- they overlap
        # at r1, so they CANNOT share a slot.
        self.assertEqual(len(allocs), 2)
        self.assertNotEqual(allocs[0].slot_name, allocs[1].slot_name)

    def test_materialize_with_fake_pool_binds_tensors(self):
        # Use the real WorkspacePool with a CPU "device" via torch.
        import torch
        from rocke.helpers import materialize_plan
        from rocke.runtime.launcher import WorkspacePool

        plan = self._two_region_plan()
        pool = WorkspacePool()
        name_to_tensor, allocs = materialize_plan(
            plan,
            pool=pool,
            device="cpu",
        )
        self.assertIn("tmp", name_to_tensor)
        # The pool should own a slot tensor that matches the colourer.
        self.assertGreaterEqual(len(allocs), 1)
        self.assertEqual(name_to_tensor["tmp"].shape, torch.Size([16]))
        self.assertEqual(name_to_tensor["tmp"].dtype, torch.float16)


class TestValidationHarness(unittest.TestCase):
    """The fusion validation runner must produce well-formed reports."""

    def test_benchmark_case_runs_torch_eager_baseline(self):
        import torch
        from rocke.helpers import BenchmarkCase, run_fusion_validation_matrix

        def ref_fn(x, y):
            return x + y

        def make_inputs(shape, dtype):
            t = getattr(torch, dtype)
            return (
                torch.randn(shape, dtype=t),
                torch.randn(shape, dtype=t),
            )

        cases = [
            BenchmarkCase(
                name="add",
                ref_fn=ref_fn,
                make_inputs=make_inputs,
                backends={"torch_eager_again": ref_fn},
            )
        ]
        reports = run_fusion_validation_matrix(
            cases=cases,
            shapes=((128,),),
            dtypes=("float32",),
            warmup=1,
            iters=3,
        )
        self.assertEqual(len(reports), 1)
        report = reports[0]
        # Reference + one backend = 2 timing rows.
        self.assertEqual(len(report.timings), 2)
        # The backend that re-runs the reference should be bit-exact.
        backend = next(t for t in report.timings if t.name == "torch_eager_again")
        self.assertEqual(backend.max_abs, 0.0)
        # speedup_vs returns ratios > 0.
        speedup = report.speedup_vs("torch_eager")
        self.assertGreater(speedup["torch_eager_again"], 0.0)

    def test_validation_report_fastest_and_speedup(self):
        from rocke.helpers import BackendTiming, ValidationReport

        report = ValidationReport(
            graph_hash="abc123",
            correctness={"torch_eager": 0.0, "rocke": 1e-3},
            timings=(
                BackendTiming("torch_eager", 2.0, max_abs=0.0),
                BackendTiming("rocke", 0.5, max_abs=1e-3),
            ),
            pattern="pointwise_add",
            shape=(128,),
            dtype="fp16",
        )
        self.assertEqual(report.fastest().name, "rocke")
        self.assertEqual(report.speedup_vs("torch_eager")["rocke"], 4.0)
        as_dict = report.as_dict()
        self.assertEqual(as_dict["graph_hash"], "abc123")
        self.assertEqual(as_dict["fastest"]["name"], "rocke")
        self.assertEqual(as_dict["shape"], [128])


class TestAttentionHarnessTimers(unittest.TestCase):
    """The attention benchmark must time every lane with one shared clock.

    The harness keeps both Triton and CK DSL apples-to-apples by:

    1. Allocating one explicit HIP stream per lane.
    2. Routing the Triton call through ``torch.cuda.stream(...)`` so its
       launches land on that stream.
    3. Passing the same HIP stream handle into the CK DSL runner so its
       raw ``hipModuleLaunchKernel`` calls share the stream.
    4. Recording HIP events on that stream via
       :func:`rocke.runtime.time_launches`.

    These tests pin down (1, 3, 4): the timer goes through
    ``time_launches`` with the caller-supplied stream and follows up
    with a per-stream release.
    """

    @staticmethod
    def _load_harness_with_fake_aiter():
        import importlib.util
        import sys
        import types
        from pathlib import Path
        from unittest import mock

        # Import torch BEFORE patching sys.modules so torch stays in the
        # parent process's module table after ``mock.patch.dict`` exits.
        import torch  # noqa: F401

        # Resolve relative to the test file so the harness is found
        # regardless of where the checkout lives. A previously hardcoded
        # absolute path only worked on one author's machine.
        module_path = (
            Path(__file__).resolve().parents[1]
            / "Python/rocke/examples/gfx950/attention/parity_unified_attention.py"
        )
        fake_aiter = types.ModuleType("aiter")
        fake_ops = types.ModuleType("aiter.ops")
        fake_triton = types.ModuleType("aiter.ops.triton")
        fake_attention = types.ModuleType("aiter.ops.triton.attention")
        fake_unified = types.ModuleType("aiter.ops.triton.attention.unified_attention")
        fake_unified.use_2d_kernel = lambda *a, **k: True
        fake_unified.unified_attention = lambda *a, **k: None
        modules = {
            "aiter": fake_aiter,
            "aiter.ops": fake_ops,
            "aiter.ops.triton": fake_triton,
            "aiter.ops.triton.attention": fake_attention,
            "aiter.ops.triton.attention.unified_attention": fake_unified,
        }
        spec = importlib.util.spec_from_file_location(
            "rocke_attention_parity_timer_test",
            module_path,
        )
        mod = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        return mod

    def test_lane_timer_routes_through_time_launches_with_stream(self):
        from unittest import mock

        mod = self._load_harness_with_fake_aiter()

        calls = []

        def fake_time_launches(fn, *, warmup, iters, stream):
            calls.append(("time_launches", warmup, iters, stream))
            fn()
            return 0.123

        def fake_sync(stream=0):
            calls.append(("sync", stream))

        with mock.patch("rocke.runtime.time_launches", fake_time_launches), mock.patch(
            "rocke.runtime.synchronize_and_release", fake_sync
        ):
            ms = mod._time_lane_ms(
                lambda: calls.append(("launch",)),
                warmup=2,
                attempts=5,
                stream=77,
            )

        self.assertEqual(ms, 0.123)
        self.assertIn(("time_launches", 2, 5, 77), calls)
        # Must release the args bucket for THIS lane's stream, not stream 0.
        self.assertIn(("sync", 77), calls)

    def test_lane_timer_is_the_only_timer(self):
        """Sanity: the harness must NOT also export a torch-event timer.

        Keeping two clocks in the harness is what produced the
        apples-to-oranges Triton-vs-CK comparison the README originally
        called out. Make that contract explicit so a future patch that
        re-introduces a torch-event timer will fail this test.
        """
        mod = self._load_harness_with_fake_aiter()
        self.assertTrue(hasattr(mod, "_time_lane_ms"))
        self.assertFalse(hasattr(mod, "_time_torch_call_loop"))
        self.assertFalse(hasattr(mod, "_time_rocke_call_loop"))


class TestConvDirectGroupedTransforms(unittest.TestCase):
    """The transform-descriptor migration must keep the kernels building."""

    def test_16c_kernel_lowers_to_llvm(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances import (
            DirectConv16cSpec,
            DirectConvProblem,
            build_direct_conv_16c,
        )

        spec = DirectConv16cSpec(
            problem=DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16)
        )
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # Smoke check: the generated LLVM IR mentions amdgpu and the
        # kernel name (proves the body got emitted).
        self.assertIn("amdgpu", ll)
        self.assertIn(kernel.name, ll)

    def test_4c_kernel_lowers_to_llvm(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances import (
            DirectConv4cSpec,
            DirectConvProblem,
            build_direct_conv_4c,
        )

        spec = DirectConv4cSpec(
            problem=DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4)
        )
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("amdgpu", ll)
        self.assertIn(kernel.name, ll)


class TestTransformsRuntimeAware(unittest.TestCase):
    """The new runtime-aware transforms must respect their contracts.

    ``PadDynamic`` and ``Indirect`` lift the original ``Pad``/``Embed``
    compile-time constraints, but they must still feed the
    :class:`TensorDescriptor` chain machinery and propagate validity
    through ``offset()``.
    """

    def test_pad_dynamic_compile_time_matches_pad(self):
        """``pad_dynamic`` with int bounds emits the same algebra as ``pad``."""
        from rocke.core.ir import I32, IRBuilder
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.helpers.transforms import TensorDescriptor, pad, pad_dynamic

        def build_with(pad_fn):
            b = IRBuilder(f"pad_test_{pad_fn.__name__}")
            b.kernel.attrs["max_workgroup_size"] = 64
            n = b.param("N", I32)
            desc = TensorDescriptor.naive(
                "X",
                lengths=[1024],
                coord_names=("idx",),
            ).transform(pad_fn("idx", lo=0, hi=32))
            off, valid = desc.offset(b, idx=n)
            # Use ``valid`` so it doesn't get DCE'd; embed it as an i32.
            b.zext(valid, I32)
            return b.kernel, off

        k_pad, _ = build_with(pad)
        k_dyn, _ = build_with(pad_dynamic)
        ll_pad = lower_kernel_to_llvm(k_pad)
        ll_dyn = lower_kernel_to_llvm(k_dyn)
        # Both should compile to amdgpu IR; the dynamic-with-ints variant
        # is operationally identical to the compile-time pad variant.
        self.assertIn("amdgpu", ll_pad)
        self.assertIn("amdgpu", ll_dyn)

    def test_pad_dynamic_runtime_hi(self):
        """``pad_dynamic`` accepts a runtime SSA Value as the upper bound."""
        from rocke.core.ir import I32, IRBuilder
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.helpers.transforms import TensorDescriptor, pad_dynamic

        b = IRBuilder("pad_rt_test")
        b.kernel.attrs["max_workgroup_size"] = 64
        idx_v = b.param("idx", I32)
        hi_v = b.param("hi", I32)
        desc = TensorDescriptor.naive(
            "X",
            lengths=[1024],
            coord_names=("idx",),
        ).transform(pad_dynamic("idx", lo=0, hi=hi_v))
        off, valid = desc.offset(b, idx=idx_v)
        self.assertIsNotNone(valid)  # validity is in flight
        b.zext(valid, I32)  # touch it so it doesn't get DCE'd
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("amdgpu", ll)
        # Two parameters: ``idx`` and ``hi``. The IR must mention ``hi``
        # otherwise our runtime upper bound was dropped.
        self.assertIn("hi", ll)

    def test_pad_dynamic_one_sided_bound(self):
        """Passing ``hi=`` only emits a one-sided bounds check."""
        from rocke.core.ir import I32, IRBuilder
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.helpers.transforms import TensorDescriptor, pad_dynamic

        b = IRBuilder("pad_one_sided")
        b.kernel.attrs["max_workgroup_size"] = 64
        idx_v = b.param("idx", I32)
        desc = TensorDescriptor.naive(
            "X",
            lengths=[1024],
            coord_names=("idx",),
        ).transform(pad_dynamic("idx", hi=128))
        _, valid = desc.offset(b, idx=idx_v)
        self.assertIsNotNone(valid)
        b.zext(valid, I32)
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("amdgpu", ll)

    def test_indirect_emits_table_lookup_in_chain(self):
        """``indirect`` adds a real ``global_load_i32`` to the chain."""
        from rocke.core.ir import I32, IRBuilder, PtrType
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.helpers.transforms import TensorDescriptor, indirect

        b = IRBuilder("indirect_test")
        b.kernel.attrs["max_workgroup_size"] = 64
        table = b.param("table", PtrType(I32, "global"), readonly=True)
        base = b.param("base", I32)
        tile_idx = b.param("tile_idx", I32)
        dim = b.param("dim", I32)
        desc = TensorDescriptor.naive(
            "kv",
            lengths=[4096, 64],
            coord_names=("physical_block", "dim"),
        ).transform(
            indirect("tile_idx", into="physical_block", table=table, base=base),
        )
        # Now the descriptor's user-facing coords are ("tile_idx", "dim").
        self.assertEqual(set(desc.upper_names), {"tile_idx", "dim"})
        off, _ = desc.offset(b, tile_idx=tile_idx, dim=dim)
        ll = lower_kernel_to_llvm(b.kernel)
        # The chain must lower to AMDGPU IR and the table+base parameters
        # must show up in the function signature (proves we issued the
        # table load).
        self.assertIn("amdgpu", ll)
        self.assertIn("table", ll)
        self.assertIn("base", ll)


class TestCkTileLowering(unittest.TestCase):
    """``lower_spec_to_cktile`` must emit a CK Tile-shaped C++ source that
    references the same template surface a hand-written CK Tile kernel
    would: ``TileGemmShape<...>``, ``GemmSpatiallyLocalTilePartitioner``,
    ``GemmPipelineAgBgCr*``, ``CShuffleEpilogue``, and -- for the kernel
    composition -- ``GemmKernel<...>`` (for ``UniversalGemmSpec``) or
    ``GroupedConvolutionForwardKernel<...>`` (for ``ImplicitGemmConvSpec``).

    Pure-Python checks; the actual compile through hipcc happens in
    `examples/lower_cktile_smoke` (out of scope here because it needs
    a working amdclang + CK Tile include dir).
    """

    def _src_for_gemm(self, *, pipeline="compv4"):
        from rocke.core import lower_spec_to_cktile
        from rocke.instances import TileSpec, TraitSpec, UniversalGemmSpec

        spec = UniversalGemmSpec(
            name="t_gemm",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline=pipeline,
                scheduler="intrawave",
                epilogue="cshuffle",
            ),
        )
        return lower_spec_to_cktile(spec), spec

    def test_gemm_source_references_cktile_template_surface(self):
        src, spec = self._src_for_gemm()
        # Required header set -- the emitted source must compile against
        # a stock CK Tile install with these (and only these) includes.
        for header in (
            "#include <ck_tile/core.hpp>",
            "#include <ck_tile/host.hpp>",
            "#include <ck_tile/ops/gemm.hpp>",
            "#include <ck_tile/ops/epilogue.hpp>",
        ):
            self.assertIn(header, src)
        # Composition surface that distinguishes CK Tile output from the
        # raw HIP IR lowering.
        for token in (
            "ck_tile::TileGemmShape",
            "ck_tile::GemmSpatiallyLocalTilePartitioner",
            "ck_tile::TileGemmUniversalTraits",
            "ck_tile::UniversalGemmPipelineProblem",
            "ck_tile::CShuffleEpilogue",
            "ck_tile::CShuffleEpilogueProblem",
            "ck_tile::GemmKernel<",
            "ck_tile::make_kernel<",
            "ck_tile::launch_kernel(",
        ):
            self.assertIn(token, src, f"emitted source missing {token!r}")
        # The host launcher must expose the spec's mangled name as an
        # extern "C" symbol so callers can dlsym it without C++ name
        # mangling getting in the way.
        self.assertIn(f'extern "C" float launch_{spec.kernel_name()}(', src)

    def test_gemm_compv4_picks_compute_v4_pipeline(self):
        src, _ = self._src_for_gemm(pipeline="compv4")
        # The pipeline enum + the matching specialisation in the
        # PipelineTypeTraits inline table must both reference COMPUTE_V4.
        self.assertIn("ck_tile::GemmPipeline::COMPUTE_V4", src)
        self.assertIn("GemmPipelineAgBgCrCompV4", src)
        # compv4 doubles the LDS buffer.
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", src)

    def test_gemm_mem_picks_memory_pipeline_single_buffer(self):
        src, _ = self._src_for_gemm(pipeline="mem")
        self.assertIn("ck_tile::GemmPipeline::MEMORY", src)
        self.assertIn("GemmPipelineAgBgCrMem", src)
        self.assertIn("static constexpr bool DoubleSmemBuffer = false;", src)

    def test_gemm_bakes_tile_shape_into_GemmConfig(self):
        src, _ = self._src_for_gemm()
        # Each tile / warp / MFMA value the spec carries must end up as a
        # ``static constexpr ck_tile::index_t`` in GemmConfig so the
        # downstream TileGemmShape<sequence<...>> picks them up at
        # compile time.
        for line in (
            "static constexpr ck_tile::index_t M_Tile = 128;",
            "static constexpr ck_tile::index_t N_Tile = 128;",
            "static constexpr ck_tile::index_t K_Tile = 32;",
            "static constexpr ck_tile::index_t M_Warp = 2;",
            "static constexpr ck_tile::index_t N_Warp = 2;",
            "static constexpr ck_tile::index_t K_Warp = 1;",
            "static constexpr ck_tile::index_t M_Warp_Tile = 32;",
            "static constexpr ck_tile::index_t N_Warp_Tile = 32;",
            "static constexpr ck_tile::index_t K_Warp_Tile = 16;",
        ):
            self.assertIn(line, src)

    def test_conv_source_references_grouped_convolution_kernel(self):
        from rocke.core import lower_spec_to_cktile
        from rocke.instances import (
            ConvProblem,
            ImplicitGemmConvSpec,
        )

        spec = ImplicitGemmConvSpec(
            problem=ConvProblem(
                N=8,
                Hi=56,
                Wi=56,
                C=64,
                K=64,
                Y=3,
                X=3,
                sH=1,
                sW=1,
                pH=1,
                pW=1,
                dH=1,
                dW=1,
            ),
            name="t_conv",
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
        src = lower_spec_to_cktile(spec)
        # Required include + the CK Tile-side type names that distinguish
        # the conv composition from a plain GEMM.
        for token in (
            "#include <ck_tile/ops/grouped_convolution.hpp>",
            "ck_tile::ConvolutionSpecialization::Default",
            "ck_tile::GroupedConvTraits<",
            "ck_tile::GroupedConvolutionForwardKernel<",
            "GroupedConvTraitsType",
            "tensor_layout::convolution::NHWGC",
            "tensor_layout::convolution::GKYXC",
            "tensor_layout::convolution::NHWGK",
            "ck_tile::GroupedConvFwdHostArgs",
            f'extern "C" float launch_{spec.kernel_name()}(',
        ):
            self.assertIn(token, src, f"emitted conv source missing {token!r}")

    def test_unsupported_dtype_raises_NotImplemented(self):
        from rocke.core.lower_cktile import lower_universal_gemm_to_cktile
        from rocke.instances import (
            DataSpec,
            TileSpec,
            UniversalGemmSpec,
        )

        spec = UniversalGemmSpec(
            name="t",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        )
        with self.assertRaises(NotImplementedError):
            lower_universal_gemm_to_cktile(spec)

    def test_unsupported_pipeline_raises_NotImplemented(self):
        from rocke.core.lower_cktile import lower_universal_gemm_to_cktile
        from rocke.instances import (
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
        )

        bad = UniversalGemmSpec(
            name="t",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            # ``compv3`` is in the IR builder but not yet wired through
            # the PipelineTypeTraits inline switch in lower_cktile.py
            # ... actually compv3 IS supported, so use a fake string.
            trait=TraitSpec(pipeline="compv3", scheduler="intrawave"),
        )
        from rocke.core.lower_cktile import _PIPELINE_MAP

        self.assertIn("compv3", _PIPELINE_MAP)
        # Force a real unsupported pipeline.
        from dataclasses import replace

        bad_pipeline = replace(bad, trait=replace(bad.trait, pipeline="not_a_pipeline"))  # type: ignore[arg-type]
        with self.assertRaises(NotImplementedError):
            lower_universal_gemm_to_cktile(bad_pipeline)

    def test_dispatcher_handles_both_spec_types(self):
        """``lower_spec_to_cktile`` must dispatch to gemm vs conv emitters
        based on the spec type without the caller knowing about them.
        """
        from rocke.core import lower_spec_to_cktile
        from rocke.instances import (
            ConvProblem,
            ImplicitGemmConvSpec,
            TileSpec,
            UniversalGemmSpec,
        )

        gemm = UniversalGemmSpec(
            name="d_gemm",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
        )
        conv = ImplicitGemmConvSpec(
            problem=ConvProblem(
                N=8,
                Hi=56,
                Wi=56,
                C=64,
                K=64,
                Y=3,
                X=3,
                sH=1,
                sW=1,
                pH=1,
                pW=1,
                dH=1,
                dW=1,
            ),
            name="d_conv",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        )
        # GEMM dispatches into ``GemmKernel<...>``; conv into
        # ``GroupedConvolutionForwardKernel<...>``. The dispatcher must
        # not cross-pollinate.
        gemm_src = lower_spec_to_cktile(gemm)
        conv_src = lower_spec_to_cktile(conv)
        self.assertIn("ck_tile::GemmKernel<", gemm_src)
        self.assertNotIn("ck_tile::GroupedConvolutionForwardKernel<", gemm_src)
        self.assertIn("ck_tile::GroupedConvolutionForwardKernel<", conv_src)
        self.assertNotIn("ck_tile::GemmKernel<", conv_src)

    def test_dispatcher_rejects_unknown_spec(self):
        from rocke.core import lower_spec_to_cktile

        with self.assertRaises(NotImplementedError) as cm:
            lower_spec_to_cktile(object())
        self.assertIn("UniversalGemmSpec", str(cm.exception))
        self.assertIn("ImplicitGemmConvSpec", str(cm.exception))


class TestHipLoweringCoverage(unittest.TestCase):
    """:func:`rocke.core.lower_kernel_to_hip` must lower every production
    instance kernel without ``NotImplementedError``.

    This pins the contract that the HIP debug lowering is a complete
    one-to-one IR-to-C++ pass rather than a stub. Each test below
    builds one of the in-tree kernel instances, lowers it, and verifies
    the resulting source has the expected structural markers: the
    typedef prologue, the ``__global__`` signature, and at least one
    of the kernel's signature ops (MFMA / async DMA / smem_alloc /
    arith.fmul / ...). Doesn't actually compile through hipcc -- that
    requires a working AMDGPU toolchain and lives in
    ``examples/ck_tile_parity`` / its peers.
    """

    def _lower_and_check(self, kernel, must_contain):
        from rocke.core.lower_hip import lower_kernel_to_hip

        hip = lower_kernel_to_hip(kernel)
        self.assertIn("#include <hip/hip_runtime.h>", hip)
        self.assertIn("__global__", hip)
        self.assertIn(f"void {kernel.name}(", hip)
        for token in must_contain:
            self.assertIn(
                token,
                hip,
                f"lowered HIP source for {kernel.name!r} missing required token "
                f"{token!r}",
            )

    def test_universal_gemm_mem_lowers(self):
        from rocke.instances import (
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
        )

        spec = UniversalGemmSpec(
            name="hip_gemm_mem",
            tile=TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16),
            trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="cshuffle"),
        )
        self._lower_and_check(
            build_universal_gemm(spec),
            must_contain=[
                "__builtin_amdgcn_mfma_f32_32x32x16_f16",
                "__syncthreads()",
                "__shared__",
            ],
        )

    def test_universal_gemm_compv4_lowers(self):
        from rocke.instances import (
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
        )

        spec = UniversalGemmSpec(
            name="hip_gemm_compv4",
            tile=TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
        )
        self._lower_and_check(
            build_universal_gemm(spec),
            must_contain=["__builtin_amdgcn_mfma_f32_32x32x16_f16", "__shared__"],
        )

    def test_implicit_gemm_conv_lowers(self):
        from rocke.instances import (
            ConvProblem,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        p = ConvProblem(
            N=8,
            Hi=56,
            Wi=56,
            C=64,
            K=64,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )
        spec = ImplicitGemmConvSpec(
            problem=p,
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
        self._lower_and_check(
            build_implicit_gemm_conv(spec),
            must_contain=[
                "__builtin_amdgcn_make_buffer_rsrc",
                "__builtin_amdgcn_raw_buffer_load",
                "__builtin_amdgcn_raw_buffer_store",
                "__builtin_amdgcn_mfma_f32_32x32x16_f16",
            ],
        )

    def test_layernorm2d_lowers(self):
        from rocke.instances import LayerNorm2DSpec, build_layernorm2d

        self._lower_and_check(
            build_layernorm2d(
                LayerNorm2DSpec(n_per_block=4096, block_size=256, vec=8, dtype="f16")
            ),
            must_contain=[
                "__builtin_amdgcn_rsqf",  # rsqrt for inv_stddev
                "__syncthreads()",  # LDS tree reduction barrier
                "__shared__",
            ],
        )

    def test_rmsnorm2d_lowers(self):
        from rocke.instances import RMSNorm2DSpec, build_rmsnorm2d

        self._lower_and_check(
            build_rmsnorm2d(RMSNorm2DSpec(n_per_block=4096, block_size=256, vec=8)),
            must_contain=["__builtin_amdgcn_rsqf", "__syncthreads()"],
        )

    def test_lower_to_hip_emits_prologue_typedefs(self):
        """The prologue must define every typedef family referenced by
        the handlers. Typedef instances are emitted via the
        ``_ROCKE_VEC`` macro (so the literal ``f16x4`` only appears
        post-preprocess); we assert the macro invocations are present
        for each family + width we rely on, plus the named singleton
        typedefs (``fp16``, ``bf16``, ``rsrc_t``, ``fp8e4m3``).
        """
        from rocke.core.lower_hip import HIP_PROLOGUE

        named = [
            "using fp16 = _Float16;",
            "using bf16 = __bf16;",
            "using fp8e4m3 = signed char",
            "using rsrc_t = __amdgpu_buffer_rsrc_t;",
        ]
        # Each (family, widths) tuple must yield a ``_ROCKE_VEC(elem, fam, N)``
        # invocation in the prologue for every width.
        vec_invocations = [
            ("f16x", [1, 2, 4, 8, 16]),
            ("bf16x", [1, 2, 4, 8, 16]),
            ("f32x", [1, 2, 4, 8, 16]),
            ("i32x", [1, 2, 3, 4, 8]),
            ("i8x", [4, 8, 16]),
        ]
        for token in named:
            self.assertIn(token, HIP_PROLOGUE, f"prologue missing {token!r}")
        import re

        for family, widths in vec_invocations:
            for n in widths:
                # The macro call form is ``_ROCKE_VEC(<elem>, <family>, <n>)``;
                # internal whitespace varies for column alignment, so we
                # match with a small regex that allows any spacing.
                pattern = re.compile(
                    rf"_ROCKE_VEC\(\s*\S+\s*,\s*{re.escape(family)}\s*,\s*{n}\s*\)"
                )
                self.assertRegex(
                    HIP_PROLOGUE,
                    pattern,
                    f"prologue missing _ROCKE_VEC for {family}{n}",
                )

    def test_lower_to_hip_include_prologue_can_be_disabled(self):
        """``include_prologue=False`` returns the bare kernel body so
        callers can embed it into a larger translation unit that
        already has the typedefs."""
        from rocke import IRBuilder, F16, I32, PtrType
        from rocke.core.lower_hip import lower_kernel_to_hip

        b = IRBuilder("bare")
        b.param("A", PtrType(F16, "global"))
        b.param("N", I32)
        b.ret()
        out = lower_kernel_to_hip(b.kernel, include_prologue=False)
        self.assertNotIn("#include <hip/hip_runtime.h>", out)
        self.assertIn("__global__", out)
        self.assertIn("void bare(", out)


class TestLauncherFenceContract(unittest.TestCase):
    """Per-launch event-fence policy in :mod:`rocke.runtime.launcher`.

    These tests pin down the host-side launch-lifecycle contract that
    fixed the M=N=K=2048 silent-corruption regression: every
    ``KernelLauncher`` call must, by default, event-synchronize on its
    own launch's completion before returning so the host never
    observes a half-finished kernel. The contract is enforced via
    ``LaunchConfig.fence`` plus the :func:`no_fence` context manager
    that batch wrappers (e.g. :func:`time_launches`) use to suppress
    per-call sync inside an outer event-timed region.

    No GPU is required: we use a fake :class:`KernelLauncher` that
    records what ``_resolved_fence`` reports inside its ``__call__``,
    then assert the public API combinations resolve as documented.
    """

    def test_launch_config_default_fence_is_on(self):
        from rocke.runtime.launcher import LaunchConfig

        self.assertTrue(LaunchConfig().fence)

    def test_no_fence_overrides_config_fence_true(self):
        from rocke.runtime.launcher import _resolved_fence, no_fence

        # Outside the context, the per-config value wins.
        self.assertTrue(_resolved_fence(True))
        self.assertFalse(_resolved_fence(False))

        # Inside, the override forces False regardless of the per-config value.
        with no_fence():
            self.assertFalse(_resolved_fence(True))
            self.assertFalse(_resolved_fence(False))

        # And the override is correctly torn down on exit.
        self.assertTrue(_resolved_fence(True))

    def test_no_fence_nests(self):
        from rocke.runtime.launcher import _resolved_fence, no_fence

        with no_fence():
            self.assertFalse(_resolved_fence(True))
            with no_fence():
                self.assertFalse(_resolved_fence(True))
            # Inner contexts don't drop the override too early.
            self.assertFalse(_resolved_fence(True))
        self.assertTrue(_resolved_fence(True))

    def test_pipeline_launcher_fences_only_last_stage(self):
        """Same-stream FIFO already orders intermediate stages, so they
        do NOT fence (a per-stage host sync would defeat pipelining).
        Only the final stage honors ``cfg.fence`` on the host's behalf.

        We pass a non-zero ``stream`` so :func:`resolve_stream` does
        not eagerly initialize torch.cuda inside a pure-Python unit
        test; that keeps this test isolated from any HIP-context
        ordering effects on other unit tests.
        """
        from rocke.runtime.launcher import LaunchConfig, PipelineLauncher

        recorded: list = []

        class FakeStage:
            kernel_name = "fake"

            def __init__(self, idx):
                self.idx = idx

            def __call__(self, values, *, config):
                recorded.append((self.idx, config.fence))
                # Return an object that looks like LaunchSummary enough.
                from rocke.runtime.launcher import LaunchSummary

                return LaunchSummary(launches=1)

        stages = [FakeStage(0), FakeStage(1), FakeStage(2)]
        pipeline = PipelineLauncher(stages)

        cfg = LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1), fence=True)
        pipeline(
            values_per_stage=[{}, {}, {}],
            configs_per_stage=[cfg, cfg, cfg],
            stream=1,
        )
        self.assertEqual(recorded, [(0, False), (1, False), (2, True)])

    def test_pipeline_launcher_honors_no_fence_on_last_stage(self):
        """If the user passed ``fence=False`` for the final stage, the
        pipeline must NOT override it: the caller is opting into
        managing their own sync (e.g. inside :func:`time_launches`).
        """
        from rocke.runtime.launcher import LaunchConfig, PipelineLauncher

        recorded: list = []

        class FakeStage:
            kernel_name = "fake"

            def __init__(self, idx):
                self.idx = idx

            def __call__(self, values, *, config):
                recorded.append((self.idx, config.fence))
                from rocke.runtime.launcher import LaunchSummary

                return LaunchSummary(launches=1)

        pipeline = PipelineLauncher([FakeStage(0), FakeStage(1)])
        cfg_nofence = LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1), fence=False)
        pipeline(
            values_per_stage=[{}, {}],
            configs_per_stage=[cfg_nofence, cfg_nofence],
            stream=1,
        )
        self.assertEqual(recorded, [(0, False), (1, False)])


class TestCkStyleLaunchKernel(unittest.TestCase):
    """CK-Tile-style :func:`make_kernel` + :func:`launch_kernel` primitive.

    Pure-Python contract checks of the new variadic launch primitive
    that mirrors ``ck_tile::launch_kernel`` /
    ``ck_tile::make_kernel`` from
    ``include/ck_tile/host/kernel_launch.hpp``. A fake "launcher" is
    used in place of :class:`KernelLauncher` so these tests exercise
    only the closure/dispatch logic and avoid any HIP context.

    The non-timing path (``time_kernel=False``) is fully covered
    here. The timing path (``time_kernel=True``) requires a HIP
    context for :func:`Runtime.wait_stream` and
    :class:`Runtime.event`; that path is exercised by the GPU smoke
    test in ``test_rocke_examples.TestCkStyleLaunchKernelGpu``.
    """

    def _record_launcher(self):
        """Return a (recorded, FakeLauncher-instance) pair.

        The fake's ``__call__`` signature matches
        :class:`KernelLauncher`'s production interface
        (``(values, *, config)``) so :func:`make_kernel` does not
        need a duck-typed shim to drive it.
        """
        from rocke.runtime.launcher import LaunchSummary

        recorded: list = []

        class _FakeLauncher:
            kernel_name = "fake"

            def __call__(self, values, *, config):
                recorded.append(
                    {
                        "values": dict(values),
                        "fence": config.fence,
                        "stream": config.stream,
                        "grid": tuple(config.grid),
                        "block": tuple(config.block),
                        "shared_bytes": config.shared_bytes,
                    }
                )
                return LaunchSummary(launches=1)

        return recorded, _FakeLauncher()

    def test_stream_config_defaults_match_cktile(self):
        """``StreamConfig`` field defaults must mirror
        ``ck_tile::stream_config`` field-for-field
        (``include/ck_tile/host/stream_config.hpp`` lines 29-39) so a
        Python benchmark driver can construct one config object that
        is valid on both sides.
        """
        from rocke.runtime.launcher import StreamConfig

        s = StreamConfig()
        self.assertEqual(s.stream_id, 0)
        self.assertFalse(s.time_kernel)
        self.assertEqual(s.log_level, 0)
        self.assertEqual(s.cold_niters, 3)
        self.assertEqual(s.nrepeat, 10)
        self.assertTrue(s.is_gpu_timer)
        self.assertFalse(s.flush_cache)
        # Frozen so accidental mutation can't desync benchmark drivers.
        with self.assertRaises(Exception):
            s.stream_id = 7  # type: ignore[misc]

    def test_launch_kernel_empty_raises(self):
        """Parity with the C++
        ``static_assert(sizeof...(callables) > 0)`` guard at
        ``kernel_launch.hpp:268``. Catching this at the Python boundary
        prevents a no-op timed loop from quietly returning 0.0 ms.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel

        with self.assertRaises(ValueError):
            launch_kernel(StreamConfig())

    def test_launch_kernel_runs_callables_in_order(self):
        """Same-stream FIFO ordering is the only correctness primitive
        between callables, but :func:`launch_kernel`'s host-side
        invocation order must also match declaration order so the
        kernels are actually enqueued in the documented sequence.
        Mirrors the C++ fold expression at ``kernel_launch.hpp:139``.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel

        order: list = []

        def _mk(idx):
            def _c(s):
                order.append((idx, s.stream_id))

            return _c

        s = StreamConfig(stream_id=42)
        ms = launch_kernel(s, _mk(0), _mk(1), _mk(2))
        self.assertEqual(ms, 0.0)
        self.assertEqual(order, [(0, 42), (1, 42), (2, 42)])

    def test_launch_kernel_returns_zero_when_not_timed(self):
        """``time_kernel=False`` is fire-and-forget: no events, no
        timing, no host-side ms. Matches the early-return path at
        ``kernel_launch.hpp:270-274``.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel

        ms = launch_kernel(StreamConfig(time_kernel=False), lambda s: None)
        self.assertEqual(ms, 0.0)

    def test_launch_kernel_short_circuits_on_error(self):
        """Mirror of the short-circuit fold expression at
        ``kernel_launch.hpp:139`` (``(... && ...)``). On the C++
        side, ``hipPeekAtLastError() != hipSuccess`` skips the
        remaining callables; on the Python side, an exception in any
        callable propagates immediately and prevents subsequent
        callables from running. Either way, the first failing launch
        aborts the rest of the group.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel

        order: list = []

        def _ok(idx):
            def _c(s):
                order.append(idx)

            return _c

        def _boom(s):
            order.append(1)
            raise RuntimeError("simulated HIP failure")

        with self.assertRaises(RuntimeError):
            launch_kernel(StreamConfig(), _ok(0), _boom, _ok(2))
        # Callables 0 and 1 ran (1 raised); callable 2 did not.
        self.assertEqual(order, [0, 1])

    def test_make_kernel_captures_args_grid_block_lds(self):
        """Each closure must carry an independent capture of
        ``(values, grid, block, lds_bytes)`` so dispatching N stages
        through one launcher does not have stage K observing stage
        K-1's values. Mirrors the C++ ``make_kernel`` capture
        semantics at ``kernel_launch.hpp:118-133`` -- the returned
        lambda takes everything by value.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel, make_kernel

        recorded, launcher = self._record_launcher()

        c0 = make_kernel(launcher, {"x": 1}, (1, 1, 1), (64, 1, 1), lds_bytes=0)
        c1 = make_kernel(launcher, {"x": 2}, (2, 1, 1), (128, 1, 1), lds_bytes=512)
        c2 = make_kernel(
            launcher, {"x": 3, "y": 7}, (3, 1, 1), (32, 2, 1), lds_bytes=4096
        )

        launch_kernel(StreamConfig(stream_id=9), c0, c1, c2)

        self.assertEqual(len(recorded), 3)
        self.assertEqual(recorded[0]["values"], {"x": 1})
        self.assertEqual(recorded[0]["grid"], (1, 1, 1))
        self.assertEqual(recorded[0]["block"], (64, 1, 1))
        self.assertEqual(recorded[0]["shared_bytes"], 0)
        self.assertEqual(recorded[1]["values"], {"x": 2})
        self.assertEqual(recorded[1]["grid"], (2, 1, 1))
        self.assertEqual(recorded[1]["block"], (128, 1, 1))
        self.assertEqual(recorded[1]["shared_bytes"], 512)
        self.assertEqual(recorded[2]["values"], {"x": 3, "y": 7})
        self.assertEqual(recorded[2]["grid"], (3, 1, 1))
        self.assertEqual(recorded[2]["block"], (32, 2, 1))
        self.assertEqual(recorded[2]["shared_bytes"], 4096)

    def test_make_kernel_uses_fence_false(self):
        """The CK-style closure must always launch with ``fence=False``
        -- the only sync point is :func:`launch_kernel`'s timing-loop
        boundary (or the caller's external drain). This is the
        critical migration hazard for callers coming from
        :class:`PipelineLauncher` (which honours the per-stage fence
        on the last stage); see the docstring on :func:`launch_kernel`.
        """
        from rocke.runtime.launcher import (
            LaunchConfig,
            StreamConfig,
            launch_kernel,
            make_kernel,
            no_fence,
        )

        recorded, launcher = self._record_launcher()

        # Default fence policy outside any context: per-config wins.
        # The closure must still emit fence=False at the launcher.
        self.assertTrue(LaunchConfig().fence)
        c = make_kernel(launcher, {}, (1, 1, 1), (64, 1, 1))
        launch_kernel(StreamConfig(), c)
        self.assertEqual(len(recorded), 1)
        self.assertFalse(recorded[0]["fence"])

        # Inside an outer no_fence: also emits False (vacuously).
        recorded.clear()
        with no_fence():
            launch_kernel(StreamConfig(), c)
        self.assertEqual(len(recorded), 1)
        self.assertFalse(recorded[0]["fence"])

    def test_make_kernel_captures_dict_copy(self):
        """:func:`make_kernel` snapshots ``values`` via
        ``dict(values)``. Mutating the original after closure
        construction must not affect the captured payload, otherwise
        a caller that re-uses one dict to build successive closures
        (the natural pattern for sweep harnesses) would silently
        observe N copies of the most recent values.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel, make_kernel

        recorded, launcher = self._record_launcher()

        vals = {"k": 1}
        c = make_kernel(launcher, vals, (1, 1, 1), (64, 1, 1))
        vals["k"] = 999
        vals["new"] = "intruder"
        launch_kernel(StreamConfig(), c)
        self.assertEqual(recorded[0]["values"], {"k": 1})

    def test_make_kernel_forwards_stream_at_call_time(self):
        """The closure reads ``stream_id`` from its
        :class:`StreamConfig` argument at call time (not at closure
        construction time), so the same closure can be replayed on
        different streams. This matches CK Tile's ``make_kernel``
        whose generated lambda references ``s.stream_id_`` (read at
        invocation time, not capture time).
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel, make_kernel

        recorded, launcher = self._record_launcher()
        c = make_kernel(launcher, {}, (1, 1, 1), (64, 1, 1))

        launch_kernel(StreamConfig(stream_id=11), c)
        launch_kernel(StreamConfig(stream_id=22), c)
        launch_kernel(StreamConfig(stream_id=33), c)

        self.assertEqual([r["stream"] for r in recorded], [11, 22, 33])

    def test_launch_kernel_accepts_bare_lambdas(self):
        """The C++ ``launch_kernel`` accepts any callable matching
        ``operator()(const stream_config&)``, including bare lambdas
        like ``[=](const stream_config& s){ hipMemset(...) }``. The
        Python equivalent must accept any ``Callable[[StreamConfig],
        None]``: this is what enables the
        ``maybe_clear_workspace``-style sibling-callable pattern at
        ``example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp:481``
        without forcing every callable through :func:`make_kernel`.
        """
        from rocke.runtime.launcher import StreamConfig, launch_kernel, make_kernel

        recorded, launcher = self._record_launcher()
        marker: list = []

        def _bare(s):
            marker.append(("clear", s.stream_id))

        c = make_kernel(launcher, {"x": 1}, (1, 1, 1), (64, 1, 1))
        launch_kernel(StreamConfig(stream_id=5), _bare, c, _bare)

        # Bare lambdas ran around the make_kernel closure, all with
        # the same stream id forwarded.
        self.assertEqual(marker, [("clear", 5), ("clear", 5)])
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["stream"], 5)


class TestRuntimeEventLifecycle(unittest.TestCase):
    """:class:`Runtime` pending-args queue: events + FIFO drain.

    Pure-Python contract checks; no GPU needed. We stand up a
    :class:`Runtime` instance and stub the ``Event``s that
    ``_reap_completed`` would normally query so the FIFO walk and
    ref-release semantics are exercised deterministically.
    """

    def _isolate_pending(self):
        """Snapshot + clear the class-level pending-args dict so this
        test doesn't observe (or leak into) other tests' state.
        """
        from rocke.runtime.hip_module import Runtime

        prior = dict(Runtime._pending_args)
        Runtime._pending_args.clear()
        return prior

    def _restore_pending(self, prior):
        from rocke.runtime.hip_module import Runtime

        Runtime._pending_args.clear()
        Runtime._pending_args.update(prior)

    def test_reap_completed_pops_fired_events_in_fifo_order(self):
        """Eager release on next launch must drop only the prefix of
        bucket entries whose events have fired; an un-fired event
        halts the walk so later entries (which may still be in
        flight) are not freed prematurely.
        """
        from rocke.runtime.hip_module import Runtime

        class FakeEvent:
            def __init__(self, fired: bool):
                self._fired = fired
                self.destroyed = False

            def query(self) -> bool:
                return self._fired

            def destroy(self) -> None:
                self.destroyed = True

        rt = Runtime()
        prior = self._isolate_pending()
        try:
            e0 = FakeEvent(fired=True)
            e1 = FakeEvent(fired=True)
            e2 = FakeEvent(fired=False)  # this one is still in flight
            e3 = FakeEvent(fired=True)
            Runtime._pending_args[42] = [
                (("ref0",), e0),
                (("ref1",), e1),
                (("ref2",), e2),
                (("ref3",), e3),
            ]
            rt._reap_completed(42)
            # The first two should be popped (FIFO from the head);
            # e2 halts the walk so e3 is NOT touched even though it
            # has fired.
            self.assertEqual(len(Runtime._pending_args[42]), 2)
            self.assertTrue(e0.destroyed and e1.destroyed)
            self.assertFalse(e2.destroyed)
            self.assertFalse(e3.destroyed)
        finally:
            self._restore_pending(prior)

    def test_reap_stops_at_none_event(self):
        """Legacy ``retain_for_stream`` calls before any launch park
        refs with ``event=None``. Those entries must NOT be reaped
        by the eager pre-launch walk (only ``sync`` releases them);
        otherwise the runtime would drop refs the GPU is still
        relying on.
        """
        from rocke.runtime.hip_module import Runtime

        rt = Runtime()
        prior = self._isolate_pending()
        try:
            Runtime._pending_args[7] = [(("legacy_ref",), None)]
            rt._reap_completed(7)
            self.assertEqual(len(Runtime._pending_args[7]), 1)
        finally:
            self._restore_pending(prior)

    def test_retain_for_stream_merges_into_latest_entry(self):
        """``retain_for_stream`` must attach to the most-recent
        launch's bucket entry so the tensors share that launch's
        completion event. If it appended a fresh entry instead the
        retain would never be released unless ``sync`` is called.
        """
        from rocke.runtime.hip_module import Runtime

        class FakeEvent:
            def query(self) -> bool:
                return False

            def destroy(self) -> None:
                pass

        rt = Runtime()
        prior = self._isolate_pending()
        try:
            e = FakeEvent()
            Runtime._pending_args[9] = [(("args_buf",), e)]
            rt.retain_for_stream(9, "tensor_A", "tensor_B", 1234)
            entries = Runtime._pending_args[9]
            self.assertEqual(len(entries), 1)
            refs, evt = entries[0]
            # ints are filtered out; tensors are merged into the same
            # tuple so they all share the launch's completion event.
            self.assertEqual(refs, ("args_buf", "tensor_A", "tensor_B"))
            self.assertIs(evt, e)
        finally:
            self._restore_pending(prior)


# ---------------------------------------------------------------------
# Build smoke tests for the FMHA / Sage / sparse-attention instance
# family. These are IR-build + LLVM-lower smoke tests (no GPU); the
# intent is to catch regressions in the builders + their lowering
# vocabulary, NOT to validate numerical correctness. End-to-end
# correctness + perf parity for these kernels is not covered by
# this suite yet -- see test_rocke_examples.py for the GPU-bound
# parity tests on the pre-existing kernels.
# ---------------------------------------------------------------------


class TestExtendedAttentionBuilds(unittest.TestCase):
    def _common(self, *, dtype="f16", mask_mode="none"):
        from rocke.instances import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
            dtype=dtype,
            mask_mode=mask_mode,
        )

    def _gqa_common(self):
        from rocke.instances import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=2),
            dtype="f16",
            mask_mode="causal",
        )

    def test_fmha_fwd_varlen_builds_and_lowers(self):
        from rocke.instances import FmhaFwdVarlenSpec, build_fmha_fwd_varlen

        spec = FmhaFwdVarlenSpec(
            common=self._common(mask_mode="causal"),
            max_seqlen_q=256,
            max_seqlen_k=256,
            batch=2,
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_varlen(spec))
        self.assertIn("@llvm.exp2.f32", ll)
        self.assertIn("define amdgpu_kernel", ll)

    def test_fmha_appendkv_with_rotary_emits_cos_sin_loads(self):
        from rocke.helpers.rotary import RotarySpec
        from rocke.instances import FmhaAppendKvSpec, build_fmha_fwd_appendkv

        spec = FmhaAppendKvSpec(
            common=self._common(),
            batch=2,
            rotary=RotarySpec(head_size=64, layout="half"),
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_appendkv(spec))
        self.assertGreaterEqual(ll.count("load float"), 32)

    def test_fmha_paged_prefill_builds(self):
        from rocke.instances import (
            FmhaFwdPagedPrefillSpec,
            build_fmha_fwd_paged_prefill,
        )

        spec = FmhaFwdPagedPrefillSpec(
            common=self._common(mask_mode="causal"),
            page_block_size=16,
            max_blocks_per_seq=32,
            batch=2,
        )
        kernel = build_fmha_fwd_paged_prefill(spec)
        self.assertIn("block_table", [p.name for p in kernel.params])

    def test_fmha_splitkv_decode_two_kernel_pipeline(self):
        from rocke.instances import (
            FmhaFwdSplitKvDecodeSpec,
            build_fmha_fwd_splitkv_decode_reduce,
            build_fmha_fwd_splitkv_decode_segment,
        )

        spec = FmhaFwdSplitKvDecodeSpec(
            common=self._common(),
            batch=4,
            num_segments=8,
        )
        seg = build_fmha_fwd_splitkv_decode_segment(spec)
        red = build_fmha_fwd_splitkv_decode_reduce(spec)
        self.assertEqual(
            sorted(p.name for p in seg.params)[:3],
            sorted(["Q", "K", "V"])[:3],
        )
        ll_red = lower_kernel_to_llvm(red)
        self.assertIn("@llvm.exp2.f32", ll_red)

    def test_fmha_head_grouping_builds_for_gqa(self):
        from rocke.instances import (
            FmhaFwdHeadGroupingSpec,
            build_fmha_fwd_head_grouping,
        )

        spec = FmhaFwdHeadGroupingSpec(
            common=self._gqa_common(),
            seqlen_q=128,
            seqlen_k=128,
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_head_grouping(spec))
        self.assertIn("@llvm.amdgcn.workgroup.id.z", ll)

    def test_fmha_bwd_uses_atomic_fadd(self):
        from rocke.instances import FmhaBwdSpec, build_fmha_bwd

        spec = FmhaBwdSpec(
            common=self._common(),
            seqlen_q=64,
            seqlen_k=64,
        )
        ll = lower_kernel_to_llvm(build_fmha_bwd(spec))
        # 3 atomic accumulators (dQ, dK, dV) per K-step per head dim.
        self.assertGreaterEqual(ll.count("atomicrmw fadd ptr addrspace(1)"), 3)

    def test_fmha_fwd_fp8_emits_cvt_fp8_intrinsic(self):
        from rocke.instances import FmhaFwdFp8Spec, build_fmha_fwd_fp8

        spec = FmhaFwdFp8Spec(
            common=self._common(),
            kv_dtype="fp8e4m3",
            seqlen_q=32,
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_fp8(spec))
        self.assertIn("@llvm.amdgcn.cvt.f32.fp8", ll)


class TestSageAttentionBuilds(unittest.TestCase):
    def _spec(self, quant_mode, *, head_size=64):
        from rocke.helpers.qk_scale import QkScaleSpec
        from rocke.instances import (
            FmhaCommonSpec,
            FmhaShape,
            SageAttentionSpec,
        )

        common = FmhaCommonSpec(
            shape=FmhaShape(head_size=head_size, num_query_heads=8, num_kv_heads=8),
            dtype="f16",
            mask_mode="none",
        )
        return SageAttentionSpec(
            common=common,
            quant_mode=quant_mode,
            q_scale=QkScaleSpec(
                layout="per_block",
                scale_block=16,
                stride_batch=128,
                stride_head=8,
                stride_block=1,
            ),
            k_scale=QkScaleSpec(
                layout="per_block",
                scale_block=64,
                stride_batch=128,
                stride_head=8,
                stride_block=1,
            ),
            seqlen_q=16,
            seqlen_k=64,
        )

    def test_fp16_baseline_no_fp8_cvt(self):
        from rocke.instances import build_sage_attention

        ll = lower_kernel_to_llvm(build_sage_attention(self._spec("fp16_bf16")))
        self.assertNotIn("@llvm.amdgcn.cvt.f32.fp8", ll)

    def test_fp8_variant_uses_fp8_cvt(self):
        from rocke.instances import build_sage_attention

        ll = lower_kernel_to_llvm(build_sage_attention(self._spec("fp8_bf16")))
        self.assertIn("@llvm.amdgcn.cvt.f32.fp8", ll)

    def test_int_variants_add_codebook_params(self):
        from rocke.instances import build_sage_attention

        # i4 sage needs head_size=128 (each lane owns one packed byte =
        # two nibbles); i8 sage works at head_size=64.
        for qm, hs in (("i8_fp8_bf16", 64), ("i4_fp8_bf16", 128)):
            kernel = build_sage_attention(self._spec(qm, head_size=hs))
            names = [p.name for p in kernel.params]
            self.assertIn("codebook_k", names, f"qm={qm}")
            self.assertIn("codebook_v", names, f"qm={qm}")


class TestSparseAttentionBuilds(unittest.TestCase):
    def _common(self):
        from rocke.instances import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
            dtype="f16",
            mask_mode="none",
        )

    def test_jenga_emits_mask_byte_guard(self):
        from rocke.instances import (
            JengaSparseSpec,
            build_jenga_sparse_attention,
        )

        spec = JengaSparseSpec(
            common=self._common(),
            seqlen_q=32,
            seqlen_k=128,
            block_q=1,
            block_k=32,
        )
        ll = lower_kernel_to_llvm(build_jenga_sparse_attention(spec))
        self.assertIn("load i8", ll)
        self.assertIn("icmp ne i8", ll)

    def test_vsa_loads_block_count_then_lut(self):
        from rocke.instances import (
            VsaSparseSpec,
            build_vsa_sparse_attention,
        )

        spec = VsaSparseSpec(
            common=self._common(),
            seqlen_q=32,
            seqlen_k=256,
            block_q=1,
            block_k=32,
            max_blocks_per_q=4,
        )
        ll = lower_kernel_to_llvm(build_vsa_sparse_attention(spec))
        self.assertGreaterEqual(ll.count("load i32"), 2)


class TestExtendedHelperBuilds(unittest.TestCase):
    def test_global_atomic_add_pk_bf16_intrinsic(self):
        from rocke.core.ir import BF16, I32, IRBuilder, PtrType

        b = IRBuilder("pk_bf16")
        ptr = b.param("p", PtrType(BF16, "global"))
        idx = b.param("i", I32)
        b.global_atomic_add_pk_bf16(ptr, idx, b.zero_vec(BF16, 2))
        b.ret()
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("@llvm.amdgcn.global.atomic.fadd.v2bf16", ll)

    def test_umul_hi_i32_lowers_via_zext_mul_lshr_trunc(self):
        from rocke.core.ir import I32, IRBuilder

        b = IRBuilder("umh")
        a = b.param("a", I32)
        c = b.param("c", I32)
        b.umul_hi_i32(a, c)
        b.ret()
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("zext i32", ll)
        self.assertIn("mul i64", ll)
        self.assertIn("lshr i64", ll)

    def test_rotary_apply_emits_fma_chain(self):
        from rocke.core.ir import F32, I32, IRBuilder, PtrType
        from rocke.helpers.rotary import (
            RotarySpec,
            apply_rotary_pair_f32,
            load_cos_sin,
        )

        b = IRBuilder("rot")
        cos_t = b.param("cos_t", PtrType(F32, "global"))
        sin_t = b.param("sin_t", PtrType(F32, "global"))
        pos = b.param("pos", I32)
        pair = b.param("pair", I32)
        cos_v, sin_v = load_cos_sin(
            b,
            cos_t,
            sin_t,
            token_pos=pos,
            pair_idx=pair,
            spec=RotarySpec(head_size=64, layout="half"),
        )
        apply_rotary_pair_f32(b, b.const_f32(1.0), b.const_f32(2.0), cos_v, sin_v)
        b.ret()
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("fmul float", ll)
        self.assertIn("fsub float", ll)

    def test_dropout_mask_pair_runs_full_philox(self):
        from rocke.core.ir import F32, I32, IRBuilder
        from rocke.helpers.rng import dropout_mask_pair_f32

        b = IRBuilder("drop")
        seed_lo = b.param("sl", I32)
        seed_hi = b.param("sh", I32)
        subseq = b.param("ss", I32)
        offset = b.param("off", I32)
        keep = b.param("keep", F32)
        dropout_mask_pair_f32(
            b,
            seed_lo=seed_lo,
            seed_hi=seed_hi,
            subseq=subseq,
            offset=offset,
            keep_prob_f32=keep,
        )
        b.ret()
        ll = lower_kernel_to_llvm(b.kernel)
        # 10 rounds of mulhilo = 20 mul i64 operations.
        self.assertGreaterEqual(ll.count("mul i64"), 20)

    def test_codebook_i8_emits_fp8_round(self):
        from rocke.core.ir import F32, I32, IRBuilder, PtrType
        from rocke.helpers.codebook import codebook_lookup_i8_to_fp8

        b = IRBuilder("cb")
        cb = b.param("cb", PtrType(F32, "global"))
        v = b.param("v", I32)
        codebook_lookup_i8_to_fp8(b, cb, v)
        b.ret()
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn("@llvm.amdgcn.cvt.pk.fp8.f32", ll)


class TestFmhaKernelBuilder(unittest.TestCase):
    """Tests for the FmhaKernelBuilder boilerplate-killer."""

    def _common(self):
        from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=2),
            dtype="f16",
            mask_mode="causal",
        )

    def test_signature_matches_old_varlen(self):
        """The builder-generated signature for fmha_varlen must match
        the canonical Q/K/V/O/cu/scale/total/batch/strides ABI exactly."""
        from rocke.instances.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe", self._common())
        kb.add_tensor("Q")
        kb.add_tensor("K")
        kb.add_tensor("V")
        kb.add_tensor("O")
        kb.add_ptr("cu_seqlens_q", dtype="i32")
        kb.add_ptr("cu_seqlens_k", dtype="i32")
        kb.add_scalar("scale_log2", "f32")
        kb.add_scalar("total_q", "i32")
        kb.add_scalar("batch", "i32")
        kb.add_strides("q", "k", "v", "o")
        sig = kb.signature()
        names = [item["name"] for item in sig]
        self.assertEqual(
            names,
            [
                "Q",
                "K",
                "V",
                "O",
                "cu_seqlens_q",
                "cu_seqlens_k",
                "scale_log2",
                "total_q",
                "batch",
                "stride_q_token",
                "stride_q_head",
                "stride_k_token",
                "stride_k_head",
                "stride_v_token",
                "stride_v_head",
                "stride_o_token",
                "stride_o_head",
            ],
        )

    def test_decode_grid_emits_gqa_div(self):
        """When ``num_queries_per_kv > 1`` the grid decode emits a
        divide on head_idx (otherwise it short-circuits to identity).
        """
        from rocke.instances.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe_grid", self._common())
        kb.add_scalar("scale_log2", "f32")
        kb.decode_grid()
        # head_idx = block_id_y, kv_head_idx = head_idx // 4 (HQ=8 / HK=2).
        # Lower and check the IR shows the divide.
        kb.builder.ret()
        ll = lower_kernel_to_llvm(kb.kernel)
        # The arith.div lowers to ``sdiv i32 ..., 4``.
        self.assertIn("sdiv i32", ll)

    def test_add_tensor_accepts_fp8_kv_dtype(self):
        """add_tensor with dtype='fp8e4m3' produces an fp8 pointer
        (used by fmha_fwd_fp8 / sage)."""
        from rocke.instances.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe_fp8", self._common())
        kb.add_tensor("K", dtype="fp8e4m3", align=8)
        sig = kb.signature()
        k_entry = next(item for item in sig if item["name"] == "K")
        # The "type" field renders as "ptr<fp8e4m3, global>".
        self.assertIn("fp8e4m3", k_entry["type"])

    def test_tensor_descriptor_naive_3d(self):
        """tensor_descriptor returns a 3-coord descriptor whose
        offset() works for an (token, head, d) triple."""
        from rocke.instances.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe_desc", self._common())
        kb.add_tensor("Q")
        kb.add_strides("q")
        desc = kb.tensor_descriptor("q")
        self.assertEqual(desc.upper_names, ("token", "head", "d"))


class TestMfmaGemm(unittest.TestCase):
    """Tests for the MFMA GEMM kernel."""

    def test_builds_emits_mfma_intrinsic(self):
        """The kernel IR must reference an AMDGCN MFMA intrinsic.

        Phase-1 ``mfma_flatmm`` worker switched ``mfma_gemm`` /
        ``flatmm`` from the legacy ``16x16x16`` atom to the K-packed
        ``16x16x32`` (and the ``32x32x16`` hero for larger tiles) on
        gfx950+ — see ``notes/instances/mfma_flatmm.md`` (1.33×–2.78×
        speedup, bit-exact parity). The test now guards the *MFMA-ness*
        of the kernel without pinning the atom shape.
        """
        from rocke.instances import MfmaGemmSpec, build_mfma_gemm

        spec = MfmaGemmSpec(M=32, N=32, K=32, dtype="f16")
        kernel = build_mfma_gemm(spec)
        ll = lower_kernel_to_llvm(kernel)
        # The default ``kpack=True`` selects the f16 16x16x32 atom on
        # gfx950+; fall back to a substring match so the test stays
        # honest if a future worker bumps the default again.
        self.assertIn("@llvm.amdgcn.mfma.f32.", ll)
        self.assertIn(".f16(", ll)

    def test_grid_one_cta_per_tile(self):
        """Grid math: (N//16, M//16, 1)."""
        from rocke.instances import MfmaGemmSpec, mfma_gemm_grid

        spec = MfmaGemmSpec(M=64, N=48, K=32)
        self.assertEqual(mfma_gemm_grid(spec), (3, 4, 1))

    def test_block_size_is_one_warp(self):
        """v1: one wave64 warp per CTA."""
        from rocke.instances import MfmaGemmSpec

        spec = MfmaGemmSpec(M=32, N=32, K=32)
        self.assertEqual(spec.block_size, 64)

    def test_spec_validation_rejects_non_atom_shape(self):
        """M / N / K must divide the active MFMA atom shape.

        With ``kpack=True`` (the new default) the active atom is
        ``16x16x32`` on the f16 path. ``M=17`` is still genuinely
        invalid (does not divide 16), so the validator should reject
        with an error string that mentions the *current* atom shape.
        """
        from rocke.instances import MfmaGemmSpec
        from rocke.instances.common.mfma_gemm import is_valid_spec

        # M=17 is invalid for both the legacy 16x16x16 atom and the new
        # 16x16x32 default — the test exercises the "non-atom shape"
        # validator path without coupling to the specific atom variant.
        spec = MfmaGemmSpec(M=17, N=32, K=32)
        ok, why = is_valid_spec(spec)
        self.assertFalse(ok)
        # The error message names the atom shape; assert on the
        # invariant prefix the validator produces (``"NxMxK atom
        # shape"`` — true regardless of whether the active atom is
        # 16x16x16 or 16x16x32).
        self.assertIn("atom shape", why)


class TestEveryKernelUsesMfma(unittest.TestCase):
    """The gremlin: assert every GEMM-shaped + attention kernel emits a
    real ``@llvm.amdgcn.mfma.*`` intrinsic in its LLVM IR.

    The kernels MUST use MFMA -- the warp-distributed scalar inner is
    not acceptable for production. This test scans the LLVM output and
    fails if the intrinsic is missing.
    """

    def _llvm_for(self, build_fn, spec):
        return lower_kernel_to_llvm(build_fn(spec))

    def test_mfma_gemm_uses_mfma(self):
        from rocke.instances import MfmaGemmSpec, build_mfma_gemm

        ll = self._llvm_for(
            build_mfma_gemm,
            MfmaGemmSpec(M=32, N=32, K=32, dtype="f16"),
        )
        # The "uses MFMA" gremlin: assert an AMDGCN MFMA intrinsic is
        # emitted regardless of the specific atom shape (legacy
        # 16x16x16 vs the new K-packed 16x16x32 — the phase-1
        # ``mfma_flatmm`` worker bumped the default).
        self.assertIn("@llvm.amdgcn.mfma.f32.", ll)

    def test_streamk_gemm_uses_mfma(self):
        from rocke.instances import StreamKGemmSpec, build_streamk_gemm

        spec = StreamKGemmSpec(
            M=32,
            N=32,
            K=32,
            tile_m=16,
            tile_n=16,
            tile_k=16,
            dtype="f16",
            num_cus=8,
            blocks_per_cu=1,
        )
        ll = self._llvm_for(build_streamk_gemm, spec)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16", ll)

    def test_block_scale_gemm_fp8_uses_mfma(self):
        from rocke.instances import BlockScaleGemmSpec
        from rocke.instances.common.block_scale_gemm import build_block_scale_gemm

        spec = BlockScaleGemmSpec(
            M=32,
            N=32,
            K=64,
            block_tile_m=16,
            block_tile_n=16,
            quant_mode="abquant",
            mantissa_dtype="fp8e4m3",
            group_size_mnk=(1, 1, 64),
        )
        ll = self._llvm_for(build_block_scale_gemm, spec)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.fp8", ll)

    def test_mx_gemm_fp8_uses_mfma(self):
        from rocke.instances import MxGemmSpec
        from rocke.instances.common.mx_gemm import build_mx_gemm

        spec = MxGemmSpec(M=16, N=16, K=64, mantissa_dtype="fp8e4m3")
        ll = self._llvm_for(build_mx_gemm, spec)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.fp8", ll)

    def test_fmha_mfma_uses_mfma(self):
        from rocke.instances import FmhaMfmaSpec, build_fmha_fwd_mfma
        from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape

        spec = FmhaMfmaSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=2, num_kv_heads=2),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=16,
            seqlen_k=16,
        )
        ll = self._llvm_for(build_fmha_fwd_mfma, spec)
        # Expect MFMA invocations for both QK and PV chains.
        n_mfma = ll.count("@llvm.amdgcn.mfma.f32.16x16x16")
        # head_size=64, atom.k=16 -> 4 QK atoms per K-tile + 4 PV atoms
        # per K-tile. Counting at least one is enough; the parity test
        # verifies the chain is correct.
        self.assertGreaterEqual(n_mfma, 1, f"got {n_mfma} MFMA calls")

    def test_universal_gemm_uses_mfma(self):
        """The pre-existing gemm_universal kernel underpins
        batched_gemm / grouped_gemm / flatmm / img2col (via implicit
        GEMM) / fused_moe; it must continue to emit MFMA."""
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
        )

        spec = UniversalGemmSpec(
            name="ugemm_smoke",
            tile=TileSpec(
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(),
            data=DataSpec(),
        )
        ll = self._llvm_for(build_universal_gemm, spec)
        self.assertTrue(
            "@llvm.amdgcn.mfma.f32" in ll,
            "gemm_universal must emit MFMA",
        )


if __name__ == "__main__":
    unittest.main()
