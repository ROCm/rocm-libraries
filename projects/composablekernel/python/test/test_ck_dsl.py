# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the ck_dsl Python authoring layer.

Tests are organised mirror-of-layer:

  - `TestCoreIR`       : `ck_dsl.core.ir` - IR construction smoke tests
  - `TestLowering`     : `ck_dsl.core.lower_llvm` - LLVM IR text shape
  - `TestTransforms`   : `ck_dsl.transforms` - coord-transform DAG
  - `TestHelpers`      : `ck_dsl.helpers` - atoms, geometry, loads,
                          epilogues
  - `TestInstances`    : `ck_dsl.instances` - end-to-end build smoke
                          tests for the parametric kernels

These run in-process (no subprocess, no GPU); they test the static
IR/lowering pipeline only. End-to-end runtime tests live in
`test_ck_dsl_examples.py`.
"""

from __future__ import annotations

import unittest

from ck_dsl import (
    F16,
    I32,
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
from ck_dsl.helpers import (
    AsyncTileLoader,
    CoalescedTileLoader,
    KernelArtifact,
    LdsLayout,
    SchedulePolicy,
    SoftwarePipeline,
    WarpGrid,
    make_gemm_manifest,
)
from ck_dsl.instances import (
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
        self.assertEqual(isa.mfma, 1)
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
        from ck_dsl.instances import (
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
        # Cross-lane softmax reduction.
        self.assertIn("@llvm.amdgcn.ds.bpermute", ll)
        # NaN-guard select on neg_inf row max.
        self.assertIn("0xFFF0000000000000", ll)
        # `qq_bias_stride_0` is the very last kernel param.
        self.assertIn("i32 %qq_bias_stride_0", ll)

    def test_unified_attention_2d_tiled_alibi_qq_bias(self):
        """ALiBi/QQ-bias variants emit sitofp + masked global load with clamp."""
        from ck_dsl.instances import (
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
        from ck_dsl.instances import (
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
        from ck_dsl.instances import (
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

    def test_fp8_cvt_intrinsic_lowering(self):
        """fp8e4m3->f32 conversion lowers to llvm.amdgcn.cvt.f32.fp8."""
        from ck_dsl.core.ir import (
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
        from ck_dsl.instances import supports_tiled_2d

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
        # head_size=256, dtype=bf16, alibi, qq_bias all supported.
        for accept in [
            dict(head_size=256),
            dict(dtype="bf16"),
            dict(use_alibi=True),
            dict(use_qq_bias=True),
        ]:
            kwargs = dict(base)
            kwargs.update(accept)
            ok, reason = supports_tiled_2d(**kwargs)
            self.assertTrue(ok, msg=f"expected accept for {accept}, got: {reason}")
        # FP8 and unsupported block_size still gated.
        for override in [
            dict(block_size=32),
            dict(use_fp8=True),
        ]:
            kwargs = dict(base)
            kwargs.update(override)
            ok, reason = supports_tiled_2d(**kwargs)
            self.assertFalse(ok, msg=f"expected reject for {override}, got: {reason}")
            self.assertTrue(reason)


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
            N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
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
            N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
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

    def test_direct_conv_16c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv16cSpec(problem=prob, block_groups=4, fold_k32=True)
        kernel = build_direct_conv_16c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # Hot loop emits both 16x16x16 and 16x16x32 MFMAs (K=32 folded).
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll)

    def test_direct_conv_4c_builds(self):
        prob = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        spec = DirectConv4cSpec(problem=prob, block_q=8, block_groups=16)
        kernel = build_direct_conv_4c(spec)
        ll = lower_kernel_to_llvm(kernel)
        # 4x4x4 atom emits one MFMA per (r, s) tile (9 per output row).
        self.assertIn("@llvm.amdgcn.mfma.f32.4x4x4f16", ll)


if __name__ == "__main__":
    unittest.main()
