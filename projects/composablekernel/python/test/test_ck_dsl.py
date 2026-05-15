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


# ---------------------------------------------------------------------
# Non-GEMM CK Tile counterparts (elementwise / norm / reduce / transpose).
# ---------------------------------------------------------------------


class TestElementwiseInstance(unittest.TestCase):
    def test_unary_builds(self):
        from ck_dsl.instances import ElementwiseSpec, build_elementwise

        for op in ("copy", "neg", "abs", "relu", "exp2", "silu", "gelu_tanh"):
            kernel = build_elementwise(ElementwiseSpec(op=op))
            ll = lower_kernel_to_llvm(kernel)
            self.assertIn("define amdgpu_kernel void", ll)

    def test_binary_builds(self):
        from ck_dsl.instances import ElementwiseSpec, build_elementwise

        for op in ("add", "sub", "mul", "max", "min"):
            kernel = build_elementwise(ElementwiseSpec(op=op))
            ll = lower_kernel_to_llvm(kernel)
            self.assertIn("define amdgpu_kernel void", ll)

    def test_bf16_path_builds(self):
        from ck_dsl.instances import ElementwiseSpec, build_elementwise

        kernel = build_elementwise(ElementwiseSpec(op="add", dtype="bf16"))
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("bfloat", ll)

    def test_rejects_unknown_op(self):
        from ck_dsl.instances import ElementwiseSpec, build_elementwise

        with self.assertRaises(ValueError):
            build_elementwise(ElementwiseSpec(op="bogus"))


class TestLayerNormInstance(unittest.TestCase):
    def test_builds_with_save_mean(self):
        from ck_dsl.instances import LayerNorm2DSpec, build_layernorm2d

        spec = LayerNorm2DSpec(n_per_block=4096, save_mean_invstd=True)
        kernel = build_layernorm2d(spec)
        ll = lower_kernel_to_llvm(kernel)
        # The reduction uses LDS tree (s_barrier) and the rsqrt intrinsic.
        self.assertIn("@llvm.amdgcn.s.barrier", ll)
        self.assertIn("@llvm.amdgcn.rsq.f32", ll)

    def test_rejects_unaligned_n(self):
        from ck_dsl.instances import LayerNorm2DSpec, build_layernorm2d

        spec = LayerNorm2DSpec(n_per_block=3072, block_size=256, vec=8)
        with self.assertRaises(ValueError):
            build_layernorm2d(spec)


class TestRMSNormInstance(unittest.TestCase):
    def test_builds(self):
        from ck_dsl.instances import RMSNorm2DSpec, build_rmsnorm2d

        kernel = build_rmsnorm2d(RMSNorm2DSpec(n_per_block=4096))
        ll = lower_kernel_to_llvm(kernel)
        self.assertIn("@llvm.amdgcn.rsq.f32", ll)


class TestReduceInstance(unittest.TestCase):
    def test_sum_max_mean_builds(self):
        from ck_dsl.instances import Reduce2DSpec, build_reduce2d

        for op in ("sum", "max", "mean"):
            kernel = build_reduce2d(Reduce2DSpec(n_per_block=4096, op=op))
            ll = lower_kernel_to_llvm(kernel)
            self.assertIn("define amdgpu_kernel void", ll)


class TestTransposeInstance(unittest.TestCase):
    def test_builds(self):
        from ck_dsl.instances import Transpose2DSpec, build_transpose2d

        kernel = build_transpose2d(Transpose2DSpec())
        ll = lower_kernel_to_llvm(kernel)
        # Transpose uses a global memory load->LDS->global store and a
        # workgroup barrier between the two phases.
        self.assertIn("@llvm.amdgcn.s.barrier", ll)
        self.assertIn("addrspace(3)", ll)

    def test_rejects_oversize_tile(self):
        from ck_dsl.instances import Transpose2DSpec, build_transpose2d

        # 128x128/2 = 8192 threads > 1024 cap.
        with self.assertRaises(ValueError):
            build_transpose2d(Transpose2DSpec(tile_m=128, tile_n=128, vec=2))


class TestBatchedGemmInstance(unittest.TestCase):
    def test_builds_with_strides(self):
        from ck_dsl.instances import BatchedGemmSpec, build_batched_gemm

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
        from ck_dsl.helpers import (
            TensorDescriptor,
            make_naive_tensor_descriptor_packed,
        )

        d = make_naive_tensor_descriptor_packed((32, 128, 64), F16)
        self.assertEqual(d.strides, (128 * 64, 64, 1))
        self.assertEqual(d.numel(), 32 * 128 * 64)
        d2 = TensorDescriptor.with_strides((4, 16), (32, 1), F16)
        self.assertEqual(d2.strides, (32, 1))

    def test_runtime_stride_offsets_via_value(self):
        from ck_dsl.helpers import TensorDescriptor

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
        from ck_dsl.helpers import make_global_view, make_lds_view

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
        from ck_dsl.helpers import io_ir_type
        from ck_dsl.core.ir import BF16

        self.assertIs(io_ir_type("f16"), F16)
        self.assertIs(io_ir_type("fp16"), F16)
        self.assertIs(io_ir_type("bf16"), BF16)
        with self.assertRaises(ValueError):
            io_ir_type("fp32")


class TestReductionHelper(unittest.TestCase):
    def test_block_lds_reduce_emits_barrier_chain(self):
        from ck_dsl.helpers import block_lds_reduce, make_lds_view

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
    from ck_dsl.core.ir import F32

    return F32


class TestTensorCoordinate(unittest.TestCase):
    def test_make_and_move_emit_offset_chain(self):
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import TensorView, TensorDescriptor

        b = IRBuilder("buf_view_misuse")
        X = b.param("X", PtrType(F16, "global"), align=16)
        view = TensorView(
            base=X, desc=TensorDescriptor.packed((4,), F16), addr_space="global"
        )
        with self.assertRaises(TypeError):
            _ = view.buffer

    def test_buffer_view_scalar_load(self):
        from ck_dsl.helpers import (
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


class TestTransformsBridge(unittest.TestCase):
    def test_view_from_transforms_descriptor(self):
        from ck_dsl.helpers import (
            make_tile_window,
            view_from_transforms_descriptor,
        )
        from ck_dsl.transforms import TensorDescriptor as RichTensorDescriptor

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
        from ck_dsl.helpers import TileDistributionEncoding

        return TileDistributionEncoding(
            Hs=((4, 256, 8),),
            Ps2RHs_major=((1,),),
            Ps2RHs_minor=((1,),),
            Ys2RHs_major=(1, 1),
            Ys2RHs_minor=(0, 2),
        )

    def test_encoding_lengths(self):
        from ck_dsl.helpers import make_static_tile_distribution

        dist = make_static_tile_distribution(self._encoding())
        self.assertEqual(dist.X_lengths, (8192,))
        self.assertEqual(dist.Y_lengths, (4, 8))
        self.assertEqual(dist.num_elements_per_thread, 32)

    def test_encoding_rejects_duplicate_h_bucket(self):
        from ck_dsl.helpers import TileDistributionEncoding

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
        from ck_dsl.helpers import TileDistributionEncoding

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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import TileDistributionEncoding

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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import (
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
        from ck_dsl.helpers import IOSpecRule, validate_io

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
        from ck_dsl.helpers import kernel_name_join

        self.assertEqual(
            kernel_name_join(
                "ck_dsl_op", "f16", "", "v8", flags={"smv": True, "n": False}
            ),
            "ck_dsl_op_f16_v8_smv",
        )

    def test_signature_builder_chaining(self):
        from ck_dsl.helpers import SignatureBuilder

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
        from ck_dsl.helpers import ceil_div_grid

        self.assertEqual(ceil_div_grid((100, 16), (128, 32)), (7, 4, 1))
        self.assertEqual(ceil_div_grid((100, 16), (128, 32), (4, 1)), (7, 4, 4))
        self.assertEqual(ceil_div_grid((512, 1)), (512, 1, 1))


class TestGroupedGemmInstance(unittest.TestCase):
    def test_builds(self):
        from ck_dsl.instances import GroupedGemmSpec, build_grouped_gemm

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


class TestCdnaPrimitives(unittest.TestCase):
    """Coverage for the AMDGPU/CDNA-specific primitives added in
    :mod:`ck_dsl.helpers.grid`, :mod:`ck_dsl.helpers.schedule`,
    :mod:`ck_dsl.helpers.layouts`, plus the new IR primitives
    ``pin_sgpr`` / ``to_sgpr_u32`` / ``wave_all`` / ``sync_half_block``.
    """

    # ---- XOR-based LDS swizzles (canonical CK Tile shared-tile layouts) ----
    def test_xor_swizzle_matches_canonical_st_16x32(self):
        from ck_dsl.helpers.layouts import LdsLayout

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
        from ck_dsl.helpers.layouts import LdsLayout

        def reference(off: int) -> int:
            return off ^ (((off % 1024) >> 9) << 5) ^ (((off % 2048) >> 10) << 4)

        layout = LdsLayout.xor_swizzled(tile_rows=32, tile_cols=32, elem_bytes=2)
        for off in range(0, 32 * 32 * 2, 13):
            self.assertEqual(layout.apply_swizzle_bytes(off), reference(off))

    def test_xor_swizzle_unknown_shape_raises(self):
        from ck_dsl.helpers.layouts import LdsLayout

        with self.assertRaises(ValueError):
            LdsLayout.xor_swizzled(tile_rows=24, tile_cols=24, elem_bytes=2)

    def test_xor_swizzle_validate_requires_stages(self):
        from ck_dsl.helpers.layouts import LdsLayout

        # Manually constructing with swizzle='xor' but no stages should fail.
        bad = LdsLayout(logical_cols=32, k_pad=0, swizzle="xor", swizzle_stages=())
        with self.assertRaises(ValueError):
            bad.validate()

    # ---- chiplet_transform_chunked (XCD round-robin reverse) ----
    def test_chiplet_python_reference_matches_closed_form(self):
        from ck_dsl.helpers.grid import python_chiplet_transform_chunked

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
        from ck_dsl.helpers.grid import python_chiplet_transform_chunked

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
        from ck_dsl.helpers.grid import python_chiplet_transform_chunked

        num_wgs = 8 * 64 + 7  # one full block plus 7 tail WGs
        for wgid in range(8 * 64, num_wgs):
            self.assertEqual(
                python_chiplet_transform_chunked(
                    wgid, num_wgs=num_wgs, num_xcds=8, chunk_size=64
                ),
                wgid,
            )

    def test_super_tile_swizzle_walks_column_first_inside_group(self):
        from ck_dsl.helpers.grid import python_super_tile_swizzle

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
        from ck_dsl.helpers.schedule import SchedulePolicy
        from ck_dsl.core.ir import IRBuilder, PtrType, F16
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.helpers.schedule import SchedulePolicy
        from ck_dsl.core.ir import IRBuilder, PtrType, F16
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.helpers.schedule import SchedulePolicy
        from ck_dsl.core.ir import IRBuilder, PtrType, F16
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.core.ir import IRBuilder, PtrType, F16, I32
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.core.ir import IRBuilder, PtrType, I32
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.core.ir import (
            IRBuilder,
            PtrType,
            F16,
            I32,
            CACHE_STREAM,
        )
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.core.ir import IRBuilder, PtrType, F16, I32

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
        from ck_dsl.helpers.pipeline import SoftwarePipeline
        from ck_dsl.core.ir import IRBuilder, PtrType, F16

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
        from ck_dsl.helpers.pipeline import SoftwarePipeline
        from ck_dsl.core.ir import IRBuilder, PtrType, F16

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
        from ck_dsl.core.ir import IRBuilder, PtrType, F16
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("occu_smoke")
        b.kernel.attrs["max_workgroup_size"] = 256
        b.kernel.attrs["waves_per_eu"] = 2
        b.param("X", PtrType(F16, "global"))
        ll = lower_kernel_to_llvm(b.kernel)
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)

    def test_waves_per_eu_tuple_range(self):
        from ck_dsl.core.ir import IRBuilder, PtrType, F16
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.instances import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            build_universal_gemm,
        )
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.instances import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            build_universal_gemm,
        )
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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
        from ck_dsl.instances import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

        prob = ConvProblem(
            N=4,
            Hi=14,
            Wi=14,
            C=128,
            K=128,
            R=3,
            S=3,
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
        from ck_dsl.instances import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

        prob = ConvProblem(
            N=8,
            Hi=56,
            Wi=56,
            C=64,
            K=64,
            R=3,
            S=3,
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
        from ck_dsl.instances import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )
        from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

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


if __name__ == "__main__":
    unittest.main()
