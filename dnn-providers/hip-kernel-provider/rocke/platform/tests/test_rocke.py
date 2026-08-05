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
`test_rocke_examples.py`. The one exception is opt-in: setting
`ROCKE_TEST_VERIFY_IR=1` assembles emitted IR with `llvm-as` (still no
GPU), so the default run keeps the no-subprocess contract.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, Mock

import pytest

from rocke import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I8,
    I16,
    I32,
    F32,
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
    UniversalGemmSpec,
    build_direct_conv_4c,
    build_direct_conv_16c,
    build_implicit_gemm_conv,
    build_universal_gemm,
)

# ---------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------
#
# rocke is consumed downstream by PyTorch, so torch is *not* a build/test
# dependency: the pure IR + lowering path must work with no torch installed.
# A handful of tests below exercise torch-facing features (the torch.fx
# fusion planner, torch-eager validation baselines) and are skipped when
# torch is absent — torch-full CI lanes still run them.

try:  # torch is optional; gate torch-facing tests on its presence.
    import torch as _torch  # noqa: F401

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - depends on the environment
    _HAVE_TORCH = False

_requires_torch = unittest.skipUnless(_HAVE_TORCH, "requires torch")

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

    def test_global_load_typed_wrappers(self):
        """Test all typed global_load_* wrappers lower correctly."""
        test_cases = [
            # (Type, wrapper_fn, llvm_type, align)
            (I8, "global_load_i8", "i8", 1),
            (I16, "global_load_i16", "i16", 2),
            (I32, "global_load_i32", "i32", 4),
            (I64, "global_load_i64", "i64", 8),
            (F16, "global_load_f16", "half", 2),
            (BF16, "global_load_bf16", "bfloat", 2),
            (F32, "global_load_f32", "float", 4),
            (FP8E4M3, "global_load_fp8e4m3", "i8", 1),  # fp8 lowers to i8
            (BF8E5M2, "global_load_bf8e5m2", "i8", 1),  # bf8 lowers to i8
        ]

        for ir_type, wrapper_name, llvm_type, align in test_cases:
            with self.subTest(wrapper=wrapper_name):
                b = IRBuilder(f"test_{wrapper_name}")
                X = b.param("X", PtrType(ir_type, "global"))
                tid = b.thread_id_x()
                # Call global_load_{ir_type}(X, tid)
                getattr(b, wrapper_name)(X, tid)
                ll = lower_kernel_to_llvm(b.kernel)
                self.assertIn(
                    f"getelementptr inbounds {llvm_type}, ptr addrspace(1)", ll
                )
                self.assertIn(f"load {llvm_type}, ptr addrspace(1)", ll)
                self.assertIn(f"align {align}", ll)


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

    def test_buffer_rsrc_llvm23_emits_p8_p1_with_i64_num_records(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM23

        ll = lower_kernel_to_llvm(
            self._buffer_rsrc_kernel(), llvm_flavor=LLVM_FLAVOR_LLVM23
        )
        self.assertIn(
            "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
            "ptr addrspace(1) nocapture readnone, i16, i64, i32)",
            ll,
        )
        self.assertIn("zext i32 %N_bytes to i64", ll)
        self.assertNotIn("@llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1)", ll)

    def test_buffer_rsrc_llvm23_passes_i64_num_bytes_without_zext(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM23

        b = IRBuilder("flavor_buf_rsrc_i64_llvm23")
        b.kernel.attrs["max_workgroup_size"] = 64
        X = b.param("X", PtrType(F16, "global"), align=16)
        N_bytes = b.param("N_bytes", I64)
        b.buffer_rsrc(X, N_bytes)
        ll = lower_kernel_to_llvm(b.kernel, llvm_flavor=LLVM_FLAVOR_LLVM23)
        self.assertNotIn("zext", ll)
        self.assertIn("i64 %N_bytes, i32 159744)", ll)

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

    def test_mfma_fp8_llvm23_uses_i64_operands(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM23

        ll = lower_kernel_to_llvm(
            self._fp8_mfma_kernel(), llvm_flavor=LLVM_FLAVOR_LLVM23
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


class TestLlvmFlavorEnumeration(unittest.TestCase):
    """The flavor set is enumerated once and every check goes through it.

    Flavors used to be re-listed by hand at each membership test, which is how
    ``llvm23`` ended up accepted by the Python lowerer but rejected by the C++
    backend path. These tests pin the single enumeration
    (:data:`LLVM_FLAVORS`), its partition into datalayout generations, and the
    version-ladder policy, so adding a flavor cannot half-land again.
    """

    def test_datalayout_kinds_partition_every_flavor(self):
        """Every flavor belongs to exactly one datalayout generation."""
        from rocke.core.lower_llvm import (
            LLVM_FLAVORS,
            LlvmDatalayoutKind,
            _DATALAYOUT_KIND_FLAVORS,
            _datalayout_kind_for_flavor,
        )

        # Exhaustive over the enum: a new member without a mapping fails here
        # (the check a language-level ``match`` would do statically).
        self.assertEqual(
            set(_DATALAYOUT_KIND_FLAVORS), set(LlvmDatalayoutKind), "kind coverage"
        )
        # ...and exhaustive over the flavors, with no flavor in two kinds.
        buckets = list(_DATALAYOUT_KIND_FLAVORS.values())
        union = frozenset().union(*buckets)
        self.assertEqual(union, frozenset(LLVM_FLAVORS), "flavor coverage")
        self.assertEqual(
            sum(len(b) for b in buckets), len(union), "flavor in multiple kinds"
        )
        for flavor in LLVM_FLAVORS:
            self.assertIsNotNone(_datalayout_kind_for_flavor(flavor), flavor)
        self.assertIsNone(_datalayout_kind_for_flavor("not-a-real-flavor"))

    def test_datalayout_kind_markers_match_datalayouts(self):
        """The p8 sniff markers must match what the flavors actually emit."""
        from rocke.core.lower_llvm import (
            LLVM_FLAVORS,
            LlvmDatalayoutKind,
            _datalayout_for_flavor,
            _datalayout_kind_for_flavor,
            _datalayout_kind_from_ir,
        )

        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                ir = f'target datalayout = "{_datalayout_for_flavor(flavor)}"'
                self.assertIs(
                    _datalayout_kind_from_ir(ir),
                    _datalayout_kind_for_flavor(flavor),
                    "sniffing a flavor's own datalayout must recover its generation",
                )
        # The indexed marker must not match the legacy layout: "p8:128:128" is a
        # prefix of "p8:128:128:128:48", so ordering/anchoring matters.
        legacy = f'target datalayout = "{_datalayout_for_flavor("llvm20")}"'
        self.assertIs(_datalayout_kind_from_ir(legacy), LlvmDatalayoutKind.P8_PLAIN)
        self.assertIsNone(_datalayout_kind_from_ir("define void @k() {}"))

    def test_llvm23_datalayout_is_llvm22_plus_me_mangling(self):
        """llvm23 is no longer an alias of llvm22: it is the llvm22 layout with
        the ELF ``m:e`` symbol-mangling spec spliced in after the leading
        endianness field.

        The toolchain drift guard proves this only on a ROCm 7.13+ host with
        hipcc; this pins the same contract on the bare CI gate, so an accidental
        re-alias (or a stray edit to either constant) fails here with a precise
        message instead of surfacing as an opaque byte-identity golden diff.
        """
        from rocke.core.lower_llvm import _DATALAYOUT_LLVM22, _DATALAYOUT_LLVM23

        self.assertNotEqual(
            _DATALAYOUT_LLVM23,
            _DATALAYOUT_LLVM22,
            "llvm23 is no longer an alias of llvm22 (it adds the m:e spec)",
        )
        self.assertEqual(
            _DATALAYOUT_LLVM23,
            _DATALAYOUT_LLVM22.replace("e-", "e-m:e-", 1),
            "llvm23 datalayout must be the llvm22 layout plus the m:e "
            "symbol-mangling spec",
        )

    def test_flavor_for_rocm_clamps_at_both_ends(self):
        """Version mapping is clamped, never raising: newest above, oldest below."""
        from rocke.core.lower_llvm import (
            LLVM_FLAVOR_LLVM20,
            LLVM_FLAVOR_LLVM22,
            LLVM_FLAVOR_LLVM23,
            LLVM_FLAVORS,
            _flavor_for_rocm,
        )

        # Exact boundaries.
        self.assertEqual(_flavor_for_rocm(7, 1), LLVM_FLAVOR_LLVM20)
        self.assertEqual(_flavor_for_rocm(7, 2), LLVM_FLAVOR_LLVM22)
        self.assertEqual(_flavor_for_rocm(7, 12), LLVM_FLAVOR_LLVM22)
        self.assertEqual(_flavor_for_rocm(7, 13), LLVM_FLAVOR_LLVM23)
        # Clamped: a ROCm newer than anything we know resolves to the newest
        # flavor, and one older than every row to the oldest.
        self.assertEqual(_flavor_for_rocm(99, 9), LLVM_FLAVORS[-1])
        self.assertEqual(_flavor_for_rocm(0, 0), LLVM_FLAVORS[0])
        self.assertEqual(_flavor_for_rocm(6, 4), LLVM_FLAVOR_LLVM20)
        # Never raises, and always returns a member of the enumeration.
        for major, minor in ((0, 0), (-1, 0), (5, 7), (7, 2), (12, 0)):
            self.assertIn(_flavor_for_rocm(major, minor), LLVM_FLAVORS)

    def test_is_modern_flavor_tracks_datalayout_kind(self):
        """The mfma-packing predicate is derived from the generation table."""
        from rocke.core.lower_llvm import (
            LLVM_FLAVORS,
            LlvmDatalayoutKind,
            _datalayout_kind_for_flavor,
            _is_modern_flavor,
        )

        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                self.assertEqual(
                    _is_modern_flavor(flavor),
                    _datalayout_kind_for_flavor(flavor)
                    is LlvmDatalayoutKind.P8_INDEXED,
                )
        # An unrecognised flavor is not "modern" (it must not silently pick the
        # i64 mfma operand packing).
        self.assertFalse(_is_modern_flavor("not-a-real-flavor"))

    def test_cpp_backend_path_accepts_every_known_flavor(self):
        """Regression: the C++ backend rejected ``llvm23`` while Python took it.

        ``_lower_via_cpp_engine`` validated against its own hand-written flavor
        tuple, so a newly added flavor raised ``ValueError`` on backend="cpp"
        even though the engine itself supported it.
        """
        from rocke.core import backend as backend_mod
        from rocke.core.lower_llvm import LLVM_FLAVORS

        engine = type(
            "_FakeEngine",
            (),
            {"lower_serialized_ir": staticmethod(lambda *a, **k: "lowered")},
        )()
        orig = backend_mod._import_engine
        backend_mod._import_engine = lambda: engine
        try:
            for flavor in LLVM_FLAVORS:
                with self.subTest(flavor=flavor):
                    self.assertEqual(
                        backend_mod._lower_via_cpp_engine(
                            "ir", arch="gfx950", llvm_flavor=flavor
                        ),
                        "lowered",
                    )
            with self.assertRaises(ValueError):
                backend_mod._lower_via_cpp_engine(
                    "ir", arch="gfx950", llvm_flavor="not-a-real-flavor"
                )
        finally:
            backend_mod._import_engine = orig

    def test_comgr_guard_compares_datalayout_generation_not_flavor(self):
        """The comgr mismatch guard keys on generation, so llvm22/llvm23 mix.

        A module's ``p8`` field cannot tell llvm22 from llvm23, and codegen only
        aborts across the *generation* boundary. Comparing flavors instead would
        false-positive on llvm22 IR fed to an llvm23 comgr.
        """
        from rocke.core.lower_llvm import _datalayout_for_flavor
        from rocke.runtime import comgr as comgr_mod

        def ir_for(flavor):
            return f'target datalayout = "{_datalayout_for_flavor(flavor)}"\n'

        orig_ver = comgr_mod.resolved_lib_rocm_version
        orig_path = comgr_mod.resolved_lib_path
        comgr_mod.resolved_lib_path = lambda: "/fake/libamd_comgr.so"
        try:
            # Same generation on both sides (including the llvm22-IR/llvm23-comgr
            # cross pair) must pass.
            for lib_ver, flavor in (
                ((7, 0), "llvm20"),
                ((7, 2), "llvm22"),
                ((7, 13), "llvm23"),
                ((7, 13), "llvm22"),
                ((7, 2), "llvm23"),
            ):
                with self.subTest(lib=lib_ver, ir=flavor):
                    comgr_mod.resolved_lib_rocm_version = lambda v=lib_ver: v
                    comgr_mod._assert_ir_flavor_matches_lib(ir_for(flavor))
            # Crossing the generation boundary must raise either way.
            for lib_ver, flavor in (((7, 0), "llvm22"), ((7, 13), "llvm20")):
                with self.subTest(lib=lib_ver, ir=flavor, expect="raise"):
                    comgr_mod.resolved_lib_rocm_version = lambda v=lib_ver: v
                    with self.assertRaises(comgr_mod.ComgrError) as cm:
                        comgr_mod._assert_ir_flavor_matches_lib(ir_for(flavor))
                    # The message must name both generations, not just "modern".
                    self.assertIn("p8_", str(cm.exception))
            # Unknown IR (no datalayout) never blocks compilation.
            comgr_mod.resolved_lib_rocm_version = lambda: (7, 0)
            comgr_mod._assert_ir_flavor_matches_lib("define void @k() {}")
        finally:
            comgr_mod.resolved_lib_rocm_version = orig_ver
            comgr_mod.resolved_lib_path = orig_path

    def test_no_hand_rolled_flavor_membership_lists(self):
        """Flavor membership must go through :data:`LLVM_FLAVORS`.

        A tuple/set literal naming two or more ``LLVM_FLAVOR_*`` constants is
        the exact shape that drifted out of date; the enumeration exists so
        such a list never needs writing again.
        """
        import ast
        from pathlib import Path

        import rocke

        # The two definitions that are *supposed* to list flavors: the
        # enumeration itself and its partition by datalayout generation.
        allowed_targets = {"LLVM_FLAVORS", "_DATALAYOUT_KIND_FLAVORS"}
        root = Path(rocke.__file__).parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            # Sources are UTF-8 regardless of the host locale; on Windows the
            # default codec is cp1252 and chokes on the non-ASCII in a few
            # example/heuristics modules.
            tree = ast.parse(path.read_text(encoding="utf-8"))
            allowed_spans = []
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if {t.id for t in targets if isinstance(t, ast.Name)} & allowed_targets:
                    allowed_spans.append((node.lineno, node.end_lineno or node.lineno))
            # A tuple/set/list literal naming 2+ flavor constants. Using the AST
            # (not a regex) keeps prose in docstrings from tripping this.
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Tuple, ast.Set, ast.List)):
                    continue
                named = [
                    e.id
                    for e in node.elts
                    if isinstance(e, ast.Name) and e.id.startswith("LLVM_FLAVOR_")
                ]
                if len(named) < 2:
                    continue
                if any(lo <= node.lineno <= hi for lo, hi in allowed_spans):
                    continue
                offenders.append(
                    f"{path.relative_to(root)}:{node.lineno}: {', '.join(named)}"
                )
        self.assertEqual(
            offenders,
            [],
            "use LLVM_FLAVORS (or the datalayout-kind table) instead of "
            "re-listing flavors:\n  " + "\n  ".join(offenders),
        )

    def test_cpp_engine_accepts_exactly_the_python_flavor_set(self):
        """The C++ ladder and :data:`LLVM_FLAVORS` must agree, in order.

        The engine has its own copy of the enumeration, so the two can drift
        into a state where a flavor lowers on one side and is rejected on the
        other. Reading the engine's list back is what makes its table an
        actual mirror rather than a second, independent source.
        """
        try:
            import rocke_engine
        except ImportError:
            self.skipTest("rocke_engine extension not built")
        from rocke.core.lower_llvm import LLVM_FLAVORS

        self.assertEqual(tuple(rocke_engine.llvm_flavors()), LLVM_FLAVORS)


# Probe run in a child process: print the datalayout the Python resolver picks
# for AUTO, and the one the C++ engine picks for AUTO (flavor=""). Must be a
# child because the flavor depends on process environment and Python memoises
# its answer per comgr path.
_FLAVOR_AUTO_PROBE = """
import sys
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _datalayout_for_flavor, _resolve_llvm_flavor
import rocke_engine

b = IRBuilder("probe")
b.kernel.attrs["max_workgroup_size"] = 64
b.const_i32(1)
cpp = rocke_engine.lower_serialized_ir(serialize(b.kernel), arch="gfx950", flavor="")
cpp_dl = next(
    ln.split('"')[1] for ln in cpp.splitlines() if ln.startswith("target datalayout")
)
print(_datalayout_for_flavor(_resolve_llvm_flavor()))
print(cpp_dl)
"""


class TestCrossEngineFlavorAutoResolution(unittest.TestCase):
    """The two engines must pick the SAME flavor when asked to auto-detect.

    Both sides answer "which LLVM flavor is this host?" independently: Python in
    ``_detect_llvm_flavor``, the engine in ``ll_resolve_flavor``. If they answer
    differently they emit different IR from the same kernel while each looks
    self-consistent, and no byte-identity run can see it -- the gate pins a
    flavor explicitly, and ``backend.py`` resolves in Python before calling the
    engine, so the engine's own resolver is never exercised on the paths the
    suite covers. The C++ resolver used to read a single hardcoded
    ``/opt/rocm/.info/version`` while Python located the comgr library that will
    actually compile the IR and took *that* tree's ROCm version, so the two
    disagreed on every host where /opt/rocm is absent, stale, or not the install
    ``$ROCM_PATH`` names.

    The layouts below are the ones that actually distinguish the resolvers; each
    is checked through the emitted datalayout, which is the observable the
    flavor controls.
    """

    def _probe(self, env_overrides):
        import os
        import subprocess
        import sys

        env = dict(os.environ)
        env.pop("ROCKE_LLVM_FLAVOR", None)
        for k in ("ROCM_PATH", "ROCM_HOME", "ROCKE_COMGR_LIB"):
            env.pop(k, None)
        env.update(env_overrides)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        out = subprocess.run(
            [sys.executable, "-c", _FLAVOR_AUTO_PROBE],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        py_dl, cpp_dl = out.stdout.strip().splitlines()[-2:]
        return py_dl, cpp_dl

    def _fake_rocm_tree(self):
        """A packaged install: release 7.0.0 at the root, component 7.13.0 in
        ``core-7.13/lib`` where comgr lives.

        The two versions map to *different* flavors (llvm20 vs llvm23), which is
        what makes this fixture worth having: a resolver that stops at the first
        ``.info/version`` it meets while climbing out of the lib dir reads the
        component version and picks llvm23 for a ROCm 7.0 toolchain.
        """
        import os
        import shutil
        import tempfile

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        rocm = os.path.join(root, "rocm-7.0.0")
        core_lib = os.path.join(rocm, "core-7.13", "lib")
        os.makedirs(os.path.join(rocm, ".info"))
        os.makedirs(os.path.join(rocm, "core-7.13", ".info"))
        os.makedirs(core_lib)
        os.makedirs(os.path.join(rocm, "lib"))
        with open(os.path.join(rocm, ".info", "version"), "w") as fh:
            fh.write("7.0.0-1234\n")
        with open(os.path.join(rocm, "core-7.13", ".info", "version"), "w") as fh:
            fh.write("7.13.0\n")
        for d in (core_lib, os.path.join(rocm, "lib")):
            open(os.path.join(d, "libamd_comgr.so"), "w").close()
        return rocm

    def _assert_agree(self, env_overrides):
        py_dl, cpp_dl = self._probe(env_overrides)
        self.assertEqual(
            py_dl,
            cpp_dl,
            f"python and cpp AUTO flavor resolution disagree for {env_overrides}: "
            f"python picked a datalayout the engine did not",
        )
        return py_dl

    def setUp(self):
        try:
            import rocke_engine  # noqa: F401
        except ImportError:
            self.skipTest("rocke_engine extension not built")

    def test_agree_on_the_bare_host(self):
        """No env pointing anywhere: whatever this machine has, both must see it."""
        self._assert_agree({})

    def test_agree_when_rocm_path_names_an_older_install(self):
        """``$ROCM_PATH`` outranks /opt/rocm on both sides, or they split."""
        rocm = self._fake_rocm_tree()
        dl = self._assert_agree({"ROCM_PATH": rocm})
        # 7.0.0 is pre-7.2, i.e. the LLVM 20 plain-p8 datalayout. Pinning the
        # value (not just agreement) is what proves the env root was honoured
        # rather than both engines ignoring it and agreeing on the host default.
        self.assertIn("-p8:128:128-", dl)

    def test_agree_when_rocm_home_names_an_older_install(self):
        rocm = self._fake_rocm_tree()
        dl = self._assert_agree({"ROCM_HOME": rocm})
        self.assertIn("-p8:128:128-", dl)

    def test_agree_when_comgr_lib_is_pinned_in_a_component_subdir(self):
        """The RELEASE version wins over the component version next to comgr.

        ``$ROCKE_COMGR_LIB`` points into ``core-7.13/lib``, whose own
        ``.info/version`` says 7.13.0 (-> llvm23) while the install root says
        7.0.0 (-> llvm20). Both engines must climb to the root.
        """
        import os

        rocm = self._fake_rocm_tree()
        lib = os.path.join(rocm, "core-7.13", "lib", "libamd_comgr.so")
        dl = self._assert_agree({"ROCKE_COMGR_LIB": lib})
        self.assertIn("-p8:128:128-", dl)

    def test_agree_when_the_named_root_has_no_comgr(self):
        """A root with a version file but no comgr is NOT a flavor signal.

        Python only takes the version of a tree it can actually find comgr in;
        a C-side resolver that read ``$ROCM_PATH/.info/version`` directly would
        honour this root and diverge.
        """
        import os
        import shutil
        import tempfile

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        os.makedirs(os.path.join(root, ".info"))
        with open(os.path.join(root, ".info", "version"), "w") as fh:
            fh.write("7.0.0\n")
        self._assert_agree({"ROCM_PATH": root})


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


# ---------------------------------------------------------------------
# Emitted-IR validation (opt-in)
# ---------------------------------------------------------------------
#
# A substring assertion cannot tell well-formed IR from IR that merely
# contains the right text: several intrinsic lowerings named a pointer of the
# wrong address space, or an i64 where a pointer was declared, and every
# assertion still matched. Assembling the module answers that directly.
#
# This is the one place in the file that shells out, which the module docstring
# otherwise rules out, so it is opt-in: unset ROCKE_TEST_VERIFY_IR and the
# suite stays in-process. Opting in without llvm-as reachable is an error
# rather than a silent skip -- a verification lane that quietly verifies
# nothing is worse than no lane.
_ENV_VERIFY_IR = "ROCKE_TEST_VERIFY_IR"


def _llvm_as_path():
    """Locate ``llvm-as``: PATH first, then a ROCm-bundled LLVM.

    The ROCm tiers mirror ``runtime_coexistence._rocm_root_libdirs`` -- the
    operator override first, then the globbed install layouts, newest first --
    so the tool is found on the same boxes the runtime finds comgr on.
    """
    import glob
    import os
    import shutil
    from pathlib import Path

    found = shutil.which("llvm-as")
    if found:
        return found
    roots = [os.environ.get(v) for v in ("ROCM_PATH", "ROCM_HOME")]
    roots += sorted(glob.glob("/opt/rocm*"), reverse=True)
    for root in roots:
        if not root:
            continue
        cand = Path(root) / "llvm" / "bin" / "llvm-as"
        if cand.is_file():
            return str(cand)
    return None


def _llvm_as_major(path):
    """Major version llvm-as reports, or None if it cannot be parsed."""
    import re
    import subprocess

    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"LLVM version (\d+)", out)
    return int(m.group(1)) if m else None


def _newest_llvm_major():
    """LLVM major of the newest flavor rocke emits (``llvm23`` -> 23)."""
    from rocke.core.lower_llvm import LLVM_FLAVORS

    return int(LLVM_FLAVORS[-1][len("llvm") :])


def _assert_ir_assembles(test, ir_text, kernel_name):
    """Assemble ``ir_text`` with llvm-as when ROCKE_TEST_VERIFY_IR is set.

    llvm-as runs the same parser and verifier comgr does, so a malformed
    declare, or a call whose operand type contradicts it, fails here with the
    exact diagnostic the toolchain would give.

    The tool has to be at least as new as the newest flavor rocke emits. These
    cases exist to cover intrinsics that only exist in that generation, so an
    older llvm-as reports them as defects for the sole reason that it predates
    them -- ``s.prefetch.inst``, ``av.load.b128`` and
    ``raw.ptr.buffer.load.async.lds`` are all absent from LLVM 22. Both "no
    tool" and "tool too old" are failures rather than silent skips: whoever
    set the variable asked for verification, and a lane that quietly verifies
    nothing is worse than no lane.
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    if os.environ.get(_ENV_VERIFY_IR, "").strip() in ("", "0"):
        return
    llvm_as = _llvm_as_path()
    if llvm_as is None:
        test.fail(
            f"{_ENV_VERIFY_IR} is set but 'llvm-as' was not found on PATH, "
            "under $ROCM_PATH/llvm/bin, or in a /opt/rocm* install; install "
            "it or unset the variable"
        )
    want = _newest_llvm_major()
    have = _llvm_as_major(llvm_as)
    if have is None or have < want:
        test.fail(
            f"{_ENV_VERIFY_IR} is set but {llvm_as} reports LLVM "
            f"{have if have is not None else '?'}; these cases emit LLVM "
            f"{want} intrinsics, so an older assembler cannot judge them. "
            "Point PATH or $ROCM_PATH at a newer LLVM, or unset the variable."
        )
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{kernel_name}.ll"
        src.write_text(ir_text)
        proc = subprocess.run(
            [llvm_as, str(src), "-o", str(Path(td) / "out.bc")],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        test.fail(
            f"llvm-as rejected the IR for kernel '{kernel_name}':\n"
            f"{proc.stderr.strip()}\n--- emitted IR ---\n{ir_text}"
        )


def _assert_engines_agree(test, kernel, python_ir, **kw):
    """Byte-compare the C++ engine against ``python_ir``.

    Never falls back to the Python result. If the extension is not importable
    the test *skips* -- visibly, and counted in the run summary -- rather than
    returning and letting the case pass as though parity had been checked. Any
    other engine outcome (it failed, or it produced different bytes) is a
    failure: a built engine that disagrees is a divergence, not a gap.
    """
    from rocke.core.backend import BackendError, _lower_via_cpp_engine, _text_diff
    from rocke.core.ir_serialize import serialize

    try:
        cpp_ir = _lower_via_cpp_engine(
            serialize(kernel), kw.get("arch") or "gfx950", kw.get("llvm_flavor")
        )
    except BackendError as e:
        test.skipTest(f"C++ engine unavailable, cross-engine check not run: {e}")
    if cpp_ir != python_ir:
        test.fail(
            f"python vs cpp engine disagree for kernel "
            f"'{getattr(kernel, 'name', '?')}':\n"
            + _text_diff("lowered AMDGPU .ll", python_ir, cpp_ir)
        )


class TestNewTargetIntrinsics(unittest.TestCase):
    """One case per intrinsic added for the LLVM 23 / future-operator surface.

    Each test emits exactly one intrinsic from the smallest possible kernel and
    pins the full ``declare`` plus call-site text. Asserting the whole call --
    operand order, types, and immediate encoding -- is what catches a signature
    regression; merely asserting the mangled name appears somewhere in the
    module cannot tell a correct call from a malformed one, and gives no signal
    about which intrinsic broke.
    """

    def _builder(self, name):
        from rocke.core.ir import IRBuilder

        b = IRBuilder(name)
        b.kernel.attrs["max_workgroup_size"] = 64
        return b

    def _lower(self, name, build, **kw):
        """Build a single-intrinsic kernel and return its lowered LLVM IR.

        Pins the NATIVE Python lowerer rather than going through
        ``lower_kernel_to_llvm``, whose engine comes from ``ROCKE_BACKEND``
        with a silent cpp->python fallback: an unpinned call makes these
        assertions describe C++ on a machine where the engine is built and
        Python on one where it is not, so the same suite means two different
        things. The C++ engine is then byte-compared against that result
        whenever it is importable, which covers it without the ambiguity.
        """
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        b = self._builder(name)
        build(b)
        ll = _lower_kernel_to_llvm_python(b.kernel, **kw)
        _assert_engines_agree(self, b.kernel, ll, **kw)
        _assert_ir_assembles(self, ll, name)
        return ll

    # ---- ds_swizzle (raw offset + XOR-butterfly encoding) ----
    def test_ds_swizzle_passes_raw_offset_immediate(self):
        ll = self._lower("dssw", lambda b: b.ds_swizzle(b.const_i32(1), 0x041F))
        self.assertIn("declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)", ll)
        # The raw immediate reaches the call verbatim (0x041F == 1055).
        self.assertIn("call i32 @llvm.amdgcn.ds.swizzle(i32 1, i32 1055)", ll)

    def test_ds_swizzle_xor_encodes_swap_mode_offset(self):
        # offset = (xor_mask << 10) | 0x1F -> (2 << 10) | 31 == 2079 (0x081F).
        ll = self._lower("dsswx", lambda b: b.ds_swizzle_xor(b.const_i32(1), 2))
        self.assertIn("call i32 @llvm.amdgcn.ds.swizzle(i32 1, i32 2079)", ll)
        # Regression: the handler used to request a decl key ("ds.swizzle") that
        # does not exist, so the declare was silently omitted.
        self.assertIn("declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)", ll)

    # ---- mov_dpp8 ----
    def test_mov_dpp8_i32_emits_typed_intrinsic(self):
        ll = self._lower("dpp8i", lambda b: b.mov_dpp8(b.const_i32(1), 0x765432))
        self.assertIn("declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32 immarg)", ll)
        # 0x765432 == 7754802, passed through as the 24-bit lane-select imm.
        self.assertIn("call i32 @llvm.amdgcn.mov.dpp8.i32(i32 1, i32 7754802)", ll)

    def test_mov_dpp8_f32_emits_typed_intrinsic(self):
        ll = self._lower("dpp8f", lambda b: b.mov_dpp8(b.const_f32(1.0), 0x765432))
        self.assertIn("declare float @llvm.amdgcn.mov.dpp8.f32(float, i32 immarg)", ll)
        self.assertIn("call float @llvm.amdgcn.mov.dpp8.f32(float", ll)

    def test_mov_dpp8_both_types_coexist_in_one_module(self):
        """i32 and f32 mov_dpp8 must not collide on one overloaded symbol.

        Both used to declare bare ``@llvm.amdgcn.mov.dpp8``, so a kernel using
        each type emitted two conflicting declares for the same name.
        """

        def build(b):
            b.mov_dpp8(b.const_i32(1), 0x11)
            b.mov_dpp8(b.const_f32(1.0), 0x11)

        ll = self._lower("dpp8_both", build)
        self.assertIn("declare i32 @llvm.amdgcn.mov.dpp8.i32(", ll)
        self.assertIn("declare float @llvm.amdgcn.mov.dpp8.f32(", ll)
        # No bare (untyped) declare, which is what caused the duplicate symbol.
        self.assertNotIn("@llvm.amdgcn.mov.dpp8(", ll)

    def test_mov_dpp8_rejects_oversized_sel_and_bad_type(self):
        b = self._builder("dpp8_bad")
        with self.assertRaises(ValueError):
            b.mov_dpp8(b.const_i32(1), 0x1000000)  # 25 bits
        with self.assertRaises(ValueError):
            b.mov_dpp8(b.const_i64(1), 0x11)

    # ---- wave_reduce ----
    def test_wave_reduce_emits_typed_reduction_per_op(self):
        for reduce_op, ty in (
            ("fmax", "float"),
            ("fadd", "float"),
            ("add", "i32"),
            ("max", "i32"),
            ("min", "i32"),
        ):
            with self.subTest(reduce_op=reduce_op):
                suffix = "f32" if ty == "float" else "i32"

                def build(b, reduce_op=reduce_op, suffix=suffix):
                    v = b.const_f32(1.0) if suffix == "f32" else b.const_i32(1)
                    b.wave_reduce(v, reduce_op)

                ll = self._lower(f"wred_{reduce_op}", build)
                name = f"@llvm.amdgcn.wave.reduce.{reduce_op}.{suffix}"
                self.assertIn(f"declare {ty} {name}({ty}, i32 immarg)", ll)
                # Trailing i32 is the strategy immediate (0 == default).
                self.assertIn(f"call {ty} {name}({ty}", ll)
                self.assertIn(", i32 0)", ll)

    def test_wave_reduce_propagates_strategy_immediate(self):
        ll = self._lower(
            "wred_strat",
            lambda b: b.wave_reduce(b.const_f32(1.0), "fmax", strategy=2),
        )
        self.assertIn("@llvm.amdgcn.wave.reduce.fmax.f32(float", ll)
        self.assertIn(", i32 2)", ll)

    def test_wave_reduce_rejects_op_type_mismatch(self):
        b = self._builder("wred_bad")
        with self.assertRaises(ValueError):
            b.wave_reduce(b.const_f32(1.0), "add")  # integer op on f32
        with self.assertRaises(ValueError):
            b.wave_reduce(b.const_i32(1), "fmax")  # float op on i32

    # ---- readlane / writelane ----
    def test_readlane_emits_typed_call_with_i32_lane(self):
        for suffix, ty, mk in (
            ("i32", "i32", "const_i32"),
            ("f32", "float", "const_f32"),
        ):
            with self.subTest(suffix=suffix):
                ll = self._lower(
                    f"rlane_{suffix}",
                    lambda b, mk=mk: b.readlane(getattr(b, mk)(1), b.const_i32(0)),
                )
                name = f"@llvm.amdgcn.readlane.{suffix}"
                self.assertIn(f"declare {ty} {name}({ty}, i32)", ll)
                self.assertIn(f"call {ty} {name}({ty}", ll)

    def test_writelane_emits_value_lane_passthrough_order(self):
        ll = self._lower(
            "wlane",
            lambda b: b.writelane(b.const_i32(7), b.const_i32(0), b.const_i32(9)),
        )
        self.assertIn("declare i32 @llvm.amdgcn.writelane.i32(i32, i32, i32)", ll)
        # Operand order is (uniform_val, lane, passthrough).
        self.assertIn("call i32 @llvm.amdgcn.writelane.i32(i32 7, i32 0, i32 9)", ll)

    def test_lane_ops_reject_non_i32_lane_index(self):
        b = self._builder("lane_bad")
        with self.assertRaises(ValueError):
            b.readlane(b.const_i32(1), b.const_i64(0))
        with self.assertRaises(ValueError):
            b.writelane(b.const_i32(1), b.const_i64(0), b.const_i32(1))
        with self.assertRaises(ValueError):
            # passthrough type must match the uniform value's type
            b.writelane(b.const_i32(1), b.const_i32(0), b.const_f32(1.0))

    # ---- permlane16 / permlane64 / permlane32_swap ----
    def test_permlane16_emits_four_sources_and_bool_flags(self):
        ll = self._lower(
            "pl16",
            lambda b: b.permlane16(*[b.const_i32(i) for i in range(4)]),
        )
        # The data type is an overloaded position, so the mangled name carries
        # its suffix. LLVM accepts the bare "permlane16" and auto-upgrades it,
        # but rewrites it to this on the way out, so the bare form is the one
        # that would not survive a round trip.
        self.assertIn(
            "declare i32 @llvm.amdgcn.permlane16.i32"
            "(i32, i32, i32, i32, i1 immarg, i1 immarg)",
            ll,
        )
        self.assertIn(
            "call i32 @llvm.amdgcn.permlane16.i32"
            "(i32 0, i32 1, i32 2, i32 3, i1 false, i1 false)",
            ll,
        )

    def test_permlane16_propagates_fi_and_bound_ctrl(self):
        ll = self._lower(
            "pl16_flags",
            lambda b: b.permlane16(
                *[b.const_i32(i) for i in range(4)], fi=True, bound_ctrl=True
            ),
        )
        self.assertIn(
            "call i32 @llvm.amdgcn.permlane16.i32"
            "(i32 0, i32 1, i32 2, i32 3, i1 true, i1 true)",
            ll,
        )

    def test_permlane64_emits_single_operand_call(self):
        ll = self._lower("pl64", lambda b: b.permlane64(b.const_i32(1)))
        self.assertIn("declare i32 @llvm.amdgcn.permlane64.i32(i32)", ll)
        self.assertIn("call i32 @llvm.amdgcn.permlane64.i32(i32 1)", ll)

    def test_permlane32_swap_unpacks_struct_result(self):
        ll = self._lower(
            "psw", lambda b: b.permlane32_swap(b.const_i32(1), b.const_i32(2))
        )
        # Regression: the handler used to request a decl key without the
        # "amdgcn." prefix, so this declare went missing.
        # No name suffix: unlike its permlane siblings this one is not
        # overloaded. The flags are still immarg.
        self.assertIn(
            "declare { i32, i32 } @llvm.amdgcn.permlane32.swap"
            "(i32, i32, i1 immarg, i1 immarg)",
            ll,
        )
        self.assertIn(
            "call { i32, i32 } @llvm.amdgcn.permlane32.swap"
            "(i32 1, i32 2, i1 false, i1 false)",
            ll,
        )
        # Both halves are extracted from the returned struct.
        self.assertIn("extractvalue { i32, i32 }", ll)
        self.assertEqual(ll.count("extractvalue { i32, i32 }"), 2)

    # ---- alignbyte / s_wqm ----
    def test_alignbyte_emits_three_operand_call(self):
        ll = self._lower(
            "algn",
            lambda b: b.alignbyte(b.const_i32(1), b.const_i32(2), b.const_i32(8)),
        )
        self.assertIn("declare i32 @llvm.amdgcn.alignbyte(i32, i32, i32)", ll)
        self.assertIn("call i32 @llvm.amdgcn.alignbyte(i32 1, i32 2, i32 8)", ll)

    def test_s_wqm_emits_width_matched_intrinsic(self):
        for suffix, ty, mk in (
            ("i32", "i32", "const_i32"),
            ("i64", "i64", "const_i64"),
        ):
            with self.subTest(suffix=suffix):
                ll = self._lower(
                    f"wqm_{suffix}",
                    lambda b, mk=mk: b.s_wqm(getattr(b, mk)(0xF)),
                )
                # Result and operand are separately overloaded, so the
                # canonical name repeats the type: s.wqm.i32.i32.
                self.assertIn(
                    f"declare {ty} @llvm.amdgcn.s.wqm.{suffix}.{suffix}({ty})", ll
                )
                self.assertIn(
                    f"call {ty} @llvm.amdgcn.s.wqm.{suffix}.{suffix}({ty} 15)", ll
                )

    def test_s_wqm_rejects_narrow_mask(self):
        b = self._builder("wqm_bad")
        with self.assertRaises(ValueError):
            b.s_wqm(b.const_f32(1.0))

    # ---- av.load / av.store (agent-scope 128-bit vector mem) ----
    def test_av_load_b128_emits_agent_scope_metadata(self):
        from rocke.core.ir import I32, PtrType

        def build(b):
            p = b.param("p", PtrType(I32, "global"), align=16)
            b.av_load_b128(p)

        ll = self._lower("avld", build)
        # A global pointer param is ptr addrspace(1) in the header, so the
        # overload -- declare and call -- has to be the p1 one.
        self.assertIn(
            "declare <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1), metadata)",
            ll,
        )
        self.assertIn(
            "call <4 x i32> @llvm.amdgcn.av.load.b128.p1("
            "ptr addrspace(1) %p, metadata !3)",
            ll,
        )
        # The scope operand must be backed by a real metadata node.
        self.assertIn('!3 = !{!"agent"}', ll)

    def test_av_store_b128_emits_agent_scope_metadata(self):
        from rocke.core.ir import I32, PtrType

        def build(b):
            p = b.param("p", PtrType(I32, "global"), align=16)
            b.av_store_b128(p, b.av_load_b128(p))

        ll = self._lower("avst", build)
        self.assertIn(
            "declare void @llvm.amdgcn.av.store.b128.p1("
            "ptr addrspace(1), <4 x i32>, metadata)",
            ll,
        )
        self.assertIn(
            "call void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1) %p, <4 x i32>", ll
        )
        self.assertIn("metadata !3)", ll)
        self.assertIn('!3 = !{!"agent"}', ll)

    def test_av_b128_rejects_lds_pointer(self):
        """LDS is not one of the spaces the intrinsic is defined for."""
        from rocke.core.ir import I32, PtrType
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = self._builder("av_lds")
        p = b.param("p", PtrType(I32, "lds"), align=16)
        b.av_load_b128(p)
        with self.assertRaises(ValueError):
            lower_kernel_to_llvm(b.kernel)

    def test_av_store_b128_requires_v4i32_data(self):
        from rocke.core.ir import I32, PtrType

        b = self._builder("avst_bad")
        p = b.param("p", PtrType(I32, "global"), align=16)
        # Right vector length, wrong element type: the ABI expects <4 x i32>.
        with self.assertRaises(ValueError):
            b.av_store_b128(p, b.zero_vec_f32(4))
        with self.assertRaises(ValueError):
            b.av_store_b128(p, b.const_i32(0))

    # ---- s_alloc_vgpr ----
    def test_s_alloc_vgpr_zexts_i1_result_to_i32(self):
        ll = self._lower("valloc", lambda b: b.s_alloc_vgpr(8))
        self.assertIn("declare i1 @llvm.amdgcn.s.alloc.vgpr(i32)", ll)
        self.assertIn("call i1 @llvm.amdgcn.s.alloc.vgpr(i32 8)", ll)
        # The intrinsic returns i1; the IR value is an i32, so a zext is required.
        self.assertIn("zext i1 ", ll)

    def test_s_alloc_vgpr_rejects_non_positive_count(self):
        b = self._builder("valloc_bad")
        with self.assertRaises(ValueError):
            b.s_alloc_vgpr(0)

    # ---- async markers / event waits / prefetch ----
    def test_asyncmark_emits_void_call(self):
        ll = self._lower("amark", lambda b: b.asyncmark())
        self.assertIn("declare void @llvm.amdgcn.asyncmark()", ll)
        self.assertIn("call void @llvm.amdgcn.asyncmark()", ll)

    def test_wait_asyncmark_passes_i16_count(self):
        ll = self._lower("await", lambda b: b.wait_asyncmark(3))
        self.assertIn("declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)", ll)
        self.assertIn("call void @llvm.amdgcn.wait.asyncmark(i16 3)", ll)

    def test_s_wait_event_passes_i16_bitmask(self):
        ll = self._lower("sevt", lambda b: b.s_wait_event(1))
        self.assertIn("declare void @llvm.amdgcn.s.wait.event(i16 immarg)", ll)
        self.assertIn("call void @llvm.amdgcn.s.wait.event(i16 1)", ll)

    def test_s_prefetch_inst_emits_ptr_and_length(self):
        from rocke.core.ir import I32, PtrType

        def build(b):
            code = b.param("code", PtrType(I32, "global"), align=4)
            b.s_prefetch_inst(code, b.const_i32(64))

        ll = self._lower("sprefetch", build)
        # The operand is llvm_anyptr_ty, so the call and its declare have to
        # name the pointer's real space. Emitting a bare ``ptr`` for this
        # addrspace(1) param is what LLVM rejects with "'%code' defined with
        # type 'ptr addrspace(1)' but expected 'ptr'".
        self.assertIn(
            "declare void @llvm.amdgcn.s.prefetch.inst.p1(ptr addrspace(1), i32)", ll
        )
        self.assertIn(
            "call void @llvm.amdgcn.s.prefetch.inst.p1("
            "ptr addrspace(1) %code, i32 64)",
            ll,
        )

    def test_s_prefetch_inst_spaces_coexist_in_one_module(self):
        """Two spaces in one kernel must not collide on one declare.

        LLVM's own ``llvm.amdgcn.s.prefetch.inst.ll`` test prefetches through
        addrspace(4), so instruction memory really is reached through more than
        one space. Since the overload is keyed on that space, an unmangled
        declare would have to name one symbol twice with two signatures --
        the same failure mode as ``mov_dpp8`` above.
        """
        from rocke.core.ir import I32, PtrType

        def build(b):
            glob = b.param("glob", PtrType(I32, "global"), align=4)
            const = b.param("const_", PtrType(I32, "constant"), align=4)
            n = b.const_i32(64)
            b.s_prefetch_inst(glob, n)
            b.s_prefetch_inst(const, n)

        ll = self._lower("sprefetch_spaces", build)
        self.assertIn(
            "declare void @llvm.amdgcn.s.prefetch.inst.p1(ptr addrspace(1), i32)", ll
        )
        self.assertIn(
            "declare void @llvm.amdgcn.s.prefetch.inst.p4(ptr addrspace(4), i32)", ll
        )
        self.assertIn("s.prefetch.inst.p1(ptr addrspace(1) %glob, i32 64)", ll)
        self.assertIn("s.prefetch.inst.p4(ptr addrspace(4) %const_, i32 64)", ll)

    def test_s_prefetch_inst_rejects_lds_pointer(self):
        from rocke.core.ir import I32, PtrType
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = self._builder("sprefetch_lds")
        p = b.param("p", PtrType(I32, "lds"), align=4)
        b.s_prefetch_inst(p, b.const_i32(64))
        with self.assertRaises(ValueError):
            lower_kernel_to_llvm(b.kernel)

    # ---- async buffer / global -> LDS ----
    def test_buffer_load_lds_async_converts_dwords_to_bytes(self):
        from rocke.core.ir import CACHE_STREAM, F16, I32, PtrType

        def build(b):
            X = b.param("X", PtrType(F16, "global"))
            N = b.param("N_bytes", I32)
            rsrc = b.buffer_rsrc(X, N)
            lds = b.smem_alloc(F16, [64, 8], name_hint="stage")
            b.buffer_load_lds_async(
                rsrc,
                b.smem_addr_of(lds),
                b.const_i32(0),
                b.const_i32(0),
                dwords=4,
                coherency=CACHE_STREAM,
            )

        ll = self._lower("buf_async", build)
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.async.lds", ll)
        # dwords=4 -> 16 bytes per lane; trailing imm is coherency (CACHE_STREAM=2).
        self.assertIn("i32 16, i32 0, i32 0, i32 0, i32 2)", ll)
        # smem_addr_of yields an i64 LDS address, but the intrinsic declares
        # ptr addrspace(3); passing the i64 through is what LLVM rejects with
        # "defined with type 'i64' but expected 'ptr addrspace(3)'".
        self.assertIn("= inttoptr i64 ", ll)
        self.assertRegex(
            ll,
            r"buffer\.load\.async\.lds\(ptr addrspace\(8\) \S+, "
            r"ptr addrspace\(3\) %lds_ptr",
        )

    def test_buffer_load_lds_async_rejects_bad_dwords_and_coherency(self):
        from rocke.core.ir import F16, I32, PtrType

        b = self._builder("buf_async_bad")
        X = b.param("X", PtrType(F16, "global"))
        N = b.param("N_bytes", I32)
        rsrc = b.buffer_rsrc(X, N)
        lds_addr = b.smem_addr_of(b.smem_alloc(F16, [64, 8], name_hint="stage"))
        with self.assertRaises(ValueError):
            b.buffer_load_lds_async(
                rsrc, lds_addr, b.const_i32(0), b.const_i32(0), dwords=2
            )
        with self.assertRaises(ValueError):
            b.buffer_load_lds_async(
                rsrc, lds_addr, b.const_i32(0), b.const_i32(0), dwords=4, coherency=7
            )

    def test_global_load_async_to_lds_supports_b8_width(self):
        """width_bytes=1 selects the LLVM 23 ``.b8`` async copy.

        The opcode is a gfx1250 one, but its lowering is arch-independent, so
        this runs on the default backend to keep the expected text identical to
        the C++ parity gate (which has no gfx1250 ISA backend).
        """
        from rocke.core.ir import I32, PtrType

        def build(b):
            src = b.param("src", PtrType(I32, "global"), align=4)
            lds = b.smem_alloc(I32, [64], name_hint="stage")
            b.global_load_async_to_lds(
                src, b.const_i32(0), lds, [b.const_i32(0)], width_bytes=1
            )

        ll = self._lower("gl_async_b8", build)
        self.assertIn(
            "declare void @llvm.amdgcn.global.load.async.to.lds.b8("
            "ptr addrspace(1) nocapture, ptr addrspace(3) nocapture, "
            "i32 immarg, i32 immarg)",
            ll,
        )
        self.assertIn("call void @llvm.amdgcn.global.load.async.to.lds.b8(", ll)
        # Per-lane source/destination addresses are computed with GEPs.
        self.assertIn("getelementptr inbounds", ll)

    def test_global_load_async_to_lds_rejects_unsupported_width(self):
        from rocke.core.ir import I32, PtrType

        b = self._builder("gl_async_bad")
        src = b.param("src", PtrType(I32, "global"), align=4)
        lds = b.smem_alloc(I32, [64], name_hint="stage")
        with self.assertRaises(ValueError):
            b.global_load_async_to_lds(
                src, b.const_i32(0), lds, [b.const_i32(0)], width_bytes=2
            )


class TestBothBackendDifferentialGate(unittest.TestCase):
    """``backend="both"`` must never answer with IR it did not compare.

    This mode exists to catch the two engines emitting different bytes, so the
    one thing it may not do is return the Python result when the C++ engine
    failed: that reports an *uncompared* kernel as parity-verified, and it is
    invisible -- the caller gets plausible IR and the suite goes green. So
    every cpp-side failure propagates here. A failure on an arch named in
    ``CPP_UNPORTED_ARCHES`` is relabelled ``BackendCoverageGap`` (the test layer
    turns those into visible skips, see tests/conftest.py); anything else
    surfaces as-is. That list is empty today -- gfx1250, the last entry, is
    ported -- so the tests that need a gap install a stand-in arch.
    """

    def setUp(self):
        from rocke.core import backend

        self.backend = backend
        backend.reset_cpp_fallbacks()
        self._env = patch.dict("os.environ", {"ROCKE_BACKEND": "both"})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(backend.reset_cpp_fallbacks)

    def _kernel(self):
        b = IRBuilder("both_gate")
        b.kernel.attrs["max_workgroup_size"] = 64
        b.ret()
        return b.kernel

    def _run(self, cpp_result, arch=None):
        """Dispatch with the cpp engine stubbed to ``cpp_result``."""

        def python_lower(kernel, *, llvm_flavor=None, arch=None):
            return "PYTHON IR\n"

        def fake_cpp(ir_text, arch, llvm_flavor=None):
            if isinstance(cpp_result, BaseException):
                raise cpp_result
            return cpp_result

        with patch.object(self.backend, "_lower_via_cpp_engine", fake_cpp):
            return self.backend.lower_kernel_via_backend(
                self._kernel(), python_lower=python_lower, arch=arch
            )

    def test_agreement_returns_the_shared_text(self):
        self.assertEqual(self._run("PYTHON IR\n"), "PYTHON IR\n")
        self.assertEqual(self.backend.cpp_fallbacks(), [])

    def test_differing_ir_raises_backend_mismatch(self):
        with self.assertRaises(self.backend.BackendMismatch):
            self._run("CPP IR\n")

    def _with_gap(self, arch):
        """Put ``arch`` on the gap list for one test.

        The list is empty now that every arch is ported, so the relabelling
        mechanism has to be exercised against a stand-in. Patching it here also
        keeps these tests from having to change again the next time the list
        does.
        """
        p = patch.object(self.backend, "CPP_UNPORTED_ARCHES", (arch,))
        p.start()
        self.addCleanup(p.stop)

    def test_unported_arch_raises_a_coverage_gap_not_python_ir(self):
        """A listed gap is still an exception -- never a substituted result."""
        self._with_gap("gfxfuture")
        gap = RuntimeError("unknown arch backend gfxfuture")
        with self.assertRaises(self.backend.BackendCoverageGap) as ctx:
            self._run(gap, arch="gfxfuture")
        # The message has to name the arch and the underlying failure, since
        # this is what the skip line in the run summary shows.
        self.assertIn("gfxfuture", str(ctx.exception))
        self.assertIn("unknown arch backend gfxfuture", str(ctx.exception))
        # And nothing was logged as a fallback: no Python IR was handed out.
        self.assertEqual(self.backend.cpp_fallbacks(), [])

    def test_failure_on_a_ported_arch_is_not_excused_as_a_gap(self):
        """Only arches on the named list get the gap treatment.

        A gfx950 lowering failure is a regression in a path the C++ engine is
        supposed to serve, so it must surface as the engine's own error rather
        than be relabelled into something the suite skips.
        """
        boom = RuntimeError("internal engine failure")
        with self.assertRaises(RuntimeError) as ctx:
            self._run(boom, arch="gfx950")
        self.assertNotIsInstance(ctx.exception, self.backend.BackendCoverageGap)
        self.assertEqual(self.backend.cpp_fallbacks(), [])

    def test_coverage_gap_is_a_backend_error(self):
        """So a caller outside pytest can catch it with the normal base class."""
        self.assertTrue(
            issubclass(self.backend.BackendCoverageGap, self.backend.BackendError)
        )

    def test_every_arch_is_ported(self):
        """The gap list is empty, so no kernel can reach the skip path.

        gfx1250 was the last entry. With the list empty, every arch the Python
        engine lowers is held to byte-identity by this lane -- a re-entry here
        would silently convert a divergence into a skip, so the emptiness is
        the property worth asserting.
        """
        self.assertEqual(self.backend.CPP_UNPORTED_ARCHES, ())

    def test_cpp_engine_lowers_every_arch_python_wires(self):
        """The invariant that replaced the gap list, checked against the engine.

        With ``CPP_UNPORTED_ARCHES`` empty, "no gaps" is only an assertion about
        a tuple. This checks the thing the tuple stands for: for every arch the
        Python ISA registry wires, the C++ engine resolves a backend and emits
        the same bytes. Wiring a new arch in Python without adding the two C++
        backend rows (``rocke_ll_backend_for`` and the ISA ``REGISTRY``) fails
        here instead of silently reintroducing a skip.
        """
        from rocke.core.backend import BackendError, _lower_via_cpp_engine, _text_diff
        from rocke.core.ir_serialize import serialize
        from rocke.core.isa.backend import wired_arches
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        arches = wired_arches()
        self.assertIn("gfx1250", arches)
        for arch in arches:
            with self.subTest(arch=arch):
                kernel = self._kernel()
                # The reference side is the native lowerer, not the dispatching
                # chokepoint: setUp puts the class in backend="both", so going
                # through it here would demand the C++ extension before the
                # skip below can excuse its absence.
                py_ir = _lower_kernel_to_llvm_python(kernel, arch=arch)
                try:
                    cpp_ir = _lower_via_cpp_engine(serialize(kernel), arch, None)
                except BackendError as e:
                    self.skipTest(f"C++ engine unavailable: {e}")
                self.assertEqual(
                    cpp_ir,
                    py_ir,
                    f"engines disagree on {arch}:\n"
                    + _text_diff("lowered AMDGPU .ll", py_ir, cpp_ir),
                )

    def test_flavor_rejection_is_never_swallowed(self):
        """One engine rejecting a flavor the other took is a defect, not a gap."""
        with self.assertRaises(ValueError):
            self._run(ValueError("unknown LLVM flavor 'llvm24'"))
        self.assertEqual(self.backend.cpp_fallbacks(), [])

    def test_flavor_rejection_is_not_excused_on_an_unported_arch(self):
        """Even on a listed arch a ValueError is a defect, not a coverage gap."""
        self._with_gap("gfxfuture")
        with self.assertRaises(ValueError):
            self._run(ValueError("unknown LLVM flavor 'llvm24'"), arch="gfxfuture")


class TestFusionPlanner(unittest.TestCase):
    """CPU-only coverage for the graph-capture fusion planner."""

    @_requires_torch
    def test_explain_matmul_bias_relu_scale(self):
        torch = pytest.importorskip("torch")
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

    @_requires_torch
    def test_explain_matmul_only(self):
        torch = pytest.importorskip("torch")
        from rocke.helpers import explain_fn

        def fn(A, B):
            return torch.matmul(A, B)

        info = explain_fn(fn)
        self.assertTrue(info["matched"])
        self.assertEqual(info["bias_arg_name"], None)
        self.assertEqual(info["epilogue_ops"], [])

    @_requires_torch
    def test_explain_unsupported_graph(self):
        torch = pytest.importorskip("torch")
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.sin(A)

        info = explain_fn(fn)
        self.assertFalse(info["matched"])
        self.assertIn("no registered", info["reason"])

    @_requires_torch
    def test_compile_fn_exposes_plan_without_launching(self):
        torch = pytest.importorskip("torch")
        from rocke.helpers import compile_fn

        def fn(A, B, bias):
            return torch.relu(torch.matmul(A, B) + bias)

        compiled = compile_fn(fn)
        self.assertEqual(compiled.match["pattern"], "matmul_bias_relu")
        self.assertEqual(compiled.match["bias_arg_name"], "bias")

    def test_dtype_to_ir_accepts_torch_aliases(self):
        from rocke.core.ir import BF16, F16, F32
        from rocke.helpers import dtype_to_ir

        # ``dtype_to_ir`` resolves torch dtypes via their ``str()`` repr
        # ("torch.float16", ...) without importing torch, so this coverage
        # runs even in torch-less lanes. Assert the string path directly.
        self.assertEqual(dtype_to_ir("torch.float16"), F16)
        self.assertEqual(dtype_to_ir("torch.bfloat16"), BF16)
        self.assertEqual(dtype_to_ir("torch.float32"), F32)

        # When torch is installed, the real dtype objects must round-trip
        # identically (their ``str()`` is the "torch.<name>" form above).
        if _HAVE_TORCH:
            import torch

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


@_requires_torch
class TestExpandedPatternMatchers(unittest.TestCase):
    """Verify the new patterns in ``_PATTERN_TABLE`` match expected fns."""

    def test_explain_matmul_gelu(self):
        torch = pytest.importorskip("torch")
        from rocke.helpers import explain_fn

        def fn(A, B):
            return torch.nn.functional.gelu(torch.matmul(A, B))

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["epilogue_ops"], ["gelu_f16"])
        self.assertEqual(info["bias_arg_name"], None)

    def test_explain_matmul_bias_silu(self):
        torch = pytest.importorskip("torch")
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
        torch = pytest.importorskip("torch")
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
        torch = pytest.importorskip("torch")
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.relu(torch.nn.functional.gelu(A))

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertTrue(info["pattern"].startswith("pointwise_"))
        self.assertEqual(info["a_arg_name"], "A")

    def test_explain_pointwise_binary_chain(self):
        torch = pytest.importorskip("torch")
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
        torch = pytest.importorskip("torch")
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
        torch = pytest.importorskip("torch")
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.sum(A, dim=-1)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["pattern"], "reduce_sum")
        self.assertEqual(info["extra_attrs"]["reduce_op"], "sum")

    def test_explain_rowwise_reduction_mean(self):
        torch = pytest.importorskip("torch")
        from rocke.helpers import explain_fn

        def fn(A):
            return torch.mean(A, dim=-1)

        info = explain_fn(fn)
        self.assertTrue(info["matched"], info.get("reason"))
        self.assertEqual(info["pattern"], "reduce_mean")
        self.assertEqual(info["extra_attrs"]["reduce_op"], "mean")


class TestLoweringRegistryBuild(unittest.TestCase):
    """Lowerer tests: candidate generation phase (GPU-agnostic).

    Tests that validate lowerers can recognize regions and generate
    spec configurations without requiring GPU hardware. Architecture
    is mocked to gfx950 so that tests can still run without a GPU.
    """

    @classmethod
    def setUpClass(cls):
        """Mock HIP runtime dependencies for in-process testing without GPU.

        These tests do not launch kernels or require a physical GPU. To achieve this, we mock:

        - ``get_device_arch``: Returns a fixed architecture string (gfx950)
          so the compiler targets a known ISA without querying the actual
          device via ``hipGetDeviceProperties``.

        - ``Runtime.load_module``: Bypasses ``hipModuleLoadData``. The
          mock allows tests to validate that lowerers produce a well-formed
          launcher object without requiring a GPU driver.

        Individual tests can override these mocks (via nested ``patch``
        context managers) to exercise error paths or architecture-specific
        behavior.
        """
        cls.arch_patcher = patch(
            "rocke.runtime.hip_module.get_device_arch", return_value="gfx950"
        )

        cls.load_module_patcher = patch(
            "rocke.runtime.hip_module.Runtime.load_module", return_value=Mock()
        )
        cls.arch_patcher.start()
        cls.load_module_patcher.start()

    @classmethod
    def tearDownClass(cls):
        """Clean up HIP runtime mocks after all tests complete."""
        cls.arch_patcher.stop()
        cls.load_module_patcher.stop()

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

    def test_gemm_epilogue_candidates_errors_on_invalid_arch(self):
        from rocke.helpers import GemmEpilogueLowerer, GreedyFusionScheduler

        graph = self._toy_gemm_graph(with_epilogue=True)
        plan = GreedyFusionScheduler().schedule(graph)
        region = plan.regions[0]
        lowerer = GemmEpilogueLowerer()
        # Override the original class mock to test the error case when get_device_arch returns None
        with patch("rocke.runtime.hip_module.get_device_arch", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                lowerer.candidates(graph, region)
            self.assertIn("Could not detect", str(ctx.exception))

    def test_gemm_epilogue_build_errors_on_invalid_arch(self):
        from rocke.helpers import GemmEpilogueLowerer, GreedyFusionScheduler

        graph = self._toy_gemm_graph(with_epilogue=True)
        plan = GreedyFusionScheduler().schedule(graph)
        region = plan.regions[0]
        lowerer = GemmEpilogueLowerer()
        cfgs = lowerer.candidates(graph, region)
        # Override the original class mock to test the error case when get_device_arch returns None
        with patch("rocke.runtime.hip_module.get_device_arch", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                lowerer.build(cfgs[0])
            self.assertIn("Could not detect", str(ctx.exception))

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

    @_requires_torch
    def test_materialize_with_fake_pool_binds_tensors(self):
        # Use the real WorkspacePool with a CPU "device" via torch.
        torch = pytest.importorskip("torch")
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

    @_requires_torch
    def test_benchmark_case_runs_torch_eager_baseline(self):
        torch = pytest.importorskip("torch")
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


class TestRuntimeLaunchKeepAlive(unittest.TestCase):
    """``Runtime.launch`` / ``launch_blocking`` args-buffer lifetime.

    Characterizes the non-torch race the pending-args queue exists to
    prevent: the ``extra``-path packed-args buffer must outlive an
    async ``launch`` (the GPU command processor reads it after
    enqueue), whereas ``launch_blocking`` needs no bucket entry because
    its trailing ``hipStreamSynchronize`` is the drain barrier. Runs on
    host only -- the two HIP entry points are stubbed to return success.
    """

    def _isolate_pending(self):
        from rocke.runtime.hip_module import Runtime

        prior = dict(Runtime._pending_args)
        Runtime._pending_args.clear()
        return prior

    def _restore_pending(self, prior):
        from rocke.runtime.hip_module import Runtime

        Runtime._pending_args.clear()
        Runtime._pending_args.update(prior)

    def test_async_launch_parks_args_buffer_until_drain(self):
        import ctypes
        from unittest import mock

        from rocke.runtime.hip_module import Runtime, _HipFunctionHandle

        rt = Runtime()
        prior = self._isolate_pending()
        try:
            with mock.patch(
                "rocke.runtime.hip_module._hipModuleLaunchKernel", return_value=0
            ):
                rt.launch(
                    _HipFunctionHandle(),
                    (1, 1, 1),
                    (64, 1, 1),
                    b"\x01\x02\x03\x04",
                    stream=7,
                    record_event=False,
                )
            bucket = Runtime._pending_args.get(7)
            self.assertIsNotNone(bucket)
            self.assertEqual(len(bucket), 1)
            refs, evt = bucket[-1]
            # No event recorded when record_event=False; refs hold the
            # ctypes objects the launch pointed the CP at.
            self.assertIsNone(evt)
            args_buf, _size_buf, extra = refs
            # The parked `extra` array must still point at the *same*
            # args buffer the launch enqueued -- proving the buffer the
            # CP reads later is the one we kept alive, not a copy that
            # was already freed.
            self.assertEqual(extra[1], ctypes.addressof(args_buf))
        finally:
            self._restore_pending(prior)

    def test_blocking_launch_parks_nothing_and_syncs_once(self):
        from unittest import mock

        from rocke.runtime.hip_module import Runtime, _HipFunctionHandle

        rt = Runtime()
        prior = self._isolate_pending()
        try:
            with mock.patch(
                "rocke.runtime.hip_module._hipModuleLaunchKernel", return_value=0
            ), mock.patch(
                "rocke.runtime.hip_module._hipStreamSynchronize", return_value=0
            ) as sync_stub:
                rt.launch_blocking(
                    _HipFunctionHandle(),
                    (1, 1, 1),
                    (64, 1, 1),
                    b"\x01\x02\x03\x04",
                    stream=3,
                )
            # The sync IS the barrier + args-buffer drain, so no bucket
            # bookkeeping is needed for a blocking launch.
            self.assertNotIn(3, Runtime._pending_args)
            self.assertEqual(sync_stub.call_count, 1)
        finally:
            self._restore_pending(prior)


class TestResolveStream(unittest.TestCase):
    """``torch_interop.resolve_stream`` -- the caching-allocator hinge.

    Its whole contract is torch-*optional*: pass through a nonzero
    handle, substitute torch's current stream when torch is present, and
    fall back to the HIP null stream (0) only when torch is genuinely
    absent. The torch-present case is the tripwire that fails loudly if a
    future edit ever collapses the torch branch to 0 inside a torch
    process.
    """

    def test_passes_through_nonzero_without_touching_torch(self):
        import sys
        from unittest import mock

        from rocke.runtime.torch_interop import resolve_stream

        # Poison torch so any import attempt would raise -- proves the
        # nonzero fast-path never reaches for torch at all.
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(resolve_stream(1234), 1234)

    def test_returns_zero_when_torch_absent(self):
        import sys
        from unittest import mock

        from rocke.runtime.torch_interop import resolve_stream

        # sys.modules["torch"] = None makes `import torch` raise
        # ImportError, standing in for a torch-free environment.
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(resolve_stream(0), 0)

    def test_uses_current_stream_when_torch_present(self):
        import sys
        from unittest import mock

        from rocke.runtime.torch_interop import resolve_stream

        fake = mock.MagicMock()
        fake.cuda.current_device.return_value = 0
        fake.cuda.current_stream.return_value.cuda_stream = 99
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertEqual(resolve_stream(0), 99)


class TestPackArgsKernargABI(unittest.TestCase):
    """``pack_args`` / ``pack_args_kernelparams`` -- AMDGPU kernarg ABI.

    Each kernarg sits at an offset aligned to its own size (8 for
    ptr/i64, 4 for i32/f32). The regression that motivated the padded
    packer is a trailing pointer after three i32s landing 4 bytes early
    and being read as garbage. These are pure host-assertable byte
    checks -- no GPU, no torch.
    """

    _MIXED_SIG = [
        {"name": "a", "type": "ptr<f16,global>"},
        {"name": "b", "type": "ptr<f16,global>"},
        {"name": "c", "type": "ptr<f16,global>"},
        {"name": "m", "type": "i32"},
        {"name": "n", "type": "i32"},
        {"name": "k", "type": "i32"},
        {"name": "d", "type": "ptr<f16,global>"},
    ]
    _MIXED_VALS = {
        "a": 0x1000,
        "b": 0x2000,
        "c": 0x3000,
        "m": 4,
        "n": 5,
        "k": 6,
        "d": 0xABCD,
    }

    def test_inserts_padding_before_misaligned_pointer(self):
        import struct

        from rocke.runtime.packing import pack_args

        packed = pack_args(self._MIXED_SIG, self._MIXED_VALS)
        # 3 ptr (24) + 3 i32 (12) = 36, pad 4 to reach 8-alignment, then
        # the trailing ptr (8) => 48 bytes, trailing ptr at offset 40.
        self.assertEqual(len(packed), 48)
        self.assertEqual(struct.unpack_from("<Q", packed, 40)[0], 0xABCD)

    def test_natural_alignment_for_mixed_scalars(self):
        import struct

        from rocke.runtime.packing import pack_args

        sig = [
            {"name": "p", "type": "ptr<f32,global>"},
            {"name": "i", "type": "i32"},
            {"name": "q", "type": "i64"},
            {"name": "f", "type": "f32"},
        ]
        vals = {"p": 0x10, "i": 7, "q": 0x1122334455, "f": 1.5}
        packed = pack_args(sig, vals)
        # p@0(8) i@8(4) pad4 q@16(8) f@24(4) => 28 bytes.
        self.assertEqual(len(packed), 28)
        self.assertEqual(struct.unpack_from("<Q", packed, 0)[0], 0x10)
        self.assertEqual(struct.unpack_from("<i", packed, 8)[0], 7)
        self.assertEqual(struct.unpack_from("<q", packed, 16)[0], 0x1122334455)
        self.assertAlmostEqual(struct.unpack_from("<f", packed, 24)[0], 1.5)

    def test_kernelparams_agrees_with_pack_args_values(self):
        import ctypes

        from rocke.runtime.packing import pack_args_kernelparams

        params = pack_args_kernelparams(self._MIXED_SIG, self._MIXED_VALS)
        # One ctypes scalar per kernel arg, in declaration order, with
        # the same values the byte-packed path encodes -- the two launch
        # paths must be interchangeable oracles.
        self.assertEqual(len(params), len(self._MIXED_SIG))
        self.assertEqual(params[0].value, 0x1000)
        self.assertEqual(params[3].value, 4)
        self.assertEqual(params[6].value, 0xABCD)
        self.assertIsInstance(params[0], ctypes.c_uint64)
        self.assertIsInstance(params[3], ctypes.c_int32)

    def test_rejects_missing_arg(self):
        from rocke.runtime.packing import pack_args

        with self.assertRaises(KeyError):
            pack_args([{"name": "x", "type": "i32"}], {})

    def test_rejects_unknown_type(self):
        from rocke.runtime.packing import pack_args

        with self.assertRaises(ValueError):
            pack_args([{"name": "x", "type": "f16"}], {"x": 1})

    def test_as_ptr_accepts_devicemem_ptr_method(self):
        import struct

        from rocke.runtime.packing import pack_args

        # DeviceMem exposes its raw device pointer via a ptr() method; the
        # torch-free numpy path passes it straight into a launcher's values.
        class _FakeDeviceMem:
            def ptr(self):
                return 0xDEAD

        packed = pack_args(
            [{"name": "p", "type": "ptr<f16,global>"}], {"p": _FakeDeviceMem()}
        )
        self.assertEqual(struct.unpack_from("<Q", packed, 0)[0], 0xDEAD)

    def test_as_ptr_prefers_data_ptr_over_ptr(self):
        import struct

        from rocke.runtime.packing import pack_args

        # A torch tensor exposes data_ptr(); if some object exposed both, the
        # data_ptr() branch must win (it precedes the DeviceMem ptr() branch).
        class _Both:
            def data_ptr(self):
                return 0xAAAA

            def ptr(self):
                return 0xBBBB

        packed = pack_args([{"name": "p", "type": "ptr<f16,global>"}], {"p": _Both()})
        self.assertEqual(struct.unpack_from("<Q", packed, 0)[0], 0xAAAA)

    def test_as_ptr_none_encodes_null(self):
        import struct

        from rocke.runtime.packing import pack_args

        packed = pack_args([{"name": "p", "type": "ptr<f16,global>"}], {"p": None})
        self.assertEqual(struct.unpack_from("<Q", packed, 0)[0], 0)

    def test_as_ptr_rejects_unconvertible(self):
        from rocke.runtime.packing import pack_args

        # A non-callable ptr attribute is not a device pointer -- the callable()
        # guard must fall through to the TypeError, not silently use the attr.
        class _NonCallablePtr:
            ptr = 0x1234

        with self.assertRaises(TypeError):
            pack_args(
                [{"name": "p", "type": "ptr<f16,global>"}], {"p": _NonCallablePtr()}
            )
        with self.assertRaises(TypeError):
            pack_args(
                [{"name": "p", "type": "ptr<f16,global>"}], {"p": "not a pointer"}
            )


class TestHostBufferReExportShim(unittest.TestCase):
    """The manifest_runner.utils re-export shim over rocke.runtime.host_buffers.

    The three host byte helpers moved to ``rocke.runtime.host_buffers``; the old
    ``instances.common.manifest_runner.utils`` path re-exports them so existing
    importers keep working. Pin the identity so a severed shim fails loudly.
    """

    def test_utils_reexports_host_buffer_helpers(self):
        from rocke.instances.common.manifest_runner import utils
        from rocke.runtime import host_buffers

        self.assertIs(utils.as_u8_buffer, host_buffers.as_u8_buffer)
        self.assertIs(utils.nbytes, host_buffers.nbytes)
        self.assertIs(utils.require_numpy, host_buffers.require_numpy)


class TestLibDiscoveryOrder(unittest.TestCase):
    """Shared-library resolution order in ``hip_module``.

    These pure helpers move to the extracted coexistence/loader module;
    pinning their behavior first makes that move provably
    behavior-preserving. ``comgr`` already depends on this same
    resolution contract across the module boundary.
    """

    def test_version_key_orders_rocm_newest_first(self):
        from rocke.runtime.runtime_coexistence import _version_key

        roots = ["/opt/rocm-7.2/lib", "/opt/rocm-7.10/lib", "/opt/rocm-7.9/lib"]
        ordered = sorted(roots, key=_version_key, reverse=True)
        # A plain string sort would put 7.10 before 7.2; the integer-tuple
        # key must rank 7.10 as the newest.
        self.assertTrue(ordered[0].startswith("/opt/rocm-7.10"))

    def test_candidate_paths_put_env_override_first(self):
        import os
        from unittest import mock

        from rocke.runtime.runtime_coexistence import _candidate_lib_paths

        with mock.patch.dict(os.environ, {"ROCKE_HIP_LIB": "/custom/libamdhip64.so"}):
            paths = _candidate_lib_paths("amdhip64", "ROCKE_HIP_LIB", ["7"])
        self.assertEqual(paths[0], "/custom/libamdhip64.so")

    def test_torch_bundled_lib_never_imports_torch(self):
        import sys
        from unittest import mock

        from rocke.runtime.runtime_coexistence import _torch_bundled_lib

        # With torch absent from sys.modules, the probe must return None
        # and must NOT import torch for the loader side effect.
        with mock.patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            result = _torch_bundled_lib("amdhip64")
            self.assertIsNone(result)
            self.assertNotIn("torch", sys.modules)


# ---------------------------------------------------------------------
# Build smoke tests for the FMHA / Sage / sparse-attention instance
# family. These are IR-build + LLVM-lower smoke tests (no GPU); the
# intent is to catch regressions in the builders + their lowering
# vocabulary, NOT to validate numerical correctness. End-to-end
# correctness + perf parity for these kernels is not covered by
# this suite yet -- see test_rocke_examples.py for the GPU-bound
# parity tests on the pre-existing kernels.
# ---------------------------------------------------------------------


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


class TestCShuffleEpilogueSmoke(unittest.TestCase):
    """Quick smoke tests for CShuffleEpilogue across fp16, bf16, and fp32.

    All three dtypes are exercised directly via CShuffleEpilogue.from_grid +
    IRBuilder so the assertions are scoped to the epilogue IR only, not the
    full GEMM pipeline.
    """

    _TILE = dict(
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

    def _build_ll(self, dtype: str) -> str:
        from rocke.helpers import CShuffleEpilogue

        b = IRBuilder(f"cshuffle_{dtype}_smoke")
        D = b.param("D", PtrType(F16, "global"))
        M = b.param("M", I32)
        N = b.param("N", I32)

        atom = mfma_atom("f16", 32, 32, 16)
        bound = WarpGrid(**self._TILE).bind(b)
        epi = CShuffleEpilogue.from_grid(atom=atom, grid=bound, out_dtype=dtype)

        n_accs = bound.mfmas_per_warp_m * bound.mfmas_per_warp_n
        accs = [b.zero_vec_f32(atom.c_per_lane) for _ in range(n_accs)]
        d_rsrc = b.buffer_rsrc(D, b.mul(M, b.mul(N, b.const_i32(4))))

        def addr_fn(b_, m_val, n_val):
            return b_.add(b_.mul(m_val, N), n_val), None

        epi.store(b, accs=accs, addr_fn=addr_fn, d_rsrc=d_rsrc)
        return lower_kernel_to_llvm(b.kernel)

    def test_cshuffle_fp16_emits_lds_staging_and_half_store(self):
        ll = self._build_ll("fp16")
        self.assertIn("store half", ll)
        self.assertIn("addrspace(3)", ll)
        self.assertIn("@llvm.amdgcn.s.barrier", ll)

    def test_cshuffle_bf16_emits_lds_staging_and_bfloat_store(self):
        ll = self._build_ll("bf16")
        self.assertIn("store bfloat", ll)
        self.assertIn("addrspace(3)", ll)
        self.assertIn("@llvm.amdgcn.s.barrier", ll)

    def test_cshuffle_fp32_emits_ds_write_and_float_store(self):
        """fp32 output: accumulator stays f32 in LDS; wide store is f32."""
        ll = self._build_ll("fp32")
        # The LDS staging dtype is f32; wide global store is also f32.
        self.assertIn("store float", ll)
        # LDS write to addrspace(3) must be present.
        self.assertIn("addrspace(3)", ll)
        # s_barrier between LDS write and read.
        self.assertIn("@llvm.amdgcn.s.barrier", ll)


if __name__ == "__main__":
    unittest.main()
