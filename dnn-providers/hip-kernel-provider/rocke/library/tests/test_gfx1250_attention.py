# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 unified-attention tests (no-GPU lowering + GPU probes).

Covers the wave32 WMMA attention surface:
  * the standalone dense WMMA attention forward;
  * the tiled-2D prefill kernel (gate / lower / dispatch);
  * the split-KV tiled-3D decode kernel (gate / lower / dispatch /
    multi-wave regression);
  * the scalar fp8-KV unified-attention leg;
  * Qwen3-30B-A3B decode/prefill routing through the unified instance and the
    KV-cache dequant / append-RoPE kernels.

The GPU probe class is skipped unless run on a gfx1250 device.

    PYTHONPATH=Python python3 -m pytest rocke/tests/test_gfx1250_attention.py -k "not Gpu"
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest

_PYDIR = pathlib.Path(__file__).resolve().parents[2] / "Python"  # rocKE/Python


def _device_arch():
    """(arch_str_or_None) via the rocke HIP runtime (no torch dependency)."""
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0)
    except Exception:
        return None


_ARCH = _device_arch()


class TestGfx1250DenseAttention(unittest.TestCase):
    def test_attention_fwd_lowers(self):
        # The standalone gfx1250 dense WMMA attention forward (BLOCK_K=32) must
        # build and lower to the K=32 WMMA intrinsic. H=64 -> 2 QK d-tiles x 2
        # N-sub-tiles + 4 PV d-tiles = 8 WMMA calls per K-loop iteration.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.wmma_attention_fwd import (
            WmmaAttentionFwdSpec,
            build_wmma_attention_fwd,
        )

        spec = WmmaAttentionFwdSpec(head_size=64, num_query_heads=4)
        ll = lower_kernel_to_llvm(
            build_wmma_attention_fwd(spec, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16", ll)


class TestGfx1250TiledAttention2D(unittest.TestCase):
    def test_gate_accepts_fp8_rejects_bf16_kv(self):
        from rocke.instances.gfx1250.attention_tiled_2d import supports_tiled_2d

        ok, why = supports_tiled_2d(
            head_size=64,
            block_size=32,
            dtype="bf16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=True,
            q_dtype=None,
            num_warps=1,
            block_m_per_warp=16,
            kv_storage_dtype="fp8e4m3",
            tile_size=32,
            arch="gfx1250",
        )
        self.assertTrue(ok, why)

        ok, why = supports_tiled_2d(
            head_size=64,
            block_size=32,
            dtype="bf16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
            num_warps=1,
            block_m_per_warp=16,
            kv_storage_dtype=None,
            tile_size=32,
            arch="gfx1250",
        )
        self.assertFalse(ok)
        self.assertIn("fp8e4m3", why)

    def test_lowers_to_wmma_and_fp8_dequant(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.attention_tiled_2d import (
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
            sliding_window=128,
            has_softcap=False,
            num_seqs=4,
            kv_storage_dtype="fp8e4m3",
            tile_size=32,
        )
        ll = lower_kernel_to_llvm(
            build_unified_attention_2d_tiled(spec, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16", ll)
        self.assertIn("llvm.amdgcn.cvt.pk.f32.fp8", ll)

    def test_common_dispatch_selects_wave32_tiled_meta(self):
        from rocke.instances.common import attention_unified as au

        old_arch = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx1250"
        try:
            problem = au.UnifiedAttentionProblem(
                total_q=4,
                num_seqs=4,
                num_query_heads=64,
                num_kv_heads=8,
                head_size=64,
                block_size=32,
                max_seqlen_q=1,
                max_seqlen_k=256,
                dtype="bf16",
                use_sinks=True,
                sliding_window=128,
                use_fp8=True,
            )
            ok, why = au.supports_native_unified_attention_tiled(problem)
            self.assertTrue(ok, why)
            spec = au._tiled_spec_from_problem(problem)
            self.assertEqual(spec.kernel_name().split("_")[2], "tiled")
            self.assertEqual(spec.num_warps, 1)
            self.assertEqual(spec.block_q, 2)
            meta = au._get_2d_launch_meta(problem, au._tiled_cache_key(problem))
            self.assertEqual(meta.block, (32, 1, 1))
            self.assertEqual(meta.grid, (8, 6, 1))
        finally:
            au._RESOLVED_ATTENTION_ARCH = old_arch


class TestGfx1250TiledAttention3D(unittest.TestCase):
    def test_gate_and_lowers(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
            supports_tiled_3d,
        )

        ok, why = supports_tiled_3d(
            head_size=64,
            block_size=16,
            dtype="bf16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=True,
            q_dtype=None,
            kv_storage_dtype="fp8e4m3",
            arch="gfx1250",
        )
        self.assertTrue(ok, why)
        bad, why_bad = supports_tiled_3d(
            head_size=64,
            block_size=64,
            dtype="bf16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
            kv_storage_dtype=None,
            arch="gfx1250",
        )
        self.assertFalse(bad)
        self.assertIn("block_size", why_bad)

        seg = UnifiedAttention3DTiledSpec(
            head_size=64,
            block_size=16,
            num_query_heads=32,
            num_kv_heads=4,
            dtype="bf16",
            use_sinks=True,
            sliding_window=0,
            has_softcap=False,
            num_segments=16,
            num_seqs=2,
            kv_storage_dtype="fp8e4m3",
        )
        ll_seg = lower_kernel_to_llvm(
            build_unified_attention_3d_tiled(seg, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16", ll_seg)
        self.assertIn("llvm.amdgcn.cvt.pk.f32.fp8", ll_seg)
        red = UnifiedAttentionReduceTiledSpec(
            head_size=64,
            num_query_heads=32,
            num_kv_heads=4,
            dtype="bf16",
            num_segments=16,
        )
        ll_red = lower_kernel_to_llvm(
            build_unified_attention_reduce_tiled(red, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("llvm.amdgcn.s.barrier", ll_red)

    def test_large_batch_regression_binary_search_floor(self):
        # Regression: large-batch decode binary search uses 32+ iterations
        # (the historically flaky 256-seq fp8 shape).
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        seg256 = UnifiedAttention3DTiledSpec(
            head_size=64,
            block_size=16,
            num_query_heads=32,
            num_kv_heads=4,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_segments=16,
            num_seqs=256,
            kv_storage_dtype="fp8e4m3",
        )
        self.assertGreaterEqual(seg256.binary_search_iters, 32)
        ll_256 = lower_kernel_to_llvm(
            build_unified_attention_3d_tiled(seg256, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16", ll_256)

    def test_multi_wave_widths_lower(self):
        # Cooperative multi-wave32 CTA (LDS inter-wave reduction) lowers for
        # every supported width.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        for waves in (1, 2, 4, 8):
            spec = UnifiedAttention3DTiledSpec(
                head_size=64,
                block_size=16,
                num_query_heads=32,
                num_kv_heads=4,
                dtype="bf16",
                use_sinks=True,
                sliding_window=0,
                has_softcap=False,
                num_segments=16,
                num_seqs=256,
                kv_storage_dtype="fp8e4m3",
                num_waves=waves,
                use_wide_lds_reads=(waves == 1),
            )
            ll = lower_kernel_to_llvm(
                build_unified_attention_3d_tiled(spec, arch="gfx1250"), arch="gfx1250"
            )
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16", ll)

    def test_dtla_prefetch_bf16_lowers(self):
        # DTLA bf16: async global->LDS V staging double-buffer + prefetch.
        # gfx1250 uses global_load_async_to_lds_b128 + the dedicated ASYNC
        # counter (s_wait_asynccnt), NOT the gfx9 buffer/global load-to-LDS
        # intrinsics (which do not select on gfx1250). The 32-token tile spans
        # 2 paged blocks at block_size=16, 1 at 32; both must lower (and
        # compile -- see test_dtla_prefetch_bf16_compiles).
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        for bs in (16, 32):
            spec = UnifiedAttention3DTiledSpec(
                head_size=64,
                block_size=bs,
                num_query_heads=32,
                num_kv_heads=4,
                dtype="bf16",
                use_sinks=True,
                sliding_window=0,
                has_softcap=False,
                num_segments=16,
                num_seqs=2,
                kv_storage_dtype="bf16",
                use_dtla_prefetch=True,
                use_wide_lds_reads=False,
            )
            self.assertIn("dtla", spec.kernel_name())
            ll = lower_kernel_to_llvm(
                build_unified_attention_3d_tiled(spec, arch="gfx1250"), arch="gfx1250"
            )
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16", ll)
            self.assertIn("llvm.amdgcn.global.load.async.to.lds.b128", ll)
            self.assertIn("llvm.amdgcn.s.wait.asynccnt", ll)
            # The gfx9 DirectToLDS intrinsics must NOT appear (not selectable).
            self.assertNotIn("raw.ptr.buffer.load.lds", ll)
            self.assertNotIn("global.load.lds", ll)

    def test_dtla_prefetch_bf16_compiles(self):
        # Regression guard: lowering to LLVM IR is NOT enough -- the gfx9
        # buffer/global load-to-LDS intrinsics lower cleanly but FAIL to select
        # on gfx1250. Compile to a real code object so a future regression to a
        # non-selectable async path is caught here, not on the GPU box.
        from rocke.instances.gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        try:
            from rocke.helpers.compile import compile_kernel
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"compile toolchain unavailable: {e}")

        for bs in (16, 32):
            spec = UnifiedAttention3DTiledSpec(
                head_size=64,
                block_size=bs,
                num_query_heads=32,
                num_kv_heads=4,
                dtype="bf16",
                use_sinks=True,
                sliding_window=0,
                has_softcap=False,
                num_segments=16,
                num_seqs=2,
                kv_storage_dtype="bf16",
                use_dtla_prefetch=True,
                use_wide_lds_reads=False,
            )
            try:
                art = compile_kernel(
                    build_unified_attention_3d_tiled(spec, arch="gfx1250"),
                    arch="gfx1250",
                )
            except Exception as e:  # pragma: no cover - env-dependent
                self.skipTest(f"gfx1250 comgr compile unavailable: {e}")
            self.assertGreater(art.hsaco_bytes, 0)

    def test_dtla_prefetch_rejects_incompatible_levers(self):
        # DTLA owns the V_lds layout + pipeline; guard the mutually exclusive
        # combinations and the not-yet-implemented fp8 path.
        from rocke.instances.gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
        )

        base = dict(
            head_size=64,
            block_size=16,
            num_query_heads=32,
            num_kv_heads=4,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_segments=16,
            num_seqs=2,
            kv_storage_dtype="bf16",
            use_dtla_prefetch=True,
            use_wide_lds_reads=False,
        )
        with self.assertRaises(ValueError):
            UnifiedAttention3DTiledSpec(**{**base, "use_wide_lds_reads": True})
        with self.assertRaises(ValueError):
            UnifiedAttention3DTiledSpec(**{**base, "num_waves": 2})
        with self.assertRaises(ValueError):
            UnifiedAttention3DTiledSpec(**{**base, "use_register_p": True})
        with self.assertRaises(ValueError):
            UnifiedAttention3DTiledSpec(**{**base, "kv_storage_dtype": "fp8e4m3"})

    def test_common_dispatch_selects_3d(self):
        from rocke.instances.common import attention_unified as au

        old_arch = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx1250"
        try:
            problem = au.UnifiedAttentionProblem(
                total_q=2,
                num_seqs=2,
                num_query_heads=32,
                num_kv_heads=4,
                head_size=64,
                block_size=16,
                max_seqlen_q=1,
                max_seqlen_k=1024,
                dtype="bf16",
                q_dtype="bf16",
                sliding_window=0,
                use_sinks=True,
                use_fp8=True,
            )
            ok, why = au.supports_native_unified_attention_3d_tiled(problem)
            self.assertTrue(ok, why)
            spec = au._tiled_3d_spec_from_problem(problem)
            self.assertEqual(spec.block_q, 2)
            self.assertGreaterEqual(spec.num_segments, 1)
            Spec, _, _, _, _ = au._tiled_3d_impl("gfx1250")
            self.assertEqual(
                Spec.__module__, "rocke.instances.gfx1250.attention_tiled_3d"
            )
        finally:
            au._RESOLVED_ATTENTION_ARCH = old_arch


class TestGfx1250ScalarFp8Attention(unittest.TestCase):
    @staticmethod
    def _small_fp8_problem(**overrides):
        from rocke.instances.common.attention_unified import UnifiedAttentionProblem

        kwargs = dict(
            total_q=8,
            num_seqs=1,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=64,
            block_size=16,
            max_seqlen_q=8,
            max_seqlen_k=16,
            dtype="bf16",
            q_dtype="bf16",
            use_fp8=True,
        )
        kwargs.update(overrides)
        return UnifiedAttentionProblem(**kwargs)

    def test_scalar_gate_accepts_fp8_kv_bf16_query(self):
        from rocke.instances.common.attention_unified import (
            supports_native_unified_attention,
        )

        ok, reason = supports_native_unified_attention(self._small_fp8_problem())
        self.assertTrue(ok, reason)
        self.assertIn("supported", reason)

    def test_scalar_gate_keeps_unsupported_biases_rejected(self):
        from rocke.instances.common.attention_unified import (
            supports_native_unified_attention,
        )

        ok_alibi, reason_alibi = supports_native_unified_attention(
            self._small_fp8_problem(use_alibi=True)
        )
        ok_qq, reason_qq = supports_native_unified_attention(
            self._small_fp8_problem(use_qq_bias=True)
        )
        self.assertFalse(ok_alibi)
        self.assertIn("ALiBi", reason_alibi)
        self.assertFalse(ok_qq)
        self.assertIn("QQ bias", reason_qq)

    def test_scalar_fp8_kv_lowers_dequant_and_scale(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.attention_unified import (
            UnifiedAttention2DSpec,
            build_unified_attention_2d,
        )

        kernel = build_unified_attention_2d(
            UnifiedAttention2DSpec(self._small_fp8_problem())
        )
        ll = lower_kernel_to_llvm(kernel, arch="gfx1250")
        self.assertIn("@llvm.amdgcn.cvt.f32.fp8", ll)
        self.assertIn("%k_scale", ll)
        self.assertIn("%v_scale", ll)


class TestGfx1250Qwen3AttentionRouting(unittest.TestCase):
    """Qwen3 decode/prefill route through the unified attention instance."""

    @staticmethod
    def _decode_problem(shape):
        from rocke.instances.common.attention_unified import UnifiedAttentionProblem

        return UnifiedAttentionProblem(
            total_q=shape.total_q,
            num_seqs=shape.num_seqs,
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size,
            block_size=shape.block_size,
            max_seqlen_q=shape.max_seqlen_q,
            max_seqlen_k=shape.max_seqlen_k,
            dtype=shape.dtype,
            q_dtype=shape.dtype,
            use_sinks=False,
            use_fp8=shape.kv_storage_dtype in ("fp8e4m3", "bf8e5m2"),
        )

    def test_decode_3d_routes_through_unified_split_kv(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            qwen3_decode_attention_shapes,
        )
        from rocke.instances.common import attention_unified as au

        old_arch = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx1250"
        try:
            for kv_storage in ("bf16", "fp8e4m3"):
                shapes = qwen3_decode_attention_shapes(kv_storage_dtype=kv_storage)
                for shape in shapes:
                    problem = self._decode_problem(shape)
                    ok, why = au.supports_native_unified_attention_3d_tiled(problem)
                    self.assertTrue(ok, f"{kv_storage} kv{shape.kv_len}: {why}")
                    (_SegSpec, ReduceSpec, build_seg, build_red, _) = au._tiled_3d_impl(
                        "gfx1250"
                    )
                    seg_spec = au._tiled_3d_spec_from_problem(problem)
                    ll = lower_kernel_to_llvm(
                        build_seg(seg_spec, arch="gfx1250"), arch="gfx1250"
                    )
                    self.assertIn(
                        "llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16", ll
                    )
                    if kv_storage == "fp8e4m3":
                        self.assertIn("llvm.amdgcn.cvt.pk.f32.fp8", ll)
                    red_spec = ReduceSpec(
                        head_size=shape.head_size,
                        num_query_heads=shape.num_query_heads,
                        num_kv_heads=shape.num_kv_heads,
                        dtype=shape.dtype,
                        num_segments=seg_spec.num_segments,
                    )
                    lower_kernel_to_llvm(
                        build_red(red_spec, arch="gfx1250"), arch="gfx1250"
                    )
        finally:
            au._RESOLVED_ATTENTION_ARCH = old_arch

    def test_prefill_2d_routes_through_unified_scalar(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            qwen3_prefill_attention_shapes,
        )
        from rocke.instances.common.attention_unified import (
            UnifiedAttention2DSpec,
            UnifiedAttentionProblem,
            build_unified_attention_2d,
            supports_native_unified_attention,
        )

        for shape in qwen3_prefill_attention_shapes():
            problem = UnifiedAttentionProblem(
                total_q=shape.total_q,
                num_seqs=shape.num_seqs,
                num_query_heads=shape.num_query_heads,
                num_kv_heads=shape.num_kv_heads,
                head_size=shape.head_size,
                block_size=shape.block_size,
                max_seqlen_q=shape.max_seqlen_q,
                max_seqlen_k=shape.max_seqlen_k,
                dtype=shape.dtype,
                use_sinks=False,
            )
            ok, why = supports_native_unified_attention(problem)
            self.assertTrue(ok, f"prefill q{shape.q_len}: {why}")
            ll = lower_kernel_to_llvm(
                build_unified_attention_2d(
                    UnifiedAttention2DSpec(
                        problem=problem, name="rocke_gfx1250_qwen3_prefill2d_scalar"
                    )
                ),
                arch="gfx1250",
            )
            self.assertIn("rocke_gfx1250_qwen3_prefill2d_scalar", ll)
            self.assertIn("define amdgpu_kernel void", ll)
            self.assertIn("addrspace(1)", ll)


class TestGfx1250Qwen3KvCache(unittest.TestCase):
    def test_fp8_kv_dequant_lowers_with_explicit_scale_multiply(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.qwen3_kv_cache import (
            Qwen3KvDequantSpec,
            build_qwen3_kv_dequant_smoke,
        )

        ll = lower_kernel_to_llvm(
            build_qwen3_kv_dequant_smoke(
                Qwen3KvDequantSpec(kv_storage_dtype="fp8e4m3"),
                arch="gfx1250",
            ),
            arch="gfx1250",
        )
        self.assertIn("llvm.amdgcn.cvt.pk.f32.fp8", ll)
        self.assertIn("fmul float", ll)
        self.assertNotIn("cvt.scalef32.pk.f32.fp8", ll)

    def test_bf8_kv_dequant_lowers_with_explicit_scale_multiply(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.qwen3_kv_cache import (
            Qwen3KvDequantSpec,
            build_qwen3_kv_dequant_smoke,
        )

        ll = lower_kernel_to_llvm(
            build_qwen3_kv_dequant_smoke(
                Qwen3KvDequantSpec(kv_storage_dtype="bf8e5m2"),
                arch="gfx1250",
            ),
            arch="gfx1250",
        )
        self.assertIn("llvm.amdgcn.cvt.pk.f32.bf8", ll)
        self.assertIn("fmul float", ll)
        self.assertNotIn("cvt.scalef32.pk.f32.bf8", ll)

    def test_kv_append_rope_lowers_for_bf16_fp8_bf8_storage(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.qwen3_kv_cache import (
            Qwen3KvAppendRopeSpec,
            build_qwen3_kv_append_rope,
        )

        expectations = {
            "bf16": "store bfloat",
            "fp8e4m3": "llvm.amdgcn.cvt.pk.fp8.f32",
            "bf8e5m2": "llvm.amdgcn.cvt.pk.bf8.f32",
        }
        for kv_storage_dtype, needle in expectations.items():
            ll = lower_kernel_to_llvm(
                build_qwen3_kv_append_rope(
                    Qwen3KvAppendRopeSpec(kv_storage_dtype=kv_storage_dtype),
                    arch="gfx1250",
                ),
                arch="gfx1250",
            )
            self.assertIn("rocke_gfx1250_qwen3_kv_append_rope", ll)
            self.assertIn("fsub float", ll)
            self.assertIn("fadd float", ll)
            self.assertIn(needle, ll)


@unittest.skipUnless(
    _ARCH == "gfx1250",
    f"gfx1250 WMMA GPU probe; running on {_ARCH!r} (needs a gfx1250 device)",
)
class TestGfx1250Gpu(unittest.TestCase):
    """GPU probe: build + numerically verify WMMA GEMM / attention on hardware."""

    def _run(self, module, *args, timeout=300):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_PYDIR)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=str(_PYDIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr)

    def test_wmma_probe_16x16x32(self):
        rc, out = self._run(
            "rocke.examples.gfx1250.wmma_probe", "--m", "16", "--n", "16", "--k", "32"
        )
        self.assertEqual(rc, 0, f"gfx1250 WMMA probe 16x16x32 failed:\n{out[-2500:]}")
        self.assertIn("PASS", out)

    def test_wmma_probe_64x64x64(self):
        rc, out = self._run(
            "rocke.examples.gfx1250.wmma_probe", "--m", "64", "--n", "64", "--k", "64"
        )
        self.assertEqual(rc, 0, f"gfx1250 WMMA probe 64x64x64 failed:\n{out[-2500:]}")
        self.assertIn("PASS", out)

    def test_attention_fwd_verify(self):
        rc, out = self._run(
            "rocke.examples.gfx1250.attention.wmma_attention_fwd_verify",
            "--seqlen-q",
            "64",
            "--seqlen-k",
            "64",
            "--head-size",
            "64",
            "--heads",
            "4",
        )
        self.assertEqual(rc, 0, f"gfx1250 attention fwd failed:\n{out[-2500:]}")
        self.assertIn("PASS", out)

    def test_attention_fwd_causal(self):
        # Causal currently verifies through the HIP/clang path. The direct LLVM
        # path still has a WMMA scheduling/codegen issue that produces
        # deterministic NaNs; hipcc emits a correct code object for the same IR.
        rc, out = self._run(
            "rocke.examples.gfx1250.attention.wmma_attention_fwd_verify",
            "--seqlen-q",
            "64",
            "--seqlen-k",
            "64",
            "--head-size",
            "64",
            "--heads",
            "4",
            "--causal",
        )
        self.assertEqual(rc, 0, f"gfx1250 causal attention fwd failed:\n{out[-2500:]}")
        self.assertIn("PASS", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
