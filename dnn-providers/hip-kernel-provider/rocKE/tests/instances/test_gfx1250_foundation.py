# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU foundation contracts for the gfx1250 (gfx1250-class) port.

Consolidated arch/ISA/WMMA-lowering facts plus the Qwen3-30B-A3B shape table and
element-wise quant contracts. Arch-level facts that span multiple targets live
in ``test/test_rocke_multiarch.py``; this file is the gfx1250-specific surface
the operator sandboxes rely on.

    PYTHONPATH=Python python3 -m pytest rocke/tests/test_gfx1250_foundation.py
"""

from __future__ import annotations

import unittest


class TestGfx1250Arch(unittest.TestCase):
    """gfx1250 ArchTarget / ISA backend / WMMA catalog facts."""

    def test_gfx1250_capabilities_are_exposed(self):
        from rocke.core.arch import ArchTarget, known_arches

        self.assertIn("gfx1250", known_arches())
        target = ArchTarget.from_gfx("gfx1250")
        self.assertEqual(target.family, "cdna")
        self.assertEqual(target.target_family, "gfx12_cdna")
        self.assertEqual(target.wave_size, 32)
        self.assertEqual(target.isa_triple, "amdgcn-amd-amdhsa--gfx1250")
        self.assertEqual(target.matrix_path, "wmma")
        self.assertFalse(target.has_mfma)
        self.assertTrue(target.has_wmma)
        self.assertEqual(target.waitcnt_model, "split_gfx1250")
        self.assertEqual(target.barrier_model, "split_named_cluster")
        self.assertEqual(target.virtual_address_bits, 57)
        self.assertTrue(target.requires_shader_end_padding)
        self.assertTrue(target.wgp_cache_lds_shared)
        # gfx1250 HAS async global->LDS DMA via its own GFX12 family
        # (global_load_async_to_lds_b128 + s_wait_asynccnt), GPU-validated via the
        # DTLA path -- distinct from the gfx9 buffer_load_lds (which does not
        # select here). TDM is not verified on this part.
        self.assertTrue(target.memory.has_async_global_lds)
        self.assertFalse(target.memory.has_tdm)
        # has_async_lds / has_ds_read_tr denote the gfx9/gfx950 ds_read_tr / async
        # ABI (MFMA-pipeline gating), which gfx1250 lacks; its async/transpose use
        # distinct GFX12 opcodes gated in the ISA backend, so these stay False.
        self.assertFalse(target.memory.has_async_lds)
        self.assertFalse(target.memory.has_ds_read_tr)

    def test_wmma_atom_is_k32(self):
        from rocke.core.arch import ArchTarget

        t = ArchTarget.from_gfx("gfx1250")
        op = t.mma.by_op_id("wmma_gfx1250_f32_16x16x32_f16")
        self.assertIsNotNone(op)
        self.assertEqual((op.m, op.n, op.k), (16, 16, 32))
        # A/B 16 elems per lane, accumulator 8 f32 per lane.
        self.assertEqual(op.a_frag_len, 16)
        self.assertEqual(op.b_frag_len, 16)
        self.assertEqual(op.c_frag_len, 8)
        # No plain 'mma' family; the wave32 WMMA fp16/bf16 atom is K=32 (not
        # gfx1201's K=16).
        self.assertEqual(
            t.mma.enumerate(a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"), []
        )
        wmma = t.mma.enumerate(
            family="wmma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"
        )
        self.assertEqual([o.shape for o in wmma], [(16, 16, 32)])

    def test_backend_selection(self):
        from rocke.core.isa import backend_for
        from rocke.core.isa.backend import Gfx1250Backend

        self.assertIsInstance(backend_for("gfx1250"), Gfx1250Backend)


class TestGfx1250WmmaLowering(unittest.TestCase):
    """The K=32 WMMA atom lowers to the confirmed gfx1250 intrinsics."""

    def test_wmma_gemm_lowers_to_gfx1250_intrinsic(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.wmma_gemm import (
            WmmaGemmSpec,
            build_wmma_gemm,
        )

        ll = lower_kernel_to_llvm(
            build_wmma_gemm(WmmaGemmSpec(), arch="gfx1250"), arch="gfx1250"
        )
        # The 8-operand gfx1250 WMMA call (i1,A,i1,B,i16,C,i1,i1) and the
        # on-demand <16 x half> declaration must both be present.
        self.assertIn(
            "call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(",
            ll,
        )
        self.assertIn("i1 false, <16 x half>", ll)
        self.assertIn("i16 0, <8 x float>", ll)
        self.assertIn(
            "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16("
            "i1 immarg, <16 x half>, i1 immarg, <16 x half>, i16 immarg, "
            "<8 x float>, i1 immarg, i1 immarg)",
            ll,
        )

    def test_bf16_lowers_without_i16_bitcast(self):
        # gfx1250 bf16 WMMA uses <16 x bfloat> directly (v16bf16), unlike the
        # gfx11/gfx12 path which bitcasts <N x bfloat> -> <N x i16>.
        from rocke.core.ir import BF16, F16, IRBuilder, PtrType
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        b = IRBuilder("bf16_wmma_smoke")
        b.kernel.attrs["max_workgroup_size"] = 32
        C = b.param("C", PtrType(F16, "global"), writeonly=True, align=16)
        a = b.zero_vec(BF16, 16)
        acc = b.zero_vec_f32(8)
        d = b.wmma_gfx1250_f32_16x16x32_bf16(a, a, acc)
        b.global_store(C, b.const_i32(0), b.trunc_f32_to_f16(b.vec_extract(d, 0)))
        ll = lower_kernel_to_llvm(b.kernel, arch="gfx1250")
        self.assertIn("v8f32.v16bf16", ll)
        self.assertNotIn("to <16 x i16>", ll)

    def test_hip_bf16_wmma_lowering_uses_gfx1250_builtin(self):
        from rocke.core.ir import BF16, F16, IRBuilder, PtrType
        from rocke.core.lower_hip import lower_kernel_to_hip

        b = IRBuilder("hip_bf16_wmma_smoke")
        b.kernel.attrs["max_workgroup_size"] = 32
        out = b.param("out", PtrType(F16, "global"), writeonly=True, align=16)
        frag = b.zero_vec(BF16, 16)
        acc = b.zero_vec_f32(8)
        result = b.wmma_gfx1250_f32_16x16x32_bf16(frag, frag, acc)
        b.global_store(
            out, b.const_i32(0), b.trunc_f32_to_f16(b.vec_extract(result, 0))
        )

        hip = lower_kernel_to_hip(b.kernel, arch="gfx1250")
        self.assertIn("__builtin_amdgcn_wmma_f32_16x16x32_bf16", hip)
        self.assertIn("bf16x16", hip)


class TestQwen3A3BShapeTable(unittest.TestCase):
    """The shared Qwen3-30B-A3B geometry the day-0 sandboxes import."""

    def test_decode_shapes_match_qwen_a3b_geometry(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            decode_attention_shapes,
            decode_gemm_shapes,
            moe_shape,
            qwen3_30b_a3b_config,
        )

        cfg = qwen3_30b_a3b_config()
        self.assertEqual(cfg.hidden, 2048)
        self.assertEqual(cfg.moe_intermediate, 768)
        self.assertEqual(cfg.num_experts, 128)
        self.assertEqual(cfg.topk, 8)
        self.assertEqual(
            (cfg.num_query_heads, cfg.num_kv_heads, cfg.head_dim), (32, 4, 64)
        )
        self.assertEqual(cfg.qkv_width, 2560)

        gemms = {shape.name: shape for shape in decode_gemm_shapes()}
        self.assertEqual(
            (
                gemms["qkv_decode_bf16"].m,
                gemms["qkv_decode_bf16"].n,
                gemms["qkv_decode_bf16"].k,
            ),
            (2, 2560, 2048),
        )
        self.assertEqual(
            (
                gemms["o_decode_bf16"].m,
                gemms["o_decode_bf16"].n,
                gemms["o_decode_bf16"].k,
            ),
            (2, 2048, 2048),
        )
        self.assertEqual(
            (
                gemms["router_decode_bf16"].m,
                gemms["router_decode_bf16"].n,
                gemms["router_decode_bf16"].k,
            ),
            (2, 128, 2048),
        )

        attn = decode_attention_shapes()
        self.assertEqual(
            [shape.max_seqlen_k for shape in attn], [512, 1024, 2048, 4096]
        )
        self.assertTrue(all(shape.max_seqlen_q == 1 for shape in attn))
        self.assertTrue(all(shape.block_size == 16 for shape in attn))
        self.assertTrue(all(shape.num_queries_per_kv == 8 for shape in attn))

        moe = moe_shape()
        self.assertEqual(moe.active_pairs, 16)
        self.assertEqual(
            (moe.experts, moe.topk, moe.hidden, moe.intermediate), (128, 8, 2048, 768)
        )

    def test_prefill_shapes_are_recorded(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            qwen3_prefill_attention_shapes,
        )

        shapes = qwen3_prefill_attention_shapes()
        self.assertEqual([s.q_len for s in shapes], [64, 128])
        self.assertTrue(all(s.mode == "prefill_2d" for s in shapes))

    def test_quantization_modes_are_explicit(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            quantization_modes_for_day0,
        )

        self.assertEqual(quantization_modes_for_day0(), ("bf16", "fp8e4m3", "bf8e5m2"))
        self.assertEqual(
            quantization_modes_for_day0(include_int4=True),
            ("bf16", "fp8e4m3", "bf8e5m2", "i4_fp8", "i4_bf8"),
        )


class TestGfx1250ElementwiseQuant(unittest.TestCase):
    """Add-RMSNorm rdquant accepts the gfx1250 wave32 fp8/bf8/i8 outputs."""

    def test_rdquant_accepts_cdna_fp8_bf8_outputs(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            RDQUANT_OUT_DTYPES,
            WAVE_SIZE,
            add_rmsnorm_rdquant_specs,
        )
        from rocke.instances import (
            build_add_rmsnorm2d_rdquant,
            is_valid_add_rmsnorm2d_rdquant_spec,
        )

        specs = add_rmsnorm_rdquant_specs()
        self.assertEqual(tuple(spec.out_dtype for spec in specs), RDQUANT_OUT_DTYPES)
        for spec in specs:
            self.assertEqual(spec.wave_size, WAVE_SIZE)
            ok, why = is_valid_add_rmsnorm2d_rdquant_spec(spec, "gfx1250")
            self.assertTrue(ok, why)
            kernel = build_add_rmsnorm2d_rdquant(spec, "gfx1250")
            self.assertEqual(kernel.attrs["max_workgroup_size"], spec.block_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
