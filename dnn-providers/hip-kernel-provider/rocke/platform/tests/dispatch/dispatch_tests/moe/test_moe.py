# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Element-path + arch-gating tests for the fused-MoE dispatcher family."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.moe import MoeRequest, dispatch_moe, moe_candidates


def _moe(arch="gfx950", dtype="fp16", **kw):
    base = {
        "num_tokens": 128,
        "hidden": 7168,
        "intermediate": 2048,
        "num_experts": 256,
        "top_k": 8,
        "arch": arch,
        "dtype": dtype,
    }
    base.update(kw)
    return MoeRequest(**base)


class TestMoeDispatch(unittest.TestCase):
    def test_fp16_selects_f16_mega(self):
        r = dispatch_moe(_moe(dtype="fp16"))
        self.assertEqual(r.candidate.spec_id, "mega_f16")

    def test_bf16_selects_f16_mega(self):
        r = dispatch_moe(_moe(dtype="bf16"))
        self.assertEqual(r.candidate.spec_id, "mega_f16")
        self.assertEqual(r.spec.dtype, "bf16")

    def test_fp8_selects_fp8_mega(self):
        r = dispatch_moe(_moe(dtype="fp8"))
        self.assertEqual(r.candidate.spec_id, "mega_fp8")
        # fp8 hero atom K=128.
        self.assertEqual(r.spec.gate_up_k, 128)

    def test_fp8_mxfp4_situ_selects_packed_weight_path(self):
        r = dispatch_moe(
            _moe(
                dtype="fp8",
                weight_dtype="mxfp4",
                activation="situ",
                activation_beta=2.0,
                activation_linear_beta=3.0,
            )
        )
        self.assertEqual(r.candidate.spec_id, "mega_mxfp4")
        self.assertEqual(r.spec.weight_dtype, "mxfp4")
        self.assertEqual((r.spec.gate_up_k, r.spec.down_k), (128, 128))
        self.assertTrue(r.spec.mxfp4_native)
        self.assertEqual(r.spec.warp_n, 8)
        self.assertTrue(r.spec.prefetch_routing_meta)
        self.assertTrue(r.spec.mxfp4_preshuffled)
        self.assertTrue(r.spec.pipeline_native_gateup)
        self.assertFalse(r.spec.pipeline_native_down)
        self.assertFalse(r.spec.use_dtla)

    def test_native_mxfp4_down_pipeline_is_t1_only(self):
        common = {
            "dtype": "fp8",
            "weight_dtype": "mxfp4",
            "activation": "situ",
        }
        self.assertTrue(
            dispatch_moe(_moe(num_tokens=1, **common)).spec.pipeline_native_down
        )
        self.assertFalse(
            dispatch_moe(_moe(num_tokens=8, **common)).spec.pipeline_native_down
        )

    def test_rejects_unknown_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(dtype="i8"))

    def test_rejects_topk_gt_experts(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(num_experts=4, top_k=8))

    def test_rejects_gfx942_no_atom(self):
        # gfx942 lacks the 16x16x32 MoE f16 atom -> unsupported.
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx942", dtype="fp16"))

    def test_rejects_rdna_arch(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx1151"))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx000"))

    def test_candidates_dtype_exclusive(self):
        req = _moe(dtype="fp8")
        supported = [c for c in moe_candidates() if c.admits(req)[0]]
        self.assertEqual([c.spec_id for c in supported], ["mega_fp8"])

    def test_mxfp4_candidate_is_weight_dtype_exclusive(self):
        req = _moe(dtype="fp8", weight_dtype="mxfp4", activation="situ")
        supported = [c for c in moe_candidates() if c.admits(req)[0]]
        self.assertEqual([c.spec_id for c in supported], ["mega_mxfp4"])

    def test_unique_candidate_names(self):
        names = [c.name for c in moe_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_block_size_default(self):
        r = dispatch_moe(_moe(dtype="fp16"))
        # warp_m=1 * warp_n=4 * wave_size=64 = 256.
        self.assertEqual(r.spec.block_size, 256)


if __name__ == "__main__":
    unittest.main()
