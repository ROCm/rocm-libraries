# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention's tuned candidates keep the shared tuning contract
(:func:`rocke.dispatch.tuning.testing.assert_tuning_contract`)."""

from __future__ import annotations

import unittest

from dispatch.attention import ATTENTION_EXECUTION_REGISTRY, AttentionRequest
from rocke.dispatch.tuning.testing import assert_tuning_contract, representative


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _check(test, prefixes, requests, **kwargs):
    for candidate in representative(
        ATTENTION_EXECUTION_REGISTRY.candidates(), prefixes
    ):
        with test.subTest(candidate=candidate.name):
            counts = assert_tuning_contract(candidate, requests, **kwargs)
            test.assertGreater(counts["replays"], 0)


class TestAttentionTuningContract(unittest.TestCase):
    def test_gfx950_dense_candidates(self):
        _check(
            self,
            [
                "attention_gfx950_dense_grid",
                "attention_gfx950_dense_persist",
                "attention_gfx950_dense_persist_widedma",
            ],
            [_req()],
            other_requests=[_req(batch=2, seqlen_q=4096, seqlen_k=4096)],
            default_knobs={"pv_priority": 1},
            refused_knobs=[{"seqlen_q": 64}, {"lds_num_buffers": 3}],
        )

    def test_gfx942_dense(self):
        _check(
            self,
            ["attention_gfx942_dense"],
            [
                _req(arch="gfx942", dtype="fp16"),
                _req(arch="gfx942", hdim_q=64, hdim_v=64),
            ],
            other_requests=[
                _req(arch="gfx942", dtype="fp16", seqlen_q=4096, seqlen_k=4096)
            ],
            refused_knobs=[{"lazy_rescale": False}],
        )

    def test_unified_geometries(self):
        _check(
            self,
            [
                "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
                "attention_gfx950_u2d_transposed32_nw1_mw32_t2xb_hipcc",
            ],
            [_req()],
            other_requests=[_req(batch=4, seqlen_q=2048, seqlen_k=2048)],
            default_knobs={"use_register_pv": False},
            refused_knobs=[{"not_a_knob": True}],
        )
        _check(
            self,
            ["attention_gfx950_u3d_splitkv_seg16"],
            [_req(seqlen_q=1, seqlen_k=4096, batch=4)],
        )
        _check(
            self,
            ["attention_gfx942_u2d_narrow", "attention_gfx942_u2d_gfx942_4warp"],
            [_req(arch="gfx942")],
            refused_knobs=[{"use_k_hbm_direct": True}],
        )

    def test_rdna_wmma_candidates(self):
        # Prefill: plain, causal with tails, a wave-pair D256 object, MLA
        # widths and a runtime-head-width bucket. Decode: GQA decode,
        # speculative rows on the D256 bucket, and FP8 K/V with softcap.
        def rdna(arch, **kw):
            shape = dict(arch=arch, dtype="fp16", nhead_q=16, nhead_k=4, num_cus=40)
            shape.update(mask_type=0)
            shape.update(kw)
            return _req(**shape)

        for arch in ("gfx11-generic", "gfx1151", "gfx12-generic", "gfx1201"):
            tag = arch.replace("-", "_")
            _check(
                self,
                [f"attention_{tag}_wmma_prefill"],
                [
                    rdna(arch),
                    rdna(arch, seqlen_q=100, seqlen_k=100, mask_type=1),
                    rdna(arch, hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4),
                    rdna(arch, hdim_q=192, hdim_v=128),
                    rdna(arch, seqlen_q=17, seqlen_k=17, runtime_head_dims=True),
                ],
                other_requests=[
                    rdna(arch, seqlen_q=37, seqlen_k=53, mask_type=1),
                    rdna(arch, hdim_q=96, hdim_v=96, seqlen_q=2048, seqlen_k=2048),
                ],
                refused_knobs=[{"not_a_knob": True}, {"num_waves": 2}],
            )
            _check(
                self,
                [f"attention_{tag}_wmma_decode"],
                [
                    rdna(arch, seqlen_q=1, seqlen_k=4096, mask_type=2),
                    rdna(arch, seqlen_q=4, seqlen_k=777, hdim_q=256, hdim_v=256),
                    rdna(
                        arch, seqlen_q=1, seqlen_k=4096, use_fp8=True, use_softcap=True
                    ),
                ],
                other_requests=[
                    rdna(arch, seqlen_q=2, seqlen_k=100, batch=8, mask_type=2),
                ],
                refused_knobs=[{"not_a_knob": True}, {"block_m": 64}],
            )


if __name__ == "__main__":
    unittest.main()
