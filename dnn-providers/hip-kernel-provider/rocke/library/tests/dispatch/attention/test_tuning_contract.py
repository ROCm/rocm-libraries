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
    def test_gfx950_dense_variants(self):
        _check(
            self,
            [
                "attention_gfx950_dense_grid_default",
                "attention_gfx950_dense_persist_default",
                "attention_gfx950_dense_persist_widedma_default",
                "attention_gfx950_dense_grid_bm128",
                "attention_gfx950_dense_persist_bm128",
                "attention_gfx950_dense_persist_widedma_bm128",
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


if __name__ == "__main__":
    unittest.main()
