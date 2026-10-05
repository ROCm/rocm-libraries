# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch wiring for the gfx942 lightning indexer (CPU-only, no GPU).

Asserts the candidate is registered and selectable, that it produces the spec /
grid / signature, that the arch gate rejects gfx950, and that a non-DSA op is
rejected (the DSA and standard-attention candidate sets are mutually exclusive).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[2]
_LIBRARY = _TESTS.parent
_PLATFORM_PYTHON = _LIBRARY.parent / "platform" / "python"
for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dispatch.dsa import (  # noqa: E402
    DSA_INDEXER_REGISTRY,
    IndexerRequest,
    dispatch_lightning_indexer,
    indexer_candidates,
    indexer_sweep_space,
)


def _req(**over) -> IndexerRequest:
    base = dict(
        seqlen_q=8,
        seqlen_k=64,
        n_index_heads=4,
        index_head_dim=16,
        arch="gfx942",
        block_size=64,
    )
    base.update(over)
    return IndexerRequest(**base)


class TestRegistration(unittest.TestCase):
    def test_gfx942_candidates_registered(self):
        # The shared registry also holds the gfx950 candidates; assert the gfx942
        # pair is present rather than an exact set.
        names = {c.name for c in indexer_candidates()}
        self.assertLessEqual(
            {"lightning_indexer_gfx942_mfma", "lightning_indexer_gfx942"}, names
        )

    def test_candidate_has_builder(self):
        for c in indexer_candidates():
            self.assertTrue(callable(c.build), f"{c.name} missing build")

    def test_require_build(self):
        self.assertTrue(DSA_INDEXER_REGISTRY.require_build)


class TestRouting(unittest.TestCase):
    def test_aligned_selects_mfma(self):
        res = dispatch_lightning_indexer(
            _req(seqlen_q=16, seqlen_k=64, index_head_dim=128)
        )
        self.assertEqual(res.candidate.name, "lightning_indexer_gfx942_mfma")
        self.assertEqual(res.candidate.algorithm, "mfma_v1")
        self.assertEqual(res.spec.body, "mfma")
        self.assertEqual(res.grid, (1, 4, 1))  # (seqlen_q/16, seqlen_k/16, 1)
        self.assertEqual(res.block, (64, 1, 1))

    def test_unaligned_falls_back_to_scalar(self):
        # seqlen_q=8 is not a multiple of 16, so the MFMA candidate is filtered out.
        res = dispatch_lightning_indexer(_req(seqlen_q=8))
        self.assertEqual(res.candidate.name, "lightning_indexer_gfx942")
        self.assertEqual(res.candidate.algorithm, "scalar_v1")
        self.assertEqual(res.spec.body, "scalar")
        self.assertEqual(res.grid, (8, 1, 1))

    def test_signature_stable_across_bodies(self):
        for req in (_req(seqlen_q=16, index_head_dim=128), _req(seqlen_q=8)):
            res = dispatch_lightning_indexer(req)
            self.assertEqual(
                [s["name"] for s in res.signature],
                ["index_q", "index_k", "w", "scores", "q_pos_base"],
            )

    def test_sweep_space(self):
        # aligned: both candidates admit; unaligned: only scalar.
        self.assertEqual(
            len(indexer_sweep_space(_req(seqlen_q=16, index_head_dim=128))), 2
        )
        self.assertEqual(len(indexer_sweep_space(_req(seqlen_q=8))), 1)


class TestGates(unittest.TestCase):
    def test_gfx950_now_supported(self):
        # gfx950 is a supported arch; it selects the gfx950 candidate.
        res = dispatch_lightning_indexer(
            _req(seqlen_q=16, index_head_dim=128, arch="gfx950")
        )
        self.assertEqual(res.candidate.name, "lightning_indexer_gfx950_mfma")

    def test_arch_gate_rejects_unsupported(self):
        with self.assertRaises(ValueError):
            dispatch_lightning_indexer(_req(arch="gfx1151"))

    def test_wrong_op_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_lightning_indexer(_req(op="attention"))

    def test_nonpositive_geometry_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_lightning_indexer(_req(seqlen_k=0))


if __name__ == "__main__":
    unittest.main()
