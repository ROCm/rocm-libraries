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
    base = dict(seqlen_q=8, seqlen_k=64, n_index_heads=4, index_head_dim=16,
                arch="gfx942", block_size=64)
    base.update(over)
    return IndexerRequest(**base)


class TestRegistration(unittest.TestCase):
    def test_single_candidate_registered(self):
        names = [c.name for c in indexer_candidates()]
        self.assertEqual(names, ["lightning_indexer_gfx942"])

    def test_candidate_has_builder(self):
        for c in indexer_candidates():
            self.assertTrue(callable(c.build), f"{c.name} missing build")

    def test_require_build(self):
        self.assertTrue(DSA_INDEXER_REGISTRY.require_build)


class TestRouting(unittest.TestCase):
    def test_selects_indexer(self):
        res = dispatch_lightning_indexer(_req())
        self.assertEqual(res.candidate.name, "lightning_indexer_gfx942")
        self.assertEqual(res.candidate.algorithm, "scalar_v1")

    def test_spec_grid_signature(self):
        res = dispatch_lightning_indexer(_req(seqlen_q=8))
        self.assertEqual(res.grid, (8, 1, 1))
        self.assertEqual(res.block, (64, 1, 1))
        self.assertEqual(
            [s["name"] for s in res.signature],
            ["index_q", "index_k", "w", "scores", "q_pos_base"],
        )

    def test_sweep_space_size_one(self):
        self.assertEqual(len(indexer_sweep_space(_req())), 1)


class TestGates(unittest.TestCase):
    def test_arch_gate_rejects_gfx950(self):
        with self.assertRaises(ValueError):
            dispatch_lightning_indexer(_req(arch="gfx950"))

    def test_wrong_op_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_lightning_indexer(_req(op="attention"))

    def test_nonpositive_geometry_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_lightning_indexer(_req(seqlen_k=0))


if __name__ == "__main__":
    unittest.main()
