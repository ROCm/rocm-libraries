# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch wiring for the gfx950 lightning indexer (CPU-only, no GPU).

The kernel is arch-neutral; this asserts the gfx950 candidates are registered and
route the same way as gfx942 (mfma for 16-aligned shapes, scalar otherwise).
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
    IndexerRequest,
    dispatch_lightning_indexer,
    indexer_candidates,
    indexer_sweep_space,
)


def _req(**over) -> IndexerRequest:
    base = dict(
        seqlen_q=16,
        seqlen_k=64,
        n_index_heads=32,
        index_head_dim=128,
        arch="gfx950",
        block_size=64,
    )
    base.update(over)
    return IndexerRequest(**base)


class TestRegistration(unittest.TestCase):
    def test_gfx950_candidates_registered(self):
        names = {c.name for c in indexer_candidates()}
        self.assertLessEqual(
            {"lightning_indexer_gfx950_mfma", "lightning_indexer_gfx950"}, names
        )


class TestRouting(unittest.TestCase):
    def test_aligned_selects_mfma(self):
        res = dispatch_lightning_indexer(_req())
        self.assertEqual(res.candidate.name, "lightning_indexer_gfx950_mfma")
        self.assertEqual(res.candidate.algorithm, "mfma_v1")
        self.assertEqual(res.spec.body, "mfma")
        self.assertEqual(res.grid, (1, 4, 1))
        self.assertEqual(res.block, (64, 1, 1))

    def test_unaligned_falls_back_to_scalar(self):
        res = dispatch_lightning_indexer(
            _req(seqlen_q=8, n_index_heads=4, index_head_dim=16)
        )
        self.assertEqual(res.candidate.name, "lightning_indexer_gfx950")
        self.assertEqual(res.candidate.algorithm, "scalar_v1")
        self.assertEqual(res.spec.body, "scalar")

    def test_sweep_space(self):
        self.assertEqual(len(indexer_sweep_space(_req())), 2)
        self.assertEqual(
            len(
                indexer_sweep_space(
                    _req(seqlen_q=8, n_index_heads=4, index_head_dim=16)
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
