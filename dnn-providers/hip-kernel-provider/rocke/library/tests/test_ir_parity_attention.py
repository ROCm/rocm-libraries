# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention IR-parity golden gate (library layer).

The representative cross-flavor IR byte-stability gate lives in the platform
harness (rocke_ir_parity_harness) for the platform kernel families; the
unified-attention cases moved here with the rest of the SDPA/MHA vertical. This
reuses the platform harness's generic golden machinery (imported as a top-level
module via the library conftest path wiring -- library legally consumes platform
test infra) driven over the attention cases through its ``cases_fn`` hook, and
checks against the library-owned golden.

    PYTHONPATH=library:platform/Python python3 -m pytest \
        rocke/library/tests/test_ir_parity_attention.py
"""

from __future__ import annotations

import unittest
from pathlib import Path

import rocke_ir_parity_harness as _h  # platform generic golden machinery
import rocke_ir_parity_harness_attention as _att  # library attention cases

_GOLDEN = Path(__file__).resolve().parent / "golden" / "rocke_attention_ir_sha256.json"


class TestAttentionIrParityCoverage(unittest.TestCase):
    def test_cases_cover_unified_attention(self):
        cs = _att.cases()
        self.assertIn("unified_attention", {c["family"] for c in cs})
        self.assertIn("gfx950", {c["arch"] for c in cs})
        self.assertEqual(len(cs), len({c["case_id"] for c in cs}), "duplicate case ids")


class TestAttentionIrParityGolden(unittest.TestCase):
    def test_ir_matches_golden(self):
        self.assertTrue(
            _GOLDEN.exists(),
            f"golden missing: {_GOLDEN} (bless: rocke_ir_parity_harness.build_golden"
            "(cases_fn=rocke_ir_parity_harness_attention.cases))",
        )
        drift = _h.check_golden(_GOLDEN, _h.current_flavor(), cases_fn=_att.cases)
        self.assertEqual(
            drift, [], "attention IR drift vs golden:\n  " + "\n  ".join(drift)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
