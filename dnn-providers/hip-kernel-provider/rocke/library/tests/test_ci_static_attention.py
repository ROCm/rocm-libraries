# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI-facing static checks for library attention shapes.

Moved from platform/tests/test_rocke_ci_static.py
(TestExampleProblemSamples.test_gfx950_attention_trace_samples_decode_and_prefill)
because it references the library ``builders`` package and therefore belongs in
the library test layer.
"""

from __future__ import annotations

import unittest
from importlib.resources import files
from pathlib import Path


def _read_jsonl(path, limit=None):
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                import json

                rows.append(json.loads(line))
    return rows


class TestExampleAttentionShapes(unittest.TestCase):
    """Keep non-GPU CI anchored to attention shapes shipped with examples."""

    def test_gfx950_attention_trace_samples_decode_and_prefill(self):
        rows = _read_jsonl(
            files("builders.gfx950.attention").joinpath("aiter_ua_shapes.json")
        )
        kinds = {row["kind"] for row in rows}
        head_sizes = {int(row["head_size"]) for row in rows}
        max_q = {int(row["max_seqlen_q"]) for row in rows}

        self.assertIn("2d", kinds)
        self.assertIn(64, head_sizes)
        self.assertTrue(any(q == 1 for q in max_q), "decode sample missing")
        self.assertTrue(any(q > 1 for q in max_q), "prefill sample missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
