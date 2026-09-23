# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contract tests for ``attention_sweep_space`` -- the multi-engine benchmarking
primitive.

The ticket requires that benchmarking workflows can *evaluate multiple engines
for the same problem*. ``attention_sweep_space(req)`` is that entry point: it
returns the deduped ``select_spec`` of every candidate that supports ``req``.
These CPU-only tests make its behavior a tested contract (previously it had no
callers) so the benchmark ``sweep`` lane can rely on it.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    ATTENTION_REGISTRY,
    AttentionRequest,
    attention_sweep_space,
)


def _gfx942_fp16_mha(**kw) -> AttentionRequest:
    base = dict(
        batch=2,
        nhead_q=16,
        nhead_k=16,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="fp16",
    )
    base.update(kw)
    return AttentionRequest(**base)


class _PinnedArch:
    def __init__(self, arch: str):
        self._arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self._arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestSweepSpace(unittest.TestCase):
    def test_invalid_request_yields_empty(self):
        # Malformed request (hdim_q != hdim_v) -> no specs, no raise.
        bad = _gfx942_fp16_mha(hdim_v=64)
        self.assertEqual(attention_sweep_space(bad), ())

    def test_covers_all_supported_candidates(self):
        with _PinnedArch("gfx942"):
            req = _gfx942_fp16_mha()
            supported = ATTENTION_REGISTRY.supported(req)
            specs = attention_sweep_space(req)
        # One spec per distinct supported candidate spec (deduped by hash). This
        # shape has >1 supported candidate (dense_pipe + unified_2d), so the sweep
        # genuinely spans multiple engines.
        self.assertGreater(len(supported), 1)
        self.assertGreaterEqual(len(specs), 1)
        self.assertLessEqual(len(specs), len(supported))

    def test_specs_are_deduped(self):
        with _PinnedArch("gfx942"):
            specs = attention_sweep_space(_gfx942_fp16_mha())
        # No two returned specs are identical.
        self.assertEqual(len(specs), len({repr(s) for s in specs}))

    def test_sweep_matches_manual_candidate_selection(self):
        with _PinnedArch("gfx942"):
            req = _gfx942_fp16_mha()
            manual = []
            seen = set()
            for c in ATTENTION_REGISTRY.supported(req):
                s = c.select_spec(req)
                if repr(s) not in seen:
                    seen.add(repr(s))
                    manual.append(s)
            specs = attention_sweep_space(req)
        self.assertEqual(list(specs), manual)


if __name__ == "__main__":
    unittest.main()
