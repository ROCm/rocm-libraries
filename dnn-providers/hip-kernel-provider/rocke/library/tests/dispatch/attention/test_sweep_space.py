# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contract tests for ``attention_sweep_space`` -- the multi-engine benchmarking
primitive.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    attention_sweep_space,
    dispatch_attention_all,
    registered_attention_combos,
)
from rocke.dispatch.core import spec_identity


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
        bad = _gfx942_fp16_mha(hdim_v=64)
        self.assertEqual(attention_sweep_space(bad), ())

    def test_covers_unified_tuning_but_excludes_dense_specs(self):
        with _PinnedArch("gfx942"):
            req = _gfx942_fp16_mha()
            combos = registered_attention_combos(req)
            specs = attention_sweep_space(req)
        names = {c.name for c, _spec in combos}
        self.assertIn("attention_gfx942_dense", names)
        self.assertTrue(any(c.algorithm == "unified_tuning" for c, _ in combos))
        self.assertNotIn("attention_unified_2d", names)
        self.assertNotIn("attention_gfx942_dense_pipe", names)
        self.assertGreater(len(specs), 1)
        self.assertTrue(any(hasattr(s, "tuning_id") for s in specs))
        self.assertTrue(all(hasattr(s, "path") for s in specs))
        self.assertTrue(ATTENTION_EXECUTION_REGISTRY.require_build)
        self.assertTrue(ATTENTION_EXECUTION_REGISTRY.require_torch_binding)

    def test_specs_are_deduped(self):
        with _PinnedArch("gfx942"):
            specs = attention_sweep_space(_gfx942_fp16_mha())
        self.assertEqual(len(specs), len({spec_identity(s) for s in specs}))

    def test_sweep_matches_manual_candidate_selection(self):
        with _PinnedArch("gfx942"):
            req = _gfx942_fp16_mha()
            manual = []
            seen = set()
            for _candidate, spec in registered_attention_combos(req):
                if not hasattr(spec, "path"):
                    continue
                key = spec_identity(spec)
                if key not in seen:
                    seen.add(key)
                    manual.append(spec)
            specs = attention_sweep_space(req)
        self.assertEqual(list(specs), manual)

    def test_dispatch_all_is_one_result_per_combo(self):
        with _PinnedArch("gfx942"):
            req = _gfx942_fp16_mha()
            combos = registered_attention_combos(req)
            results = dispatch_attention_all(req)
        self.assertGreater(len(results), 1)
        self.assertEqual(len(results), len(combos))
        self.assertEqual(
            [r.candidate.name for r in results],
            [c.name for c, _spec in combos],
        )


if __name__ == "__main__":
    unittest.main()
