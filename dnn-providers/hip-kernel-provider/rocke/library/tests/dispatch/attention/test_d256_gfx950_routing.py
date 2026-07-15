# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Routing tests for the gfx950 bf16 D256 prefill dispatch candidate.

CPU-only: exercises the ``attention_gfx950_d256`` candidate's selection through
``dispatch_attention`` (filter -> priority -> pick). No GPU required.
"""

from __future__ import annotations

import unittest

from dispatch.attention import AttentionRequest, _problem, dispatch_attention
from kernels.common import attention_unified as au


def _d256(arch="gfx950", **kw):
    base = dict(
        batch=2,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=4096,
        seqlen_k=4096,
        hdim_q=256,
        hdim_v=256,
        arch=arch,
        dtype="bf16",
        mask_type=1,  # causal prefill
    )
    base.update(kw)
    return AttentionRequest(**base)


def _routed_spec_id(req):
    """spec_id of the winning candidate, or None if nothing supports the req."""
    try:
        return dispatch_attention(req).candidate.spec_id
    except ValueError:
        return None


class TestD256Gfx950Routing(unittest.TestCase):
    def test_selects_gfx950_d256_for_cohort(self):
        r = dispatch_attention(_d256())
        self.assertEqual(r.candidate.spec_id, "gfx950_d256")
        self.assertEqual(r.spec.path, "2d")

    def test_outranks_generic_2d(self):
        # priority 5 must beat the generic unified_2d (priority 10).
        self.assertEqual(dispatch_attention(_d256()).candidate.priority, 5)

    def test_force_by_algorithm(self):
        r = dispatch_attention(_d256(algorithm="d256_gfx950"))
        self.assertEqual(r.candidate.spec_id, "gfx950_d256")

    def test_rejects_non_gfx950(self):
        self.assertNotEqual(_routed_spec_id(_d256(arch="gfx942")), "gfx950_d256")

    def test_rejects_non_bf16(self):
        self.assertNotEqual(_routed_spec_id(_d256(dtype="fp16")), "gfx950_d256")

    def test_rejects_non_d256(self):
        self.assertNotEqual(
            _routed_spec_id(_d256(hdim_q=128, hdim_v=128)), "gfx950_d256"
        )

    def test_rejects_decode(self):
        # q=1 decode over long kv routes to the 3D split-KV path, not our cohort.
        self.assertNotEqual(
            _routed_spec_id(
                _d256(batch=1, nhead_q=16, nhead_k=16, seqlen_q=1, seqlen_k=8192)
            ),
            "gfx950_d256",
        )

    def test_rejects_sliding_window(self):
        self.assertNotEqual(_routed_spec_id(_d256(sliding_window=256)), "gfx950_d256")


class TestD256Gfx950SpecOverrides(unittest.TestCase):
    """MVP: the dispatcher declares the knobs; the orchestrator accepts them."""

    EXPECTED_KEYS = {
        "use_mfma_32x32",
        "use_transposed_qk_32x32",
        "use_q_direct_reg",
        "use_transposed_half_local_pv",
        "use_transposed_scalar_state",
        "use_transposed_mask_once",
        "use_transposed_mask_limit",
        "use_mask_phase_split",
        "use_register_pv",
        "use_k_single_buffer",
        "use_v_double_buffer",
        "use_early_v_schedule",
        "use_sched_barrier",
        "use_softmax_mfma_interleave",
        "softmax_interleave_mode",
        "softmax_interleave_groups",
        "use_fast_paged_kv_desc",
        "use_mfma32_skip_legacy_qreg",
        "use_kq_lds_pad",
        "kq_lds_pad_halves",
    }

    def setUp(self):
        self._saved = getattr(au, "_RESOLVED_ATTENTION_ARCH", None)
        au._RESOLVED_ATTENTION_ARCH = "gfx950"

    def tearDown(self):
        au._RESOLVED_ATTENTION_ARCH = self._saved

    def test_ssot_constellation(self):
        ov = au._d256_gfx950_spec_overrides()
        self.assertEqual(set(ov), self.EXPECTED_KEYS)
        self.assertIs(ov["use_softmax_mfma_interleave"], True)
        self.assertEqual(ov["softmax_interleave_mode"], 2)
        self.assertIs(ov["use_register_pv"], False)

    def test_dispatched_spec_carries_overrides(self):
        # the candidate declares the knobs on the dispatched spec (via the SSOT).
        spec = dispatch_attention(_d256()).spec
        self.assertEqual(
            spec.tiled_overrides,
            tuple(sorted(au._d256_gfx950_spec_overrides().items())),
        )

    def test_generic_candidate_has_no_overrides(self):
        # a non-cohort shape (hd128) routes to the generic candidate -> empty.
        spec = dispatch_attention(_d256(hdim_q=128, hdim_v=128)).spec
        self.assertEqual(spec.tiled_overrides, ())

    def test_orchestrator_accepts_overrides(self):
        problem = _problem(_d256())
        # no-arg path: _impl already applies the D256 override, and
        # _resolve_lds_budget still runs (a valid spec is returned).
        base = au._tiled_spec_from_problem(problem)
        self.assertIs(base.use_softmax_mfma_interleave, True)
        self.assertIs(base.use_register_pv, False)
        # passing the SSOT overrides explicitly reproduces the same valid spec
        # (proves overrides= is plumbed + applied, and the resolver still runs).
        ov = au._tiled_spec_from_problem(
            problem, overrides=au._d256_gfx950_spec_overrides()
        )
        self.assertEqual(ov, base)

    def test_overrides_reach_the_spec(self):
        # an override that violates a spec invariant is raised by the spec's
        # __post_init__ -> proves overrides= is really applied, not ignored.
        problem = _problem(_d256())
        with self.assertRaises(ValueError):
            au._tiled_spec_from_problem(problem, overrides={"use_register_pv": True})


if __name__ == "__main__":
    unittest.main()
