# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 ``attention_dense`` dispatcher wiring (AICK-1664).

Required by ``library/dispatch/AGENTS.md`` step 4. Covers:
  - the candidate is registered and discoverable, with the right spec_id/algorithm
  - OPT-IN ONLY: ``algorithm="auto"`` never selects it, despite priority 3 outranking
    every other attention candidate
  - ``spec_id`` is an equivalent opt-in door
  - routing on gfx942, and rejection of every out-of-scope request
  - ``dense_persistent``: 'auto' resolves to off (accepted), explicit 'on' is rejected
    rather than silently downgraded
  - the dispatched ``kernel_name_override`` is batch-unique and matches what
    ``build_attention_dense`` actually emits

The priority-3 tests are the load-bearing ones: the arm sorts ahead of every other
candidate, so the opt-in check is the ONLY thing keeping a correctness-first P0 kernel
off the default gfx942 path.
"""

from __future__ import annotations

import dataclasses
import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionRequest,
    _dense_spec,
    attention_candidates,
    dispatch_attention,
)
from kernels.gfx942.attention_dense import build_attention_dense

_NAME = "attention_gfx942_dense"
_SPEC_ID = "gfx942_attention_dense"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=128,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        mask_type=1,
        algorithm="attention_dense",
    )
    base.update(kw)
    return AttentionRequest(**base)


def _candidate():
    return next(c for c in attention_candidates() if c.name == _NAME)


class _Gfx942Arch:
    """Pin _RESOLVED_ATTENTION_ARCH so routing does not depend on the host GPU."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestGfx942DenseRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, [c.name for c in attention_candidates()])

    def test_spec_id_and_algorithm(self):
        c = _candidate()
        self.assertEqual(c.spec_id, _SPEC_ID)
        self.assertEqual(c.algorithm, "attention_dense")

    def test_priority_outranks_every_other_candidate(self):
        """Documents WHY the opt-in gate matters: nothing else holds this arm back."""
        c = _candidate()
        others = [o for o in attention_candidates() if o.name != _NAME]
        self.assertTrue(all(c.priority <= o.priority for o in others))


class TestGfx942DenseOptIn(unittest.TestCase):
    def test_auto_algorithm_never_selects_it(self):
        with _Gfx942Arch():
            ok, why = _candidate().supports(_req(algorithm="auto", spec_id="auto"))
            self.assertFalse(ok, "attention_dense must never be auto-selected")
            self.assertIn("opt-in", why)
            routed = dispatch_attention(_req(algorithm="auto", spec_id="auto"))
            self.assertNotEqual(routed.candidate.name, _NAME)

    def test_spec_id_is_an_equivalent_opt_in(self):
        with _Gfx942Arch():
            ok, why = _candidate().supports(_req(algorithm="auto", spec_id=_SPEC_ID))
            self.assertTrue(ok, why)

    def test_routes_on_explicit_algorithm(self):
        with _Gfx942Arch():
            r = dispatch_attention(_req())
            self.assertEqual(r.candidate.name, _NAME)
            self.assertEqual(r.spec.path, "2d")
            self.assertEqual(r.spec.name, "rocke_attention_dense_gfx942")


class TestGfx942DenseSupportGates(unittest.TestCase):
    def test_rejects_non_gfx942_arch(self):
        ok, why = _candidate().supports(_req(arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("gfx942", why)

    def test_rejects_unsupported_dtype(self):
        with _Gfx942Arch():
            self.assertFalse(_candidate().supports(_req(dtype="fp8"))[0])

    def test_rejects_sliding_window(self):
        with _Gfx942Arch():
            ok, why = _candidate().supports(_req(sliding_window=64))
            self.assertFalse(ok)
            self.assertIn("sliding-window", why)

    def test_rejects_ragged_sequence_length(self):
        """_dense_spec sets ragged=True for any non-256-multiple self-attention
        length -- most real serving shapes. P0 must decline, not select-then-fail."""
        with _Gfx942Arch():
            ok, why = _candidate().supports(_req(seqlen_q=1000, seqlen_k=1000))
            self.assertFalse(ok)
            self.assertIn("ragged", why)


class TestGfx942DensePersistent(unittest.TestCase):
    def test_auto_persistent_is_resolved_off_and_accepted(self):
        """'auto' would turn persistent ON at this size; the candidate resolves it to
        off (persistent is P4) rather than rejecting the request."""
        with _Gfx942Arch():
            ok, why = _candidate().supports(
                _req(seqlen_q=8192, seqlen_k=8192, dense_persistent="auto")
            )
            self.assertTrue(ok, why)

    def test_explicit_persistent_on_is_rejected_not_downgraded(self):
        """An explicit 'on' must NOT be silently answered with a default-grid kernel."""
        with _Gfx942Arch():
            ok, why = _candidate().supports(_req(dense_persistent="on"))
            self.assertFalse(ok)
            self.assertIn("P4", why)


class TestGfx942DenseSpecIdentity(unittest.TestCase):
    def test_kernel_name_override_is_batch_unique(self):
        """The kernel bakes batch into the buffer extents; the dispatched identity
        must disambiguate it or a name-keyed cache serves the B=1 binary."""
        with _Gfx942Arch():
            names = {
                dispatch_attention(_req(batch=b)).spec.kernel_name_override
                for b in (1, 2, 4)
            }
            self.assertEqual(len(names), 3, names)

    def test_support_implies_the_dispatched_spec_builds(self):
        """The dispatch-level half of the supports/build contract."""
        with _Gfx942Arch():
            req = _req()
            self.assertTrue(_candidate().supports(req)[0])
            spec = _dense_spec(dataclasses.replace(req, dense_persistent="off"))
            kd = build_attention_dense(spec, arch="gfx942")
            self.assertEqual(kd.name, dispatch_attention(req).spec.kernel_name_override)


if __name__ == "__main__":
    unittest.main()
