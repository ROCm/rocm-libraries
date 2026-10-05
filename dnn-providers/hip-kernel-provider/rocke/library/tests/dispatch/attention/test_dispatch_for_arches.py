# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contract tests for :func:`dispatch.attention.dispatch_for_arches` (AICK-2146).

The helper is the user-facing payoff of making selection arch-driven: ask one
host what every arch would pick. It shipped with no tests, so this file pins the
three things a caller can actually depend on -- the input form, the canonical
keying, and what happens when one arch fails -- before anything starts calling
it.

CPU-only. ``num_cus`` is pinned on every request on purpose: with it unset the
CU count is host-derived (see ``_resolve_num_cus``), which would make these
assertions depend on the runner's GPU.
"""

from __future__ import annotations

import unittest

from dispatch.attention import (
    AttentionRequest,
    dispatch_for_arches,
)
from rocke.dispatch.core import DispatchResult

# An arch no attention candidate declares -- dispatch_attention raises ValueError
# for it. Used to exercise the failure path without inventing a fake registry.
_UNSUPPORTED_ARCH = "gfx1030"


def _request(arch: str = "gfx942", **kw) -> AttentionRequest:
    defaults = dict(
        batch=2,
        nhead_q=16,
        nhead_k=16,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="fp16",
        num_cus=304,  # pinned: keeps the result independent of the runner's box
    )
    defaults.update(kw)
    return AttentionRequest(**defaults)


class TestArchesInputForms(unittest.TestCase):
    """``arches`` takes a sequence; the comma-split is the CLI boundary only."""

    def test_sequence_of_arches(self):
        results = dispatch_for_arches(_request(), ["gfx942", "gfx950"])
        self.assertEqual(list(results), ["gfx942", "gfx950"])

    def test_tuple_is_accepted(self):
        results = dispatch_for_arches(_request(), ("gfx942", "gfx950"))
        self.assertEqual(list(results), ["gfx942", "gfx950"])

    def test_comma_string_matches_the_sequence_form(self):
        """The CLI form must be a pure spelling of the structured one."""
        from_str = dispatch_for_arches(_request(), "gfx942,gfx950")
        from_seq = dispatch_for_arches(_request(), ["gfx942", "gfx950"])
        self.assertEqual(list(from_str), list(from_seq))
        for arch in from_seq:
            self.assertEqual(
                from_str[arch].candidate.name, from_seq[arch].candidate.name
            )

    def test_single_arch_sequence(self):
        results = dispatch_for_arches(_request(), ["gfx950"])
        self.assertEqual(list(results), ["gfx950"])

    def test_a_sequence_entry_is_never_split_on_commas(self):
        """Splitting a Sequence[str] would cut a name in half; it must not."""
        with self.assertRaises(ValueError):
            dispatch_for_arches(_request(), ["gfx942,gfx950"])

    def test_empty_input_is_an_error_not_an_empty_dict(self):
        for empty in ([], (), ""):
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                dispatch_for_arches(_request(), empty)

    def test_invalid_entry_raises(self):
        for bad in (["gfx942", ""], ["gfx942", "   "], ["gfx942", None]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dispatch_for_arches(_request(), bad)


class TestCanonicalKeying(unittest.TestCase):
    """Keys are canonical arches, and canonical duplicates collapse to one."""

    def test_case_and_feature_suffix_normalize_to_one_key(self):
        results = dispatch_for_arches(
            _request(), ["GFX950", "gfx950:sramecc+", "gfx950"]
        )
        self.assertEqual(list(results), ["gfx950"])

    def test_dedup_dispatches_once_not_repeatedly(self):
        """Dedup must drop the work, not just the duplicate key."""
        calls = []
        original = dispatch_for_arches.__globals__["dispatch_attention"]

        def counting(req, **kw):
            calls.append(req.arch)
            return original(req, **kw)

        dispatch_for_arches.__globals__["dispatch_attention"] = counting
        try:
            dispatch_for_arches(_request(), ["GFX950", "gfx950:sramecc+", "gfx942"])
        finally:
            dispatch_for_arches.__globals__["dispatch_attention"] = original
        self.assertEqual(calls, ["gfx950", "gfx942"])

    def test_first_spelling_sets_the_position(self):
        results = dispatch_for_arches(_request(), ["gfx950:sramecc+", "gfx942"])
        self.assertEqual(list(results), ["gfx950", "gfx942"])

    def test_base_request_arch_does_not_leak_into_the_results(self):
        """Every result is dispatched for its own arch, not the request's."""
        results = dispatch_for_arches(_request(arch="gfx942"), ["gfx950"])
        self.assertIn("gfx950", results["gfx950"].explanation[0])
        self.assertNotIn("gfx942", results["gfx950"].explanation[0])


class TestPerArchFailureHandling(unittest.TestCase):
    """One unsupported arch must not be able to discard the others' results."""

    def test_strict_default_propagates(self):
        with self.assertRaises(ValueError):
            dispatch_for_arches(_request(), ["gfx942", _UNSUPPORTED_ARCH, "gfx950"])

    def test_non_strict_keeps_the_successful_arches(self):
        results = dispatch_for_arches(
            _request(),
            ["gfx942", _UNSUPPORTED_ARCH, "gfx950"],
            strict=False,
        )
        self.assertEqual(list(results), ["gfx942", _UNSUPPORTED_ARCH, "gfx950"])
        self.assertIsInstance(results["gfx942"], DispatchResult)
        self.assertIsInstance(results["gfx950"], DispatchResult)
        self.assertIsInstance(results[_UNSUPPORTED_ARCH], Exception)

    def test_non_strict_reports_the_real_exception(self):
        """The mapped value is the exception itself, not a flattened string."""
        results = dispatch_for_arches(_request(), [_UNSUPPORTED_ARCH], strict=False)
        failure = results[_UNSUPPORTED_ARCH]
        self.assertIsInstance(failure, ValueError)
        self.assertIn(_UNSUPPORTED_ARCH, str(failure))

    def test_non_strict_does_not_swallow_a_malformed_arches_argument(self):
        """A bad argument is the caller's bug, not a per-arch dispatch failure."""
        with self.assertRaises(ValueError):
            dispatch_for_arches(_request(), ["gfx942", ""], strict=False)


class TestCrossArchComparison(unittest.TestCase):
    """The use case the helper exists for."""

    def test_two_arches_can_select_differently_from_one_host(self):
        # fp16 hd128 prefill: gfx942 claims it with its dense-pipe candidate,
        # gfx950 falls through to the generic 2D path.
        results = dispatch_for_arches(_request(), ["gfx942", "gfx950"])
        self.assertNotEqual(
            results["gfx942"].candidate.name,
            results["gfx950"].candidate.name,
            "this shape is meant to route differently per arch; if the two now "
            "agree the comparison is no longer demonstrating anything -- repick "
            "the shape rather than deleting the assertion",
        )

    def test_result_is_identical_to_dispatching_each_arch_directly(self):
        from dispatch.attention import dispatch_attention

        req = _request()
        batched = dispatch_for_arches(req, ["gfx942", "gfx950"])
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                from dataclasses import replace

                direct = dispatch_attention(replace(req, arch=arch))
                self.assertEqual(batched[arch].candidate.name, direct.candidate.name)
                self.assertEqual(batched[arch].spec, direct.spec)


if __name__ == "__main__":
    unittest.main()
