# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Open/closed invariant for the attention candidate registry.

The ticket's primary acceptance criterion is: *new kernel specs can be added
without modifying existing implementations*. This test encodes that as an
executable invariant -- registering a brand-new candidate must NOT change the
``supports()`` verdict or the ``select_spec()`` result of any pre-existing
candidate.

It is CPU-only and does NOT mutate the shipped ``ATTENTION_REGISTRY`` singleton:
a fresh ``CandidateRegistry`` is seeded from ``attention_candidates()`` and the
example candidate is registered only into that copy. The example candidate is a
throwaway defined in this module (never shipped) -- it exists solely to prove the
registration mechanics, following the ``_make_d256_decode_candidate`` factory
shape in ``dispatch/attention/generic.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict
from typing import Tuple

import kernels.common.attention_unified as au
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    ShapeRange,
)
from dispatch.attention import (
    _FAMILY,
    ATTENTION_ABI_VERSION,
    ATTENTION_DIM_VOCABULARY,
    AttentionRequest,
    AttentionSpec,
    _problem,
    _request_errors,
    attention_candidates,
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


def _gfx950_d256_decode(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=16,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=8192,
        hdim_q=256,
        hdim_v=256,
        arch="gfx950",
        dtype="bf16",
    )
    base.update(kw)
    return AttentionRequest(**base)


_SAMPLE_REQUESTS = (_gfx942_fp16_mha(), _gfx950_d256_decode())


class _PinnedArch:
    def __init__(self, arch: str):
        self._arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self._arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _make_example_candidate(*, priority: int = 7) -> KernelCandidate:
    """A throwaway specialized candidate: gfx942 fp16 2D only.

    Deliberately narrow so it overlaps the shipped ``gfx942_dense_pipe`` /
    ``unified_2d`` cohort -- the overlap is what makes the non-interference
    assertion meaningful.
    """
    spec_id = "example_probe"
    name = "attention_example_probe"

    def support(req) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if req.arch != "gfx942":
            return False, f"example requires gfx942 (got {req.arch!r})"
        if req.dtype != "fp16":
            return False, f"example is fp16-only (got {req.dtype!r})"
        problem = _problem(req)
        if problem.select_path() != "2d":
            return False, "problem routes to 3D, not 2D"
        return True, "ok"

    def select(req) -> AttentionSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_example_probe",
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="example_probe",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16",),
            shapes=(ShapeRange("hdim_q", min=1),),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


def _fresh_registry_with(extra: KernelCandidate | None = None) -> CandidateRegistry:
    reg = CandidateRegistry(_FAMILY, dim_vocabulary=ATTENTION_DIM_VOCABULARY)
    reg.extend(attention_candidates())
    if extra is not None:
        reg.register(extra)
    return reg


def _supports_snapshot(reg: CandidateRegistry) -> dict:
    """{(request_index, candidate_name): (spec_id, priority, admits-verdict)}.

    Captured as plain data over ``reg.supported()`` -- so it records the
    registry's actual membership/verdicts at a point in time, not references to
    live candidate objects. Comparing two such snapshots (before vs after adding
    a candidate) is a real diff; comparing the shared objects to themselves is
    not.
    """
    return {
        (i, c.name): (c.spec_id, c.priority, c.admits(req))
        for i, req in enumerate(_SAMPLE_REQUESTS)
        for c in reg.supported(req)
    }


def _select_spec_snapshot(reg: CandidateRegistry) -> dict:
    """{(request_index, candidate_name): asdict(select_spec)} over supported().

    Serialized to plain dicts so the before/after comparison is against recorded
    state rather than the same live objects."""
    return {
        (i, c.name): asdict(c.select_spec(req))
        for i, req in enumerate(_SAMPLE_REQUESTS)
        for c in reg.supported(req)
    }


class TestAdditiveRegistration(unittest.TestCase):
    def test_example_registers_without_touching_singleton(self):
        before = {c.name for c in attention_candidates()}
        _fresh_registry_with(_make_example_candidate())
        after = {c.name for c in attention_candidates()}
        # The shipped singleton is untouched by seeding a copy.
        self.assertEqual(before, after)
        self.assertNotIn("attention_example_probe", after)

    def test_new_candidate_is_discoverable_in_copy(self):
        reg = _fresh_registry_with(_make_example_candidate())
        self.assertIn("attention_example_probe", {c.name for c in reg.candidates()})

    def test_priority_orders_new_candidate_correctly(self):
        # priority 7 sits between the specialists (5) and the generics (10).
        reg = _fresh_registry_with(_make_example_candidate(priority=7))
        ordered = [c.name for c in reg.candidates()]
        i_example = ordered.index("attention_example_probe")
        i_generic = ordered.index("attention_unified_2d")
        self.assertLess(i_example, i_generic)

    def test_existing_supports_verdicts_unchanged(self):
        # The open/closed invariant: adding the example changes NO pre-existing
        # candidate's admits() verdict for any sample request. Snapshot the
        # baseline registry's state as plain data BEFORE the example exists, then
        # diff against the state after -- so the comparison is before-vs-after,
        # not an object compared to itself.
        with _PinnedArch("gfx942"):
            reg = _fresh_registry_with(None)
            before = _supports_snapshot(reg)
            reg.register(_make_example_candidate())
            after = _supports_snapshot(reg)
        # Guard the premise: the snapshots must actually cover shipped candidates
        # (an empty baseline would make the invariant vacuous).
        self.assertTrue(before, "no shipped candidate supported any sample request")
        # Every pre-existing entry is unchanged; the only new key is the example.
        self.assertEqual(before, {k: v for k, v in after.items() if k in before})
        new_keys = set(after) - set(before)
        self.assertTrue(
            all(name == "attention_example_probe" for _, name in new_keys),
            f"adding the example perturbed candidates other than itself: {new_keys}",
        )

    def test_existing_select_specs_unchanged(self):
        # And it changes NO pre-existing candidate's select_spec() output.
        with _PinnedArch("gfx942"):
            reg = _fresh_registry_with(None)
            before = _select_spec_snapshot(reg)
            reg.register(_make_example_candidate())
            after = _select_spec_snapshot(reg)
        self.assertTrue(before, "no shipped candidate supported any sample request")
        self.assertEqual(before, {k: v for k, v in after.items() if k in before})

    def test_selection_winner_unchanged_by_lower_priority_add(self):
        # Open/closed at the selection level: the example (priority 7) overlaps
        # the shipped cohort but never outranks the specialist (priority 5), so
        # the winning candidate for each sample request must be unchanged. This
        # is the assertion that actually fails if a newly registered spec
        # perturbs routing for an existing problem.
        with _PinnedArch("gfx942"):
            reg = _fresh_registry_with(None)
            before = {
                i: reg.select(req).name
                for i, req in enumerate(_SAMPLE_REQUESTS)
                if reg.supported(req)
            }
            reg.register(_make_example_candidate())
            after = {i: reg.select(_SAMPLE_REQUESTS[i]).name for i in before}
        self.assertTrue(before, "no sample request had a supported candidate")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
