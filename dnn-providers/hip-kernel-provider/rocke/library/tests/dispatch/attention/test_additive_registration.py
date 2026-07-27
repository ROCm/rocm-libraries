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
shape in ``dispatch/attention.py``.
"""

from __future__ import annotations

import unittest
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


def _support_verdicts(reg: CandidateRegistry) -> dict:
    """{(candidate_name, request_index): (ok, why)} over the sample requests."""
    out = {}
    for i, req in enumerate(_SAMPLE_REQUESTS):
        for c in reg.candidates():
            out[(c.name, i)] = c.admits(req)
    return out


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
        # candidate's supports() verdict for any sample request.
        with _PinnedArch("gfx942"):
            baseline = _support_verdicts(_fresh_registry_with(None))
            with_example = _support_verdicts(
                _fresh_registry_with(_make_example_candidate())
            )
        for key, verdict in baseline.items():
            self.assertEqual(
                verdict,
                with_example[key],
                msg=f"supports() changed for {key} after adding example candidate",
            )

    def test_existing_select_specs_unchanged(self):
        # And it changes NO pre-existing candidate's select_spec() output.
        with _PinnedArch("gfx942"):
            base = _fresh_registry_with(None)
            withx = _fresh_registry_with(_make_example_candidate())
            for i, req in enumerate(_SAMPLE_REQUESTS):
                for c in base.candidates():
                    ok, _ = c.admits(req)
                    if not ok:
                        continue
                    other = next(
                        x for x in withx.candidates() if x.name == c.name
                    )
                    self.assertEqual(
                        c.select_spec(req),
                        other.select_spec(req),
                        msg=f"select_spec changed for {c.name} on request #{i}",
                    )


if __name__ == "__main__":
    unittest.main()
