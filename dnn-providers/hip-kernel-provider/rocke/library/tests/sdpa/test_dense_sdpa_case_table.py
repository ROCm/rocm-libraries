# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense SDPA binding over the shared case table (host-only).

For every ``cases.py`` case on each CDNA arch: an admitted request reports
only features the binding carries, each named dense candidate's own
``admits()`` accepts the pinned translated request, and the spec it builds
keeps exactly the keys the request asks for; a declined request says why.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.dispatch.core import opt_in_probe

from dispatch.attention import ATTENTION_ROUTE_REGISTRY
from dispatch.sdpa.bind_dense import (
    SDPA_DENSE_FEATURES,
    sdpa_dense_admits,
    sdpa_dense_candidates,
)
from dispatch.sdpa.bind_shared import kept_keys, to_attention_request
from tests.sdpa import cases as C
from tests.sdpa.request_bridge import request_from_case

ARCHES = ("gfx942", "gfx950")
_CANDIDATES = {c.name: c for c in ATTENTION_ROUTE_REGISTRY.candidates()}
_PAIRS = [
    (case, arch) for case in C.all_cases() for arch in ARCHES if arch in case.archs
]


def _spec_keeps(spec, req):
    """Independent of the gate: the spec's mask (``q - k < sliding_window``,
    ``k <= d + right_bound`` when causal) row by row, against the request's."""
    window = int(spec.sliding_window)
    right = int(getattr(spec, "right_bound", 0)) if spec.causal else -1
    top_left = not getattr(spec, "causal_bottom_right", False)
    got = kept_keys(req.s_q, req.s_kv, top_left, window - 1 if window else -1, right)
    want = kept_keys(
        req.s_q, req.s_kv, req.diagonal == "top_left", req.left_bound, req.right_bound
    )
    return all(np.array_equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("case,arch", _PAIRS, ids=[f"{c.id}-{a}" for c, a in _PAIRS])
def test_admitted_features_are_carried_and_declines_explained(case, arch):
    req = request_from_case(case, arch)
    ok, reason = sdpa_dense_admits(req)
    if not ok:
        assert reason
        return
    assert req.features() <= SDPA_DENSE_FEATURES
    attn = to_attention_request(req)
    for name in sdpa_dense_candidates(req):
        candidate = _CANDIDATES[name]
        probe = opt_in_probe(attn, candidate)
        assert candidate.admits(probe)[0], name
        assert _spec_keeps(candidate.select_spec(probe).kernel_spec, req), name


@pytest.mark.parametrize("arch,floor", [("gfx942", 15), ("gfx950", 150)])
def test_the_table_admits_cases_on_both_arches(arch, floor):
    """Guards against a gate that passes the test above by declining everything."""
    admitted = [
        case.id
        for case, a in _PAIRS
        if a == arch and sdpa_dense_admits(request_from_case(case, arch))[0]
    ]
    assert len(admitted) >= floor, (arch, len(admitted))
