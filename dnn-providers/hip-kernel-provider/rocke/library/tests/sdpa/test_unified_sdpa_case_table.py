# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unified SDPA binding over the shared case table (host-only).

For every ``cases.py`` case on each CDNA arch it claims: an admitted request
reports only features the binding carries, and a unified route candidate's own
``admits()`` accepts the translated request; a declined request says why.
"""

from __future__ import annotations

import pytest

from dispatch.attention import ATTENTION_ROUTE_REGISTRY
from dispatch.sdpa.bind_shared import to_attention_request
from dispatch.sdpa.bind_unified import (
    SDPA_UNIFIED_FEATURES,
    sdpa_unified_admits,
    sdpa_unified_candidates,
)
from tests.sdpa import cases as C
from tests.sdpa.request_bridge import request_from_case

ARCHES = ("gfx942", "gfx950")
_CANDIDATES = {c.name: c for c in ATTENTION_ROUTE_REGISTRY.candidates()}
_PAIRS = [
    (case, arch) for case in C.all_cases() for arch in ARCHES if arch in case.archs
]


@pytest.mark.parametrize("case,arch", _PAIRS, ids=[f"{c.id}-{a}" for c, a in _PAIRS])
def test_admitted_features_are_carried_and_declines_explained(case, arch):
    req = request_from_case(case, arch)
    ok, reason = sdpa_unified_admits(req)
    if not ok:
        assert reason
        return
    assert req.features() <= SDPA_UNIFIED_FEATURES
    names = sdpa_unified_candidates(req)
    attn = to_attention_request(req)
    assert any(_CANDIDATES[n].admits(attn)[0] for n in names)


def test_the_table_admits_cases_on_both_arches():
    """Guards against a gate that passes the test above by declining everything."""
    for arch in ARCHES:
        admitted = [
            case.id
            for case, a in _PAIRS
            if a == arch and sdpa_unified_admits(request_from_case(case, arch))[0]
        ]
        assert len(admitted) >= 50, (arch, len(admitted))
