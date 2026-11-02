# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dispatch.sdpa``'s family-level sweep entry points.

``registered_sdpa_combos`` / ``sdpa_sweep_space`` / ``dispatch_sdpa_all`` add
no candidate or knob of their own: they translate an ``SdpaRequest`` to the
``AttentionRequest`` the unified and dense bindings already use
(``to_attention_request``) and delegate entirely to the existing
``dispatch.attention`` registry.
"""

from __future__ import annotations

from dispatch.sdpa import (
    dispatch_sdpa_all,
    registered_sdpa_combos,
    sdpa_sweep_space,
)
from dispatch.sdpa.bind_dense import sdpa_dense_candidates
from dispatch.sdpa.bind_shared import to_attention_request
from dispatch.sdpa.bind_unified import sdpa_unified_candidates

from tests.sdpa.request_cases import make_request


def _request(arch="gfx942", d=128, dtype="fp16", **overrides):
    return make_request(arch=arch, d=d, dtype=dtype, **overrides)


def test_combos_are_nonempty_for_an_admitted_request():
    req = _request()
    combos = registered_sdpa_combos(req)
    assert combos
    assert all(c.family == "attention_unified" for c, _ in combos)


def test_combos_cover_every_unified_and_dense_candidate_name():
    req = _request()
    names = {c.name for c, _ in registered_sdpa_combos(req)}
    assert set(sdpa_unified_candidates(req)) <= names
    assert set(sdpa_dense_candidates(req)) <= names


def test_sweep_space_matches_the_underlying_attention_sweep():
    req = _request()
    from dispatch.attention import attention_sweep_space

    assert sdpa_sweep_space(req) == attention_sweep_space(to_attention_request(req))


def test_dispatch_all_matches_registered_combos():
    req = _request()
    results = dispatch_sdpa_all(req)
    combos = registered_sdpa_combos(req)
    assert len(results) == len(combos)
    assert [(r.candidate, r.spec) for r in results] == list(combos)


def test_all_three_are_empty_for_something_that_is_not_an_sdpa_request():
    assert registered_sdpa_combos("not a request") == ()
    assert sdpa_sweep_space("not a request") == ()
    assert dispatch_sdpa_all("not a request") == ()
