# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only differential table: this PR's routed policy against the policy
of the checkout it is built on (develop, where the RDNA general tables of
:mod:`.gfx1151_tuning` / :mod:`.gfx1201_tuning` are empty -- Phase A
registered the modules in :data:`routing.ARCH_TUNING` with no entries yet),
swept over the dispatcher's own exhaustive request domain
(:mod:`tests.sdpa.bwd_cases`, the same case table :mod:`test_registry` uses).

:func:`routing.plan_attention_bwd` and :func:`attn_bwd_support` are pure
functions of ``(request, arch, policy)``, so running both policies in the
same process (:mod:`routing.arch_tuned_policy` reads each arch module's
``GENERAL_TUNED_TABLE`` fresh on every call) is a faithful develop-vs-PR
comparison without a second checkout.
"""

from __future__ import annotations

import pytest

from dispatch.attention_bwd import gfx1151_tuning, gfx1201_tuning, routing
from dispatch.attention_bwd import tuned_tables as tt
from kernels.common.attention_bwd_plan import attn_bwd_support, attn_bwd_tuning_key

from ...sdpa.bwd_cases import select_bwd
from ...sdpa.bwd_kernel_cases import (
    HEAD_PACK_CASES,
    SUPPLEMENTARY_CASES,
    ZERO_LENGTH_CASES,
    request_from_case,
)


def _requests():
    cases = list(select_bwd("exhaustive")) + list(SUPPLEMENTARY_CASES)
    cases += list(HEAD_PACK_CASES) + list(ZERO_LENGTH_CASES)
    seen, out = set(), []
    for c in cases:
        if c.id not in seen:
            seen.add(c.id)
            out.append((c.id, request_from_case(c)))
    return out


_REQUESTS = _requests()


def _develop_and_pr_policies():
    """``(develop, PR)`` routed policies: identical except the RDNA general
    tables, empty on develop (:func:`routing.arch_tuned_policy` reads the
    modules' ``GENERAL_TUNED_TABLE`` at call time, so monkeypatching it to
    ``{}`` for the call reproduces the pre-PR policy exactly)."""
    pr_policy = routing.routed_policy()
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(gfx1151_tuning, "GENERAL_TUNED_TABLE", {})
        mp.setattr(gfx1201_tuning, "GENERAL_TUNED_TABLE", {})
        develop_policy = routing.routed_policy()
    finally:
        mp.undo()
    return develop_policy, pr_policy


_DEVELOP_POLICY, _PR_POLICY = _develop_and_pr_policies()


def _decision(req, arch, policy):
    """A hashable-by-equality summary of the routed outcome: the decline
    reason code, or the served route and its full spec/runtime."""
    verdict = attn_bwd_support(req, arch, policy=policy)
    if not verdict.ok:
        return ("declined", verdict.code)
    plan = routing.plan_attention_bwd(req, arch, policy=policy)
    return ("served", plan.route, plan.specs, plan.runtime)


def test_cdna_routing_is_unaffected_by_the_rdna_tables():
    """gfx942 / gfx950 decisions are byte-identical between develop and this
    PR for every request of the exhaustive domain: :func:`arch_tuned_policy`
    joins disjoint per-arch dicts, so adding gfx1151 / gfx1201 keys cannot
    move a gfx942 / gfx950 decision."""
    checked = 0
    for arch in ("gfx942", "gfx950"):
        for _cid, req in _REQUESTS:
            if not attn_bwd_support(req, arch, policy=_PR_POLICY).ok:
                continue
            checked += 1
            assert _decision(req, arch, _DEVELOP_POLICY) == _decision(
                req, arch, _PR_POLICY
            ), (arch, req)
    assert checked  # the domain sweep reaches at least one served CDNA request


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
def test_rdna_routing_differs_only_at_shipped_tuned_keys(arch):
    """An RDNA decision changes between develop and this PR only for a
    request whose tuning key is one of this PR's shipped general entries;
    a request of an untuned key, or one the predicate declines either way,
    is unaffected (no unintended change in the unified request, the predicate,
    or any other key's knobs). The exhaustive domain alone rarely lands a
    request's fill bucket on the tuned ``over`` bucket, so it is swept
    alongside a representative request of every shipped key
    (:func:`tuned_tables.general_key_requests`, the same representatives the
    entry check itself plans)."""
    mod = routing.ARCH_TUNING[arch]
    assert mod.GENERAL_TUNED_TABLE  # otherwise this test is vacuous
    domain = list(_REQUESTS) + [
        (f"tuned-{key!r}", req)
        for key in mod.GENERAL_TUNED_TABLE
        for req in tt.general_key_requests(arch, key)
    ]
    changed_keys = set()
    checked = 0
    for _cid, req in domain:
        if not attn_bwd_support(req, arch, policy=_PR_POLICY).ok:
            continue
        checked += 1
        before = _decision(req, arch, _DEVELOP_POLICY)
        after = _decision(req, arch, _PR_POLICY)
        if before == after:
            continue
        key = attn_bwd_tuning_key(req, arch)
        assert key in mod.GENERAL_TUNED_TABLE, (req, key, before, after)
        changed_keys.add(key)
    assert checked
    assert changed_keys  # the sweep reaches at least one shipped tuned key
