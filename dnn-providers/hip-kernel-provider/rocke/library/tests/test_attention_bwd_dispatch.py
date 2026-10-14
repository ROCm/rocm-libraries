# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Routing among the attention backward families (CPU, no device).

Every request of the backward case table (all tiers) and of the supplementary
kernel cases that the general predicate accepts, on gfx942, gfx950 and
gfx1151:

* routes to exactly one family and gets a plan of that family only (one
  main launch: ``"main"`` for the general family, ``"dense"`` for the dense
  family);
* routes dense exactly when it is dense-eligible and its dense key is in the
  tuned dense table; never dense for an ineligible request, even with its key
  in the table;
* with an empty tuned dense table routes general, so routing changes
  nothing until a dense key passes its gate; with the shipped table only the
  requests of a packed key route dense;
* a forced general route on a packed key plans the general family; a forced
  dense route on an ineligible request raises.

Every policy here is the shipped one (the routed policy's per-arch tables on
the shipped kernel-level policy); only the dense table is replaced where the
point is a table (every key packed, or none).
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention_bwd import (
    DENSE_BWD_TUNED,
    attention_bwd_route,
    dense_policy,
    plan_attention_bwd,
    routed_policy,
)
from dispatch.attention_bwd.routing import arch_tuned_policy
from kernels.common.attention_bwd_plan import (
    _dense_key,
    attn_bwd_support,
    dense_bwd_eligibility,
    shipped_policy,
)

from .sdpa.bwd_cases import select_bwd
from .sdpa.bwd_kernel_cases import (
    HEAD_PACK_CASES,
    SUPPLEMENTARY_CASES,
    ZERO_LENGTH_CASES,
    request_from_case,
)

ARCHS = ("gfx942", "gfx950", "gfx1151")


def _requests():
    cases = list(select_bwd("exhaustive")) + list(SUPPLEMENTARY_CASES)
    cases += list(HEAD_PACK_CASES) + list(ZERO_LENGTH_CASES)
    seen, out = set(), []
    for c in cases:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append((c.id, request_from_case(c)))
    return out


_REQUESTS = _requests()


def _shipped():
    """The shipped policy with the per-arch general tables (no dense table:
    each test adds the dense table it is about)."""
    return arch_tuned_policy(shipped_policy())


def _main_stages(plan):
    return [s.stage for s in plan.launches if s.stage in ("main", "dense")]


@pytest.mark.parametrize("arch", ARCHS)
def test_exactly_one_family_and_dense_only_when_eligible(arch):
    base = _shipped()
    # every key in the table, eligible or not: eligibility must still gate
    table = {(arch, _dense_key(req)): {} for _cid, req in _REQUESTS}
    policy = dense_policy(base, table)
    n_ok = n_dense = 0
    for cid, req in _REQUESTS:
        if not attn_bwd_support(req, arch, policy=policy).ok:
            with pytest.raises(ValueError):
                plan_attention_bwd(req, arch, policy=policy)
            continue
        n_ok += 1
        eligible, _tag = dense_bwd_eligibility(req, arch)
        route, _ = attention_bwd_route(req, arch, policy=policy)
        assert route == ("dense" if eligible else "general"), cid
        plan = plan_attention_bwd(req, arch, policy=policy)
        assert plan.route == route, cid
        assert _main_stages(plan) == ["dense" if eligible else "main"], cid
        assert plan.perf_class[:2] == (arch, route), cid
        n_dense += route == "dense"
    assert n_ok > 0
    if arch in ("gfx942", "gfx950"):
        assert n_dense > 0
    else:
        assert n_dense == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_empty_table_routes_general(arch):
    policy = dense_policy(_shipped(), {})
    shipped = routed_policy()
    for cid, req in _REQUESTS[::7]:
        if not attn_bwd_support(req, arch, policy=policy).ok:
            continue
        assert attention_bwd_route(req, arch, policy=policy)[0] == "general", cid
        assert _main_stages(plan_attention_bwd(req, arch, policy=policy)) == ["main"]
        packed = (arch, _dense_key(req)) in DENSE_BWD_TUNED
        want = "dense" if packed and dense_bwd_eligibility(req, arch)[0] else "general"
        assert attention_bwd_route(req, arch, policy=shipped)[0] == want, cid
        assert attention_bwd_route(req, arch)[0] == want, cid  # no policy given


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_unified_requests_have_their_own_entry(arch):
    """A rocKE unified (paged KV) request is planned by the unified family
    (route "unified"); a declined one raises with its reason code."""
    from dispatch.attention_bwd import plan_attention_bwd_unified
    from kernels.common.attention_bwd_unified_plan import AttnBwdUnifiedRequest

    req = AttnBwdUnifiedRequest(
        num_seqs=3, h_q=8, h_kv=2, d_qk=128, d_v=128, total_q=100,
        max_seqlen_q=60, max_seqlen_kv=300, page=16, num_blocks=50,
        max_blocks_per_seq=20,
    )  # fmt: skip
    plan = plan_attention_bwd_unified(req, arch)
    assert plan.route == "unified"
    assert [s.stage for s in plan.launches].count("main") == 1
    with pytest.raises(ValueError, match="SOFTCAP"):
        plan_attention_bwd_unified(dataclasses.replace(req, softcap=30.0), arch)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_forced_routes(arch):
    base = _shipped()
    eligible = [
        (cid, req) for cid, req in _REQUESTS if dense_bwd_eligibility(req, arch)[0]
    ]
    ineligible = [
        (cid, req)
        for cid, req in _REQUESTS
        if not dense_bwd_eligibility(req, arch)[0]
        and attn_bwd_support(req, arch, policy=base).ok
    ]
    assert eligible and ineligible
    cid, req = eligible[0]
    policy = dense_policy(base, {(arch, _dense_key(req)): {}})
    assert plan_attention_bwd(req, arch, policy=policy).route == "dense"
    forced = plan_attention_bwd(req, arch, policy=policy, route="general")
    assert forced.route == "general" and _main_stages(forced) == ["main"]
    assert forced.perf_class[:2] == (arch, "general")
    # a forced dense route on a key outside the table takes the dense defaults
    other = dense_policy(base, {})
    assert plan_attention_bwd(req, arch, policy=other).route == "general"
    forced = plan_attention_bwd(req, arch, policy=other, route="dense")
    assert forced.route == "dense" and _main_stages(forced) == ["dense"]
    knobs = {
        "epilogue", "oob", "edge_tiles", "ws_layout", "dkv_mode", "dq_mode",
        "persistent", "persist_decode", "xcd_remap", "body_knobs", "body_variant",
    }  # fmt: skip
    assert not knobs & {k for k, _ in forced.specs["dense"].non_default_fields()}
    cid, req = ineligible[0]
    with pytest.raises(ValueError, match="dense-eligible"):
        plan_attention_bwd(
            req, arch, policy=dense_policy(base, {(arch, _dense_key(req)): {}}),
            route="dense",
        )  # fmt: skip
    with pytest.raises(ValueError, match="route"):
        plan_attention_bwd(req, arch, policy=base, route="unified")
