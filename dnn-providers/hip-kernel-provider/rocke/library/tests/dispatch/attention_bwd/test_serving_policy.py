# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Serving under the shipped backward policy (CPU, no device).

The shipped policy serves every request the predicate accepts and the plan
builds, on every declared target; the per-class performance status
(``verified`` / ``unverified`` / ``below_baseline``) is information only:

* every case-table request the predicate accepts on gfx942 / gfx950 is served
  by the default policy and planned by the default routing;
* the status of a class is recorded in the plan and published in the
  declaration, and never changes the verdict, the route or the knobs;
* every other reason code still declines under the default policy;
* the opt-in performance gate (``AttnBwdPolicy.perf_gate``) still declines
  ``PERF_BELOW_BASELINE`` / ``PERF_UNVERIFIED``;
* tuned keys plan their tuned configuration, untuned requests the defaults.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention_bwd import routing, tuned_tables
from kernels.common.attention_bwd_plan import (
    ANY_BUCKET,
    FEATURE_GROUPS,
    PERF_STATUSES,
    REASON_CODES,
    AttnBwdPolicy,
    AttnBwdRequest,
    AttnBwdSpec,
    _dense_key,
    attn_bwd_declaration,
    attn_bwd_perf_class,
    attn_bwd_plan,
    attn_bwd_support,
    attn_bwd_tuning_key,
    shipped_policy,
)
from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

from ...sdpa.bwd_cases import select_bwd
from ...sdpa.bwd_kernel_cases import SUPPLEMENTARY_CASES, request_from_case

CDNA = ("gfx942", "gfx950")
GATE = AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True)


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=256, s_kv=256, d_qk=128, d_v=128)
    base.update(kw)
    return AttnBwdRequest(**base)


def _case_requests():
    seen, out = set(), []
    for c in list(select_bwd("exhaustive")) + list(SUPPLEMENTARY_CASES):
        if c.id not in seen:
            seen.add(c.id)
            out.append((c.id, request_from_case(c)))
    return out


def _same_plan(a, b):
    return (a.route, a.specs, a.runtime, a.launches) == (
        b.route,
        b.specs,
        b.runtime,
        b.launches,
    )


# ---------------------------------------------------------------------------
# serve everything legal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", CDNA)
def test_default_policy_serves_every_accepted_case(arch):
    """The shipped default's verdict is the functional one (every group
    declared, performance not consulted), and every accepted request plans
    under the default routing."""
    routed = routing.routed_policy()
    assert not shipped_policy().perf_gate and not routed.perf_gate
    n_ok = 0
    for cid, req in _case_requests():
        v = attn_bwd_support(req, arch)
        assert (v.ok, v.code) == (
            attn_bwd_support(req, arch, policy=GATE).ok,
            attn_bwd_support(req, arch, policy=GATE).code,
        ), cid
        assert v.code not in ("PERF_UNVERIFIED", "PERF_BELOW_BASELINE"), cid
        assert attn_bwd_support(req, arch, policy=routed).code == v.code, cid
        if not v.ok:
            continue
        n_ok += 1
        plan = routing.plan_attention_bwd(req, arch)
        assert plan.perf_class == attn_bwd_perf_class(req, arch, plan.route)
        assert plan.perf_status == routed.status_of(arch, plan.route, plan.perf_class)
    assert n_ok > 100


_REPRESENTATIVE = {
    "d128 bf16 mha none": _req(dtype="bf16"),
    "d64 fp16 causal": _req(dtype="fp16", d_qk=64, d_v=64, causal=True),
    "gqa bottom-right": _req(
        dtype="bf16", h_k=2, h_v=2, s_q=64, right_bound=0, bottom_right=True
    ),
    "h_k != h_v window": _req(dtype="fp16", h_k=4, h_v=2, left_bound=31, right_bound=0),
    "s_q 1": _req(dtype="bf16", s_q=1, right_bound=0, bottom_right=True),
    "padded": _req(dtype="bf16", padding=True, has_seq_len_q=True, has_seq_len_kv=True),
    "thd": _req(dtype="fp16", layout="thd"),
    "narrow": _req(dtype="bf16", tensor_alignment=4),
    "d32": _req(dtype="fp16", d_qk=32, d_v=32),
    "device hints": _req(dtype="bf16", causal=True, num_cus=256, num_xcds=8),
}


@pytest.mark.parametrize("arch", CDNA)
@pytest.mark.parametrize("name", sorted(_REPRESENTATIVE))
def test_representative_requests_are_served(arch, name):
    req = _REPRESENTATIVE[name]
    assert attn_bwd_support(req, arch).ok
    assert attn_bwd_support(req, arch, policy=routing.routed_policy()).ok
    plan = routing.plan_attention_bwd(req, arch)
    assert plan.perf_status in PERF_STATUSES


def test_rdna_is_unchanged():
    """RDNA targets were never declined on performance: still served, with
    the ``degraded`` status, under the default and the gated policy."""
    gated = dataclasses.replace(shipped_policy(), perf_gate=True)
    for arch in ("gfx1151", "gfx1201"):
        for policy in (None, gated):
            assert attn_bwd_support(_req(dtype="bf16"), arch, policy=policy).ok
        assert routing.plan_attention_bwd(_req(dtype="bf16"), arch).perf_status == (
            "degraded"
        )
        d = routing.attention_bwd_declaration(arch)
        assert {e["perf"] for e in d["perf"]} == {"degraded"}


# ---------------------------------------------------------------------------
# performance status: information only
# ---------------------------------------------------------------------------


def _class_request(arch, pclass):
    """A request of general class ``pclass`` the gate policy plans."""
    key = tuple(pclass) + (ANY_BUCKET, ANY_BUCKET)
    for req in tuned_tables.general_key_requests(arch, key):
        if attn_bwd_support(req, arch, policy=GATE).ok:
            return req
    raise AssertionError(f"no plannable request of {pclass}")


@pytest.mark.parametrize("arch", CDNA)
def test_status_tables_are_well_formed(arch):
    mod = routing.ARCH_TUNING[arch]
    for table, route in (
        (mod.GENERAL_PERF_STATUS, "general"),
        (mod.DENSE_PERF_STATUS, "dense"),
    ):
        for pclass, status in table.items():
            assert pclass[:2] == (arch, route)
            # unverified is the default: the tables list measured outcomes only
            assert status in ("verified", "below_baseline")
    # a dense status names a key of the shipped dense pack
    assert {(arch, k[2:]) for k in mod.DENSE_PERF_STATUS} <= set(mod.DENSE_TUNED)
    routed = routing.routed_policy()
    for pclass, status in {**mod.GENERAL_PERF_STATUS, **mod.DENSE_PERF_STATUS}.items():
        assert routed.perf_status[pclass] == status


@pytest.mark.parametrize("arch", CDNA)
def test_status_is_reported_per_class(arch):
    """Every status-table class is served under the default routing, and the
    plan records the class's status; the declaration publishes one status per
    general class and per dense-pack key, matching the tables."""
    mod = routing.ARCH_TUNING[arch]
    for pclass, status in mod.GENERAL_PERF_STATUS.items():
        req = _class_request(arch, pclass)
        assert attn_bwd_support(req, arch).ok
        plan = routing.plan_attention_bwd(req, arch, route="general")
        assert (plan.perf_class, plan.perf_status) == (pclass, status)
    for pclass, status in mod.DENSE_PERF_STATUS.items():
        req = tuned_tables.dense_key_requests(arch, pclass[2:])[0]
        if (arch, _dense_key(req)) != (arch, pclass[2:]):
            raise AssertionError("dense representative of another key")
        plan = routing.plan_attention_bwd(req, arch)
        assert (plan.route, plan.perf_class, plan.perf_status) == (
            "dense",
            pclass,
            status,
        )
    decl = routing.attention_bwd_declaration(arch)
    assert decl["perf_policy"] == {
        "gate": "information_only",
        "unverified": "served",
        "below_baseline": "served",
    }
    general = {
        tuple(e[k] for k in ("arch", "route", "d", "dtype", "mask", "heads"))
        + (e["length_relation"], e["length_mode"], e["stage_vec"]): e["perf"]
        for e in decl["perf"]
        if e.get("route") == "general"
    }
    dense = {
        (arch, "dense") + tuple(_tuple(e["key"])): e["perf"]
        for e in decl["perf"]
        if e.get("route") == "dense"
    }
    assert {c: s for c, s in general.items() if s != "unverified"} == dict(
        mod.GENERAL_PERF_STATUS
    )
    assert set(general.values()) <= set(PERF_STATUSES)
    for pclass, status in dense.items():
        assert status == mod.DENSE_PERF_STATUS.get(pclass, "unverified")
    # the kernels-level declaration (no dispatch tables) claims nothing
    assert {e["perf"] for e in attn_bwd_declaration(arch)["perf"]} == {"unverified"}


def _tuple(x):
    return tuple(_tuple(v) for v in x) if isinstance(x, list) else x


@pytest.mark.parametrize("arch", CDNA)
@pytest.mark.parametrize(
    "req",
    [_req(dtype="bf16"), _req(dtype="fp16", h_k=2, h_v=2, causal=True, num_cus=304)],
    ids=["mha", "gqa"],
)
def test_status_never_changes_serving_or_the_plan(arch, req):
    base = routing.routed_policy()
    route = routing.attention_bwd_route(req, arch, policy=base)[0]
    pclass = attn_bwd_perf_class(req, arch, route)
    plans = {}
    for status in PERF_STATUSES:
        pol = dataclasses.replace(
            base, perf_status={**base.perf_status, pclass: status}
        )
        assert attn_bwd_support(req, arch, policy=pol).ok
        assert routing.attention_bwd_route(req, arch, policy=pol)[0] == route
        plans[status] = routing.plan_attention_bwd(req, arch, policy=pol)
        assert plans[status].perf_status == status
    ref = plans["verified"]
    assert all(_same_plan(p, ref) for p in plans.values())


# ---------------------------------------------------------------------------
# the remaining declines
# ---------------------------------------------------------------------------

_DECLINES = {
    "DROPOUT": _req(dropout=0.1),
    "BIAS_IN_BACKWARD": _req(has_bias=True),
    "DETERMINISTIC": _req(deterministic=True),
    "DTYPE": _req(dtype="fp32"),
    "HEAD_DIM_256_BWD": _req(d_qk=256, d_v=256),
    "HEAD_DIM": _req(d_qk=96, d_v=96),
    "DQK_NE_DV": _req(d_v=64),
    "GQA_RATIO": _req(h_k=3, h_v=3),
    "EMPTY": _req(b=0),
}


@pytest.mark.parametrize("arch", CDNA)
@pytest.mark.parametrize("code", sorted(_DECLINES))
def test_other_reason_codes_still_decline(arch, code):
    req = _DECLINES[code]
    for policy in (None, routing.routed_policy()):
        v = attn_bwd_support(req, arch, policy=policy)
        assert (v.ok, v.code) == (False, code), v
    with pytest.raises(ValueError, match=code):
        routing.plan_attention_bwd(req, arch)


@pytest.mark.parametrize("arch", CDNA)
def test_resource_and_arch_refusals_still_decline(arch):
    req = _req()  # fp16: the instance named below
    name = AttnBwdSpec(head_size=128, mask_class="none").kernel_name("main")
    pol = dataclasses.replace(
        routing.routed_policy(), instance_unavailable=frozenset({name})
    )
    assert attn_bwd_support(req, arch, policy=pol).code == "INSTANCE_UNAVAILABLE"
    pol = dataclasses.replace(
        routing.routed_policy(), not_validated_archs=frozenset({arch})
    )
    assert attn_bwd_support(req, arch, policy=pol).code == "ARCH_NOT_VALIDATED"
    assert attn_bwd_support(req, "gfx90a").code == "ARCH_UNSUPPORTED"
    pol = dataclasses.replace(
        routing.routed_policy(), declared_groups=frozenset({"base"})
    )
    v = attn_bwd_support(_req(dtype="bf16", causal=True), arch, policy=pol)
    assert v.code == "NOT_YET_DECLARED"


@pytest.mark.parametrize("arch", CDNA)
def test_opt_in_perf_gate_still_declines(arch):
    req = _req(dtype="bf16")
    pclass = attn_bwd_perf_class(req, arch)
    gated = dataclasses.replace(shipped_policy(), perf_gate=True)
    assert attn_bwd_support(req, arch, policy=gated).code == "PERF_UNVERIFIED"
    waived = dataclasses.replace(gated, waive_unverified=True)
    assert attn_bwd_support(req, arch, policy=waived).ok
    below = dataclasses.replace(waived, perf_status={pclass: "below_baseline"})
    assert attn_bwd_support(req, arch, policy=below).code == "PERF_BELOW_BASELINE"
    ok = dataclasses.replace(gated, perf_status={pclass: "verified"})
    assert attn_bwd_support(req, arch, policy=ok).ok
    d = attn_bwd_declaration(arch, policy=gated)
    assert d["perf_policy"] == {
        "gate": "declines",
        "unverified": "declined",
        "below_baseline": "declined",
    }
    assert {"PERF_UNVERIFIED", "PERF_BELOW_BASELINE"} <= set(REASON_CODES)


# ---------------------------------------------------------------------------
# tuned keys plan their tuned configuration; untuned requests the defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", CDNA)
def test_tuned_keys_route_to_their_tuned_configs(arch):
    mod = routing.ARCH_TUNING[arch]
    default_routing = routing.routed_policy()
    n = 0
    for key, knobs in mod.GENERAL_TUNED_TABLE.items():
        tuned = dataclasses.replace(GATE, tuned_table={key: dict(knobs)})
        for req in tuned_tables.general_key_requests(arch, key):
            if attn_bwd_tuning_key(req, arch) != key:
                continue  # a representative of an ANY_BUCKET key
            if not attn_bwd_support(req, arch, policy=GATE).ok:
                continue
            if routing.attention_bwd_route(req, arch, policy=default_routing)[0] != (
                "general"
            ):
                continue
            want = attn_bwd_plan(req, arch, policy=tuned)
            got = routing.plan_attention_bwd(req, arch)
            assert _same_plan(got, want), (key, req)
            n += 1
            break
    assert n == len(mod.GENERAL_TUNED_TABLE)
    for (parch, dkey), knobs in mod.DENSE_TUNED.items():
        req = tuned_tables.dense_key_requests(arch, dkey)[0]
        want = attn_dense_bwd_plan(
            req,
            arch,
            policy=routing.dense_policy(GATE, {(parch, dkey): dict(knobs)}),
        )
        got = routing.plan_attention_bwd(req, arch)
        assert got.route == "dense" and _same_plan(got, want), dkey


@pytest.mark.parametrize("arch", CDNA)
def test_untuned_requests_plan_the_defaults(arch):
    """A request without ``num_cus`` (fill bucket ``unknown``) matches no
    tuned key and plans the defaults; so does a class with no entry."""
    mod = routing.ARCH_TUNING[arch]
    reqs = [_req(dtype="bf16"), _req(dtype="fp16", d_qk=64, d_v=64, causal=True)]
    for key in mod.GENERAL_TUNED_TABLE:
        for req in tuned_tables.general_key_requests(arch, key):
            if req.num_cus is None and attn_bwd_support(req, arch, policy=GATE).ok:
                reqs.append(req)
                break
    for req in reqs:
        assert attn_bwd_tuning_key(req, arch) not in mod.GENERAL_TUNED_TABLE
        if (arch, _dense_key(req)) in mod.DENSE_TUNED:
            continue
        got = routing.plan_attention_bwd(req, arch)
        assert _same_plan(got, attn_bwd_plan(req, arch, policy=GATE)), req
