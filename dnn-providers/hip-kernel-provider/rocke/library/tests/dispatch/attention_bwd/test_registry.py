# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The attention backward dispatcher: candidates, ranker, entry points.

CPU only (plans, no device; the launch path is covered with the runners
replaced, and on the device by ``test_attention_bwd_dispatch_numeric.py``):

* one candidate per (family, arch), each with a declared capability that is
  a superset of its family's predicate;
* every case-table request: a candidate admits it exactly when its family's
  predicate accepts it under the shipped routed policy (no test policy), and
  :func:`dispatch_attention_bwd` plans it exactly as the routing does, on
  every declared target;
* the ranker puts the tuned tables first, then the defaults: the dense family
  exactly for a tuned dense key, the general family otherwise; pins select a
  family; a ranker cannot introduce an unsupported candidate;
* a declined request raises with the predicate's reason code first;
* the identity, signature and launch entry points.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from dispatch.attention_bwd import registry as reg
from dispatch.attention_bwd import routing, tuned_tables
from dispatch.attention_bwd import tuning_common as tc
from dispatch.attention_bwd.unified_tuning import UNIFIED_TUNED_PAGE_TILES
from kernels.common.attention_bwd_plan import (
    DECLARED_ARCHS,
    AttnBwdRequest,
    _dense_key,
    attn_bwd_support,
    dense_bwd_eligibility,
)
from kernels.common.attention_bwd_unified_plan import (
    VALIDATED_ARCHS,
    AttnBwdUnifiedRequest,
    attn_bwd_unified_support,
)
from kernels.common.attention_dense_bwd_spec import DENSE_BWD_ARCHS
from rocke.dispatch.core import CandidateRegistry

from ...sdpa.bwd_cases import select_bwd
from ...sdpa.bwd_kernel_cases import (
    HEAD_PACK_CASES,
    SUPPLEMENTARY_CASES,
    ZERO_LENGTH_CASES,
    request_from_case,
)

R = reg.AttentionBwdRequest


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


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=256, s_kv=256, d_qk=128, d_v=128)
    base.update(dtype="bf16", causal=True)
    base.update(kw)
    return AttnBwdRequest(**base)


def _unified(**kw):
    base = dict(
        num_seqs=3, h_q=8, h_kv=2, d_qk=128, d_v=128, total_q=100,
        max_seqlen_q=60, max_seqlen_kv=300, page=16, num_blocks=50,
        max_blocks_per_seq=20,
    )  # fmt: skip
    base.update(kw)
    return AttnBwdUnifiedRequest(**base)


def _same_plan(a, b):
    return (a.route, a.specs, a.runtime, a.launches, a.workspace_bytes) == (
        b.route,
        b.specs,
        b.runtime,
        b.launches,
        b.workspace_bytes,
    )


def _dense_requests(arch):
    """A representative request of every shipped dense key of ``arch``."""
    mod = routing.ARCH_TUNING.get(arch)
    if mod is None:
        return []
    return [tuned_tables.dense_key_requests(arch, k[1])[0] for k in mod.DENSE_TUNED]


# ------------------------------------------------------------------ registration


def test_one_candidate_per_family_and_arch():
    want = {f"attention_bwd_general_{a}" for a in DECLARED_ARCHS}
    want |= {f"attention_bwd_dense_{a}" for a in DENSE_BWD_ARCHS}
    want |= {f"attention_bwd_unified_{a}" for a in VALIDATED_ARCHS}
    cands = reg.attention_bwd_candidates()
    assert {c.name for c in cands} == want
    for c in cands:
        algorithm, arch = c.name.split("_")[2], c.name.split("_")[3]
        assert c.family == reg.FAMILY and c.algorithm == algorithm
        assert c.capability.arches == (arch,)
        assert c.spec_id == f"attn_bwd_{algorithm}"
        assert c.abi_version.startswith("rocke.attn_")
        assert not c.opt_in
    # without a ranker, the general family precedes the dense one
    assert reg._PRIORITY["general"] < reg._PRIORITY["dense"]
    manifest = reg.ATTENTION_BWD_REGISTRY.coverage()
    json.dumps(manifest)
    assert len(manifest["candidates"]) == len(want)
    for arch in DECLARED_ARCHS:
        assert reg.ATTENTION_BWD_REGISTRY.for_arch(arch)


def test_a_fresh_registry_gets_the_same_candidates():
    fresh = CandidateRegistry(
        reg.FAMILY, dim_vocabulary=reg.ATTENTION_BWD_DIM_VOCABULARY
    )
    reg.register(fresh)
    assert [c.name for c in fresh.candidates()] == [
        c.name for c in reg.attention_bwd_candidates()
    ]
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(fresh)


def test_request_type_and_identity():
    r = R(_req(), "gfx950")
    assert r.features() == frozenset() and not r.paged
    assert set(r.dims()) <= reg.ATTENTION_BWD_DIM_VOCABULARY
    u = R(_unified(), "gfx950")
    assert u.features() == {reg.PAGED_KV} and u.paged
    assert set(u.dims()) <= reg.ATTENTION_BWD_DIM_VOCABULARY
    for x in (r, u):
        assert json.loads(json.dumps(x.normalized()))["arch"] == "gfx950"
    with pytest.raises(TypeError):
        R({"b": 1}, "gfx950")
    with pytest.raises(TypeError):
        R(_req(), None)
    # the request hash follows the problem
    a = reg.dispatch_attention_bwd(r).kernel_id
    b = reg.dispatch_attention_bwd(R(_req(s_q=512, s_kv=512), "gfx950")).kernel_id
    assert a.request_hash != b.request_hash
    assert a == reg.dispatch_attention_bwd(R(_req(), "gfx950")).kernel_id


# ------------------------------------------------------------------ admits = predicate


@pytest.mark.parametrize("arch", DECLARED_ARCHS)
def test_candidates_admit_exactly_what_their_family_serves(arch):
    """Shipped routed policy, no test policy: the general candidate admits a
    request iff the predicate accepts it; the dense one iff, in addition, it
    is dense-eligible; the capability never refuses an accepted request."""
    routed = routing.routed_policy()
    cands = {c.algorithm: c for c in reg.ATTENTION_BWD_REGISTRY.for_arch(arch)}
    n_ok = 0
    for cid, prob in _REQUESTS:
        req = R(prob, arch)
        v = attn_bwd_support(prob, arch, policy=routed)
        ok, why = cands["general"].admits(req)
        assert ok == v.ok, (cid, why)
        if v.ok:
            n_ok += 1
            assert cands["general"].capability.check(req)[0], cid
        if "dense" in cands:
            eligible = dense_bwd_eligibility(prob, arch)[0]
            ok, why = cands["dense"].admits(req)
            assert ok == (v.ok and eligible), cid
            if v.ok and not eligible:
                assert "dense-eligible" in why, (cid, why)
        if "unified" in cands:
            assert not cands["unified"].admits(req)[0], cid
    assert n_ok > 100


@pytest.mark.parametrize("arch", DECLARED_ARCHS)
def test_dispatch_plans_as_the_routing_does(arch):
    """Every accepted case-table request (and every shipped dense key):
    the dispatcher's selection is the routing's family and its plan is the
    routing's plan, under the shipped policy."""
    n = n_dense = 0
    reqs = [p for _cid, p in _REQUESTS] + _dense_requests(arch)
    for prob in reqs:
        if not attn_bwd_support(prob, arch).ok:
            with pytest.raises(ValueError):
                reg.dispatch_attention_bwd(R(prob, arch))
            continue
        res = reg.dispatch_attention_bwd(R(prob, arch))
        want = routing.plan_attention_bwd(prob, arch)
        assert res.candidate.algorithm == want.route
        assert res.candidate.capability.arches == (arch,)
        assert _same_plan(res.spec, want)
        assert res.spec.perf_status in ("unverified", "degraded")
        n += 1
        n_dense += want.route == "dense"
    assert n > 100
    if arch in ("gfx942", "gfx950"):
        assert n_dense == len(routing.ARCH_TUNING[arch].DENSE_TUNED)
    else:
        assert n_dense == 0


@pytest.mark.parametrize("arch", sorted(VALIDATED_ARCHS))
def test_unified_requests_dispatch_to_the_unified_family(arch):
    for prob in (
        _unified(),
        _unified(page=64, d_qk=64, d_v=64),
        _unified(block_table_injective=True, h_kv=8),
        _unified(dtype="fp16", right_bound=-1, bottom_right=False),
    ):
        assert attn_bwd_unified_support(prob, arch).ok
        res = reg.dispatch_attention_bwd(R(prob, arch))
        assert res.candidate.name == f"attention_bwd_unified_{arch}"
        assert _same_plan(res.spec, routing.plan_attention_bwd_unified(prob, arch))
    with pytest.raises(ValueError, match="SOFTCAP"):
        reg.dispatch_attention_bwd(R(_unified(softcap=30.0), arch))


def test_unified_on_an_unvalidated_target_declines():
    for arch in sorted(set(DECLARED_ARCHS) - set(VALIDATED_ARCHS)):
        with pytest.raises(ValueError, match="^ARCH_NOT_VALIDATED"):
            reg.dispatch_attention_bwd(R(_unified(), arch))


# ------------------------------------------------------------------ ranker


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_ranker_puts_the_tuned_tables_first(arch):
    """Dense-eligible requests: the dense family first exactly when the
    dense key is in the tuned dense pack, the general family otherwise (both
    admit); the general family's tuned general keys rank the general first;
    the registered order alone serves the general family."""
    pack = routing.routed_policy().dense_pack
    seen_tuned = seen_untuned = 0
    for _cid, prob in _REQUESTS + [(None, r) for r in _dense_requests(arch)]:
        req = R(prob, arch)
        sup = reg.ATTENTION_BWD_REGISTRY.supported(req)
        if not sup:
            continue
        ranked = reg.attention_bwd_ranker(req, sup)
        assert {c.name for c in ranked} == {c.name for c in sup}
        tuned = (arch, _dense_key(prob)) in pack
        eligible = dense_bwd_eligibility(prob, arch)[0]
        assert [c.algorithm for c in sup][0] == "general"
        if eligible:
            assert {c.algorithm for c in sup} == {"general", "dense"}
            first = ranked[0].algorithm
            assert first == ("dense" if tuned else "general")
            assert first == routing.attention_bwd_route(prob, arch)[0]
            seen_tuned += tuned
            seen_untuned += not tuned
        prio = reg.dispatch_attention_bwd(req, ranker=reg.priority_ranker)
        assert prio.candidate.algorithm == "general"
    assert seen_untuned
    assert seen_tuned == len(routing.ARCH_TUNING[arch].DENSE_TUNED)


def test_ranker_reads_the_tables_it_is_given(monkeypatch):
    """A dense key added to the routed dense table moves the dense family
    first; with an empty table the general family is first."""
    arch = "gfx950"
    prob = next(
        p
        for _cid, p in _REQUESTS
        if dense_bwd_eligibility(p, arch)[0] and attn_bwd_support(p, arch).ok
    )
    req = R(prob, arch)
    assert reg.dispatch_attention_bwd(req).candidate.algorithm == "general"
    monkeypatch.setattr(routing, "DENSE_BWD_TUNED", {(arch, _dense_key(prob)): {}})
    res = reg.dispatch_attention_bwd(req)
    assert res.candidate.algorithm == "dense" and res.spec.route == "dense"
    assert routing.plan_attention_bwd(prob, arch).route == "dense"
    monkeypatch.setattr(routing, "DENSE_BWD_TUNED", {})
    for p in _dense_requests(arch):
        assert reg.dispatch_attention_bwd(R(p, arch)).candidate.algorithm == "general"


def test_ranker_general_and_unified_tables():
    """A request of a tuned general key ranks the general family as tuned;
    a unified request of a tuned page tile ranks the unified family."""
    for arch, mod in sorted(routing.ARCH_TUNING.items()):
        key = next(iter(mod.GENERAL_TUNED_TABLE))
        prob = next(
            r
            for r in tuned_tables.general_key_requests(arch, key)
            if attn_bwd_support(r, arch).ok and r.num_cus is not None
        )
        cand = reg.ATTENTION_BWD_REGISTRY.get(f"attention_bwd_general_{arch}")
        assert reg._tuned_rank(R(prob, arch), cand) == 1
        bare = dataclasses.replace(prob, num_cus=None)  # fill bucket unknown
        assert reg._tuned_rank(R(bare, arch), cand) is None
    for (arch, d, page), _name in UNIFIED_TUNED_PAGE_TILES.items():
        cand = reg.ATTENTION_BWD_REGISTRY.get(f"attention_bwd_unified_{arch}")
        prob = _unified(d_qk=d, d_v=d, page=page, h_kv=8)
        assert reg._tuned_rank(R(prob, arch), cand) == 1
        assert reg._tuned_rank(R(_unified(page=16), arch), cand) is None


def test_ranker_cannot_introduce_a_candidate():
    req = R(_req(), "gfx950")
    other = reg.ATTENTION_BWD_REGISTRY.get("attention_bwd_general_gfx942")
    with pytest.raises(ValueError, match="unsupported candidate"):
        reg.dispatch_attention_bwd(req, ranker=lambda _r, _c: [other])
    with pytest.raises(ValueError, match="no candidates"):
        reg.dispatch_attention_bwd(req, ranker=lambda _r, _c: [])


# ------------------------------------------------------------------ pins, declines


def test_pins_select_a_family():
    arch = "gfx950"
    prob = next(
        p
        for _cid, p in _REQUESTS
        if dense_bwd_eligibility(p, arch)[0] and attn_bwd_support(p, arch).ok
    )
    res = reg.dispatch_attention_bwd(R(prob, arch, algorithm="dense"))
    assert res.candidate.algorithm == "dense"
    assert _same_plan(res.spec, routing.plan_attention_bwd(prob, arch, route="dense"))
    res = reg.dispatch_attention_bwd(R(prob, arch, spec_id="ATTN_BWD_GENERAL"))
    assert res.candidate.algorithm == "general"
    with pytest.raises(ValueError):
        reg.dispatch_attention_bwd(R(_req(), arch, algorithm="unified"))
    with pytest.raises(ValueError):
        reg.dispatch_attention_bwd(R(_req(), arch, algorithm="nope"))
    # a dense pin on an ineligible request declines
    ineligible = next(
        p
        for _cid, p in _REQUESTS
        if not dense_bwd_eligibility(p, arch)[0] and attn_bwd_support(p, arch).ok
    )
    assert reg.dispatch_attention_bwd(R(ineligible, arch)).candidate.algorithm == (
        "general"
    )
    with pytest.raises(ValueError):
        reg.dispatch_attention_bwd(R(ineligible, arch, algorithm="dense"))
    # every admitting family, ranked
    names = [
        r.candidate.algorithm for r in reg.dispatch_attention_bwd_all(R(prob, arch))
    ]
    assert names == ["general", "dense"]


@pytest.mark.parametrize("arch", DECLARED_ARCHS)
@pytest.mark.parametrize(
    "code,kw",
    [
        ("DROPOUT", dict(dropout=0.1)),
        ("BIAS_IN_BACKWARD", dict(has_bias=True)),
        ("DTYPE", dict(dtype="fp32")),
        ("HEAD_DIM_256_BWD", dict(d_qk=256, d_v=256)),
        ("HEAD_DIM", dict(d_qk=96, d_v=96)),
        ("EMPTY", dict(b=0)),
    ],
)
def test_declines_lead_with_the_reason_code(arch, code, kw):
    with pytest.raises(ValueError, match=f"^{code}: "):
        reg.dispatch_attention_bwd(R(_req(**kw), arch))
    assert reg.dispatch_attention_bwd_all(R(_req(**kw), arch)) == ()


def test_unsupported_arch_declines():
    with pytest.raises(ValueError, match="^ARCH_UNSUPPORTED"):
        reg.dispatch_attention_bwd(R(_req(), "gfx90a"))


def test_known_wrong_values_decline_the_candidate(monkeypatch):
    """A plan carrying a known-wrong value is not admitted (here a runtime
    value of the default plan, made known wrong for the test)."""
    req = R(_req(), "gfx950")
    plan = reg.dispatch_attention_bwd(req).spec
    assert reg.plan_known_wrong(plan) == ()
    monkeypatch.setattr(
        tc,
        "BWD_KNOWN_WRONG_KNOBS",
        {"gfx950": {"scale_placement": frozenset({plan.runtime["scale_placement"]})}},
    )
    assert reg.plan_known_wrong(plan) == ("scale_placement='fold_ds'",)
    with pytest.raises(ValueError, match="known wrong on gfx950"):
        reg.dispatch_attention_bwd(req)
    # another target is untouched
    assert reg.dispatch_attention_bwd(R(_req(), "gfx942")).candidate.algorithm
    # a main-spec field and a dQ-kernel field are found too
    monkeypatch.setattr(
        tc,
        "BWD_KNOWN_WRONG_KNOBS",
        {"gfx950": {"edge_tiles": frozenset({plan.specs["main"].edge_tiles})}},
    )
    assert reg.plan_known_wrong(plan)


# ------------------------------------------------------------------ result, launch


def test_result_fields_describe_the_plan():
    for prob, arch in ((_req(), "gfx942"), (_unified(), "gfx950")):
        res = reg.dispatch_attention_bwd(R(prob, arch))
        plan = res.spec
        main = [s for s in plan.launches if s.stage in ("main", "dense")]
        assert len(main) == 1
        assert res.grid == tuple(main[0].grid)
        assert res.block == (int(main[0].block), 1, 1)
        assert [s["kernel"] for s in res.signature] == [s.kernel for s in plan.launches]
        assert res.kernel_id.abi_version == res.candidate.abi_version
        assert res.kernel_id.op == reg.FAMILY
        assert res.explanation[0].startswith("selected ")
        assert res.candidate.sweep_space(res.request)[0].launches == plan.launches
    # different families on the same arch, different identities
    a = reg.dispatch_attention_bwd(R(_req(), "gfx950")).kernel_id
    u = reg.dispatch_attention_bwd(R(_unified(), "gfx950")).kernel_id
    assert a.spec_hash != u.spec_hash and a.candidate != u.candidate


def test_launch_uses_the_family_runner(monkeypatch):
    from kernels.common import attention_bwd_run, attention_bwd_unified_run

    calls = []

    def fake(name):
        def run(plan, **kw):
            calls.append((name, plan.route, kw["workspace_size"], kw["stream"]))
            return [s.kernel for s in plan.launches]

        return run

    monkeypatch.setattr(attention_bwd_run, "run_attn_bwd", fake("general"))
    monkeypatch.setattr(
        attention_bwd_unified_run, "run_attn_bwd_unified", fake("unified")
    )
    for prob, runner in ((_req(), "general"), (_unified(), "unified")):
        res = reg.dispatch_attention_bwd(R(prob, "gfx950"))
        ws = res.spec.workspace_bytes
        res2, names = reg.run_attention_bwd(
            R(prob, "gfx950"), tensors={}, workspace=0, workspace_size=ws, stream=7
        )
        assert names == [s.kernel for s in res.spec.launches]
        assert calls[-1] == (runner, res.spec.route, ws, 7)
        assert res2.kernel_id == res.kernel_id
    # a dense selection runs on the general runner (its stages are packed there)
    prob = _dense_requests("gfx950")[0]
    res, _names = reg.run_attention_bwd(
        R(prob, "gfx950"), tensors={}, workspace=0, workspace_size=1 << 40
    )
    assert res.spec.route == "dense" and calls[-1][0] == "general"
    n = len(calls)
    with pytest.raises(ValueError, match="smaller than the plan"):
        big = R(_req(s_q=4096, s_kv=4096), "gfx950")
        reg.run_attention_bwd(big, tensors={}, workspace=0, workspace_size=16)
    assert len(calls) == n  # nothing launched


def _general_meta(prob):
    """TensorMeta of every role of a dense-layout general request (exact
    shapes and strides, room to spare)."""
    from kernels.common.attention_bwd_plan import _lse_kernarg_strides
    from kernels.common.attention_bwd_run import TensorMeta

    big = 1 << 40
    meta = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        shape = (prob.b, prob.tensor_heads(n), prob.tensor_seq(n), prob.d_qk)
        meta[n] = TensorMeta(shape, prob.tensor_strides(n), prob.dtype, big)
    lse = (prob.b, prob.h_q, prob.s_q)
    meta["lse"] = TensorMeta(lse, _lse_kernarg_strides(prob), "fp32", big)
    return meta


def test_launch_checks_the_callers_tensors_first(monkeypatch):
    """With ``tensor_meta`` the entry checks the caller's tensors against the
    request before anything is compiled or launched."""
    from kernels.common import attention_bwd_run, attention_bwd_unified_run
    from kernels.common.attention_bwd_run import TensorMeta

    calls = []

    def run(plan, **kw):
        calls.append(plan.route)
        return [s.kernel for s in plan.launches]

    monkeypatch.setattr(attention_bwd_run, "run_attn_bwd", run)
    monkeypatch.setattr(attention_bwd_unified_run, "run_attn_bwd_unified", run)
    prob = _req()
    meta = _general_meta(prob)
    kw = dict(tensors={}, workspace=0, workspace_size=1 << 40)
    reg.run_attention_bwd(R(prob, "gfx950"), tensor_meta=meta, **kw)
    assert calls == ["general"]
    bad = dict(meta)
    q = meta["q"]
    bad["q"] = TensorMeta((q.shape[0], q.shape[1], q.shape[2] - 1, q.shape[3]),
                          q.strides, q.dtype, q.nbytes)  # fmt: skip
    with pytest.raises(ValueError, match="^TENSOR_SHAPE"):
        reg.run_attention_bwd(R(prob, "gfx950"), tensor_meta=bad, **kw)
    small = dict(meta, dk=TensorMeta(meta["dk"].shape, meta["dk"].strides,
                                     meta["dk"].dtype, 64))  # fmt: skip
    with pytest.raises(ValueError, match="^TENSOR_EXTENT"):
        reg.run_attention_bwd(R(prob, "gfx950"), tensor_meta=small, **kw)
    missing = {k: v for k, v in meta.items() if k != "lse"}
    with pytest.raises(ValueError, match="^MISSING_TENSOR"):
        reg.run_attention_bwd(R(prob, "gfx950"), tensor_meta=missing, **kw)
    # the unified entry checks its own roles
    with pytest.raises(ValueError):
        reg.run_attention_bwd(R(_unified(), "gfx950"), tensor_meta={}, **kw)
    assert calls == ["general"]  # nothing launched after a failed check
    # without metadata the pointers are trusted (the runner is reached)
    reg.run_attention_bwd(R(_unified(), "gfx950"), **kw)
    assert calls == ["general", "unified"]


# ------------------------------------------------------------------ shipped policy


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_shipped_policy_serves_cdna_without_override(arch):
    """The shipped dispatcher (no policy argument anywhere) serves legal
    requests on the CDNA targets, with no per-class status claimed."""
    assert not routing.routed_policy().perf_status
    for prob in (
        _req(),
        _req(dtype="fp16", d_qk=64, d_v=64, causal=False),
        _req(h_k=2, h_v=2, causal=False, right_bound=0, bottom_right=True),
        _req(layout="thd", causal=False),
        _req(num_cus=256, num_xcds=8),
    ):
        res = reg.dispatch_attention_bwd(R(prob, arch))
        assert res.spec.perf_status == "unverified"
        assert res.candidate.algorithm in ("general", "dense")
