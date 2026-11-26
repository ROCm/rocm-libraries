# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics through the backward dispatcher's library entry point.

Each case is planned by :func:`dispatch.attention_bwd.dispatch_attention_bwd`
under the shipped policy (no test policy, no override) and launched by
:func:`dispatch.attention_bwd.launch_attention_bwd`, on the device harnesses
(``_attention_bwd_harness`` / ``_attention_bwd_unified_harness``: oracle,
sentinels, canaries, poisoned workspace):

* ``test_general_cases``: smoke-tier table cases the general predicate
  accepts on the device;
* ``test_tuned_general_keys`` (CDNA): one case per shipped tuned general key
  of a fixed-length class (its request carries the key's ``num_cus``): the
  dispatcher plans the tuned configuration and it passes;
* ``test_tuned_dense_keys`` (gfx950): the shipped dense keys whose float64
  oracle fits the host (the dense family is selected and passes);
* ``test_unified_cases`` (gfx942 / gfx950): smoke-tier unified cases without
  forced knobs, in the page mode the shipped plan chooses;
* ``test_negative_control_general`` / ``test_negative_control_unified``: the
  same entry with the main launch dropped fails the checks.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dispatch.attention_bwd import registry as reg
from dispatch.attention_bwd import routing, tuned_tables
from kernels.common.attention_bwd_plan import (
    _dense_key,
    _lse_kernarg_strides,
    _table_knobs,
    attn_bwd_support,
    attn_bwd_tuning_key,
)
from kernels.common.attention_bwd_run import TensorMeta
from kernels.common.attention_bwd_unified_plan import (
    VALIDATED_ARCHS,
    attn_bwd_unified_support,
)

from . import _attention_bwd_harness as H
from . import _attention_bwd_unified_harness as HU
from .sdpa.bwd_cases import _mkb, compare_bwd, select_bwd
from .sdpa.bwd_kernel_cases import SUPPLEMENTARY_CASES, request_from_case
from .sdpa.bwd_unified_cases import select_unified

ARCH = H.ARCH

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in H.DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

# float64 oracle budget: batches * heads * s_q * s_kv
_ORACLE_CELLS = 160 * 1024 * 1024
_CACHE: dict = {}


def _case_meta(case, req):
    """TensorMeta of the harness's buffers for a batched case (``None`` for a
    ragged case, whose storage the harness lays out itself): the request's
    shapes and strides, the bytes from each view's offset to its buffer's
    end."""
    if case.length_mode == "ragged":
        return None
    inp, _ref = H.reference(case)
    meta = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        dsc = inp.descs[n]
        shape = (req.b, req.tensor_heads(n), req.tensor_seq(n), req.d_qk)
        nbytes = (int(inp.sizes[dsc.buffer]) - int(dsc.offset)) * 2
        meta[n] = TensorMeta(shape, req.tensor_strides(n), req.dtype, nbytes)
    lse_bytes = int(np.size(inp.buffers["lse"])) * 4
    meta["lse"] = TensorMeta(
        (req.b, req.h_q, req.s_q), _lse_kernarg_strides(req), "fp32", lse_bytes
    )
    if req.padding:
        for n in ("seq_len_q", "seq_len_kv"):
            meta[n] = TensorMeta((req.b,), (1,), "int32", 4 * req.b)
    return meta


@pytest.fixture
def through_dispatcher(monkeypatch):
    """Plan and launch the general harness through the dispatcher entry.

    The harness's planner becomes ``dispatch_attention_bwd`` on the case's
    request (plus the test's request fields) and its run function
    ``launch_attention_bwd`` of that result; the results are recorded."""
    results, metas = [], []

    def plan_case(case, *, arch, config, overrides=None, request=None):
        assert overrides is None, "the dispatcher takes no knob overrides"
        req = request_from_case(case)
        if request:
            req = dataclasses.replace(req, **request)
        res = reg.dispatch_attention_bwd(reg.AttentionBwdRequest(req, arch))
        results.append(res)
        metas.append(_case_meta(case, req))
        return res.spec

    def run(plan, **kw):
        res = results[-1]
        assert res.spec is plan
        # the entry checks the harness's buffers against the request first
        return reg.launch_attention_bwd(res, tensor_meta=metas[-1], **kw)

    monkeypatch.setattr(H, "plan_case", plan_case)
    monkeypatch.setattr(H, "run_attn_bwd", run)
    return results


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _accepted(case):
    return attn_bwd_support(request_from_case(case), ARCH).ok


def _general_params():
    if ARCH not in H.DEVICE_ARCHS:
        return [pytest.param(None, id="no-device")]
    cases = list(select_bwd("smoke")) + [
        c for c in SUPPLEMENTARY_CASES if "smoke" in c.tags
    ]
    out, seen = [], set()
    for c in cases:
        if c.id in seen or ARCH not in c.archs or not _accepted(c):
            continue
        seen.add(c.id)
        out.append(pytest.param(c, id=c.id))
    return out


@pytest.mark.parametrize("case", _general_params())
def test_general_cases(case, through_dispatcher):
    run = H.run_case(case, arch=ARCH, config="default", cache=_CACHE)
    res = through_dispatcher[-1]
    assert run.plan is res.spec
    assert res.candidate.capability.arches == (ARCH,)
    assert (
        res.spec.route == routing.attention_bwd_route(request_from_case(case), ARCH)[0]
    )
    _passes(run)


def _without_main(res):
    """``res`` with its main launch dropped (negative control)."""
    plan = res.spec
    cut = tuple(s for s in plan.launches if s.stage not in ("main", "dense"))
    assert len(cut) == len(plan.launches) - 1
    return dataclasses.replace(res, spec=dataclasses.replace(plan, launches=cut))


def test_negative_control_general(through_dispatcher, monkeypatch):
    """The same entry with the main launch dropped fails the oracle: the
    passes above come from the dispatched kernels."""
    params = [p for p in _general_params() if p.values[0] is not None]
    if not params:
        pytest.skip(f"no general case on {ARCH}")
    case = params[0].values[0]

    def broken(plan, **kw):
        res = through_dispatcher[-1]
        assert res.spec is plan
        return reg.launch_attention_bwd(_without_main(res), **kw)

    monkeypatch.setattr(H, "run_attn_bwd", broken)
    run = H.run_case(case, arch=ARCH, config="default", cache=_CACHE)
    assert compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref) or run.integrity


_MASK_CASE = {
    "none": {},
    "causal_tl": {"diagonal": "top_left", "right_bound": 0},
    "causal_br": {"diagonal": "bottom_right", "right_bound": 0},
}


def _general_key_case(arch, key):
    """A fixed-length bhsd / bshd case of tuned key ``key`` and the request
    fields that put it in the key (``num_cus``, ``num_xcds``), or None."""
    for req in tuned_tables.general_key_requests(arch, key):
        if req.layout != "dense" or req.padding or req.num_cus is None:
            continue
        mask = key[4]
        if mask not in _MASK_CASE or req.b * req.h_q * req.s_q * req.s_kv > (
            _ORACLE_CELLS
        ):
            continue
        layout = "bhsd" if req.strides is None else "bshd"
        tag = "_".join(str(x) for x in key[2:]).replace("*", "any")
        case = _mkb(
            f"disp_{arch}_{tag}_b{req.b}h{req.h_q}k{req.h_k}s{req.s_q}x{req.s_kv}",
            reqs=(1,),
            tier="full",
            dtype=req.dtype,
            d=req.d_qk,
            b=req.b,
            h_q=req.h_q,
            h_k=req.h_k,
            h_v=req.h_v,
            s_q=req.s_q,
            s_kv=req.s_kv,
            layout=layout,
            **_MASK_CASE[mask],
        )
        extra = {"num_cus": req.num_cus, "num_xcds": req.num_xcds}
        got = dataclasses.replace(request_from_case(case), **extra)
        if attn_bwd_tuning_key(got, arch) == key and attn_bwd_support(got, arch).ok:
            return case, extra
    return None


def _tuned_general_params():
    if ARCH not in routing.ARCH_TUNING:
        return [pytest.param(None, None, None, id="no-tables")]
    out = []
    for key in routing.ARCH_TUNING[ARCH].GENERAL_TUNED_TABLE:
        if key[7] != "fixed":
            continue
        found = _general_key_case(ARCH, key)
        if found is not None:
            out.append(pytest.param(key, *found, id=found[0].id))
    return out


@pytest.mark.parametrize("key,case,extra", _tuned_general_params())
def test_tuned_general_keys(key, case, extra, through_dispatcher):
    if key is None:
        pytest.skip(f"no tuned tables on {ARCH}")
    run = H.run_case(case, arch=ARCH, config="default", request=extra, cache=_CACHE)
    plan = through_dispatcher[-1].spec
    assert plan.tuning_key == key
    # the general entry applies (a packed dense key routes dense instead)
    assert plan.route == "dense" or _table_knobs(
        routing.routed_policy().tuned_table, key
    )
    _passes(run)


def _dense_params():
    mod = routing.ARCH_TUNING.get(ARCH)
    if mod is None or not mod.DENSE_TUNED:
        return [pytest.param(None, id="no-dense-keys")]
    out = []
    for _arch, k in mod.DENSE_TUNED:
        s_q, s_kv, h_q, h_k, d, dtype, layout, mask, (left, right, br) = k
        if h_q * s_q * s_kv > _ORACLE_CELLS:
            continue  # the float64 oracle does not fit the host budget
        band = {}
        if (left, right) != (-1, -1):
            band = dict(
                left_bound=left,
                right_bound=right,
                diagonal="bottom_right" if br else "top_left",
            )
        case = _mkb(
            f"disp_dense_{ARCH}_{s_q}x{s_kv}_h{h_q}k{h_k}_d{d}_{dtype}_{layout}_{mask}",
            reqs=(1,),
            tier="full",
            dtype=dtype,
            d=d,
            b=1,
            h_q=h_q,
            h_k=h_k,
            s_q=s_q,
            s_kv=s_kv,
            layout=layout,
            **band,
        )
        out.append(pytest.param(case, id=case.id))
    return out or [pytest.param(None, id="no-dense-key-in-budget")]


@pytest.mark.parametrize("case", _dense_params())
def test_tuned_dense_keys(case, through_dispatcher):
    if case is None:
        pytest.skip(f"no shipped dense key (within the oracle budget) on {ARCH}")
    req = request_from_case(case)
    assert (ARCH, _dense_key(req)) in routing.ARCH_TUNING[ARCH].DENSE_TUNED
    run = H.run_case(case, arch=ARCH, config="default", cache=_CACHE)
    res = through_dispatcher[-1]
    assert res.candidate.algorithm == "dense" and run.plan.route == "dense"
    _passes(run)


def _unified_params():
    if ARCH not in VALIDATED_ARCHS:
        return [pytest.param(None, id="no-unified-target")]
    out = []
    for case in select_unified("smoke"):
        if case.knobs:
            continue
        out.append(pytest.param(case, id=case.id))
    return out


def _unified_meta(inp, req):
    """TensorMeta of the unified harness's buffers: the request's shapes and
    strides, each buffer's own bytes."""
    from kernels.common.attention_bwd_unified_plan import _lse_layout

    dt = req.dtype

    def nbytes(name):
        if name in inp.buffers:
            return int(np.size(inp.buffers[name])) * 2
        return int(inp.sizes[name]) * 2

    meta = {}
    for n in ("q", "o", "do", "dq"):
        meta[n] = TensorMeta(
            (req.total_q, req.h_q, req.d_qk), req.tensor_strides(n), dt, nbytes(n)
        )
    for role, n in (("k_cache", "k"), ("v_cache", "v"), ("dk_cache", "dk"),
                    ("dv_cache", "dv")):  # fmt: skip
        meta[role] = TensorMeta(
            tuple(req.tensor_dims(n)), req.tensor_strides(n), dt, nbytes(role)
        )
    _why, (lt, lh) = _lse_layout(req)
    meta["lse"] = TensorMeta(
        (req.total_q, req.h_q), (lt, lh), "fp32", int(np.size(inp.lse)) * 4
    )
    for role, values in (
        ("cu_seqlens_q", inp.cu_seqlens_q),
        ("seq_lens_kv", inp.seq_lens_kv),
        ("block_table", inp.block_table),
    ):
        size = int(np.size(values))
        meta[role] = TensorMeta((size,), (1,), "int32", 4 * size)
    return meta


def _run_unified(case, monkeypatch, *, control: bool = False):
    """Plan ``case`` by the dispatcher (shipped policy) and launch it by the
    entry point on the unified harness; ``control``: the main launch dropped
    (negative control)."""
    results, metas = [], []

    def plan_unified_case(case_, *, arch, mode, overrides=None):
        assert overrides is None
        inp = HU.inputs_of(case_)
        req = HU.request_from_unified(inp)
        assert attn_bwd_unified_support(req, arch).ok
        res = reg.dispatch_attention_bwd(reg.AttentionBwdRequest(req, arch))
        assert res.spec.runtime["page_mode"] == mode
        results.append(res)
        metas.append(_unified_meta(inp, req))
        return res.spec

    def run(plan, **kw):
        res = results[-1]
        assert res.spec is plan
        if control:
            return reg.launch_attention_bwd(_without_main(res), **kw)
        # the entry checks the harness's buffers against the request first
        return reg.launch_attention_bwd(res, tensor_meta=metas[-1], **kw)

    monkeypatch.setattr(HU, "plan_unified_case", plan_unified_case)
    monkeypatch.setattr(HU, "run_attn_bwd_unified", run)
    mode = "direct" if case.injective else "atomic"
    out = HU.run_unified_case(case, arch=ARCH, mode=mode)
    assert results[-1].candidate.name == f"attention_bwd_unified_{ARCH}"
    return out


@pytest.mark.parametrize("case", _unified_params())
def test_unified_cases(case, monkeypatch):
    if case is None:
        pytest.skip(f"the unified family is not validated on {ARCH}")
    run = _run_unified(case, monkeypatch)
    assert not run.integrity, run.integrity
    assert not run.problems, run.problems


def test_negative_control_unified(monkeypatch):
    """The unified entry with the main launch dropped fails the checks."""
    case = _unified_params()[0].values[0]
    if case is None:
        pytest.skip(f"the unified family is not validated on {ARCH}")
    run = _run_unified(case, monkeypatch, control=True)
    assert run.integrity or run.problems
