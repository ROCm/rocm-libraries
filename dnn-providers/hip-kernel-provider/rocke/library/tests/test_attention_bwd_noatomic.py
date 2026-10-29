# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The atomic-free dK / dV sink (``dkv_sink = "slices"``) and the atomic-free
backward (``dq_mode = "split"`` with it).

CPU:

* the knob: values, inert rules, kernel names, the not-built ``last_cta``
  value (validator, emission check, plan refusal, registry);
* the plan: launch list (``reduce`` stages instead of the dK / dV converts),
  prep without the kv zero fill, the slice workspace terms, the reduce
  kernargs, the slice ownership (every slice row of a kv head written by
  exactly one CTA), the deterministic policy and the declaration;
* the IR: no atomic instruction in any launch of a split + slices plan on
  every epilogue form and target; with atomic dQ the main kernel's atomics are
  exactly the dQ sink's; default plans keep their kernels and IR;
* the sweep: stage R crosses the sink with the atomic points; the custom
  test configuration caps a rule's head split per request.

Device (gfx942 / gfx950 for split dQ; the slice sink alone on every target):
every case passes the float64 oracle and the harness integrity facts; with
split dQ two runs give the same dQ / dK / dV bits, with atomic dQ the same
dK / dV bits.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re

import numpy as np
import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from benchmarks.common import attention_bwd_sweep as sw
from dispatch.attention_bwd import tuning_common as tc
from kernels.common.attention_bwd import (
    DKV_SINK_NOT_BUILT,
    DKV_SINKS,
    INERT_KNOBS,
    SPLIT_DQ_ARCHS,
    AttnBwdSpec,
    _not_built,
    attn_bwd_params,
    build_attn_bwd_main,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_plan import (
    FEATURE_GROUPS,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_declaration,
    attn_bwd_plan,
    attn_bwd_support,
    attn_bwd_workspace_layout,
)
from kernels.common.attention_bwd_run import _BUILDERS
from ._attention_bwd_test_utils import find_bwd_case

ALL_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
_ATOMIC = re.compile(r"\b(atomicrmw|cmpxchg)\b")


def _policy(knobs=None, **kw) -> AttnBwdPolicy:
    knobs = dict(knobs or {})
    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=lambda *a: dict(knobs),
        **kw,
    )


def _req(**kw) -> AttnBwdRequest:
    base = dict(
        b=2, h_q=8, h_k=1, h_v=1, s_q=150, s_kv=170, d_qk=64, d_v=64,
        dtype="fp16", causal=True,
    )  # fmt: skip
    base.update(kw)
    return AttnBwdRequest(**base)


def _atomics(ir: str) -> int:
    return sum(1 for ln in ir.splitlines() if _ATOMIC.search(ln))


def _ir(plan, step) -> str:
    k = _BUILDERS[step.stage](plan.specs[step.stage], arch=plan.arch)
    return lower(k, arch=plan.arch)


# ---------------------------------------------------------------------------
# the knob
# ---------------------------------------------------------------------------


def test_sink_values_and_kernel_names():
    assert DKV_SINKS == ("atomic", "slices", "last_cta")
    assert AttnBwdSpec(head_size=64).dkv_sink == "atomic"
    base = AttnBwdSpec(head_size=64, dkv_mode="atomic")
    sl = dataclasses.replace(base, dkv_sink="slices")
    assert sl.kernel_name() != base.kernel_name()
    assert ("dkv_sink", "slices") in sl.non_default_fields()
    with pytest.raises(ValueError, match="dkv_sink"):
        AttnBwdSpec(head_size=64, dkv_sink="ring")


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_last_cta_is_registered_not_built(arch):
    s = validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64, dkv_mode="atomic", dkv_sink="last_cta"), arch
    )
    assert "dkv_sink='last_cta'" in _not_built(s)
    with pytest.raises(NotImplementedError, match="last_cta"):
        build_attn_bwd_main(s, arch=arch)
    # inert (and so built) in direct mode
    d = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dkv_sink="last_cta"), arch)
    assert not _not_built(d)
    reason = DKV_SINK_NOT_BUILT["last_cta"]
    assert "fence" in reason and "s_barrier" in reason
    with pytest.raises(ValueError, match="dkv_sink='last_cta' is not built"):
        attn_bwd_plan(
            _req(), arch, policy=_policy({"dkv_sink": "last_cta", "g_split": 2})
        )
    # inert in direct mode: planned like the default sink
    attn_bwd_plan(_req(), arch, policy=_policy({"dkv_sink": "last_cta"}))
    axis = tc.backward_axis("dkv_sink")
    assert axis.default == "atomic" and axis.not_built == ("last_cta",)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1151"))
def test_inert_rules(arch):
    direct = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dkv_sink="slices"), arch)
    assert INERT_KNOBS["dkv_sink"](direct, arch)
    a = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dkv_mode="atomic"), arch)
    s = dataclasses.replace(a, dkv_sink="slices")
    assert not INERT_KNOBS["dkv_sink"](s, arch)
    # the slice sink has no atomic row guard
    assert INERT_KNOBS["dkv_atomic_guard"](s, arch)
    assert not INERT_KNOBS["dkv_atomic_guard"](a, arch)
    # the store width is the slice rows' width, unless stored per slot
    assert INERT_KNOBS["dkv_store_vec"](a, arch)
    assert not INERT_KNOBS["dkv_store_vec"](s, arch)
    if not s.acc_in_lds:
        assert INERT_KNOBS["dkv_store_vec"](
            dataclasses.replace(s, dkv_epilogue="direct"), arch
        )
        assert not INERT_KNOBS["dkv_epilogue"](s, arch)
    else:
        assert INERT_KNOBS["dkv_epilogue"](s, arch)

    # an inert value emits the default IR (apart from the kernel name)
    def body(spec):
        ir = lower(build_attn_bwd_main(spec, arch=arch), arch=arch)
        return ir.replace(spec.kernel_name("main"), "K")

    assert body(direct) == body(dataclasses.replace(direct, dkv_sink="atomic"))
    sg = dataclasses.replace(s, dkv_atomic_guard="select")
    assert body(sg) == body(s)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _ownership(h_q, h_k, h_v, g_split):
    """``{("dk"|"dv", head, slice): count}`` of the CTA decode
    (``head_range`` and the slice formula of ``_slice_slab``)."""
    gk, gv = h_q // h_k, h_q // h_v
    group = math.gcd(gk, gv)
    per = group // g_split
    seen = {}
    for unit in range(h_q // group):
        for split in range(g_split):
            h0 = unit * group + split * per
            for which, g in (("dk", gk), ("dv", gv)):
                key = (which, (unit * group) // g, (h0 % g) // per)
                seen[key] = seen.get(key, 0) + 1
    return seen


@pytest.mark.parametrize(
    "h_q,h_k,h_v,g_split",
    [
        (8, 1, 1, 2), (8, 1, 1, 4), (8, 1, 1, 8), (8, 2, 2, 2), (8, 2, 2, 4),
        (32, 8, 8, 4), (8, 4, 2, 1), (8, 4, 2, 2), (8, 2, 8, 1), (12, 2, 3, 1),
        (64, 8, 8, 8), (16, 2, 4, 4),
    ],
)  # fmt: skip
def test_every_slice_row_has_one_writer(h_q, h_k, h_v, g_split):
    req = _req(h_q=h_q, h_k=h_k, h_v=h_v)
    plan = attn_bwd_plan(
        req, "gfx942", policy=_policy({"g_split": g_split, "dkv_sink": "slices"})
    )
    red = [s for s in plan.launches if s.stage == "reduce"]
    n_k, n_v = (s.scalars["n_slices"] for s in red)
    seen = _ownership(h_q, h_k, h_v, g_split)
    want = {("dk", h, i) for h in range(h_k) for i in range(n_k)}
    want |= {("dv", h, i) for h in range(h_v) for i in range(n_v)}
    assert set(seen) == want and set(seen.values()) == {1}


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_slice_plan(arch):
    for req, g in ((_req(), 4), (_req(h_q=8, h_k=4, h_v=2), 1)):
        knobs = {"g_split": g, "dkv_sink": "slices"}
        atomic = attn_bwd_plan(req, arch, policy=_policy({"g_split": g}))
        plan = attn_bwd_plan(req, arch, policy=_policy(knobs))
        stages = [s.stage for s in plan.launches]
        assert stages == ["prep", "main", "convert", "reduce", "reduce"]
        assert [s.stage for s in atomic.launches][3:] == ["convert"] * 2
        assert plan.runtime["dkv_sink"] == "slices"
        assert "dkv_sink" not in atomic.runtime
        main = plan.specs["main"]
        assert main.dkv_mode == "atomic" and main.dkv_sink == "slices"
        prep = plan.launches[0]
        assert prep.scalars["zero_kv"] == 0 and atomic.launches[0].scalars["zero_kv"]
        # no kv-zero role in the prep grid
        assert prep.grid[1] == req.h_q < atomic.launches[0].grid[1]
        group = math.gcd(req.h_q // req.h_k, req.h_q // req.h_v)
        per = group // g
        rows_kv = req.b * req.s_kv
        for step, heads, gq in (
            (plan.launches[3], req.h_k, req.h_q // req.h_k),
            (plan.launches[4], req.h_v, req.h_q // req.h_v),
        ):
            assert step.scalars["n_slices"] == gq // per
            assert step.scalars["s_stride"] == rows_kv * heads * req.d_qk
            assert step.kernel == plan.specs["reduce"].kernel_name("reduce")
            names = [n for n, _ in attn_bwd_params("reduce")]
            assert sorted(names) == sorted(list(step.pointers) + list(step.scalars))
        lay, base = plan.workspace_layout, atomic.workspace_layout
        n_k, n_v = (
            plan.launches[3].scalars["n_slices"],
            plan.launches[4].scalars["n_slices"],
        )
        assert lay["WS_DK"][1] == base["WS_DK"][1] * n_k
        assert lay["WS_DV"][1] == base["WS_DV"][1] * n_v
        assert lay == attn_bwd_workspace_layout(req, g_split=g, dkv_sink="slices")
        # the convert ABI is unchanged; the reduce adds the slice stride / count
        conv = [n for n, _ in attn_bwd_params("convert")]
        assert [n for n, _ in attn_bwd_params("reduce")] == (
            conv[:6] + ["s_stride"] + conv[6:] + ["n_slices"]
        )


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_default_plans_are_unchanged(arch):
    """No sink value: the plan of today (no runtime key, the converts, the
    kv zero fill, the same kernels); the explicit default value too."""
    for req, knobs in (
        (_req(), {"g_split": 4}),
        (_req(h_q=8, h_k=4, h_v=2), {}),
        (_req(h_k=8, h_v=8), {}),
    ):
        a = attn_bwd_plan(req, arch, policy=_policy(knobs))
        b = attn_bwd_plan(req, arch, policy=_policy({**knobs, "dkv_sink": "atomic"}))
        assert a == b
        assert "dkv_sink" not in a.runtime and "reduce" not in a.specs
        assert all(s.stage != "reduce" for s in a.launches)
        assert not a.deterministic
        # in direct mode the sink is inert (a salted name, same launches)
        if a.specs["main"].dkv_mode == "direct":
            s = attn_bwd_plan(
                req, arch, policy=_policy({**knobs, "dkv_sink": "slices"})
            )
            assert [x.stage for x in s.launches] == [x.stage for x in a.launches]
            assert "dkv_sink" not in s.runtime


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"dkv_epilogue": "lds_staged_pair"},
        {"dkv_epilogue": "direct"},
        {"dkv_store_vec": 4},
        {"dkv_store_vec": 1},
        {"ws_layout": "token_major"},
        {"block_m": 32, "pt_route": "lds", "warp_grid_g02": (2, 4),
         "warp_grid_g13": (2, 4)},
        {"waves": 2, "block_m": 16, "block_n": 32, "warp_grid_g4": (1, 2),
         "atom_g02": "16x16x16", "atom_g13": "16x16x16", "atom_g4": "16x16x16",
         "acc_in_lds": True, "agpr_alloc": (0, 0)},
    ],
    ids=["staged", "pair", "direct", "vec4", "vec1", "token_major", "grid2xw",
         "acc_in_lds"],
)  # fmt: skip
def test_atomic_free_plan_has_no_atomic_instruction(arch, extra):
    for req, g in ((_req(), 8), (_req(h_q=8, h_k=4, h_v=2), 1)):
        knobs = {"g_split": g, "dkv_sink": "slices", "dq_mode": "split", **extra}
        plan = attn_bwd_plan(req, arch, policy=_policy(knobs))
        assert plan.atomic_free and plan.deterministic
        assert [s.stage for s in plan.launches] == [
            "prep", "main", "dq", "reduce", "reduce"
        ]  # fmt: skip
        for step in plan.launches:
            assert _atomics(_ir(plan, step)) == 0, (step.stage, step.kernel)


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_slices_main_keeps_only_the_dq_atomics(arch):
    """With atomic dQ the slice main's atomics are the dQ sink's: exactly
    those of the same kernel in direct mode (no dK / dV atomic)."""
    req = _req(h_q=8, h_k=4, h_v=2)
    plan = attn_bwd_plan(req, arch, policy=_policy({"dkv_sink": "slices"}))
    spec = plan.specs["main"]
    sl = _atomics(lower(build_attn_bwd_main(spec, arch=arch), arch=arch))
    direct = dataclasses.replace(spec, dkv_mode="direct", dkv_sink="atomic")
    want = _atomics(lower(build_attn_bwd_main(direct, arch=arch), arch=arch))
    atomic = dataclasses.replace(spec, dkv_sink="atomic")
    more = _atomics(lower(build_attn_bwd_main(atomic, arch=arch), arch=arch))
    assert sl == want and more > sl
    for step in plan.launches:
        if step.stage != "main":
            assert _atomics(_ir(plan, step)) == 0


def test_reduce_kernel_sums_in_slice_order():
    plan = attn_bwd_plan(
        _req(), "gfx942", policy=_policy({"g_split": 4, "dkv_sink": "slices"})
    )
    ir = lower(_BUILDERS["reduce"](plan.specs["reduce"], arch="gfx942"), arch="gfx942")
    assert _atomics(ir) == 0
    assert "rslice" in ir  # the fixed-order slice loop
    assert "n_slices" in ir and "s_stride" in ir


# ---------------------------------------------------------------------------
# determinism: policy, plan, declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_deterministic_requests(arch):
    req = _req(deterministic=True)
    v = attn_bwd_support(req, arch, policy=_policy())
    assert not v.ok and v.code == "DETERMINISTIC"
    pol = _policy(deterministic=True)
    v = attn_bwd_support(req, arch, policy=pol)
    if arch not in SPLIT_DQ_ARCHS:
        assert not v.ok and v.code == "DETERMINISTIC"
        return
    assert v.ok, v
    plan = attn_bwd_plan(req, arch, policy=pol)
    assert plan.deterministic and plan.specs["main"].dq_mode == "split"
    assert plan.specs["main"].dkv_mode == "direct"  # MQA, no head split
    # a head split: the slices are forced
    pol4 = _policy({"g_split": 4}, deterministic=True)
    plan = attn_bwd_plan(req, arch, policy=pol4)
    assert plan.specs["main"].dkv_sink == "slices" and plan.deterministic
    # the flag wins over a contradicting tuned value (an admitted request plans)
    for tuned in ({"g_split": 2, "dkv_sink": "atomic"}, {"dq_mode": "atomic"}):
        plan = attn_bwd_plan(req, arch, policy=_policy(tuned, deterministic=True))
        assert plan.deterministic and plan.atomic_free
        assert plan.specs["main"].dq_mode == "split"
        if plan.specs["main"].dkv_mode == "atomic":
            assert plan.specs["main"].dkv_sink == "slices"
    # a request without the flag keeps the tuned values
    plan = attn_bwd_plan(
        _req(), arch, policy=_policy({"g_split": 2}, deterministic=True)
    )
    assert not plan.deterministic


def test_declaration_states_the_atomic_free_mode():
    d = attn_bwd_declaration("gfx942")
    det = d["determinism"]
    assert "dkv_sink='slices'" in det["dk_dv"] and "atomic_free" in det
    assert "n_k" in " ".join(d["workspace"]["terms"])
    assert "slices" in d["workspace"]


# ---------------------------------------------------------------------------
# registry, sweep, test configuration
# ---------------------------------------------------------------------------


def test_registry_and_sweep_stage_r():
    axis = tc.backward_axis("dkv_sink")
    assert axis.family == 6 and axis.stage == "codegen" and axis.kind == "compile"
    assert "dkv_sink" in sw.CODEGEN_EXTRA
    assert axis.applies("slices", {"dkv_mode": "atomic"})
    assert not axis.applies("slices", {"dkv_mode": "direct"})
    assert not tc.backward_axis("dkv_atomic_guard").applies(
        "select", {"dkv_mode": "atomic", "dkv_sink": "slices"}
    )
    rel = sw.Release(sw.ALL_RELEASED, "production")
    assert sw._sink_values(sw.primary_slice("gfx942", "general", 64), rel) == ["slices"]
    assert (
        sw._sink_values(sw.primary_slice("gfx942", "general", 64), sw.Release()) == []
    )
    # stage R: the sink, and each slice-only lever on top of it, on atomic points
    sl = sw.primary_slice("gfx942", "general", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    pts = sw.sink_points(c, rel)
    assert pts[0] == {"dkv_sink": "slices"}
    assert {"dkv_sink": "slices", "ws_bounds": "buffer"} in pts
    assert {"dkv_sink": "slices", "dkv_epilogue": "direct", "dkv_cell_vec": 4} in pts
    assert {"dkv_sink": "slices", "dkv_store_vec": 4} in pts
    stage = sw.stage_candidates(
        "runtime", sl, rel=rel, prior={"swizzle": [c]},
        cells=sw.representative_cells(sl),
    )  # fmt: skip
    sliced = [x for x in stage if x.k.get("dkv_sink") == "slices"]
    assert sliced and all(int(x.r.get("g_split", 1)) > 1 for x in sliced)
    rdna = sw.primary_slice("gfx1151", "general", 64)
    rc = sw.Candidate.make(rdna, sw.default_knobs(rdna))
    assert not any("dkv_cell_vec" in p for p in sw.sink_points(rc, rel))


def test_custom_rule_head_split_is_a_cap(monkeypatch):
    from .sdpa.bwd_kernel_cases import CUSTOM_KNOBS_ENV, config_policy

    monkeypatch.setenv(
        CUSTOM_KNOBS_ENV, '[{"knobs": {"g_split": 8, "dkv_sink": "slices"}}]'
    )
    pol = config_policy("custom", "gfx942")
    for (hq, hk, hv), want in (
        ((8, 8, 8), 1), ((8, 2, 2), 4), ((8, 1, 1), 8), ((8, 4, 2), 2),
        ((6, 2, 2), 1),
    ):  # fmt: skip
        req = _req(h_q=hq, h_k=hk, h_v=hv)
        assert pol.tuning(req, "gfx942", "general", ())["g_split"] == want
    pol = config_policy("custom", "gfx942", {"g_split": 2})
    assert pol.tuning(_req(), "gfx942", "general", ())["g_split"] == 2


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

try:
    from ._attention_bwd_harness import ARCH, DEVICE_ARCHS, run_case
    from .sdpa.bwd_cases import compare_bwd
    from .sdpa.bwd_kernel_cases import THD_HK_NE_HV_CASES, selected_configs, shard_of
except ImportError:  # pragma: no cover - the harness needs the device stack
    ARCH, DEVICE_ARCHS = None, ()

# (case id, head splits): MQA (G = 8), GQA 4, h_k != h_v, THD h_k != h_v
DEVICE_CASES = (
    ("bwd_gqa_mqa_causal_tl", (2, 4, 8)),
    ("bwd_gqa_gqa4_causal_tl", (2, 4)),
    ("bwd_kern_hkv_842_tl_fp16_d64", (1, 2)),
)
# slice-store forms (each on the MQA case at g_split = 4)
EPILOGUES = {
    "staged": {},
    "pair": {"dkv_epilogue": "lds_staged_pair"},
    "direct": {"dkv_epilogue": "direct"},
    "vec4": {"dkv_store_vec": 4},
    "vec1": {"dkv_store_vec": 1},
    "token_major": {"ws_layout": "token_major"},
}
SLICES = {"dkv_sink": "slices"}


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _device_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    cases = list(DEVICE_CASES)
    if THD_HK_NE_HV_CASES:
        cases.append((THD_HK_NE_HV_CASES[0], (1, 2)))
    out = []
    for cid, splits in cases:
        if shard_of(cid, n) != i:
            continue
        for g in splits:
            for split_dq in (False, True):
                if split_dq and ARCH not in SPLIT_DQ_ARCHS:
                    continue
                for cfg in selected_configs(ARCH):
                    tag = f"{cid}-g{g}-{'split' if split_dq else 'atomicdq'}-{cfg}"
                    out.append(pytest.param(cid, g, split_dq, cfg, id=tag))
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("case_id,g_split,split_dq,config", _device_params())
def test_slices_pass_and_repeat_bitwise(case_id, g_split, split_dq, config):
    if ARCH not in DEVICE_ARCHS:
        pytest.skip("no supported device")
    case = find_bwd_case(case_id)
    ov = dict(SLICES, g_split=g_split)
    if split_dq:
        ov["dq_mode"] = "split"
    first = run_case(case, arch=ARCH, config=config, overrides=ov)
    plan = first.plan
    assert plan.specs["main"].dkv_mode == "atomic"
    assert plan.runtime["dkv_sink"] == "slices"
    assert plan.deterministic == split_dq
    _passes(first)
    second = run_case(case, arch=ARCH, config=config, overrides=ov)
    _passes(second)
    assert np.array_equal(first.dk, second.dk), "dK differs between two runs"
    assert np.array_equal(first.dv, second.dv), "dV differs between two runs"
    if split_dq:
        assert np.array_equal(first.dq, second.dq), "dQ differs between two runs"


def _epilogue_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(name, cfg, id=f"{name}-{cfg}")
        for name in EPILOGUES
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("name,config", _epilogue_params())
def test_slice_store_forms_agree(name, config):
    """Every slice-store form writes the same fp32 partials to the same slice
    elements, so dK / dV are bit for bit those of the staged default."""
    if ARCH not in DEVICE_ARCHS:
        pytest.skip("no supported device")
    from ._attention_bwd_harness import plan_case

    case = find_bwd_case("bwd_gqa_mqa_causal_tl")
    ov = dict(SLICES, g_split=4)
    spec = plan_case(case, arch=ARCH, config=config, overrides=ov).specs["main"]
    try:
        validate_attn_bwd_spec(dataclasses.replace(spec, **EPILOGUES[name]), ARCH)
    except ValueError as exc:
        pytest.skip(f"{EPILOGUES[name]} not legal on {spec.kernel_name()}: {exc}")
    base = run_case(case, arch=ARCH, config=config, overrides=ov)
    _passes(base)
    run = run_case(case, arch=ARCH, config=config, overrides={**ov, **EPILOGUES[name]})
    _passes(run)
    assert np.array_equal(run.dk, base.dk), f"dK differs under {name}"
    assert np.array_equal(run.dv, base.dv), f"dV differs under {name}"
