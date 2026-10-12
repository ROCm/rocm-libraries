# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward numerics on a device (gfx942 / gfx950).

Every dense-eligible case (the backward case table of the selected tier, the
supplementary kernel cases, census-shaped cases at reduced batch and heads,
and length tails for the ragged kernel) runs through the dense route:
``prep -> dense main -> convert`` from :func:`attn_dense_bwd_plan`, through
the general harness (sentinel-filled outputs and stride gaps, a NaN-poisoned
workspace with a canary, exact-zero checks). Each run is held to the
backward tolerance against the float64 oracle (:func:`compare_bwd`), and its
dK / dV must equal, bit for bit, the general family's run of the same case
on the same tile with edge tiles on and head packing off (the dense kernel
changes addressing, band classification and the batch source, never the
math or its order).

The knob variants (``oob = "clamp"``, every epilogue, the token-major
workspace, edge tiles off, both persistent decodes, the chiplet remap, the
reverse and interleave load-balance orders, every multiplier applied at run
time (``static_scale = "runtime"``), ``noalias`` workspace pointers, a pinned
batch) must give dK / dV bit for bit equal to the default dense run; atomic
dK / dV (``g_split = 2``) is held to the tolerance.

Selection: ``ROCKE_BWD_TIER`` (smoke / full / exhaustive) and
``ROCKE_BWD_SHARD`` (``i/n``) as for the general numeric suite.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from kernels.common.attention_bwd_plan import dense_bwd_eligibility
from kernels.common.attention_dense_bwd_plan import (
    attn_dense_bwd_plan,
    dense_bwd_spec_for,
)

from ._attention_bwd_harness import ARCH, run_case
from .sdpa.bwd_cases import _mkb, compare_bwd, select_bwd, validate_bwd_case
from .sdpa.bwd_kernel_cases import (
    SUPPLEMENTARY_CASES,
    config_policy,
    request_from_case,
    shard_of,
)
from .sdpa.cases import _mask

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in ("gfx942", "gfx950"),
        reason=f"the dense backward runs on gfx942 / gfx950; device is {ARCH}",
    ),
]

_CENSUS = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "common"
    / "attention_bwd_census.json"
)
# Census-shaped cases keep each cell's head size, dtype, mask, layout and
# head ratio, with batch 1, at most two kv heads and both lengths capped, so
# the float64 oracle stays affordable.
_CENSUS_S_CAP = 1024
_TAIL_LENGTHS = ((200, 200), (136, 72), (72, 1000), (1000, 136), (17, 33))


def _census_cases():
    try:
        census = json.loads(_CENSUS.read_text())
    except OSError:
        return []
    out, seen = [], set()
    for cfg in census.get("configs", []):
        layout = cfg.get("layout")
        if layout not in ("bshd", "bhsd") or cfg.get("hv") not in (None, cfg["hkv"]):
            continue
        if cfg.get("padded_q") or cfg.get("padded_kv") or cfg.get("seqlens"):
            continue
        d, dt, mask = int(cfg["d"]), cfg["dtype"], cfg["mask"]
        hq, hk = int(cfg["hq"]), int(cfg["hkv"])
        if d not in (64, 128) or hk < 1 or hq % hk:
            continue
        ratio = hq // hk
        kv = min(hk, 2)
        sq, skv = min(int(cfg["sq"]), _CENSUS_S_CAP), min(
            int(cfg["skv"]), _CENSUS_S_CAP
        )
        if mask == "swa":
            m = dict(_mask("br"), left_bound=int(cfg["window"]) - 1)
        else:
            m = {"none": {}, "causal_tl": _mask("tl"), "causal_br": _mask("br")}.get(
                mask
            )
        if m is None:
            continue
        key = (d, dt, ratio, kv, sq, skv, mask, cfg.get("window"), layout)
        if key in seen:
            continue
        seen.add(key)
        win = f"w{cfg['window']}" if mask == "swa" else ""
        cid = (
            f"kern_dense_census_{dt}_d{d}_g{ratio}_kv{kv}_{sq}x{skv}_{mask}{win}_"
            f"{layout}"
        )
        case = _mkb(
            cid, reqs=(1, 15), tier="full", dtype=dt, d=d, b=1, h_q=ratio * kv,
            h_k=kv, s_q=sq, s_kv=skv, layout=layout, rdna=False, **m,
        )  # fmt: skip
        if not validate_bwd_case(case):
            out.append(case)
    return out


def _tail_cases():
    out = []
    for (sq, skv), (dt, d, mask) in zip(
        _TAIL_LENGTHS,
        (
            ("bf16", 128, "tl"),
            ("fp16", 64, "br"),
            ("bf16", 64, "none"),
            ("fp16", 128, "tl"),
            ("bf16", 128, "br"),
        ),
    ):
        case = _mkb(
            f"kern_dense_tail_{sq}x{skv}_{mask}_{dt}_d{d}", reqs=(8, 15),
            tier="smoke", dtype=dt, d=d, b=2, h_q=4, h_k=2, s_q=sq, s_kv=skv,
            layout="bshd", rdna=False, **_mask(mask),
        )  # fmt: skip
        problems = validate_bwd_case(case)
        if problems:
            raise AssertionError(f"{case.id}: {problems}")
        out.append(case)
    return out


def dense_cases(arch, tier=None):
    """Dense-eligible cases on ``arch`` (selected tier, sharded)."""
    tier = tier or os.environ.get("ROCKE_BWD_TIER", "smoke")
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    pool = list(select_bwd(tier)) + [s for s in SUPPLEMENTARY_CASES if tier in s.tags]
    pool += _tail_cases()
    if tier != "smoke":
        pool += _census_cases()
    out, seen = [], set()
    for c in pool:
        if c.id in seen or shard_of(c.id, n) != i:
            continue
        seen.add(c.id)
        if dense_bwd_eligibility(request_from_case(c), arch)[0]:
            out.append(c)
    return out


def _params():
    if ARCH not in ("gfx942", "gfx950"):
        return [pytest.param(None, id="no-device")]
    return [pytest.param(c, id=c.id) for c in dense_cases(ARCH)]


def _dense_run(case, *, spec=None, runtime=None, request=None):
    req = request_from_case(case)
    if request:
        req = dataclasses.replace(req, **request)
    pol = config_policy("default", ARCH)
    plan = attn_dense_bwd_plan(req, ARCH, policy=pol, spec=spec, runtime=runtime)

    def use_dense(_general):
        return plan

    return run_case(case, arch=ARCH, config="default", mutate_plan=use_dense)


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity
    assert run.plan.route == "dense"
    assert [s.stage for s in run.plan.launches][:2] == ["prep", "dense"]


_DEFAULT: dict = {}


@pytest.mark.parametrize("case", _params())
def test_dense_case(case):
    run = _dense_run(case)
    _passes(run)
    _DEFAULT[case.id] = run
    # the general twin on the same body: edge tiles on, no head packing (the
    # general plan packs short query spans, another summation order)
    general = run_case(
        case,
        arch=ARCH,
        config="default",
        overrides={"edge_tiles": True, "head_pack": False},
    )
    body = run.plan.specs["dense"].body_spec().resolved(ARCH)
    gspec = general.plan.specs["main"]
    assert not gspec.head_pack
    for f in ("waves", "block_m", "block_n", "block_k4", "atom_g02", "atom_g13"):
        assert getattr(gspec, f) == getattr(body, f), f
    assert np.array_equal(run.dk, general.dk), "dK differs from the general family"
    assert np.array_equal(run.dv, general.dv), "dV differs from the general family"


# Variant knobs: (id, spec fields, runtime values, request fields)
_VARIANTS = (
    ("clamp", dict(oob="clamp"), None, None),
    ("epilogue_direct", dict(epilogue="direct"), None, None),
    ("epilogue_pair", dict(epilogue="lds_staged_pair"), None, None),
    ("token_major", dict(ws_layout="token_major"), None, None),
    ("no_edge_tiles", dict(edge_tiles=False), None, None),
    ("persist_kv", dict(persistent=True), dict(num_persistent=7), None),
    (
        "persist_hkv",
        dict(persistent=True, persist_decode="hkv_major"),
        dict(num_persistent=5),
        None,
    ),
    ("xcd", dict(xcd_remap=True), dict(xcd_chunk="gx"), dict(num_xcds=4)),
    ("reverse", {}, dict(lb_order="reverse"), None),
    (
        "interleave",
        dict(body_knobs={"lb_interleave": True}),
        dict(lb_order="interleave"),
        None,
    ),
    ("static_scale_runtime", dict(body_knobs={"static_scale": "runtime"}), None, None),
    ("noalias_workspace", dict(body_knobs={"ptr_noalias": "workspace"}), None, None),
    ("pinned_batch", dict(batch="b"), None, None),
    # body levers built apart from the dense shell (bit for bit dK / dV)
    ("barrier_merge", dict(body_knobs={"barrier_merge": "on"}), None, None),
    ("counted", dict(body_knobs={"loop_sync": "counted"}), None, None),
    (
        "stage_table_atomics",
        dict(body_knobs={"sched": "stage_table_atomics"}),
        None,
        None,
    ),
    ("xt_lds", dict(body_knobs={"transpose_source": "xt_lds"}), None, None),
    ("stats_vec4", dict(body_knobs={"stats_vec": 4}), None, None),
    (
        "g4_slice_db",
        dict(body_knobs={"g4_slicing": "slice_db", "block_k4": 32}),
        None,
        None,
    ),
)
_VARIANT_CASES = (
    "bwd_kern_dense_tail_200x200_tl_bf16_d128",
    "bwd_kern_dense_tail_136x72_br_fp16_d64",
)


_SKIP_SHARD = pytest.mark.skip(reason="no dense device, or not shard 0")


def _first_shard() -> bool:
    """The knob variants run once per sharded suite (in shard 0)."""
    return os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/")[0] == "0"


def _variant_params():
    if ARCH not in ("gfx942", "gfx950") or not _first_shard():
        return [pytest.param(None, None, id="no-device-or-shard", marks=_SKIP_SHARD)]
    cases = {c.id: c for c in _tail_cases()}
    return [
        pytest.param(cases[cid], v, id=f"{cid}-{v[0]}")
        for cid in _VARIANT_CASES
        for v in _VARIANTS
    ]


@pytest.mark.parametrize("case,variant", _variant_params())
def test_dense_variants_agree(case, variant):
    _vid, fields, runtime, request = variant
    if case.id not in _DEFAULT:
        run = _dense_run(case)
        _passes(run)
        _DEFAULT[case.id] = run
    base = _DEFAULT[case.id]
    req = request_from_case(case)
    if request:
        req = dataclasses.replace(req, **request)
    fields = dict(fields)
    if fields.get("batch") == "b":
        fields["batch"] = case.b
    spec = dataclasses.replace(dense_bwd_spec_for(req, ARCH), **fields)
    runtime = dict(runtime or {})
    if runtime.get("xcd_chunk") == "gx":
        runtime["xcd_chunk"] = base.plan.launches[1].grid[0]
    run = _dense_run(case, spec=spec, runtime=runtime, request=request)
    _passes(run)
    assert np.array_equal(run.dk, base.dk), "dK differs across a bitwise knob"
    assert np.array_equal(run.dv, base.dv), "dV differs across a bitwise knob"


@pytest.mark.parametrize(
    "case_id",
    (
        [pytest.param(c, id=c) for c in _VARIANT_CASES]
        if ARCH in ("gfx942", "gfx950") and _first_shard()
        else [pytest.param(None, id="no-device-or-shard", marks=_SKIP_SHARD)]
    ),
)
def test_dense_atomic_g_split(case_id):
    case = {c.id: c for c in _tail_cases()}[case_id]
    req = request_from_case(case)
    spec = dataclasses.replace(dense_bwd_spec_for(req, ARCH), dkv_mode="atomic")
    run = _dense_run(case, spec=spec, runtime={"g_split": 2})
    _passes(run)
    assert [s.stage for s in run.plan.launches].count("convert") == 3
