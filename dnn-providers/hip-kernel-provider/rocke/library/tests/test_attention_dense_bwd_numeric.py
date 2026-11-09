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

Split dQ (``dq_mode = "split"``: the dense main kernel computes dK / dV only,
the dense dQ kernel dQ): every variant (rings, head splits, both OOB modes,
the dQ kernel's edge / interior loops, head packing, key prefetch, row tile
and chiplet remap, the best gfx950 tiles) passes the oracle and gives dQ, dK
and dV bit for bit equal to the general family's split run; dQ is
deterministic; dead rows get exactly zero dQ, and a dQ kernel without the
dead-row select (negative control) does not.

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


def _general_twin(case, **kw):
    """The general-route twin run of a dense run (which has already passed
    the oracle). A twin whose main kernel the validator refuses for its
    resources (RESOURCE_REFUSED: it fails the resource gate on the
    production compiler) is not run: the test skips the bitwise comparison
    with the validator's reason, after the dense run's oracle check."""
    try:
        return run_case(case, **kw)
    except ValueError as exc:
        if "RESOURCE_REFUSED" not in str(exc):
            raise
        pytest.skip(
            f"the dense run passed the oracle; its general twin is refused: {exc}"
        )


@pytest.mark.parametrize("case", _params())
def test_dense_case(case):
    run = _dense_run(case)
    _passes(run)
    _DEFAULT[case.id] = run
    # the general twin on the same body: edge tiles on, no head packing (the
    # general plan packs short query spans, another summation order)
    general = _general_twin(
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


# The Q / dO prefetch ring on the dense shell: (id, body knobs, dense spec
# fields, runtime values, arches). Each runs through the dense hook path (the
# plan derives atomic dK / dV from g_split and drops head_pack = False) and
# must match its general twin (the same body knobs and runtime values, edge
# tiles as the dense run, no head packing) bit for bit in dK / dV: the ring
# changes where Q / dO are staged, never the math or its order (two head
# splits add two partials, exact in either order).
_BOTH, _950 = ("gfx942", "gfx950"), ("gfx950",)
_WIN_A = {
    "atom_g02": "16x16x32", "atom_g13": "16x16x16", "atom_g4": "16x16x32",
    "block_k4": 32, "block_m": 32, "block_n": 128, "pt_route": "relabel",
    "transpose_source": "tr_read", "warp_grid_g02": (1, 8),
    "warp_grid_g13": (8, 1), "warp_grid_g4": (2, 4), "waves": 8,
    "waves_per_eu": 1, "stats_vec": 4, "stats_fma": True,
}  # fmt: skip
_DMA2 = {"global_path": "dma", "ring_depth": 2}
_RING_VARIANTS = (
    ("vgpr2", {"ring_depth": 2}, {}, {}, _BOTH),
    ("vgpr2_clamp", {"ring_depth": 2}, {"oob": "clamp"}, {}, _BOTH),
    ("vgpr2_g2", {"ring_depth": 2}, {}, {"g_split": 2}, _BOTH),
    (
        "vgpr2_persist",
        {"ring_depth": 2},
        {"persistent": True},
        {"num_persistent": 7},
        _BOTH,
    ),
    ("vgpr3", {"ring_depth": 3}, {}, {}, _950),
    ("dma1", {"global_path": "dma"}, {}, {}, _950),
    ("dma2", _DMA2, {}, {}, _950),
    ("dma3", {"global_path": "dma", "ring_depth": 3}, {}, {}, _950),
    ("dma2_clamp", _DMA2, {"oob": "clamp"}, {}, _950),
    ("dma2_g2", _DMA2, {}, {"g_split": 2}, _950),
    ("dma2_xcd", _DMA2, {}, {"xcd_chunk": "gx", "num_xcds": 4}, _950),
    (
        "dma2_winA_stats_g2",
        dict(_WIN_A, **_DMA2),
        {"edge_tiles": False},
        {"g_split": 2},
        _950,
    ),
)
_RUNTIME_HINTS = ("num_xcds",)


def _ring_cases():
    """The two tail (ragged) variant cases plus aligned (non-ragged) cases:
    the global-pointer view of buffer_oob does not clamp there."""
    cases = {c.id: c for c in _tail_cases()}
    out = [cases[c] for c in _VARIANT_CASES]
    out.append(
        _mkb(
            "kern_dense_ring_256x256_tl_bf16_d128", reqs=(8, 15), tier="smoke",
            dtype="bf16", d=128, b=2, h_q=4, h_k=2, s_q=256, s_kv=256,
            layout="bshd", rdna=False, **_mask("tl"),
        )
    )  # fmt: skip
    out.append(
        _mkb(
            "kern_dense_ring_128x384_br_fp16_d64_bhsd", reqs=(8, 15), tier="smoke",
            dtype="fp16", d=64, b=2, h_q=4, h_k=2, s_q=128, s_kv=384,
            layout="bhsd", rdna=False, **_mask("br"),
        )
    )  # fmt: skip
    for c in out[2:]:
        problems = validate_bwd_case(c)
        if problems:
            raise AssertionError(f"{c.id}: {problems}")
    return out


def _ring_params():
    if ARCH not in ("gfx942", "gfx950") or not _first_shard():
        return [pytest.param(None, None, id="no-device-or-shard", marks=_SKIP_SHARD)]
    out = []
    for case in _ring_cases():
        for v in _RING_VARIANTS:
            if ARCH not in v[4] or (v[0].startswith("dma2_winA") and case.d != 128):
                continue
            out.append(pytest.param(case, v, id=f"{case.id}-{v[0]}"))
    return out


@pytest.mark.parametrize("case,variant", _ring_params())
def test_dense_ring_matches_the_general_twin(case, variant):
    _vid, body, fields, runtime, _archs = variant
    req = request_from_case(case)
    hints = {k: runtime[k] for k in _RUNTIME_HINTS if k in runtime}
    tuning = {k: v for k, v in runtime.items() if k not in _RUNTIME_HINTS}
    if hints:
        req = dataclasses.replace(req, **hints)
    knobs = {**body, **fields, **tuning, "head_pack": False}
    plan = attn_dense_bwd_plan(req, ARCH, policy=config_policy("default", ARCH, knobs))
    spec = plan.specs["dense"]
    resolved = spec.body_spec().resolved(ARCH)
    for k, v in body.items():
        assert getattr(resolved, k) == v, k
    assert (spec.dkv_mode == "atomic") == (runtime.get("g_split", 1) > 1)
    run = run_case(case, arch=ARCH, config="default", mutate_plan=lambda _g: plan)
    _passes(run)
    general_knobs = {
        **body,
        **{k: v for k, v in tuning.items() if k != "num_persistent"},
        "edge_tiles": spec.edge_tiles,
        "head_pack": False,
    }
    general = _general_twin(
        case,
        arch=ARCH,
        config="default",
        overrides=general_knobs,
        request=hints or None,
    )
    gspec = general.plan.specs["main"]
    for f in ("waves", "block_m", "block_n", "global_path", "ring_depth"):
        assert getattr(gspec, f) == getattr(resolved, f), f
    assert np.array_equal(run.dk, general.dk), "dK differs from the general twin"
    assert np.array_equal(run.dv, general.dv), "dV differs from the general twin"


# Split dQ on the dense route (dq_mode = "split": the dense main kernel
# computes dK / dV only, the dense dQ kernel dQ): (id, body knobs, dense spec
# fields, runtime values, dQ kernel values, arches). Each runs through the
# dense hook path and must pass the oracle and the integrity checks, and its
# dK / dV and dQ must equal its general split twin's (the same body knobs,
# runtime and dQ values, edge tiles as the dense run, no head packing) bit for
# bit: the dense kernels change addressing and the band classification (the
# dQ kernel drops key loops proved empty, in the same key order), never the
# math or its order.
_W4_D64 = {
    "atom_g02": "16x16x32", "atom_g13": "16x16x16", "atom_g4": "16x16x32",
    "block_k4": 32, "block_m": 32, "block_n": 128, "pt_route": "relabel",
    "transpose_source": "tr_read", "warp_grid_g02": (1, 4),
    "warp_grid_g13": (4, 1), "warp_grid_g4": (2, 2), "waves": 4,
    "waves_per_eu": 2, "stats_vec": 4, "stats_fma": True, "bwd_exp2": "fast",
    "kv_residency": "reg", "kt_source": "reg",
}  # fmt: skip
_SPLIT_VARIANTS = (
    ("split", {}, {}, {}, {}, _BOTH),
    ("split_vgpr2", {"ring_depth": 2}, {}, {}, {}, _BOTH),
    ("split_vgpr2_g2", {"ring_depth": 2}, {}, {"g_split": 2}, {}, _BOTH),
    ("split_clamp_edge", {}, {"oob": "clamp"}, {}, {"dq_edge_tiles": True}, _BOTH),
    ("split_dq_edge", {}, {}, {}, {"dq_edge_tiles": True}, _BOTH),
    # the dQ kernel applies dq_mult (the convert multiplier folded in)
    ("split_convert", {}, {}, {"scale_placement": "convert"}, {}, _BOTH),
    (
        "split_dq_before_atomic",
        {},
        {},
        {"scale_placement": "dq_before_atomic"},
        {},
        _BOTH,
    ),
    (
        "split_dq_group_prefetch",
        {},
        {},
        {},
        {"dq_head_loop": "group", "dq_kv_prefetch": True},
        _BOTH,
    ),
    ("split_dq_bm128", {}, {}, {}, {"dq_block_m": 128, "dq_edge_tiles": True}, _BOTH),
    (
        "split_dq_xcd",
        {},
        {},
        {"num_xcds": 4},
        {"dq_xcd_chunk": "gx"},
        _BOTH,
    ),
    ("split_dma2", _DMA2, {}, {}, {}, _950),
    ("split_dma2_g2", _DMA2, {}, {"g_split": 2}, {"dq_edge_tiles": True}, _950),
    (
        "split_dma2_persist",
        _DMA2,
        {"persistent": True},
        {"num_persistent": 7},
        {},
        _950,
    ),
    (
        "split_dma2_winA_stats_g2",
        dict(_WIN_A, **_DMA2),
        {"edge_tiles": False},
        {"g_split": 2},
        {"dq_edge_tiles": True},
        _950,
    ),
    (
        "split_dma2_w4_d64",
        dict(_W4_D64, **_DMA2),
        {},
        {},
        {"dq_block_m": 128, "dq_edge_tiles": True},
        _950,
    ),
)


def _split_cases():
    """The ring cases plus an aligned unmasked d64 MHA case (every key tile
    interior: the dense dQ kernel runs its interior loop only)."""
    out = _ring_cases()
    case = _mkb(
        "kern_dense_split_256x256_none_bf16_d64_mha", reqs=(8, 15), tier="smoke",
        dtype="bf16", d=64, b=2, h_q=4, h_k=4, s_q=256, s_kv=256, layout="bshd",
        rdna=False, **_mask("none"),
    )  # fmt: skip
    problems = validate_bwd_case(case)
    if problems:
        raise AssertionError(f"{case.id}: {problems}")
    return out + [case]


def _split_params():
    if ARCH not in ("gfx942", "gfx950") or not _first_shard():
        return [pytest.param(None, None, id="no-device-or-shard", marks=_SKIP_SHARD)]
    out = []
    for case in _split_cases():
        for v in _SPLIT_VARIANTS:
            if ARCH not in v[5]:
                continue
            if v[0] == "split_dma2_winA_stats_g2" and case.d != 128:
                continue
            if v[0] == "split_dma2_w4_d64" and case.d != 64:
                continue
            if "g2" in v[0] and case.h_q == case.h_k:
                continue  # MHA: no head group to split
            out.append(pytest.param(case, v, id=f"{case.id}-{v[0]}"))
    return out


@pytest.mark.parametrize("case,variant", _split_params())
def test_dense_split_matches_the_general_split_twin(case, variant):
    _vid, body, fields, runtime, dq_knobs, _archs = variant
    req = request_from_case(case)
    hints = {k: runtime[k] for k in _RUNTIME_HINTS if k in runtime}
    tuning = {k: v for k, v in runtime.items() if k not in _RUNTIME_HINTS}
    if hints:
        req = dataclasses.replace(req, **hints)
    knobs = {**body, **fields, **tuning, **dq_knobs, "dq_mode": "split"}
    knobs["head_pack"] = False
    plan = attn_dense_bwd_plan(req, ARCH, policy=config_policy("default", ARCH, knobs))
    spec = plan.specs["dense"]
    assert spec.dq_mode == "split"
    stages = [s.stage for s in plan.launches]
    assert stages[:3] == ["prep", "dense", "dense_dq"] and "dq" not in stages
    assert plan.launches[0].scalars["zero_dq"] == 0
    run = run_case(case, arch=ARCH, config="default", mutate_plan=lambda _g: plan)
    _passes(run)
    general_knobs = {
        **body,
        **{k: v for k, v in tuning.items() if k != "num_persistent"},
        **dq_knobs,
        "dq_mode": "split",
        "edge_tiles": spec.edge_tiles,
        "head_pack": False,
    }
    general = _general_twin(
        case,
        arch=ARCH,
        config="default",
        overrides=general_knobs,
        request=hints or None,
    )
    gspec = general.plan.specs["main"]
    assert gspec.dq_mode == "split"
    resolved = spec.body_spec().resolved(ARCH)
    for f in ("waves", "block_m", "block_n", "global_path", "ring_depth"):
        assert getattr(gspec, f) == getattr(resolved, f), f
    assert general.plan.specs["dq"] == plan.specs["dense_dq"].dq
    assert np.array_equal(run.dk, general.dk), "dK differs from the general twin"
    assert np.array_equal(run.dv, general.dv), "dV differs from the general twin"
    assert np.array_equal(run.dq, general.dq), "dQ differs from the general twin"
    if _vid == "split":
        # deterministic: a second run gives the same dQ bits (no atomics)
        again = run_case(case, arch=ARCH, config="default", mutate_plan=lambda _g: plan)
        assert np.array_equal(run.dq, again.dq)


@pytest.mark.parametrize(
    "case_id",
    (
        [pytest.param(c, id=c) for c in _VARIANT_CASES]
        if ARCH in ("gfx942", "gfx950") and _first_shard()
        else [pytest.param(None, id="no-device-or-shard", marks=_SKIP_SHARD)]
    ),
)
def test_dense_split_dead_row_select_is_load_bearing(case_id):
    """Dead rows (LSE ``-inf``, a NaN Dsum as an ungated prep would leave it)
    get exactly zero dQ from the dense dQ kernel; the negative control built
    without the dead-row term of the dS select does not (so the check can see
    a missing sentinel)."""
    from unittest import mock

    from kernels.common import attention_bwd_run
    from kernels.common.attention_dense_bwd_dq import emit_dense_bwd_dq

    from ._attention_bwd_harness import dead_rows

    case = {c.id: c for c in _tail_cases()}[case_id]
    rows = dead_rows(case)
    req = request_from_case(case)
    pol = config_policy(
        "default", ARCH, {"dq_mode": "split", "head_pack": False, "dq_edge_tiles": True}
    )
    plan = attn_dense_bwd_plan(req, ARCH, policy=pol)
    assert plan.specs["dense"].ws_layout == "head_major"

    def poison_lse(values):
        out = values.copy()
        out.reshape(case.b, case.h_q, case.s_q)[rows] = -np.inf
        return out

    def ungated_dsum(ws, p):
        off, _ = p.workspace_layout["WS_DSUM"]
        n_rows = case.b * case.s_q
        dsum = ws[off // 4 : off // 4 + n_rows * case.h_q]
        for bi, h, q in np.argwhere(rows):
            dsum[h * n_rows + bi * case.s_q + q] = np.nan

    kw = dict(
        arch=ARCH, config="default", mutate_plan=lambda _g: plan,
        mutate_inputs=poison_lse, after_prep=ungated_dsum,
    )  # fmt: skip
    shipped = run_case(case, **kw)
    assert np.isfinite(shipped.dq).all()
    assert np.all(shipped.dq[rows] == 0.0), "dQ of dead rows is not exactly 0"

    def build(spec, *, arch):
        return emit_dense_bwd_dq(spec, arch, sentinel=False)

    build.cache_variant = "emit_dense_bwd_dq:sentinel=False"
    with mock.patch.dict(attention_bwd_run._BUILDERS, {"dense_dq": build}):
        control = run_case(case, cache={}, **kw)
    assert not np.isfinite(control.dq[rows]).all(), (
        "the negative control (no dead-row term) did not fire: the test cannot "
        "see a missing sentinel"
    )


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
