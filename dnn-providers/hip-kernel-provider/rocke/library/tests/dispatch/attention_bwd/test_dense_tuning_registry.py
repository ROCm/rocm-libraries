# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense attention backward tuning registry (CPU).

Every dense spec tuning field has one axis whose default is the field's
default; not-built values are registered and refused by the kernel; every
built value of every dense axis builds a legal spec on its arch; the body
knobs are the general registry's axes; every shipped tuned entry plans on a
census cell of its key.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention_bwd import (
    DENSE_BWD_KNOB_AXES,
    DENSE_BWD_TUNED,
    BWD_KNOB_AXES,
    dense_axis,
    dense_body_axes,
    dense_runtime_axes,
    dense_spec_legality,
    dense_tuning_axes,
)
from dispatch.attention_bwd.tuning_common import KIND_RUNTIME
from kernels.common.attention_dense_bwd_spec import DENSE_BODY_KNOBS
from kernels.gfx942.attention_dense_bwd import Gfx942AttentionDenseBwdSpec
from kernels.gfx950.attention_dense_bwd import Gfx950AttentionDenseBwdSpec

_SPEC = {"gfx942": Gfx942AttentionDenseBwdSpec, "gfx950": Gfx950AttentionDenseBwdSpec}
_BASE = dict(
    seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=2, head_size=128
)


def test_defaults_match_the_spec():
    fields = {
        f.name: f.default for f in dataclasses.fields(Gfx942AttentionDenseBwdSpec)
    }
    for ax in DENSE_BWD_KNOB_AXES:
        if ax.kind == KIND_RUNTIME:
            continue
        assert ax.default == fields[ax.name], ax.name


def test_body_axes_are_the_general_axes():
    names = {ax.name for ax in dense_body_axes()}
    assert names == set(DENSE_BODY_KNOBS)
    general = {ax.name: ax for ax in BWD_KNOB_AXES}
    for ax in dense_body_axes():
        assert ax is general[ax.name]
    assert dense_axis("waves") is general["waves"]
    assert dense_axis("oob").name == "oob"
    with pytest.raises(KeyError):
        dense_axis("head_size")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_every_built_value_is_legal_and_not_built_values_refused(arch):
    cls = _SPEC[arch]
    for ax in DENSE_BWD_KNOB_AXES:
        if ax.kind == KIND_RUNTIME or ax.plan_owned:
            continue
        for v in ax.values(arch, "full"):
            kw = dict(_BASE)
            kw[ax.name] = v
            if ax.name == "persist_decode" and v != "kv_tile_major":
                kw["persistent"] = True
            try:
                spec = cls(**kw)
            except ValueError:
                assert v in ax.not_built, (ax.name, v)
                continue
            got = dense_spec_legality(spec, arch)
            want = "not_built" if v in ax.not_built else "legal"
            assert got == want, (ax.name, v, got)
    spec = cls(**_BASE, dkv_mode="atomic")
    assert dense_spec_legality(spec, arch) == "legal"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_every_built_staging_value_is_legal_on_the_dense_shell(arch):
    """The Q / dO staging axes (global_path, ring_depth) of the shared body:
    every built value of the arch is a legal dense spec (the prefetch ring
    is built on the dense shell); the mirror pass is not built there."""
    cls = _SPEC[arch]
    n = 0
    for path in dense_axis("global_path").values(arch, "full"):
        for depth in dense_axis("ring_depth").values(arch, "full"):
            knobs = {"global_path": path, "ring_depth": depth}
            try:
                spec = cls(**_BASE, body_knobs=knobs)
                verdict = dense_spec_legality(spec, arch)
            except ValueError:
                continue
            if verdict == "illegal":
                continue
            assert verdict == "legal", (path, depth)
            n += 1
    assert n == {"gfx942": 2, "gfx950": 6}[arch]
    mirror = cls(**_BASE, body_knobs={"lb_mirror": True})
    assert dense_spec_legality(mirror, arch) == "not_built"


def _tuned_entry_plans(table):
    """Plan every ``DENSE_BWD_TUNED`` entry on a census cell of its key."""
    from benchmarks.common import attention_bwd_bench as bb
    from benchmarks.common import attention_bwd_rocke_arm as ra
    from benchmarks.common import attention_bwd_sweep as sw
    from kernels.common.attention_bwd_plan import AttnBwdPolicy, FEATURE_GROUPS
    from kernels.common.attention_bwd_plan import _dense_key, attn_bwd_route
    from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

    cells = [c for c in bb.census_cells(bb.load_census()) if not c.is_thd]
    by_key = {}
    for c in cells:
        try:
            by_key.setdefault(_dense_key(sw.cell_request(c)), c)
        except ValueError:
            continue
    out = []
    for (arch, key), knobs in table.items():
        cell = by_key[key]
        req = sw.cell_request(cell)
        rt = {k: v for k, v in knobs.items() if k not in ra.REQUEST_HINTS}

        def tuning(_r, _a, _route, _p, rt=rt):
            return dict(rt)

        pol = AttnBwdPolicy(
            declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True,
            tuning=tuning, dense_pack=frozenset({(arch, key)}),
        )  # fmt: skip
        assert attn_bwd_route(req, arch, policy=pol)[0] == "dense"
        out.append(attn_dense_bwd_plan(req, arch, policy=pol))
    return out


def test_tuned_entries_plan_on_their_keys():
    """Every shipped DENSE_BWD_TUNED entry plans on a census cell of its key;
    the check itself is exercised on an entry the sweep would emit."""
    from benchmarks.common import attention_bwd_sweep as sw
    from kernels.common.attention_bwd_plan import _dense_key

    plans = _tuned_entry_plans(DENSE_BWD_TUNED)
    assert len(plans) == len(DENSE_BWD_TUNED) > 0
    assert all(p.route == "dense" for p in plans)
    sl = sw.primary_slice("gfx950", "dense", 128)
    gsl = sw.primary_slice("gfx950", "general", 128)
    knobs = dict(sw.default_knobs(gsl), global_path="dma", ring_depth=2)
    c = sw.Candidate.make(sl, knobs, {"g_split": 2, "route": "dense"})
    cell = sw.dense_cells(sl, sw.representative_cells(sl), 128)[0]
    key = _dense_key(sw.cell_request(cell))
    value = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in sw.dense_tuned_value(c).items()
    }
    (plan,) = _tuned_entry_plans({("gfx950", key): value})
    spec = plan.specs["dense"]
    assert spec.dkv_mode == "atomic"
    body = spec.body_spec().resolved("gfx950")
    assert (body.global_path, body.ring_depth) == ("dma", 2)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_arch_values_of_the_body_variant(arch):
    vals = dense_axis("body_variant").values(arch)
    assert vals[0] == "general" and len(vals) == 2
    assert all(v in _SPEC[arch].BODY_VARIANTS for v in vals)


def test_tuning_axes_and_table():
    names = {ax.name for ax in dense_tuning_axes("gfx942", "full")}
    assert {"oob", "epilogue", "persistent", "waves", "block_n"} <= names
    assert {"g_split", "lb_order", "xcd_chunk", "scale_placement"} <= names
    # the joined per-arch tables: (arch, dense key) keys of the CDNA targets
    assert all(arch in ("gfx942", "gfx950") for arch, _key in DENSE_BWD_TUNED)
    assert all(len(key) == 9 for _arch, key in DENSE_BWD_TUNED)


def test_runtime_axes_are_the_general_runtime_axes():
    from kernels.common.attention_bwd_plan import AUX_KNOBS, RUNTIME_KNOBS

    general = {ax.name: ax for ax in BWD_KNOB_AXES}
    got = {ax.name for ax in dense_runtime_axes()}
    assert got == (set(RUNTIME_KNOBS) - {"use_worklist"}) | set(AUX_KNOBS)
    for ax in dense_runtime_axes():
        assert ax is general[ax.name]
        assert ax.kind == KIND_RUNTIME
    assert dense_axis("lb_order") is general["lb_order"]
    with pytest.raises(KeyError):
        dense_axis("use_worklist")
    # the dense remap is derived by the plan, like the general one
    assert dense_axis("xcd_remap").plan_owned
