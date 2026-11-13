# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 tuned tables of the attention backward and their plumbing.

CPU only: the routed policy joins the per-arch general tables and status,
the dense table joins the per-arch dense tables; the sweep's emitted dense
entries become checked table entries (gfx950, the key's shape, knobs that
plan on the key), and a general entry reaches the plan through the routed
policy. The shipped entries themselves are checked in
``test_tuned_tables.py``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from dispatch.attention_bwd import gfx942_tuning
from dispatch.attention_bwd import gfx950_tuning as g
from dispatch.attention_bwd import routing
from dispatch.attention_bwd.dense_tuning import DENSE_BWD_TUNED
from kernels.common.attention_bwd_plan import (
    AttnBwdRequest,
    attn_bwd_tuning_key,
    shipped_policy,
)


def test_tables_are_registered_and_joined():
    assert g.ARCH == "gfx950"
    assert routing.ARCH_TUNING == {"gfx942": gfx942_tuning, "gfx950": g}
    assert g.GENERAL_TUNED_TABLE and g.DENSE_TUNED
    assert all(k[:2] == ("gfx950", "general") for k in g.GENERAL_PERF_STATUS)
    assert all(k[:2] == ("gfx950", "dense") for k in g.DENSE_PERF_STATUS)
    assert all(k[:2] == ("gfx950", "general") for k in g.GENERAL_TUNED_TABLE)
    assert all(k[0] == "gfx950" for k in g.DENSE_TUNED)
    base = shipped_policy()
    pol = routing.routed_policy()
    joined = routing.arch_tuned_policy(base)
    for mod in routing.ARCH_TUNING.values():
        for key, knobs in mod.GENERAL_TUNED_TABLE.items():
            assert pol.tuned_table[key] == knobs == joined.tuned_table[key]
        assert set(mod.DENSE_TUNED) <= set(DENSE_BWD_TUNED) <= pol.dense_pack
        for key, knobs in mod.DENSE_TUNED.items():
            assert DENSE_BWD_TUNED[key] == knobs
    assert len(DENSE_BWD_TUNED) == sum(
        len(m.DENSE_TUNED) for m in routing.ARCH_TUNING.values()
    )
    ref = routing.dense_policy(joined)
    for f in ("tuned_table", "perf_status", "dense_pack", "declared_groups"):
        assert getattr(pol, f) == getattr(ref, f), f
    # the per-arch status tables join the policy's status (information only)
    status = {}
    for mod in routing.ARCH_TUNING.values():
        status.update(mod.GENERAL_PERF_STATUS)
        status.update(mod.DENSE_PERF_STATUS)
    assert pol.perf_status == {**status, **base.perf_status}


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=256, s_kv=256, d_qk=128, d_v=128)
    base.update(kw)
    return AttnBwdRequest(dtype="bf16", causal=True, **base)


class _Mod:
    ARCH = "gfx950"

    def __init__(self, table=None, status=None):
        self.GENERAL_TUNED_TABLE = table or {}
        self.GENERAL_PERF_STATUS = status or {}
        self.DENSE_PERF_STATUS = {}
        self.DENSE_TUNED = {}


def _gate_policy():
    """A policy that admits every declared feature group and unverified
    classes (the gate runs' policy), so the request plans on any target."""
    from kernels.common.attention_bwd_plan import FEATURE_GROUPS, AttnBwdPolicy

    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True
    )


def test_general_entry_reaches_the_plan(monkeypatch):
    """A gfx950 general entry joins the routed policy's tuned table (a base
    entry of the same key wins) and plans: the request takes its knobs."""
    req = _req()
    key = attn_bwd_tuning_key(req, "gfx950")
    g.check_general_entry(key, {"edge_tiles": True})
    base = _gate_policy()

    def plan(arch):
        pol = routing.dense_policy(routing.arch_tuned_policy(base))
        return routing.plan_attention_bwd(req, arch, policy=pol)

    assert plan("gfx950").specs["main"].edge_tiles is not True
    other0 = plan("gfx942").specs["main"]
    monkeypatch.setattr(
        routing, "ARCH_TUNING", {"gfx950": _Mod({key: {"edge_tiles": True}})}
    )
    assert plan("gfx950").specs["main"].edge_tiles is True
    assert plan("gfx942").specs["main"] == other0  # another arch: untouched
    # the shipped routed policy joins the per-arch table too
    assert routing.routed_policy().tuned_table[key] == {"edge_tiles": True}
    # a base entry of the same key wins
    pinned = dataclasses.replace(base, tuned_table={key: {"edge_tiles": False}})
    assert routing.arch_tuned_policy(pinned).tuned_table[key] == {"edge_tiles": False}


def test_general_entries_are_checked():
    req = _req()
    key = attn_bwd_tuning_key(req, "gfx950")
    good = {"edge_tiles": True}
    doc = {"entries": [{"key": json.loads(json.dumps(key)), "knobs": good}]}
    assert g.general_entries(doc) == {key: good}
    for bad in (
        attn_bwd_tuning_key(req, "gfx942"),  # another arch
        ("gfx950", "dense") + key[2:],  # another route
        ("gfx950",),
    ):
        with pytest.raises(ValueError):
            g.check_general_entry(bad, {})
    # the knobs are planned on the key: a wave count without its warp grid
    # does not plan, and the plan's reason is the error
    with pytest.raises(ValueError, match="warp_grid_g4"):
        g.general_entries([{"key": list(key), "knobs": {"waves": 4}}])
    with pytest.raises(ValueError, match="two entries"):
        g.general_entries(doc["entries"] + [{"key": list(key), "knobs": {}}])


def _dense_doc():
    """A dense table document as the sweep emits it (``--emit-dense-table``)
    for one gfx950 key, and the key's request."""
    from benchmarks.common import attention_bwd_sweep as sw
    from kernels.common.attention_bwd_plan import _dense_key

    sl = sw.primary_slice("gfx950", "dense", 128)
    gsl = sw.primary_slice("gfx950", "general", 128)
    knobs = dict(sw.default_knobs(gsl), global_path="dma", ring_depth=2)
    c = sw.Candidate.make(sl, knobs, {"g_split": 2, "route": "dense"})
    cell = sw.dense_cells(sl, sw.representative_cells(sl), 128)[0]
    req = sw.cell_request(cell)
    key = _dense_key(req)
    entry = {
        "arch": "gfx950",
        "key": key,
        "knobs": sw.dense_tuned_value(c),
        "cells": [cell.id],
    }
    return json.loads(json.dumps({"entries": [entry]})), key, req


def test_dense_entries_from_the_sweep_plan_on_their_keys():
    doc, key, req = _dense_doc()
    table = g.dense_entries(doc, {key: req})
    assert list(table) == [("gfx950", key)]
    (knobs,) = table.values()
    assert knobs["ring_depth"] == 2 and knobs["g_split"] == 2
    # the routing takes the entry as its dense route (dense pack and knobs)
    pol = routing.dense_policy(_gate_policy(), table=table)
    plan = routing.plan_attention_bwd(req, "gfx950", policy=pol)
    assert plan.route == "dense"
    body = plan.specs["dense"].body_spec().resolved("gfx950")
    assert (body.global_path, body.ring_depth) == ("dma", 2)


def test_dense_entries_are_checked():
    doc, key, req = _dense_doc()
    entry = doc["entries"][0]
    for bad, match in (
        ({**entry, "arch": "gfx942"}, "not a gfx950"),
        ({**entry, "key": entry["key"][:3]}, "not a dense key"),
        ({**entry, "knobs": {**entry["knobs"], "oob": "nope"}}, None),
    ):
        with pytest.raises(ValueError, match=match):
            g.dense_entries([bad], {key: req})
    with pytest.raises(ValueError, match="is not"):  # the request of another key
        other = dataclasses.replace(req, s_q=req.s_q * 2)
        g.dense_entries(doc, {key: other})
    with pytest.raises(ValueError, match="two entries"):
        twin = {**entry, "knobs": {**entry["knobs"], "g_split": 1}}
        g.dense_entries([entry, twin])
