# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The shipped per-arch tuned tables of the attention backward and their gate.

CPU only (plans, no device):

* the entry check (:mod:`dispatch.attention_bwd.tuned_tables`) plans an
  entry on representative requests of its key: an unknown knob, an illegal
  value or a value that needs a device hint the key does not carry is
  refused with the plan's reason, a valid entry passes;
* every shipped entry (gfx942 and gfx950, general and dense) passes it;
* routing takes the tuned configuration for requests of a tuned key on each
  arch (the dense route for a packed dense key), and keeps the defaults for
  requests of an untuned key.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention_bwd import gfx942_tuning, gfx950_tuning, routing
from dispatch.attention_bwd import tuned_tables as tt
from dispatch.attention_bwd.dense_tuning import dense_axis
from kernels.common.attention_bwd_plan import (
    FEATURE_GROUPS,
    AttnBwdPolicy,
    AttnBwdRequest,
    _dense_key,
    attn_bwd_plan,
    attn_bwd_tuning_key,
)
from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

MODS = {"gfx942": gfx942_tuning, "gfx950": gfx950_tuning}
_GATE = AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True)


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=4096, s_kv=4096, d_qk=128, d_v=128)
    base.update(dtype="bf16", causal=True, num_cus=64)
    base.update(kw)
    return AttnBwdRequest(**base)


def _routed():
    """The shipped routed policy (joined per-arch tables, dense pack) with the
    gate's waiver, so every class plans."""
    return dataclasses.replace(
        routing.routed_policy(),
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
    )


def _hook(knobs):
    values = dict(knobs)
    return dataclasses.replace(_GATE, tuning=lambda *_a: dict(values))


# ------------------------------------------------------------------ the check


@pytest.mark.parametrize("arch", sorted(MODS))
def test_general_check_refuses_an_unknown_knob(arch):
    key = attn_bwd_tuning_key(_req(), arch)
    # refused by the registry check before anything is planned
    with pytest.raises(ValueError, match="'bogus' is not a registered knob"):
        MODS[arch].check_general_entry(key, {"bogus": 1})
    with pytest.raises(ValueError, match="not a registered knob"):
        MODS[arch].general_entries([{"key": list(key), "knobs": {"bogus": 1}}])


@pytest.mark.parametrize("arch", sorted(MODS))
def test_general_check_refuses_an_illegal_value(arch):
    key = attn_bwd_tuning_key(_req(), arch)
    with pytest.raises(ValueError, match="lb_order='sideways'"):
        MODS[arch].check_general_entry(key, {"lb_order": "sideways"})
    with pytest.raises(ValueError, match="global_path"):
        MODS[arch].check_general_entry(key, {"global_path": "nope"})


@pytest.mark.parametrize("arch", sorted(MODS))
def test_general_check_refuses_a_value_the_key_does_not_fix(arch):
    """A g_split that does not divide every head group of a GQA class, and
    mirror pairing on a fill bucket whose lower edge does not overfill the
    device, plan on some requests of the key but not on all."""
    gqa = attn_bwd_tuning_key(_req(h_k=2, h_v=2), arch)
    assert gqa[5] == "gqa"
    assert attn_bwd_plan(_req(h_k=2, h_v=2), arch, policy=_hook({"g_split": 2}))
    with pytest.raises(ValueError, match="must divide the head group"):
        MODS[arch].check_general_entry(gqa, {"g_split": 2})
    full = attn_bwd_tuning_key(_req(num_cus=2 * 8 * 32), arch)
    assert full[-1] == "full"
    with pytest.raises(ValueError, match="mirror pairing"):
        MODS[arch].check_general_entry(full, {"lb_order": "mirror"})


@pytest.mark.parametrize("arch", sorted(MODS))
def test_general_check_passes_a_valid_entry(arch):
    key = attn_bwd_tuning_key(_req(), arch)
    MODS[arch].check_general_entry(key, {"edge_tiles": True})
    doc = {"entries": [{"key": list(key), "knobs": {"edge_tiles": True}}]}
    assert MODS[arch].general_entries(doc) == {key: {"edge_tiles": True}}
    reqs = tt.general_key_requests(arch, key)
    assert reqs and all(attn_bwd_tuning_key(r, arch) == key for r in reqs)


def test_general_check_keeps_only_requests_that_plan_without_the_entry(monkeypatch):
    """A representative the gate policy declines without the entry (here a
    deterministic request: the key does not carry the flag) is skipped, not
    held against the entry; a key none of whose representatives plans is
    refused."""
    key = attn_bwd_tuning_key(_req(), "gfx950")
    reps = tt.general_key_requests("gfx950", key)
    declined = dataclasses.replace(reps[0], deterministic=True)
    assert attn_bwd_tuning_key(declined, "gfx950") == key
    with pytest.raises(ValueError, match="DETERMINISTIC"):
        attn_bwd_plan(declined, "gfx950", policy=_GATE)
    monkeypatch.setattr(tt, "general_key_requests", lambda *_a: reps + [declined])
    gfx950_tuning.check_general_entry(key, {"edge_tiles": True})
    monkeypatch.setattr(tt, "general_key_requests", lambda *_a: [declined])
    with pytest.raises(ValueError, match="no plannable representative"):
        gfx950_tuning.check_general_entry(key, {"edge_tiles": True})


def test_general_check_refuses_malformed_keys_and_knobs():
    key = attn_bwd_tuning_key(_req(), "gfx950")
    for bad in (key[:-1], ("gfx942",) + key[1:], key[:1] + ("dense",) + key[2:]):
        with pytest.raises(ValueError, match="key"):
            gfx950_tuning.check_general_entry(bad, {})
    with pytest.raises(ValueError, match="not in"):
        gfx950_tuning.check_general_entry(key[:4] + ("diagonal",) + key[5:], {})
    with pytest.raises(ValueError, match="name -> value"):
        gfx950_tuning.check_general_entry(key, {1: 2})


def _dense_key_of(arch):
    req = _req(strides=None)
    return (arch, _dense_key(req)), req


@pytest.mark.parametrize("arch", sorted(MODS))
def test_dense_check_plans_without_a_request(arch):
    """The dense check no longer needs the caller's request: it plans the
    entry on the key's representatives."""
    key, req = _dense_key_of(arch)
    with pytest.raises(ValueError, match="unknown"):
        MODS[arch].check_dense_entry(key, {"bogus": 1})
    with pytest.raises(ValueError):
        MODS[arch].check_dense_entry(key, {"oob": "nope"})
    with pytest.raises(ValueError, match="xcd_chunk needs num_xcds"):
        MODS[arch].check_dense_entry(key, {"xcd_chunk": "gx"})
    MODS[arch].check_dense_entry(key, {"edge_tiles": True})
    MODS[arch].check_dense_entry(key, {"edge_tiles": True}, req)
    reqs = tt.dense_key_requests(arch, key[1])
    assert reqs and all(_dense_key(r) == key[1] for r in reqs)


# ---------------------------------------------------------- shipped entries


def _entries(table_name):
    out = []
    for arch, mod in sorted(MODS.items()):
        for key, knobs in getattr(mod, table_name).items():
            out.append(pytest.param(arch, key, knobs, id=f"{arch}-{key!r}"))
    return out


@pytest.mark.parametrize("arch,key,knobs", _entries("GENERAL_TUNED_TABLE"))
def test_shipped_general_entry_plans_on_its_key(arch, key, knobs):
    MODS[arch].check_general_entry(key, knobs)
    assert knobs.get("sched") != "iglp1"


@pytest.mark.parametrize("arch,key,knobs", _entries("DENSE_TUNED"))
def test_shipped_dense_entry_plans_on_its_key(arch, key, knobs):
    MODS[arch].check_dense_entry(key, knobs)
    # registered dense values (the import-time pass's registry check)
    tt.check_dense_entry(arch, key, knobs, axis_of=dense_axis)
    assert knobs.get("sched") != "iglp1"


def test_both_arches_ship_tuned_keys_and_gfx950_dense_keys():
    for mod in MODS.values():
        assert mod.GENERAL_TUNED_TABLE
    assert gfx950_tuning.DENSE_TUNED
    pack = routing.routed_policy().dense_pack
    for mod in MODS.values():
        assert set(mod.DENSE_TUNED) <= pack


# ------------------------------------------------------------------ routing


def _first_plannable(arch, reqs, policy):
    for r in reqs:
        try:
            attn_bwd_plan(r, arch, policy=policy)
        except ValueError:
            continue
        return r
    raise AssertionError("no plannable representative")


@pytest.mark.parametrize("arch", sorted(MODS))
def test_routing_takes_the_tuned_general_config(arch):
    """For a representative request of every tuned general key, the routed
    plan is the plan of the entry's knobs (and differs from the defaults on
    at least one key)."""
    pol = _routed()
    differs = 0
    for key, knobs in MODS[arch].GENERAL_TUNED_TABLE.items():
        req = _first_plannable(arch, tt.general_key_requests(arch, key), _GATE)
        assert attn_bwd_tuning_key(req, arch) == key
        plan = routing.plan_attention_bwd(req, arch, policy=pol)
        if routing.attention_bwd_route(req, arch, policy=pol)[0] == "dense":
            continue  # a packed dense key wins over the general entry
        assert plan.route == "general"
        want = attn_bwd_plan(req, arch, policy=_hook(knobs))
        assert plan.specs == want.specs, key
        assert plan.runtime == want.runtime, key
        differs += plan.specs != attn_bwd_plan(req, arch, policy=_GATE).specs
    assert differs


@pytest.mark.parametrize("arch", sorted(MODS))
def test_routing_takes_the_tuned_dense_config(arch):
    pol = _routed()
    for key, knobs in MODS[arch].DENSE_TUNED.items():
        for req in tt.dense_key_requests(arch, key[1], (None, 1024)):
            assert routing.attention_bwd_route(req, arch, policy=pol)[0] == "dense"
            plan = routing.plan_attention_bwd(req, arch, policy=pol)
            assert plan.route == "dense"
            want = attn_dense_bwd_plan(req, arch, policy=_hook(knobs))
            assert plan.specs == want.specs, key
            # a forced general route keeps the general family
            forced = routing.plan_attention_bwd(req, arch, policy=pol, route="general")
            assert forced.route == "general"


@pytest.mark.parametrize("arch", sorted(MODS))
def test_untuned_keys_keep_the_defaults(arch):
    """A request whose tuning key is in no table (here: no num_cus, the
    ``unknown`` fill bucket the tables never key; a head size no entry has)
    plans exactly as without the tables."""
    pol = _routed()
    tuned = set(MODS[arch].GENERAL_TUNED_TABLE)
    for req in (
        _req(num_cus=None),
        _req(num_cus=None, h_k=2, h_v=2),
        _req(d_qk=64, d_v=64, s_q=1000, s_kv=3000, causal=False),
        _req(dtype="fp16", h_k=4, h_v=2, num_cus=None),
    ):
        key = attn_bwd_tuning_key(req, arch)
        assert key not in tuned
        assert routing.attention_bwd_route(req, arch, policy=pol)[0] == "general"
        got = routing.plan_attention_bwd(req, arch, policy=pol)
        assert got.specs == attn_bwd_plan(req, arch, policy=_GATE).specs


def test_a_tuned_shape_without_num_cus_keeps_the_defaults():
    """The tables key the fill bucket the measured cells had on the device;
    the same shape without ``num_cus`` is fill ``unknown`` and untuned."""
    pol = _routed()
    for arch, mod in sorted(MODS.items()):
        key = next(iter(mod.GENERAL_TUNED_TABLE))
        req = _first_plannable(arch, tt.general_key_requests(arch, key), _GATE)
        bare = dataclasses.replace(req, num_cus=None)
        assert attn_bwd_tuning_key(bare, arch)[-1] == "unknown"
        if routing.attention_bwd_route(bare, arch, policy=pol)[0] != "general":
            continue
        got = routing.plan_attention_bwd(bare, arch, policy=pol)
        assert got.specs == attn_bwd_plan(bare, arch, policy=_GATE).specs


def test_rdna_targets_have_no_tables():
    assert set(routing.ARCH_TUNING) == {"gfx942", "gfx950"}
    pol = _routed()
    for arch in ("gfx1151", "gfx1201"):
        assert not any(k[0] == arch for k in pol.tuned_table)
        assert not any(k[0] == arch for k in pol.dense_pack)
