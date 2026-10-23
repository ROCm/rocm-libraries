# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Known-wrong backward knob values (``BWD_KNOWN_WRONG_KNOBS``), CPU only.

* the shipped mapping is explicit and empty on every declared target (no
  backward knob value is known to compute wrong values: every such kernel
  was fixed or is refused exactly by its validator);
* a value made known wrong (for the test) is held out of every sweep level of
  the general, dense and unified registries, is illegal for the legality
  verdicts of the three families, is refused by the tuned-table check and
  declines the dispatcher's candidate (``test_registry.py``); another target
  is untouched;
* a default value can never be known wrong.
"""

from __future__ import annotations

import pytest

from dispatch.attention_bwd import dense_tuning, tuned_tables, unified_tuning
from dispatch.attention_bwd import tuning_common as tc
from kernels.common.attention_bwd import AttnBwdSpec
from kernels.common.attention_bwd_plan import DECLARED_ARCHS, AttnBwdRequest
from kernels.common.attention_bwd_unified import AttnBwdUnifiedSpec


def test_shipped_mapping_is_explicit_and_empty():
    assert set(tc.BWD_KNOWN_WRONG_KNOBS) == set(DECLARED_ARCHS)
    for arch in DECLARED_ARCHS:
        assert dict(tc.BWD_KNOWN_WRONG_KNOBS[arch]) == {}
    tc.check_known_wrong({a.name: a for a in tc.BWD_KNOB_AXES})


def _mark(monkeypatch, arch, knob, *values):
    monkeypatch.setattr(tc, "BWD_KNOWN_WRONG_KNOBS", {arch: {knob: frozenset(values)}})


@pytest.mark.parametrize("level", tc.SWEEP_LEVELS)
def test_held_out_of_every_level_and_registry(monkeypatch, level):
    for registry, name, value in (
        (tc.backward_axis, "edge_tiles", True),
        (tc.backward_axis, "lb_order", "reverse"),
        (dense_tuning.dense_axis, "persistent", True),
        (dense_tuning.dense_axis, "oob", "clamp"),
        (unified_tuning.unified_axis, "sub_page", 64),
        (unified_tuning.unified_axis, "page_tile", "k64"),
    ):
        ax = registry(name)
        before = ax.values("gfx950", level)
        assert value in before, (name, value)
        _mark(monkeypatch, "gfx950", name, value)
        # exactly the marked value goes; every other value stays
        assert ax.values("gfx950", level) == tuple(v for v in before if v != value)
        assert value in ax.values("gfx942", level), (name, value)
        monkeypatch.undo()


def test_general_legality_refuses(monkeypatch):
    spec = AttnBwdSpec(head_size=128, edge_tiles=True)
    assert tc.backward_spec_legality(spec, "gfx950").status == "legal"
    _mark(monkeypatch, "gfx950", "edge_tiles", True)
    got = tc.backward_spec_legality(spec, "gfx950")
    assert got.status == "illegal" and got.known_wrong == ("edge_tiles=True",)
    assert "known wrong on gfx950" in got.error
    assert tc.backward_spec_legality(spec, "gfx942").status == "legal"
    knobs = tc.backward_spec_legality({"edge_tiles": True}, "gfx950", head_size=64)
    assert knobs.status == "illegal"


def test_dense_legality_refuses(monkeypatch):
    from kernels.common.attention_dense_bwd_plan import dense_bwd_spec_class

    cls = dense_bwd_spec_class("gfx950")
    spec = cls(
        seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=8,
        head_size=128, oob="clamp",
    )  # fmt: skip
    assert dense_tuning.dense_spec_legality(spec, "gfx950") == "legal"
    _mark(monkeypatch, "gfx950", "oob", "clamp")
    assert dense_tuning.dense_spec_legality(spec, "gfx950") == "illegal"
    assert dense_tuning.dense_known_wrong(
        spec, spec.body_spec().resolved("gfx950"), "gfx950"
    ) == ("oob='clamp'",)
    # a value of the resolved body counts too
    body_default = spec.body_spec().resolved("gfx950").kt_source
    monkeypatch.undo()
    plain = cls(
        seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=8,
        head_size=128,
    )  # fmt: skip
    assert dense_tuning.dense_spec_legality(plain, "gfx950") == "legal"
    _mark(monkeypatch, "gfx950", "kt_source", body_default)
    assert dense_tuning.dense_spec_legality(plain, "gfx950") == "illegal"


def test_unified_legality_refuses(monkeypatch):
    spec = AttnBwdUnifiedSpec(head_size=128, sub_page=64)
    assert unified_tuning.unified_spec_legality(spec, "gfx950") == "legal"
    _mark(monkeypatch, "gfx950", "sub_page", 64)
    assert unified_tuning.unified_spec_legality(spec, "gfx950") == "illegal"
    assert unified_tuning.unified_spec_legality(spec, "gfx942") == "legal"


def test_tuned_table_check_refuses(monkeypatch):
    from kernels.common.attention_bwd_plan import attn_bwd_tuning_key

    req = AttnBwdRequest(
        b=2, h_q=8, h_k=8, h_v=8, s_q=256, s_kv=256, d_qk=128, d_v=128,
        dtype="bf16", causal=True,
    )  # fmt: skip
    key = attn_bwd_tuning_key(req, "gfx950")
    tuned_tables.check_general_entry("gfx950", key, {"edge_tiles": True}, limit=1)
    _mark(monkeypatch, "gfx950", "edge_tiles", True)
    with pytest.raises(ValueError, match="known wrong on gfx950: edge_tiles=True"):
        tuned_tables.check_general_entry("gfx950", key, {"edge_tiles": True}, limit=1)


def test_a_default_is_never_known_wrong(monkeypatch):
    axes = {a.name: a for a in tc.BWD_KNOB_AXES}
    _mark(monkeypatch, "gfx950", "edge_tiles", axes["edge_tiles"].default)
    with pytest.raises(AssertionError, match="default is known wrong"):
        tc.check_known_wrong(axes)
    monkeypatch.setattr(tc, "BWD_KNOWN_WRONG_KNOBS", {"gfx90a": {}})
    with pytest.raises(AssertionError, match="undeclared target"):
        tc.check_known_wrong(axes)
    monkeypatch.setattr(tc, "BWD_KNOWN_WRONG_KNOBS", {"gfx950": {"waves": {4}}})
    with pytest.raises(AssertionError, match="frozenset"):
        tc.check_known_wrong(axes)
