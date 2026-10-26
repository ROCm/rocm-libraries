# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The BURS generator (`tools/isa_burs.py`): patterns parse, rule costs are the
description's issue cycles times the unit weights, the checked-in selector is
what the grammar generates, and the isel.rs hooks apply to their anchors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM / "tools"))

import isa_burs
import isa_tables

CRATE = PLATFORM / "rust" / "rocke-codeobj"


def load_desc():
    return isa_tables.load_desc(isa_tables.Tree(CRATE))


def test_patterns_parse_into_terminals_and_variables():
    p = isa_burs.parse_pattern("add(shl(x:v,k:konst),y:v)")
    assert p.term == "add" and p.kids[0].term == "shl"
    found: list[tuple[str, str]] = []
    isa_burs.pattern_vars(p, found)
    assert found == [("x", "v"), ("k", "konst"), ("y", "v")]
    with pytest.raises(isa_tables.DescError):
        isa_burs.parse_pattern("add(x:v")


def test_rule_costs_are_issue_cycles_times_unit_weight():
    desc = load_desc()
    grammar = isa_burs.parse_grammar(desc)
    weight = {
        r.args[0]: int(r.args[1]) for r in desc.section("burs") if r.kind == "weight"
    }
    unit_issue = {r.args[0]: int(r.get("issue") or 1) for r in desc.section("unit")}
    insts = {r.args[0]: r for r in desc.insts}
    patterns = [r for r in grammar.rules if r.kind == "pattern"]
    assert patterns
    for rule in patterns:
        rec = insts[rule.inst]
        issue = int(rec.get("issue") or unit_issue[rec.get("unit")])
        assert rule.cost == issue * weight.get(rec.get("unit"), 1), rule.text
    assert all(r.cost == 0 for r in grammar.rules if r.kind != "pattern")


def test_checked_in_selector_is_what_the_grammar_generates():
    text = (CRATE / isa_tables.GEN_BURS).read_bytes().decode().replace("\r\n", "\n")
    assert isa_burs.generate_burs(load_desc()) == text


def test_isel_hooks_apply_to_their_anchors():
    text = (
        isa_burs.MOD_OLD
        + "\n}\n"
        + isa_burs.FIELD_OLD
        + "    f: MFunc,\n}\n"
        + "fn select() {\n"
        + isa_burs.INIT_OLD
        + "    };\n"
        + isa_burs.PLAN_OLD
        + "}\n"
        + "fn op() {\n"
        + isa_burs.HOOK_OLD
        + "}\n"
    )
    for name, done, apply in isa_burs.isel_hunks():
        assert done not in text, name
        text = apply(text)
        assert text is not None and done in text, name
