#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the convolution backward-weights bundle generator.

They check that the checked-in ConvolutionWrw sweeps are what the generator
writes (nobody edits them by hand), that every case is legal and traceable to
the plan, and that the fill recipes of a case are keyed by the uids of the
operands the plan states them for: dy is uid 1 and x is uid 2 in the wrw graph,
and the dgrad graph numbers its operands differently (there uid 2 is w), so a
recipe table copied from there would fill x with the recipe of a weight tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import conv_wrw_plan as P  # noqa: E402
import generate_conv_wrw_bundles as G  # noqa: E402

BUNDLES = SCRIPTS.parent / "integration-test-bundles"


@pytest.fixture(scope="module")
def built():
    result, _ = G.build()
    return result


def test_checked_in_bundles_are_what_the_generator_writes(built, tmp_path):
    files = G.write_tree(built, tmp_path, dry_run=True)
    for rel, text in files.items():
        if rel.name == "support.json":
            continue  # claims are harvested from runs, never generated
        on_disk = (BUNDLES / rel).read_text(encoding="utf-8")
        assert (
            on_disk == text
        ), f"{rel} differs from the generator output; regenerate it"


def test_generator_is_deterministic():
    first, _ = G.build()
    second, _ = G.build()
    ids = lambda r: {
        lv: [cid for _, cid in items] for lv, items in r.items()
    }  # noqa: E731
    assert ids(first) == ids(second)


def test_every_case_is_legal_and_ids_are_unique(built):
    seen = set()
    for level, items in built.items():
        for c, cid in items:
            assert P.legal(c) is None, f"{level}/{cid}"
            assert cid == cid.lower() and " " not in cid
            assert (level, cid) not in seen
            seen.add((level, cid))
    assert {lv: len(i) for lv, i in built.items()}["quick"] > 0


def test_recipes_follow_the_operand_not_the_position(built):
    """The recipe of dy lands on the dy uid and the recipe of x on the x uid, for
    every value class. ``zero`` sets dy alone, so a table keyed by the wrong uid
    shows as a zero-filled x (dw would then be all zero for every dy)."""
    template = G.template()
    node = template["nodes"][0]
    dy_uid = node["inputs"]["dy_tensor_uid"]
    x_uid = node["inputs"]["x_tensor_uid"]
    checked = set()
    for items in built.values():
        for c, cid in items:
            case = G.sweep_case(c, cid)
            vals = next(
                (i[3:] for i in case["metadata"]["plan_items"] if i.startswith("F21")),
                "uniform",
            )
            inputs = case["metadata"]["inputs"]
            assert set(inputs) == {str(dy_uid), str(x_uid)}, case["id"]
            dy, x = inputs[str(dy_uid)], inputs[str(x_uid)]
            if vals == "zero":
                assert dy["kind"] == "fixed" and dy["value"] == 0.0, case["id"]
                assert x["kind"] == "free", case["id"]
            elif vals == "const":
                assert dy == {**dy, "kind": "fixed", "value": 1.0}, case["id"]
                assert x["kind"] == "fixed" and x["value"] == 1.0, case["id"]
            checked.add(vals)
    assert {"zero", "const", "positive", "large", "tiny", "uniform"} <= checked


def test_value_classes_do_not_overflow(built):
    """Non-cancelling classes must keep the largest possible dw element (every
    product at the recipe's maximum) inside the data type."""
    for items in built.values():
        for c, cid in items:
            rec = P.value_recipes(c)
            if rec is None or c["vals"] not in ("const", "positive", "large"):
                continue
            hi = lambda spec: spec[1] if spec[0] == "fixed" else spec[2]  # noqa: E731
            peak = hi(rec["dy"]) * hi(rec["x"]) * P.accum_len(c)
            assert peak <= P.DTYPE_MAX[c["xt"]], cid


def test_quick_stays_a_smoke_set(built):
    assert len(built["quick"]) <= 10
    for c, cid in built["quick"]:
        assert P.purpose(c) == "claim", cid
        assert P.fits(c, "quick"), cid
