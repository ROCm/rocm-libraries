#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the SDPA backward bundle generator and its plan module.

They check that the checked-in SdpaBackward sweeps and the generated block
of ASM_SDPA_ENGINE.toml are what the generator writes, that the generator is
deterministic, that every written case is legal and traceable to the plan,
that the cases up to nightly cover every required pair of axis values, and
that every case id resolves to exactly one tolerance row."""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import generate_sdpa_bwd_bundles as G  # noqa: E402
import sdpa_bwd_plan as P  # noqa: E402

BUNDLES = SCRIPTS.parent / "integration-test-bundles"
ENGINE_TOML = (
    SCRIPTS.parents[1] / "hip-kernel-provider" / "config" / "ASM_SDPA_ENGINE.toml"
)
NUMPY_MISSING = importlib.util.find_spec("numpy") is None


@pytest.fixture(scope="module")
def cases():
    return G.build(G.Calibration(P.CALIBRATION_FILE))


@pytest.fixture(scope="module")
def written(cases):
    return [c for c in cases if c["runnable_now"]]


@pytest.fixture(scope="module")
def files(cases):
    return G.write_tree(cases, BUNDLES, dry_run=True)


def _sweep_cases(files):
    for rel, text in files.items():
        if rel.name == "sweep.json":
            level, _, topo = rel.parts[:3]
            for e in json.loads(text)["cases"]:
                yield level, topo, e


# --------------------------------------------------------------------------
# Determinism and the checked-in tree
# --------------------------------------------------------------------------


def test_generation_is_deterministic(files):
    again = G.write_tree(
        G.build(G.Calibration(P.CALIBRATION_FILE)), BUNDLES, dry_run=True
    )
    assert again == files


def test_checked_in_sweeps_are_generated(files):
    """The tree holds exactly what the generator writes: no hand edits, no
    stale topology directory."""
    for rel, text in files.items():
        path = BUNDLES / rel
        assert path.is_file(), f"{rel} is not checked in; rerun the generator"
        assert (
            path.read_text() == text
        ), f"{rel} differs from the generator output; rerun the generator"
    owned = {"graph.template.json", "sweep.json", "support.json"}
    on_disk = {
        p.relative_to(BUNDLES)
        for p in BUNDLES.glob("*/SdpaBackward/*/*")
        if p.name in owned
    }
    assert on_disk == set(
        files
    ), f"stale generator files: {sorted(on_disk - set(files))}"


def test_engine_toml_block_is_generated():
    text = ENGINE_TOML.read_text()
    i, j = text.index(G.TOML_BEGIN), text.index(G.TOML_END)
    assert text[i : j + len(G.TOML_END) + 1] == G.toml_block()


def test_calibration_has_no_stale_entries():
    calib = G.Calibration(P.CALIBRATION_FILE)
    G.build(calib)
    assert set(calib.entries) == calib.used


# --------------------------------------------------------------------------
# Written cases
# --------------------------------------------------------------------------


def test_notrun_cases_are_not_written(cases, files):
    ids = {e["id"] for _, _, e in _sweep_cases(files)}
    assert ids == {c["case_id"] for c in cases if c["runnable_now"]}
    for c in cases:
        if not c["runnable_now"]:
            assert c["level"] == P.NOTRUN and c["oracle_status"] in P.GATES, c["id"]
            assert c.get("target_level"), c["id"]


def test_every_written_case_is_legal(written):
    for c in written:
        rec = c["record"]
        assert P.legality_violations(rec) == [], (
            c["case_id"],
            P.legality_violations(rec),
        )
        layer = P.layer(rec, "develop")[0]
        assert layer == "U" if c.get("rejection") else layer in ("WF", "EXT"), (
            c["case_id"],
            layer,
        )


def test_sweep_values_match_the_case(written, files):
    by_id = {c["case_id"]: c for c in written}
    for level, topo, e in _sweep_cases(files):
        c = by_id[e["id"]]
        assert (topo, level) == (c["topology"], P.LEVEL_DIR[c["level"]])
        rec = c["record"]
        tensors = {t["uid"]: t for t in e["values"]["tensors"]}
        assert set(tensors) == {t["uid"] for t in rec["strides"].values()}
        for t in rec["strides"].values():
            got = tensors[t["uid"]]
            assert (got["dims"], got["strides"], got["data_type"]) == (
                t["dims"],
                t["strides"],
                t["data_type"],
            )
        attrs = e["values"]["attributes"]
        lb, rb = P.bounds_written(c)
        assert (
            attrs["parameters__left_bound"] == lb
            and attrs["parameters__right_bound"] == rb
        )
        assert attrs["parameters__diagonal_alignment"] in ("TOP_LEFT", "BOTTOM_RIGHT")
        meta = e["metadata"]
        assert (
            meta["format_version"] == 1
            and meta["generator"]
            and meta["reference_source"]
        )
        assert meta["purpose"] in ("claim", "probe", "regression")
        assert meta["size_class"] in ("S", "L", "M", "H", "X")
        assert e["id"].endswith("_" + P.TOKEN[meta["tol_class"]])
        assert {int(k) for k in meta["inputs"]} <= set(tensors) - {
            P.UID["O"],
            P.UID["Stats"],
        }
        assert len(e["id"]) <= G.ID_CAP and re.fullmatch(r"[a-z0-9_]+", e["id"])


def test_templates_cover_every_topology():
    for topo in P.TOPOLOGIES:
        tpl = G.template(topo)
        uids = [t["uid"] for t in tpl["tensors"]]
        assert len(uids) == len(set(uids)) == len(G.template_tensors(topo))
        attrs = {p for p in G.placeholders(tpl) if p.startswith("attributes.")}
        expected = {"attributes.compute_data_type"} | {
            "attributes.parameters__" + k for k in G.BAND_PARAMETERS
        }
        assert attrs == expected | (
            {"attributes.parameters__padding_mask"} if topo == "Padding" else set()
        )
    scale = [
        t for t in G.template("ScaleTensor")["tensors"] if t["uid"] == P.UID["Scale"]
    ][0]
    assert scale["value"] == G.SCALE_CONST and not scale["is_runtime_pass_by_value"]


def test_window_semantics_switch_keeps_ids(written, monkeypatch):
    window = [
        c
        for c in written
        if c.get("W") is not None
        and not (c.get("attrs") or {}).get("explicit_unbounded")
    ]
    assert window
    for c in window:
        lb = P.bounds_written(c)[0]
        monkeypatch.setattr(P, "WINDOW_SEMANTICS", "after_sem001")
        assert P.bounds_written(c)[0] == lb + 1 == c["W"]
        assert G.case_id(c, c["dtype"], c["tol_class"]) == c["case_id"]
        monkeypatch.setattr(P, "WINDOW_SEMANTICS", "develop")


# --------------------------------------------------------------------------
# Coverage, tolerance rows and traceability
# --------------------------------------------------------------------------


def test_all_pairs_up_to_nightly(written):
    cs = [
        (c["axis_values"], c["purpose_final"])
        for c in written
        if c["level"] != "full" and c["axis_values"]
    ]
    r = P.pairwise_check(cs)
    assert r["required"] > 0 and r["uncovered"] == [], r["uncovered"][:5]
    assert r["claim_uncovered"] == [], r["claim_uncovered"][:5]
    assert r["forbidden"] == []


def _toml_rows(text):
    rows = []
    for block in text.split("[[")[1:]:
        if not block.startswith("tolerance_overrides]]"):
            continue
        filters = re.findall(
            r'"([^"]+)"', re.search(r"filters\s*=\s*\[([^\]]*)\]", block).group(1)
        )
        atol = float(re.search(r"^atol\s*=\s*(\S+)", block, re.M).group(1))
        rtol = float(re.search(r"^rtol\s*=\s*(\S+)", block, re.M).group(1))
        rows.append((filters, atol, rtol))
    return rows


def test_every_case_id_matches_exactly_one_toml_row(cases):
    rows = _toml_rows(ENGINE_TOML.read_text())
    assert rows
    for c in cases:
        name = G.gtest_name(c)
        hits = [r for r in rows if any(fnmatch.fnmatchcase(name, f) for f in r[0])]
        assert len(hits) == 1, (name, hits)
        assert (hits[0][1], hits[0][2]) == P.tolerance(c["tol_class"], c["dtype"]), name
    others = [
        "quick_SdpaFwd_Default.c01_bfp16_none_told",
        "standard_SdpaBackward_Default_fwd_pair_gl01_bfp16_none_told_Small.Small",
        "quick_SdpaBackward_bhsd_bf16_hd128_causal_Small.Small",
    ]
    assert not [
        n for n in others for r in rows if any(fnmatch.fnmatchcase(n, f) for f in r[0])
    ]


def test_test_skips_match_written_cases(written):
    """Every SdpaBackward [[test_skips]] filter of the engine TOML still
    selects a written case (no stale skip)."""
    text = ENGINE_TOML.read_text()
    names = [G.gtest_name(c) for c in written]
    for block in text.split("[[test_skips]]")[1:]:
        for f in re.findall(r'"([^"]*SdpaBackward[^"]*)"', block.split("[[")[0]):
            assert any(fnmatch.fnmatchcase(n, f) for n in names), f


def test_plan_items_are_traceable(cases):
    items = G.plan_items()
    item_ids = {c["id"] for c in cases}
    shapes = (
        set(P.S_SHAPES)
        | set(P.H_SHAPES)
        | set(P.X_SHAPES)
        | set(P.L_SHAPES)
        | set(P.M_MODELS)
    )
    shapes |= {s for v in G.MERGED.values() for s in v.split()}
    tags = {
        "AP",
        "TP-A",
        "TP-B",
        "TP-C",
        "TP-D",
        "TP-E",
        "G-A",
        "G-B",
        "G-C",
        "G-D",
        "G-E",
        "G-F",
    }
    known = set(P.FEATURES) | item_ids | shapes | tags
    covered = set()
    for c in cases:
        own = c["record"]["plan_items"]
        assert c["id"] in own
        unknown = [
            i
            for i in own
            if i not in known and not re.fullmatch(r"(PRB|REG)-[A-Z]*\d+[a-z]?", i)
        ]
        assert unknown == [], (c["id"], unknown)
        covered |= set(own)
    missing = [
        f for f in P.FEATURES if f not in covered and f not in P.FEATURE_DISPOSITION
    ]
    assert missing == []
    for carrier, shapes_ in G.MERGED.items():
        it = next(i for i in items if i["id"] == carrier)
        assert set(shapes_.split()) <= set(it["items"]), carrier


def test_ceilings_hold(written):
    for c in written:
        assert P.budget_seconds(c["est_s"], c["est_s_alt"]) <= P.CEIL_S[c["level"]], c[
            "case_id"
        ]
        cap = P.CEIL_BYTES[c["level"]]
        assert cap is None or c["record"]["bytes"] <= cap, c["case_id"]


def test_cumulative_budgets_hold(written):
    total = 0.0
    for lv in P.LEVELS:
        total += sum(
            P.budget_seconds(c["est_s"], c["est_s_alt"])
            for c in written
            if c["level"] == lv
        )
        assert total <= P.CUM_S[lv], (lv, total)


# --------------------------------------------------------------------------
# Plan module
# --------------------------------------------------------------------------


def test_kept_pairs_matches_brute_force():
    rng = random.Random(7)
    for _ in range(2000):
        Sq, Skv = rng.randint(1, 40), rng.randint(1, 40)
        W, R = rng.choice([None] + list(range(1, 45))), rng.choice(
            [None] + list(range(0, 45))
        )
        al = rng.choice(["TL", "BR"])
        off = 0 if al == "TL" else Skv - Sq
        brute = sum(
            1
            for i in range(Sq)
            for j in range(Skv)
            if (R is None or j <= i + off + R) and (W is None or j >= i + off - W + 1)
        )
        assert P.kept_pairs(Sq, Skv, W, R, al) == brute, (Sq, Skv, W, R, al)


def test_model_reduction_fits_its_target():
    for r in P.model_variants():
        if (
            r["id"].endswith((".pr", ".n"))
            and r["oracle"] == "ready"
            and r["id"].split(".")[0] not in P.M_VARIANTS
        ):
            target = (
                P.M_PR_TARGET_S if r["id"].endswith(".pr") else P.M_NIGHTLY_TARGET_S
            )
            assert P._m_cost(r)["s"] <= target, r["id"]


@pytest.mark.skipif(NUMPY_MISSING, reason="the magnitude estimator needs numpy")
def test_estimator_reproduces_the_calibration(cases):
    calib = P.load_calibration()
    items = {it["id"]: it for it in G.plan_items()}
    sample = [
        c for c in cases if f"{c['id']}/{c['dtype']}" in calib and c["n_pairs"] < 50000
    ][:6]
    assert sample
    for c in sample:
        stored = calib[f"{c['id']}/{c['dtype']}"]
        km = P.kernel_model(c, c["dtype"], c["runnable_now"])
        unscaled = (
            items[c["id"]]["inputs"] if c["id"] in items else c["inputs"]
        )  # before the dO scaling
        params = P.estimate_params(dict(c, inputs=unscaled), c["dtype"], km)
        assert P.fingerprint(params) == stored["fp"], c["id"]
        got = P.estimate(params)
        assert got["lambda"] == pytest.approx(stored["lambda"], rel=1e-9)
        for g in ("dQ", "dK", "dV"):
            assert got["max_ref"][g] == pytest.approx(
                stored["max_ref"][g], rel=1e-6, abs=1e-12
            )
        if stored["atol_needed"] is None:
            assert got["atol_needed"] is None
        else:  # the kernel model's FP32 sums may round differently across BLAS builds
            assert got["atol_needed"] == pytest.approx(stored["atol_needed"], rel=1e-2)
