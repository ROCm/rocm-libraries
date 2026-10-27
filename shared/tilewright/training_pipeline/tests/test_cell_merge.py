# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repair on refusal (lib/cell_merge.py): which per-cell gate findings it
repairs and up to how many cells, the order of its actions, the splits a
carried subtree revokes and the trees that honour them, the incumbent found
by content, the all-group check and the judgement, and installs and
withdrawals that an interruption anywhere leaves finishable."""

import json
import os
import types

import numpy as np
import pytest

import test_collapse_repair as tcr
from lib import audit as al
from lib import cell_merge as cm
from lib import collapse_repair as cr
from lib import run_bundles as rb
from lib import subcells as sc

BASE, OTHER = tcr.BASE, tcr.OTHER
THIRD = "Small|Small|MidK|Bnone"
FOURTH = "Tiny|Tiny|TinyK|Bnone"


def _gate(groups, passed=False):
    return {"passed": passed, "groups": groups}


def _cell(cell, ratio=0.97, p=0.01):
    return {"cell": cell, "n": 8, "ratio": ratio, "ratio_ci": [0.95, 0.99], "p": p}


def _tail(cell, shape=tcr.SHAPE, decides=True):
    return {
        "shape": list(shape),
        "cell": cell,
        "before": 0.9,
        "after": 0.3,
        "confirmed": True,
        "decides": decides,
    }


def _group(deciding=True, regressed=(), floor=(), tail=(), protocol="v2"):
    return {
        "protocol": protocol,
        "deciding": deciding,
        "regressed": list(regressed),
        "floor": list(floor),
        "tail": list(tail),
    }


# ── the findings a repair takes ─────────────────────────────────────────────


def test_the_flagged_cells_are_the_deciding_findings_by_base_cell():
    gate = _gate(
        {
            "exclusive": _group(
                regressed=[_cell(OTHER)],
                floor=[_cell(THIRD)],
                tail=[_tail(BASE), _tail(FOURTH, decides=False)],
            ),
            # a context's cells are reported, never decide
            "cu64_cot": _group(deciding=False, regressed=[_cell(FOURTH)]),
            # one GEMM collapsing in two contexts is one base cell
            "cu128_cot": _group(deciding=False, tail=[_tail(tcr.LEAF)]),
            "cu192_cot": {"protocol": "v2", "reason": "does not pair"},
        }
    )
    got = cm.flagged_cells(gate)
    assert sorted(got) == sorted([BASE, OTHER, THIRD])
    assert [f["kind"] for f in got[BASE]] == ["collapse", "collapse"]
    assert {f["group"] for f in got[BASE]} == {"exclusive", "cu128_cot"}
    assert got[OTHER] == [
        {
            "kind": "regressed",
            "group": "exclusive",
            "n": 8,
            "ratio": 0.97,
            "ratio_ci": [0.95, 0.99],
            "p": 0.01,
            "before": None,
            "after": None,
        }
    ]
    assert got[THIRD][0]["kind"] == "floor"
    assert cm.cell_rule_cells(got) == sorted([OTHER, THIRD])
    assert "collapsed 30.00% (was 90.00%)" in cm.finding_text(got[BASE][0])
    assert cm.finding_text(got[OTHER][0]) == "exclusive regressed x0.9700 p 0.01"
    assert cm.finding_text({"kind": "floor", "group": "g", "ratio": 0.9}) == (
        "g fell below the floor x0.9000"
    )
    lines = cm.flagged_lines(got)
    assert len(lines) == 3 and lines[0].startswith(f"{BASE}: exclusive ")
    assert "; cu128_cot " in lines[0]


def test_a_refusal_is_repaired_up_to_the_cap_and_refused_whole_above_it():
    two = _gate({"exclusive": _group(regressed=[_cell(OTHER)], tail=[_tail(BASE)])})
    ok, why, flagged = cm.repairable(two, 2)
    assert ok and sorted(flagged) == sorted([BASE, OTHER])
    assert why.startswith("2 base cell(s) flagged")
    ok, why, flagged = cm.repairable(two, 1)
    assert not ok and len(flagged) == 2
    assert why == (
        "2 base cells flagged, more than gate.repair_max_cells 1: refused whole"
    )
    assert cm.repairable(two, cm.DEFAULT_MAX_CELLS)[0]
    # the cap counts base cells: one GEMM collapsing in several groups, and
    # a leaf's collapse beside its base cell's, flag one cell
    one = _gate(
        {
            "exclusive": _group(
                tail=[_tail(BASE), _tail(tcr.LEAF, shape=(9, 9, 9, 1))]
            ),
            "cu64_cot": _group(deciding=False, tail=[_tail(BASE)]),
        }
    )
    ok, why, flagged = cm.repairable(one, 1)
    assert ok and list(flagged) == [BASE] and len(flagged[BASE]) == 3
    assert why == f"1 base cell(s) flagged: {BASE}"
    # only a refusal is repaired, and only for findings that name a cell
    assert cm.repairable({**two, "passed": True}, 3)[:2] == (
        False,
        "the per-cell gate did not keep it out",
    )
    assert not cm.repairable({**two, "passed": None}, 3)[0]
    assert not cm.repairable(None, 3)[0]
    assert cm.repairable(_gate({"exclusive": _group()}), 3)[1] == (
        "no finding of the per-cell gate decides"
    )
    nameless = _gate({"exclusive": _group(tail=[_tail("")])})
    assert cm.repairable(nameless, 3)[1] == (
        "a deciding finding of the per-cell gate names no cell"
    )


# ── the order of the actions ────────────────────────────────────────────────


def _run(cand, ship, gate, audit, **kw):
    return cr.repair(
        cand,
        ship,
        cr.collapses_of(gate),
        audit,
        pack=tcr.pack,
        leaf_data=tcr.leaf_data(),
        view_key_hash=tcr.view_key_hash,
        **kw,
    )


def test_a_collapse_gets_its_tier_key_and_a_cell_rule_finding_its_subtree():
    """The collapse in BASE's leaf is lifted by a tier key, which keeps the
    new networks there; OTHER, which the cell rules flagged, is carried with
    every label of its subtree from the incumbent."""
    cand = tcr._bundle({tcr.LEAF: [tcr.H_BAD], tcr.SIBLING: [tcr.H_GOOD]})
    ship = tcr._bundle({BASE: [tcr.H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    gate = tcr._gate(("cu128_cot", tcr.SHAPE, 0.9, 0.3))
    out = _run(cand, ship, gate, tcr.FakeAudit(), carry_also=[OTHER])
    rec = out.record
    assert rec["carried_for_cell_rules"] == [OTHER]
    assert rec["n_keys_added"] == 1 and list(rec["carried_cells"]) == [OTHER]
    assert out.bundle["models"][tcr.LEAF]["view_key_hashes"] == [
        tcr.H_BAD,
        tcr.H_GOOD,
    ]
    assert out.bundle["models"][OTHER] is ship["models"][OTHER]
    assert "(1 for a cell rule's finding)" in rec["outcome"]
    flagged = {BASE: [{"kind": "collapse"}], OTHER: [{"kind": "regressed"}]}
    acts = cm.cell_actions(flagged, rec, "c" * 64, "i" * 64)
    assert acts[BASE]["action"] == "tier add"
    assert acts[BASE]["origin_sha256"] == "c" * 64
    assert acts[BASE]["tier_keys_added"] == {tcr.LEAF: [tcr.H_GOOD]}
    assert acts[OTHER]["action"] == "subtree carry"
    assert acts[OTHER]["origin_sha256"] == "i" * 64


def test_a_cell_flagged_by_both_rules_is_carried_and_its_tier_key_dropped():
    cand = tcr._bundle({tcr.LEAF: [tcr.H_BAD], tcr.SIBLING: [tcr.H_GOOD]})
    ship = tcr._bundle({BASE: [tcr.H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    gate = tcr._gate(("cu128_cot", tcr.SHAPE, 0.9, 0.3))
    out = _run(cand, ship, gate, tcr.FakeAudit(), carry_also=[BASE])
    rec = out.record
    (tried,) = rec["tier_adds"][tcr.LEAF]["tried"]
    assert tried["kept"] is True
    assert rec["tier_adds_dropped_by_carry"] == {tcr.LEAF: [tcr.H_GOOD]}
    assert rec["carried_cells"] == {
        BASE: {"removed": [tcr.LEAF, tcr.SIBLING], "carried": [BASE]}
    }
    assert sorted(out.bundle["models"]) == [BASE, OTHER]
    assert cr.kept_tier_adds(rec) == {}


def test_cell_rule_findings_alone_carry_their_cells():
    cand = tcr._bundle({tcr.LEAF: [tcr.H_BAD], tcr.SIBLING: [tcr.H_GOOD]})
    ship = tcr._bundle({BASE: [tcr.H_GOOD]}, labels=(BASE, tcr.LEAF, OTHER), tag="ship")
    audit = tcr.FakeAudit()
    out = _run(cand, ship, {"groups": {}}, audit, carry_also=[tcr.LEAF])
    assert audit.calls == []
    assert out.record["carried_cells"] == {
        BASE: {"removed": [tcr.LEAF, tcr.SIBLING], "carried": [BASE, tcr.LEAF]}
    }
    assert out.record["outcome"] == (
        "0 tier key(s) added, 1 base cell(s) carried from the bundle that ships "
        "for a cell rule's finding"
    )
    assert sorted(out.bundle["models"]) == [BASE, tcr.LEAF, OTHER]
    missing = tcr._bundle({}, labels=(OTHER,), tag="ship")
    out = _run(cand, missing, {"groups": {}}, audit, carry_also=[BASE])
    assert out.bundle is None and out.record["outcome"].startswith("cannot repair")


def test_the_repair_summary_names_cells_origins_and_revocations():
    rec = {
        "outcome": "o",
        "cells": {BASE: {"action": "subtree carry"}},
        "origins": {"candidate": "c", "incumbent": "i"},
        "revoked_splits": [{"cell": tcr.LEAF}],
        "barred_cells": [BASE],
    }
    s = cr.summary(rec)
    assert s["cells"] == rec["cells"] and s["origins"] == rec["origins"]
    assert s["revoked_splits"] == rec["revoked_splits"]
    assert s["barred_cells"] == [BASE]
    assert "cells" not in cr.summary({"outcome": "o"})


# ── revoked splits ───────────────────────────────────────────────────────────

SPLIT = sc.build_split_rule(BASE, "M", 1024)
DEEP = sc.build_split_rule(tcr.LEAF, "K", 64)
OTHER_SPLIT = sc.build_split_rule(OTHER, "N", 300)


def test_a_carried_subtree_revokes_the_splits_the_incumbent_lacks():
    tree = {r.cell: r for r in (SPLIT, DEEP, OTHER_SPLIT)}
    # the incumbent serves BASE unsplit: both splits under BASE go
    got = cm.revoked_rules(tree, [BASE, OTHER], [BASE])
    assert [r.cell for r in got] == [BASE, tcr.LEAF]
    # the incumbent has the first split: only the deeper one goes
    got = cm.revoked_rules(tree, [tcr.LEAF, tcr.SIBLING], [BASE])
    assert [r.cell for r in got] == [tcr.LEAF]
    # nothing outside the carried cells, nothing when the subtrees agree
    leaves = [sc.make_subcell_label(tcr.LEAF, "K", "<=", 64), tcr.SIBLING]
    assert cm.revoked_rules(tree, leaves, [BASE]) == []
    # an incumbent split the run's tree does not hold is refused
    other = [sc.make_subcell_label(BASE, "M", "<=", 2048)]
    with pytest.raises(ValueError, match="the incumbent splits"):
        cm.revoked_rules(tree, other, [BASE])
    with pytest.raises(ValueError, match="has no split there"):
        cm.revoked_rules({}, [tcr.LEAF], [BASE])


def _round(run, k, splits=(), revoked=None, barred=()):
    rd = run / f"round_{k}"
    sc.write_splits_json(rd / "stage04b" / "splits.json", k, list(splits))
    if revoked is not None:
        sc.write_revocations(rd / "stage05" / sc.REVOKED_SPLITS, revoked, barred)
    return rd


def test_a_round_on_a_final_retrain_inherits_its_revocations(tmp_path):
    run = tmp_path / "run"
    r0, r1 = _round(run, 0, [SPLIT]), _round(run, 1)
    final = run / rb.FINAL_RETRAIN_DIR
    sc.write_revocations(final / "stage05" / sc.REVOKED_SPLITS, [SPLIT], [BASE])
    r2 = run / "round_2"
    (r2 / "stage05").mkdir(parents=True)
    assert cm.inherit_revocations(r2, r1, [r0, r1]) is False
    assert not (r2 / "stage05" / sc.REVOKED_SPLITS).exists()
    assert cm.inherit_revocations(r2, final, [r0, r1]) is True
    doc = sc.revocations(r2)
    assert [r["cell"] for r in doc["revoked"]] == [BASE]
    assert doc["barred_cells"] == [BASE] and doc["inherited_from"] == "final_retrain"
    assert sc.load_cumulative_split_tree([r0, r1, r2]) == {}
    # taken once: the same call again writes nothing
    text = (r2 / "stage05" / sc.REVOKED_SPLITS).read_text()
    assert cm.inherit_revocations(r2, final, [r0, r1]) is False
    assert (r2 / "stage05" / sc.REVOKED_SPLITS).read_text() == text
    # a round whose own repair revoked a split keeps it and its record's fields
    r3 = run / "round_3"
    sc.write_revocations(
        r3 / "stage05" / sc.REVOKED_SPLITS, [OTHER_SPLIT], [OTHER], candidate_sha256="c"
    )
    assert cm.inherit_revocations(r3, final, [r0, r1])
    doc = sc.revocations(r3)
    assert [r["cell"] for r in doc["revoked"]] == sorted([BASE, OTHER])
    assert doc["barred_cells"] == sorted([BASE, OTHER])
    assert doc["candidate_sha256"] == "c" and doc["inherited_from"] == "final_retrain"


# ── the incumbent, by content ────────────────────────────────────────────────


def _bundle_dir(d, models, staged=None, record_models=True):
    (d / "stage05").mkdir(parents=True, exist_ok=True)
    (d / "stage05" / "models.pt").write_bytes(models)
    if staged is not None:
        (d / "stage06").mkdir(parents=True, exist_ok=True)
        rec = {"sha256": staged}
        if record_models:
            rec["models_sha256"] = cm.sha256_file(d / "stage05" / "models.pt")
        (d / "stage06" / rb.DEPLOY_RECORD).write_text(json.dumps(rec))
    return d


def test_the_incumbent_is_found_by_the_sha256_of_its_models_not_its_name(
    tmp_path,
):
    run = tmp_path / "run"
    _bundle_dir(run / "round_1", b"one", "s1")
    old = _bundle_dir(run / "stale" / "20000101" / rb.FINAL_RETRAIN_DIR, b"fr3", "s3")
    # final_retrain/ now holds another bundle than the one the gate scored
    _bundle_dir(run / rb.FINAL_RETRAIN_DIR, b"fr8", "s8")
    got = cm.find_incumbent(run, staged_sha256="s3", hint=run / rb.FINAL_RETRAIN_DIR)
    assert got is not None and got.dir == old
    assert got.models_sha256 == cm.sha256_file(old / "stage05" / "models.pt")
    assert got.verified == "deploy record: staged model and models.pt"
    assert got.to_json(run)["dir"] == f"stale/20000101/{rb.FINAL_RETRAIN_DIR}"
    by_models = cm.find_incumbent(run, models_sha256=got.models_sha256)
    assert by_models is not None and by_models.dir == old
    assert cm.find_incumbent(run, staged_sha256="nowhere") is None
    assert cm.find_incumbent(run, hint=run / "round_1") is None
    # models.pt changed since its deploy packed it: not that bundle any more
    (old / "stage05" / "models.pt").write_bytes(b"retrained")
    assert cm.find_incumbent(run, staged_sha256="s3") is None
    assert cm.find_incumbent(run, staged_sha256="s1", models_sha256="0" * 64) is None
    legacy = _bundle_dir(run / "round_2", b"two", "s2", record_models=False)
    got = cm.find_incumbent(run, staged_sha256="s2")
    assert got.dir == legacy and got.verified == "deploy record: staged model"


def test_the_repair_script_refuses_a_baseline_of_other_content(tmp_path, capsys):
    import repair_bundle
    from test_run_orchestrator import make_config

    cfg = make_config(tmp_path)
    train, base, out = tmp_path / "cand" / "stage05", tmp_path / "base", tmp_path / "o"
    _bundle_dir(train.parent, b"candidate")
    _bundle_dir(base, b"incumbent")
    gate = tmp_path / "cell_gate.json"
    gate.write_text(json.dumps(tcr._gate(("cu128_cot", tcr.SHAPE, 0.9, 0.3))))
    argv = [
        "--train-dir",
        str(train),
        "--baseline-dir",
        str(base),
        "--cell-gate",
        str(gate),
        "--out-dir",
        str(out),
        "--audit-dir",
        str(tmp_path / "audit"),
        "--config",
        str(cfg),
    ]
    assert repair_bundle.main([*argv, "--baseline-sha256", "0" * 64]) == 1
    assert "is not the bundle to carry cells from" in capsys.readouterr().err
    two = tcr._gate(("cu128_cot", tcr.SHAPE, 0.9, 0.3))
    two["groups"]["exclusive"] = {"deciding": True, "regressed": [_cell(OTHER)]}
    gate.write_text(json.dumps(two))
    assert repair_bundle.main([*argv, "--max-cells", "1"]) == 3
    rec = json.loads((out / cr.RECORD).read_text())
    assert (
        rec["outcome"] == "not repaired: 2 base cells flagged, more than --max-cells 1"
    )
    assert not (out / "models.pt").exists()


def _script_run(tmp_path, monkeypatch, incumbent_labels):
    """round_2's bundle splits BASE at M 1024 (its stage04b), round_1's, the
    incumbent, serves `incumbent_labels`, and round_2 holds the revocation of
    OTHER's split (round_0's) it inherited; the audit set and the leaves'
    GEMMs are test_collapse_repair's fakes. The repair script's argv."""
    import torch

    import repair_bundle
    from test_run_orchestrator import make_config

    run = tmp_path / "run"
    r0 = _round(run, 0, [OTHER_SPLIT])
    r1 = _round(run, 1)
    r2 = _round(run, 2, [SPLIT], revoked=[OTHER_SPLIT], barred=[OTHER])
    for d in (r1, r2):
        (d / "stage05").mkdir(parents=True, exist_ok=True)
    cand = tcr._bundle({tcr.LEAF: [tcr.H_BAD], tcr.SIBLING: [tcr.H_GOOD]})
    torch.save(cand, str(r2 / "stage05" / "models.pt"))
    ship = tcr._bundle({}, labels=incumbent_labels, tag="ship")
    torch.save(ship, str(r1 / "stage05" / "models.pt"))
    args = {
        "prior_round_dir": [str(r0), str(r1)],
        "current_round_dir": str(r2),
        "enriched_csv_dir": [str(r2 / "stage04")],
        "validation_frac": 0.2,
        "seed": 7,
    }
    (r2 / "stage05" / "metrics.json").write_text(json.dumps({"args": args}))
    gate = tmp_path / "cell_gate.json"
    gate.write_text(json.dumps(_gate({"exclusive": _group(regressed=[_cell(BASE)])})))
    monkeypatch.setattr(cr, "AuditView", lambda *a, **k: tcr.FakeAudit())
    monkeypatch.setattr(
        repair_bundle, "leaf_data_reader", lambda *a, **k: tcr.leaf_data()
    )
    monkeypatch.setattr(repair_bundle.hwlib, "device_hardware", lambda *a, **k: None)
    argv = ["--train-dir", str(r2 / "stage05"), "--baseline-dir", str(r1)]
    argv += ["--baseline-sha256", cm.sha256_file(r1 / "stage05" / "models.pt")]
    argv += ["--cell-gate", str(gate), "--out-dir", str(tmp_path / "out")]
    argv += [
        "--audit-dir",
        str(tmp_path / "audit"),
        "--config",
        str(make_config(tmp_path)),
    ]
    argv += ["--library-dir", str(tmp_path / "library"), "--max-cells", "3", "--quiet"]
    return argv, r1, r2


def test_the_repair_script_carries_a_flagged_cell_and_revokes_its_splits(
    tmp_path, monkeypatch
):
    """A cell rule's finding in BASE carries BASE's whole subtree from the
    incumbent, which serves it unsplit: the repaired bundle revokes round_2's
    split of BASE, bars BASE, and keeps the revocation round_2 held before."""
    import torch

    import repair_bundle

    argv, r1, r2 = _script_run(tmp_path, monkeypatch, (BASE, OTHER))
    assert repair_bundle.main(argv) == 0
    out = tmp_path / "out"
    repaired = torch.load(str(out / "models.pt"), weights_only=True)
    assert sorted(repaired["models"]) == [BASE, OTHER]
    assert repaired["models"][BASE]["tag"] == "ship"
    assert repaired["models"][OTHER]["tag"] == "cand"
    doc = json.loads((out / sc.REVOKED_SPLITS).read_text())
    assert doc["schema"] == sc.REVOKED_SCHEMA
    assert doc["revoked"] == [
        dict(SPLIT.to_json(), base_cell=BASE),
        dict(OTHER_SPLIT.to_json(), base_cell=OTHER),
    ]
    assert doc["barred_cells"] == [BASE, OTHER]
    cand_sha = cm.sha256_file(r2 / "stage05" / "models.pt")
    inc_sha = cm.sha256_file(r1 / "stage05" / "models.pt")
    assert (doc["candidate_sha256"], doc["incumbent_sha256"]) == (cand_sha, inc_sha)
    rec = json.loads((out / cr.RECORD).read_text())
    assert rec["carried_for_cell_rules"] == [BASE]
    assert rec["carried_cells"] == {
        BASE: {"removed": [tcr.LEAF, tcr.SIBLING], "carried": [BASE]}
    }
    assert rec["origins"] == {"candidate": cand_sha, "incumbent": inc_sha}
    assert rec["cells"][BASE]["action"] == "subtree carry"
    assert rec["cells"][BASE]["origin_sha256"] == inc_sha
    assert rec["revoked_splits"] == doc["revoked"]
    assert rec["barred_cells"] == [BASE, OTHER]
    met = json.loads((out / "metrics.json").read_text())
    assert met["repair"]["revoked_splits"] == doc["revoked"]
    # the repaired bundle's stage05 drops the split for every tree read
    # through it, and the inherited revocation stays
    (r2 / "stage05" / sc.REVOKED_SPLITS).write_text(
        (out / sc.REVOKED_SPLITS).read_text()
    )
    assert sc.load_split_tree([r2.parent / "round_0", r1], r2) == {}


def test_the_repair_script_refuses_an_incumbent_split_otherwise(tmp_path, monkeypatch):
    import repair_bundle

    other = (
        sc.make_subcell_label(BASE, "M", "<=", 2048),
        sc.make_subcell_label(BASE, "M", ">", 2048),
        OTHER,
    )
    argv, _, _ = _script_run(tmp_path, monkeypatch, other)
    assert repair_bundle.main(argv) == cm.NOT_REPAIRED_EXIT
    out = tmp_path / "out"
    rec = json.loads((out / cr.RECORD).read_text())
    assert rec["outcome"].startswith(
        f"cannot repair: the incumbent splits {BASE} at M2048, the run's split "
        f"tree has M1024 there"
    )
    assert not (out / "models.pt").exists()
    assert not (out / sc.REVOKED_SPLITS).exists()


# ── the all-group check and the judgement ───────────────────────────────────


def _shapes(n, seed):
    rng = np.random.default_rng(seed)
    return [
        [int(m), int(n_), int(k), 1] for m, n_, k in rng.integers(16, 8192, size=(n, 3))
    ]


EXCL, CU = _shapes(160, 1), _shapes(120, 2)


def _record(excl=1.0, cu=1.0, what="x", noise=0.0, seed=0, v2=True):
    """An audit record over EXCL and CU, every GEMM at 0.9 times the group's
    scale, with a v2 record of the same readings unless `v2` is false."""
    rng = np.random.default_rng(seed)

    def rows(shapes, scale):
        return [
            s + [float(min(1.0, 0.9 * scale * (1 + noise * rng.standard_normal())))]
            for s in shapes
        ]

    groups = {
        "exclusive": {"per_gemm": rows(EXCL, excl)},
        "cu64_cot": {"per_gemm": rows(CU, cu)},
    }
    rec = {"status": "ok", "what": what, "groups": groups}
    if v2:
        rec["v2"] = {"status": "ok", "groups": groups}
    return rec


SETTINGS = dict(al.CELL_GATE_DEFAULTS)


def test_the_all_group_check_flags_a_group_significantly_worse_beyond_the_tolerance():
    before = _record()
    worse = cm.group_check(before, _record(cu=0.96), SETTINGS, 400)
    assert worse["passed"] is False and worse["worse"] == ["cu64_cot"]
    g = worse["groups"]["cu64_cot"]
    assert g["ratio"] == pytest.approx(0.96) and g["p"] < 0.05 and g["worse"]
    assert worse["groups"]["exclusive"]["worse"] is False
    assert worse["reason"].startswith(
        "1 group(s) significantly worse beyond 1.0% (family-wise alpha 0.05 over "
        "2 groups): cu64_cot x0.9600"
    )
    # a consistent decline within the tolerance does not flag
    small = cm.group_check(before, _record(cu=0.995), SETTINGS, 400)
    assert small["passed"] is True and small["groups"]["cu64_cot"]["p"] < 0.05
    # noise around equal does not flag
    noisy = cm.group_check(
        _record(noise=0.05, seed=1), _record(noise=0.05, seed=2), SETTINGS, 400
    )
    assert noisy["passed"] is True
    # too few sign flips to reach alpha: not tested
    assert cm.group_check(before, _record(cu=0.9), SETTINGS, 10)["passed"] is None
    assert cm.group_check({"status": "ok"}, before, SETTINGS, 400)["passed"] is None


def test_a_repaired_bundle_passes_only_every_rule_under_the_refusals_protocols():
    before, same = _record(what="I"), _record(what="M")
    refusal = {"groups": {"exclusive": {"protocol": "v1"}}}
    v = cm.judge(before, same, refusal, SETTINGS, 400, "sustained", mean_rule=True)
    assert v.status == "pass" and v.not_worse[0] is True
    assert v.reason.startswith("not worse on the audit set: cell mean x1.0000")
    assert v.record()["all_groups"]["passed"] is True
    # a group the refusal judged under v2 is not judged again under v1
    stronger = {"groups": {"cu64_cot": {"protocol": "v2"}}}
    v1 = _record(what="M", v2=False)
    v = cm.judge(before, v1, stronger, SETTINGS, 400, "sustained", mean_rule=True)
    assert v.status == "wait" and v.weaker == ["cu64_cot"]
    v = cm.judge(before, same, stronger, SETTINGS, 400, "sustained", mean_rule=True)
    assert v.status == "pass"
    assert (
        cm.weaker_groups(stronger, {"groups": {"cu64_cot": {"protocol": "v2"}}}, "v2")
        == []
    )
    # a context group worse beyond the tolerance keeps it out, the mean
    # rule only where it applies
    worse = _record(cu=0.96, what="M")
    v = cm.judge(before, worse, refusal, SETTINGS, 400, "sustained", mean_rule=False)
    assert v.status == "fail" and v.not_worse is None
    assert v.reason.startswith("the all-group check keeps it out: 1 group(s)")
    collapse = _record(what="M")
    collapse["groups"]["exclusive"]["per_gemm"][0][4] = 0.3
    v = cm.judge(before, collapse, refusal, SETTINGS, 400, "sustained", mean_rule=True)
    assert v.status == "fail"
    assert v.reason.startswith("worse on the audit set: cell mean x")
    assert "; the per-cell gate keeps it out: 1 new pick(s) below 50.0%" in v.reason


# ── installs and withdrawals, interrupted anywhere ───────────────────────────


def _world(root):
    """A round whose stage06 refused its bundle, with a repair in repair/
    ready to install that carries OTHER from round_1 and revokes BASE's
    split."""
    run = root / "run"
    base, rd = run / "round_1", run / "round_2"
    s5, d = rd / "stage05", rd / cm.REPAIR_DIR
    for p in (base / "stage05", s5, d, rd / "stage06"):
        p.mkdir(parents=True)
    (base / "stage05" / "models.pt").write_text("incumbent models.pt")
    for name in ("models.pt", "cells.json", "metrics.json", "whitelist_tier.json"):
        (s5 / name).write_text(f"unrepaired {name}")
        (d / name).write_text(f"repaired {name}")
    (s5 / rb.CELL_ORIGINS).write_text("unrepaired origins")
    sc.write_revocations(d / sc.REVOKED_SPLITS, [SPLIT], [BASE])
    (rd / "stage06" / rb.DEPLOY_RECORD).write_text(json.dumps({"deployed": False}))
    (rd / "stage06" / "cell_gate.json").write_text("refusal")
    rec = {
        "outcome": "carried",
        "carried_cells": {OTHER: {"removed": [OTHER], "carried": [OTHER]}},
        "incumbent": {"models_sha256": cm.sha256_file(base / "stage05" / "models.pt")},
    }
    seen = []

    def origins(s5_, b, labels):
        seen.append((b, labels))
        from lib.fs import atomic_write_text

        atomic_write_text(s5_ / rb.CELL_ORIGINS, json.dumps({"from": b.name}))

    def copy(d_, r, b):
        cm.copy_repair(rd, d_, r, b, origins=origins)

    return run, rd, d, rec, base, copy, seen


class _Stops:
    """os.replace and os.rename that raise at the `stop`-th call."""

    def __init__(self, monkeypatch):
        self.n, self.stop = 0, None
        for name in ("replace", "rename"):
            real = getattr(os, name)
            monkeypatch.setattr(os, name, self._wrap(real))

    def _wrap(self, real):
        def call(*a, **kw):
            if self.n == self.stop:
                raise KeyboardInterrupt
            self.n += 1
            return real(*a, **kw)

        return call


def _installed(rd, d):
    s5 = rd / "stage05"
    return all(
        (s5 / n).read_text() == (d / n).read_text()
        for n in (*cm.REPAIRED_FILES, sc.REVOKED_SPLITS)
    ) and json.loads((s5 / rb.CELL_ORIGINS).read_text()) == {"from": "round_1"}


def test_a_round_install_interrupted_anywhere_is_finished_before_stage06(
    tmp_path, monkeypatch
):
    stops = _Stops(monkeypatch)
    run, rd, d, rec, base, copy, _ = _world(tmp_path / "whole")
    stops.n = 0
    cm.install(rd, d, rec, base, copy=copy, keep_stage06=True)
    steps = stops.n
    assert _installed(rd, d)
    kept = d / cm.UNREPAIRED
    assert (kept / "models.pt").read_text() == "unrepaired models.pt"
    assert (kept / "stage06" / "cell_gate.json").read_text() == "refusal"
    inst = json.loads((d / cm.INSTALLED).read_text())["installed"]
    assert inst["state"] == "installed" and inst["carried_from"] == "round_1"
    assert inst["unrepaired_sha256"] == cm.sha256_file(kept / "models.pt")
    for stop in range(steps):
        run, rd, d, rec, base, copy, seen = _world(tmp_path / f"stop{stop}")
        stops.n, stops.stop = 0, stop
        with pytest.raises(KeyboardInterrupt):
            cm.install(rd, d, rec, base, copy=copy, keep_stage06=True)
        stops.stop = None
        s5 = rd / "stage05"
        # models.pt goes last: an unfinished install leaves the unrepaired one
        assert (s5 / "models.pt").read_text() == "unrepaired models.pt", stop
        got = cm.installed_repair(
            rd, d, finish=lambda r: copy(d, r, cm.resolve_base(r, run))
        )
        if not (d / cm.INSTALLED).is_file():
            assert got is None and not _installed(rd, d), stop
            continue
        assert got is not None and _installed(rd, d), stop
        assert seen[-1] == (base, [OTHER]), stop


def test_a_withdrawal_interrupted_anywhere_is_finished(tmp_path, monkeypatch):
    stops = _Stops(monkeypatch)

    def installed_world(name):
        run, rd, d, rec, base, copy, _ = _world(tmp_path / name)
        cm.install(rd, d, rec, base, copy=copy, keep_stage06=True)
        # stage06 ran again on the repaired bundle and kept it out
        (rd / "stage06" / rb.DEPLOY_RECORD).write_text(json.dumps({"merged": 1}))
        return rd, d

    def withdrawn(rd, d):
        s5 = rd / "stage05"
        return (
            (s5 / "models.pt").read_text() == "unrepaired models.pt"
            and (s5 / "cells.json").read_text() == "unrepaired cells.json"
            and (s5 / rb.CELL_ORIGINS).read_text() == "unrepaired origins"
            and not (s5 / sc.REVOKED_SPLITS).exists()
            and (rd / "stage06" / "cell_gate.json").read_text() == "refusal"
            and json.loads((rd / "stage06" / rb.DEPLOY_RECORD).read_text())
            == {"deployed": False}
        )

    rd, d = installed_world("whole")
    stops.n = 0
    assert cm.withdraw(rd, d)
    steps = stops.n
    assert withdrawn(rd, d)
    assert json.loads((d / "stage06.repaired" / rb.DEPLOY_RECORD).read_text()) == {
        "merged": 1
    }
    assert cm.installed_repair(rd, d, finish=lambda r: None) is None
    assert cm.finish_withdrawal(rd, d)
    for stop in range(steps):
        rd, d = installed_world(f"w{stop}")
        stops.n, stops.stop = 0, stop
        with pytest.raises(KeyboardInterrupt):
            cm.withdraw(rd, d)
        stops.stop = None
        state = json.loads((d / cm.INSTALLED).read_text())["installed"]["state"]
        if state == "installed":
            # interrupted before the record said so: the install stands whole
            s5 = rd / "stage05"
            assert all(
                (s5 / n).read_text() == (d / n).read_text() for n in cm.REPAIRED_FILES
            ), stop
            continue
        assert cm.finish_withdrawal(rd, d), stop
        assert withdrawn(rd, d), stop


def test_a_withdrawal_puts_back_the_revocations_the_round_inherited(tmp_path):
    run, rd, d, rec, base, copy, _ = _world(tmp_path)
    s5 = rd / "stage05"
    inherited = sc.write_revocations(
        s5 / sc.REVOKED_SPLITS, [OTHER_SPLIT], [OTHER], inherited_from="final_retrain"
    )
    cm.install(rd, d, rec, base, copy=copy, keep_stage06=True)
    assert sc.revocations(rd) == json.loads((d / sc.REVOKED_SPLITS).read_text())
    assert cm.withdraw(rd, d)
    assert sc.revocations(rd) == inherited


def test_a_revoked_split_bars_its_base_cell_without_a_barred_list(tmp_path):
    rd = _round(tmp_path / "run", 1, [SPLIT], revoked=[DEEP], barred=())
    assert sc.revocations(rd)["barred_cells"] == []
    assert sc.barred_base_cells([rd]) == {BASE: "round_1"}


def _fix(run, rd):
    """A RoundRepair of `rd` whose steps need no pipeline."""
    return cm.RoundRepair(
        types.SimpleNamespace(run_dir=run),
        rd,
        2,
        lambda: 0,
        lambda *a: None,
        rd / "repair_bundle.py",
        rd / "audit_eval.py",
        None,
    )


def _refused_world(root):
    """`_world`'s repair installed, and the complete stage06 that measured
    the repaired bundle keeping it out, before the round decided."""
    run, rd, d, rec, base, copy, _ = _world(root)
    cm.install(rd, d, rec, base, copy=copy, keep_stage06=True)
    s06 = rd / "stage06"
    rec06 = {
        "deployed": False,
        "not_deployed": "below the target",
        "deploy_gate": {"passed": False, "reason": "below the target"},
        "models_sha256": cm.sha256_file(rd / "stage05" / "models.pt"),
    }
    (s06 / rb.DEPLOY_RECORD).write_text(json.dumps(rec06))
    (s06 / rb.STAGE_MARKER).write_text("{}")
    (d / cm.DECISION).write_text(json.dumps({"outcome": "installed", "reason": "r"}))
    return run, rd, d, rec06


def _withdrawn(rd, d):
    s5 = rd / "stage05"
    dec = json.loads((d / cm.DECISION).read_text())
    return (
        (s5 / "models.pt").read_text() == "unrepaired models.pt"
        and (s5 / "cells.json").read_text() == "unrepaired cells.json"
        and not (s5 / sc.REVOKED_SPLITS).exists()
        and json.loads((rd / "stage06" / rb.DEPLOY_RECORD).read_text())
        == {"deployed": False}
        and json.loads((d / cm.INSTALLED).read_text())["installed"]["state"]
        == "withdrawn"
        and dec["outcome"] == "refused"
        and dec["deployed"] is False
        and dec["stage06"]["deploy_gate"]["passed"] is False
        and "its stage06 keeps it out: below the target" in dec["reason"]
    )


def test_a_refusal_is_recorded_before_its_withdrawal_and_finished_anywhere(
    tmp_path, monkeypatch
):
    """The refusal of an installed repaired bundle writes the decision
    before the withdrawal moves anything, so that an interruption anywhere
    leaves a refusal the next invocation finishes (`resume`), never a
    repair to decide again; a stage06 that kept the bundle out before the
    decision said so is a refusal too."""
    stops = _Stops(monkeypatch)
    run, rd, d, rec06 = _refused_world(tmp_path / "whole")
    stops.n = 0
    dec = json.loads((d / cm.DECISION).read_text())
    assert _fix(run, rd).refuse(dec, rec06) == 0
    steps = stops.n
    assert _withdrawn(rd, d)
    decided = json.loads((d / cm.DECISION).read_text())["decided"]
    assert _fix(run, rd).resume() == 0
    assert _withdrawn(rd, d)
    assert json.loads((d / cm.DECISION).read_text())["decided"] == decided
    for stop in range(steps):
        run, rd, d, rec06 = _refused_world(tmp_path / f"stop{stop}")
        dec = json.loads((d / cm.DECISION).read_text())
        stops.n, stops.stop = 0, stop
        with pytest.raises(KeyboardInterrupt):
            _fix(run, rd).refuse(dec, rec06)
        stops.stop = None
        state = json.loads((d / cm.INSTALLED).read_text())["installed"]["state"]
        outcome = json.loads((d / cm.DECISION).read_text())["outcome"]
        assert outcome == "refused" or (stop == 0 and state == "installed"), stop
        assert _fix(run, rd).resume() == 0, stop
        assert _withdrawn(rd, d), stop


def test_a_refused_repair_still_installed_is_withdrawn_whatever_stage06_says(
    tmp_path,
):
    run, rd, d, _ = _refused_world(tmp_path / "deployed")
    dec = json.loads((d / cm.DECISION).read_text())
    dec.update(outcome="refused", reason="kept out")
    (d / cm.DECISION).write_text(json.dumps(dec))
    rc, rec = _fix(run, rd).decide({"deployed": True})
    assert rc == 0 and rec == {"deployed": False}
    assert (rd / "stage05" / "models.pt").read_text() == "unrepaired models.pt"
    assert json.loads((d / cm.DECISION).read_text())["reason"] == "kept out"
    # with nothing kept to put back the round stops, its refusal recorded
    run, rd, d, rec06 = _refused_world(tmp_path / "lost")
    (d / cm.UNREPAIRED / "models.pt").unlink()
    dec = json.loads((d / cm.DECISION).read_text())
    assert _fix(run, rd).refuse(dec, rec06) == 1
    assert json.loads((d / cm.DECISION).read_text())["outcome"] == "refused"
    assert (rd / "stage05" / "models.pt").read_text() == "repaired models.pt"
    assert _fix(run, rd).resume() == 1


def test_a_withdrawn_repair_is_a_decision_for_its_bundle(tmp_path):
    """A repair its stage06 kept out and the round withdrew stands for the
    bundle under the same settings, even when decision.json does not say
    so; other settings, or another bundle, are decided again."""
    run, rd, d, rec06 = _refused_world(tmp_path)
    assert cm.withdraw(rd, d)
    now = cm.sha256_file(rd / "stage05" / "models.pt")
    inst = json.loads((d / cm.INSTALLED).read_text())["installed"]
    st = {"max_cells": 3}
    fix = _fix(run, rd)
    assert not fix.decided({"settings": {"max_cells": 2}}, inst, now, st)
    assert not fix.decided({"settings": st}, inst, "0" * 64, st)
    assert not fix.decided({"settings": st, "outcome": "repair failed"}, {}, now, st)
    dec = {"settings": st, "outcome": "installed", "candidate_sha256": now}
    assert fix.decided(dec, inst, now, st)
    got = json.loads((d / cm.DECISION).read_text())
    assert got["outcome"] == "refused" and got["deployed"] is False
    assert got["reason"].startswith("the repaired bundle was withdrawn after its")
    for outcome in cm.FINAL_OUTCOMES:
        dec = {"settings": st, "outcome": outcome, "candidate_sha256": now}
        assert fix.decided(dec, {}, now, st), outcome
        # a decision about another bundle (the round trained again) does not
        # stand for this one
        other = dict(dec, candidate_sha256="0" * 64)
        assert not fix.decided(other, {}, now, st), outcome


def test_a_recorded_refusal_is_withdrawn_and_an_unfinished_stage06_decides_nothing(
    tmp_path,
):
    # recorded: withdrawn, whatever the round's stage06 holds
    run, rd, d, _ = _refused_world(tmp_path / "recorded")
    (rd / "stage06" / rb.STAGE_MARKER).unlink()
    dec = json.loads((d / cm.DECISION).read_text())
    dec.update(outcome="refused", reason="kept out")
    (d / cm.DECISION).write_text(json.dumps(dec))
    assert _fix(run, rd).resume() == 0
    assert (rd / "stage05" / "models.pt").read_text() == "unrepaired models.pt"
    # the refusal a stage06 of the repaired bundle wrote stands without the
    # marker the orchestrator did not get to write
    run, rd, d, _ = _refused_world(tmp_path / "unmarked")
    (rd / "stage06" / rb.STAGE_MARKER).unlink()
    assert _fix(run, rd).resume() == 0
    assert _withdrawn(rd, d)
    # a stage06 of the repaired bundle that wrote no refusal decided nothing:
    # the next one measures it
    for name, change in [
        ("failed install", {"not_deployed": None, "error": "the build failed"}),
        ("refusal and error", {"error": "the build failed"}),
        ("no reason", {"not_deployed": ""}),
        ("deployed", {"deployed": True, "not_deployed": None}),
    ]:
        run, rd, d, rec06 = _refused_world(tmp_path / name)
        (rd / "stage06" / rb.STAGE_MARKER).unlink()
        rec06 = {k: v for k, v in {**rec06, **change}.items() if v is not None}
        (rd / "stage06" / rb.DEPLOY_RECORD).write_text(json.dumps(rec06))
        assert _fix(run, rd).resume() == 0, name
        assert (rd / "stage05" / "models.pt").read_text() == "repaired models.pt"
        assert json.loads((d / cm.DECISION).read_text())["outcome"] == "installed"


def test_a_refusal_is_the_deploy_record_that_keeps_the_bundle_out():
    sha = "a" * 64
    rec = {"models_sha256": sha, "deployed": False, "not_deployed": "below"}
    assert cm.kept_out(rec, sha)
    assert not cm.kept_out(rec, "b" * 64) and not cm.kept_out(rec, None)
    assert not cm.kept_out(dict(rec, deployed=True), sha)
    assert not cm.kept_out({k: v for k, v in rec.items() if k != "deployed"}, sha)
    assert not cm.kept_out(dict(rec, not_deployed=""), sha)
    assert not cm.kept_out(dict(rec, error="the build failed"), sha)


def test_an_installed_repair_is_measured_in_one_attempt(tmp_path):
    run, rd, d, rec, base, copy, _ = _world(tmp_path)
    calls = []
    fix = cm.RoundRepair(
        types.SimpleNamespace(run_dir=run),
        rd,
        2,
        lambda **kw: calls.append(kw) or 0,
        None,
        rd,
        rd,
        None,
    )
    assert fix.measure() == 0 and calls == [{}]
    cm.install(rd, d, rec, base, copy=copy, keep_stage06=True)
    assert fix.measure() == 0 and calls[-1] == {"retries": 0}
    installed = (rd / "stage05" / "models.pt").read_text()
    (rd / "stage05" / "models.pt").write_text("trained again")
    assert fix.measure() == 0 and calls[-1] == {}
    (rd / "stage05" / "models.pt").write_text(installed)
    assert cm.withdraw(rd, d)
    assert fix.measure() == 0 and calls[-1] == {}
