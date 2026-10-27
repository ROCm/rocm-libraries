# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repair on refusal inside one `--mode full` command (lib/cell_merge.py):
a round's bundle the per-cell gate keeps out for a few base cells is
repaired, judged again, installed and deployed, and the rounds after it
and the final retrain build on the repaired bundle; above the cap, when the
repair does not pass, or when its stage06 keeps it out the round's bundle
stays out whole, as without the repair; and a final retrain kept out by
the cell rules is repaired like one kept out by a collapse. With stub
stages and stub audit and repair scripts."""

import hashlib
import json
import shutil
import types
from pathlib import Path

import pytest

import run_orchestrator as ro
import test_onecmd_audit as toa
import test_run_orchestrator as tro
from lib import audit as al
from lib import cell_merge as cm
from lib import run_bundles as rb
from lib import subcells as sc
from test_cell_merge import _Stops
from test_onecmd_audit import _final_config, audit_stubs, values  # noqa: F401
from test_run_orchestrator import LARGE, TINY, arg, by_stage, calls, run

stubs = tro.stubs
env = tro.env

# The per-cell gate of a round's stage06 (appended to the stage06 stub):
# STUB_MERGE names, per round, the base cells whose collapse keeps the
# round's own bundle out (a repaired one too with refuse_repaired), the
# protocol it judged under and whether the deploy gate passes the
# candidate and the repaired bundle. Its deploy record holds what stage06
# writes for a repair (deploy_gate, models_sha256, and repair from the
# stage's own `repair_record`), and the log names the bundle it measured;
# with kill_repaired the stage dies by SIGKILL once it wrote the record of
# a repaired bundle.
STAGE06_MERGE = r"""
if "--cell-gate-baseline" in argv and os.environ.get("STUB_MERGE"):
    import hashlib
    spec = json.loads(os.environ["STUB_MERGE"])
    base = Path(opt("--cell-gate-baseline"))
    brec = json.loads((base / "stage06" / "deploy_record.json").read_text())
    data = (Path(opt("--train-dir")) / "models.pt").read_bytes()
    repaired = data.endswith(b"repaired")
    if os.environ.get("STUB_LOG"):
        with open(os.environ["STUB_LOG"], "a") as f:
            f.write(json.dumps({"stage": "stage06_measured", "round": rnd,
                                "bundle": "repaired" if repaired else "candidate"})
                    + "\n")
    flagged = [] if repaired and not spec.get("refuse_repaired") else spec.get(rnd, [])
    group = {"protocol": spec.get("protocol", "sustained"), "deciding": True,
             "regressed": [], "floor": [], "tail": [
                 {"shape": [8192, 64, 8192, 1], "cell": c, "before": 0.9,
                  "after": 0.3, "confirmed": True, "decides": True}
                 for c in flagged]}
    gate = {"passed": not flagged, "reason": "stub gate", "n_tail": len(flagged),
            "baseline": {"dir": str(base), "sha256": brec.get("sha256")},
            "groups": {"exclusive": group}}
    (o / "cell_gate.json").write_text(json.dumps(gate))
    dec = json.loads((o / "deploy_record.json").read_text())
    ok = spec.get("deploy_gate_" + ("repaired" if repaired else "candidate"), True)
    dec["deploy_gate"] = {"passed": ok, "reason": "stub deploy gate"}
    dec["models_sha256"] = hashlib.sha256(data).hexdigest()
    sys.path[:0] = [os.environ["STUB_PIPELINE"],
                    os.path.join(os.environ["STUB_PIPELINE"], "stages")]
    import stage06_deploy_weights
    dec["repair"] = stage06_deploy_weights.repair_record(Path(opt("--train-dir")))
    if flagged or not ok:
        dec.update(deployed=False, not_deployed=(
            "below the target" if not ok else "the per-cell gate keeps it out"))
    (o / "deploy_record.json").write_text(json.dumps(dec))
    if repaired and spec.get("kill_repaired"):
        os.kill(os.getpid(), signal.SIGKILL)
"""

# The repair stub of test_onecmd_audit, writing the repaired bundle's
# metrics.json as scripts/repair_bundle.py does, the splits
# STUB_REPAIR_REVOKED names into its revocation record, and failing with
# STUB_REPAIR_RC.
REVOKE = r"""
revoked = json.loads(os.environ.get("STUB_REPAIR_REVOKED", "[]"))
if revoked:
    from lib import subcells as sc

    rules = [sc.SplitRule.from_json(r) for r in revoked]
    sc.write_revocations(out / sc.REVOKED_SPLITS, rules,
                         {sc.base_cell(r.cell) for r in rules})
    rec["revoked_splits"] = revoked
    (out / "repair.json").write_text(json.dumps(rec))
if not (train / "metrics.json").is_file():
    (out / "metrics.json").write_text(json.dumps(cr.metrics_doc({}, bundle, rec)))
if os.environ.get("STUB_REPAIR_RC"):
    (out / "models.pt").unlink()
    sys.exit(int(os.environ["STUB_REPAIR_RC"]))
"""

SPLIT = sc.build_split_rule(LARGE, "M", 4096)
CARRIED = {LARGE: {"removed": [SPLIT.lo_label, SPLIT.hi_label], "carried": [LARGE]}}


@pytest.fixture
def merge_stubs(audit_stubs, monkeypatch):  # noqa: F811
    s06 = audit_stubs / ro.STAGE_DEFS["stage06"].script
    s06.write_text(s06.read_text() + STAGE06_MERGE)
    stub = Path(str(ro.REPAIR_SCRIPT))
    stub.write_text(toa.REPAIR_STUB + REVOKE)
    for k in ("STUB_MERGE", "STUB_REPAIR_REVOKED", "STUB_REPAIR_RC"):
        monkeypatch.delenv(k, raising=False)
    return audit_stubs


def _world(tmp_path, monkeypatch, merge, repaired=(1.0, 1.0, 1.0, 1.0), **cfg):
    """A full run whose round_2 the per-cell gate keeps out for the base
    cells `merge["round_2"]` names, and whose repair stub carries LARGE
    from the incumbent and revokes its split, giving a bundle the audit set
    scores `repaired`."""
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    monkeypatch.setenv("STUB_SPLITS", json.dumps({"round_2": {LARGE: SPLIT.to_json()}}))
    monkeypatch.setenv("STUB_MERGE", json.dumps(merge))
    monkeypatch.setenv("STUB_REPAIR", "0 tier key(s) added, 1 base cell(s) carried")
    monkeypatch.setenv("STUB_REPAIR_CARRIED", json.dumps(CARRIED))
    monkeypatch.setenv("STUB_REPAIR_REVOKED", json.dumps([SPLIT.to_json()]))
    scores = {"repair/models.pt": list(repaired), "default": [0.9, 0.9, 0.9, 0.9]}
    monkeypatch.setenv("STUB_AUDIT", json.dumps(scores))
    return _final_config(tmp_path, **{"gate.repair": True, **cfg})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage06_of(log, rd):
    return [
        c for c in by_stage(log)["stage06"] if Path(arg(c, "--out-dir")).parent == rd
    ]


def test_a_round_the_per_cell_gate_keeps_out_ships_repaired(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _world(tmp_path, monkeypatch, {"round_2": [LARGE]}, **{"audit.v2": "rounds"})
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    r1, r2 = run_dir / "round_1", run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR
    log = calls(env)
    (rep,) = by_stage(log)["repair"]
    assert arg(rep, "--train-dir") == str(r2 / "stage05")
    assert arg(rep, "--baseline-dir") == str(r1)
    assert arg(rep, "--baseline-sha256") == _sha(r1 / "stage05" / "models.pt")
    assert arg(rep, "--cell-gate") == str(d / "cell_gate.json")
    assert arg(rep, "--max-cells") == "3"
    (ev,) = [
        c
        for c in by_stage(log)["audit_eval"]
        if arg(c, "--out") == str(d / "audit.json")
    ]
    assert values(ev, "--round-dir") == [str(r1)]
    assert values(ev, "--model-file") == [str(d / "models.pt")]
    assert values(ev, "--trained-round-dir") == [
        str(run_dir / f"round_{k}") for k in range(3)
    ]
    # protocol v2 measures the repaired bundle's picks before the audit set
    # judges it
    v2 = [m for c in by_stage(log)["audit_v2"] for m in values(c, "--model")]
    assert f"{run_dir.name}.round_2={d / 'models.pt'}" in v2
    # stage06 refused the round's bundle, then deployed the repaired one
    first, second = _stage06_of(log, r2)
    s5 = r2 / "stage05"
    assert (s5 / "models.pt").read_bytes().endswith(b"repaired")
    kept = d / cm.UNREPAIRED
    assert (kept / "models.pt").read_bytes() + b"repaired" == (
        s5 / "models.pt"
    ).read_bytes()
    assert (
        json.loads((kept / "stage06" / rb.DEPLOY_RECORD).read_text())["deployed"]
        is False
    )
    rec = json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())
    assert rec["deployed"] is True
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is True
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "repaired" and dec["deployed"] is True
    assert dec["candidate_sha256"] == _sha(kept / "models.pt")
    assert list(dec["flagged"]) == [LARGE]
    assert dec["flagged"][LARGE][0]["kind"] == "collapse"
    assert dec["incumbent"]["dir"] == "round_1"
    assert dec["incumbent"]["models_sha256"] == _sha(r1 / "stage05" / "models.pt")
    assert dec["cells"][LARGE]["action"] == "subtree carry"
    assert dec["cells"][LARGE]["origin_sha256"] == dec["incumbent"]["models_sha256"]
    assert dec["judged"]["status"] == "pass"
    assert dec["judged"]["all_groups"]["passed"] is True
    assert dec["selected_by_audit"] is True
    assert "fresh held-out audits" in dec["selection_note"]
    assert dec["settings"]["max_cells"] == 3
    assert dec["stage06"]["deployed"] is True
    assert dec["stage06"]["models_sha256"] == _sha(s5 / "models.pt")
    assert dec["repair"]["installed"]["state"] == "installed"
    assert dec["repair"]["installed"]["models_sha256"] == _sha(s5 / "models.pt")
    assert rec["repair"]["selected_by_audit"] is True
    assert rec["repair"]["selection_note"] == cm.SELECTION_NOTE
    assert rec["repair"]["outcome"] == dec["repair"]["outcome"]
    # the carried cell's origin is the network the incumbent served: the one
    # round_0 trained, which round_1 carried
    origins = json.loads((s5 / rb.CELL_ORIGINS).read_text())
    assert origins["origins"][origins["cells"][LARGE]["origin"]]["bundle"] == "round_0"
    # the round's split of the carried cell is revoked for every later reader
    assert LARGE in json.loads((r2 / "stage04b" / "splits.json").read_text())["splits"]
    assert sc.revoked_parents(r2) == [LARGE]
    rounds = ro.complete_prefix(run_dir)
    assert LARGE not in sc.load_cumulative_split_tree(rounds)
    assert sc.barred_base_cells(rounds) == {LARGE: "round_2"}
    # the final retrain is judged against the repaired round, which ships
    final = run_dir / ro.FINAL_RETRAIN_DIR
    fdec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert fdec["baseline"]["round"] == "round_2" and fdec["baseline"]["deployed"]
    # decided: the same command neither repairs nor deploys again
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "repair" not in by_stage(calls(env)[n:])


def test_the_round_after_a_repaired_one_builds_on_the_repaired_bundle(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE]},
        **{"full.total_rounds": 4, "full.final_retrain.enabled": False},
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    r2, r3 = run_dir / "round_2", run_dir / "round_3"
    st = by_stage(calls(env))
    (s04b,) = [c for c in st["stage04b"] if Path(arg(c, "--out-dir")).parent == r3]
    assert arg(s04b, "--models-dir") == str(r2 / "stage05")
    assert arg(s04b, "--prior-models") == str(
        run_dir / "round_1" / "stage05" / "models.pt"
    )
    (s05,) = [c for c in st["stage05"] if Path(arg(c, "--output-dir")).parent == r3]
    assert values(s05, "--prior-round-dir")[-1] == str(r2)
    (s06,) = _stage06_of(calls(env), r3)
    assert arg(s06, "--cell-gate-baseline") == str(r2)
    assert (r2 / "stage05" / "models.pt").read_bytes().endswith(b"repaired")


def test_more_flagged_cells_than_the_cap_keep_the_round_out_whole(
    tmp_path, merge_stubs, env, monkeypatch
):
    """Without a repair the round's bundle stays out, as before gate.repair:
    the next round still builds on it, while its gate compares with the
    bundle the run ships."""
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE, TINY]},
        **{
            "gate.repair_max_cells": 1,
            "full.total_rounds": 4,
            "full.final_retrain.enabled": False,
        },
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    r1, r2, r3 = (run_dir / f"round_{k}" for k in (1, 2, 3))
    st = by_stage(calls(env))
    assert "repair" not in st
    dec = json.loads((r2 / cm.REPAIR_DIR / cm.DECISION).read_text())
    assert dec["outcome"] == "not repairable"
    assert dec["reason"] == (
        "2 base cells flagged, more than gate.repair_max_cells 1: refused whole"
    )
    assert sorted(dec["flagged"]) == sorted([LARGE, TINY])
    assert (
        json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())["deployed"] is False
    )
    assert not (r2 / "stage05" / "models.pt").read_bytes().endswith(b"repaired")
    (s04b,) = [c for c in st["stage04b"] if Path(arg(c, "--out-dir")).parent == r3]
    assert arg(s04b, "--models-dir") == str(r2 / "stage05")
    (s06,) = _stage06_of(calls(env), r3)
    assert arg(s06, "--cell-gate-baseline") == str(r1)


def test_a_repair_that_does_not_pass_keeps_the_round_out_and_is_not_tried_again(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE]},
        repaired=(1.0, 1.0, 1.0, 0.3),
        **{"full.final_retrain.enabled": False},
    )
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    d = r2 / cm.REPAIR_DIR
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "refused" and dec["deployed"] is False
    assert dec["reason"].startswith(
        "its repair (0 tier key(s) added, 1 base cell(s) carried) does not pass: "
        "the per-cell gate keeps it out: 1 new pick(s) below 50.0%"
    )
    assert dec["judged"]["status"] == "fail"
    assert not (d / cm.UNREPAIRED).exists() and not (d / cm.INSTALLED).exists()
    assert not (r2 / "stage05" / "models.pt").read_bytes().endswith(b"repaired")
    assert not (r2 / "stage05" / sc.REVOKED_SPLITS).exists()
    assert len(_stage06_of(calls(env), r2)) == 1
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is False
    # the round run again measures its bundle again, never repairs it again
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    assert len(st["stage06"]) == 1 and "repair" not in st
    assert json.loads((d / cm.DECISION).read_text())["decided"] == dec["decided"]
    # a bundle the round trained again is another candidate: the refusal of
    # the first one does not stand for it, and it is repaired on its own
    (r2 / "stage05" / "models.pt").write_bytes(b"trained again")
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env)[n:])["repair"]) == 1
    again = json.loads((d / cm.DECISION).read_text())
    assert again["candidate_sha256"] == hashlib.sha256(b"trained again").hexdigest()
    assert again["outcome"] == "refused" and again["decided"] != dec["decided"]


def test_a_repaired_bundle_its_stage06_keeps_out_is_withdrawn(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE], "deploy_gate_repaired": False},
        **{"full.final_retrain.enabled": False},
    )
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    d = r2 / cm.REPAIR_DIR
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "refused"
    assert "its stage06 keeps it out: below the target" in dec["reason"]
    assert dec["stage06"]["deploy_gate"] == {
        "passed": False,
        "reason": "stub deploy gate",
    }
    s5 = r2 / "stage05"
    assert (s5 / "models.pt").read_bytes() == (
        d / cm.UNREPAIRED / "models.pt"
    ).read_bytes()
    assert not (s5 / sc.REVOKED_SPLITS).exists()
    rec = json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())
    assert rec["deployed"] is False
    assert rec["not_deployed"] == "the per-cell gate keeps it out"
    assert (
        json.loads((r2 / "stage06" / "cell_gate.json").read_text())["passed"] is False
    )
    assert (d / "stage06.repaired" / rb.DEPLOY_RECORD).is_file()
    inst = json.loads((d / cm.INSTALLED).read_text())["installed"]
    assert inst["state"] == "withdrawn"
    assert LARGE in sc.load_cumulative_split_tree(ro.complete_prefix(run_dir))


def test_a_round_its_deploy_gate_keeps_out_too_is_not_repaired(
    tmp_path, merge_stubs, env, monkeypatch
):
    """A repair answers the per-cell gate's findings, not a bundle below its
    deploy gate: a round both keep out stays out whole, and a round only the
    deploy gate keeps out is no repair's business."""
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE], "deploy_gate_candidate": False},
        **{"full.final_retrain.enabled": False},
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    d = run_dir / "round_2" / cm.REPAIR_DIR
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "not repairable"
    assert dec["reason"] == "its deploy gate keeps it out too: stub deploy gate"
    assert "repair" not in by_stage(calls(env))
    alone = tmp_path / "alone"
    alone.mkdir()
    cfg = _world(
        alone,
        monkeypatch,
        {"round_2": [], "deploy_gate_candidate": False},
        **{"full.final_retrain.enabled": False},
    )
    assert run(merge_stubs, cfg, alone / "run", "--mode", "full") == 0
    r2 = alone / "run" / "round_2"
    assert (
        json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())["deployed"] is False
    )
    assert not (r2 / cm.REPAIR_DIR).exists()


def test_an_interrupted_install_is_finished_before_stage06(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _world(tmp_path, monkeypatch, {"round_2": [LARGE]})
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    r2 = run_dir / "round_2"
    d, s5 = r2 / cm.REPAIR_DIR, r2 / "stage05"
    # the install stopped before its last step: models.pt is the unrepaired one
    (s5 / "models.pt").write_bytes((d / cm.UNREPAIRED / "models.pt").read_bytes())
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    assert "repair" not in st
    assert (s5 / "models.pt").read_bytes() == (d / "models.pt").read_bytes()
    (s06,) = _stage06_of(calls(env)[n:], r2)
    assert json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())["deployed"]


def test_an_install_its_stage06_has_not_measured_is_measured(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The run stopped as the install of round_2's repair completed: the
    round's stage06 is still the one that kept the unrepaired bundle out,
    which decides nothing about the repaired one; the next invocation
    measures the repaired bundle and deploys it."""
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE]},
        **{"full.final_retrain.enabled": False},
    )
    real = cm.install

    def stop_after(*a, **kw):
        real(*a, **kw)
        raise KeyboardInterrupt

    monkeypatch.setattr(cm, "install", stop_after)
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 130
    monkeypatch.setattr(cm, "install", real)
    r2 = run_dir / "round_2"
    assert (r2 / "stage05" / "models.pt").read_bytes().endswith(b"repaired")
    assert (
        json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())["deployed"] is False
    )
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert _measured(calls(env)[n:], "round_2") == ["repaired"]
    dec = json.loads((r2 / cm.REPAIR_DIR / cm.DECISION).read_text())
    assert dec["outcome"] == "repaired" and dec["deployed"] is True
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is True


def test_a_repair_measured_under_weaker_protocols_waits(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _world(tmp_path, monkeypatch, {"round_2": [LARGE]})
    stub = Path(str(ro.AUDIT_EVAL_SCRIPT))
    stub.write_text(
        stub.read_text().replace(
            'if os.environ.get("STUB_AUDIT_V2"):',
            'if os.environ.get("STUB_AUDIT_V2") and not model.endswith('
            '"/repair/models.pt"):',
        )
    )
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR
    # the same command tries again, gate.repair_max_attempts (3) times in all
    for attempt in (1, 2):
        assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 1
        assert not (r2 / ro.ROUND_MARKER).exists()
        assert not (d / cm.UNREPAIRED).exists()
        rec = json.loads((d / cm.REPAIR_RECORD).read_text())
        assert rec["judged"]["status"] == "wait"
        assert rec["judged"]["weaker_groups"] == ["exclusive"]
        dec = json.loads((d / cm.DECISION).read_text())
        assert dec["outcome"] == "wait" and dec["attempts"] == attempt
        assert dec["reason"] == rec["judged"]["reason"]
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env))["repair"]) == 3
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "not repaired" and dec["attempts"] == 3
    assert dec["reason"].endswith(
        "wait for their v2 measurement, in each of 3 attempt(s) "
        "(gate.repair_max_attempts 3): refused whole"
    )
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is False


def test_without_gate_repair_a_refused_round_stays_out(
    tmp_path, merge_stubs, env, monkeypatch
):
    assert ro.cfg_get({}, "gate.repair") is False
    assert ro.cfg_get({}, "gate.repair_max_cells") == cm.DEFAULT_MAX_CELLS
    errors, _ = ro.validate_config({"gate": {"repair_max_cells": 0}})
    assert any(e.startswith("gate.repair_max_cells:") for e in errors)
    assert ro.cfg_get({}, "gate.repair_max_attempts") == cm.DEFAULT_MAX_ATTEMPTS == 3
    for bad in (0, -1, "3", 1.5):
        errors, _ = ro.validate_config({"gate": {"repair_max_attempts": bad}})
        assert any(e.startswith("gate.repair_max_attempts:") for e in errors), bad
    errors, _ = ro.validate_config({"gate": {"repair_max_attempts": 1}})
    assert not any("repair_max_attempts" in e for e in errors)
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE]},
        **{"gate.repair": False, "full.final_retrain.enabled": False},
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "repair" not in by_stage(calls(env))
    assert not (run_dir / "round_2" / cm.REPAIR_DIR).exists()


def test_gate_repair_without_a_per_cell_gate_warns(tmp_path):
    import yaml

    path = _final_config(tmp_path, **{"gate.repair": True})
    cfg = yaml.safe_load(Path(path).read_text())
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and not [w for w in warnings if "gate.repair" in w]
    cfg["gate"]["per_cell"] = False
    _, warnings = ro.validate_config(cfg)
    assert any("gate.per_cell false turns off" in w for w in warnings)
    cfg["gate"]["per_cell"] = True
    cfg["audit"].pop("dir")
    cfg["full"]["final_retrain"]["gate"] = "none"
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and any("no audit.dir turns off" in w for w in warnings)


# ── the final retrain ───────────────────────────────────────────────────────


def _regressed(monkeypatch, cells):
    """The per-cell gate finds `cells` significantly worse in the final
    retrain as it trained, and judges its repair as it is."""
    real = al.cell_gate

    def gate(before, after, settings=None, bootstrap=2000, official=None):
        out = real(before, after, settings, bootstrap, official=official)
        model = str(after.get("model") or "")
        if after.get("what") == ro.FINAL_RETRAIN_DIR and model.endswith(
            "stage05/models.pt"
        ):
            g = out["groups"]["exclusive"]
            g["regressed"] = [
                {"cell": c, "n": 4, "ratio": 0.95, "ratio_ci": None, "p": 0.01}
                for c in cells
            ]
            out.update(passed=False, n_regressed=len(cells), reason="stub: regressed")
        return out

    monkeypatch.setattr(al, "cell_gate", gate)


def _final_world(tmp_path, monkeypatch, cells, **cfg):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    monkeypatch.setenv("STUB_REPAIR", "0 tier key(s) added, 1 base cell(s) carried")
    carried = {TINY: {"removed": [TINY], "carried": [TINY]}}
    monkeypatch.setenv("STUB_REPAIR_CARRIED", json.dumps(carried))
    scores = {
        "models.pt": [1.0, 1.0, 1.0, 1.0],
        "repair/models.pt": [1.0, 1.0, 1.0, 1.0],
        "default": [0.9, 0.9, 0.9, 0.9],
    }
    monkeypatch.setenv("STUB_AUDIT", json.dumps(scores))
    _regressed(monkeypatch, cells)
    return _final_config(tmp_path, **{"audit.v2": "final", **cfg})


def test_a_final_retrain_kept_out_by_a_cell_rule_ships_repaired(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _final_world(tmp_path, monkeypatch, [TINY])
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    final, r2 = run_dir / ro.FINAL_RETRAIN_DIR, run_dir / "round_2"
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] is True
    assert dec["unrepaired"]["cell_gate"]["n_regressed"] == 1
    assert dec["unrepaired"]["cell_gate"]["n_tail"] == 0
    assert dec["repair"]["incumbent"]["models_sha256"] == _sha(
        r2 / "stage05" / "models.pt"
    )
    assert dec["repair"]["judged"]["all_groups"]["passed"] is True
    assert dec["reason"].startswith("repaired (0 tier key(s) added, 1 base cell(s)")
    assert "; all groups: no group significantly worse" in dec["reason"]
    (rep,) = by_stage(calls(env))["repair"]
    assert arg(rep, "--max-cells") == "3"
    assert arg(rep, "--baseline-sha256") == _sha(r2 / "stage05" / "models.pt")
    assert (final / "stage05" / "models.pt").read_bytes().endswith(b"repaired")


def test_a_final_retrain_with_more_flagged_cells_than_the_cap_stays_out(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _final_world(
        tmp_path, monkeypatch, [TINY, LARGE], **{"gate.repair_max_cells": 1}
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    dec = json.loads(
        (run_dir / ro.FINAL_RETRAIN_DIR / ro.FinalRetrain.DECISION).read_text()
    )
    assert dec["accepted"] is False and "repair" not in dec
    assert "the per-cell gate keeps it out: stub: regressed" in dec["reason"]
    assert "repair" not in by_stage(calls(env))


# ── what a round decided stays decided ──────────────────────────────────────


def _measured(log, rnd):
    """The bundles the stage06 stub measured for round `rnd`, in order."""
    return [
        c["bundle"]
        for c in by_stage(log).get("stage06_measured", [])
        if c["round"] == rnd
    ]


def test_a_refusal_interrupted_anywhere_is_never_repaired_or_measured_again(
    tmp_path, merge_stubs, env, monkeypatch
):
    """stage06 keeps the installed repaired bundle out. Wherever an
    interruption stops its refusal (the decision, then the withdrawal step
    by step), and before the decision is written at all, the next
    invocation finishes it: the round stays out with its unrepaired bundle,
    and the repaired bundle is neither repaired nor measured again, although
    a new measurement would pass it now."""
    merge = {"round_2": [LARGE], "deploy_gate_repaired": False}
    cfg = _world(tmp_path, monkeypatch, merge, **{"full.final_retrain.enabled": False})
    run_dir, snap = tmp_path / "run", tmp_path / "snap"
    r2 = run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR
    real = cm.RoundRepair.refuse

    def stop(self, dec, rec):
        raise KeyboardInterrupt

    # the first invocation stops as its stage06 has kept the repaired bundle
    # out, before the round decided anything
    monkeypatch.setattr(cm.RoundRepair, "refuse", stop)
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 130
    monkeypatch.setattr(cm.RoundRepair, "refuse", real)
    assert _measured(calls(env), "round_2") == ["candidate", "repaired"]
    assert json.loads((d / cm.DECISION).read_text())["outcome"] == "installed"
    shutil.copytree(run_dir, snap, symlinks=True)
    unrepaired = (d / cm.UNREPAIRED / "models.pt").read_bytes()
    monkeypatch.setenv("STUB_MERGE", json.dumps(dict(merge, deploy_gate_repaired=True)))
    stops = _Stops(monkeypatch)

    def refusal_stopped_at(stop):
        """The snapshot with the round's refusal stopped at its `stop`-th
        file step (None: whole); the number of steps it took."""
        shutil.rmtree(run_dir)
        shutil.copytree(snap, run_dir, symlinks=True)
        fix = cm.RoundRepair(
            types.SimpleNamespace(run_dir=run_dir), r2, 2, None, None, d, d, None
        )
        dec = json.loads((d / cm.DECISION).read_text())
        rec = json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())
        stops.n, stops.stop = 0, stop
        try:
            if stop is None:
                assert fix.refuse(dec, rec) == 0
            else:
                with pytest.raises(KeyboardInterrupt):
                    fix.refuse(dec, rec)
        finally:
            stops.stop = None
        return stops.n

    steps = refusal_stopped_at(None)
    assert steps >= 5
    for at in [*range(steps), "whole", "snapshot", "recorded only by install"]:
        if at == "snapshot":
            shutil.rmtree(run_dir)
            shutil.copytree(snap, run_dir, symlinks=True)
        else:
            refusal_stopped_at(None if isinstance(at, str) else at)
        if at == "recorded only by install":
            # a withdrawal whose decision never said so
            dec = json.loads((d / cm.DECISION).read_text())
            dec.update(outcome="installed", deployed=None)
            (d / cm.DECISION).write_text(json.dumps(dec))
        n = len(calls(env))
        assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0, at
        new = calls(env)[n:]
        assert "repair" not in by_stage(new), at
        assert _measured(new, "round_2") == ["candidate"], at
        dec = json.loads((d / cm.DECISION).read_text())
        assert dec["outcome"] == "refused" and dec["deployed"] is False, at
        assert (r2 / "stage05" / "models.pt").read_bytes() == unrepaired, at
        inst = json.loads((d / cm.INSTALLED).read_text())["installed"]
        assert inst["state"] == "withdrawn", at
        assert not (r2 / "stage05" / sc.REVOKED_SPLITS).exists(), at
        assert (
            json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())["deployed"]
            is False
        ), at
        assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is False
    # a refusal with nothing kept to put back stops the round before any
    # stage06
    refusal_stopped_at(1)
    assert json.loads((d / cm.DECISION).read_text())["outcome"] == "refused"
    (d / cm.UNREPAIRED / "models.pt").unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") != 0
    assert _measured(calls(env)[n:], "round_2") == []
    assert not (r2 / ro.ROUND_MARKER).exists()


def _kept_out_unmarked(tmp_path, monkeypatch, env, stubs_dir, merge, first):
    """round_2's repaired bundle kept out by a stage06 that the first
    invocation (`first(cfg, run_dir)`, its exit status) leaves unmarked;
    then a second invocation under which a new measurement would pass the
    repaired bundle: (the bundles stage06 measured in each, the decision)."""
    cfg = _world(tmp_path, monkeypatch, merge, **{"full.final_retrain.enabled": False})
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    n = len(calls(env))
    assert first(cfg, run_dir) != 0
    rec = json.loads((r2 / "stage06" / rb.DEPLOY_RECORD).read_text())
    assert rec["deployed"] is False and rec["not_deployed"] == "below the target"
    assert not (r2 / "stage06" / rb.STAGE_MARKER).is_file()
    d = r2 / cm.REPAIR_DIR
    assert json.loads((d / cm.DECISION).read_text())["outcome"] == "installed"
    before = _measured(calls(env)[n:], "round_2")
    passing = dict(merge, deploy_gate_repaired=True, kill_repaired=False)
    monkeypatch.setenv("STUB_MERGE", json.dumps(passing))
    n = len(calls(env))
    assert run(stubs_dir, cfg, run_dir, "--mode", "full") == 0
    new = calls(env)[n:]
    assert "repair" not in by_stage(new)
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is False
    assert json.loads((d / cm.INSTALLED).read_text())["installed"]["state"] == (
        "withdrawn"
    )
    return before, _measured(new, "round_2"), json.loads((d / cm.DECISION).read_text())


def test_a_refusal_its_stage06_wrote_stands_when_a_code_change_stops_it(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The code changed as the stage06 of round_2's repaired bundle ended:
    it wrote its refusal and the orchestrator stopped before it marked the
    stage complete. The next invocation takes the refusal: the repaired
    bundle is neither measured nor repaired again, although a new
    measurement would pass it."""
    merge = {"round_2": [LARGE], "deploy_gate_repaired": False}
    real = ro.Pipeline.pin_drift
    fired = []

    def drift(self):
        r2 = self.run_dir / "round_2"
        rec = cm._read(r2 / "stage06" / rb.DEPLOY_RECORD) or {}
        inst = (cm._read(r2 / cm.REPAIR_DIR / cm.INSTALLED) or {}).get("installed")
        if not fired and cm.kept_out(rec, (inst or {}).get("models_sha256")):
            fired.append(1)
            return ["code: changed while stage06 ran"]
        return real(self)

    def first(cfg, run_dir):
        monkeypatch.setattr(ro.Pipeline, "pin_drift", drift)
        try:
            return run(merge_stubs, cfg, run_dir, "--mode", "full")
        finally:
            monkeypatch.setattr(ro.Pipeline, "pin_drift", real)

    before, after, dec = _kept_out_unmarked(
        tmp_path, monkeypatch, env, merge_stubs, merge, first
    )
    assert fired and before == ["candidate", "repaired"]
    assert after == ["candidate"]
    assert dec["outcome"] == "refused" and dec["deployed"] is False
    assert "its stage06 keeps it out: below the target" in dec["reason"]


def test_a_stage06_killed_after_its_refusal_of_a_repair_is_not_tried_again(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The stage06 of round_2's repaired bundle dies by a signal after it
    wrote its refusal: it is not attempted again (runtime.stage_retries is
    1), and the next invocation takes the refusal."""
    merge = {"round_2": [LARGE], "deploy_gate_repaired": False, "kill_repaired": True}

    def first(cfg, run_dir):
        return run(merge_stubs, cfg, run_dir, "--mode", "full")

    before, after, dec = _kept_out_unmarked(
        tmp_path, monkeypatch, env, merge_stubs, merge, first
    )
    assert before == ["candidate", "repaired"]
    assert after == ["candidate"]
    assert dec["outcome"] == "refused" and dec["deployed"] is False
    # the same for the stage06 that measures an install an interruption left
    # unmeasured, which the round's own stage06 runs
    real = cm.install

    def stop_after(*a, **kw):
        real(*a, **kw)
        raise KeyboardInterrupt

    def interrupted(cfg, run_dir):
        monkeypatch.setattr(cm, "install", stop_after)
        try:
            assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 130
        finally:
            monkeypatch.setattr(cm, "install", real)
        n = len(calls(env))
        rc = run(merge_stubs, cfg, run_dir, "--mode", "full")
        assert _measured(calls(env)[n:], "round_2") == ["repaired"]
        return rc

    again = tmp_path / "interrupted"
    again.mkdir()
    before, after, dec = _kept_out_unmarked(
        again, monkeypatch, env, merge_stubs, merge, interrupted
    )
    assert before == ["candidate", "repaired"] and after == ["candidate"]
    assert dec["outcome"] == "refused" and dec["deployed"] is False


def test_a_decision_is_made_again_under_other_settings_only(
    tmp_path, merge_stubs, env, monkeypatch
):
    """Two flagged cells over a cap of 1 keep round_2 out whole; run again
    under the same settings it stays decided, under a cap of 3 it is
    decided again, repaired and shipped, and the run summary says the
    audit GEMMs that flagged its cells also judged it."""
    merge = {"round_2": [LARGE, TINY]}
    off = {"full.final_retrain.enabled": False}
    cfg = _world(tmp_path, monkeypatch, merge, **{"gate.repair_max_cells": 1, **off})
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    first = json.loads((d / cm.DECISION).read_text())
    assert first["outcome"] == "not repairable"
    assert first["settings"]["max_cells"] == 1
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "repair" not in by_stage(calls(env)[n:])
    assert json.loads((d / cm.DECISION).read_text()) == first
    more = _world(tmp_path, monkeypatch, merge, **off)
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, more, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env)[n:])["repair"]) == 1
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "repaired" and dec["settings"]["max_cells"] == 3
    (old,) = (r2 / "stale").glob(f"*/{cm.REPAIR_DIR}")
    assert json.loads((old / cm.DECISION).read_text()) == first
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is True
    said = (run_dir / "orchestrator.log").read_text().split("mode=full")[-1]
    assert f"round_2 ships repaired ({dec['repair']['outcome']}): " in said
    assert cm.SELECTION_NOTE in said


def test_a_round_trained_again_after_its_repair_is_judged_as_its_own(
    tmp_path, merge_stubs, env, monkeypatch
):
    """After round_2's repair shipped, its stage05 trained a new bundle: the
    install of the old repair is not that bundle's, so its refusal is not
    withdrawn over it; the new bundle is repaired from scratch, with the
    old repair moved aside."""
    off = {"full.final_retrain.enabled": False}
    cfg = _world(tmp_path, monkeypatch, {"round_2": [LARGE]}, **off)
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    d, s5 = r2 / cm.REPAIR_DIR, r2 / "stage05"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    first = json.loads((d / cm.INSTALLED).read_text())
    (s5 / "models.pt").write_bytes(b"trained again")
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env)[n:])["repair"]) == 1
    assert _measured(calls(env)[n:], "round_2") == ["candidate", "repaired"]
    assert (s5 / "models.pt").read_bytes() == b"trained againrepaired"
    assert (d / cm.UNREPAIRED / "models.pt").read_bytes() == b"trained again"
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "repaired"
    assert dec["candidate_sha256"] == hashlib.sha256(b"trained again").hexdigest()
    (old,) = (r2 / "stale").glob(f"*/{cm.REPAIR_DIR}")
    assert json.loads((old / cm.INSTALLED).read_text()) == first


def _unmeasured_world(tmp_path, monkeypatch, **cfg):
    """`_world` with an audit set that never measures a repaired bundle."""
    path = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE]},
        **{"full.final_retrain.enabled": False, **cfg},
    )
    stub = Path(str(ro.AUDIT_EVAL_SCRIPT))
    old = 'rec = {"status": "ok", "status_text": "complete", "what": what,'
    if old in stub.read_text():
        stub.write_text(
            stub.read_text().replace(
                old,
                'rec = {"status": "missing" if model.endswith("/repair/models.pt") '
                'else "ok", "status_text": "complete", "what": what,',
            )
        )
    return path


def test_a_repair_the_audit_set_never_measures_is_tried_up_to_the_cap(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The audit set does not measure the repaired bundle: the round stops
    with 1 and records that it waits, the same command repairs and tries
    again, and the attempt that waits for the gate.repair_max_attempts-th
    time refuses the round whole; the run goes on, and the round is not
    repaired again."""
    cfg = _unmeasured_world(tmp_path, monkeypatch, **{"gate.repair_max_attempts": 2})
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 1
    assert not (r2 / ro.ROUND_MARKER).exists()
    assert not (d / cm.UNREPAIRED).exists()
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "wait" and dec["attempts"] == 1
    assert dec["deployed"] is False and "judged" not in dec
    assert dec["reason"].startswith(
        "the audit set did not measure the repaired bundle and the incumbent"
    )
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env))["repair"]) == 2
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "not repaired" and dec["attempts"] == 2
    assert dec["reason"].endswith(
        ", in each of 2 attempt(s) (gate.repair_max_attempts 2): refused whole"
    )
    assert json.loads((r2 / ro.ROUND_MARKER).read_text())["deployed"] is False
    assert not (r2 / "stage05" / "models.pt").read_bytes().endswith(b"repaired")
    # decided: the round entered again is not repaired again
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "repair" not in by_stage(calls(env)[n:])
    assert json.loads((d / cm.DECISION).read_text()) == dec


def test_the_attempts_of_a_repair_count_for_one_bundle_under_its_settings(
    tmp_path, merge_stubs, env, monkeypatch
):
    """A round trained again, or its gate settings changed, after a repair
    waited, starts counting its attempts anew."""
    cfg = _unmeasured_world(tmp_path, monkeypatch, **{"gate.repair_max_attempts": 2})
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR

    def attempts():
        dec = json.loads((d / cm.DECISION).read_text())
        return dec["outcome"], dec["attempts"]

    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 1
    assert attempts() == ("wait", 1)
    (r2 / "stage05" / "models.pt").write_bytes(b"trained again")
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 1
    assert attempts() == ("wait", 1)
    other = _unmeasured_world(
        tmp_path,
        monkeypatch,
        **{"gate.repair_max_attempts": 2, "gate.repair_max_cells": 2},
    )
    assert run(merge_stubs, other, run_dir, "--mode", "full") == 1
    assert attempts() == ("wait", 1)
    assert run(merge_stubs, other, run_dir, "--mode", "full") == 0
    assert attempts() == ("not repaired", 2)


def test_a_repair_that_failed_is_tried_again_and_one_not_possible_is_not(
    tmp_path, merge_stubs, env, monkeypatch
):
    off = {"full.final_retrain.enabled": False}
    cfg = _world(tmp_path, monkeypatch, {"round_2": [LARGE]}, **off)
    run_dir = tmp_path / "run"
    r2 = run_dir / "round_2"
    d = r2 / cm.REPAIR_DIR
    monkeypatch.setenv("STUB_REPAIR_RC", "1")
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "repair failed" and "rc=1" in dec["reason"]
    monkeypatch.delenv("STUB_REPAIR_RC")
    (r2 / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env)[n:])["repair"]) == 1
    assert json.loads((d / cm.DECISION).read_text())["outcome"] == "repaired"
    # the repair script finds no repair possible (its exit 3): not repaired,
    # for the script's own reason, and decided
    none = tmp_path / "none"
    none.mkdir()
    cfg = _world(none, monkeypatch, {"round_2": [LARGE]}, **off)
    monkeypatch.delenv("STUB_REPAIR")
    rd = none / "run" / "round_2"
    assert run(merge_stubs, cfg, none / "run", "--mode", "full") == 0
    dec = json.loads((rd / cm.REPAIR_DIR / cm.DECISION).read_text())
    assert dec["outcome"] == "not repaired"
    assert dec["reason"] == "no repair: cannot repair: the stub repairs nothing"
    assert dec["repair"]["outcome"] == "cannot repair: the stub repairs nothing"
    (rd / ro.ROUND_MARKER).unlink()
    n = len(calls(env))
    assert run(merge_stubs, cfg, none / "run", "--mode", "full") == 0
    assert "repair" not in by_stage(calls(env)[n:])
    assert json.loads((rd / cm.REPAIR_DIR / cm.DECISION).read_text()) == dec
    # no bundle of the run is the incumbent: not repaired, and decided
    alone = tmp_path / "alone"
    alone.mkdir()
    cfg = _world(alone, monkeypatch, {"round_2": [LARGE]}, **off)
    monkeypatch.setattr(cm, "find_incumbent", lambda *a, **k: None)
    rd = alone / "run" / "round_2"
    assert run(merge_stubs, cfg, alone / "run", "--mode", "full") == 0
    dec = json.loads((rd / cm.REPAIR_DIR / cm.DECISION).read_text())
    assert dec["outcome"] == "not repaired"
    assert dec["reason"].startswith(
        "no bundle of the run is the one the per-cell gate compared with"
    )
    (rd / ro.ROUND_MARKER).unlink()
    assert run(merge_stubs, cfg, alone / "run", "--mode", "full") == 0
    assert json.loads((rd / cm.REPAIR_DIR / cm.DECISION).read_text()) == dec


def test_an_incumbent_the_audit_set_scored_elsewhere_is_not_carried_from(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The incumbent's directory now points the audit set at another model
    than the one its gate compared with: the repair is not judged on it."""
    cfg = _world(
        tmp_path,
        monkeypatch,
        {"round_2": [LARGE]},
        **{"full.final_retrain.enabled": False},
    )
    stub = Path(str(ro.AUDIT_EVAL_SCRIPT))
    old = 'str(Path(rd) / "stage05" / "models.pt"))'
    assert old in stub.read_text()
    stub.write_text(
        stub.read_text().replace(old, 'str(Path(rd) / "stage06" / "elsewhere.bin"))')
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    d = run_dir / "round_2" / cm.REPAIR_DIR
    dec = json.loads((d / cm.DECISION).read_text())
    assert dec["outcome"] == "not repaired" and "judged" not in dec
    assert dec["reason"].startswith("the audit set scored ")
    assert "elsewhere.bin" in dec["reason"]
    assert "not the model its gate compared with" in dec["reason"]
    assert not (d / cm.UNREPAIRED).exists()


def test_rounds_on_a_repaired_final_retrain_keep_its_revocations(
    tmp_path, merge_stubs, env, monkeypatch
):
    """A final retrain ships repaired by a carry that revoked LARGE's split,
    and the run goes on: round_2 builds on it (its stage01 samples on the
    final retrain's leaves, its stage04b evaluates the final retrain's
    bundle), judges the cells it carries on a tree without the split, and
    takes the revocation into its own stage05, also when it resumes after
    its stage05, so that later rounds leave the split out."""
    monkeypatch.setenv("STUB_SPLITS", json.dumps({"round_1": {LARGE: SPLIT.to_json()}}))
    cfg = _final_world(tmp_path, monkeypatch, [LARGE], **{"full.total_rounds": 2})
    monkeypatch.setenv("STUB_REPAIR_CARRIED", json.dumps(CARRIED))
    monkeypatch.setenv("STUB_REPAIR_REVOKED", json.dumps([SPLIT.to_json()]))
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert json.loads((final / ro.FinalRetrain.DECISION).read_text())["accepted"]
    assert sc.revoked_parents(final) == [LARGE]
    monkeypatch.setenv("STUB_SPLITS", "{}")
    monkeypatch.setenv("STUB_FAIL", "stage06@round_2")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    trees = []
    real = ro.carried_cells

    def carried(lineage, cells, tree):
        trees.append(dict(tree))
        return real(lineage, cells, tree)

    monkeypatch.setattr(ro, "carried_cells", carried)
    more = _final_config(tmp_path, **{"audit.v2": "final", "full.total_rounds": 4})
    n = len(calls(env))
    assert run(merge_stubs, more, run_dir, "--mode", "full") != 0
    r = [run_dir / f"round_{k}" for k in range(4)]
    (s04b,) = [
        c
        for c in by_stage(calls(env)[n:])["stage04b"]
        if arg(c, "--out-dir") == str(r[2] / "stage04b")
    ]
    assert arg(s04b, "--models-dir") == str(final / "stage05")
    assert trees and all(LARGE not in t for t in trees)
    (s01,) = [
        c
        for c in by_stage(calls(env)[n:])["stage01"]
        if arg(c, "--out-dir") == str(r[2] / "stage01")
    ]
    assert arg(s01, "--base-bundle-dir") == str(final)
    assert values(s01, "--prior-round-dir") == [str(r[0]), str(r[1])]
    # stopped after its stage05, round_2 takes the revocation as it resumes
    (r[2] / "stage05" / sc.REVOKED_SPLITS).unlink(missing_ok=True)
    assert run(merge_stubs, more, run_dir, "--mode", "full") == 0
    doc = sc.revocations(r[2])
    assert [x["cell"] for x in doc["revoked"]] == [LARGE]
    assert doc["inherited_from"] == ro.FINAL_RETRAIN_DIR
    assert len(trees) >= 2 and all(LARGE not in t for t in trees)
    assert LARGE not in sc.load_cumulative_split_tree(r)
    assert sc.barred_base_cells(r) == {LARGE: "round_2"}
    # round_3 builds on round_2, one of its prior rounds
    (s01,) = [
        c
        for c in by_stage(calls(env)[n:])["stage01"]
        if arg(c, "--out-dir") == str(r[3] / "stage01")
    ]
    assert arg(s01, "--base-bundle-dir") is None


def test_a_final_retrain_repairs_against_the_bundle_found_by_content(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The bundle the final retrain's gate scored is found by its sha256 in
    another directory than the one the run ships from: the repair carries
    from it, and the audit set scores it there."""
    cfg = _final_world(tmp_path, monkeypatch, [TINY])
    run_dir = tmp_path / "run"
    r2, final = run_dir / "round_2", run_dir / ro.FINAL_RETRAIN_DIR
    moved = run_dir / "stale" / "20000101_000000" / ro.FINAL_RETRAIN_DIR
    real = cm.find_incumbent

    def elsewhere(run, **kw):
        inc = real(run, **kw)
        if inc is None or Path(inc.dir) != r2:
            return inc
        if not moved.exists():
            shutil.copytree(r2, moved, symlinks=True)
        return cm.Incumbent(moved, inc.staged_sha256, inc.models_sha256, inc.verified)

    monkeypatch.setattr(cm, "find_incumbent", elsewhere)
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    (rep,) = st["repair"]
    assert arg(rep, "--baseline-dir") == str(moved)
    rd = final / ro.FinalRetrain.REPAIR_DIR
    (ev,) = [c for c in st["audit_eval"] if arg(c, "--out") == str(rd / "audit.json")]
    assert values(ev, "--round-dir") == [str(moved)]
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] is True
    assert dec["repair"]["incumbent"]["dir"] == str(moved.relative_to(run_dir))
    assert dec["repair"]["installed"]["carried_from"] == moved.name


def test_a_final_retrain_whose_incumbent_is_not_found_stays_out(
    tmp_path, merge_stubs, env, monkeypatch
):
    cfg = _final_world(tmp_path, monkeypatch, [TINY])
    monkeypatch.setattr(cm, "find_incumbent", lambda *a, **k: None)
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    dec = json.loads(
        (run_dir / ro.FINAL_RETRAIN_DIR / ro.FinalRetrain.DECISION).read_text()
    )
    assert dec["accepted"] is False and dec["repair"] == {}
    assert (
        "no repair: no bundle of the run is the one the per-cell gate compared with"
        in dec["reason"]
    )
    assert "repair" not in by_stage(calls(env))


def test_a_final_retrain_repair_scored_against_another_model_stays_out(
    tmp_path, merge_stubs, env, monkeypatch
):
    """The audit set scores another model for the incumbent when it judges
    the repair than the one the final retrain's gate compared with: the
    final retrain stays out, unrepaired."""
    cfg = _final_world(tmp_path, monkeypatch, [TINY])
    stub = Path(str(ro.AUDIT_EVAL_SCRIPT))
    old = 'str(Path(rd) / "stage05" / "models.pt"))'
    assert old in stub.read_text()
    stub.write_text(
        stub.read_text().replace(
            old,
            'str(Path(rd) / ("stage06/elsewhere.bin" if values("--out")[0]'
            '.endswith("/repair/audit.json") else "stage05/models.pt")))',
        )
    )
    run_dir = tmp_path / "run"
    assert run(merge_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] is False and "installed" not in dec["repair"]
    assert "; no repair: the audit set scored " in dec["reason"]
    assert "elsewhere.bin" in dec["reason"]
    assert not (final / ro.FinalRetrain.REPAIR_DIR / cm.UNREPAIRED).exists()
