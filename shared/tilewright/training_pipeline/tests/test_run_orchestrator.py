# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Orchestrator behavior with stub stage scripts (no bench, no training)."""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import run_orchestrator as ro
from lib import run_layout
from lib import sustained as suslib

STEM = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950"

STUB = r"""
import json, os, signal, sys
from pathlib import Path

KEY = __KEY__
argv = sys.argv[1:]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


lease_dir = os.environ.get("TILEWRIGHT_GPU_LEASE_DIR")
leases = {
    p.name: p.read_text() for p in sorted(Path(lease_dir).glob("gpu*.lock"))
} if lease_dir else {}
if os.environ.get("STUB_ENV_LOG"):
    names = json.loads(os.environ.get("STUB_ENV", "[]"))
    with open(os.environ["STUB_ENV_LOG"], "a") as f:
        f.write(json.dumps({"stage": KEY, "env": {k: os.environ.get(k) for k in names}})
                + "\n")
if os.environ.get("STUB_LOG"):
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps({"stage": KEY, "argv": argv, "leases": leases}) + "\n")
if os.environ.get("STUB_IGNORE_TERM") == KEY:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.environ.get("STUB_LEAVE") == KEY:
    import subprocess
    left = subprocess.Popen(["sleep", "600"], start_new_session=True)
    Path(os.environ["STUB_PID_FILE"]).write_text(str(left.pid))
if os.environ.get("STUB_SLEEP") == KEY:
    import time
    Path(os.environ["STUB_PID_FILE"]).write_text(str(os.getpid()))
    time.sleep(600)
out = opt("--out-dir") or opt("--output-dir")
rnd = next((p for p in Path(out).parts if p.startswith("round_")), "validate")
for spec in filter(None, os.environ.get("STUB_FAIL", "").split(",")):
    name, _, how = spec.partition(":")
    if name not in (KEY, KEY + "@" + rnd):
        continue
    once = os.environ.get("STUB_FAIL_ONCE_DIR")
    if once:
        marker = Path(once) / (KEY + rnd + ".failed")
        if marker.exists():
            continue
        marker.write_text("1")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "partial.txt").write_text("partial")
    if KEY == "stage05":
        (Path(out) / "metrics.json").write_text(json.dumps({"cells_failed": [],
            "cells_not_trained": [{"cell": "X#M>1", "reason": "too few gemms (2)"}],
            "n_cells_trained": int(os.environ.get("STUB_TRAINED", "4"))}))
        if how == "3" and not os.environ.get("STUB_NO_BUNDLE"):
            (Path(out) / "models.pt").write_text("models of " + rnd)
            (Path(out) / "cells.json").write_text(
                json.dumps({"cells": [], "model_labels": ["X"]}))
    if how == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
    if how.startswith("SIG"):
        os.kill(os.getpid(), getattr(signal, how))
    sys.exit(int(how or 1))
o = Path(out)
o.mkdir(parents=True, exist_ok=True)
if KEY == "stage00":
    blocked = json.loads(os.environ.get("STUB_BLOCKED", "[]"))
    (o / "blocklist.json").write_text(json.dumps({"schema": "tilewright_blocklist_v1",
        "entries": [{"name": n} for n in blocked]}))
elif KEY == "stage00a":
    held = json.loads(os.environ.get("STUB_QUARANTINED", "[]"))
    (o / "quarantine.json").write_text(json.dumps({"schema": "tilewright_blocklist_v1",
        "entries": [{"name": n} for n in held]}))
elif KEY == "stage01":
    (o / "shapes.yaml").write_text("- {M: 64, N: 64, K: 64, batch_count: 1}\n")
elif KEY == "stage03":
    (o / "logs").mkdir()
    (o / "logs" / "block_0000.log").write_text("log\n")
elif KEY == "stage03d":
    for r in range(2):
        (o / "runs" / f"run_{r}" / "stage03" / "logs").mkdir(parents=True)
elif KEY == "stage04":
    (o / "chunk_0000.csv").write_text("m,n,k,batch_count\n64,64,64,1\n")
elif KEY == "stage04b":
    retrain = json.loads(os.environ.get("STUB_RETRAIN", "{}")).get(rnd, "Tiny|Tiny|TinyK|Bnone")
    se = float(os.environ.get("STUB_SEL_EFF", "0.9"))
    (o / "decisions.json").write_text(json.dumps({
        "models_dir": opt("--models-dir"),
        "aggregate": {"global_sel_eff_new": se, "n_gemms_evaluated": 40},
        "per_cell": [
        {"cell": "Tiny|Tiny|TinyK|Bnone", "sel_eff_new": se, "n_eval": 10,
         "n_training_gemms": 40},
        {"cell": "Large|Large|LargeK|Bnone", "sel_eff_new": se, "n_eval": 30,
         "n_training_gemms": 40}]}))
    (o / "retrain_cells.txt").write_text(retrain)
    splits = json.loads(os.environ.get("STUB_SPLITS", "{}")).get(rnd, {})
    (o / "splits.json").write_text(json.dumps({"splits": splits}))
    if os.environ.get("STUB_GATE"):
        for sub in ("gate", "train"):
            (o / sub).mkdir()
            (o / sub / "chunk_0000.csv").write_text("m,n,k,batch_count\n64,64,64,1\n")
elif KEY == "stage05":
    (o / "models.pt").write_text("models of " + o.parent.name)
    only = opt("--only-cells")
    trained = [c for c in json.loads(os.environ.get("STUB_CELLS", "[]"))
               if only is None or c in only.split(",")]
    labels = set(trained)
    priors = [argv[i + 1] for i, a in enumerate(argv) if a == "--prior-round-dir"]
    if priors:
        prior = json.loads((Path(priors[-1]) / "stage05" / "cells.json").read_text())
        labels.update(prior.get("model_labels", []))
    (o / "cells.json").write_text(json.dumps(
        {"cells": [{"label": c} for c in trained], "model_labels": sorted(labels)}))
    (o / "member_order.json").write_text(json.dumps({"library_digest": rnd, "keys": {}}))
    if os.environ.get("STUB_STAGES"):
        sys.path.insert(0, os.environ["STUB_STAGES"])
        import stage05_train
        stage05_train.write_json(o / stage05_train.MODEL_OPTIONS_FILE,
                                 stage05_train.command_model_options(argv))
elif KEY == "stage06":
    rec = {"train_dir": opt("--train-dir")}
    if os.environ.get("STUB_STAGED"):
        import hashlib
        model = o / "staged" / "model.tilewright.bin"
        model.parent.mkdir(parents=True)
        model.write_bytes(("staged model of " + rnd).encode())
        rec.update(sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   staged_model=str(model),
                   targets=[{"dir": os.environ["STUB_STAGED"],
                             "library_stem": opt("--library-stem"),
                             "weights_file": "w.tilewright.bin", "staged": str(model)}])
        if os.environ.get("STUB_SIDECAR"):
            side = model.parent / (opt("--library-stem") + ".tilewright.blocklist")
            side.write_text(os.environ["STUB_SIDECAR"])
            rec["targets"][0]["blocklist"] = side.name
    if "--gate-csv-dir" in argv:
        se = float(os.environ.get("STUB_GATE_SE", "0.9"))
        target = opt("--gate-target")
        passed = (
            target is None
            or se >= float(target)
            or opt("--gate-fallback") == "not-worse"
            and (opt("--gate-baseline-train-dir") is None
                 or os.environ.get("STUB_NOT_WORSE") == "1")
        )
        rec.update(deployed=passed, gate={"sel_eff": se, "passed": passed})
        if not passed:
            rec["not_deployed"] = "below the target"
        (o / "gate.json").write_text(json.dumps({
            "n_evaluated": 10, "sel_eff": se, "n_miss": 1,
            "per_cell": {"Tiny|Tiny|TinyK|Bnone": {
                "sel_eff_new": se, "n_eval": 10, "paired_sel_eff_new": se,
                "origami_sel_eff_new": 0.5, "origami_n_eval": 10}}}))
    (o / "deploy_record.json").write_text(json.dumps(rec))
elif KEY in ("stage07", "stage08"):
    (o / "summary.json").write_text(json.dumps({"ok": True}))
    sys.exit(int(os.environ.get("STUB_RC_" + KEY, "0")))
"""


@pytest.fixture
def stubs(tmp_path):
    d = tmp_path / "stages"
    d.mkdir()
    for key, sdef in ro.STAGE_DEFS.items():
        (d / sdef.script).write_text(STUB.replace("__KEY__", repr(key)))
    return d


def make_config(tmp_path, **overrides):
    cfg = {
        "config_id": "orch_test",
        "arch": "gfx950",
        "hipblaslt": {
            "a_type": "bf16_r",
            "b_type": "bf16_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scale_type": "f32_r",
            "compute_type": "c_f32_r",
            "transA": "T",
            "transB": "N",
            "library_stem": STEM,
        },
        "runtime": {
            "devices": [0, 1],
            "rocm_libraries_root": str(tmp_path / "bench"),
            "gpu_lease_wait_s": 0,
            "stage_retry_backoff_s": 0,
        },
        "seed": {"target_per_cell": 40, "shapes_per_cell": 40, "seed": 7},
        "train": {"min_cell_gemms": 20},
        "validate": {"min_cell_gemms": 10, "split_after_attempts": 0},
        "full": {"total_rounds": 3},
        "deploy": {
            "weight_dtype": "int4",
            "rocm_libraries_to_deploy": str(tmp_path / "deploy"),
            "apply_patch": True,
            "build_after_apply": True,
        },
        "stage07": {
            "parity_sample_trained": 0,
            "bench_yamls": [str(tmp_path / "w.yaml")],
        },
        "stage08": {"datasets": [{"name": "ds", "path": str(tmp_path / "ds.yaml")}]},
    }
    for dotted, value in overrides.items():
        cur = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    path = tmp_path / "orch_test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert ro.validate_config(cfg)[0] == []
    return path


def run(stubs, cfg_path, run_dir, *extra):
    return ro.main(
        [
            "--config",
            str(cfg_path),
            "--run-dir",
            str(run_dir),
            "--skip-preflight",
            "--no-report",
            "--stages-dir",
            str(stubs),
            *extra,
        ]
    )


def calls(log):
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def arg(entry, name):
    a = entry["argv"]
    return a[a.index(name) + 1] if name in a else None


@pytest.fixture
def env(monkeypatch, tmp_path):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    for k in (
        "STUB_FAIL",
        "STUB_IGNORE_TERM",
        "STUB_LEAVE",
        "STUB_NO_BUNDLE",
        "STUB_TRAINED",
        "STUB_RETRAIN",
        "STUB_FAIL_ONCE_DIR",
        "STUB_SEL_EFF",
        "STUB_GATE",
        "STUB_GATE_SE",
        "STUB_NOT_WORSE",
        "STUB_CELLS",
        "STUB_SPLITS",
        "STUB_STAGES",
        "STUB_STAGED",
        "STUB_SIDECAR",
        "STUB_BLOCKED",
        "STUB_QUARANTINED",
        "STUB_RC_stage07",
        "STUB_RC_stage08",
    ):
        monkeypatch.delenv(k, raising=False)
    return log


def stage05_runs(log):
    return {
        Path(arg(c, "--output-dir")).parent.name: c
        for c in calls(log)
        if c["stage"] == "stage05"
    }


def test_carried_cells_trained_with_other_model_options_are_retrained(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_2": ""}))
    run_dir = tmp_path / "run"

    def config(**train):
        options = {f"train.{k}": v for k, v in train.items()}
        return make_config(tmp_path, **{"full.total_rounds": 4, **options})

    changed = ("--mode", "active", "--allow-code-change")
    assert run(stubs, config(cascade_k=32), run_dir, "--mode", "initial") == 0
    assert run(stubs, config(cascade_k=32, epochs=7), run_dir, *changed) == 0
    new = config(cascade_k=16, stage1_recall_weight=0.1)
    capsys.readouterr()
    assert run(stubs, new, run_dir, *changed) == 0
    out = capsys.readouterr().out
    assert run(stubs, new, run_dir, "--mode", "active") == 0
    trained = stage05_runs(env)
    assert sorted(trained) == ["round_0", "round_1", "round_2", "round_3"]
    assert arg(trained["round_1"], "--only-cells") == TINY
    assert "--only-cells" not in trained["round_2"]["argv"]
    assert arg(trained["round_2"], "--current-round-dir") == str(run_dir / "round_2")
    assert not (run_dir / "round_2" / "stage05" / "carried_forward.json").exists()
    assert (
        "round_2 retrains 2 carried cells: cascade_k 32 -> 16, "
        "stage1_recall_weight 0 -> 0.1" in out
    )
    assert arg(trained["round_3"], "--only-cells") == TINY


def test_carried_cells_are_judged_by_the_options_their_round_recorded(
    tmp_path, stubs, env, monkeypatch, capsys, pipeline_dir
):
    import stage05_train as s5

    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    monkeypatch.setenv("STUB_STAGES", str(pipeline_dir / "stages"))
    monkeypatch.setenv(
        "STUB_RETRAIN", json.dumps({f"round_{r}": "" for r in (2, 3, 4)})
    )
    cfg = make_config(tmp_path, **{"full.total_rounds": 5})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    r0, r1 = run_dir / "round_0" / "stage05", run_dir / "round_1" / "stage05"

    def edit(path, change):
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    edit(r0 / ro.STAGE_MARKER, lambda m: m["cmd"].extend(["--boundary-margin", "1"]))
    edit(r1 / s5.MODEL_OPTIONS_FILE, lambda o: o.pop("stage1_recall_weight"))
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert (run_dir / "round_2" / "stage05" / "carried_forward.json").is_file()

    (r0 / s5.MODEL_OPTIONS_FILE).unlink()
    capsys.readouterr()
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert "round_3 retrains 1 carried cell: boundary_margin 1 -> 0" in (
        capsys.readouterr().out
    )
    edit(r1 / ro.STAGE_MARKER, lambda m: m.update(cmd=None))
    (r1 / s5.MODEL_OPTIONS_FILE).unlink()
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert "round_4 retrains 1 carried cell: 1 trained with unknown options" in (
        capsys.readouterr().out
    )
    trained = stage05_runs(env)
    assert sorted(trained) == ["round_0", "round_1", "round_3", "round_4"]
    assert arg(trained["round_3"], "--only-cells") == LARGE
    assert arg(trained["round_4"], "--only-cells") == TINY


def test_a_cell_the_round_splits_is_not_carried(tmp_path, stubs, env, monkeypatch):
    from lib import subcells as sc

    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    marker = run_dir / "round_0" / "stage05" / ro.STAGE_MARKER
    rec = json.loads(marker.read_text())
    marker.write_text(
        json.dumps(dict(rec, cmd=rec["cmd"] + ["--boundary-margin", "1"]))
    )
    rule = sc.build_split_rule(LARGE, "M", 4096)
    children = f"{rule.lo_label},{rule.hi_label}"
    monkeypatch.setenv("STUB_SPLITS", json.dumps({"round_2": {LARGE: rule.to_json()}}))
    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_2": children}))
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert arg(stage05_runs(env)["round_2"], "--only-cells") == children


def test_an_interrupted_round_retrains_the_same_carried_cells(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    run_dir = tmp_path / "run"
    first = make_config(tmp_path, **{"full.total_rounds": 2})
    assert run(stubs, first, run_dir, "--mode", "initial") == 0
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 2, "train.boundary_margin": 0.5}
    )
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    capsys.readouterr()
    assert run(stubs, cfg, run_dir, "--mode", "active", "--allow-code-change") == 1
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    attempts = [
        c
        for c in calls(env)
        if c["stage"] == "stage05"
        and Path(arg(c, "--output-dir")).parent.name == "round_1"
    ]
    assert len(attempts) == 2
    assert all("--only-cells" not in c["argv"] for c in attempts)
    assert (run_dir / "round_1" / ro.ROUND_MARKER).is_file()
    line = "round_1 retrains 1 carried cell: boundary_margin 0 -> 0.5"
    assert capsys.readouterr().out.count(line) == 2


def test_an_enhance_round_retrains_carried_cells_trained_with_other_options(
    tmp_path, stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    _, run_dir = initial_run(stubs, tmp_path)
    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_1": TINY}))
    int8 = make_config(
        tmp_path, **{"full.total_rounds": 1, "deploy.weight_dtype": "int8"}
    )
    assert (
        run(
            stubs,
            int8,
            run_dir,
            "--mode",
            "enhance",
            "--cells",
            TINY,
            "--allow-code-change",
        )
        == 0
    )
    assert "--only-cells" not in stage05_runs(env)["round_1"]["argv"]
    record = json.loads((run_dir / "round_1" / ro.ENHANCE_RECORD).read_text())
    assert record["retrain"] == [TINY]


def test_full_run_carries_models_through_an_all_pass_round(
    tmp_path, stubs, env, monkeypatch, pipeline_dir
):
    import stage05_train as s5

    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_2": ""}))
    monkeypatch.setenv("STUB_STAGES", str(pipeline_dir / "stages"))
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    for r in range(3):
        assert (run_dir / f"round_{r}" / ro.ROUND_MARKER).is_file()
    trained = [arg(c, "--output-dir") for c in log if c["stage"] == "stage05"]
    assert [Path(p).parent.name for p in trained] == ["round_0", "round_1"]
    carried = run_dir / "round_2" / "stage05"
    assert (carried / "models.pt").read_text() == "models of round_1"
    assert (carried / "carried_forward.json").is_file()
    source = run_dir / "round_1" / "stage05"
    for name in (s5.MODEL_OPTIONS_FILE, s5.MEMBER_ORDER_FILE):
        assert (carried / name).read_text() == (source / name).read_text()
    assert (
        json.loads((carried / s5.MEMBER_ORDER_FILE).read_text())["library_digest"]
        == "round_1"
    )
    assert (carried / ro.STAGE_MARKER).is_file()
    deploys = [c for c in log if c["stage"] == "stage06"]
    assert len(deploys) == 1
    assert arg(deploys[0], "--train-dir") == str(carried)
    assert (
        "--apply-patch" in deploys[0]["argv"]
        and "--build-after-apply" in deploys[0]["argv"]
    )
    assert arg(deploys[0], "--library-stem") == STEM

    by_stage = {}
    for c in log:
        by_stage.setdefault(c["stage"], []).append(c)
    assert [arg(c, "--round-index") for c in by_stage["stage01"]] == ["0", "1", "2"]
    assert "stage02" not in by_stage
    assert arg(by_stage["stage03"][0], "--in-dir").endswith("round_0/stage01")
    assert arg(by_stage["stage04"][0], "--library-stem") == STEM
    for key in ("stage04b", "stage05"):
        for c in by_stage[key]:
            assert arg(c, "--weight-dtype") == "int4"
            assert arg(c, "--library-stem") == STEM
    snaps = sorted((run_dir / "configs").glob("*_full.yaml"))
    assert len(snaps) == 1
    for c in log:
        if "--config-yaml" in c["argv"]:
            assert arg(c, "--config-yaml") == str(snaps[0])
    s04b = by_stage["stage04b"][1]
    assert arg(s04b, "--models-dir").endswith("round_1/stage05")
    assert arg(s04b, "--train-min-cell-gemms") == "20"
    assert "--final-round" in s04b["argv"]
    assert arg(s04b, "--prior-models").endswith("round_0/stage05/models.pt")
    validate = [c["stage"] for c in log if c["stage"] in ("stage07", "stage08")]
    assert validate == ["stage07", "stage08"]
    s07 = by_stage["stage07"][0]
    assert arg(s07, "--deploy-record").endswith("round_2/stage06/deploy_record.json")
    timing_dirs = [
        s07["argv"][i + 1]
        for i, a in enumerate(s07["argv"])
        if a == "--timings-csv-dir"
    ]
    assert timing_dirs == [str(run_dir / f"round_{r}" / "stage04") for r in range(3)]
    assert arg(by_stage["stage08"][0], "--train-dir") == str(carried)
    assert arg(by_stage["stage08"][0], "--pick-measurement") == "warmup"
    assert "--origami-reference" not in by_stage["stage08"][0]["argv"]
    assert arg(s04b, "--gate-frac") == "0.2" and arg(s04b, "--seed") == "7"
    assert arg(s04b, "--pick-measurement") == "warmup"
    assert all(arg(c, "--pick-measurement") == "warmup" for c in by_stage["stage05"])
    assert "--gate-csv-dir" not in deploys[0]["argv"]
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == "round_2"
    assert breakdown["source"] == "stage04b" and breakdown["n_cells"] == 2
    assert [m["round"] for m in breakdown["measurements"]] == ["round_2"]


def by_stage(log):
    out = {}
    for c in log:
        out.setdefault(c["stage"], []).append(c)
    return out


def test_each_round_ships_the_bundle_its_gate_measured(
    tmp_path, stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    r1, r2 = run_dir / "round_1", run_dir / "round_2"
    trained = [arg(c, "--output-dir") for c in st["stage05"]]
    assert [Path(p).parent.name for p in trained] == ["round_0", "round_1", "round_2"]
    for c, rd in zip(st["stage05"][1:], (r1, r2)):
        dirs = [
            c["argv"][i + 1]
            for i, a in enumerate(c["argv"])
            if a == "--enriched-csv-dir"
        ]
        assert dirs[-1] == str(rd / "stage04b" / "train")
        assert str(run_dir / "round_0" / "stage04") in dirs
    d1, d2 = st["stage06"]
    assert arg(d1, "--train-dir") == str(r1 / "stage05")
    assert arg(d1, "--gate-csv-dir") == str(r1 / "stage04b" / "gate")
    assert arg(d1, "--gate-target") == "0.95" and "--gate-fallback" not in d1["argv"]
    assert arg(d1, "--library-dir").endswith("library/gfx950")
    assert arg(d2, "--gate-fallback") == "not-worse"
    assert arg(d2, "--gate-baseline-train-dir") == str(r1 / "stage05")
    for rd in (r1, r2):
        marker = json.loads((rd / ro.ROUND_MARKER).read_text())
        assert marker["deployed"] and marker["gate_sel_eff"] == pytest.approx(0.97)
        assert marker["held_out_sel_eff"] == pytest.approx(0.9)
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["source"] == "deploy_gate"
    assert breakdown["shipped"]["round"] == "round_2"
    assert breakdown["held_out_sel_eff"] == pytest.approx(0.97)
    assert breakdown["per_cell"]["Tiny|Tiny|TinyK|Bnone"]["n_eval"] == 10
    assert breakdown["paired_origami"]["n"] == 10 and breakdown["final_bundle"] is None


def test_a_round_below_its_gate_target_does_not_deploy(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    assert [Path(arg(c, "--out-dir")).parent.name for c in st["stage06"]] == [
        "round_1",
        "round_2",
    ]
    m1 = json.loads((run_dir / "round_1" / ro.ROUND_MARKER).read_text())
    assert m1["deployed"] is False
    assert "--gate-baseline-train-dir" not in st["stage06"][1]["argv"]
    s07 = st["stage07"][0]
    assert arg(s07, "--deploy-record").endswith("round_2/stage06/deploy_record.json")
    out = capsys.readouterr()
    assert "round_1 not deployed: below the target" in out.err
    assert "origami" not in out.out.lower()


def test_the_final_round_keeps_a_worse_bundle_out(tmp_path, stubs, env, monkeypatch):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--deploy") == 0
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    (d1,) = st["stage06"]
    assert arg(d1, "--gate-baseline-train-dir") == str(run_dir / "round_0" / "stage05")
    rec = ro.latest_deploy_record(ro.complete_prefix(run_dir))
    assert rec == run_dir / "round_0" / "stage06" / "deploy_record.json"
    assert "stage07" not in st and "stage08" not in st
    vdir = run_dir / "validate"
    assert ro.validated_by(vdir / "stage07")["deploy_record"] == str(rec)
    assert ro.validated_by(vdir / "stage08")["train_dir"] == str(
        run_dir / "round_0" / "stage05"
    )
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == "round_0"
    assert breakdown["final_bundle"]["round"] == "round_1"
    assert breakdown["final_bundle"]["gate_sel_eff"] == pytest.approx(0.5)
    monkeypatch.setenv("STUB_NOT_WORSE", "1")
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full", "--from-round", "1") == 0
    rec = ro.latest_deploy_record(ro.complete_prefix(run_dir))
    assert rec == run_dir / "round_1" / "stage06" / "deploy_record.json"
    st = by_stage(calls(env)[n:])
    assert arg(st["stage07"][0], "--deploy-record") == str(rec)
    assert arg(st["stage08"][0], "--train-dir") == str(run_dir / "round_1" / "stage05")


def test_the_reference_flag_reaches_the_stages_and_the_report(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full", "--origami-reference") == 0
    assert "[reference] Origami on the same 10 GEMMs" in capsys.readouterr().out
    st = by_stage(calls(env))
    for key in ("stage04b", "stage06", "stage07", "stage08"):
        assert all("--origami-reference" in c["argv"] for c in st[key]), key
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ro.subprocess, "run", fake_run)
    ro.emit_report(run_dir, origami_reference=True)
    ro.emit_report(run_dir)
    assert seen[0][-1] == "--origami-reference" and "--origami-reference" not in seen[1]


def test_shipped_numbers_follow_the_deployed_bundle(tmp_path):
    run_dir = tmp_path / "run"
    rows = [{"cell": "A", "sel_eff_new": 0.8, "n_eval": 20, "n_miss": 2}]
    for r in range(4):
        rd = run_dir / f"round_{r}"
        (rd / "stage05").mkdir(parents=True)
        (rd / "stage05" / "models.pt").write_text(f"bundle {min(r, 2)}")
        if r:
            (rd / "stage04b").mkdir()
            prior = run_dir / f"round_{r - 1}" / "stage05"
            (rd / "stage04b" / "decisions.json").write_text(
                json.dumps(
                    {
                        "models_dir": str(prior),
                        "aggregate": {
                            "global_sel_eff_new": 0.1 * r,
                            "n_gemms_evaluated": 10 * r,
                        },
                        "per_cell": rows,
                    }
                )
            )
    s6 = run_dir / "round_2" / "stage06"
    s6.mkdir()
    (s6 / "deploy_record.json").write_text(
        json.dumps({"train_dir": str(run_dir / "round_2" / "stage05")})
    )
    ro.mark_stage_complete(s6, None)
    got = ro.shipped_bundle_numbers(run_dir)
    assert got["shipped"]["round"] == "round_2" and got["source"] == "stage04b"
    assert [m["round"] for m in got["measurements"]] == ["round_3"]
    assert got["held_out_sel_eff"] == pytest.approx(0.3) and got["n_gemms"] == 30
    assert got["per_cell"]["A"]["n_miss"] == 2
    (s6 / "gate.json").write_text(
        json.dumps({"n_evaluated": 5, "sel_eff": 0.7, "per_cell": {"B": {}}})
    )
    got = ro.shipped_bundle_numbers(run_dir)
    assert got["source"] == "deploy_gate" and got["held_out_sel_eff"] == 0.7
    assert [m["source"] for m in got["measurements"]] == ["deploy_gate", "stage04b"]
    s6b = run_dir / "round_3" / "stage06"
    s6b.mkdir()
    (s6b / "deploy_record.json").write_text(
        json.dumps(
            {"deployed": False, "not_deployed": "below", "gate": {"sel_eff": 0.2}}
        )
    )
    ro.mark_stage_complete(s6b, None)
    got = ro.shipped_bundle_numbers(run_dir)
    assert got["shipped"]["round"] == "round_2"
    assert got["final_bundle"] == {
        "round": "round_3",
        "train_dir": None,
        "deployed": False,
        "reason": "below",
        "gate_sel_eff": 0.2,
    }
    assert ro.shipped_bundle_numbers(tmp_path / "empty")["shipped"] is None


def test_rerun_skips_complete_rounds(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(calls(env)) == n
    assert len(list((run_dir / "configs").glob("*_full*.source.yaml"))) == 2


def test_failed_stage_resumes_with_fresh_output(tmp_path, stubs, env, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "stage04")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, "stage07.bench_yamls": []})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 1
    assert (run_dir / "round_0" / "stage04" / "partial.txt").is_file()
    assert not (run_dir / "round_0" / ro.ROUND_MARKER).exists()
    first = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    second = [c["stage"] for c in calls(env)[first:]]
    assert second[:2] == ["stage04", "stage05"]
    assert not (run_dir / "round_0" / "stage04" / "partial.txt").exists()
    stale = list((run_dir / "round_0" / "stale").glob("*/stage04/partial.txt"))
    assert len(stale) == 1
    assert list((run_dir / "round_0" / "stale").glob("*/stage04.log"))


def test_from_stage_needs_completed_earlier_stages(tmp_path, stubs, env):
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3") == 2
    assert calls(env) == []


def test_explicit_from_stage_reruns_from_there(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    n = len(calls(env))
    assert (
        run(
            stubs,
            cfg,
            run_dir,
            "--mode",
            "full",
            "--from-round",
            "1",
            "--from-stage",
            "4.5",
        )
        == 0
    )
    rerun = [c["stage"] for c in calls(env)[n:]]
    assert rerun[:3] == ["stage04b", "stage05", "stage06"]


def test_mode_preconditions(tmp_path, stubs, env):
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert run(stubs, cfg, run_dir, "--mode", "active") == 2
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 2
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 2
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert (run_dir / "round_1" / ro.ROUND_MARKER).is_file()


@pytest.mark.parametrize(
    "fail,rc",
    [
        ({}, 0),
        ({"STUB_RC_stage07": "3"}, 3),
        ({"STUB_FAIL": "stage08:signal"}, 137),
        ({"STUB_RC_stage07": "1", "STUB_RC_stage08": "4"}, 4),
    ],
)
def test_validate_propagates_stage_exit_codes(
    tmp_path, stubs, env, monkeypatch, fail, rc
):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full", "--deploy") == 0
    for k, v in fail.items():
        monkeypatch.setenv(k, v)
    assert run(stubs, cfg, run_dir, "--mode", "validate") == rc


def test_validate_without_anything_configured(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full", "--deploy") == 0
    bare = make_config(tmp_path, **{"stage07.bench_yamls": [], "stage08.datasets": []})
    assert run(stubs, bare, run_dir, "--mode", "validate") == 2


def test_stage07_needs_a_deployed_model(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, "stage08.datasets": []})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 2
    assert not [c for c in calls(env) if c["stage"] == "stage07"]


def test_failed_deploy_fails_the_round(tmp_path, stubs, env, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "stage06:5")
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 5
    assert (run_dir / "round_0" / ro.ROUND_MARKER).is_file()
    assert not (run_dir / "round_1" / ro.ROUND_MARKER).exists()


def test_each_invocation_gets_its_own_snapshot(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"train.epochs": 11})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    cfg = make_config(tmp_path, **{"train.epochs": 22})
    assert run(stubs, cfg, run_dir, "--mode", "active", "--allow-code-change") == 0
    snaps = {}
    for c in calls(env):
        if c["stage"] == "stage05":
            snap = Path(arg(c, "--config-yaml"))
            snaps[snap] = yaml.safe_load(snap.read_text())["train"]["epochs"]
            assert arg(c, "--epochs") == str(snaps[snap])
    assert sorted(snaps.values()) == [11, 22]
    assert len(list((run_dir / "configs").glob("*.source.yaml"))) == 2


def test_the_snapshot_states_the_bench_timing_it_resolved(tmp_path, stubs, env):
    from lib.bench_yaml import (
        ADAPTIVE_DEFAULTS,
        ADAPTIVE_LONG_DEFAULTS,
        WARMUP_LINE_DEFAULTS,
        render_bench_line,
    )
    from lib.rebench import REREAD_DEFAULTS
    from lib.shapes import Shape

    path = make_config(tmp_path)
    source = yaml.safe_load(path.read_text())
    assert "bench" not in source
    assert run(stubs, path, tmp_path / "run", "--mode", "initial") == 0
    (snap,) = {
        Path(arg(c, "--config-yaml")) for c in calls(env) if arg(c, "--config-yaml")
    }
    text = snap.read_text()
    cfg = yaml.safe_load(text)
    assert cfg["bench"] == {
        "adaptive": ADAPTIVE_DEFAULTS,
        "skip_slow_solution_ratio": 0.0,
        "kernel_order": "shuffled",
        "order_seed": 42,
        "sacrificial_kernel": True,
        "warmup_pass": True,
        "warmup_line": WARMUP_LINE_DEFAULTS,
        "canary": {
            "enabled": False,
            "tolerance": 0.1,
            "timings": 3,
            "backoff_s": 120.0,
        },
        "rebench": {"enabled": False, **REREAD_DEFAULTS},
        "adaptive_long": ADAPTIVE_LONG_DEFAULTS,
        "sustained": suslib.DEFAULTS,
    }
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert head.startswith("Bench timing of this invocation, resolved below:")
    assert "adaptive timing (always on) with warmup_time 1.0," in head
    assert "no skip screen" in head and "no near-winner re-reads" in head
    assert "roofline" not in head
    assert "kernel lists in a per-GEMM shuffle (order_seed 42)" in head
    assert "no sustained finalists" in head
    assert (
        "every bench process starts with a discarded warm-up line of 4 kernel(s) "
        "on 512x512x512" in head
    )
    timing, _, produced = head.partition(" Produced with: ")
    assert timing.rstrip().endswith("; no canary")
    assert produced.startswith("pipeline code ")
    assert ro.validate_config(cfg)[0] == []
    shape = Shape(64, 128, 256, 1)
    assert render_bench_line(shape, cfg) == render_bench_line(shape, source)


def test_every_stage_runs_with_the_runtime_env(tmp_path, stubs, env, monkeypatch):
    seen = tmp_path / "env.jsonl"
    monkeypatch.setenv("STUB_ENV_LOG", str(seen))
    monkeypatch.setenv("STUB_ENV", json.dumps(["TW_STAGE_KNOB", "TW_OTHER"]))
    monkeypatch.setenv("TW_STAGE_KNOB", "from-the-shell")
    monkeypatch.delenv("TW_OTHER", raising=False)
    path = make_config(tmp_path, **{"runtime.env": {"TW_STAGE_KNOB": "on"}})
    assert run(stubs, path, tmp_path / "run", "--mode", "full") == 0
    log = calls(env)
    records = calls(seen)
    assert [r["stage"] for r in records] == [c["stage"] for c in log]
    assert {c["stage"] for c in log} >= {"stage01", "stage03", "stage05", "stage07"}
    assert all(r["env"] == {"TW_STAGE_KNOB": "on", "TW_OTHER": None} for r in records)
    assert os.environ["TW_STAGE_KNOB"] == "from-the-shell"
    (snap,) = {Path(arg(c, "--config-yaml")) for c in log if arg(c, "--config-yaml")}
    text = snap.read_text()
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert "cotenant of this invocation runs with runtime.env: TW_STAGE_KNOB=on" in head
    assert yaml.safe_load(text)["runtime"]["env"] == {"TW_STAGE_KNOB": "on"}


def test_runtime_env_is_validated(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["runtime"]["env"] = {"HIP_VISIBLE_DEVICES": "0", "OK_KNOB": True}
    errors, _ = ro.validate_config(cfg)
    assert any(
        e.startswith("runtime.env.HIP_VISIBLE_DEVICES: the pipeline") for e in errors
    )
    assert any(e.startswith("runtime.env.OK_KNOB: expected") for e in errors)
    cfg["runtime"]["env"] = {"OK_KNOB": 1}
    assert ro.validate_config(cfg)[0] == []


def test_the_warmup_line_keys_are_validated_and_stated(tmp_path):
    cfg, raw = ro.load_config(make_config(tmp_path))
    for bad in ({"kernels": 0}, {"kernels": 65}, {"size": 0}, {"enabled": "yes"}):
        cfg["bench"] = {"warmup_line": bad}
        errors, _ = ro.validate_config(cfg)
        assert [e for e in errors if e.startswith("bench.warmup_line.")], bad
    cfg["bench"] = {"warmup_line": {"enabled": False}}
    assert ro.validate_config(cfg)[0] == []
    text = ro.write_snapshot(tmp_path / "run", cfg, raw, "full").read_text()
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert "; no warm-up line;" in head
    assert yaml.safe_load(text)["bench"]["warmup_line"] == {
        "enabled": False,
        "kernels": 4,
        "size": 512,
    }


def test_the_snapshot_names_the_screen_prepass_of_planned_lines(tmp_path):
    cfg, raw = ro.load_config(make_config(tmp_path))
    cfg["bench"] = {"skip_slow_solution_ratio": 0.45, "rebench": {"enabled": True}}
    cfg["planner"] = {"enabled": True}
    snap = ro.write_snapshot(tmp_path / "run", cfg, raw, "full")
    text = snap.read_text()
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert "skip_slow_solution_ratio 0.45 against every listed kernel's" in head
    assert "skip_screen_prepass on the planner's lines" in head
    assert "behind a sacrificial copy of the first listed kernel" in head
    assert "near-winner re-reads: the candidates within 0.05 of the winner, at" in head
    assert "most 16, read 3x below 10 us, 3x from 1000 us, 1x in between" in head
    bench = yaml.safe_load(text)["bench"]
    assert bench["skip_slow_solution_ratio"] == 0.45
    from lib.rebench import REREAD_DEFAULTS

    assert bench["rebench"] == {"enabled": True, **REREAD_DEFAULTS}


def test_the_planner_is_stage_2(tmp_path, capsys):
    planner = ro.STAGE_DEFS[run_layout.PLAN_STAGE]
    assert (planner.number, planner.script) == (2.0, "stage02_plan_benchmarks.py")
    assert planner.key == "stage02" and "stage01b" not in ro.STAGE_DEFS
    assert ro.TRAINING_STAGE_NUMBERS == (
        1.0,
        2.0,
        3.0,
        3.2,
        3.5,
        3.7,
        4.0,
        4.5,
        5.0,
        6.0,
    )
    assert sorted(s.number for s in ro.STAGE_DEFS.values()) == [
        0.0,
        0.0,
        *ro.TRAINING_STAGE_NUMBERS,
        7.0,
        8.0,
    ]
    base = ["--config", str(tmp_path / "c.yaml")]
    with pytest.raises(SystemExit):
        ro.parse_args([*base, "--mode", "validate", "--from-stage", "2"])
    assert "takes --from-stage 7 or 8" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        ro.parse_args([*base, "--mode", "full", "--from-stage", "7"])
    assert (
        "takes --from-stage 1, 2, 3, 3.2, 3.5, 3.7, 4, 4.5, 5 or 6, or the stage's "
        "name" in capsys.readouterr().err
    )
    for n in ro.TRAINING_STAGE_NUMBERS:
        args = ro.parse_args([*base, "--mode", "full", "--from-stage", f"{n:g}"])
        assert args.from_stage == n
    assert capsys.readouterr().err == ""


def test_from_stage_takes_stage_names_and_the_planners_old_ones(tmp_path, capsys):
    base = ["--config", str(tmp_path / "c.yaml")]
    for mode in (["full"], ["initial"], ["active"], ["enhance", "--cells", "a"]):
        for key in ("stage01", "stage02", "stage03c", "stage04b", "stage06"):
            argv = [*base, "--mode", *mode, "--from-stage", key]
            assert ro.parse_args(argv).from_stage == ro.STAGE_DEFS[key].number
        assert capsys.readouterr().err == ""
        for old in ("1.5", "stage01b"):
            args = ro.parse_args([*base, "--mode", *mode, "--from-stage", old])
            assert args.from_stage == 2.0
            assert (
                f"--from-stage {old} is deprecated: the planner is stage 2 (stage02) "
                f"now; use --from-stage 2" in capsys.readouterr().err
            )
    args = ro.parse_args([*base, "--mode", "validate", "--from-stage", "stage08"])
    assert args.from_stage == 8.0
    for bad, why in (
        ("stage02b", "neither a stage number nor a stage name such as stage02"),
        ("plan", "neither a stage number nor a stage name such as stage02"),
        ("stage00", "or the stage's name"),
        ("0.5", "or the stage's name"),
    ):
        with pytest.raises(SystemExit):
            ro.parse_args([*base, "--mode", "full", "--from-stage", bad])
        assert why in capsys.readouterr().err
    with pytest.raises(SystemExit):
        ro.parse_args([*base, "--mode", "validate", "--from-stage", "stage01b"])
    err = capsys.readouterr().err
    assert "is deprecated" in err and "takes --from-stage 7 or 8" in err


def test_a_config_with_a_probe_section_does_not_start(tmp_path, stubs, env, capsys):
    path = make_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["probe"] = {"duration_us": 2000, "skip_ratio_mode": "sigmoid"}
    cfg["bench"] = {"adaptive": {"enabled": False}}
    path.write_text(yaml.safe_dump(cfg))
    assert run(stubs, path, tmp_path / "run", "--mode", "initial") == 2
    err = capsys.readouterr().err
    assert "probe: the probe stage was removed and adaptive timing is always on" in err
    assert "bench.adaptive.enabled: adaptive timing is always on" in err
    assert calls(env) == [] and not (tmp_path / "run").exists()


def _probe_era(rd):
    """What the removed probe left in a round of a run written before the
    planner became stage02: its stage02 outputs, marker and log."""
    out = rd / "stage02"
    (out / "logs").mkdir(parents=True)
    (out / "shapes_probed.yaml").write_text(
        "- {M: 64, N: 64, K: 64, batch_count: 1, iters: 40, cold_iters: 40}\n"
    )
    (out / "logs" / "probe_0000_lines1-1_gpu0.log").write_text("probe log\n")
    ro.mark_stage_complete(out, ["stage02_probe.py"])
    (rd / "stage02.log").write_text("stage02\n")


def _stage02_files(*rds):
    return {
        p: p.read_bytes()
        for rd in rds
        for p in sorted(rd.rglob("*"))
        if p.is_file() and "stage02" in p.relative_to(rd).parts[0]
    }


@contextlib.contextmanager
def written_before_the_rename():
    """Invocations inside write their run or audit directory as the pipeline
    did before the planner became stage02: in stage01b/, without
    run_layout.json."""
    with pytest.MonkeyPatch.context() as mp:
        for name in ("of", "settle"):
            mp.setattr(run_layout, name, lambda root: run_layout.LEGACY)
        yield


def layout_of_the_past(planner):
    """`written_before_the_rename` for runs whose planner is in stage01b/."""
    if planner == run_layout.LEGACY_PLAN_STAGE:
        return written_before_the_rename()
    return contextlib.nullcontext()


def test_a_run_started_with_the_probe_resumes_without_it(tmp_path, stubs, env):
    """A round of a run written before the planner became stage02 (no
    run_layout.json) interrupted after its probe resumes at stage 3 on
    stage01's lines; every stage02 output of the probe stays as it is,
    unread, also when a round runs again from stage 2."""
    cfg = make_config(tmp_path, **{"full.total_rounds": 3})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    (run_dir / run_layout.LAYOUT_FILE).unlink()
    r0, r1 = run_dir / "round_0", run_dir / "round_1"
    _probe_era(r0)
    (r1 / "stage01").mkdir(parents=True)
    (r1 / "stage01" / "shapes.yaml").write_text(
        "- {M: 64, N: 64, K: 64, batch_count: 1, iters: 50, cold_iters: 50}\n"
    )
    ro.mark_stage_complete(r1 / "stage01", None)
    _probe_era(r1)
    before = _stage02_files(r0, r1)
    assert len(before) == 8
    env.write_text("")

    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    assert "stage02" not in {c["stage"] for c in log}
    resumed = [c for c in log if "round_1" in (arg(c, "--out-dir") or "")]
    assert resumed[0]["stage"] == "stage03"
    assert arg(resumed[0], "--in-dir") == str(r1 / "stage01")
    s04b = [c for c in resumed if c["stage"] == "stage04b"][0]["argv"]
    assert _after(s04b, "--rebench-lines") == [str(r1 / "stage01" / "shapes.yaml")]
    for rd in (r0, r1, run_dir / "round_2"):
        assert (rd / ro.ROUND_MARKER).is_file()
    assert _stage02_files(r0, r1) == before and not list(r1.glob("stale/*/stage02*"))

    env.write_text("")
    argv = ["--mode", "full", "--from-round", "1", "--from-stage", "3"]
    assert run(stubs, cfg, run_dir, *argv) == 0
    redo = [c for c in calls(env) if "round_1" in (arg(c, "--out-dir") or "")]
    assert [c["stage"] for c in redo][:2] == ["stage03", "stage04"]
    assert _stage02_files(r0, r1) == before

    env.write_text("")
    argv = ["--mode", "full", "--from-round", "1", "--from-stage", "2"]
    assert run(stubs, cfg, run_dir, *argv) == 0
    redo = [c for c in calls(env) if "round_1" in (arg(c, "--out-dir") or "")]
    assert [c["stage"] for c in redo][:2] == ["stage03", "stage04"]
    assert _stage02_files(r0, r1) == before and not list(r1.glob("stale/*/stage02*"))
    assert run_layout.of(run_dir) == run_layout.LEGACY
    assert not (run_dir / run_layout.LAYOUT_FILE).exists()


def test_untrained_cells_stop_the_round_and_it_resumes(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == ro.STAGE05_CELLS_NOT_TRAINED
    err = capsys.readouterr().err
    assert "stage05 did not train every requested cell" in err
    assert "X#M>1" in err and "round_1 stops before deployment" in err
    assert not (run_dir / "round_1" / ro.ROUND_MARKER).exists()
    assert not [c for c in calls(env) if c["stage"] in ("stage06", "stage07")]
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    rerun = [c["stage"] for c in calls(env)[n:]]
    assert rerun[:2] == ["stage05", "stage06"]
    assert (run_dir / "round_1" / ro.ROUND_MARKER).is_file()


def test_sigterm_stops_the_stage_and_the_round_resumes(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, "stage07.bench_yamls": []})
    run_dir = tmp_path / "run"
    pid_file = tmp_path / "stage.pid"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(ro.__file__)),
            "--config",
            str(cfg),
            "--run-dir",
            str(run_dir),
            "--skip-preflight",
            "--no-report",
            "--stages-dir",
            str(stubs),
            "--mode",
            "full",
        ],
        env=dict(os.environ, STUB_SLEEP="stage03", STUB_PID_FILE=str(pid_file)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        stage_pid = int(pid_file.read_text())
        assert os.getpgid(stage_pid) == stage_pid != os.getpgid(0)
        proc.send_signal(signal.SIGTERM)
        _out, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 128 + signal.SIGTERM
    assert "SIGTERM: stopped" in err
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(stage_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the interrupted stage is still running")
    assert not ro.stage_complete(run_dir / "round_0" / "stage03")
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]][:2] == ["stage03", "stage04"]


def test_a_run_dir_takes_one_orchestrator_at_a_time(tmp_path, stubs, env, capsys):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, "stage07.bench_yamls": []})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fd = ro.lock_run_dir(run_dir)
    assert fd is not None
    try:
        assert run(stubs, cfg, run_dir, "--mode", "full") == 2
        assert "another orchestrator is using" in capsys.readouterr().err
        assert calls(env) == []
    finally:
        os.close(fd)
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0


HOLD_GPU = """
import sys, time
sys.path.insert(0, {root!r})
from lib import gpu_lease
with gpu_lease.hold_gpus([1], "an experiment", run_dir="/runs/other"):
    print("held", flush=True)
    time.sleep(60)
"""


@pytest.fixture
def visible_gpus(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")


def test_bench_stages_run_under_the_lease_of_their_gpus(
    tmp_path, stubs, env, visible_gpus
):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    by_stage = {c["stage"]: c["leases"] for c in calls(env)}
    held = {name: json.loads(text) for name, text in by_stage["stage03"].items()}
    assert sorted(held) == ["gpu0.lock", "gpu1.lock"]
    for rec in held.values():
        assert rec["pid"] == os.getpid() and rec["purpose"] == "round_0 stage03"
        assert rec["run_dir"] == str(run_dir.resolve())
    assert set(by_stage["stage04"].values()) == {""}


def test_a_gpu_another_process_tree_leases_is_left_to_it(
    tmp_path, stubs, env, visible_gpus, capsys
):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_GPU.format(root=str(ro.THIS_DIR))],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
        err = capsys.readouterr().err
        assert "round_0 stage03: GPU 1 left to its holder: pid" in err
        assert f"pid {holder.pid}" in err and "for 'an experiment'" in err
        leases = next(c["leases"] for c in calls(env) if c["stage"] == "stage03")
        assert json.loads(leases["gpu0.lock"])["pid"] == os.getpid()
        assert json.loads(leases["gpu1.lock"])["pid"] == holder.pid

        only = make_config(tmp_path, **{"full.total_rounds": 1, "runtime.devices": [1]})
        assert run(stubs, only, tmp_path / "run1", "--mode", "initial") == 2
    finally:
        holder.kill()
        holder.wait()
    err = capsys.readouterr().err
    assert "stage03 not started: refused (one process tree per GPU): GPU 1" in err
    assert not (tmp_path / "run1" / "round_0" / "stage03").exists()
    assert run(stubs, only, tmp_path / "run1", "--mode", "initial") == 0


def test_which_stages_lease_gpus(tmp_path):
    p = _pipeline(tmp_path, make_config(tmp_path))
    assert p.stage_devices("stage03", []) == [0, 1]
    assert p.stage_devices("stage03b", []) == [0, 1]
    assert p.stage_devices("stage03c", []) == [0, 1]
    assert p.stage_devices("stage03d", []) == [0, 1]
    assert p.stage_devices("stage02", ["--seed-commands", "{}"]) == [0, 1]
    assert p.stage_devices("stage02", ["--model-file", "m.pt"]) == []
    assert p.stage_devices("stage04b", ["--rebench-misses"]) == [0, 1]
    assert p.stage_devices("stage04b", []) == []
    assert p.stage_devices("stage07", []) == [0]
    assert p.stage_devices("stage05", []) == []
    bad = ro.load_config(make_config(tmp_path))[0]
    bad["runtime"]["gpu_lease_wait_s"] = -1
    assert any(
        e.startswith("runtime.gpu_lease_wait_s") for e in ro.validate_config(bad)[0]
    )


def test_stage08_datasets_use_the_bench_build_dir(tmp_path):
    other = tmp_path / "other"
    cfg = make_config(
        tmp_path,
        **{
            "runtime.build_dir": str(tmp_path / "custom_build"),
            "stage08.datasets": [
                {
                    "name": "same",
                    "path": "/d.yaml",
                    "rocm_libraries_used": str(tmp_path / "bench"),
                },
                {"name": "other", "path": "/e", "rocm_libraries_used": str(other)},
            ],
        },
    )
    p = _pipeline(tmp_path, cfg)
    args = p.args_stage08(tmp_path / "run", tmp_path / "run" / "round_0")
    specs = [args[i + 1] for i, a in enumerate(args) if a == "--dataset-build-dir"]
    default = other / "projects" / "hipblaslt" / "build" / "release"
    assert specs == [f"same={tmp_path / 'custom_build'}", f"other={default}"]


def _pipeline(tmp_path, cfg_path):
    cfg, _ = ro.load_config(cfg_path)
    return ro.Pipeline(
        cfg=cfg,
        snapshot=cfg_path,
        run_dir=tmp_path / "run",
        paths=ro.derive_paths(cfg),
        stages_dir=Path(ro.__file__).resolve().parent / "stages",
        timing=ro.Timing(),
        force_deploy=False,
    )


def defined_options(help_text):
    """Option strings argparse lists in `--help`: the invocation part of the
    lines indented by exactly two spaces (help text is indented further)."""
    opts = set()
    for line in help_text.splitlines():
        if not re.match(r"^  -", line):
            continue
        invocation = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
        opts.update(re.findall(r"(?:^|, )(--?[A-Za-z0-9][\w-]*)", invocation))
    return opts


def test_defined_options_ignores_mentions_in_help_text():
    text = (
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --smart-k-max SMART_K_MAX\n"
        "                        cut at the knee in [--smart-k-min, --smart-k-max]\n"
        "  --bench-yaml BENCH_YAML, --yaml BENCH_YAML\n"
        "  --weight-dtype {bf16,fp32}  dtype; see --library-stem\n"
    )
    assert defined_options(text) == {
        "-h",
        "--help",
        "--smart-k-max",
        "--bench-yaml",
        "--yaml",
        "--weight-dtype",
    }


def test_serving_options_reach_stage05(tmp_path):
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(rd, step, None)
    assert "--no-non-temporal-rule" not in plain and "--cascade-k" not in plain
    cfg = make_config(
        tmp_path,
        **{"train.smart_k": 0, "train.non_temporal_rule": False, "train.cascade_k": 32},
    )
    args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    assert "--no-non-temporal-rule" in args
    assert args[args.index("--cascade-k") + 1] == "32"
    assert args[args.index("--smart-k") + 1] == "0"


def test_stage_cli_contract(tmp_path, pipeline_dir):
    """Every flag the orchestrator passes is one the real stage accepts."""
    cfg = make_config(
        tmp_path,
        **{
            "train.smart_k_max": 25,
            "train.validation_frac": 0.2,
            "train.non_temporal_rule": False,
            "train.cascade_k": 32,
            "stage08.tie_tolerance": 0.01,
            "stage08.bootstrap": 10,
            "stage08.datasets": [
                {"name": "ds", "path": "/d.yaml", "rocm_libraries_used": "/r"}
            ],
        },
    )
    p = _pipeline(tmp_path, cfg)
    p.origami_reference = True
    rd = tmp_path / "run" / "round_1"
    priors = [tmp_path / "run" / "round_0", tmp_path / "run" / "round_00"]
    (priors[0] / "stage04").mkdir(parents=True)
    initial = ro.RoundStep(0, "initial", [], None, False)
    active = ro.RoundStep(1, "active", priors, None, True)
    enhance = ro.RoundStep(1, "active", priors, None, True, targets=["a"])
    built = {
        "stage01": p.args_stage01(rd, initial)
        + p.args_stage01(rd, active)
        + p.args_stage01(rd, enhance),
        "stage03": p.args_stage03(rd),
        "stage04": p.args_stage04(rd),
        "stage04b": p.args_stage04b(rd, active, priors[0] / "stage05"),
        "stage05": p.args_stage05(rd, initial, None)
        + p.args_stage05(rd, active, "a,b")
        + p.args_stage05(rd, enhance, "a"),
        "stage06": p.args_stage06(rd)
        + p.args_stage06(rd, ro.GatePlan(rd / "g", 0.95, "not-worse", rd / "b")),
        "stage07": p.args_stage07(rd, rd / "r.json", [rd / "w.yaml"], priors[:1]),
        "stage08": p.args_stage08(rd, rd),
    }
    assert "--no-route" not in built["stage07"]
    library_mode = make_config(
        tmp_path,
        **{
            "stage07.build_dir": "/v3",
            "stage07.cpus": "4,5",
            "stage07.production_route": False,
        },
    )
    built["stage07"] += _pipeline(tmp_path, library_mode).args_stage07(
        rd, rd / "r.json", [rd / "w.yaml"], held_out=[rd / "audit.yaml"]
    )
    assert {
        "--bench-binary",
        "--library-dir",
        "--cpus",
        "--held-out-yaml",
        "--no-route",
    } <= set(built["stage07"])
    for key, args in built.items():
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        assert proc.returncode == 0, proc.stderr
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"


def _after(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_misses_are_rebenched_by_the_deciding_stages(tmp_path):
    cfg = make_config(
        tmp_path,
        **{
            "validate.rebench_runs": 4,
            "contexts.list": [{"cu_budget": 64, "cotenant": True}],
            "train.catalog": "v3",
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_1"
    active = ro.RoundStep(1, "active", [tmp_path / "run" / "round_0"], None, False)
    gate = ro.GatePlan(rd / "g", 0.95)
    s04b = p.args_stage04b(rd, active, rd / "m")
    s06 = p.args_stage06(rd, gate)
    s08 = p.args_stage08(rd, rd)
    for args in (s04b, s06, s08):
        assert "--rebench-misses" in args and _after(args, "--rebench-runs") == ["4"]
        assert _after(args, "--bench-binary") == [str(p.paths.bench_binary)]
        assert _after(args, "--devices") == ["0,1"]
        assert _after(args, "--cotenant-binary") == [
            str(ro.context_cotenant(p.cfg, p.paths))
        ]
    for args in (s04b, s06):
        assert _after(args, "--rebench-lines") == [str(rd / "stage01" / "shapes.yaml")]
    assert "--rebench-lines" not in s08
    assert "--rebench-misses" not in p.args_stage06(rd)
    pipeline_dir = Path(ro.__file__).resolve().parent
    for key, args in (("stage04b", s04b), ("stage06", s06), ("stage08", s08)):
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"
    off = _pipeline(
        tmp_path, make_config(tmp_path, **{"validate.rebench_misses": False})
    )
    for args in (
        off.args_stage04b(rd, active, rd / "m"),
        off.args_stage06(rd, gate),
        off.args_stage08(rd, rd),
    ):
        assert not any(a.startswith("--rebench") for a in args)
    assert ro.cfg_get({}, "validate.rebench_misses") is True
    assert ro.cfg_get({}, "validate.rebench_runs") == 3
    errors, _ = ro.validate_config({"validate": {"rebench_runs": 0}})
    assert any(e.startswith("validate.rebench_runs: must be >= 1") for e in errors)


def test_preflight_wants_a_module_of_this_engine(monkeypatch):
    import types

    from lib import evaluate, features

    class OldConfig:
        def __init__(self, **kwargs):
            if "attributes" in kwargs:
                raise TypeError("unexpected keyword argument 'attributes'")

    def broken(**kwargs):
        raise RuntimeError("no engine")

    def module(config):
        return types.SimpleNamespace(
            Config=config, feature_catalog_hash=features.feature_names_hash
        )

    for config, expected in (
        (lambda **kw: None, []),
        (OldConfig, ["older than this checkout's engine"]),
        (broken, ["cannot build kernel configs: RuntimeError('no engine')"]),
    ):
        tw = module(config)
        monkeypatch.setattr(evaluate, "tilewright_module", lambda m=tw: m)
        errors = ro._tilewright_errors()
        assert len(errors) == len(expected)
        assert all(want in got for want, got in zip(expected, errors))


def test_exit_code_mapping():
    assert ro.exit_code(0) == 0
    assert ro.exit_code(3) == 3
    assert ro.exit_code(-9) == 137


def test_move_aside_refuses_foreign_paths(tmp_path):
    owner = tmp_path / "round_0"
    owner.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(RuntimeError):
        ro.move_aside(owner, other)


def test_v3_options_reach_stage05(tmp_path, pipeline_dir):
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(rd, step, None)
    assert "--catalog" not in plain
    cfg = make_config(
        tmp_path,
        **{
            "train.catalog": "v3",
            "train.cascade_k": 0,
            "train.dot2_rule": True,
            "train.non_temporal_rule": True,
            "train.launch_keys": True,
            "train.served_candidates": True,
            "train.clip_features": False,
            "train.whitelist": "view-key",
            "train.min_cu_fraction": 0.25,
            "train.schedules": ["default", "dynamic"],
            "train.attribute_spec": [
                {"name": "param_a", "transform": "log2p1"},
                {"name": "param_b", "transform": "onehot", "values": [0, 2]},
            ],
        },
    )
    args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    assert args[args.index("--catalog") + 1] == "v3"
    assert args[args.index("--cascade-k") + 1] == "0"
    assert args[args.index("--whitelist") + 1] == "view-key"
    assert args[args.index("--schedules") + 1] == "default,dynamic"
    assert args[args.index("--min-cu-fraction") + 1] == "0.25"
    assert {"--dot2-rule", "--non-temporal-rule", "--no-clip-features"} <= set(args)
    assert "--launch-keys" in args and "--served-candidates" in args
    assert "--small-batched-rule" not in args and "--no-non-temporal-rule" not in args
    v3_default = make_config(tmp_path, **{"train.catalog": "v3"})
    args = _pipeline(tmp_path, v3_default).args_stage05(rd, step, None)
    assert "--cascade-k" not in args and "--non-temporal-rule" not in args
    assert "--launch-keys" not in args and "--served-candidates" not in args
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    accepted = defined_options(proc.stdout)
    full = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    assert {a for a in full if a.startswith("--")} <= accepted


STAGE05_OPTION_VALUES = {
    "validation_frac": 0.2,
    "batch_gemms": 8,
    "lr": 0.001,
    "lr_schedule": "step",
    "warmup_frac": 0.1,
    "lr_min_frac": 0.05,
    "eval_points": 20,
    "patience": 4,
    "student_temperature": "fixed",
    "qat_frac": 0.25,
    "qat_lr_frac": 0.2,
    "boundary_margin": 0.5,
    "boundary_weight": 0.3,
    "pseudo_negatives": "random",
    "unmeasured_samples": 32,
    "unmeasured_margin": 0.05,
    "unmeasured_weight": 0.3,
    "stage1_recall_weight": 0.3,
    "recipe": "per_leaf",
    "pooled_epochs": 8,
    "pooled_batch_gemms": 32,
    "pooled_lr": 0.001,
    "pooled_eval_points": 12,
    "pooled_threads": 8,
    "pooled_teacher_threads": 6,
    "fine_tune_epochs": 6,
    "fine_tune_lr": 0.0005,
}


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_stage05_sees_the_plans_its_training_rounds_were_benched_from(
    tmp_path, pipeline_dir, planner
):
    """In a run of either layout: stage01b/ in one written before the
    planner became stage02, whose stage02/ holds the probe's outputs."""
    run = tmp_path / "run"
    r0, r1 = run / "round_0", run / "round_1"
    if planner == "stage01b":
        _probe_era(r0)
    else:
        run_layout.settle(run)
    for path in (
        r0 / planner / "plan.json",
        r0 / planner / "seed" / "plan.json",
        r1 / planner / "plan.json",
        r1 / planner / "contexts" / "cu64" / "plan.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    step = ro.RoundStep(1, "active", [r0], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(r1, step, None)
    assert "--plan" not in plain and "--context-plan" not in plain
    cfg = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "train.catalog": "v3",
            "contexts.list": [{"cu_budget": 64}],
        },
    )
    p = _pipeline(tmp_path, cfg)
    args = p.args_stage05(r1, step, None)
    plans = [args[i + 1] for i, a in enumerate(args) if a == "--plan"]
    assert plans == [
        str(r0 / planner / "plan.json"),
        str(r0 / planner / "seed" / "plan.json"),
        str(r1 / planner / "plan.json"),
    ]
    assert "--context-plan" not in args
    replan = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "train.catalog": "v3",
            "contexts.list": [{"cu_budget": 64}],
            "contexts.replan": True,
        },
    )
    args = _pipeline(tmp_path, replan).args_stage05(r1, step, None)
    i = args.index("--context-plan")
    assert args[i + 1 : i + 4] == [
        "64",
        "default",
        str(r1 / planner / "contexts" / "cu64" / "plan.json"),
    ]
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert {a for a in args if a.startswith("--")} <= defined_options(proc.stdout)


def test_stage05_off_list_charge_is_configured_or_measured_by_stage04b(
    tmp_path, pipeline_dir
):
    rd = tmp_path / "run" / "round_1"
    step = ro.RoundStep(1, "active", [tmp_path / "run" / "round_0"], None, False)

    def off_list(**kw):
        cfg = make_config(tmp_path, **{f"train.{k}": v for k, v in kw.items()})
        assert not ro.validate_config(ro.load_config(cfg)[0])[0], kw
        args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
        i = [j for j, a in enumerate(args) if a.startswith("--off-list")]
        return args[i[0] : i[-1] + 2] if i else []

    assert off_list() == []
    assert off_list(off_list_sel_eff=0.8) == ["--off-list-sel-eff", "0.8"]
    auto = off_list(off_list_sel_eff="auto", off_list_fallback=0.9)
    assert auto == [
        "--off-list-sel-eff",
        "auto",
        "--off-list-min-misses",
        "30",
        "--off-list-fallback",
        "0.9",
        "--off-list-rebench",
        str(rd / "stage04b" / "rebench_misses.json"),
    ]
    assert "--off-list-fallback" not in off_list(off_list_sel_eff="auto")
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert {a for a in auto if a.startswith("--")} <= defined_options(proc.stdout)


def test_training_options_reach_stage05(tmp_path, pipeline_dir):
    assert set(STAGE05_OPTION_VALUES) == set(ro.STAGE05_OPTION_KEYS)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(rd, step, None)
    flags = {"--" + k.replace("_", "-"): v for k, v in STAGE05_OPTION_VALUES.items()}
    assert not set(flags) & set(plain)
    cfg = make_config(
        tmp_path, **{f"train.{k}": v for k, v in STAGE05_OPTION_VALUES.items()}
    )
    assert not ro.validate_config(ro.load_config(cfg)[0])[0]
    args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    for flag, value in flags.items():
        assert args[args.index(flag) + 1] == str(value)
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert set(flags) <= defined_options(proc.stdout)
    for value in STAGE05_OPTION_VALUES.values():
        if isinstance(value, str):
            assert re.search(rf"\{{[^}}]*\b{value}\b[^}}]*\}}", proc.stdout), value


def test_training_option_validation():
    assert not any("train" in e for e in ro.validate_config({"train": {}})[0])
    for bad in (
        {"batch_gemms": -1},
        {"lr": 0},
        {"lr_schedule": "linear"},
        {"warmup_frac": 1.0},
        {"lr_min_frac": -0.1},
        {"eval_points": 0},
        {"patience": -1},
        {"student_temperature": "auto"},
        {"qat_frac": -0.5},
        {"qat_lr_frac": 0},
        {"batch_gemms": 1.5},
        {"off_list_sel_eff": 0},
        {"off_list_sel_eff": 1.5},
        {"off_list_sel_eff": "measured"},
        {"off_list_fallback": 0},
        {"off_list_min_misses": 0},
        {"stage1_recall_weight": -1},
        {"unmeasured_samples": -1},
        {"unmeasured_margin": -0.1},
        {"unmeasured_weight": 0},
        {"pseudo_negatives": "nearest"},
        {"recipe": "global"},
        {"pooled_epochs": 0},
        {"pooled_batch_gemms": -1},
        {"pooled_lr": 0},
        {"pooled_eval_points": 0},
        {"pooled_threads": -1},
        {"pooled_teacher_threads": -1},
        {"fine_tune_epochs": -1},
        {"fine_tune_lr": 0},
    ):
        errors = ro.validate_config({"train": bad})[0]
        assert any(f"train.{next(iter(bad))}" in e for e in errors), bad


def test_the_pooled_recipe_needs_validation_gemms_and_no_margin(tmp_path):
    base = ro.load_config(make_config(tmp_path))[0]

    def errors(**train):
        cfg = dict(base, train={**base["train"], "recipe": "pooled_ft", **train})
        return [e for e in ro.validate_config(cfg)[0] if "recipe" in e]

    assert not errors()
    assert not errors(validation_frac=0.2, boundary_margin=0, qat_frac=0)
    for bad in ({"validation_frac": 0}, {"boundary_margin": 0.5}, {"qat_frac": 0.1}):
        found = errors(**bad)
        assert len(found) == 1 and f"train.{next(iter(bad))}" in found[0], bad


def test_pseudo_negative_draws_agree_with_their_sample_count(tmp_path):
    base = ro.load_config(make_config(tmp_path))[0]

    def errors(**train):
        cfg = dict(base, train={**base["train"], **train})
        found = ro.validate_config(cfg)[0]
        return [e for e in found if "pseudo_negatives" in e]

    assert not errors(pseudo_negatives="targeted", catalog="v3")
    assert not errors(pseudo_negatives="random", unmeasured_samples=8)
    assert not errors(pseudo_negatives="none")
    assert not errors(unmeasured_samples=8)
    assert errors(pseudo_negatives="targeted")
    assert errors(pseudo_negatives="none", unmeasured_samples=8)
    assert errors(pseudo_negatives="random")
    assert errors(pseudo_negatives="targeted", catalog="v3", unmeasured_samples=0)


def test_v3_config_validation():
    base = {"train": {"catalog": "v3"}}
    assert not any("train" in e for e in ro.validate_config(base)[0])
    for bad in (
        {"catalog": "v4"},
        {"attribute_spec": [{"transform": "identity"}]},
        {"attribute_spec": [{"name": "x", "transform": "cube"}]},
        {"schedules": ["fast"]},
        {"min_cu_fraction": 0.0},
        {"whitelist": "both"},
        {"launch_keys": "yes"},
        {"served_candidates": "yes"},
    ):
        errors = ro.validate_config({"train": dict(base["train"], **bad)})[0]
        assert any("train" in e for e in errors), bad


def test_served_candidates_are_a_v3_training_option(tmp_path):
    cfg = ro.load_config(make_config(tmp_path))[0]
    assert ro.cfg_get(cfg, "train.served_candidates") is False
    cfg.setdefault("train", {})["served_candidates"] = True
    errors = ro.validate_config(cfg)[0]
    assert any("train.served_candidates" in e and "v3" in e for e in errors)
    cfg["train"]["catalog"] = "v3"
    assert ro.validate_config(cfg)[0] == []


def test_launch_keys_are_a_v3_serving_option(tmp_path):
    cfg = ro.load_config(make_config(tmp_path))[0]
    assert ro.cfg_get(cfg, "train.launch_keys") is False
    cfg.setdefault("train", {})["launch_keys"] = True
    errors = ro.validate_config(cfg)[0]
    assert any("train.launch_keys" in e and "v3" in e for e in errors)
    cfg["train"]["catalog"] = "v3"
    assert ro.validate_config(cfg)[0] == []


def test_family_rule_and_trained_key_gate_are_v3_serving_options(
    tmp_path, pipeline_dir
):
    cfg = ro.load_config(make_config(tmp_path))[0]
    assert ro.cfg_get(cfg, "train.non_temporal_family_rule") is False
    assert ro.cfg_get(cfg, "train.trained_key_gate") is True
    cfg.setdefault("train", {})["non_temporal_family_rule"] = True
    errors = ro.validate_config(cfg)[0]
    assert any("train.non_temporal_family_rule" in e for e in errors)
    cfg["train"]["catalog"] = "v3"
    assert any(
        "train.non_temporal_family_rule" in e for e in ro.validate_config(cfg)[0]
    )
    cfg["train"]["non_temporal_rule"] = True
    assert ro.validate_config(cfg)[0] == []

    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    on = make_config(
        tmp_path,
        **{"train.catalog": "v3", "train.non_temporal_rule": True,
           "train.non_temporal_family_rule": True},
    )  # fmt: skip
    args = _pipeline(tmp_path, on).args_stage05(rd, step, None)
    assert "--non-temporal-family-rule" in args and "--no-trained-key-gate" not in args
    off = make_config(
        tmp_path, **{"train.catalog": "v3", "train.trained_key_gate": False}
    )
    args = _pipeline(tmp_path, off).args_stage05(rd, step, None)
    assert "--no-trained-key-gate" in args and "--non-temporal-family-rule" not in args


def test_preflight_checks_the_v3_catalog(monkeypatch):
    import types

    from lib import evaluate, features

    tw = types.SimpleNamespace(
        Config=lambda **kw: None,
        feature_catalog_hash=features.feature_names_hash,
        feature_catalog_hashes=lambda: [features.feature_names_hash()],
    )
    monkeypatch.setattr(evaluate, "tilewright_module", lambda: tw)
    assert ro._tilewright_errors("v2") == []
    (err,) = ro._tilewright_errors("v3")
    assert "does not compute the v3 catalog" in err
    tw.feature_catalog_hashes = lambda: [
        features.feature_names_hash(),
        features.v3_catalog_hash(),
    ]
    assert ro._tilewright_errors("v3") == []


# ── planner ──────────────────────────────────────────────────────────────────


def test_planner_is_off_by_default(tmp_path, stubs, env):
    run_dir = tmp_path / "run"
    assert run(stubs, make_config(tmp_path), run_dir, "--mode", "initial") == 0
    stages = [c["stage"] for c in calls(env)]
    assert "stage02" not in stages
    s04 = [c for c in calls(env) if c["stage"] == "stage04"][0]
    assert "--plan" not in s04["argv"] and s04["argv"].count("--log-dir") == 1


def test_planner_defaults_match_the_stage():
    import stage02_plan_benchmarks as s02

    for key, default in s02.DEFAULTS.items():
        assert ro.cfg_get({}, f"planner.{key}") == default, key


def test_planner_config_is_validated(tmp_path):
    cfg_path = make_config(tmp_path, **{"planner.enabled": True})
    cfg, _ = ro.load_config(cfg_path)
    cfg["planner"]["origami_top_k"] = 4
    cfg["planner"]["max_list"] = 4097
    cfg["planner"]["mode"] = "heuristic"
    errors, _ = ro.validate_config(cfg)
    assert "planner.origami_top_k: unknown key" in errors
    assert any(e.startswith("planner.max_list: must be <= 4096") for e in errors)
    assert any(e.startswith("planner.mode: 'heuristic'") for e in errors)


def test_planned_round_wiring(tmp_path, stubs, env):
    cfg = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.mode": "planned",
            "planner.bank_library_stems": [STEM.replace("ID75a0", "CU256_ID75a0")],
        },
    )
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    rd0 = run_dir / "round_0"
    stages0 = [c["stage"] for c in log if "round_0" in (arg(c, "--out-dir") or "")]
    assert stages0[:4] == ["stage01", "stage02", "stage03", "stage04"]
    s02 = [c for c in log if c["stage"] == "stage02"]
    assert arg(s02[0], "--in-dir") == str(rd0 / "stage01")
    assert arg(s02[0], "--out-dir") == str(rd0 / "stage02")
    seed = json.loads(arg(s02[0], "--seed-commands"))
    assert "--model-file" not in s02[0]["argv"]
    assert sorted(seed) == ["stage03", "stage04"]
    assert arg({"argv": seed["stage03"]}, "--in-dir") == str(rd0 / "stage02" / "seed")
    assert arg({"argv": seed["stage04"]}, "--log-dir").endswith("seed/stage03/logs")
    s03 = [c for c in log if c["stage"] == "stage03"][0]
    assert arg(s03, "--in-dir") == str(rd0 / "stage02")
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--plan") == str(rd0 / "stage02" / "plan.json")
    assert arg(s04, "--extra-library-stem").startswith("TensileLibrary_")
    # Later rounds rank with the latest trained model instead of seeds.
    assert arg(s02[1], "--model-file") == str(rd0 / "stage05" / "models.pt")
    assert "--seed-commands" not in s02[1]["argv"]
    for rd in (rd0, run_dir / "round_1", run_dir / "round_2"):
        assert ro.stage_complete(rd / "stage02") and (rd / "stage02.log").is_file()
    assert not list(run_dir.rglob("stage01b*"))
    layout = json.loads((run_dir / run_layout.LAYOUT_FILE).read_text())
    assert layout == {"layout": 2, "planner": "stage02"}


def test_planner_from_stage_and_stage05_seed_ranker(tmp_path, stubs, env):
    cfg = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.mode": "planned",
            "planner.ranker": "stage05",
            "planner.seed_per_cell": 4,
        },
    )
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    s02 = [c for c in calls(env) if c["stage"] == "stage02"][0]
    seed = json.loads(arg(s02, "--seed-commands"))
    assert sorted(seed) == ["stage03", "stage04", "stage05"]
    assert arg({"argv": seed["stage03"]}, "--in-dir").endswith("stage02/seed")
    s05 = {"argv": seed["stage05"]}
    assert arg(s05, "--enriched-csv-dir").endswith("stage02/seed/stage04")
    assert arg(s05, "--output-dir").endswith("stage02/seed/stage05")
    assert arg(s05, "--min-cell-gemms") == "4"
    s03 = [c for c in calls(env) if c["stage"] == "stage03"][-1]
    assert arg(s03, "--in-dir").endswith("round_0/stage02")
    for value in ("2", "stage02"):
        env.write_text("")
        argv = ["--mode", "initial", "--from-stage", value]
        assert run(stubs, cfg, run_dir, *argv) == 0
        assert [c["stage"] for c in calls(env)][:2] == ["stage02", "stage03"]
    assert len(list((run_dir / "round_0").glob("stale/*/stage02"))) == 2


def test_explicit_model_file_skips_the_seed_bootstrap(tmp_path, stubs, env):
    model = tmp_path / "given.pt"
    cfg = make_config(
        tmp_path,
        **{"planner.enabled": True, "planner.model_file": str(model)},
    )
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 0
    s02 = [c for c in calls(env) if c["stage"] == "stage02"][0]
    assert arg(s02, "--model-file") == str(model)


def _timing_rows(run_dir):
    return [
        r
        for p in sorted((run_dir / "timing").glob("*.json"))
        for r in json.loads(p.read_text())["rows"]
    ]


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_a_run_resumes_with_its_planner_where_it_wrote_it(
    tmp_path, stubs, env, monkeypatch, capsys, planner
):
    """An interrupted run resumes with the planner in stage02/, or in
    stage01b/ when it was written before the planner became stage02 (its
    rounds also hold the removed probe's stage02/): every stage reads the
    plans there, the rounds the resume adds plan there, and a round plans
    again there from --from-stage 2 or its old names. Nothing of an old run
    is renamed, moved or taken for the planner's, and it gets no
    run_layout.json."""
    cfg = make_config(tmp_path, **{"planner.enabled": True, "planner.mode": "planned"})
    run_dir = tmp_path / "run"
    r0, r1, r2 = (run_dir / f"round_{i}" for i in range(3))
    legacy = planner == "stage01b"
    with layout_of_the_past(planner):
        assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
        if legacy:
            _probe_era(r0)
        monkeypatch.setenv("STUB_FAIL", "stage03@round_1")
        assert run(stubs, cfg, run_dir, "--mode", "active") == 1
    monkeypatch.delenv("STUB_FAIL")
    assert ro.stage_complete(r1 / planner) and not ro.stage_complete(r1 / "stage03")
    (r0 / planner / "plan.json").write_text("{}")
    probe = _stage02_files(r0)
    if legacy:
        assert len(probe) == 4
    assert (run_dir / run_layout.LAYOUT_FILE).exists() != legacy
    assert run_layout.of(run_dir).planner == planner

    env.write_text("")
    monkeypatch.setenv("STUB_GATE", "1")
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    s02 = [c for c in log if c["stage"] == "stage02"]
    assert [arg(c, "--out-dir") for c in s02] == [str(r2 / planner)]
    resumed = [c for c in log if str(r1) in (arg(c, "--out-dir") or "")]
    assert resumed[0]["stage"] == "stage03"
    assert arg(resumed[0], "--in-dir") == str(r1 / planner)
    by = {c["stage"]: c for c in resumed}
    assert arg(by["stage04"], "--plan") == str(r1 / planner / "plan.json")
    assert arg(by["stage04b"], "--plan") == str(r1 / planner / "plan.json")
    s05 = [c for c in log if c["stage"] == "stage05"][0]
    assert arg(s05, "--output-dir") == str(r1 / "stage05")
    assert _after(s05["argv"], "--plan") == [str(r0 / planner / "plan.json")]
    s06 = [c for c in log if c["stage"] == "stage06"][-1]
    assert arg(s06, "--gate-plan") == str(r2 / planner / "plan.json")
    for rd in (r0, r1, r2):
        assert ro.stage_complete(rd / planner) and (rd / f"{planner}.log").is_file()
        assert (rd / ro.ROUND_MARKER).is_file()
    assert {r["stage"] for r in _timing_rows(run_dir)} & {"stage01b", "stage02"} == {
        planner
    }
    assert _stage02_files(r0) == probe
    if legacy:
        assert not (r1 / "stage02").exists() and not (r2 / "stage02").exists()
        assert not (run_dir / run_layout.LAYOUT_FILE).exists()
    else:
        assert not list(run_dir.rglob("stage01b*"))

    capsys.readouterr()
    for value in ("2", "stage02", "1.5", "stage01b"):
        env.write_text("")
        argv = ["--mode", "full", "--from-round", "2", "--from-stage", value]
        assert run(stubs, cfg, run_dir, *argv) == 0
        log = calls(env)
        assert [c["stage"] for c in log][:2] == ["stage02", "stage03"]
        assert arg(log[0], "--out-dir") == str(r2 / planner)
        assert arg(log[1], "--in-dir") == str(r2 / planner)
        err = capsys.readouterr().err
        assert ("is deprecated" in err) == (value in ("1.5", "stage01b"))
    assert len(list(r2.glob(f"stale/*/{planner}"))) == 4
    assert len(list(r2.glob(f"stale/*/{planner}.log"))) == 4
    assert _stage02_files(r0) == probe
    if legacy:
        assert not list(run_dir.glob("*/stale/*/stage02*"))
    assert (run_dir / run_layout.LAYOUT_FILE).exists() != legacy


def test_planner_preflight_needs_bench_support(tmp_path):
    cfg_path = make_config(tmp_path, **{"planner.enabled": True})
    cfg, _ = ro.load_config(cfg_path)
    bench = tmp_path / "clients" / "hipblaslt-bench"
    bench.parent.mkdir(parents=True)
    assert "cannot read" in ro._planner_bench_errors(bench)[0]
    (bench.parent / "hipblaslt_common.yaml").write_text("  - solution_index_list: x\n")
    assert "skip_screen_prepass" in ro._planner_bench_errors(bench)[0]
    (bench.parent / "hipblaslt_common.yaml").write_text(
        "  - solution_index_list: c_int32*512\n  - skip_screen_prepass: c_bool\n"
    )
    assert "fewer than 4096" in ro._planner_bench_errors(bench)[0]
    (bench.parent / "hipblaslt_common.yaml").write_text(
        "  - solution_index_list: c_int32*4096\n  - skip_screen_prepass: c_bool\n"
    )
    assert ro._planner_bench_errors(bench) == []


def _isa_topology(tmp_path, monkeypatch, *isa):
    """A fake KFD topology of GPUs with these (compiler target, ASIC
    revision)."""
    import gpu_fakes as gf
    from lib import kfd

    root = gf.write_sysfs(tmp_path / "sys", gf.GPUS[: len(isa)], isa=list(isa))
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))


def test_bench_devices_must_report_the_strict_arch(tmp_path, monkeypatch):
    from lib.arch import STRICT_SWITCH

    _isa_topology(tmp_path, monkeypatch, ("gfx1250", 0), ("gfx1250", 0), ("gfx950", 0))
    cfg = {"arch": "gfx1250-strict", "runtime": {"devices": [0, 1]}}
    errs = ro._bench_device_errors(cfg, {})
    assert len(errs) == 2
    assert errs[1].startswith(
        "runtime.devices: device 1 (GPU 1) reports gfx1250 (ASIC revision 0, "
        f"{STRICT_SWITCH} unset), so hipBLASLt loads library/gfx1250v0/ for it, "
        "not library/gfx1250-strict/: gfx1250-strict needs a gfx1250 device of "
        f"ASIC revision 0 and {STRICT_SWITCH}=0 in the environment of the run"
    )
    assert ro._bench_device_errors(cfg, {STRICT_SWITCH: "0"}) == []
    one = dict(cfg, runtime={"devices": [1]})
    (err,) = ro._bench_device_errors(one, {STRICT_SWITCH: "1"})
    assert f"(ASIC revision 0, {STRICT_SWITCH}=1)" in err
    other = dict(cfg, runtime={"devices": [2]})
    (err,) = ro._bench_device_errors(other, {STRICT_SWITCH: "0"})
    assert "device 2 (GPU 2) reports gfx950, so hipBLASLt loads library/gfx950/" in err
    visible = {STRICT_SWITCH: "0", "ROCR_VISIBLE_DEVICES": "2,1"}
    (err,) = ro._bench_device_errors(cfg, visible)
    assert err.startswith("runtime.devices: device 0 (GPU 2) reports gfx950")
    assert ro._bench_device_errors(cfg, {"ROCR_VISIBLE_DEVICES": "-1"}) == []
    (err,) = ro._bench_device_errors(
        cfg, {STRICT_SWITCH: "0", "ROCR_VISIBLE_DEVICES": "0"}
    )
    assert err.startswith("runtime.devices: device 1 is not one of the 1 GPU(s)")


def test_the_device_check_applies_the_runtime_env(tmp_path, monkeypatch):
    from lib.arch import STRICT_SWITCH

    _isa_topology(tmp_path, monkeypatch, ("gfx1250", 0), ("gfx1250", 1))
    cfg = {"arch": "gfx1250-strict", "runtime": {"devices": [0]}}
    (err,) = ro._bench_device_errors(cfg, {})
    assert err.endswith(f' (set runtime.env: {{{STRICT_SWITCH}: "0"}})')
    on = {**cfg, "runtime": {"devices": [0], "env": {STRICT_SWITCH: "0"}}}
    assert ro._bench_device_errors(on, {}) == []
    assert ro._bench_device_errors(on, {STRICT_SWITCH: "1"}) == []
    (err,) = ro._bench_device_errors(
        {**on, "runtime": {**on["runtime"], "devices": [1]}}, {}
    )
    assert "reports gfx1250 (ASIC revision 1" in err and "set runtime.env" not in err


def test_the_revision_decides_which_subtree_a_gfx1250_device_loads(
    tmp_path, monkeypatch
):
    from lib.arch import STRICT_SWITCH

    _isa_topology(tmp_path, monkeypatch, ("gfx1250", 1), ("gfx1250", 0))
    strict = {"arch": "gfx1250-strict", "runtime": {"devices": [0]}}
    (err,) = ro._bench_device_errors(strict, {STRICT_SWITCH: "0"})
    assert f"reports gfx1250 (ASIC revision 1, {STRICT_SWITCH}=0)" in err
    base = {"arch": "gfx1250", "runtime": {"devices": [0, 1]}}
    for env, loaded in (({}, "gfx1250v0"), ({STRICT_SWITCH: "0"}, "gfx1250-strict")):
        (err,) = ro._bench_device_errors(base, env)
        assert err.startswith("runtime.devices: device 1 (GPU 1)")
        assert f"loads library/{loaded}/ for it, not library/gfx1250/" in err
        assert "needs a gfx1250 device of an ASIC revision other than 0" in err


def test_the_device_check_reads_only_the_runs_devices(tmp_path, monkeypatch):
    from lib import kfd
    from lib.arch import STRICT_SWITCH

    def unread(*args, **kwargs):
        raise AssertionError("the topology was read")

    with monkeypatch.context() as m:
        m.setattr(kfd, "gpus", unread)
        m.setattr(kfd, "physical_gpu", unread)
        m.setattr(kfd, "gpu_isa", unread)
        for arch in ("gfx950", "gfx942:sramecc+:xnack-"):
            cfg = {"arch": arch, "runtime": {"devices": [0, 1]}}
            assert ro._bench_device_errors(cfg, {}) == []
    _isa_topology(tmp_path, monkeypatch, ("gfx950", 0), ("gfx1250", 0), None)
    read = []
    isa = kfd.gpu_isa
    monkeypatch.setattr(kfd, "gpu_isa", lambda g: read.append(g.index) or isa(g))
    cfg = {"arch": "gfx1250-strict", "runtime": {"devices": [0]}}
    env = {STRICT_SWITCH: "0", "ROCR_VISIBLE_DEVICES": "1"}
    assert ro._bench_device_errors(cfg, env) == [] and read == [1]
    (err,) = ro._bench_device_errors(cfg, {"ROCR_VISIBLE_DEVICES": "2"})
    assert "device 0 (GPU 2) has no gfx_target_version in the KFD topology" in err


# An architecture without a fixed queue count, made up for these tests; the
# XCD counts given with it are made up too.
UNLISTED = "gfx9990"
UNLISTED_STEM = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_"
    "Dijk_" + UNLISTED
)


def _unlisted_config(tmp_path, kernels, **overrides):
    """A v3 config of UNLISTED whose bench library holds `kernels`."""
    import bench_fakes as bf

    path = make_config(
        tmp_path,
        **{
            "arch": UNLISTED,
            "hipblaslt.library_stem": UNLISTED_STEM,
            "deploy.targets": [UNLISTED],
            "train.catalog": "v3",
            **overrides,
        },
    )
    cfg, _ = ro.load_config(path)
    paths = ro.derive_paths(cfg)
    paths.library_dir.mkdir(parents=True, exist_ok=True)
    bf.write_logic(paths.library_dir / f"{UNLISTED_STEM}.dat", kernels)
    return cfg, paths


def test_a_pool_that_takes_a_dynamic_queue_needs_the_xcd_count(tmp_path):
    from bench_fakes import solution

    stream_k = dict(tileProcessingStrategy="StreamK", workAssignment="StaticGrid")
    queue = dict(tileProcessingStrategy="StreamK", workAssignment="DynamicWorkQueue")
    static = [solution(1, 1, **stream_k), solution(2, 2, **stream_k)]
    cfg, paths = _unlisted_config(tmp_path, static)
    assert ro._queue_count_errors(cfg, paths) == []
    cfg, paths = _unlisted_config(tmp_path, static + [solution(3, 3, **queue)])
    (err,) = ro._queue_count_errors(cfg, paths)
    assert err.startswith("1 of the 3 kernels of the pool (e.g. ")
    assert f"unknown for {UNLISTED}: set hardware.num_xcd" in err
    cfg["hardware"] = {"num_xcd": 3}
    assert ro._queue_count_errors(cfg, paths) == []
    cfg["hardware"] = {}
    cfg["train"]["catalog"] = "v2"
    assert ro._queue_count_errors(cfg, paths) == []
    hybrid = dict(tileProcessingStrategy="StreamK", workAssignment="Hybrid")
    cfg, paths = _unlisted_config(tmp_path, static + [solution(3, 3, **hybrid)])
    assert ro._queue_count_errors(cfg, paths) == []
    cfg["contexts"] = {"list": [{"cu_budget": 128, "schedule": "dynamic"}]}
    assert len(ro._queue_count_errors(cfg, paths)) == 1


def test_a_pool_of_tdm_loaded_operands_refuses_what_reads_their_hints(tmp_path):
    """Preflight of a run whose pool holds a kernel the tensor data mover
    loads operands of refuses the non-temporal rules (on by default in v2)
    and attributes holding such an operand's temporal hint; a run that
    benches without training (--mode audit) is not refused."""
    from bench_fakes import solution
    from test_kernel_identity import tdm_solution

    plain = [solution(1, 1), solution(2, 2, nta=4)]
    cfg, paths = _unlisted_config(tmp_path, plain, **{"train.non_temporal_rule": True})
    assert ro._tdm_identity_errors(cfg, paths) == []
    cfg, paths = _unlisted_config(tmp_path, plain + [tdm_solution(3, 3, tdm=2)])
    assert ro._tdm_identity_errors(cfg, paths) == []
    cfg["train"]["non_temporal_rule"] = True
    (err,) = ro._tdm_identity_errors(cfg, paths)
    assert err.startswith("train.non_temporal_rule: it reads cache hints")
    assert "loads operand B of kernels of the pool" in err
    assert err.endswith("(identity rule tdm_th_v1); set it false")
    assert err in ro.preflight(cfg, paths, "full")
    assert err not in ro.preflight(cfg, paths, "audit")
    cfg["train"]["non_temporal_family_rule"] = True
    assert len(ro._tdm_identity_errors(cfg, paths)) == 2
    cfg["train"] = {"catalog": "v3", "attribute_spec": [{"name": "temporal_hint_a"}]}
    assert ro._tdm_identity_errors(cfg, paths) == []
    cfg["train"]["attribute_spec"].append({"name": "temporal_hint_b"})
    (err,) = ro._tdm_identity_errors(cfg, paths)
    assert err.startswith("train.attribute_spec: it names temporal_hint_b,")
    assert err.endswith("; leave those attributes out")
    cfg["train"] = {"catalog": "v2"}
    assert len(ro._tdm_identity_errors(cfg, paths)) == 1
    cfg["train"]["non_temporal_rule"] = False
    assert ro._tdm_identity_errors(cfg, paths) == []
    cfg["train"]["catalog"] = "v3"
    cfg["full"] = {"final_retrain": {"recipe": {"non_temporal_rule": True}}}
    (err,) = ro._tdm_identity_errors(cfg, paths)
    assert err.startswith("full.final_retrain.recipe: train.non_temporal_rule: ")


def test_the_xcd_count_of_a_whole_device_is_read_when_the_run_starts(
    tmp_path, monkeypatch
):
    import gpu_fakes as gf
    from lib import kfd

    for var in (*kfd.HIP_VARS, kfd.ROCR_VAR):
        monkeypatch.delenv(var, raising=False)
    root = gf.write_sysfs(tmp_path / "sys", gf.GPUS[:2], xcc=[(3, "SPX"), (1, "CPX")])
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    cfg, _ = _unlisted_config(tmp_path, [])
    out, (line,) = ro.resolve_num_xcd(cfg, 0)
    assert out["hardware"] == {"num_xcd": 3} and "hardware" not in cfg
    assert line.startswith("hardware.num_xcd 3, the XCCs of bench device 0")
    assert ro.resolve_num_xcd(cfg, 1) == (cfg, [])
    pinned = dict(cfg, hardware={"num_xcd": 5})
    assert ro.resolve_num_xcd(pinned, 0) == (pinned, [])
    gfx950, _ = ro.load_config(make_config(tmp_path))
    assert ro.resolve_num_xcd(gfx950, 0) == (gfx950, [])


def test_a_configured_xcd_count_must_be_the_arch_queue_count(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["hardware"] = {"num_xcd": 4}
    (err,) = ro.validate_config(cfg)[0]
    assert err.startswith("hardware.num_xcd 4: hipBLASLt's dynamic-queue kernels")
    cfg["hardware"] = {"num_xcd": 8}
    assert ro.validate_config(cfg)[0] == []


def test_preflight_checks_the_bench_devices_of_a_mode_that_benches(
    tmp_path, monkeypatch
):
    from lib.arch import STRICT_SWITCH

    _isa_topology(tmp_path, monkeypatch, ("gfx1250", 0), ("gfx1250", 0))
    hidden = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    for var in (STRICT_SWITCH, *hidden):
        monkeypatch.delenv(var, raising=False)
    overrides = {"arch": "gfx1250-strict", "stage07.bench_yamls": []}

    def device_errors(mode, **more):
        cfg, _ = ro.load_config(make_config(tmp_path, **overrides, **more))
        errs = ro.preflight(cfg, ro.derive_paths(cfg), mode)
        return [e for e in errs if e.startswith("runtime.devices:")]

    assert len(device_errors("full")) == 2
    assert len(device_errors("audit")) == 2
    assert device_errors("validate", **{"validate.rebench_misses": False}) == []
    assert len(device_errors("validate")) == 2
    monkeypatch.setenv(STRICT_SWITCH, "0")
    assert device_errors("full") == []


def test_preflight_needs_the_runs_devices_to_be_of_one_kind(tmp_path, monkeypatch):
    import gpu_fakes as gf
    from lib import kfd

    for var in (*kfd.HIP_VARS, kfd.ROCR_VAR):
        monkeypatch.delenv(var, raising=False)
    root = gf.write_sysfs(
        tmp_path / "sys", xcc=[(3, "SPX"), (1, "CPX"), (3, "SPX")], cus=[48, 16, 48]
    )
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    assert ro._device_settings_errors({"runtime": {"devices": [0, 2]}}) == []
    errs = ro._device_settings_errors({"runtime": {"devices": [0, 1]}})
    head = "runtime.devices: the run's devices disagree on the"
    assert [e[len(head) :].split(":")[0] for e in errs] == [
        " CU count",
        " compute partition mode",
        " XCC count",
    ]
    assert errs[1].endswith("GPU 0 (0000:0b:00.0) SPX, GPU 1 (0000:0a:00.0) CPX")
    both = {"runtime": {"devices": [0, 1]}}
    assert ro._device_settings_errors(both, {kfd.ROCR_VAR: "0,2"}) == []
    assert ro._device_settings_errors(both, {kfd.ROCR_VAR: "-1"}) == []
    cfg, _ = ro.load_config(make_config(tmp_path, **{"runtime.devices": [0, 1]}))
    for mode in ("full", "audit"):
        errs = ro.preflight(cfg, ro.derive_paths(cfg), mode)
        assert sum(e.startswith(head) for e in errs) == 3


def _bank_library(library_dir):
    """STEM (Prediction table [1, 2]) plus a `_CU256_` file whose Equality
    row the 0x75a0 / 256-CU device accepts and an `_ID75a8` one it does
    not."""
    import msgpack

    from bench_fakes import solution

    def write(stem, indices, hw, library):
        sols = [dict(solution(i, i), hardwarePredicate=hw) for i in indices]
        (library_dir / f"{stem}.dat").write_bytes(
            msgpack.packb({"solutions": sols, "library": library})
        )

    def equality(indices):
        table = [{"key": [i], "index": i} for i in indices]
        return {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "EqualityMatching"},
                    "library": {"type": "Matching", "table": table},
                }
            ],
        }

    def gpu(*terms):
        return {"type": "AMDGPU", "value": {"type": "And", "value": list(terms)}}

    chip = {"type": "PciChipId", "value": 0x75A0}
    library_dir.mkdir(parents=True)
    prediction = {"type": "Prediction", "table": [1, 2]}
    write(STEM, [1, 2], gpu(chip), prediction)
    cu256 = STEM.replace("ID75a0", "CU256_ID75a0")
    write(cu256, [5, 4], gpu(chip, {"type": "CUCount", "value": 256}), equality([5, 4]))
    a8 = STEM.replace("ID75a0", "ID75a8")
    write(a8, [7], gpu({"type": "PciChipId", "value": 0x75A8}), equality([7]))
    return cu256


def test_planner_bank_auto_wiring(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "-1")
    hardware = {"n_cu": 256, "lds_bytes": 65536, "l2_bytes": 4 << 20}
    cfg_path = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.bank": "auto",
            "bench.pool_only_library": True,
            "hardware": dict(hardware, pci_chip_id=0x75A0),
        },
    )
    p = _pipeline(tmp_path, cfg_path)
    cu256 = _bank_library(p.paths.library_dir)
    pool_only = p._pool_only_args()
    stems = [
        pool_only[i + 1] for i, a in enumerate(pool_only) if a == "--pool-only-library"
    ]
    assert stems == [STEM, cu256]
    s04 = p.args_stage04(tmp_path / "run" / "round_0")
    assert [s04[i + 1] for i, a in enumerate(s04) if a == "--extra-library-stem"] == [
        cu256
    ]
    assert ro._pool_errors(p.cfg, p.paths) == []
    no_chip = dict(p.cfg, hardware=hardware)
    (err,) = ro._pool_errors(no_chip, p.paths)
    assert err.startswith("planner.bank: auto: no PCI chip id")
    explicit = dict(p.cfg, planner=dict(p.cfg["planner"], bank_library_stems=[cu256]))
    assert ro._pool_errors(explicit, p.paths) == []
    errors, _ = ro.validate_config(dict(p.cfg, planner={"bank": "all"}))
    assert any(e.startswith("planner.bank: 'all'") for e in errors)


def test_library_kind_preflight_and_pool_printout(tmp_path, monkeypatch, capsys):
    """`library.kind` names the pool whether the planner runs or not:
    benches load and stages index the bank's files, preflight resolves the
    pool on the first device and prints what it is made of; origami needs
    the library's Prediction table and a bank of kernels alone a kernel."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "-1")
    hardware = {"n_cu": 256, "lds_bytes": 65536, "l2_bytes": 4 << 20}
    cfg_path = make_config(
        tmp_path,
        **{
            "library": {"kind": ["origami", "bok"]},
            "train.catalog": "v3",
            "bench.pool_only_library": True,
            "hardware": dict(hardware, pci_chip_id=0x75A0),
        },
    )
    p = _pipeline(tmp_path, cfg_path)
    cu256 = _bank_library(p.paths.library_dir)
    assert ro._pool_errors(p.cfg, p.paths) == []
    assert p.bank_stems() == [cu256]
    s04 = p.args_stage04(tmp_path / "run" / "round_0")
    assert [s04[i + 1] for i, a in enumerate(s04) if a == "--extra-library-stem"] == [
        cu256
    ]
    assert ro.TILEWRIGHT_BANK_VAR in ro.stage07_capabilities(p.cfg)
    ro.print_pool(p.cfg, p.paths)
    out = capsys.readouterr().out
    assert "[pool] library.kind [origami, bok] with bok rows freesize, equality" in out
    assert "[pool] origami: 2 kernels" in out and "[pool] bok: 2 kernels" in out
    assert "[pool] pool: 4 kernels in" in out
    (err,) = ro._pool_errors(dict(p.cfg, hardware=hardware), p.paths)
    assert err.startswith("library.kind [origami, bok] with bok rows freesize, ")
    assert err.endswith("no PCI chip id for device 0: set hardware.pci_chip_id")
    rows = dict(p.cfg, library={"kind": "bok", "bok_rows": ["gridbased"]})
    (err,) = ro._pool_errors(rows, p.paths)
    assert "no file of the problem type of" in err
    on_bank = dict(p.cfg["hipblaslt"], library_stem=cu256)
    (err,) = ro._pool_errors(
        dict(p.cfg, hipblaslt=on_bank, library={"kind": "origami"}), p.paths
    )
    assert "has no Prediction table" in err and "library.kind bok" in err
    legacy = {k: v for k, v in p.cfg.items() if k != "library"}
    assert ro._pool_errors(dict(legacy, hipblaslt=on_bank), p.paths) == []
    ro.print_pool(dict(p.cfg, hipblaslt=on_bank), p.paths)
    assert "cannot read the kernel pool" in capsys.readouterr().err


def test_the_library_section_is_validated(tmp_path):
    base = yaml.safe_load(make_config(tmp_path).read_text())

    def errors(library, **train):
        cfg = dict(base, library=library, train=dict(base["train"], **train))
        return ro.validate_config(cfg)[0]

    assert errors({"kind": "origami"}) == []
    assert errors({"kind": ["origami", "bok"]}, catalog="v3") == []
    assert errors({"kind": "all"}) == [
        "library.kind: expected one of ['origami', 'bok'] or a list of distinct "
        "ones, got 'all'"
    ]
    assert errors({"kind": ["bok", "bok"]})[0].startswith("library.kind: expected")
    assert errors({"bok_rows": ["range"]})[0].startswith("library.bok_rows: expected")
    assert errors({"kinds": "bok"}) == ["library.kinds: unknown key"]
    assert errors({"kind": "bok"}) == [
        "library.kind bok needs train.catalog v3: only a v3 model tells TensileLite "
        "to rank the bank it was trained on"
    ]
    mixed = dict(base, library={"kind": "origami"}, planner={"bank": "none"})
    (error,) = ro.validate_config(mixed)[0]
    assert error.startswith("library and planner.bank both name the kernel pool")


def test_the_library_section_is_pinned_when_set(tmp_path):
    """A config without a library section keeps the pin it had before the
    section existed; one with it pins its keys, in an audit directory too
    (they name the kernels its benches measure)."""
    cfg = yaml.safe_load(make_config(tmp_path).read_text())
    for audit in (False, True):
        assert not any(k.startswith("library.") for k in ro.pinned_config(cfg, audit))
    cfg["library"] = {"kind": ["origami", "bok"]}
    for audit in (False, True):
        pinned = ro.pinned_config(cfg, audit)
        assert pinned["library.kind"] == ["origami", "bok"]
        assert pinned["library.bok_rows"] is None


def test_the_offset_rereads_are_pinned_only_when_they_change_something(tmp_path):
    """A pin written before the re-reads' tolerance_us, inline_copies and
    align_to_original existed holds none of them, so a config that leaves
    them at the values that change nothing keeps it, in an audit directory
    too; a config that sets another value pins it."""
    from lib import pinning
    from lib.rebench import NEUTRAL_REREADS, PREDATING_REREADS, REREAD_DEFAULTS

    assert set(ro.PINNED_UNLESS_NEUTRAL) == {
        f"bench.rebench.{k}" for k in NEUTRAL_REREADS
    }
    assert all(REREAD_DEFAULTS[k] == v for k, v in NEUTRAL_REREADS.items())
    assert all(PREDATING_REREADS[k] == v for k, v in NEUTRAL_REREADS.items())
    cfg = yaml.safe_load(make_config(tmp_path).read_text())

    def with_rebench(**keys):
        return dict(cfg, bench=dict(cfg.get("bench") or {}, rebench=keys))

    def pin(values):
        return {"config": pinning.config_record(values)}

    on = {"tolerance_us": 2.5, "inline_copies": 2, "align_to_original": True}
    for audit in (False, True):
        before = ro.pinned_config(cfg, audit)
        assert "bench.rebench.tolerance" in before
        assert not set(ro.PINNED_UNLESS_NEUTRAL) & set(before)
        neutral = ro.pinned_config(with_rebench(tolerance_us=0, inline_copies=1), audit)
        assert pinning.differences(pin(before), pin(neutral), ("config",)) == []
        changed = ro.pinned_config(with_rebench(**on), audit)
        assert {k: changed[f"bench.rebench.{k}"] for k in on} == on
        diffs = pinning.differences(pin(before), pin(changed), ("config",))
        assert sorted(d.split(":")[0] for d in diffs) == [
            f"config bench.rebench.{k}" for k in sorted(on)
        ]


def test_top_k_round0_is_pinned_only_when_set(tmp_path):
    """A config without planner.top_k_round0 ranks round 0 with top_k, as
    before the key existed, and keeps its pin; one with it pins it, though
    not in an audit directory, whose benches are exhaustive. Where no round
    ranks from seeds, the key is ignored with a warning."""
    cfg = yaml.safe_load(make_config(tmp_path, **{"planner.enabled": True}).read_text())
    before = {audit: ro.pinned_config(cfg, audit) for audit in (False, True)}
    assert "planner.top_k_round0" not in set(before[False]) | set(before[True])
    cfg["planner"]["top_k_round0"] = None
    for audit in (False, True):
        assert ro.pinned_config(cfg, audit) == before[audit]
    cfg["planner"]["top_k_round0"] = 64
    assert ro.validate_config(cfg)[0] == []
    assert ro.pinned_config(cfg) == {**before[False], "planner.top_k_round0": 64}
    assert ro.pinned_config(cfg, True) == before[True]
    bad = {**cfg, "planner": {**cfg["planner"], "top_k_round0": -1}}
    assert ro.validate_config(bad)[0] == ["planner.top_k_round0: must be >= 0, got -1"]

    def moot(**planner):
        warnings = ro.validate_config({**cfg, "planner": {**cfg["planner"], **planner}})
        return [w for w in warnings[1] if w.startswith("planner.top_k_round0")]

    assert moot(mode="planned") == [] == moot(mode="exhaustive", top_k_round0=None)
    assert moot(mode="planned", model_file="m.pt") == [
        "planner.top_k_round0 ignored: planner.model_file ranks every round, "
        "round 0 included"
    ]
    assert moot(mode="exhaustive") == [
        "planner.top_k_round0 ignored: planner.mode exhaustive benches every kernel"
    ]


def test_the_seed_recall_threshold_is_never_pinned(tmp_path):
    """planner.recall_min_ceiling only decides when stage02 warns about its
    seed recall: it changes no list, so no pin holds it."""
    cfg = yaml.safe_load(make_config(tmp_path, **{"planner.enabled": True}).read_text())
    before = {audit: ro.pinned_config(cfg, audit) for audit in (False, True)}
    cfg["planner"]["recall_min_ceiling"] = 0.5
    assert ro.validate_config(cfg)[0] == []
    for audit in (False, True):
        assert "planner.recall_min_ceiling" not in before[audit]
        assert ro.pinned_config(cfg, audit) == before[audit]
    cfg["planner"]["recall_min_ceiling"] = 1.5
    assert ro.validate_config(cfg)[0] == [
        "planner.recall_min_ceiling: must be <= 1, got 1.5"
    ]


def test_every_enrichment_the_orchestrator_starts_keeps_the_bank(tmp_path):
    """The pools ranked downstream hold the bank, so every stage04 the
    orchestrator builds keeps its rows: a round's, the seed bootstrap's, the
    context template's (which enriches the held-out contexts) and an audit's
    groups and increments."""
    bank = STEM.replace("ID75a0", "CU256_ID75a0")
    cfg_path = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.bank_library_stems": [bank],
            "train.catalog": "v3",
            "contexts.holdout": [{"cu_budget": 96, "cotenant": True}],
        },
    )
    p = _pipeline(tmp_path, cfg_path)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    contexts = json.loads(arg({"argv": p.args_stage03c(rd)}, "--commands"))
    audit = ro.AuditRun(p, p.cfg, [])
    unit = {"context": None, "parts": [[[64, 64, 64, 1]]]}
    built = [
        p.args_stage04(rd),
        p.seed_commands(rd, step)["stage04"],
        contexts["stage04"],
        audit.group_stage04_args(p.run_dir, "exclusive", unit, tmp_path / "g"),
        audit.args_stage04(p.run_dir, {"exclusive": unit}),
    ]
    for argv in built:
        extra = [argv[i + 1] for i, a in enumerate(argv) if a == "--extra-library-stem"]
        assert extra == [bank]


def test_planner_stage_cli_contract(tmp_path, pipeline_dir):
    """Every flag the planned round passes is one the real stage accepts."""
    cfg = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.ranker": "stage05",
            "planner.bank_library_stems": [STEM],
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    seed = p.seed_commands(rd, step)
    built = {
        "stage02": p.args_stage02(rd, step),
        "stage03": p.args_stage03(rd) + seed["stage03"][1:],
        "stage04": p.args_stage04(rd) + seed["stage04"][1:],
        "stage05": seed["stage05"][1:],
    }
    for key, args in built.items():
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        assert proc.returncode == 0, proc.stderr
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"


def test_rebench_runs_between_bench_and_enrich(tmp_path, stubs, env, pipeline_dir):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert run(stubs, make_config(plain), plain / "run", "--mode", "initial") == 0
    assert "stage03b" not in [c["stage"] for c in calls(env)]
    s04 = [c for c in calls(env) if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(plain / "run" / "round_0" / "stage03" / "logs")

    env.write_text("")
    cfg = make_config(tmp_path, **{"bench.rebench": {"enabled": True, "runs": 5}})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    log = calls(env)
    stages = [c["stage"] for c in log]
    assert stages[stages.index("stage03") : stages.index("stage03") + 3] == [
        "stage03",
        "stage03b",
        "stage04",
    ]
    rd = run_dir / "round_0"
    s03 = [c for c in log if c["stage"] == "stage03"][0]
    s03b = [c for c in log if c["stage"] == "stage03b"][0]
    assert arg(s03b, "--in-dir") == arg(s03, "--in-dir")
    assert arg(s03b, "--stage03-dir") == str(rd / "stage03")
    assert arg(s03b, "--out-dir") == str(rd / "stage03b")
    assert arg(s03b, "--devices") == arg(s03, "--devices")
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(rd / "stage03b" / "logs")
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.2") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03b", "stage04"]

    proc = subprocess.run(
        [
            sys.executable,
            str(pipeline_dir / "stages" / ro.STAGE_DEFS["stage03b"].script),
            "--help",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert proc.returncode == 0, proc.stderr
    built = _pipeline(tmp_path, cfg).args_stage03b(rd)
    assert {a for a in built if a.startswith("--")} <= defined_options(proc.stdout)


def test_sustained_finalists_run_before_enrich(tmp_path, stubs, env, pipeline_dir):
    sustained = {"enabled": True, "top_k": 6, "min_us": 50.0}
    cfg = make_config(
        tmp_path,
        **{"bench.rebench": {"enabled": True}, "bench.sustained": sustained},
    )
    run_dir = tmp_path / "run"
    rd = run_dir / "round_0"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    log = calls(env)
    stages = [c["stage"] for c in log]
    i = stages.index("stage03")
    assert stages[i : i + 4] == ["stage03", "stage03b", "stage03d", "stage04"]
    s03 = [c for c in log if c["stage"] == "stage03"][0]
    s03d = [c for c in log if c["stage"] == "stage03d"][0]
    assert arg(s03d, "--in-dir") == arg(s03, "--in-dir")
    assert arg(s03d, "--stage03-dir") == str(rd / "stage03b")
    assert arg(s03d, "--out-dir") == str(rd / "stage03d")
    assert arg(s03d, "--devices") == arg(s03, "--devices")
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(rd / "stage03b" / "logs")
    dirs = [str(d) for d in _pipeline(tmp_path, cfg).sustained_log_dirs(rd)]
    argv = s04["argv"]
    got = [argv[k + 1] for k, a in enumerate(argv) if a == "--sustained-log-dir"]
    assert got == dirs and len(dirs) == 2
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.7") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03d", "stage04"]
    snap = Path(arg(s03d, "--config-yaml")).read_text()
    assert yaml.safe_load(snap)["bench"]["sustained"] == dict(
        suslib.DEFAULTS, **sustained
    )
    assert "sustained finalists of power-capped GEMMs (winner >= 50 us" in " ".join(
        ln[2:] for ln in snap.splitlines() if ln.startswith("# ")
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(pipeline_dir / "stages" / ro.STAGE_DEFS["stage03d"].script),
            "--help",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert proc.returncode == 0, proc.stderr
    built = _pipeline(tmp_path, cfg).args_stage03d(rd)
    assert {a for a in built if a.startswith("--")} <= defined_options(proc.stdout)

    plain = tmp_path / "plain"
    plain.mkdir()
    env.write_text("")
    assert run(stubs, make_config(plain), plain / "run", "--mode", "initial") == 0
    assert "stage03d" not in [c["stage"] for c in calls(env)]
    s04 = [c for c in calls(env) if c["stage"] == "stage04"][0]
    assert "--sustained-log-dir" not in s04["argv"]
    bad = {"bench": {"sustained": {"measure_time": 500.0}}}
    errors = ro.validate_config(dict(yaml.safe_load(cfg.read_text()), **bad))[0]
    assert "bench.sustained.measure_time exceeds max_measure_time" in errors


def test_rebench_config_is_validated():
    bad_values = (
        {"tolerance": 0},
        {"tolerance_us": -1.0},
        {"runs": -1},
        {"fast_runs": -1},
        {"max_kernels": 1},
        {"inline_copies": 0},
        {"align_to_original": "yes"},
        {"window": 2},
        {"gemm_order": "random"},
    )
    for bad in bad_values:
        errors = ro.validate_config({"bench": {"rebench": {"enabled": True, **bad}}})[0]
        assert any(e.startswith("bench.rebench.") for e in errors), bad


def _with_bench(tmp_path, **bench):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["bench"] = {**dict(cfg.get("bench") or {}), **bench}
    return cfg


def test_rebench_rules_the_schema_cannot_state(tmp_path):
    rebench = {"enabled": True, "slow_min_iters": 20}
    errors = ro.validate_config(
        _with_bench(tmp_path, rebench=rebench, adaptive={"max_iters": 10})
    )[0]
    assert errors == ["bench.rebench.slow_min_iters exceeds bench.adaptive.max_iters"]
    rebench = {"fast_below_us": 2000.0}
    errors = ro.validate_config(_with_bench(tmp_path, rebench=rebench))[0]
    assert errors == ["bench.rebench.fast_below_us exceeds slow_from_us"]
    rebench = {"inline_copies": 300}
    errors = ro.validate_config(_with_bench(tmp_path, rebench=rebench))[0]
    assert errors == [
        "bench.rebench.inline_copies x max_kernels lists more kernels than a "
        f"bench line takes ({ro.MAX_BENCH_LIST})"
    ]
    rebench = {"inline_copies": 2, "max_kernels": 40}
    assert ro.validate_config(_with_bench(tmp_path, rebench=rebench))[0] == []
    rebench = {"align_to_original": True, "with_original": False}
    errors = ro.validate_config(_with_bench(tmp_path, rebench=rebench))[0]
    assert errors == [
        "bench.rebench.align_to_original shifts the re-reads to the original "
        "timings they merge with: it needs with_original"
    ]


def test_aligning_the_rereads_of_one_copy_is_warned_of(tmp_path):
    """With one inline copy, a GEMM with two near-winners gives a run too
    few differences to shift it, so align_to_original warns; two copies
    do not."""
    from lib import rebench as rb

    def warned(**rebench):
        cfg = _with_bench(tmp_path, rebench=rebench)
        errors, warnings = ro.validate_config(cfg)
        assert errors == []
        return [w for w in warnings if w.startswith("bench.rebench.")]

    assert warned() == [] and warned(inline_copies=1) == []
    assert warned(align_to_original=True, inline_copies=1) == [
        "bench.rebench." + rb.align_copies_warning(True, 1)
    ]
    assert warned(align_to_original=True) == warned(
        align_to_original=True, inline_copies=1
    )
    assert warned(align_to_original=True, inline_copies=2) == []


def test_rebench_defaults_match_the_stage(tmp_path):
    import stage03b_rebench_near_ties as s03b
    from lib import rebench as rb

    assert set(ro.SCHEMA.children["bench"].children["rebench"].children) == {
        "enabled",
        *s03b.CONFIG_KEYS,
    }
    for key in s03b.CONFIG_KEYS:
        assert ro.cfg_get({}, f"bench.rebench.{key}") == rb.REREAD_DEFAULTS[key]
        assert f"bench.rebench.{key}" in ro.BENCH_TIMING_KEYS
    old = {"bench": {"rebench": {"enabled": True, **rb.PREDATING_REREADS}}}
    assert ro.validate_config(_with_bench(tmp_path, **old["bench"]))[0] == []
    text = ro.bench_timing_text(old)
    assert "within 0.03 of the winner" in text and "read 3x below 10 us" in text
    assert "the median of the re-reads" in text and "sacrificial" not in text
    text = ro.bench_timing_text({"bench": {"rebench": {"enabled": True}}})
    assert "read 3x below 10 us, 3x from 1000 us, 1x in between" in text
    assert "a sacrificial first kernel, GEMMs shuffled" in text
    assert "the median of the original and the re-reads" in text
    band = {"enabled": True, **rb.PREDATING_SLOW_BAND}
    text = ro.bench_timing_text({"bench": {"rebench": band}})
    assert "read 3x below 10 us, 1x from 1000 us in windows of at least 10" in text
    plain = ro.bench_timing_text({"bench": {"rebench": {"enabled": True}}})
    for absent in ("whichever is wider", "listing every candidate", "shifted"):
        assert absent not in plain
    knobs = {
        "enabled": True,
        "tolerance_us": 2.5,
        "inline_copies": 2,
        "align_to_original": True,
    }
    text = ro.bench_timing_text({"bench": {"rebench": knobs}})
    assert (
        "the candidates within 0.05 or 2.5 us, whichever is wider, of the winner, "
        "at most 16" in text
    )
    assert "each in a bench of its own listing every candidate 2x, no screen" in text
    assert (
        "labelled by the median of the original and the re-reads, each run's "
        "re-reads shifted by their median difference from the original;" in text
    )


def test_the_v2_set_width_is_configured_and_never_pinned(tmp_path):
    from lib import audit_v2 as v2

    cfg, _ = ro.load_config(make_config(tmp_path))
    for key in ("set_tolerance", "set_tolerance_us", "set_max"):
        assert ro.cfg_get(cfg, f"audit.v2_{key}") == v2.DEFAULTS[key]
        assert f"audit.v2_{key}" in ro.UNPINNED_KEYS
    width = {"v2_set_tolerance": 0.25, "v2_set_tolerance_us": 2.5, "v2_set_max": 24}
    cfg["audit"] = dict(cfg.get("audit") or {}, **width)
    assert ro.validate_config(cfg)[0] == []
    for audit in (False, True):
        assert not any(k.startswith("audit.v2") for k in ro.pinned_config(cfg, audit))
    for bad in (
        {"v2_set_tolerance": -0.1},
        {"v2_set_tolerance_us": -1.0},
        {"v2_set_max": 0},
    ):
        cfg["audit"] = bad
        errors = ro.validate_config(cfg)[0]
        assert [e for e in errors if e.startswith("audit.v2_set")], bad


def test_long_lines_are_configured_and_stated(tmp_path):
    from lib.bench_yaml import ADAPTIVE_LONG_DEFAULTS

    on = {"bench": {"adaptive_long": {"enabled": True, "bandwidth_gbs": 100.0}}}
    assert ro.validate_config(_with_bench(tmp_path, **on["bench"]))[0] == []
    for key in ADAPTIVE_LONG_DEFAULTS:
        assert (
            ro.cfg_get({}, f"bench.adaptive_long.{key}") == ADAPTIVE_LONG_DEFAULTS[key]
        )
        assert f"bench.adaptive_long.{key}" in ro.BENCH_TIMING_KEYS
    snap = ro.resolve_bench_timing({})
    assert snap["bench"]["adaptive_long"] == ADAPTIVE_LONG_DEFAULTS
    text = ro.bench_timing_text(on)
    assert "on lines whose roofline is at least 100 us" in text
    assert "warmup_time 0.5, sample_time 0.25, max_measure_time 4.0" in text
    assert "roofline" not in ro.bench_timing_text({})
    bad = {"enabled": True, "max_measure_time": 1.0}
    assert ro.validate_config(_with_bench(tmp_path, adaptive_long=bad))[0] == [
        "bench.adaptive with bench.adaptive_long: measure_time exceeds "
        "max_measure_time"
    ]
    off = {"max_measure_time": 1.0}
    assert ro.validate_config(_with_bench(tmp_path, adaptive_long=off))[0] == []
    for bad in ({"min_roofline_us": 0}, {"peak_tflops": 0}, {"warmup_time": -1}):
        errors = ro.validate_config(_with_bench(tmp_path, adaptive_long=bad))[0]
        assert any(e.startswith("bench.adaptive_long.") for e in errors), bad


def test_the_roofline_rates_are_read_when_the_run_starts(monkeypatch):
    from lib import hardware as hwlib

    calls = []

    def rates(cfg, device):
        calls.append(device)
        return hwlib.RooflineRates(321.0, "a peak", None, "no memory")

    monkeypatch.setattr(hwlib, "roofline_rates", rates)
    off = {"bench": {"adaptive_long": {"peak_tflops": None}}}
    assert ro.resolve_roofline(off, 0) == (off, [])
    given = {"bench": {"adaptive_long": {"enabled": True, "peak_tflops": 1.0}}}
    given["bench"]["adaptive_long"]["bandwidth_gbs"] = 2.0
    assert ro.resolve_roofline(given, 0) == (given, []) and calls == []
    cfg = {"bench": {"adaptive_long": {"enabled": True}, "rebench": {"runs": 2}}}
    out, (peak, bw) = ro.resolve_roofline(cfg, 3)
    assert calls == [3] and "peak_tflops" not in cfg["bench"]["adaptive_long"]
    assert out["bench"]["adaptive_long"] == {"enabled": True, "peak_tflops": 321.0}
    assert out["bench"]["rebench"] == {"runs": 2}
    assert peak == "bench.adaptive_long.peak_tflops 321 (bench device 3: a peak)"
    assert bw.startswith("bench.adaptive_long.bandwidth_gbs unknown (no memory)")
    kept = {"bench": {"adaptive_long": {"enabled": True, "bandwidth_gbs": 5.0}}}
    out, (line,) = ro.resolve_roofline(kept, 0)
    assert out["bench"]["adaptive_long"]["bandwidth_gbs"] == 5.0
    assert out["bench"]["adaptive_long"]["peak_tflops"] == 321.0
    monkeypatch.setattr(
        hwlib,
        "roofline_rates",
        lambda cfg, d: hwlib.RooflineRates(None, "no peak", None, "no memory"),
    )
    with pytest.raises(ro.ConfigError, match="neither peak_tflops nor bandwidth_gbs"):
        ro.resolve_roofline(cfg, 0)
    # Without a peak the bound is the memory term, which needs every data
    # type's size: a type of unknown size leaves no bound to classify a line.
    monkeypatch.setattr(
        hwlib,
        "roofline_rates",
        lambda cfg, d: hwlib.RooflineRates(None, "no peak", 5000.0, "a memory"),
    )
    types = {"a_type": "bf16_r", "b_type": "bf16_r", "c_type": "bf16_r"}
    known = dict(cfg, hipblaslt=dict(types, d_type="bf16_r"))
    out, _ = ro.resolve_roofline(known, 0)
    assert out["bench"]["adaptive_long"]["bandwidth_gbs"] == 5000.0
    odd = dict(cfg, hipblaslt=dict(types, d_type="mx9_r"))
    with pytest.raises(ro.ConfigError, match="gives no peak") as e:
        ro.resolve_roofline(odd, 0)
    assert "bench.adaptive_long.peak_tflops" in str(e.value.errors)
    odd["bench"] = {"adaptive_long": {"enabled": True, "peak_tflops": 1.0}}
    assert ro.resolve_roofline(odd, 0)[0]["bench"]["adaptive_long"] == {
        "enabled": True,
        "peak_tflops": 1.0,
        "bandwidth_gbs": 5000.0,
    }


def test_runtime_env_launch_overrides_are_warned_about(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["runtime"]["env"] = {"TENSILE_PRECISE_SM_TARGET": "1", "MY_KNOB": "x"}
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert any(
        w.startswith("runtime.env sets the launch overrides TENSILE_PRECISE_SM_TARGET")
        for w in warnings
    )
    assert not [w for w in warnings if "hybrid schedule" in w]
    # The pipeline sets neither hybrid force mode (a context's schedule is a
    # bench option): runtime.env may set one, with a warning of its own.
    cfg["runtime"]["env"] = {"TENSILE_PERSISTENT_HYBRID_FORCE_MODE": "1"}
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    (forced,) = [w for w in warnings if "hybrid schedule" in w]
    assert forced.startswith("runtime.env sets TENSILE_PERSISTENT_HYBRID_FORCE_MODE")
    assert "--hybrid_assignment_policy" in forced


def test_warmup_pass_default_matches_the_stage():
    import argparse

    import stage03_load_balance_offline_tuning as s03

    unset = argparse.Namespace(warmup_pass=None)
    assert s03._resolve_warmup(unset, {}) == (
        ro.cfg_get({}, "bench.warmup_pass"),
        "default",
    )


def test_canary_defaults_match_the_stage():
    import argparse

    import stage03_load_balance_offline_tuning as s03
    from bench_fakes import Gemm, bench_line

    keys = ("canary", "canary_tolerance", "canary_timings", "canary_backoff_s")
    unset = argparse.Namespace(**dict.fromkeys(keys))
    line = bench_line(Gemm(64, 64, 64), scaleA=0, scaleB=0)
    assert ro.cfg_get({}, "bench.canary.enabled") is False
    assert s03._resolve_canary(unset, {}, [line]) == (None, "turned off")
    on = {"bench": {"canary": {"enabled": True}}}
    assert ro.cfg_get(on, "bench.canary.enabled") is True
    canary, _ = s03._resolve_canary(unset, on, [line])
    for key in ("tolerance", "timings", "backoff_s"):
        assert ro.cfg_get({}, f"bench.canary.{key}") == getattr(canary, key)
    text = ro.bench_timing_text(on)
    assert "canary timed 3x before and after every bench process's shapes" in text
    assert "benched again once, after 120 s, when a reading is over 0.1" in text
    errors, _ = ro.validate_config(
        {"bench": {"canary": {"tolerance": 0, "timings": 0, "backoff_s": -1}}}
    )
    for key in ("tolerance", "timings", "backoff_s"):
        assert any(e.startswith(f"bench.canary.{key}") for e in errors)


def test_planner_keys_without_enabled_warn(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["planner"] = {"mode": "planned"}
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert "planner: mode ignored (planner.enabled is false)" in warnings


TINY = "Tiny|Tiny|TinyK|Bnone"
LARGE = "Large|Large|LargeK|Bnone"


def initial_run(stubs, tmp_path, **overrides):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, **overrides})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    return cfg, run_dir


def write_cells(run_dir, rnd, rows):
    """round_<rnd>/stage05/cells.json with (label, validation sel_eff,
    training GEMMs) entries."""
    cells = [
        {
            "label": label,
            "n_gemms": n,
            "deployed": {"sel_eff": min(1.0, se + 0.02), "n_eval": n},
            "deployed_val": {"sel_eff": se, "n_eval": n // 4},
        }
        for label, se, n in rows
    ]
    path = run_dir / f"round_{rnd}" / "stage05" / "cells.json"
    path.write_text(json.dumps({"cells": cells}))


def test_enhance_round_retrains_only_the_chosen_cells(
    tmp_path, stubs, env, monkeypatch
):
    cfg, run_dir = initial_run(stubs, tmp_path)
    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_1": f"{TINY},{LARGE}"}))
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 0
    by_stage = {}
    for c in calls(env)[n:]:
        by_stage.setdefault(c["stage"], []).append(c)
    assert arg(by_stage["stage01"][0], "--only-cells") == TINY
    assert arg(by_stage["stage01"][0], "--mode") == "active"
    assert "--final-round" in by_stage["stage04b"][0]["argv"]
    assert arg(by_stage["stage05"][0], "--only-cells") == TINY
    assert len(by_stage["stage06"]) == 1
    rd = run_dir / "round_1"
    marker = json.loads((rd / ro.ROUND_MARKER).read_text())
    assert marker["mode"] == "enhance" and marker["deployed"]
    record = json.loads((rd / ro.ENHANCE_RECORD).read_text())
    assert record["targets"] == [TINY]
    assert record["retrain"] == [TINY]
    assert record["deferred"] == [LARGE]
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert (run_dir / "round_2" / ro.ROUND_MARKER).is_file()


def test_enhance_dry_run_writes_nothing(tmp_path, stubs, env, capsys):
    cfg, run_dir = initial_run(stubs, tmp_path)
    write_cells(run_dir, 0, [(TINY, 0.80, 40), (LARGE, 0.97, 40)])
    before = sorted(str(p) for p in run_dir.rglob("*"))
    n = len(calls(env))
    capsys.readouterr()
    rc = run(
        stubs,
        cfg,
        run_dir,
        "--mode",
        "enhance",
        "--weakest",
        "3",
        "--shapes-per-cell",
        "60",
        "--enhance-rounds",
        "2",
        "--dry-run",
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert sorted(str(p) for p in run_dir.rglob("*")) == before
    assert len(calls(env)) == n
    assert out.count("--only-cells") == 4
    assert f"--only-cells '{TINY}'" in out
    assert LARGE not in out
    assert "--shapes-per-cell 60" in out
    assert "round_1" in out and "round_2" in out and "(final round)" in out
    assert out.count("[stage06] ") == 2 and out.count("stage06_") == 1


def test_enhance_selects_weak_cells_from_training_evidence(tmp_path, stubs, env):
    _, run_dir = initial_run(stubs, tmp_path)
    mid = "Mid|Mid|MidK|Bnone"
    write_cells(run_dir, 0, [(TINY, 0.80, 40), (LARGE, 0.90, 400), (mid, 0.97, 40)])
    rounds = ro.complete_prefix(run_dir)

    def pick(**kw):
        opts = dict(cells=[], weakest=None, below=None, rank_by="sel_eff")
        opts.update(kw)
        return ro.select_enhance_cells(run_dir, rounds, target=0.95, **opts)

    assert pick(weakest=1)["targets"] == [TINY]
    assert pick(weakest=1, rank_by="gain")["targets"] == [LARGE]
    assert pick(weakest=5)["targets"] == [TINY, LARGE]
    assert pick(below=0.85)["targets"] == [TINY]
    assert pick(below=0.99)["targets"] == [TINY, LARGE, mid]
    assert pick(cells=[mid], weakest=1)["targets"] == [mid, TINY]
    sel = pick(weakest=1)["selection"]
    assert sel["picked"][0]["source"] == "validation"
    assert sel["evidence_rounds"] == ["round_0"]
    with pytest.raises(ro.ConfigError):
        pick(below=0.5)
    with pytest.raises(ro.ConfigError):
        pick(cells=["Nope|Tiny|TinyK|Bnone"])

    s8 = run_dir / "validate" / "stage08"
    s8.mkdir(parents=True, exist_ok=True)
    (s8 / "summary.json").write_text(
        json.dumps({"per_cell": {mid: {"sel_eff": 0.1}, LARGE: {"sel_eff": 0.1}}})
    )
    (s8 / "per_gemm.csv").write_text(f"cell,sel_eff\n{mid},0.1\n")
    incomplete = run_dir / "round_1" / "stage04b"
    incomplete.mkdir(parents=True)
    (incomplete / "decisions.json").write_text(
        json.dumps({"per_cell": [{"cell": mid, "sel_eff_new": 0.1, "n_eval": 50}]})
    )
    assert pick(weakest=1)["targets"] == [TINY]
    assert pick(weakest=1, rank_by="gain")["targets"] == [LARGE]


def test_enhance_resumes_with_the_recorded_selection(tmp_path, stubs, env, monkeypatch):
    cfg, run_dir = initial_run(stubs, tmp_path)
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 1
    assert run(stubs, cfg, run_dir, "--mode", "active") == 2
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", LARGE) == 0
    rerun = calls(env)[n:]
    assert [c["stage"] for c in rerun][:2] == ["stage05", "stage06"]
    assert arg(rerun[0], "--only-cells") == TINY


def test_enhance_splits_hand_their_children_to_stage05(tmp_path):
    from lib import subcells as sc

    p = _pipeline(tmp_path, make_config(tmp_path, **{"train.min_cell_gemms": 20}))
    run_dir = tmp_path / "run"
    rd = run_dir / "round_1"
    (rd / "stage04b").mkdir(parents=True)
    rule = sc.build_split_rule(LARGE, "M", 4096)
    sc.write_splits_json(rd / "stage04b" / "splits.json", 1, [rule])
    rows = [
        {"cell": LARGE, "n_training_gemms": 80},
        {"cell": rule.lo_label, "n_training_gemms": 40},
        {"cell": rule.hi_label, "n_training_gemms": 40},
        {"cell": TINY, "n_training_gemms": 5},
    ]
    (rd / "stage04b" / "decisions.json").write_text(json.dumps({"per_cell": rows}))
    step = ro.RoundStep(
        1, "active", [run_dir / "round_0"], None, True, targets=[LARGE, TINY]
    )
    listed = f"{rule.lo_label},{rule.hi_label},Mid|Mid|MidK|Bnone"
    got = p._enhance_retrain(rd, step, listed)
    assert got.split(",") == [rule.lo_label, rule.hi_label]
    record = json.loads((rd / ro.ENHANCE_RECORD).read_text())
    assert record["too_few_gemms"] == [TINY]
    assert record["deferred"] == ["Mid|Mid|MidK|Bnone"]


def test_enhance_preconditions(tmp_path, stubs, env, capsys):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for argv in (
        ["--mode", "active", "--cells", TINY],
        ["--mode", "full", "--dry-run"],
        ["--mode", "enhance"],
        ["--mode", "enhance", "--weakest", "0"],
        ["--mode", "enhance", "--below", "95"],
        ["--mode", "enhance", "--cells", TINY, "--from-round", "1"],
    ):
        with pytest.raises(SystemExit) as e:
            run(stubs, cfg, run_dir, *argv)
        assert e.value.code == 2, argv
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 2
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    bad = "Huge|Tiny|TinyK|Bnone"
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", bad) == 2
    assert not (run_dir / "round_1").exists()
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--below", "0.5") == 2
    assert not (run_dir / "round_1").exists()
    assert [c["stage"] for c in calls(env)].count("stage01") == 1


# ── validation markers, run status, stage 7 library directory ───────────────


def status_of(run_dir):
    return json.loads((run_dir / ro.RUN_STATUS).read_text())


def test_validation_resumes_until_it_passes(tmp_path, stubs, env, monkeypatch):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    monkeypatch.setenv("STUB_RC_stage07", "1")
    assert run(stubs, cfg, run_dir, "--mode", "full") == 1
    status = status_of(run_dir)
    assert status["rounds"]["complete"] == 2 and not status["complete"]
    v = status["validation"]
    assert v["stage07"]["status"] == "failed" and v["stage07"]["rc"] == 1
    assert v["stage08"]["status"] == "passed"
    assert status["invocation"]["exit_status"] == 1
    assert ro.validated_by(run_dir / "validate" / "stage07") is None
    record = run_dir / "round_1" / "stage06" / "deploy_record.json"
    assert ro.validated_by(run_dir / "validate" / "stage08")["train_dir"] == str(
        record.parent.parent / "stage05"
    )

    monkeypatch.delenv("STUB_RC_stage07")
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == ["stage07"]
    status = status_of(run_dir)
    assert status["complete"] and status["invocation"]["exit_status"] == 0
    assert status["validation"]["stage07"]["status"] == "passed"
    assert status["validation"]["stage08"]["earlier"]
    assert ro.validated_by(run_dir / "validate" / "stage07")["deploy_record"] == str(
        record
    )
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert calls(env)[n:] == []
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 2, "stage07.timing_repetitions": 5}
    )
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == ["stage07"]
    n = len(calls(env))
    cfg = make_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "stage07.timing_repetitions": 5,
            "train.repeat_aggregation": "min",
        },
    )
    assert run(stubs, cfg, run_dir, "--mode", "full") == 2
    assert calls(env)[n:] == []
    assert run(stubs, cfg, run_dir, "--mode", "full", "--allow-code-change") == 0
    assert [c["stage"] for c in calls(env)[n:]] == ["stage08"]


def test_a_pass_of_another_bundle_reads_stale(tmp_path, stubs, env, monkeypatch):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert status_of(run_dir)["complete"]
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1")
    assert run(stubs, cfg, run_dir, "--mode", "full", "--from-round", "1") == 1
    status = status_of(run_dir)
    assert {k: v["status"] for k, v in status["validation"].items()} == {
        "stage07": "stale",
        "stage08": "stale",
    }
    passed = status["validation"]["stage07"]["validated"]["deploy_record"]
    assert passed.endswith("round_1/stage06/deploy_record.json")
    assert not status["complete"] and status["deployed"] is None


def test_a_pass_recorded_before_the_record_reads_passed_unrecorded(
    tmp_path, stubs, env
):
    """A validation marker without VALIDATED (written before markers said
    what a pass vouches for) reads as a pass that is not recorded, which the
    next --mode full runs again; a summary reads failed only when it reports
    a failure."""
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    vdir = run_dir / "validate"
    for key in ("stage07", "stage08"):
        marker = json.loads((vdir / key / ro.STAGE_MARKER).read_text())
        del marker[ro.VALIDATED]
        (vdir / key / ro.STAGE_MARKER).write_text(json.dumps(marker))
    status = ro.RunStatus(run_dir, "initial")
    status.cfg = ro.load_config(cfg)[0]
    for key in ("stage07", "stage08"):
        entry = status.validation_entry(key, True)
        assert entry["status"] == "passed_unrecorded"
        assert entry["marker"] == str(vdir / key / ro.STAGE_MARKER)
    assert not status.doc()["complete"]
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == ["stage07", "stage08"]
    assert status_of(run_dir)["complete"]

    for key, summary in (
        ("stage07", {"ok": True}),
        ("stage08", {"datasets": [{"name": "ds", "rc": 0}]}),
    ):
        (vdir / key / ro.STAGE_MARKER).unlink()
        (vdir / key / "summary.json").write_text(json.dumps(summary))
        assert status.validation_entry(key, True)["status"] == "pending"
    (vdir / "stage07" / "summary.json").write_text(json.dumps({"ok": False}))
    (vdir / "stage08" / "summary.json").write_text(
        json.dumps({"datasets": [{"name": "ds", "rc": 1}]})
    )
    for key in ("stage07", "stage08"):
        assert status.validation_entry(key, True)["status"] == "failed"


def test_a_pass_with_the_origami_reference_stands_for_one_without(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert ro.validated_by(run_dir / "validate" / "stage07")["origami_reference"] is (
        False
    )
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full", "--origami-reference") == 0
    (s07,) = [c for c in calls(env)[n:] if c["stage"] == "stage07"]
    assert "--origami-reference" in s07["argv"]
    assert [c["stage"] for c in calls(env)[n:]] == ["stage07"]
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert calls(env)[n:] == []
    want = ro.validated_by(run_dir / "validate" / "stage07")
    assert want["origami_reference"] is True
    assert ro.validation_covers(want, dict(want, origami_reference=False))
    assert not ro.validation_covers(dict(want, origami_reference=False), want)
    assert not ro.validation_covers(want, dict(want, model_sha256="other"))
    assert not ro.validation_covers(None, want)


def test_stage07_production_route_turns_the_route_off(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    (s07,) = [c for c in calls(env) if c["stage"] == "stage07"]
    assert "--no-route" not in s07["argv"]
    want = ro.validated_by(run_dir / "validate" / "stage07")
    assert want["stage07"]["production_route"] is True
    n = len(calls(env))
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 2, "stage07.production_route": False}
    )
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    (s07,) = [c for c in calls(env)[n:] if c["stage"] == "stage07"]
    assert "--no-route" in s07["argv"]
    assert [c["stage"] for c in calls(env)[n:]] == ["stage07"]


def test_full_preflight_refuses_a_run_that_deploys_nothing_for_stage7(
    tmp_path, stubs, env, capsys, monkeypatch
):
    def errors(run_dir, deploy=False, **overrides):
        cfg, _ = ro.load_config(make_config(tmp_path, **overrides))
        total = ro.cfg_get(cfg, "full.total_rounds")
        steps = ro.plan_training("full", run_dir, total, None, None)
        return ro.deploy_errors(cfg, run_dir, steps, deploy)

    fresh = tmp_path / "fresh"
    assert errors(fresh) == []
    one = {"full.total_rounds": 1}
    (err,) = errors(fresh, **one)
    assert err.endswith(
        "a run of one round deploys only with --deploy; pass --deploy or turn "
        "stage 7 off"
    )
    assert errors(fresh, True, **one) == []
    assert errors(fresh, **one, **{"stage07.bench_yamls": []}) == []
    later = {"deploy.on_final_round": False}
    capsys.readouterr()
    assert errors(fresh, **later) == []
    assert "ends with stage 7 not run (exit 2) when no round reaches it" in (
        capsys.readouterr().err
    )
    (err,) = errors(fresh, **later, **{"deploy.target_sel_eff": None})
    assert err.endswith(
        "and deploy.target_sel_eff is unset; pass --deploy, set "
        "deploy.on_final_round or turn stage 7 off"
    )
    (err,) = errors(fresh, **later, **{"deploy.gate_frac": 0})
    assert "and deploy.gate_frac 0 sets no gate GEMMs aside" in err

    run_dir = tmp_path / "run"
    cfg = make_config(tmp_path, **one)
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    (err,) = errors(run_dir, **one)
    assert err.endswith(
        "every planned round is complete; deploy round_0 with --from-round 0 "
        "--from-stage 6 --deploy or turn stage 7 off"
    )
    argv = ["--config", str(cfg), "--run-dir", str(run_dir), "--no-report"]
    argv += ["--stages-dir", str(stubs), "--mode", "full"]
    monkeypatch.setattr(ro, "preflight", lambda *a, **k: [])
    n = len(calls(env))
    assert ro.main(argv) == 2
    assert calls(env)[n:] == []
    stage07 = status_of(run_dir)["validation"]["stage07"]
    assert stage07["status"] == "not_run" and stage07["reason"] == err
    assert (
        run(
            stubs,
            cfg,
            run_dir,
            *"--mode full --from-round 0 --from-stage 6".split(),
            "--deploy",
        )
        == 0
    )
    assert errors(run_dir, **one) == []
    assert errors(run_dir, **{"full.total_rounds": 2}) == []


def test_audit_mode_warns_about_what_only_stage07_needs(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["deploy"].update(rocm_libraries_to_deploy=None, apply_patch=False)
    cfg["deploy"]["build_after_apply"] = False
    errors, _ = ro.validate_config(cfg, "full")
    (err,) = errors
    assert err.startswith("stage07 needs a hipBLASLt build")
    errors, warnings = ro.validate_config(cfg, "audit")
    assert errors == [] and f"{err} (--mode audit runs no stage 7)" in warnings
    assert ro.validate_config(cfg)[0] == [err]


def test_full_mode_fails_while_stage07_has_no_deployed_bundle(
    tmp_path, stubs, env, capsys
):
    """Without the preflight that refuses it (--skip-preflight), such a run
    ends with 2."""
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 2
    assert "stage 7 needs a model deployed by this run" in capsys.readouterr().err
    assert not [c for c in calls(env) if c["stage"] == "stage07"]
    v = status_of(run_dir)["validation"]
    assert v["stage07"] == {
        "enabled": True,
        "status": "not_run",
        "reason": "this run has deployed no model",
    }
    assert v["stage08"]["status"] == "passed"
    assert run(stubs, cfg, tmp_path / "stepwise", "--mode", "initial") == 0
    assert status_of(tmp_path / "stepwise")["validation"]["stage07"]["status"] == (
        "not_run"
    )


def test_run_status_records_the_rounds_and_the_one_that_failed(
    tmp_path, stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1")
    cfg = make_config(tmp_path, **{"full.total_rounds": 3})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 1
    status = status_of(run_dir)
    rounds = status["rounds"]
    assert (rounds["planned"], rounds["complete"]) == (3, 1)
    assert rounds["failed"] == {"round": 1, "rc": 1}
    assert [r["complete"] for r in rounds["per_round"]] == [True, False]
    assert status["validation"]["stage07"]["status"] == "pending"
    assert status["report"] == {"status": "off"} and not status["complete"]
    assert status["invocation"]["mode"] == "full"
    assert status["invocation"]["exit_status"] == 1
    marker = json.loads((run_dir / "round_0" / ro.ROUND_MARKER).read_text())
    assert 0 < marker["bytes"] <= ro.round_bytes(run_dir / "round_0")


def test_stage07_serves_the_staged_model_from_a_library_directory(
    tmp_path, stubs, env, monkeypatch
):
    release = tmp_path / "bench" / "projects" / "hipblaslt" / "build" / "release"
    lib = release / "Tensile" / "library" / "gfx950"
    lib.mkdir(parents=True)
    (lib / f"{STEM}.dat").write_bytes(b"logic")
    other = "TensileLibrary_HH_HH_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    (lib / "tilewright_index").write_text(f"{STEM}\tshipped.bin\n{other}\to.bin\n")
    v3 = tmp_path / "v3build"
    cfg = make_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "stage07.build_dir": str(v3),
            "stage07.cpus": "4,5",
            "deploy.rocm_libraries_to_deploy": None,
            "deploy.apply_patch": False,
            "deploy.build_after_apply": False,
        },
    )
    monkeypatch.setenv("STUB_STAGED", "gfx950")
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    (s07,) = by_stage(calls(env))["stage07"]
    libdir = run_dir / "validate" / "stage07_lib" / "gfx950"
    assert arg(s07, "--build-dir") == str(v3)
    assert arg(s07, "--bench-binary") == str(v3 / "clients" / "hipblaslt-bench")
    assert arg(s07, "--library-dir") == str(libdir)
    assert arg(s07, "--cpus") == "4,5"
    rec = json.loads(
        (run_dir / "round_1" / "stage06" / "deploy_record.json").read_text()
    )
    staged = Path(rec["staged_model"]).read_bytes()
    assert (libdir / "w.tilewright.bin").read_bytes() == staged
    assert (libdir / f"{STEM}.dat").resolve() == (lib / f"{STEM}.dat").resolve()
    assert (libdir / "tilewright_index").read_text() == (
        f"{other}\to.bin\n{STEM}\tw.tilewright.bin\n"
    )
    assert ro.validated_by(run_dir / "validate" / "stage07")["build_dir"] == str(v3)


def test_stage07_times_a_sample_of_the_audit_set(tmp_path):
    from lib import audit as auditlib
    from lib.bench_yaml import parse_shapes_yaml

    audit = tmp_path / "audit"
    (audit / "stage01").mkdir(parents=True)
    (audit / "stage01" / "shapes.yaml").write_text(
        "- {M: 64, N: 128, K: 256, batch_count: 1}\n"
        "- {M: 512, N: 512, K: 512, batch_count: 4}\n"
    )
    gens = [
        {"stage04": ["bench/exclusive/stage04", "bench/cu64_cot/stage04"]},
        {"stage04": "increments/inc_001/stage04"},
    ]
    (audit / auditlib.MANIFEST).write_text(
        json.dumps({"schema": auditlib.SCHEMA_VERSION, "generations": gens})
    )
    cfg = make_config(
        tmp_path, **{"audit.dir": str(audit), "stage07.parity_sample_audit": 5}
    )
    p = _pipeline(tmp_path, cfg)
    vdir = tmp_path / "run" / "validate"
    yamls, held_out = p._stage07_bench_yamls(vdir)
    assert yamls == [tmp_path / "w.yaml"]
    assert held_out == [vdir / "stage07_audit_5.yaml"]
    assert sorted(parse_shapes_yaml(held_out[0])) == [
        (64, 128, 256, 1),
        (512, 512, 512, 4),
    ]
    args = p.args_stage07(vdir, vdir / "r.json", yamls, held_out=held_out)
    assert _after(args, "--bench-yaml") == [str(tmp_path / "w.yaml")]
    assert _after(args, "--held-out-yaml") == [str(held_out[0])]
    assert _after(args, "--timings-csv-dir") == [
        str(audit / "bench" / "exclusive" / "stage04"),
        str(audit / "increments" / "inc_001" / "stage04"),
    ]


def test_stage07_needs_a_build_that_serves_the_model(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["deploy"].update(rocm_libraries_to_deploy=None, apply_patch=False)
    cfg["deploy"]["build_after_apply"] = False
    errors, _ = ro.validate_config(cfg)
    assert any(e.startswith("stage07 needs a hipBLASLt build") for e in errors)
    cfg["stage07"]["build_dir"] = "/v3build"
    assert ro.validate_config(cfg)[0] == []
    cfg["stage07"]["cpus"] = "four"
    (err,) = ro.validate_config(cfg)[0]
    assert err.startswith("stage07.cpus: 'four' is not a CPU list")
    cfg["stage07"]["cpus"] = "4,5"
    cfg["stage07"]["parity_sample_audit"] = 50
    (err,) = ro.validate_config(cfg)[0]
    assert "stage07.parity_sample_audit samples the audit set" in err


def test_preflight_checks_what_the_stage07_build_serves(tmp_path):
    import fake_hipblaslt as fh

    v3 = tmp_path / "v3"
    fh.write_fake_build(v3, "gfx950", STEM, capabilities=("MLREC_v2",))
    cfg, _ = ro.load_config(
        make_config(tmp_path, **{"stage07.build_dir": str(v3), "train.catalog": "v3"})
    )
    paths = ro.derive_paths(cfg)
    (err,) = ro.stage07_build_errors(cfg, paths)
    assert err.startswith(f"stage07.build_dir {v3}: its TensileLite")
    assert "cannot load MLREC_v3 models (train.catalog v3); no 'MLREC_v3'" in err
    fh.write_fake_build(v3, "gfx950", STEM, capabilities=("MLREC_v3",))
    assert ro.stage07_build_errors(cfg, paths) == []
    cfg["planner"] = {"enabled": True, "bank_library_stems": [STEM + "_bank"]}
    (err,) = ro.stage07_build_errors(cfg, paths)
    assert (
        "does not serve a bank of kernels" in err and "TENSILE_TILEWRIGHT_BANK" in err
    )
    fh.write_fake_build(
        v3, "gfx950", STEM, capabilities=("MLREC_v3", "TENSILE_TILEWRIGHT_BANK")
    )
    assert ro.stage07_build_errors(cfg, paths) == []
    cfg["stage07"]["cpus"] = f"0,{os.cpu_count()}"
    (err,) = ro.stage07_build_errors(cfg, paths)
    assert err.startswith(
        f"stage07.cpus names CPUs this host does not have: [{os.cpu_count()}]"
    )
    cfg["stage07"]["cpus"] = None
    for f in (v3 / "tensilelite").iterdir():
        f.unlink()
    (v3 / "clients" / "hipblaslt-bench").unlink()
    errs = ro.stage07_build_errors(cfg, paths)
    assert errs[0].startswith("stage07.build_dir hipblaslt-bench not found")
    assert "holds no TensileLite library" in errs[1]
    deploy, _ = ro.load_config(make_config(tmp_path))
    errs = ro.stage07_build_errors(deploy, ro.derive_paths(deploy))
    assert errs[0].startswith("the deploy build hipblaslt-bench not found")
    assert errs[1].startswith("deploy library directory is missing")


def test_preflight_checks_the_stage08_datasets(tmp_path):
    entry = {
        "M": 64,
        "N": 64,
        "K": 64,
        "transA": "T",
        "transB": "N",
        "a_type": "bf16_r",
        "compute_type": "c_f32_r",
    }
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump([entry, dict(entry, transA="N")]))
    other = tmp_path / "nn.yaml"
    other.write_text(yaml.safe_dump([dict(entry, transA="N", transB="N")]))
    enriched = tmp_path / "enriched"
    enriched.mkdir()
    (enriched / "chunk_0000.csv").write_text(
        "m,n,k,batch_count,transA,transB,a_type,compute_type\n"
        "64,64,64,1,T,N,bf16_r,f32_r\n"
    )
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    datasets = [
        {"name": n, "path": str(p)}
        for n, p in (
            ("good", good),
            ("nn", other),
            ("gone", tmp_path / "gone.yaml"),
            ("notes", notes),
            ("csv", enriched),
        )
    ]
    cfg, _ = ro.load_config(make_config(tmp_path, **{"stage08.datasets": datasets}))
    errs = ro._dataset_errors(cfg)
    assert len(errs) == 3
    assert errs[0].startswith("stage08.datasets nn: none of the first 1000 problems of")
    assert errs[0].endswith("; its first one has transA=N")
    assert errs[1] == f"stage08.datasets gone: {tmp_path / 'gone.yaml'} does not exist"
    assert errs[2].startswith("stage08.datasets notes: ") and "neither" in errs[2]


def problem_log(g):
    import bench_fakes as bf

    return ["Is supported 1 / Total solutions: 1", *bf.measured_lines(0, g, 7, 10.0)]


def test_dataset_preflight_reads_until_a_problem_matches(tmp_path, monkeypatch):
    """A bench yaml is read a line at a time and only until its first
    problem of the config's types; directories are read file after file."""
    import bench_fakes as bf

    nn = bf.bench_line(bf.Gemm(64, 64, 64, transB="T"))
    tn = bf.bench_line(bf.Gemm(64, 64, 64))
    lines = tmp_path / "lines.yaml"
    lines.write_text(nn * 2 + tn + "- {M: [unclosed\n")
    block = tmp_path / "block.yaml"
    block.write_text(yaml.safe_dump([{"M": 8, "transA": "T", "transB": "N"}]))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "block_0000.log").write_text("\n".join(bf.BANNER) + "\n")
    (logs / "block_0001.log").write_text(
        "\n".join(
            bf.BANNER
            + problem_log(bf.Gemm(32, 32, 32, transB="T"))
            + problem_log(bf.Gemm(32, 32, 32))
        )
        + "\n"
    )
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    head = "m,n,k,batch_count,transA,transB,a_type,compute_type\n"
    (chunks / "chunk_0000.csv").write_text(head + "64,64,64,1,N,N,f8_r,f32_r\n")
    (chunks / "chunk_0001.csv").write_text(head + "64,64,64,1,T,N,f8_r,f32_r\n")
    datasets = [
        {"name": n, "path": str(p)}
        for n, p in (
            ("lines", lines),
            ("block", block),
            ("logs", logs),
            ("csv", chunks),
        )
    ]
    f8 = {f"hipblaslt.{t}_type": "f8_r" for t in "ab"}
    f8.update({"hipblaslt.scaleA": 3, "hipblaslt.scaleB": 3})
    cfg, _ = ro.load_config(
        make_config(tmp_path, **f8, **{"stage08.datasets": datasets})
    )
    assert ro._dataset_errors(cfg) == []
    monkeypatch.setattr(ro, "DATASET_SCAN", 2)
    errs = ro._dataset_errors(cfg)
    assert [e.split(":")[0] for e in errs] == ["stage08.datasets lines"]
    assert "none of the first 2 problems" in errs[0] and "transB=T" in errs[0]


def test_full_preflight_wants_a_built_audit_set_unless_the_run_builds_it(
    tmp_path, capsys
):
    from lib import audit as auditlib

    audit = tmp_path / "audit"
    over = {"audit.dir": str(audit), "audit.build": "off"}
    cfg, _ = ro.load_config(make_config(tmp_path, **over))
    paths = ro.derive_paths(cfg)
    (err,) = ro._audit_preflight(cfg, paths)
    assert f"audit.dir {audit} holds no audit set (audit_set.json)" in err
    assert "audit.build is off" in err
    assert ro.preflight(cfg, paths, "validate").count(err) == 0
    assert ro.preflight(cfg, paths, "full").count(err) == 1
    for build in ("before", "parallel"):
        over = {"audit.dir": str(audit), "audit.build": build}
        if build == "parallel":
            over["audit.devices"] = [5]
        built, _ = ro.load_config(make_config(tmp_path, **over))
        # the run builds the audit set itself (AuditBuild), so a missing one is
        # no error of its own
        assert ro._audit_preflight(built, paths) == []
        assert not any(
            "holds no audit set" in e for e in ro.preflight(built, paths, "full")
        )
    audit.mkdir()
    (audit / auditlib.MANIFEST).write_text(
        json.dumps({"schema": auditlib.SCHEMA_VERSION, "generations": []})
    )
    capsys.readouterr()
    assert ro._audit_preflight(cfg, paths) == []
    assert capsys.readouterr().err == ""


def test_disk_space_is_judged_by_the_largest_active_round(
    tmp_path, stubs, env, monkeypatch, capsys
):
    import collections

    usage = collections.namedtuple("usage", "total used free")

    def free(gb):
        monkeypatch.setattr(ro.shutil, "disk_usage", lambda p: usage(0, 0, gb * 1e9))

    def recorded(rd, gb):
        marker = rd / ro.ROUND_MARKER
        marker.write_text(json.dumps(dict(json.loads(marker.read_text()), bytes=gb)))

    cfg = make_config(tmp_path, **{"full.total_rounds": 4})
    run_dir = tmp_path / "run"
    assert ro.disk_problems(run_dir, 3) == ([], [])
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    marker = json.loads((run_dir / "round_0" / ro.ROUND_MARKER).read_text())
    assert 0 < marker["bytes"] <= ro.round_bytes(run_dir / "round_0")
    recorded(run_dir / "round_0", int(20e9))
    free(5)
    errs, warns = ro.disk_problems(run_dir, 3)
    assert errs == [] and "its latest complete round, round_0, takes 20.0 GB" in (
        warns[0]
    )
    assert "when it holds the seed benches" in warns[0]
    capsys.readouterr()
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert "round_0, takes 20.0 GB" in capsys.readouterr().err
    recorded(run_dir / "round_1", int(4e9))
    assert ro.disk_problems(run_dir, 1) == ([], [])
    (warn,) = ro.disk_problems(run_dir, 2)[1]
    assert warn.endswith(
        "its largest complete active round, round_1, takes 4.0 GB, "
        "so 2 more rounds may not fit"
    )

    free(3)
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 2
    assert calls(env)[n:] == []
    assert "round_2 not started: 3.0 GB free" in capsys.readouterr().err
    assert status_of(run_dir)["rounds"]["failed"]["round"] == 2
    loaded, _ = ro.load_config(cfg)
    errs = ro.preflight(loaded, ro.derive_paths(loaded), "full", run_dir)
    assert any(
        "round_1, takes 4.0 GB: free space before the next round" in e for e in errs
    )

    floor = make_config(tmp_path, **{"full.total_rounds": 4, "runtime.min_free_gb": 2})
    assert run(stubs, floor, run_dir, "--mode", "active") == 0
    free(1)
    (err,) = ro.disk_problems(run_dir, 1, 2)[0]
    assert "runtime.min_free_gb asks for 2 GB before each round" in err
    assert run(stubs, floor, run_dir, "--mode", "full") == 2
    off = make_config(tmp_path, **{"full.total_rounds": 4, "runtime.min_free_gb": 0})
    assert ro.disk_problems(run_dir, 1, 0) == ([], [])
    assert run(stubs, off, run_dir, "--mode", "full") == 0


def test_the_report_runs_to_its_limit_and_pairs_origami_from_the_config(
    tmp_path, stubs, env, monkeypatch
):
    seen = []
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if Path(str(cmd[1])).name != "make_report.py":
            return real_run(cmd, **kw)
        seen.append((cmd, kw.get("timeout")))
        if len(seen) == 2:
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        Path(cmd[3]).write_text("report")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ro.subprocess, "run", fake_run)
    monkeypatch.setenv("STUB_GATE", "1")
    cfg = make_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "report.origami_reference": True,
            "report.timeout_s": 120,
        },
    )
    run_dir = tmp_path / "run"
    argv = ["--config", str(cfg), "--run-dir", str(run_dir), "--skip-preflight"]
    argv += ["--stages-dir", str(stubs), "--mode", "full"]
    assert ro.main(argv) == 0
    ((cmd, timeout),) = seen
    assert cmd[-1] == "--origami-reference" and timeout == 120
    st = by_stage(calls(env))
    for key in ("stage04b", "stage06", "stage07", "stage08"):
        assert all("--origami-reference" in c["argv"] for c in st[key]), key
    report = status_of(run_dir)["report"]
    assert report["status"] == "written" and report["origami_reference"]
    assert ro.main(argv) == 0
    assert status_of(run_dir)["report"]["status"] == "timed_out"
    assert ro.cfg_get({}, "report.timeout_s") == 3600
    flag = ro.parse_args(argv[:2] + ["--mode", "full", "--origami-reference"])
    assert ro.origami_reference_on(flag, {})
    assert not ro.origami_reference_on(ro.parse_args(argv), {})
