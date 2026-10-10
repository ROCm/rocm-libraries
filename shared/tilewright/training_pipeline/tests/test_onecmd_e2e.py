# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One command on a production-shaped fake world (tests/onecmd_fakes.py):
the audit set, then `--mode full` through every round and stage (v3, the
planner with its seed bootstrap and bank, trained, re-planned and held-out
execution contexts, the near-winner re-reads, splits, the deploy gate, the
audit evaluations, stage08 and the report), the same command on the
finished run, and runs killed with SIGKILL that the same command resumes."""

import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import fake_hipblaslt as fh
import onecmd_fakes as of
import run_orchestrator as ro
from gpu_fakes import wait_for
from lib import attempts, pinning, run_layout

pytestmark = pytest.mark.e2e


@pytest.fixture
def world(tmp_path, monkeypatch):
    pytest.importorskip("tilewright")
    fh.unhide_devices(monkeypatch)
    for key, value in of.build_world(tmp_path).items():
        monkeypatch.setenv(key, value)
    return tmp_path


def _config(base, **overrides):
    cfg = of.config(base, **overrides)
    assert ro.validate_config(cfg)[0] == []
    path = base / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _argv(cfg, mode, run_dir):
    return ["--config", str(cfg), "--mode", mode, "--run-dir", str(run_dir)]


def _rows(enriched):
    for path in sorted(Path(enriched).glob("chunk_*.csv")):
        with path.open() as f:
            yield from csv.DictReader(f)


def _markers(run_dir):
    return sorted(Path(run_dir).glob("round_*/**/" + ro.STAGE_MARKER))


def test_one_command_runs_every_round_and_stage(world):
    cfg = _config(world)
    audit, run_dir = world / "audit", world / "run"
    assert ro.main(_argv(cfg, "audit", audit)) == 0
    assert ro.main(_argv(cfg, "full", run_dir)) == 0

    n_pool = len(of.pool_and_bank()[0])
    for r in range(3):
        rd = run_dir / f"round_{r}"
        assert (rd / ro.ROUND_MARKER).is_file()
        keys = ["stage01", "stage02", "stage03", "stage03b", "stage03c", "stage04"]
        keys += ["stage04b", "stage06"] if r else []
        for key in keys + ["stage05"]:
            assert ro.stage_complete(rd / key), f"{rd.name}/{key}"
        assert (rd / "stage03b" / "rebench_summary.json").is_file()
        rows = list(_rows(rd / "stage04"))
        assert {r_["cu_budget"] or "0" for r_ in rows} == {"0", "128"}
        assert any(int(r_["sol_idx_global"]) >= n_pool for r_ in rows)
        assert list(_rows(rd / "stage03c" / "holdout" / "stage04"))
        result = rd / ("stage06" if r else "stage05") / "audit.json"
        assert json.loads(result.read_text())["status"] == "ok"
        if r:
            assert (rd / "stage02" / "contexts" / "cu128_cot" / "plan.json").is_file()
            assert "splits" in json.loads((rd / "stage04b" / "splits.json").read_text())
            decisions = json.loads((rd / "stage04b" / "decisions.json").read_text())
            assert "192/default" in decisions["aggregate"]["context_holdout"]
    assert list(_rows(run_dir / "round_0" / "stage02" / "seed" / "stage04"))
    record = json.loads(
        (run_dir / "round_2" / "stage06" / "deploy_record.json").read_text()
    )
    assert record["gate"]["n_evaluated"] > 0
    stage08 = json.loads(
        (run_dir / "validate" / "stage08" / "summary.json").read_text()
    )
    assert [d["rc"] for d in stage08["datasets"]] == [0]
    assert (run_dir / "report.html").is_file()

    pin = pinning.read_run_pin(run_dir)["pin"]
    markers = _markers(run_dir)
    assert markers and all(
        json.loads(m.read_text())["pin"] == pinning.summary(pin) for m in markers
    )
    assert not (run_dir / attempts.RUNNING_FILE).exists()

    before = [m.stat().st_mtime_ns for m in markers]
    assert ro.main(_argv(cfg, "full", run_dir)) == 0
    assert [m.stat().st_mtime_ns for m in _markers(run_dir)] == before


def test_one_command_gives_every_bundle_its_whitelist_tier(world):
    """train.whitelist_tier_k through every round and a pooled_ft final
    retrain: each stage05 builds the tier of its bundle, each stage06
    records it against the pool it deploys for, and the same command on the
    finished run runs nothing again."""
    import torch

    from lib import whitelist_tier as wt

    recipe = {
        "recipe": "pooled_ft",
        "pooled_epochs": 1,
        "pooled_eval_points": 1,
        "pooled_n_estimators": 8,
        "pooled_threads": 1,
        "pooled_teacher_threads": 1,
        "fine_tune_epochs": 1,
    }
    cfg = _config(
        world,
        **{
            "train.whitelist_tier_k": 3,
            "full.final_retrain.enabled": True,
            "full.final_retrain.recipe": recipe,
        },
    )
    audit, run_dir = world / "audit", world / "run"
    assert ro.main(_argv(cfg, "audit", audit)) == 0
    assert ro.main(_argv(cfg, "full", run_dir)) == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    deployed = [run_dir / "round_1", run_dir / "round_2"]
    if json.loads((final / "decision.json").read_text())["accepted"]:
        deployed.append(final)
    for d in [run_dir / f"round_{r}" for r in range(3)] + [final]:
        s05 = d / "stage05"
        bundle = torch.load(
            str(s05 / "models.pt"), map_location="cpu", weights_only=True
        )
        tier = bundle[wt.RECORD]
        assert tier["k"] == 3 and tier["n_cells"] == len(bundle["models"]), d.name
        assert bundle["serving"]["whitelist_tier"] and (s05 / wt.TIER_FILE).is_file()
        options = json.loads((s05 / "model_options.json").read_text())
        assert options["whitelist_tier_k"] == 3
        if d in deployed:
            rec = json.loads((d / "stage06" / "deploy_record.json").read_text())
            assert rec["whitelist_tier"]["k"] == 3
            assert rec["whitelist_tier"]["served_pool"]["tier_keys_missing"] == 0
    trained = torch.load(str(final / "stage05" / "models.pt"), weights_only=True)
    assert trained["recipe"] == "pooled_ft"

    markers = _markers(run_dir) + sorted(final.glob("**/" + ro.STAGE_MARKER))
    before = [m.stat().st_mtime_ns for m in markers]
    assert ro.main(_argv(cfg, "full", run_dir)) == 0
    assert [m.stat().st_mtime_ns for m in markers] == before


@pytest.mark.parametrize(
    "where, planner",
    [
        ("round_0/stage02/seed/stage03/logs", True),
        ("round_0/stage01b/seed/stage03/logs", True),
        ("round_0/stage03/logs", False),
    ],
    ids=["seed_bench", "seed_bench_of_an_old_run", "bench"],
)
def test_a_run_killed_with_sigkill_resumes_with_the_same_command(world, where, planner):
    """SIGKILL the orchestrator while a bench runs: the seed bench is a
    sub-stage of the planner in its process group, the bench stage03 itself.
    Every process of the attempt stops within seconds, long before its work
    would end, and the same command resumes the round; in stage01b/ when
    the run was written before the planner became stage02."""
    overrides = {"full.total_rounds": 1, "audit.dir": None, "stage08.datasets": []}
    if not planner:
        overrides.update({"planner.enabled": False, "contexts.replan": False})
    cfg = _config(world, **overrides)
    run_dir = world / "run"
    old = run_layout.LEGACY_PLAN_STAGE in where
    if old:
        (run_dir / "round_0" / run_layout.LEGACY_PLAN_STAGE).mkdir(parents=True)
    out = (world / "killed.out").open("w")
    proc = subprocess.Popen(
        [sys.executable, str(Path(ro.__file__)), *_argv(cfg, "full", run_dir)],
        env=dict(os.environ, FAKE_SLEEP_PER_KERNEL="0.1"),
        stdout=out,
        stderr=subprocess.STDOUT,
    )
    try:
        assert wait_for(
            lambda: any((run_dir / where).glob("block_*.log"))
            or proc.poll() is not None,
            300,
        )
        assert proc.poll() is None, (world / "killed.out").read_text()[-3000:]
        (token, rec), *_ = attempts.Running(run_dir).records().items()
        assert attempts.processes_of(token)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert wait_for(lambda: not attempts.processes_of(token), 20), "orphans run on"
    assert ro.main(_argv(cfg, "full", run_dir)) == 0
    rd = run_dir / "round_0"
    assert (rd / ro.ROUND_MARKER).is_file()
    stage = rec["label"].split()[-1]
    assert list((rd / "stale").glob(f"*/{stage}"))
    assert not (run_dir / attempts.RUNNING_FILE).exists()
    if planner:
        planned = where.split("/")[1]
        assert stage == planned and ro.stage_complete(rd / planned)
        assert list(_rows(rd / planned / "seed" / "stage04"))
        assert (rd / "stage02").exists() != old
    assert (run_dir / run_layout.LAYOUT_FILE).exists() != old
