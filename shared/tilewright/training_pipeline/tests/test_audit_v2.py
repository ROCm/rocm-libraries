# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Audit protocol v2 (lib/audit_v2.py) through its entry points on the fake
bench: scripts/audit_v2.py plans the sets and benches them with workers
under scripts/gpu_run.py leases (fake GPUs), FAKE_SLOW_PLAN slows the
canary and some GEMMs in chosen processes, scripts/audit_eval.py scores v1
and v2 side by side and benches an increment (--v2-bench), and the report
labels each number's protocol."""

import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import audit_v2 as audit_v2_script
import gpu_fakes as gf
import test_audit as ta
from lib import audit as al
from lib import audit_v2 as v2
from lib.bench_yaml import line_solution_list, read_bench_lines, shape_of_line

PIPELINE = ta.PIPELINE
AUDIT_V2 = PIPELINE / "scripts" / "audit_v2.py"
SETTINGS = [
    "--set-max",
    "1",
    "--canary-attempts",
    "2",
    "--canary-backoff-s",
    "0",
    "--canary-reference-readings",
    "3",
]


def _run(cmd, env=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def v2world(tmp_path_factory):
    """The audit world of test_audit with a v2 measurement: generation 0
    planned with no model (sets of one kernel, the winner) and benched by
    one worker on fake GPU 1 under a slow plan; records of both rounds
    then; then an increment for both rounds' picks through audit_eval
    --v2-bench on GPUs 1 and 2, and the records after it."""
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("audit_v2")
    mp = pytest.MonkeyPatch()
    try:
        world = ta._build_world(root, mp)
        state = root / "gpu_state"
        gf.write_sysfs(state / "sys")
        mp.setenv("TILEWRIGHT_GPU_LEASE_DIR", str(state / "leases"))
        mp.setenv("TILEWRIGHT_SYSFS_ROOT", str(state / "sys"))
        mp.setenv("TILEWRIGHT_AMD_SMI", "")
        audit, cfg = world["audit"], world["cfg_path"]
        base = [sys.executable, str(AUDIT_V2)]
        audit_files = ta._files(audit)
        plan_out = _run(
            base + ["plan", "--audit-dir", str(audit), "--config", str(cfg)] + SETTINGS
        )
        d = v2.v2_dir(audit)
        m = v2.manifest(d)
        p0 = json.loads((d / "gen_000" / "plan.json").read_text())
        excl = p0["groups"][al.EXCLUSIVE]
        canary = m["canary"]
        last = [tuple(e["shape"]) for e in excl[-1] if e["shape"] != canary["shape"]]
        slow = [
            {
                "shape": canary["shape"],
                "alone": True,
                "factor": 1.5,
                "processes": [3, 8, 10],
            },
            {"shape": list(last[0]), "factor": 2.0, "processes": [0, 1]},
            {"shape": list(last[1]), "factor": 2.0, "processes": [0, 5]},
        ]
        plan_file = root / "slow.json"
        plan_file.write_text(json.dumps(slow))
        mp.setenv("FAKE_SLOW_PLAN", str(plan_file))
        gen0_out = _run(
            base
            + ["run", "--audit-dir", str(audit), "--config", str(cfg), "--gpus", "1"]
        )
        rounds = ta._rounds(world)
        args = [x for rd in rounds for x in ("--round-dir", str(rd))]
        before, before_log = ta.audit_eval(world, audit, root / "before.json", *args)
        after, after_log = ta.audit_eval(
            world, audit, root / "after.json", "--v2-bench", "1,2", *args
        )
        yield {
            **world,
            "v2": d,
            "plan_out": plan_out,
            "gen0_out": gen0_out,
            "excl_parts": excl,
            "slow": slow,
            "before": before,
            "before_log": before_log,
            "after": after,
            "after_log": after_log,
            "audit_files": audit_files,
        }
    finally:
        mp.undo()


def _result(d, gen, label, i):
    return json.loads((v2.part_path(d, gen, label, i) / v2.RESULT).read_text())


def _per_gemm(group):
    return {tuple(r[:4]): r[4] for r in group["per_gemm"]}


# ── sets ─────────────────────────────────────────────────────────────────────


def _g(shape, rows, skips=()):
    cands = [dict(sol_idx_global=s, us=u, is_skip=False) for s, u in rows]
    cands += [dict(sol_idx_global=s, us=u, is_skip=True) for s, u in skips]
    m, n, k, b = shape
    tested = [u for _, u in rows]
    return dict(m=m, n=n, k=k, batch_count=b, candidates=cands, winner_us=min(tested))


def test_candidates_are_the_kernels_within_the_tolerance_fastest_first():
    g = _g((64, 64, 64, 1), [(1, 10.0), (2, 10.9), (3, 11.2), (4, 10.5)], [(5, 10.2)])
    assert v2.candidates(g, 0.10, 20) == [1, 5, 4, 2]
    assert v2.candidates(g, 0.10, 2) == [1, 5]
    assert v2.candidates(g, 0.0, 20) == [1]


def test_the_canary_is_the_least_work_exclusive_gemm_and_its_winner():
    gemms = [
        _g((64, 64, 64, 1), [(1, 9.0), (2, 8.0)]),
        _g((3, 1, 3, 1), [(7, 4.0), (8, 4.1)], [(9, 3.0)]),
    ]
    assert v2.choose_canary(gemms) == ((3, 1, 3, 1), 7)


def test_medians_need_enough_readings_and_spread_is_relative_to_the_median():
    assert v2.medians({1: [1.0, 3.0, 2.0], 2: [5.0, None, None]}, 3) == {1: 2.0}
    assert v2.spread([2.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert v2.spread([]) is None


def test_an_increment_plans_only_gemms_whose_sets_lack_a_pick(tmp_path):
    a, b = (64, 64, 64, 1), (128, 64, 64, 1)
    gemms = {
        (al.EXCLUSIVE, a): _g(a, [(1, 10.0), (2, 10.5), (3, 20.0)]),
        (al.EXCLUSIVE, b): _g(b, [(1, 8.0), (2, 30.0), (3, 40.0)]),
    }
    lines = {
        s: f"- {{M: {s[0]}, N: {s[1]}, K: {s[2]}, batch_count: {s[3]}}}" for s in (a, b)
    }
    parts = {al.EXCLUSIVE: [[a, b]]}
    first = v2.Model("m1", "1" * 64, "x", {(al.EXCLUSIVE, a): 2, (al.EXCLUSIVE, b): 1})
    assert (
        v2.plan(tmp_path, gemms=gemms, lines=lines, parts=parts, new_models=[first])
        == 0
    )
    assert v2.current_sets(tmp_path) == {
        (al.EXCLUSIVE, a): [1, 2],
        (al.EXCLUSIVE, b): [1],
    }
    same = v2.Model("m2", "2" * 64, "x", {(al.EXCLUSIVE, a): 1, (al.EXCLUSIVE, b): 1})
    assert (
        v2.plan(tmp_path, gemms=gemms, lines=lines, parts=parts, new_models=[same])
        is None
    )
    new = v2.Model("m3", "3" * 64, "x", {(al.EXCLUSIVE, a): 3, (al.EXCLUSIVE, b): 1})
    assert (
        v2.plan(tmp_path, gemms=gemms, lines=lines, parts=parts, new_models=[new]) == 1
    )
    p1 = json.loads((tmp_path / "gen_001" / "plan.json").read_text())
    (entry,) = [e for part in p1["groups"][al.EXCLUSIVE] for e in part]
    assert tuple(entry["shape"]) == a and entry["kernels"] == [1, 2, 3]
    assert entry["picks"] == {"m1": 2, "m2": 1, "m3": 3}
    assert line_solution_list(entry["line"]) == [1, 2, 3]
    clash = v2.Model("m1", "9" * 64, "x", {})
    with pytest.raises(v2.V2Error, match="another sha256"):
        v2.plan(tmp_path, gemms=gemms, lines=lines, parts=parts, new_models=[clash])


# ── the bench ────────────────────────────────────────────────────────────────


def test_every_part_is_benched_in_shuffled_separate_repetitions(v2world):
    d = v2world["v2"]
    root = v2.part_path(d, 0, al.EXCLUSIVE, 0) / "a2"
    reps = [read_bench_lines(root / f"rep_{r}" / "shapes.yaml") for r in range(5)]
    orders = [[shape_of_line(ln) for ln in lines] for lines in reps]
    assert all(sorted(o) == sorted(orders[0]) for o in orders)
    assert len({tuple(o) for o in orders}) > 1
    for lines in reps:
        for ln in lines:
            assert "adaptive: true" in ln and "skip_slow_solution_ratio: 0" in ln
    for r in range(5):
        assert list((root / f"rep_{r}" / "stage03" / "logs").glob("block_*.log"))
    res = _result(d, 0, al.EXCLUSIVE, 0)
    assert len(res["bench"]["attempts"][-1]["reps"]) == 5
    for g in res["gemms"]:
        assert all(len(v) == 5 for v in g["readings"].values())


def test_workers_bench_one_gpu_each_under_gpu_run(v2world):
    d = v2world["v2"]
    assert (d / "canary" / "reference_1.json").is_file()
    assert (d / "canary" / "reference_2.json").is_file()
    gpus = {
        json.loads(p.read_text())["gpu"] for p in d.glob("gen_*/*/part_*/result.json")
    }
    assert gpus == {"1", "2"}
    assert (d / "worker_gpu1.log").is_file() and (d / "worker_gpu2.log").is_file()


def test_contexts_bench_through_the_cotenant_wrapper(v2world):
    d = v2world["v2"]
    m = v2.manifest(d)
    for label in ta.CONTEXTS:
        wrapper = Path(m["bench"]["wrappers"][label])
        assert wrapper.parent == d / "bench" / label
        assert any((wrapper.parent / "cotenant_logs").iterdir())
    now = [f for f in ta._files(v2world["audit"]) if not f[0].startswith("v2/")]
    assert now == v2world["audit_files"]


def test_the_canary_rebenches_a_part_and_flags_it_after_its_attempts(v2world):
    d = v2world["v2"]
    ref = json.loads((d / "canary" / "reference_1.json").read_text())
    assert len(ref["readings"]) == 3
    r0 = _result(d, 0, al.EXCLUSIVE, 0)
    a = r0["bench"]["attempts"]
    assert [x["canary_ok"] for x in a] == [False, True] and not r0["canary_flagged"]
    assert a[0]["canary_start"]["reading"] == pytest.approx(
        1.5 * ref["median"], rel=1e-4
    )
    assert len(a[0]["canary_start"]["timings"]) == 3
    r1 = _result(d, 0, al.EXCLUSIVE, 1)
    assert [x["canary_ok"] for x in r1["bench"]["attempts"]] == [False, False]
    assert r1["canary_flagged"]


def test_a_wide_spread_is_rebenched_once_then_flagged(v2world):
    d = v2world["v2"]
    res = _result(d, 0, al.EXCLUSIVE, len(v2world["excl_parts"]) - 1)
    by = {tuple(g["shape"]): g for g in res["gemms"]}
    calm, stuck = (tuple(r["shape"]) for r in v2world["slow"][1:])
    assert by[calm]["rebenched"] and not by[calm]["flagged"]
    assert by[stuck]["rebenched"] and by[stuck]["flagged"]
    assert res["n_spread_flagged"] == 1
    (k,) = by[stuck]["readings"]
    vals = by[stuck]["readings"][k]
    assert max(vals) == pytest.approx(2 * min(vals), rel=1e-4)
    assert by[stuck]["median"][k] == pytest.approx(min(vals))
    others = [g for s, g in by.items() if s not in (calm, stuck)]
    assert others and not any(g["rebenched"] for g in others)


# ── scores ───────────────────────────────────────────────────────────────────


def test_records_keep_v1_labelled_and_add_v2(v2world):
    for rec in v2world["before"]["models"] + v2world["after"]["models"]:
        assert rec["protocol"] == "v1" and rec["protocol_text"] == al.PROTOCOLS["v1"]
        assert rec["v2"]["protocol"] == "v2"
    v1_before = [r["groups"] for r in v2world["before"]["models"]]
    v1_after = [r["groups"] for r in v2world["after"]["models"]]
    assert v1_before == v1_after


def test_v2_is_null_for_picks_its_sets_lack_until_the_increment(v2world):
    v1 = v2world["before"]["models"][0]
    off = [r for r in v1["groups"][al.EXCLUSIVE]["per_gemm"] if r[4] < 1]
    assert off, "the fake world needs a pick that is not the winner"
    sub = v2world["before"]["models"][0]["v2"]
    gs = sub["group_status"][al.EXCLUSIVE]
    assert gs["status"] == "needs_increment" and gs["n_needs_increment"] > 0
    assert sub["groups"][al.EXCLUSIVE] is None
    assert "need an increment" in gs["reason"]
    assert "audit v2" in v2world["before_log"]


def test_the_increment_benches_whole_sets_and_completes_v2(v2world):
    d = v2world["v2"]
    m = v2.manifest(d)
    assert len(m["generations"]) == 2
    assert sorted(v2.models(d)) == sorted(
        f"{rd.parent.name}.{rd.name}" for rd in ta._rounds(v2world)
    )
    p1 = json.loads((d / "gen_001" / "plan.json").read_text())
    for label, parts in p1["groups"].items():
        for i, part in enumerate(parts):
            res = _result(d, 1, label, i)
            for e, g in zip(part, res["gemms"]):
                assert set(e["picks"].values()) <= set(g["kernels"])
                assert e["candidates"][0] in g["kernels"]
                assert all(len(v) == 5 for v in g["readings"].values())
    for rec in v2world["after"]["models"]:
        assert rec["v2"]["status"] == "ok", rec["v2"]["group_status"]
        assert rec["v2"]["generations"] == 2


def test_v2_medians_score_like_the_noise_free_exhaustive_pass(v2world):
    """The fake bench reads every kernel the same way each time, so the
    medians reproduce the exhaustive rows: every v2 score equals v1's."""
    for rec in v2world["after"]["models"]:
        for label, g1 in rec["groups"].items():
            g2 = rec["v2"]["groups"][label]
            p1, p2 = _per_gemm(g1), _per_gemm(g2)
            assert p1.keys() == p2.keys()
            for s in p1:
                assert p2[s] == pytest.approx(p1[s], rel=1e-9), (label, s)
            assert g2["classes"] == g1["classes"]
            for c, v in g2["per_cell"].items():
                assert v["ceiling_rules"] == pytest.approx(
                    g1["per_cell"][c]["ceiling_rules"]
                )
        gs = rec["v2"]["group_status"][al.EXCLUSIVE]
        assert {"n_canary_flagged", "n_spread_flagged"} <= set(gs)


def _unrecorded_after_a_rebuild(v2world, tmp_path):
    """Copies of the world's bench and audit set whose v2 measurement is as
    written before kernel identities were recorded (no records, no bench
    library), after a rebuild that renumbers the pool: (audit dir, config,
    the world with that config, audit_eval's --round-dir arguments)."""
    import yaml

    import bench_fakes as bf
    from lib import kernel_identity as ki
    from lib.dat import read_tensile_logic

    cfg = yaml.safe_load(v2world["cfg_path"].read_text())
    bench = shutil.copytree(cfg["runtime"]["rocm_libraries_root"], tmp_path / "bench")
    cfg["runtime"]["rocm_libraries_root"] = str(bench)
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    audit = shutil.copytree(v2world["audit"], tmp_path / "audit_set")
    d = v2.v2_dir(audit)
    for p in [*d.glob("gen_*/plan.json"), *(d / "models").glob("*.json")]:
        doc = json.loads(p.read_text())
        doc.pop(ki.RECORD_KEY)
        p.write_text(json.dumps(doc))
    m = v2.manifest(d)
    m["canary"].pop(ki.RECORD_KEY)
    m["bench"].pop("library")
    m["bench"]["config_yaml"] = str(cfg_path)
    (d / v2.MANIFEST).write_text(json.dumps(m))
    libdir = bench / "projects" / "hipblaslt" / "build" / "release" / "Tensile"
    logic = libdir / "library" / ta.ARCH / f"{bf.LIBRARY_STEM}.dat"
    sols = read_tensile_logic(logic)["solutions"]
    bf.write_logic(logic, [dict(s, index=int(s["index"]) + 50) for s in sols])
    args = [x for rd in ta._rounds(v2world) for x in ("--round-dir", str(rd))]
    return audit, cfg_path, dict(v2world, cfg_path=cfg_path), args


def test_v2_records_without_identities_follow_the_audit_digests_in_a_rebuild(
    v2world, tmp_path
):
    """The v2 measurement as written before kernel identities were recorded,
    after a rebuild that renumbers the pool: scoring reads its medians at
    their kernels' current indices by the audit set's kernel digests and
    gives the scores of before the rebuild; plan and bench refuse it, since
    its sets and bench lines name other kernels now (the bench reads the
    library from the plan's config, as the manifest records none)."""
    audit, cfg_path, world, args = _unrecorded_after_a_rebuild(v2world, tmp_path)
    rec, log = ta.audit_eval(world, audit, tmp_path / "out.json", *args)
    assert "by the audit set's kernel digests" in log
    for new, old in zip(rec["models"], v2world["after"]["models"]):
        assert new["v2"]["status"] == "ok", new["v2"]["group_status"]
        for label, g in old["v2"]["groups"].items():
            assert _per_gemm(new["v2"]["groups"][label]) == _per_gemm(g), label
    plan = subprocess.run(
        [sys.executable, str(AUDIT_V2), "plan", "--audit-dir", str(audit)]
        + ["--config", str(cfg_path), *args],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert plan.returncode == 2
    assert "records no kernel identities, and by the audit set's" in (
        plan.stdout + plan.stderr
    )
    bench_run = subprocess.run(
        [sys.executable, str(AUDIT_V2), "bench", "--audit-dir", str(audit)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert bench_run.returncode == 2
    out = bench_run.stdout + bench_run.stderr
    assert "its bench lines would time other kernels" in out
    assert str(tmp_path / "bench") in out


def test_v2_records_without_identities_score_by_the_generations_of_their_plan(
    v2world, tmp_path
):
    """The v2 measurement as written before kernel identities were recorded,
    then an audit increment, finished after its plans, that benched a kernel
    (since removed) under the index of the kernel most of its winners are,
    and a rebuild that renumbers the pool. By the generations finished
    before its plans that index still names its winner, and the scores are
    those of before. With its plans' times unknown the index names either
    kernel: every GEMM with a median under it needs a new measurement."""
    audit, _, world, args = _unrecorded_after_a_rebuild(v2world, tmp_path)
    d = v2.v2_dir(audit)
    winners = Counter(
        int(g["winner"])
        for p in d.glob(f"gen_*/*/part_*/{v2.RESULT}")
        for g in json.loads(p.read_text())["gemms"]
        if g["winner"] is not None
    )
    x = winners.most_common(1)[0][0]
    m = v2.manifest(d)
    planned = max(datetime.fromisoformat(g["created"]) for g in m["generations"])
    finished = (planned + timedelta(seconds=1)).isoformat(timespec="seconds")
    am = al.manifest(audit)
    am["generations"].append(
        {
            "index": len(am["generations"]),
            "stage04": "increments/inc_001/stage04",
            "kernels": {str(x): "f" * 16},
            "finished": finished,
        }
    )
    (audit / "increments" / "inc_001" / "stage04").mkdir(parents=True)
    (audit / al.MANIFEST).write_text(json.dumps(am))
    rec, _ = ta.audit_eval(world, audit, tmp_path / "out.json", *args)
    for new, old in zip(rec["models"], v2world["after"]["models"]):
        assert new["v2"]["status"] == "ok", new["v2"]["group_status"]
        for label, g in old["v2"]["groups"].items():
            assert _per_gemm(new["v2"]["groups"][label]) == _per_gemm(g), label

    for g in m["generations"]:
        del g["created"]
    (d / v2.MANIFEST).write_text(json.dumps(m))
    under_x = {k for k, mm in v2.measurements(d).items() if x in mm.median}
    rec, log = ta.audit_eval(world, audit, tmp_path / "unknown.json", *args)
    assert f"the audit generations do not tell which kernel solution {x} named" in log
    for model in rec["models"]:
        gs = model["v2"]["group_status"]
        assert sum(s["n_unplaced"] for s in gs.values()) == len(under_x)
        for label in {label for label, _ in under_x}:
            assert gs[label]["status"] == "needs_new_measurement"
            assert "need a new v2 measurement" in gs[label]["reason"]
            assert model["v2"]["groups"][label] is None


def test_paired_changes_are_given_per_protocol(v2world):
    after = v2world["after"]
    assert [c["protocol"] for c in after["changes"]] == ["v1"]
    assert [c["protocol"] for c in after["changes_v2"]] == ["v2"]
    c1 = after["changes"][0]["groups"][al.EXCLUSIVE]
    c2 = after["changes_v2"][0]["groups"][al.EXCLUSIVE]
    assert c2["cell_mean_ratio"] == pytest.approx(c1["cell_mean_ratio"])
    before = v2world["before"]["changes_v2"][0]["groups"][al.EXCLUSIVE]
    assert "reason" in before
    rec0, rec1 = after["models"]
    c, _, proto = al.paired_change(rec0, rec1)
    assert proto == "v2" and c["protocol"] == "v2"
    old = dict(rec0, v2=v2world["before"]["models"][0]["v2"])
    _, _, proto = al.paired_change(old, rec1)
    assert proto == "v1"
    assert "[v2 median of repeated joint benches]" in v2world["after_log"]


def test_status_lines_and_the_v2_only_option(v2world, tmp_path):
    line = v2.status_line(v2world["v2"])
    assert "planned GEMM benches done" in line and "flagged by the canary" in line
    rd = ta._rounds(v2world)[0]
    rec, _ = ta.audit_eval(
        v2world,
        v2world["audit"],
        tmp_path / "v1.json",
        "--round-dir",
        str(rd),
        "--protocol",
        "v1",
    )
    assert "v2" not in rec and rec["protocol"] == "v1"
    rec, _ = ta.audit_eval(
        v2world,
        v2world["audit"],
        tmp_path / "none.json",
        "--round-dir",
        str(rd),
        "--v2-dir",
        str(tmp_path / "nowhere"),
    )
    assert rec["v2"]["status"] == "missing" and rec["v2"]["groups"] is None


def test_the_v2_record_is_written_by_the_stages_too(v2world, tmp_path):
    rd = ta._rounds(v2world)[1]
    data = Path(
        json.loads((rd / "stage06" / "deploy_record.json").read_text())["staged_model"]
    ).read_bytes()
    spec = al.Spec(
        audit_dir=v2world["audit"],
        library_dir=v2world["libdir"],
        library_stem=ta.bf.LIBRARY_STEM,
        bootstrap=50,
        trained_rounds=al.rounds_until(rd, True),
    )
    from lib.hardware import DeviceHardware

    rec = al.evaluate_to(
        tmp_path / "audit.json",
        data,
        spec,
        v2world["cfg"],
        hardware=DeviceHardware(**v2world["cfg"]["hardware"]),
        what="round_1",
    )
    assert rec["v2"]["status"] == "ok"
    lines = al.summary_lines(rec)
    assert any(ln.startswith("audit v2 exclusive: cell mean") for ln in lines)
    assert any(ln.startswith("audit exclusive: cell mean") for ln in lines)
    assert math.isclose(
        rec["v2"]["headline"]["cell_mean"], rec["headline"]["cell_mean"], rel_tol=1e-9
    )


def test_the_planner_refuses_an_incomplete_audit(v2world, tmp_path):
    copy = ta._benching_copy(v2world, tmp_path / "audit")
    shutil.rmtree(copy / v2.DIRNAME)
    proc = subprocess.run(
        [sys.executable, str(AUDIT_V2), "plan", "--audit-dir", str(copy)]
        + ["--config", str(v2world["cfg_path"])],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert proc.returncode == 2 and "not complete" in proc.stdout + proc.stderr
    assert not (copy / v2.DIRNAME / v2.MANIFEST).exists()


def test_a_worker_keys_its_canary_reference_by_its_gpu(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "-1")
    assert audit_v2_script.canary_key("2") == "device2"
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    assert audit_v2_script.canary_key("0") == "3"
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    assert audit_v2_script.canary_key("5") == "5"
