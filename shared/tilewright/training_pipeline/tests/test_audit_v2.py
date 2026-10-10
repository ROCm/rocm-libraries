# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Audit protocol v2 (lib/audit_v2.py) and the sustained composite
(lib/audit_sustained.py) through their entry points on the fake bench:
scripts/audit_v2.py plans the sets and benches them with workers under
scripts/gpu_run.py leases (fake GPUs), then plans and benches the
power-capped exclusive GEMMs in long windows (FAKE_SUSTAINED: kernels read
slower there, some more than others), FAKE_SLOW_PLAN slows the canary and
some GEMMs in chosen processes, scripts/audit_eval.py scores v1, v2 and the
composite side by side and benches an increment (--v2-bench), and the
report labels each number's protocol."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import audit_v2 as audit_v2_script
import bench_fakes as bf
import gpu_fakes as gf
import test_audit as ta
from lib import audit as al
from lib import audit_sustained as asus
from lib import audit_v2 as v2
from lib import rebench as rb
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
# The sustained measurement's first plan: its canary as quick as v2's, and a
# compute-bound rate the fake world's large GEMMs reach.
SUSTAINED = [
    "--sustained-canary-attempts",
    "2",
    "--sustained-canary-backoff-s",
    "0",
    "--sustained-canary-reference-readings",
    "3",
    "--sustained-compute-bound-tflops",
    "20",
]
# How much slower a kernel reads in long self-soaked windows.
SOAK = {"default": 1.05, "8": 1.5, "9": 1.5}


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
        mp.setenv("FAKE_SUSTAINED", json.dumps(SOAK))
        gen0_out = _run(
            base
            + ["run", "--audit-dir", str(audit), "--config", str(cfg), "--gpus", "1"]
            + SUSTAINED
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


# ── the kernel order of every repetition, without a bench ────────────────────


def _listed_line(shape, kernels=()):
    m, n, k, b = shape
    return (
        f"- {{M: {m}, N: {n}, K: {k}, batch_count: {b}, adaptive: true, "
        f"solution_index_list: [{', '.join(str(x) for x in kernels)}]}}"
    )


def _runner(calls, us_of=lambda shape, sid, rank: 10.0 + sid):
    """A Runner that records the lines of every process and prints, for
    each, a bench log timing its kernels in list order (`us_of`)."""

    def run(run_dir, lines, binary):
        calls.append(list(lines))
        out = list(bf.BANNER)
        for ln in lines:
            shape = shape_of_line(ln)
            g = bf.Gemm(*shape)
            ks = line_solution_list(ln)
            out += bf.preamble(len(ks))
            best = None
            for rank, sid in enumerate(ks):
                us = us_of(shape, sid, rank)
                out += bf.measured_lines(rank, g, sid, us)
                if best is None or us < best[2]:
                    best = (rank, sid, us)
            if len(ks) > 1:
                out += bf.winner(best[0], g, best[1], best[2])
        return v2.Run(0, [rb.read_log_lines("\n".join(out) + "\n")], 0.0)

    return run


class _Steady(v2.Bench):
    """A worker whose canary always reads its reference."""

    def reference(self):
        return 1.0

    def canary_reading(self, run_dir):
        return {"timings": [1.0], "reading": 1.0, "rc": 0, "taken": "now"}


def _bench(tmp_path, settings, calls, us_of=None):
    m = {"settings": settings, "bench": {"binary": "hipblaslt-bench"}}
    runner = _runner(calls) if us_of is None else _runner(calls, us_of)
    return _Steady(tmp_path / "v2", m, runner, "0")


A, B = (256, 256, 256, 1), (512, 128, 64, 1)
ENTRIES = [
    {"shape": list(A), "kernels": [11, 12, 13, 14, 15], "picks": {"m": 15}},
    {"shape": list(B), "kernels": [21, 22], "picks": {"m": 22}},
]
for _e in ENTRIES:
    _e["line"] = _listed_line(tuple(_e["shape"]), _e["kernels"])


def test_v2_shuffles_each_set_whole_per_repetition_behind_a_sacrificial_copy(
    tmp_path,
):
    calls = []
    st = {**v2.DEFAULTS, "repetitions": 6, "canary_attempts": 1}

    def cold_first(shape, sid, rank):
        return 10.0 + sid + (100.0 if rank == 0 else 0.0)

    w = _bench(tmp_path, st, calls, cold_first)
    assert w.st["kernel_order"] == "shuffled" and w.st["sacrificial_kernel"]
    assert w.st["warmup_line"] is True
    out = w.bench_round(tmp_path / "part", al.EXCLUSIVE, ENTRIES, "g0/exclusive/0")
    assert len(calls) == 6
    for e in ENTRIES:
        s = tuple(e["shape"])
        orders = out["orders"][s]
        assert len(orders) == 6
        assert all(sorted(o) == sorted(e["kernels"]) for o in orders)
        for r, o in enumerate(orders):
            (ln,) = [x for x in calls[r] if shape_of_line(x) == s]
            assert line_solution_list(ln) == [o[0], *o]
            assert o == v2.rep_order(e["kernels"], st, r, ("g0/exclusive/0", 1, *s))
        # each reading is the kernel's own: the copy's slow first call is set aside
        for k, v in out["readings"][s].items():
            assert v == [10.0 + k] * 6
    a = out["orders"][A]
    assert len({tuple(o) for o in a}) > 1
    assert {o.index(15) for o in a} != {4}, "the pick is not always timed last"
    reps = out["attempts"][-1]["reps"]
    for r, rep in enumerate(reps):
        assert rep["gemm_order"] == [list(shape_of_line(x)) for x in calls[r]]


def test_a_set_planned_before_the_order_settings_keeps_its_rotation(tmp_path):
    """A measurement whose settings predate `kernel_order`,
    `sacrificial_kernel` and `warmup_line` is benched as it was: each set as
    planned, rotated by the repetition, no copy, no warm-up line."""
    calls = []
    old = {k: v for k, v in v2.DEFAULTS.items() if k not in v2.PREDATING}
    w = _bench(tmp_path, {**old, "repetitions": 3, "canary_attempts": 1}, calls)
    assert w.st["kernel_order"] == "rotated" and not w.st["sacrificial_kernel"]
    assert w.st["warmup_line"] is False
    out = w.bench_round(tmp_path / "part", al.EXCLUSIVE, ENTRIES, "g0/exclusive/0")
    for r in range(3):
        for e in ENTRIES:
            s = tuple(e["shape"])
            (ln,) = [x for x in calls[r] if shape_of_line(x) == s]
            assert ln == rb.rebench_line(e["line"], rb.rotate(e["kernels"], r))
            assert out["orders"][s][r] == rb.rotate(e["kernels"], r)


def test_a_part_records_the_order_of_the_readings_it_keeps(tmp_path):
    """result.json holds every GEMM's kernel order per repetition; a GEMM
    benched again for its spread keeps the orders of the readings that
    replaced its own."""
    calls = []
    st = {**v2.DEFAULTS, "repetitions": 3, "canary_attempts": 1}

    def us_of(shape, sid, rank):
        # GEMM B's pick reads far slower in the main round's first process
        return (
            10.0 + sid + (60.0 if shape == B and sid == 22 and len(calls) == 1 else 0)
        )

    w = _bench(tmp_path, st, calls, us_of)
    w.d.mkdir(parents=True)
    res = w.bench_part(0, al.EXCLUSIVE, 0, ENTRIES)
    by = {tuple(g["shape"]): g for g in res["gemms"]}
    assert res["settings"]["kernel_order"] == "shuffled"
    assert res["settings"]["sacrificial_kernel"] is True
    assert by[B]["rebenched"] and not by[A]["rebenched"]
    spread = res["spread_bench"]["attempts"][-1]["reps"]
    assert len(spread) == 3 and "orders" not in res["bench"]
    for g in by.values():
        assert len(g["orders"]) == 3
        assert all(sorted(o) == sorted(g["kernels"]) for o in g["orders"])
    stored = json.loads((v2.part_path(w.d, 0, al.EXCLUSIVE, 0) / v2.RESULT).read_text())
    assert stored["gemms"] == res["gemms"]


def test_the_runner_gives_stage03_the_warm_up_line_of_the_measurement(
    tmp_path, monkeypatch
):
    """stage03 runs with the warm-up line the measurement's settings say,
    its kernels one of each library file of the bench setup; without a
    setting, as the config says."""
    seen = []

    def fake_run(cmd, **kw):
        seen.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(v2.subprocess, "run", fake_run)
    library = {"dir": "/lib", "stems": ["Pool", "Bank"]}
    for flag, want in ((True, "--warmup-line"), (False, "--no-warmup-line")):
        run = v2.stage03_runner(None, warmup_line=flag, library=library)
        run(tmp_path / "r", ["- {M: 1, N: 1, K: 1}"], Path("bench"))
        cmd = seen[-1]
        assert want in cmd
        stems = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--identity-stem"]
        if flag:
            assert cmd[cmd.index("--identity-dir") + 1] == "/lib"
            assert stems == ["Pool", "Bank"]
        else:
            assert "--identity-dir" not in cmd and stems == []
    v2.stage03_runner(None)(tmp_path / "r", ["- {M: 1, N: 1, K: 1}"], Path("bench"))
    assert not {"--warmup-line", "--no-warmup-line"} & set(seen[-1])


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


def test_every_v2_bench_process_starts_with_the_warm_up_line(v2world):
    """A measurement planned with the warm-up line on gives every stage03 of
    its parts and canary readings the line, over a kernel of each library
    file of its bench setup, and their platform.json records it."""
    d = v2world["v2"]
    m = v2.manifest(d)
    assert m["settings"]["warmup_line"] is True
    stems = m["bench"]["library"]["stems"]
    records = [
        json.loads(p.read_text())["warmup_line"]
        for p in d.rglob("stage03/platform.json")
    ]
    assert len(records) > 10
    for rec in records:
        assert rec["enabled"] and rec["shape"] == [512, 512, 512, 1]
        assert rec["kernels"] and {k["file"] for k in rec["kernels"]} <= set(stems)
    assert any(p.name.startswith("warmup_") for p in d.rglob("stage03/logs/*.log"))


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


def test_an_increment_times_each_set_shuffled_behind_a_sacrificial_copy(v2world):
    """Each repetition times a GEMM's whole set in a shuffle of its own, led
    by a copy of its first kernel whose row the readers set aside, and the
    part records the order of every repetition it keeps."""
    d = v2world["v2"]
    p1 = json.loads((d / "gen_001" / "plan.json").read_text())
    moved = 0
    for label, parts in p1["groups"].items():
        for i in range(len(parts)):
            res = _result(d, 1, label, i)
            assert res["settings"]["kernel_order"] == "shuffled"
            assert res["settings"]["sacrificial_kernel"] is True
            root = v2.part_path(d, 1, label, i) / f"a{len(res['bench']['attempts'])}"
            reps = [
                {shape_of_line(ln): ln for ln in read_bench_lines(p)}
                for p in sorted(root.glob("rep_*/shapes.yaml"))
            ]
            for g in res["gemms"]:
                assert all(sorted(o) == sorted(g["kernels"]) for o in g["orders"])
                if not g["rebenched"]:
                    for o, lines in zip(g["orders"], reps):
                        ln = lines[tuple(g["shape"])]
                        assert line_solution_list(ln) == [o[0], *o]
                if len(g["kernels"]) > 1:
                    moved += len({tuple(o) for o in g["orders"]}) > 1
    assert moved


def _picks(v2world, rec):
    """{shape: pick} of the exclusive group of the model `rec` evaluated."""
    (name,) = [n for n in v2.models(v2world["v2"]) if n.endswith("." + rec["what"])]
    return {
        s: sid
        for (label, s), sid in v2.picks_of(v2.models(v2world["v2"])[name]).items()
        if label == al.EXCLUSIVE
    }


def _window(line, key):
    m = re.search(rf"(?<![\w]){key}:\s*([0-9.]+)", line)
    return float(m.group(1)) if m else None


def test_the_power_capped_gemms_are_benched_again_in_long_windows(v2world):
    sd = asus.sustained_dir(v2world["audit"])
    m = v2.manifest(sd)
    assert m["regime"] == "sustained" and len(m["generations"]) == 2
    st = v2.settings_of(m)
    assert st["compute_bound_tflops"] == 20.0 and "given" in m["compute_bound"]
    entries = asus.current_entries(sd)
    assert entries, "the fake world needs power-capped GEMMs"
    strata = set()
    for (label, s), mm in v2.measurements(v2world["v2"]).items():
        if label == al.EXCLUSIVE and asus.stratum(s, mm.winner_us, st):
            assert s in entries
    for s, e in entries.items():
        strata.add(e["stratum"])
        assert (al.EXCLUSIVE, s) in v2.measurements(v2world["v2"])
    assert "compute_bound" in strata and strata & {"long", "very_long"}
    for gen, g in enumerate(m["generations"]):
        for i in range(g["n_parts"]):
            res = _result(sd, gen, al.EXCLUSIVE, i)
            assert res["settings"]["repetitions"] == 3
            assert res["settings"]["sacrificial_kernel"] is False
            root = v2.part_path(sd, gen, al.EXCLUSIVE, i)
            root = root / f"a{len(res['bench']['attempts'])}"
            for r in range(3):
                lines = read_bench_lines(root / f"rep_{r}" / "shapes.yaml")
                assert lines
                for ln in lines:
                    assert _window(ln, "warmup_time") == 250.0
                    assert _window(ln, "measure_time") == 300.0
                    assert _window(ln, "max_measure_time") == 400.0
                    (gm,) = [
                        x
                        for x in res["gemms"]
                        if tuple(x["shape"]) == shape_of_line(ln)
                    ]
                    if not gm["rebenched"]:
                        assert line_solution_list(ln) == gm["orders"][r]
    assert (sd / "worker_gpu1.log").is_file()
    status = _run(
        [sys.executable, str(AUDIT_V2), "status", "--audit-dir", str(v2world["audit"])]
    )
    assert "v2: " in status and "sustained: " in status


def test_the_composite_scores_power_capped_gemms_on_their_long_windows(v2world):
    """The fake bench reads every kernel the same way each time and SOAK
    times slower in long windows: each sustained median is the GEMM's v2
    median times its kernel's factor, a power-capped GEMM's score is the
    fastest of them over its pick's, and every other score is v2's."""
    sd = asus.sustained_dir(v2world["audit"])
    entries = asus.current_entries(sd)
    short, soaked = v2.measurements(v2world["v2"]), v2.measurements(sd)
    for key, ms in soaked.items():
        for k, us in ms.median.items():
            factor = SOAK.get(str(k), SOAK["default"])
            assert us == pytest.approx(short[key].median[k] * factor, rel=1e-4)
    moved = 0
    for rec in v2world["after"]["models"]:
        assert rec["official"] == "sustained"
        sus = rec["sustained"]
        assert sus["status"] == "ok", sus["group_status"]
        picks = _picks(v2world, rec)
        comp = _per_gemm(sus["groups"][al.EXCLUSIVE])
        flat = _per_gemm(rec["v2"]["groups"][al.EXCLUSIVE])
        assert comp.keys() == flat.keys()
        for s in comp:
            if s not in entries:
                assert comp[s] == pytest.approx(flat[s], rel=1e-9)
                continue
            med = soaked[(al.EXCLUSIVE, s)].median
            want = min(1.0, min(med.values()) / med[picks[s]])
            assert comp[s] == pytest.approx(want, rel=1e-9)
            moved += abs(comp[s] - flat[s]) > 1e-6
        for label in ta.CONTEXTS:
            assert sus["groups"][label] == rec["v2"]["groups"][label]
        gs = sus["group_status"][al.EXCLUSIVE]
        assert 0 < gs["n_power_capped"] == sum(gs["strata"].values())
        assert gs["n_power_capped"] == len(set(entries) & set(comp))
        assert sus["headline"]["cell_mean"] == sus["groups"][al.EXCLUSIVE]["cell_mean"]
        g, proto = al.group_view(rec, al.EXCLUSIVE)
        assert proto == "sustained" and g == sus["groups"][al.EXCLUSIVE]
    assert moved, "the long windows changed no score of the fake world"
    log = v2world["after_log"]
    assert "audit sustained exclusive: cell mean" in log
    assert "official exclusive number" in log


def test_the_composite_is_null_for_picks_its_sets_lack_until_the_increment(v2world):
    sub = v2world["before"]["models"][0]["sustained"]
    gs = sub["group_status"][al.EXCLUSIVE]
    assert gs["status"] == "needs_increment" and sub["groups"][al.EXCLUSIVE] is None
    assert gs["n_needs_increment"] > 0 and gs["n_power_capped"] > 0


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
    assert proto == "sustained" and c["protocol"] == "sustained"
    c, _, proto = al.paired_change(rec0, rec1, official="v2")
    assert proto == "v2" and c["protocol"] == "v2"
    for label in ta.CONTEXTS:
        assert al.paired_change(rec0, rec1, label)[2] == "v2"
    assert [c["protocol"] for c in after["changes_sustained"]] == ["sustained"]
    assert list(after["changes_sustained"][0]["groups"]) == [al.EXCLUSIVE]
    was = v2world["before"]["models"][0]
    old = dict(rec0, v2=was["v2"], sustained=was["sustained"])
    _, _, proto = al.paired_change(old, rec1)
    assert proto == "v1"
    assert "[v2 median of repeated joint benches]" in v2world["after_log"]
    assert f"[{al.PROTOCOLS['sustained']}]" in v2world["after_log"]


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


def test_run_gives_its_workers_one_guard_session_and_process_tree(
    tmp_path, monkeypatch
):
    from lib import gpu_health, host_load

    monkeypatch.delenv(host_load.RUN_ROOT_VAR, raising=False)
    env = audit_v2_script.worker_env(tmp_path / "v2", {"KEEP": "1"})
    assert env[gpu_health.GUARD_DIR_VAR] == str(tmp_path / "v2" / gpu_health.GUARD_DIR)
    assert env[gpu_health.SESSION_VAR].startswith("audit_v2_")
    assert env[host_load.RUN_ROOT_VAR] == str(os.getpid()) and env["KEEP"] == "1"
    named = {
        gpu_health.GUARD_DIR_VAR: str(tmp_path / "run" / "guard"),
        gpu_health.SESSION_VAR: "20261010_full",
        host_load.RUN_ROOT_VAR: str(os.getppid()),
    }
    assert audit_v2_script.worker_env(tmp_path / "v2", named) == named


# ── hung lines ───────────────────────────────────────────────────────────────


def test_a_line_that_hangs_again_is_quarantined_for_v2_alone(tmp_path, monkeypatch):
    """The v2 runner benches a hung line once more in stage03's retry pass,
    then quarantines it under audit_v2 (`Run.hung`); a later run of that
    protocol leaves it out (`Run.left_out`), the canary's runner neither."""
    from lib import gpu_health

    bench = bf.write_fake_bench(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"hang": ["x".join(map(str, A))], "state_dir": str(state)})
    )
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))
    monkeypatch.setenv(gpu_health.GUARD_DIR_VAR, str(tmp_path / "guard"))
    quick = dict(stall_timeout_s=1.0, startup_grace_s=1.0)
    lines = [e["line"] for e in ENTRIES]
    run = v2.stage03_runner(None, **quick)
    got = run(tmp_path / "r0", lines, bench)
    assert (got.hung, got.left_out) == ([A], [])
    assert set(v2.readings_of_logs(got.logs)) == {B}
    again = run(tmp_path / "r1", lines, bench)
    assert (again.hung, again.left_out) == ([], [A])
    guard = gpu_health.Guard(tmp_path / "guard", "x")
    assert guard.quarantined_shapes(v2.hung_scope()) == {A}
    assert guard.quarantined_shapes() == set()
    assert guard.quarantined_shapes(v2.hung_scope("sustained")) == set()
    canary = v2.stage03_runner(
        None, **quick, scope=gpu_health.NO_SCOPE, retry_hung=False
    )
    seen = canary(tmp_path / "c0", lines, bench)
    assert (seen.hung, seen.left_out) == ([], [])
    assert guard.quarantined_shapes(v2.hung_scope()) == {A}


def _dropping(calls, shape, at):
    """A Runner as `_runner`'s whose run `at` (0-based) hangs on `shape`'s
    line again after its retry, and whose later runs leave it out."""
    base = _runner(calls)

    def run(run_dir, lines, binary):
        n = len(calls)
        if n >= at:
            lines = [ln for ln in lines if shape_of_line(ln) != shape]
        out = base(run_dir, lines, binary)
        if n == at:
            out.hung = [shape]
        elif n > at:
            out.left_out = [shape]
        return out

    return run


def test_a_part_drops_a_gemm_whose_line_hung_and_records_why(tmp_path):
    calls = []
    st = {**v2.DEFAULTS, "repetitions": 3, "canary_attempts": 1}
    m = {"settings": st, "bench": {"binary": "hipblaslt-bench"}}
    said = []
    w = _Steady(tmp_path / "v2", m, _dropping(calls, A, 1), "0", log=said.append)
    w.d.mkdir(parents=True)
    res = w.bench_part(0, al.EXCLUSIVE, 0, ENTRIES)
    by = {tuple(g["shape"]): g for g in res["gemms"]}
    assert by[A]["dropped"] == {"why": "hung", "attempt": 1, "rep": 1}
    assert "dropped" not in by[B] and res["n_dropped"] == 1
    assert not by[A]["rebenched"] and by[A]["median"] == {}
    reps = res["bench"]["attempts"][-1]["reps"]
    assert [r.get("dropped") for r in reps] == [None, [list(A)], [list(A)]]
    assert any("dropped in repetition 1: its line hung" in s for s in said)
    stored = json.loads((v2.part_path(w.d, 0, al.EXCLUSIVE, 0) / v2.RESULT).read_text())
    assert stored["gemms"] == res["gemms"] and "dropped" not in stored["bench"]


def _result_of(d, gen, rows):
    """Write generation `gen`'s exclusive part 0 result: {shape: (median,
    dropped)}."""
    res = {
        "canary_flagged": False,
        "gemms": [
            {
                "shape": list(s),
                "kernels": sorted(med) or [1],
                "median": {str(k): v for k, v in med.items()},
                "flagged": False,
                **({"dropped": why} if why else {}),
            }
            for s, (med, why) in rows.items()
        ],
    }
    path = v2.part_path(d, gen, al.EXCLUSIVE, 0) / v2.RESULT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))


def test_a_dropped_gemm_keeps_its_earlier_measurement_and_says_why(tmp_path):
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
    hung = {"why": "hung", "attempt": 1, "rep": 2}
    _result_of(tmp_path, 0, {a: ({1: 10.0, 2: 10.5}, None), b: ({}, hung)})
    new = v2.Model("m3", "3" * 64, "x", {(al.EXCLUSIVE, a): 3, (al.EXCLUSIVE, b): 1})
    assert v2.plan(tmp_path, gemms=gemms, lines=lines, parts=parts, new_models=[new])
    _result_of(tmp_path, 1, {a: ({}, hung)})
    got = v2.measurements(tmp_path)
    ma, mb = got[(al.EXCLUSIVE, a)], got[(al.EXCLUSIVE, b)]
    assert ma.median == {1: 10.0, 2: 10.5} and ma.generation == 0
    assert ma.dropped == {**hung, "generation": 1}
    assert mb.median == {} and mb.dropped == {**hung, "generation": 0}
    assert v2.progress(tmp_path)["gemms_dropped"] == 2
    assert "2 GEMM bench(es) dropped" in v2.status_line(tmp_path)
    t = al._counts()
    al._missed(t, (al.EXCLUSIVE, b), mb, {})
    al._missed(t, (al.EXCLUSIVE, a), ma, {(al.EXCLUSIVE, a): 1})
    assert (t["dropped"], t["benching"], t["needs_increment"]) == (1, 1, 0)
    text = al._with_dropped("complete", {al.EXCLUSIVE: t}, "v2 bench")
    assert text == (
        "complete; 1 GEMM(s) left out: their v2 bench hung, again when benched "
        "once more"
    )
    assert al._with_dropped("complete", {al.EXCLUSIVE: al._counts()}, "x") == (
        "complete"
    )


def test_a_run_of_quarantined_lines_alone_leaves_them_out(tmp_path, monkeypatch):
    """stage03 benches nothing when the run quarantined every line it is
    given, and its summary still names them, so that the part drops their
    GEMMs rather than measuring them as empty."""
    from lib import gpu_health

    bench = bf.write_fake_bench(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"hang": ["x".join(map(str, A))], "state_dir": str(state)})
    )
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))
    monkeypatch.setenv(gpu_health.GUARD_DIR_VAR, str(tmp_path / "guard"))
    run = v2.stage03_runner(None, stall_timeout_s=1.0, startup_grace_s=1.0)
    assert run(tmp_path / "r0", [e["line"] for e in ENTRIES], bench).hung == [A]
    again = run(tmp_path / "r1", [ENTRIES[0]["line"]], bench)
    assert (again.rc, again.logs, again.hung, again.left_out) == (1, [], [], [A])
    assert not again.unmeasured
    assert again.problems == [
        f"every line of {tmp_path / 'r1' / 'shapes.yaml'} is quarantined in the run "
        f"({v2.hung_scope()})"
    ]


def test_a_part_of_quarantined_gemms_alone_drops_them(tmp_path):
    st = {**v2.DEFAULTS, "repetitions": 2, "canary_attempts": 1}
    m = {"settings": st, "bench": {"binary": "hipblaslt-bench"}}

    def left_out(run_dir, lines, binary):
        return v2.Run(1, [], 0.0, left_out=[A], problems=["every line ..."])

    w = _Steady(tmp_path / "v2", m, left_out, "0")
    w.d.mkdir(parents=True)
    (g,) = w.bench_part(0, al.EXCLUSIVE, 0, ENTRIES[:1])["gemms"]
    assert g["dropped"] == {"why": "quarantined", "attempt": 1, "rep": 0}


# ── unmeasured parts and held GPUs ───────────────────────────────────────────


def _benching_nothing(calls):
    """A Runner whose stage03 runs end without a bench, as on a GPU the run
    holds past bench.health.max_wait_s."""

    def run(run_dir, lines, binary):
        calls.append(list(lines))
        return v2.Run(2, [], 0.0, problems=["every GPU stopped taking blocks"])

    return run


def test_a_part_whose_runs_bench_nothing_is_left_unwritten(tmp_path):
    assert v2.Run(2, [], 0.0).unmeasured
    assert not v2.Run(0, [], 0.0).unmeasured
    assert not v2.Run(2, [["x"]], 0.0).unmeasured
    assert not v2.Run(1, [], 0.0, hung=[A]).unmeasured
    assert not v2.Run(1, [], 0.0, left_out=[A]).unmeasured
    calls = []
    st = {**v2.DEFAULTS, "repetitions": 3, "canary_attempts": 2}
    m = {"settings": st, "bench": {"binary": "hipblaslt-bench"}}
    w = _Steady(tmp_path / "v2", m, _benching_nothing(calls), "0")
    w.d.mkdir(parents=True)
    with pytest.raises(v2.Unmeasured, match=r"nothing \(exit 2\): every GPU stopped"):
        w.bench_part(0, al.EXCLUSIVE, 0, ENTRIES)
    assert len(calls) == 1
    assert not (v2.part_path(w.d, 0, al.EXCLUSIVE, 0) / v2.RESULT).exists()
    m["canary"] = {"line": ENTRIES[0]["line"], "shape": list(A), "kernel": 11}
    plain = v2.Bench(
        tmp_path / "v2", m, _runner([]), "0", canary_runner=_benching_nothing(calls)
    )
    with pytest.raises(v2.Unmeasured):
        plain.reference()
    assert len(calls) == 2
    assert not list((tmp_path / "v2" / "canary").glob("reference_*"))


def _planned(d):
    """A v2 measurement in `d` planned with two exclusive parts of one GEMM
    each; its manifest."""
    a, b = (64, 64, 64, 1), (128, 64, 64, 1)
    gemms = {(al.EXCLUSIVE, s): _g(s, [(1, 10.0), (2, 10.5)]) for s in (a, b)}
    lines = {
        s: f"- {{M: {s[0]}, N: {s[1]}, K: {s[2]}, batch_count: {s[3]}}}" for s in (a, b)
    }
    model = v2.Model("m1", "1" * 64, "x", {(al.EXCLUSIVE, a): 2, (al.EXCLUSIVE, b): 1})
    st = {"repetitions": 3, "canary_attempts": 1}
    parts = {al.EXCLUSIVE: [[a], [b]]}
    bench = {"binary": "hipblaslt-bench"}
    got = v2.plan(
        d,
        gemms=gemms,
        lines=lines,
        parts=parts,
        new_models=[model],
        settings=st,
        bench=bench,
    )
    assert got == 0
    return v2.manifest(d)


def test_a_worker_takes_no_part_while_the_run_holds_its_gpu(tmp_path):
    d = tmp_path / "v2"
    m = _planned(d)
    calls, said, slept = [], [], []
    holds = iter(["GPU 1: power management is off"] * 2)
    w = _Steady(
        d,
        m,
        _runner(calls),
        "1",
        log=said.append,
        sleep=slept.append,
        held=lambda: next(holds, None),
        poll_s=7.0,
    )
    assert v2.work(w) == 2 and not v2.pending(d)
    assert slept == [7.0, 7.0] and len(calls) == 6
    noted = [s for s in said if "the run holds this worker's GPU" in s]
    assert noted == [
        "the run holds this worker's GPU (GPU 1: power management is off): it takes "
        "no part while it does; 2 GEMM bench(es) left"
    ]


def test_a_held_worker_returns_once_no_part_is_left_or_its_wait_is_over(tmp_path):
    d = tmp_path / "v2"
    m = _planned(d)
    calls = []
    other = _Steady(d, m, _runner(calls), "0")
    w = _Steady(
        d, m, _runner([]), "1", held=lambda: "held", sleep=lambda _s: v2.work(other)
    )
    assert v2.work(w) == 0 and not v2.pending(d) and len(calls) == 6
    d2 = tmp_path / "again"
    said = []
    w = _Steady(
        d2,
        _planned(d2),
        _runner([]),
        "1",
        log=said.append,
        held=lambda: "held",
        sleep=lambda _s: time.sleep(0.05),
        max_wait_s=0.2,
    )
    assert v2.work(w) == 0 and len(v2.pending(d2)) == 2
    assert any(
        "for longer than bench.health.max_wait_s 0.2 s (held)" in s for s in said
    )


def test_a_part_left_when_the_run_holds_the_gpu_is_benched_once_it_does_not(
    tmp_path,
):
    d = tmp_path / "v2"
    m = _planned(d)
    calls, said = [], []
    real = _runner(calls)
    state = {"held": None, "failed": False}

    def runner(run_dir, lines, binary):
        if not state["failed"]:
            state.update(failed=True, held="GPU 1: power cap 50 W")
            return v2.Run(2, [], 0.0, problems=["every GPU stopped taking blocks"])
        return real(run_dir, lines, binary)

    w = _Steady(
        d,
        m,
        runner,
        "1",
        log=said.append,
        held=lambda: state["held"],
        sleep=lambda _s: state.update(held=None),
    )
    assert v2.work(w) == 2 and not v2.pending(d) and len(calls) == 6
    assert any(
        f"gen 0 {al.EXCLUSIVE} part 0 is left unwritten: the stage03 run in" in s
        for s in said
    )


def test_a_worker_whose_gpu_cannot_bench_stops_and_leaves_its_part(tmp_path):
    d = tmp_path / "v2"
    w = _Steady(d, _planned(d), _benching_nothing([]), "0")
    with pytest.raises(v2.Unmeasured):
        v2.work(w)
    assert len(v2.pending(d)) == 2
    claim = v2.claim(v2.part_path(d, 0, al.EXCLUSIVE, 0))
    assert claim is not None
    claim.release()


def test_run_reports_gemm_benches_no_worker_could_bench(tmp_path, capsys):
    d = tmp_path / "v2"
    m = _planned(d)
    assert audit_v2_script._summary(d, "v2", time.time()) == 1
    assert "2 GEMM bench(es) of the v2 measurement are left" in capsys.readouterr().err
    v2.work(_Steady(d, m, _runner([]), "0"))
    assert audit_v2_script._summary(d, "v2", time.time()) == 0


def test_a_worker_reads_the_runs_hold_of_its_gpu(tmp_path, monkeypatch):
    import yaml

    from lib import gpu_health, kfd

    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(gf.write_sysfs(tmp_path / "sys")))
    guard = gpu_health.Guard(tmp_path / "guard", "s")
    g0, g1, _ = kfd.gpus()
    k0, k1 = gpu_health.gpu_key(g0), gpu_health.gpu_key(g1)
    off = [("power_management", "GPU 1: power management is off")]
    guard.settle(gpu_health.Check([k0, k1, gpu_health.HOST], {k1: off}))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"bench": {"health": {"max_wait_s": 60}}}))
    env = {gpu_health.GUARD_DIR_VAR: str(guard.dir), kfd.ROCR_VAR: "0,1"}
    held, health = audit_v2_script.held_check("1", cfg, env)
    assert held() == "GPU 1: power management is off" and health.max_wait_s == 60
    assert audit_v2_script.held_check("0", cfg, env)[0]() is None
    assert audit_v2_script.held_check("1", cfg, {kfd.ROCR_VAR: "0,1"})[0]() is None
    no = tmp_path / "off.yaml"
    no.write_text(yaml.safe_dump({"bench": {"health": {"enabled": False}}}))
    assert audit_v2_script.held_check("1", no, env)[0]() is None
    assert audit_v2_script.held_check("1", None, env)[0]() is not None
    guard.settle(
        gpu_health.Check([gpu_health.HOST], {gpu_health.HOST: [("ttm", "stuck")]})
    )
    assert "stuck" in audit_v2_script.held_check("0", cfg, env)[0]()


def test_a_v2_run_on_a_gpu_the_run_holds_alone_measures_nothing_at_once(
    tmp_path, monkeypatch
):
    """With no limit on the health waits, the audit's stage03 run still
    leaves a GPU the run holds alone out at once, so that its part goes
    back to the workers on the other GPUs instead of waiting for the GPU."""
    import yaml

    from lib import gpu_health, kfd

    root = gf.write_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1")
    for gpu in (0, 1):
        dev = root / "class" / "drm" / f"renderD{gf.GPUS[gpu].render_minor}"
        (dev / "device" / "pp_features").write_text("00. DPM ( 0) : enabled\n")
    guard = gpu_health.Guard(tmp_path / "guard", "s0")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    dev0 = root / "class" / "drm" / f"renderD{gf.GPUS[0].render_minor}" / "device"
    (dev0 / "pp_features").write_text("00. DPM ( 0) : disabled\n")
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)
    assert guard.settle(gpu_health.check(guard, [g0, g1], **quick))[0] == "raised"
    monkeypatch.setenv(gpu_health.GUARD_DIR_VAR, str(guard.dir))
    cfg = tmp_path / "cfg.yaml"
    health = {"poll_s": 1.0, "canary": {"enabled": False}}
    cfg.write_text(
        yaml.safe_dump({"bench": {"quiet_host": {"enabled": False}, "health": health}})
    )
    bench = bf.write_fake_bench(tmp_path)
    (tmp_path / "state").mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"state_dir": str(tmp_path / "state")}))
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))
    run = v2.stage03_runner(cfg, device="0", stall_timeout_s=5.0, startup_grace_s=5.0)
    got = run(tmp_path / "r0", [e["line"] for e in ENTRIES], bench)
    assert got.unmeasured and got.rc == 2 and got.logs == []
    assert "--leave-held-gpus leaves it out at once" in got.problems[0]
