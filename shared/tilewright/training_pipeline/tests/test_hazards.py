# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The hazard canary (lib/hazards.py, stages/stage00a_hazard_canary.py) on a
fake bench and a fake library: classes from logic parameters and names,
the canary's problems, the store's verdicts, and runs that pass, hang,
fault, run nothing or meet a stopped device."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

import bench_fakes as bf
import fake_hipblaslt as fh
from lib import correctness as cx
from lib import hazards as hz
from lib import kfd
import test_run_orchestrator as tro

stubs = tro.stubs
env = tro.env

# A made-up architecture: the canary reads no architecture-specific value.
ARCH = "gfx9990"
STEM = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_"
    + ARCH
)


def _name(mt, cd=(1, 1), pgl=0, tdmim=0):
    return (
        f"Cijk_Alik_Bljk_BBS_BH_MT{mt[0]}x{mt[1]}x{mt[2]}_MI16x16x1_CD{cd[0]}_{cd[1]}"
        f"_CLR0_PGL{pgl}_PGR2_TDMI3_TDMIM{tdmim}_WG16_8_1"
    )


def kernel(mt=(128, 64, 64), cd=(1, 1), pgl=0, tdmim=0, cluster=None):
    k = {
        "kernel_name": _name(mt, cd, pgl, tdmim),
        "mt_m": mt[0],
        "mt_n": mt[1],
        "mt_k": mt[2],
        "attributes": {},
    }
    if cluster is not None:
        k["attributes"] = {"cluster_dim_x": cluster[0], "cluster_dim_y": cluster[1]}
    return k


# ── classes ──────────────────────────────────────────────────────────────────


def test_name_values_read_the_parameters_tensilelite_abbreviates():
    name = _name((256, 128, 64), cd=(2, 4), pgl=2, tdmim=3)
    assert hz.name_values(name, "CD") == (2, 4)
    assert hz.name_values(name, "PGL") == (2,)
    assert hz.name_values(name, "TDMIM") == (3,)
    assert hz.name_values(name, "TDMI") == (3,)
    assert hz.name_values(name, "MT") is None
    assert hz.name_values("A_TDMIMn1_B", "TDMIM") == (-1,)
    assert hz.name_values("A_XCD2_4_B", "CD") is None


@pytest.mark.parametrize(
    "k, classes",
    [
        (kernel(), ()),
        (kernel(cd=(2, 4)), ("cluster_dim_y4",)),
        (kernel(cd=(4, 4)), ("cluster_dim_y4",)),
        (kernel(cd=(4, 2)), ()),
        (kernel(cluster=(2, 4)), ("cluster_dim_y4",)),
        (kernel(pgl=2), ("prefetch_gl2_2",)),
        (kernel(pgl=1), ()),
        (kernel(tdmim=3), ("tdm_iterate_mode_3",)),
        (kernel(tdmim=2), ()),
        (
            kernel(cd=(2, 4), pgl=2, tdmim=3),
            ("cluster_dim_y4", "prefetch_gl2_2", "tdm_iterate_mode_3"),
        ),
    ],
)
def test_hazard_classes_from_logic_parameters_and_names(k, classes):
    assert hz.hazard_classes(k) == classes


def test_the_canary_problems_cover_clusters_blocks_and_partial_tiles():
    full, edge = hz.canary_shapes(kernel(mt=(128, 64, 96), cd=(2, 4)), mx_block=32)
    assert (full.m, full.n, full.k, full.batch) == (512, 512, 384, 1)
    assert (edge.m, edge.n, edge.k, edge.batch) == (512 + 65, 512 + 33, 384, 2)
    plain = hz.canary_shapes(kernel(mt=(64, 64, 64)))[0]
    assert (plain.m, plain.n, plain.k) == (128, 128, 256)


def test_canary_lines_time_one_kernel_alone(config):
    from lib.bench_yaml import line_solution_list, line_skip_ratio, shape_of_line

    k = dict(kernel(mt=(128, 64, 64), cd=(2, 4)), sol_idx_global=7)
    lines = hz.canary_yaml(k, config, run=2)
    assert len(lines) == 2
    for line in lines:
        assert line_solution_list(line) == [7]
        assert line_skip_ratio(line) == 0
        assert "canary_7_r2_" in line
    assert shape_of_line(lines[0]) == (512, 512, 256, 1)


# ── the store ────────────────────────────────────────────────────────────────


def _rec(event, token, name, verdict=None, **kw):
    out = dict(event=event, token=token, name=name, arch=ARCH, kernel_id="k:" + name)
    if verdict:
        out["verdict"] = verdict
    out.update(kw)
    return out


def test_the_store_quarantines_ended_and_unfinished_runs(tmp_path):
    store = hz.CanaryStore(tmp_path)
    store.add(
        [
            _rec("start", "t1", "A"),
            _rec("end", "t1", "A", "pass", device="d", code="c", protocol="p"),
            _rec("start", "t2", "B"),
            _rec("end", "t2", "B", "hang"),
            _rec("start", "t3", "C"),
            _rec("start", "t4", "D"),
            _rec("end", "t4", "D", "interrupted"),
            _rec("start", "t5", "E", arch="gfx9991"),
            _rec("start", "t6", "F"),
            _rec("end", "t6", "F", "untested"),
        ]
    )
    held = hz.CanaryStore(tmp_path).quarantined(ARCH)
    assert sorted(held) == ["B", "C", "F"]
    assert held["B"]["verdict"] == "hang" and held["C"]["verdict"] == "unfinished"
    assert sorted(store.quarantined(ARCH, running=["t3"])) == ["B", "F"]
    assert store.passed("k:A", "d", "c", "p") is not None
    assert store.passed("k:A", "d", "other build", "p") is None
    (tmp_path / hz.CACHE_FILE).open("a").write("not json\n{}\n")
    assert sorted(hz.CanaryStore(tmp_path).quarantined(ARCH)) == ["B", "C", "F"]


def test_quarantined_names_follow_the_config(tmp_path):
    hz.CanaryStore(tmp_path).add([_rec("start", "t", "A")])
    cfg = {"arch": ARCH, "hazards": {"cache_dir": str(tmp_path)}}
    assert hz.quarantined_names(cfg) == {"A"}
    assert hz.quarantined_names(dict(cfg, arch=ARCH + ":xnack+")) == {"A"}
    assert hz.quarantined_names({"arch": ARCH}) == set()
    off = dict(cfg, hazards={"cache_dir": str(tmp_path), "enabled": False})
    assert hz.quarantined_names(off) == set()
    ks = [{"kernel_name": "A"}, {"kernel_name": "B"}]
    assert hz.without_quarantined(cfg, ks) == [{"kernel_name": "B"}]


def test_the_invocations_blocklist_adds_the_kernels_it_keeps_out(tmp_path):
    hz.CanaryStore(tmp_path).add([_rec("start", "t", "A")])
    blocklist = tmp_path / "invocation.json"
    cfg = {
        "arch": ARCH,
        "hazards": {"cache_dir": str(tmp_path)},
        "correctness": {"blocklist": str(blocklist)},
    }
    assert hz.quarantined_names(cfg) == {"A"}
    cx.write_blocklist(
        blocklist,
        cx.Blocklist(
            [
                cx.Entry(name="A", source=hz.SOURCE),
                cx.Entry(name="B", reason="hazard canary: unchecked", source=hz.SOURCE),
                cx.Entry(name="C", reason="wrong results", source="correctness gate"),
            ]
        ),
    )
    assert hz.quarantined_names(cfg) == {"A", "B"}
    blocklist.write_text("{not json")
    assert hz.quarantined_names(cfg) == {"A"}


# ── verdicts of one run ──────────────────────────────────────────────────────


def _log(tmp_path, *problems, name="x.log"):
    lines = list(bf.BANNER)
    for sid in problems:
        lines += bf.preamble(1)
        if sid is None:
            lines += ["NO solution found"]
        else:
            lines += bf.measured_lines(0, bf.Gemm(64, 64, 64), sid, 10.0)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    "problems, rc, reason, stopping, verdict",
    [
        ((7, 7), 0, None, False, "pass"),
        ((7, None), 0, None, False, "pass"),
        ((None, None), 0, None, False, "untested"),
        ((8,), 0, None, False, "untested"),
        ((7,), -9, "stall", False, "hang"),
        ((7,), -9, "fatal", False, "fault"),
        ((7,), -6, None, False, "fault"),
        ((7,), 134, None, False, "fault"),
        ((7,), -15, None, False, "interrupted"),
        ((7,), -9, None, True, "interrupted"),
    ],
)
def test_judge(tmp_path, problems, rc, reason, stopping, verdict):
    r = hz.judge(7, _log(tmp_path, *problems), rc, reason, 5.0, stopping)
    assert r.verdict == verdict, r.reason
    assert r.problems == sum(1 for p in problems if p == 7)


# ── canary passes ────────────────────────────────────────────────────────────

FAKE_BENCH = r"""
import json, os, signal, sys, time
sys.path.insert(0, __TESTS__)
import bench_fakes as bf
argv = sys.argv[1:]
yaml_path = argv[argv.index("--yaml") + 1]
plan = json.loads(os.environ.get("FAKE_CANARY", "{}"))
names = json.loads(os.environ["FAKE_CANARY_NAMES"])
with open(os.environ["FAKE_CANARY_LOG"], "a") as f:
    f.write(json.dumps({"rocr": os.environ.get("ROCR_VISIBLE_DEVICES"),
                        "yaml": open(yaml_path).read()}) + "\n")
print("\n".join(bf.BANNER), flush=True)
for line in open(yaml_path):
    if not line.strip().startswith("-"):
        continue
    sid = int(line.split("solution_index_list: [")[1].split("]")[0])
    shape = [int(line.split(f" {k}: ")[1].split(",")[0]) for k in ("M", "N", "K")]
    how = plan.get(str(sid), "pass")
    print("\n".join(bf.preamble(1)), flush=True)
    if how == "hang":
        time.sleep(600)
    if how == "fault":
        print("Memory access fault by GPU node-1 on address 0x1000.", flush=True)
        os.abort()
    if how == "fatal":
        print("HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION", flush=True)
        time.sleep(600)
    if how == "unsupported":
        print("NO solution found", flush=True)
        continue
    if how == "killed":
        os.kill(os.getpid(), signal.SIGKILL)
    rows = bf.measured_lines(0, bf.Gemm(*shape), sid, 12.5)
    name = names[str(sid)] if how != "wrong_name" else "Cijk_another_kernel"
    print("\n".join(rows).replace(bf.kernel_name((128, 128, 64)), name), flush=True)
"""


@pytest.fixture
def config(tmp_path):
    return {
        "config_id": "canary",
        "arch": ARCH,
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
        "hardware": {"n_cu": 7, "lds_bytes": 65536, "l2_bytes": 1 << 20},
        "runtime": {"devices": [0, 1], "rocm_libraries_root": str(tmp_path)},
        "hazards": {"timeout_s": 2.0, "cache_dir": str(tmp_path / "store")},
    }


SOLUTIONS = [
    ((64, 64, 64), {}),
    ((128, 64, 64), {"cd": (2, 4)}),
    ((128, 128, 64), {"pgl": 2}),
    ((64, 128, 64), {"tdmim": 3}),
    ((256, 64, 64), {"cd": (4, 4), "tdmim": 3}),
]


@pytest.fixture
def world(tmp_path, monkeypatch, config):
    fh.unhide_devices(monkeypatch)
    monkeypatch.setenv(kfd.ROCR_VAR, "2,3")
    libdir = tmp_path / "lib"
    libdir.mkdir()
    sols = []
    for i, (mt, kw) in enumerate(SOLUTIONS):
        s = bf.solution(i, i, mt=mt)
        s["name"] = _name(mt, **kw)
        if "cd" in kw:
            s["sizeMapping"]["clusterDim"] = list(kw["cd"]) + [1]
        sols.append(s)
    bf.write_logic(libdir / f"{STEM}.dat", sols, table=[s["index"] for s in sols])
    bench = tmp_path / "hipblaslt-bench"
    bench.write_text(
        f"#!{sys.executable}\n"
        + FAKE_BENCH.replace("__TESTS__", repr(str(Path(bf.__file__).parent)))
    )
    bench.chmod(bench.stat().st_mode | stat.S_IXUSR)
    names = {str(s["index"]): s["name"] for s in sols}
    monkeypatch.setenv("FAKE_CANARY_NAMES", json.dumps(names))
    monkeypatch.setenv("FAKE_CANARY_LOG", str(tmp_path / "bench_calls.jsonl"))
    monkeypatch.delenv("FAKE_CANARY", raising=False)
    return tmp_path, libdir, bench


def _spec(world, config, **kw):
    base, libdir, bench = world
    args = dict(
        cfg=config,
        dat_dir=libdir,
        library_stem=STEM,
        out_dir=base / "out",
        cache_dir=base / "store",
        bench_binary=bench,
        devices=[0],
        startup_grace_s=0.5,
    )
    args.update(kw)
    return hz.Spec(**args)


def _bench_calls(world):
    path = world[0] / "bench_calls.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _listed(call):
    from lib.bench_yaml import line_solution_list

    lines = [ln for ln in call["yaml"].splitlines() if ln.strip().startswith("-")]
    return sorted({sid for ln in lines for sid in line_solution_list(ln)})


def test_every_hazard_kernel_runs_alone_once(world, config):
    summary = hz.run_canary(_spec(world, config))
    assert summary["rc"] == 0 and summary["hazard"] == 4 and summary["checked"] == 4
    assert summary["by_class"] == {
        "cluster_dim_y4": 2,
        "tdm_iterate_mode_3": 2,
        "prefetch_gl2_2": 1,
    }
    calls = _bench_calls(world)
    assert [_listed(c) for c in calls] == [[0], [1], [2], [3], [4]]
    assert {c["rocr"] for c in calls} == {"2"}
    out = world[0] / "out"
    verdicts = json.loads((out / hz.VERDICTS_FILE).read_text())["kernels"]
    assert [v["verdict"] for v in verdicts] == ["pass"] * 4
    assert cx.read_list(out / hz.QUARANTINE_FILE).names == frozenset()
    again = hz.run_canary(_spec(world, config))
    assert again["checked"] == 0 and again["passed_before"] == 4
    assert len(_bench_calls(world)) == 5


def test_kernels_that_hang_fault_or_run_nothing_are_quarantined(
    world, config, monkeypatch
):
    monkeypatch.setenv(
        "FAKE_CANARY", json.dumps({"1": "hang", "2": "fault", "3": "unsupported"})
    )
    summary = hz.run_canary(_spec(world, config))
    assert summary["rc"] == 0 and summary["quarantined"] == 3
    out = world[0] / "out"
    held = cx.read_list(out / hz.QUARANTINE_FILE)
    by_name = json.loads(os.environ["FAKE_CANARY_NAMES"])
    assert held.names == {by_name["1"], by_name["2"], by_name["3"]}
    reasons = {e.name: e.reason for e in held.entries}
    assert reasons[by_name["1"]].startswith("hazard canary: hang (ClusterDim with")
    assert "fault" in reasons[by_name["2"]] and "untested" in reasons[by_name["3"]]
    calls = [_listed(c) for c in _bench_calls(world)]
    # The control runs before the first kernel and after the hang and the
    # fault.
    assert calls == [[0], [1], [0], [2], [0], [3], [4]]
    monkeypatch.delenv("FAKE_CANARY")
    again = hz.run_canary(_spec(world, config))
    assert again["checked"] == 0 and again["quarantined"] == 3
    assert again["quarantined_before"] == 3


def test_a_run_that_never_ended_quarantines_its_kernel(world, config):
    name = json.loads(os.environ["FAKE_CANARY_NAMES"])["4"]
    hz.CanaryStore(world[0] / "store").add([_rec("start", "x", name)])
    summary = hz.run_canary(_spec(world, config))
    assert summary["quarantined_before"] == 1 and summary["checked"] == 3
    (entry,) = cx.read_list(world[0] / "out" / hz.QUARANTINE_FILE).entries
    assert "never ended" in entry.reason
    assert [4] not in [_listed(c) for c in _bench_calls(world)]


def test_a_run_killed_from_outside_keeps_its_kernel_out_of_the_invocation(
    world, config, monkeypatch
):
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"2": "killed"}))
    summary = hz.run_canary(_spec(world, config))
    assert summary["rc"] == 0 and summary["checked"] == 4
    assert (summary["quarantined"], summary["kept_out"]) == (0, 1)
    out = world[0] / "out"
    (entry,) = cx.read_list(out / hz.QUARANTINE_FILE).entries
    assert entry.name == json.loads(os.environ["FAKE_CANARY_NAMES"])["2"]
    assert entry.reason.startswith("hazard canary: unchecked (PrefetchGL2 2)")
    assert "killed by SIGKILL from outside" in entry.reason
    assert hz.CanaryStore(world[0] / "store").quarantined(ARCH) == {}
    monkeypatch.delenv("FAKE_CANARY")
    again = hz.run_canary(_spec(world, config))
    assert (again["checked"], again["passed_before"], again["kept_out"]) == (1, 3, 0)
    assert cx.read_list(out / hz.QUARANTINE_FILE).names == frozenset()


def test_a_failing_control_stops_the_canary_before_any_kernel(
    world, config, monkeypatch
):
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"0": "fault"}))
    summary = hz.run_canary(_spec(world, config))
    assert summary["rc"] == 3 and summary["checked"] == 0
    assert "no control kernel passed before the first run" in summary["stopped"]
    assert [_listed(c) for c in _bench_calls(world)] == [[0]]
    assert hz.CanaryStore(world[0] / "store").quarantined(ARCH) == {}


def test_controls_that_run_no_canary_problem_stop_the_canary_blaming_none(
    world, config, monkeypatch
):
    """A canary line that does not fit the library runs nothing with any
    kernel: no kernel is quarantined for it."""
    plan = {str(i): "unsupported" for i in range(len(SOLUTIONS))}
    monkeypatch.setenv("FAKE_CANARY", json.dumps(plan))
    with pytest.raises(hz.CanaryError, match="no control kernel ran a canary"):
        hz.run_canary(_spec(world, config))
    assert [_listed(c) for c in _bench_calls(world)] == [[0]]
    assert hz.CanaryStore(world[0] / "store").quarantined(ARCH) == {}


def test_a_stuck_driver_process_after_a_hang_stops_the_device(
    world, config, monkeypatch
):
    seen = []

    def stuck():
        seen.append(len(_bench_calls(world)))
        return [(4242, "bench")] if len(seen) > 1 else []

    monkeypatch.setattr(hz, "stuck_processes", stuck)
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"1": "hang"}))
    summary = hz.run_canary(_spec(world, config))
    assert summary["rc"] == 3 and "uninterruptible sleep after a hang" in (
        summary["stopped"]
    )
    assert [_listed(c) for c in _bench_calls(world)] == [[0], [1]]
    assert (summary["quarantined"], summary["kept_out"]) == (1, 3)


def test_without_a_control_a_fault_stops_the_device(
    world, config, tmp_path, monkeypatch
):
    names = json.loads(os.environ["FAKE_CANARY_NAMES"])
    lst = tmp_path / "known.tsv"
    lst.write_text(f"X\t{names['0']}\n")
    cfg = dict(config, correctness={"lists": [str(lst)]})
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"2": "fault"}))
    summary = hz.run_canary(_spec(world, cfg))
    assert summary["rc"] == 3 and "no kernel outside the hazard classes" in (
        summary["stopped"]
    )
    assert [_listed(c) for c in _bench_calls(world)] == [[1], [2]]
    assert (summary["quarantined"], summary["kept_out"]) == (1, 2)


def test_a_hang_is_found_once_the_library_is_loaded(world, config, monkeypatch):
    """A canary bench that hangs after its library load is killed after
    hazards.timeout_s, however long the startup grace."""
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"1": "hang"}))
    summary = hz.run_canary(_spec(world, config, startup_grace_s=600.0))
    assert summary["rc"] == 0 and summary["quarantined"] == 1
    assert summary["seconds"] < 120


def test_a_run_another_invocation_still_runs_blames_no_kernel(world, config):
    names = json.loads(os.environ["FAKE_CANARY_NAMES"])
    store = hz.CanaryStore(world[0] / "store")
    me = hz.process_owner()
    assert me is not None and hz.owner_alive(me)
    gone = dict(me, start=me["start"] + 1)
    assert not hz.owner_alive(gone) and not hz.owner_alive(dict(me, boot="b"))
    assert not hz.owner_alive(None)
    store.add(
        [
            _rec("start", "live", names["4"], owner=me),
            _rec("start", "dead", names["3"], owner=gone),
        ]
    )
    assert sorted(store.quarantined(ARCH)) == [names["3"]]
    assert store.in_flight(ARCH) == {names["4"]}
    summary = hz.run_canary(_spec(world, config))
    assert summary["running_elsewhere"] == 1 and summary["quarantined"] == 1
    assert [_listed(c) for c in _bench_calls(world)] == [[0], [1], [2]]
    (kept,) = [
        e
        for e in cx.read_list(world[0] / "out" / hz.QUARANTINE_FILE).entries
        if e.name == names["4"]
    ]
    assert "another invocation was running it alone" in kept.reason
    recs = [r for r in hz.CanaryStore(world[0] / "store").records() if r["token"]]
    assert all(r["owner"] == me for r in recs if r["token"] not in ("live", "dead"))


def test_a_process_stuck_in_the_driver_keeps_the_canary_from_starting(
    world, config, monkeypatch, tmp_path
):
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    (proc / "4242" / "stat").write_text("4242 (bench) D 1 2 3\n")
    (proc / "4242" / "comm").write_text("bench\n")
    monkeypatch.setattr(kfd, "PROC", proc)
    (kfd.kfd_root() / "proc" / "4242").mkdir(parents=True)
    summary = hz.run_canary(_spec(world, config))
    assert summary["rc"] == 3 and "4242 (bench)" in summary["stopped"]
    assert _bench_calls(world) == []


def test_a_bench_that_runs_another_kernel_stops_the_canary(world, config, monkeypatch):
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"1": "wrong_name"}))
    with pytest.raises(hz.CanaryError, match="does not load the library"):
        hz.run_canary(_spec(world, config))
    assert hz.CanaryStore(world[0] / "store").quarantined(ARCH) == {}


def test_listed_kernels_and_a_dry_run_run_nothing(world, config, tmp_path):
    names = json.loads(os.environ["FAKE_CANARY_NAMES"])
    lst = tmp_path / "known.tsv"
    lst.write_text("".join(f"X\t{names[s]}\n" for s in ("1", "2", "3", "4")))
    cfg = dict(config, correctness={"lists": [str(lst)]})
    summary = hz.run_canary(_spec(world, cfg))
    assert summary["hazard"] == 0 and summary["listed"] == 4
    assert _bench_calls(world) == []
    dry = hz.run_canary(_spec(world, config, dry_run=True))
    assert dry["checked"] == 0 and _bench_calls(world) == []
    rows = json.loads((world[0] / "out" / hz.VERDICTS_FILE).read_text())["kernels"]
    assert {r["verdict"] for r in rows} == {"unchecked"}
    assert dry["kept_out"] == 0
    assert cx.read_list(world[0] / "out" / hz.QUARANTINE_FILE).names == frozenset()


def test_the_stage_reports_through_its_exit_status(world, config, monkeypatch):
    import yaml

    import stage00a_hazard_canary as s00a

    path = world[0] / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    base = [
        "--config-yaml",
        str(path),
        "--dat-dir",
        str(world[1]),
        "--bench-binary",
        str(world[2]),
        "--startup-grace-s",
        "0.5",
        "--quiet",
    ]
    assert s00a.main(base + ["--out-dir", str(world[0] / "a")]) == 0
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"0": "fault"}))
    hz.CanaryStore(world[0] / "store").path.unlink()
    assert s00a.main(base + ["--out-dir", str(world[0] / "b")]) == 3
    assert s00a.main(base + ["--out-dir", str(world[0] / "c"), "--dat-dir", "x"]) == 1


# ── the orchestrator ─────────────────────────────────────────────────────────


def _orch(tmp_path, **more):
    """A stub run's config whose library holds a kernel of a hazard
    class."""
    import run_orchestrator as ro
    from test_run_orchestrator import STEM, make_config

    over = {"hazards.enabled": True, "full.total_rounds": 1, **more}
    path = make_config(tmp_path, **over)
    cfg, _ = ro.load_config(path)
    libdir = ro.derive_paths(cfg).library_dir
    libdir.mkdir(parents=True, exist_ok=True)
    sols = [bf.solution(0, 0), bf.solution(1, 1, mt=(128, 64, 64))]
    sols[1]["name"] = _name((128, 64, 64), cd=(2, 4))
    bf.write_logic(libdir / f"{STEM}.dat", sols, table=[0, 1])
    return path


def test_the_canary_runs_first_and_every_stage_applies_its_quarantine(
    tmp_path, stubs, env, monkeypatch
):
    import yaml

    import run_orchestrator as ro
    from test_run_orchestrator import arg, calls, run

    monkeypatch.setenv("STUB_QUARANTINED", json.dumps(["Cijk_q"]))
    run_dir = tmp_path / "run"
    cfg = _orch(tmp_path, **{"planner.enabled": True})
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    seen = calls(env)
    canary = seen[0]
    assert canary["stage"] == "stage00a" and canary["leases"] == {}
    assert "stage00" not in [c["stage"] for c in seen]
    assert arg(canary, "--out-dir") == str(run_dir / "hazards" / "canary.new")
    assert arg(canary, "--cache-dir") == str(run_dir / "hazards" / "cache")
    assert arg(canary, "--devices") == "0" and arg(canary, "--hardware-device") == "0"
    snapshot = yaml.safe_load(Path(arg(canary, "--config-yaml")).read_text())
    blocklist = Path(snapshot["correctness"]["blocklist"])
    assert cx.read_list(blocklist).names == {"Cijk_q"}
    assert (run_dir / "hazards" / "canary" / ro.STAGE_MARKER).is_file()
    assert not (run_dir / "hazards" / "canary.new").exists()
    marker = json.loads((run_dir / "round_0" / "stage01" / ro.STAGE_MARKER).read_text())
    assert marker["blocklist"] == cx.blocklist_record(blocklist)


def test_the_gate_leaves_the_quarantine_unchecked(tmp_path, stubs, env, monkeypatch):
    import yaml

    from test_run_orchestrator import arg, calls, run

    monkeypatch.setenv("STUB_QUARANTINED", json.dumps(["Cijk_q"]))
    monkeypatch.setenv("STUB_BLOCKED", json.dumps(["Cijk_q", "Cijk_y"]))
    harness = tmp_path / "errmap"
    harness.write_text("")
    run_dir = tmp_path / "run"
    cfg = _orch(
        tmp_path,
        **{
            "planner.enabled": True,
            "correctness.enabled": True,
            "correctness.harness": str(harness),
        },
    )
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    seen = calls(env)
    assert [c["stage"] for c in seen[:2]] == ["stage00a", "stage00"]
    quarantine = run_dir / "hazards" / "canary" / hz.QUARANTINE_FILE
    assert arg(seen[1], "--quarantine") == str(quarantine)
    snapshot = yaml.safe_load(Path(arg(seen[1], "--config-yaml")).read_text())
    assert cx.read_list(snapshot["correctness"]["blocklist"]).names == {
        "Cijk_q",
        "Cijk_y",
    }


def test_a_quarantine_needs_the_planner(tmp_path, stubs, env, monkeypatch, capsys):
    from test_run_orchestrator import calls, run

    monkeypatch.setenv("STUB_QUARANTINED", json.dumps(["Cijk_q"]))
    cfg = _orch(tmp_path, **{"planner.enabled": False})
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 2
    assert "set planner.enabled" in capsys.readouterr().err
    assert [c["stage"] for c in calls(env)] == ["stage00a"]
    monkeypatch.setenv("STUB_QUARANTINED", "[]")
    assert run(stubs, cfg, tmp_path / "run2", "--mode", "initial") == 0


def test_a_stopped_canary_stops_the_invocation(
    tmp_path, stubs, env, monkeypatch, capsys
):
    from test_run_orchestrator import calls, run

    monkeypatch.setenv("STUB_FAIL", "stage00a:3")
    cfg = _orch(tmp_path, **{"planner.enabled": True})
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 3
    assert "reset the device, then run the same command" in capsys.readouterr().err
    assert [c["stage"] for c in calls(env)] == ["stage00a"]


def test_without_the_canary_nothing_runs_before_the_rounds(tmp_path, stubs, env):
    import yaml

    from test_run_orchestrator import arg, calls, run

    cfg = _orch(tmp_path, **{"hazards.enabled": False})
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 0
    seen = calls(env)
    assert seen[0]["stage"] == "stage01"
    snapshot = yaml.safe_load(Path(arg(seen[0], "--config-yaml")).read_text())
    assert "blocklist" not in (snapshot.get("correctness") or {})
