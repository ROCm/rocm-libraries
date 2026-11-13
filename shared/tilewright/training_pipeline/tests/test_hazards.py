# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The hazard canary (lib/hazards.py, stages/stage00a_hazard_canary.py) on a
fake bench and a fake library: classes from logic parameters and names,
the canary's problems, the store's verdicts, and runs that pass, hang,
fault, run nothing or meet a stopped device; the hang screen's problems,
lines and passes, and screens whose processes stop at a kernel that hangs
or faults alone too, at one that does not, after a problem the bench finds
no solution for, or on a device that no longer answers."""

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
            _rec("start", "t7", "G", blame=False),
            _rec("start", "t8", "H", blame=True),
        ]
    )
    held = hz.CanaryStore(tmp_path).quarantined(ARCH)
    assert sorted(held) == ["B", "C", "F", "H"]
    assert held["B"]["verdict"] == "hang" and held["C"]["verdict"] == "unfinished"
    store.add([_rec("end", "t8", "H", "cleared")])
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

# Runs the listed kernels of each line in list order, as hipblaslt-bench
# does, each as FAKE_CANARY says: by "<solution>@<MxNxKxB>" on that problem,
# else by "<solution>"; "hang_with_others" hangs in a line that lists other
# kernels too, "slow" sleeps 2.5 s before its row, "fault" prints a memory
# fault and aborts, "abort" aborts without a word. A line that runs no kernel
# prints "NO solution found" instead of the "Is supported" line, as the bench
# does, and "exit@<MxNxKxB>" ends the process with that status once it has
# printed the problem.
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
    listed = line.split("solution_index_list: [")[1].split("]")[0]
    sids = [int(t) for t in listed.split(",")]
    shape = [int(line.split(f" {k}: ")[1].split(",")[0])
             for k in ("M", "N", "K", "batch_count")]
    at = "x".join(str(v) for v in shape)
    hows = [plan.get(f"{sid}@{at}", plan.get(str(sid), "pass")) for sid in sids]
    runs = [(sid, how) for sid, how in zip(sids, hows) if how != "unsupported"]
    if runs:
        print("\n".join(bf.preamble(len(runs))), flush=True)
    else:
        print("NO solution found", flush=True)
    for rank, (sid, how) in enumerate(runs):
        if how == "hang" or (how == "hang_with_others" and len(sids) > 1):
            time.sleep(600)
        if how == "slow":
            time.sleep(2.5)
        if how == "fault":
            print("Memory access fault by GPU node-1 on address 0x1000.", flush=True)
            os.abort()
        if how == "abort":
            os.abort()
        if how == "fatal":
            print("HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION", flush=True)
            time.sleep(600)
        if how == "killed":
            os.kill(os.getpid(), signal.SIGKILL)
        rows = bf.measured_lines(rank, bf.Gemm(*shape), sid, 12.5)
        name = names[str(sid)] if how != "wrong_name" else "Cijk_another_kernel"
        print("\n".join(rows).replace(bf.kernel_name((128, 128, 64)), name), flush=True)
    if len(runs) > 1:
        rows = bf.winner(0, bf.Gemm(*shape), runs[0][0], 12.5)
        name = names[str(runs[0][0])]
        print("\n".join(rows).replace(bf.kernel_name((128, 128, 64)), name), flush=True)
    if f"exit@{at}" in plan:
        sys.exit(plan[f"exit@{at}"])
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


def _write_library(libdir, solutions):
    """The logic file of `solutions` ((macro tile, hazard parameters) each,
    solution indices in order); {solution index: name}."""
    sols = []
    for i, (mt, kw) in enumerate(solutions):
        s = bf.solution(i, i, mt=mt)
        s["name"] = _name(mt, **kw)
        if "cd" in kw:
            s["sizeMapping"]["clusterDim"] = list(kw["cd"]) + [1]
        sols.append(s)
    bf.write_logic(libdir / f"{STEM}.dat", sols, table=[s["index"] for s in sols])
    return {str(s["index"]): s["name"] for s in sols}


@pytest.fixture
def world(tmp_path, monkeypatch, config):
    fh.unhide_devices(monkeypatch)
    monkeypatch.setenv(kfd.ROCR_VAR, "2,3")
    libdir = tmp_path / "lib"
    libdir.mkdir()
    names = _write_library(libdir, SOLUTIONS)
    bench = tmp_path / "hipblaslt-bench"
    bench.write_text(
        f"#!{sys.executable}\n"
        + FAKE_BENCH.replace("__TESTS__", repr(str(Path(bf.__file__).parent)))
    )
    bench.chmod(bench.stat().st_mode | stat.S_IXUSR)
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
    # The fault aborts its bench at once: the error it printed, not the exit
    # status, is the reason, however soon the watchdog looked.
    assert reasons[by_name["2"]].startswith("hazard canary: fault (")
    assert reasons[by_name["2"]].endswith(": its bench reported a fatal GPU error")
    assert "untested" in reasons[by_name["3"]]
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


def test_a_hung_kernel_stuck_in_the_driver_is_not_killed_and_stops_the_device(
    world, config, monkeypatch
):
    from lib import procs

    left = []

    def stuck(proc):
        """The first bench killed stays stuck in the driver."""
        if not left:
            left.append(proc)
        return [proc.pid] if proc is left[0] else []

    monkeypatch.setattr(procs, "stuck_members", stuck)
    monkeypatch.setattr(procs, "STUCK_WAIT_S", 0.3)
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"1": "hang"}))
    try:
        summary = hz.run_canary(_spec(world, config))
        assert summary["rc"] == 3 and "a killed bench did not end" in summary["stopped"]
        assert "stuck in the GPU driver and is left running" in summary["stopped"]
        assert left[0].poll() is None
    finally:
        for p in left:
            p.kill()
            p.wait()


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


# ── the hang screen ──────────────────────────────────────────────────────────

# Kernels of no hazard class besides SOLUTIONS' first: solutions 5 to 8.
PLAIN = [(128, 128, 64), (256, 128, 64), (128, 256, 64), (256, 256, 64)]
SCREENED = [0, 5, 6, 7, 8]
SCREEN = {
    "enabled": True,
    "shapes": [[300, 200, 100], [700, 500, 300, 2]],
    "per_cell": 0,
    "stall_s": 2.0,
}


@pytest.fixture
def screen_world(world, monkeypatch):
    names = _write_library(world[1], SOLUTIONS + [(mt, {}) for mt in PLAIN])
    monkeypatch.setenv("FAKE_CANARY_NAMES", json.dumps(names))
    return world


def _screened(config, **screen):
    return dict(config, hazards=dict(config["hazards"], screen=dict(SCREEN, **screen)))


def _names():
    return json.loads(os.environ["FAKE_CANARY_NAMES"])


def test_screen_lines_list_their_kernels_each_timed_once(config):
    from lib.bench_yaml import line_knob, line_skip_ratio, line_solution_list
    from lib.bench_yaml import shape_of_line
    from lib.shapes import Shape

    line = hz.screen_line(Shape(700, 500, 300, 2), config, [8, 0, 5], "screen1_p0")
    assert line_solution_list(line) == [8, 0, 5]
    assert shape_of_line(line) == (700, 500, 300, 2)
    assert line_skip_ratio(line) == 0 and "screen1_p0" in line
    knobs = {k: line_knob(line, k) for k in hz.SCREEN_KNOBS}
    assert knobs == {"iters": "1", "cold_iters": "1", "flush": "false", "rotating": "0"}
    assert line_knob(line, "adaptive") is None and line_knob(line, "min_iters") is None


def test_the_screen_problems_are_its_shapes_then_a_draw_in_every_cell(config):
    from lib import grid
    from lib.shapes import Shape

    cfg = dict(config, bench={"device_memory_gib": 64})
    given = hz.ScreenSettings.of(
        {"shapes": [[300, 200, 100], [700, 500, 300, 2], [300, 200, 100, 1]]}
    )
    assert given.per_cell == 1 and not given.enabled and given.stall_s == 120.0
    explicit = hz.ScreenSettings.of(dict(SCREEN, shapes=given.shapes))
    assert hz.screen_shapes(cfg, explicit) == [
        Shape(300, 200, 100, 1),
        Shape(700, 500, 300, 2),
    ]
    drawn = hz.screen_shapes(cfg, hz.ScreenSettings.of({"seed": 3}))
    assert sorted(grid.cell_key(*s.as_tuple()) for s in drawn) == sorted(
        grid.all_cell_labels()
    )
    assert hz.screen_shapes(cfg, hz.ScreenSettings.of({"seed": 3})) == drawn
    assert hz.screen_shapes(cfg, hz.ScreenSettings.of({"seed": 4})) != drawn
    both = hz.screen_shapes(cfg, given)
    assert both[:2] == hz.screen_shapes(cfg, explicit)
    assert both[2:] == hz.screen_shapes(cfg, hz.ScreenSettings.of({}))
    a, b = Shape(1, 2, 3, 1), Shape(4, 5, 6, 2)
    assert hz.screen_protocol("h", [a, b]) == hz.screen_protocol("h", [b, a])
    assert hz.screen_protocol("h", [a, b]) != hz.screen_protocol("h", [a])
    assert hz.screen_protocol("h", [a]) != hz.screen_protocol("other build", [a])


def test_a_kernel_that_hangs_on_a_problem_is_confirmed_alone_and_quarantined(
    screen_world, config, monkeypatch
):
    """The fake bench makes solution 6 hang on the second problem, which the
    bench does not run solution 5 on: the screen's process stops after 0's
    row, 5 runs nothing alone, 6 also hangs when it runs alone and is
    quarantined, and the problem resumes after it. The next invocation
    screens nothing."""
    monkeypatch.setenv(
        "FAKE_CANARY",
        json.dumps({"5@700x500x300x2": "unsupported", "6@700x500x300x2": "hang"}),
    )
    cfg = _screened(config)
    summary = hz.run_canary(_spec(screen_world, cfg))
    assert summary["rc"] == 0 and summary["checked"] == 4
    calls = [_listed(c) for c in _bench_calls(screen_world)]
    canary, screen = calls[:5], calls[5:]
    assert canary == [[0], [1], [2], [3], [4]]
    # The control runs before the screen, after its process stops and after
    # the hang alone.
    assert screen == [[0], SCREENED, SCREENED, [0], [5], [6], [0], [7, 8]]
    hung = _names()["6"]
    held = hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH)
    assert list(held) == [hung] and held[hung]["verdict"] == "hang"
    out = screen_world[0] / "out"
    (entry,) = cx.read_list(out / hz.QUARANTINE_FILE).entries
    assert entry.name == hung and entry.source == hz.SOURCE
    assert entry.reason.startswith("hazard canary: hang (hang screen at 700x500x300x2)")
    assert not hz.kept_out_unchecked(entry)
    sc = summary["screen"]
    assert sc["problems"] == ["300x200x100x1", "700x500x300x2"]
    assert (sc["kernels"], sc["screened"], sc["passed"], sc["quarantined"]) == (
        5,
        5,
        4,
        1,
    )
    (stopped,) = sc["stops"]
    assert (stopped["shape"], stopped["ended"], stopped["rows"]) == (
        "700x500x300x2",
        "hung",
        1,
    )
    assert [a["verdict"] for a in stopped["alone"]] == ["untested", "hang"]
    rows = json.loads((out / hz.VERDICTS_FILE).read_text())["kernels"]
    (row,) = [r for r in rows if r["solution_index"] == 6]
    assert (row["verdict"], row["source"], row["classes"]) == (
        "quarantined",
        "screen",
        [],
    )
    again = hz.run_canary(_spec(screen_world, cfg))
    assert again["screen"]["kernels"] == 4 and again["screen"]["passed_before"] == 4
    assert again["screen"]["screened"] == 0
    assert len(_bench_calls(screen_world)) == len(calls)
    assert cx.read_list(out / hz.QUARANTINE_FILE).names == {hung}


def test_kernels_that_fault_among_others_and_alone_are_quarantined(
    screen_world, config, monkeypatch
):
    """Solution 8, listed last, reports a fatal GPU error on the first
    problem, and solution 6 aborts on the second, among other kernels and
    alone: each process stops at it, it faults alone too and is
    quarantined, and the second problem's process resumes after 6."""
    monkeypatch.setenv(
        "FAKE_CANARY",
        json.dumps({"8@300x200x100x1": "fatal", "6@700x500x300x2": "abort"}),
    )
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    # The control runs before the screen, after each stop and after each
    # fault alone.
    assert screen == [[0], SCREENED, [0], [8], [0], [0, 5, 6, 7], [0], [6], [0], [7]]
    sc = summary["screen"]
    stops = [
        (s["shape"], s["ended"], s["rows"], s["suspect"]["solution"])
        for s in sc["stops"]
    ]
    assert stops == [
        ("300x200x100x1", "reported a fatal GPU error", 4, 8),
        ("700x500x300x2", "ended with rc=-6", 2, 6),
    ]
    assert [[a["verdict"] for a in s["alone"]] for s in sc["stops"]] == [
        ["fault"],
        ["fault"],
    ]
    assert (sc["passed"], sc["quarantined"]) == (3, 2)
    names = _names()
    held = hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH)
    assert {n: r["verdict"] for n, r in held.items()} == {
        names["8"]: "fault",
        names["6"]: "fault",
    }
    out = screen_world[0] / "out"
    reasons = {e.name: e.reason for e in cx.read_list(out / hz.QUARANTINE_FILE).entries}
    assert reasons == {
        names["8"]: "hazard canary: fault (hang screen at 300x200x100x1): its "
        "bench reported a fatal GPU error",
        names["6"]: "hazard canary: fault (hang screen at 700x500x300x2): its "
        "bench ended with rc=-6",
    }
    rows = json.loads((out / hz.VERDICTS_FILE).read_text())["kernels"]
    by_sid = {r["solution_index"]: (r["verdict"], r["source"]) for r in rows}
    assert by_sid[6] == by_sid[8] == ("quarantined", "screen")


def test_a_stall_that_its_kernel_does_not_repeat_alone_blames_none(
    screen_world, config, monkeypatch
):
    """The fake bench makes solution 6 hang among other kernels but not
    alone: no kernel is quarantined, each problem resumes after it, and it
    passes, since the screen judges a kernel by what it does alone."""
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"6": "hang_with_others"}))
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    once = [SCREENED, [0], [6], [7, 8]]
    assert screen == [[0], *once, *once]
    store = hz.CanaryStore(screen_world[0] / "store")
    assert store.quarantined(ARCH) == {}
    sc = summary["screen"]
    assert (sc["passed"], sc["quarantined"], sc["kept_out"]) == (5, 0, 0)
    assert [a["verdict"] for s in sc["stops"] for a in s["alone"]] == ["pass", "pass"]
    alone = [r for r in store.records() if (r.get("screen") or {}).get("shape")]
    assert {r["verdict"] for r in alone if r["event"] == "end"} == {"cleared"}
    assert cx.read_list(screen_world[0] / "out" / hz.QUARANTINE_FILE).names == set()


def test_a_kernel_that_stops_before_any_row_is_blamed_once_others_run(
    screen_world, config, monkeypatch
):
    """The fake bench makes solution 0, the first kernel listed, hang on the
    second problem: its run alone is held back until the rest of the
    problem's kernels run to their rows, and it is quarantined then. A
    held-back run that never ended would blame it for nothing."""
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"0@700x500x300x2": "hang"}))
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    assert screen == [[0], SCREENED, SCREENED, [0], [0], [0], [5, 6, 7, 8]]
    store = hz.CanaryStore(screen_world[0] / "store")
    hung = _names()["0"]
    held = store.quarantined(ARCH)
    assert list(held) == [hung] and held[hung]["verdict"] == "hang"
    assert held[hung]["reason"].endswith("; other kernels run on the problem")
    alone = [r for r in store.records() if r.get("name") == hung and "screen" in r]
    assert [r["verdict"] for r in alone if r["event"] == "end"] == [
        "unconfirmed",
        "hang",
    ]
    assert (summary["screen"]["passed"], summary["screen"]["left"]) == (4, [])
    (start,) = [r for r in alone if r["event"] == "start"]
    assert start["blame"] is False
    died = hz.CanaryStore(screen_world[0] / "died")
    died.add([dict(start, owner=None)])
    assert died.quarantined(ARCH) == {}


def test_a_problem_no_kernel_runs_on_blames_none(screen_world, config, monkeypatch):
    """Every kernel hangs on the second problem: two stop alone before any
    runs to its row, and the screen leaves the problem with neither
    blamed; no kernel passes the screen."""
    plan = {f"{s}@700x500x300x2": "hang" for s in SCREENED}
    monkeypatch.setenv("FAKE_CANARY", json.dumps(plan))
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    rest = [5, 6, 7, 8]
    assert screen == [[0], SCREENED, SCREENED, [0], [0], [0], rest, [0], [5], [0]]
    assert hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH) == {}
    sc = summary["screen"]
    assert (sc["left"], sc["passed"], sc["quarantined"]) == (["700x500x300x2"], 0, 0)
    assert cx.read_list(screen_world[0] / "out" / hz.QUARANTINE_FILE).names == set()


def test_a_process_that_stops_after_no_solution_runs_no_kernel_alone(
    screen_world, config, monkeypatch
):
    """The bench finds no solution among the kernels on the second problem
    and its process then ends with exit 1: it ran none of them there, so
    once the control kernel passes they all clear the problem without a run
    alone, and none is suspected."""
    plan = {f"{s}@700x500x300x2": "unsupported" for s in SCREENED}
    plan["exit@700x500x300x2"] = 1
    monkeypatch.setenv("FAKE_CANARY", json.dumps(plan))
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    assert screen == [[0], SCREENED, SCREENED, [0]]
    sc = summary["screen"]
    (stopped,) = sc["stops"]
    assert (stopped["shape"], stopped["ended"], stopped["rows"]) == (
        "700x500x300x2",
        "ended with rc=1",
        0,
    )
    assert (stopped["suspect"], stopped["alone"]) == (None, [])
    assert (sc["passed"], sc["quarantined"], sc["left"]) == (5, 0, [])
    store = hz.CanaryStore(screen_world[0] / "store")
    assert not [r for r in store.records() if r.get("event") == hz.STOPPED]
    assert store.quarantined(ARCH) == {}


def test_a_kernel_only_slower_than_stall_s_is_not_blamed(
    screen_world, config, monkeypatch
):
    """A run alone has the canary's timeout_s when that is longer than the
    screen's stall_s."""
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"6@700x500x300x2": "slow"}))
    cfg = _screened(
        dict(config, hazards=dict(config["hazards"], timeout_s=6.0)), stall_s=1.0
    )
    summary = hz.run_canary(_spec(screen_world, cfg))
    assert summary["rc"] == 0
    (stopped,) = summary["screen"]["stops"]
    assert stopped["ended"] == "hung" and stopped["alone"][0]["verdict"] == "pass"
    assert hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH) == {}
    assert summary["screen"]["passed"] == 5


def test_a_problem_lists_its_kernels_in_processes_of_the_benchs_limit(
    screen_world, config, monkeypatch
):
    """With a limit of two kernels a line, each problem takes three
    processes; a hang in the second resumes with the third's kernel."""
    from lib import planner

    monkeypatch.setattr(planner, "MAX_BENCH_LIST", 2)
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"7@700x500x300x2": "hang"}))
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    chunks = [[0, 5], [6, 7], [8]]
    assert screen == [[0], *chunks, [0, 5], [6, 7], [0], [7], [0], [8]]
    assert list(hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH)) == [
        _names()["7"]
    ]
    assert summary["screen"]["passed"] == 4


def test_the_problems_are_shared_by_the_devices(screen_world, config, monkeypatch):
    """Each device checks itself with the control kernel and takes the
    next problem; a hang is quarantined whichever device finds it."""
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"6@700x500x300x2": "hang"}))
    summary = hz.run_canary(_spec(screen_world, _screened(config), devices=[0, 1]))
    assert summary["rc"] == 0 and summary["devices"] == [0, 1]
    logs = screen_world[0] / "out" / "logs"
    for d in (0, 1):
        assert (logs / f"control_before_screen_dev{d}.log").is_file()
    screened = [c for c in _bench_calls(screen_world) if len(_listed(c)) > 1]
    assert [_listed(c) for c in screened] == [SCREENED, SCREENED, [7, 8]]
    assert {c["rocr"] for c in screened} <= {"2", "3"}
    assert list(hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH)) == [
        _names()["6"]
    ]
    assert summary["screen"]["passed"] == 4


def test_screen_passes_hold_for_the_problems_they_were_made_on(screen_world, config):
    """A pass of the screen holds for its problems whatever its patience; a
    screen of other problems runs its kernels again."""
    first = hz.run_canary(_spec(screen_world, _screened(config)))
    assert first["screen"]["passed"] == 5 and first["screen"]["stops"] == []
    seen = len(_bench_calls(screen_world))
    again = hz.run_canary(_spec(screen_world, _screened(config, stall_s=30.0)))
    assert (again["screen"]["passed_before"], again["screen"]["screened"]) == (5, 0)
    assert len(_bench_calls(screen_world)) == seen
    other = hz.run_canary(_spec(screen_world, _screened(config, shapes=[[64, 64, 64]])))
    assert (other["screen"]["passed_before"], other["screen"]["passed"]) == (0, 5)
    assert [_listed(c) for c in _bench_calls(screen_world)[seen:]] == [[0], SCREENED]


def test_without_the_screen_the_canary_runs_as_before(screen_world, config):
    summary = hz.run_canary(_spec(screen_world, _screened(config, enabled=False)))
    assert summary["screen"] is None and summary["rc"] == 0
    assert [_listed(c) for c in _bench_calls(screen_world)] == [[0], [1], [2], [3], [4]]
    records = hz.CanaryStore(screen_world[0] / "store").records()
    assert records and not [r for r in records if "screen" in r]


def test_with_the_screen_off_its_quarantine_still_leaves_the_pool(
    screen_world, config, tmp_path
):
    """With nothing to run alone, stage00a takes no GPU and still writes a
    kernel the screen quarantined into its quarantine."""
    names = _names()
    lst = tmp_path / "known.tsv"
    lst.write_text("".join(f"X\t{names[s]}\n" for s in ("1", "2", "3", "4")))
    cfg = dict(config, correctness={"lists": [str(lst)]})
    hz.CanaryStore(screen_world[0] / "store").add(
        [_rec("end", "t", names["6"], "hang", screen={"shape": "700x500x300x2"})]
    )
    summary = hz.run_canary(_spec(screen_world, cfg))
    assert summary["screen"] is None and _bench_calls(screen_world) == []
    (entry,) = cx.read_list(screen_world[0] / "out" / hz.QUARANTINE_FILE).entries
    assert entry.name == names["6"]
    assert entry.reason.startswith("hazard canary: hang (hang screen at 700x500x300x2)")


def test_a_run_alone_cut_short_keeps_its_kernel_out_unchecked(
    screen_world, config, monkeypatch
):
    """Solution 6 is killed from outside on the second problem, among others
    and alone: it is not quarantined but stays out of the invocation, and
    the next invocation screens it again there, alone, since a process
    stopped at it."""
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"6@700x500x300x2": "killed"}))
    summary = hz.run_canary(_spec(screen_world, _screened(config)))
    assert summary["rc"] == 0
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    assert screen == [[0], SCREENED, SCREENED, [0], [6], [7, 8]]
    assert summary["screen"]["stops"][0]["ended"] == "was killed by SIGKILL"
    out = screen_world[0] / "out"
    (entry,) = cx.read_list(out / hz.QUARANTINE_FILE).entries
    assert entry.name == _names()["6"] and hz.kept_out_unchecked(entry)
    assert entry.reason.startswith(
        "hazard canary: unchecked (hang screen at 700x500x300x2)"
    )
    assert hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH) == {}
    assert (summary["screen"]["passed"], summary["kept_out"]) == (4, 1)
    monkeypatch.delenv("FAKE_CANARY")
    seen = len(_bench_calls(screen_world))
    again = hz.run_canary(_spec(screen_world, _screened(config)))
    assert [_listed(c) for c in _bench_calls(screen_world)[seen:]] == [[0], [6]]
    assert (again["screen"]["passed"], again["kept_out"]) == (1, 0)
    assert again["screen"]["swept_before"] == 1
    (suspect,) = again["screen"]["suspects"]
    assert (suspect["solution"], suspect["alone"][0]["verdict"]) == (6, "pass")


def test_a_device_that_does_not_answer_after_a_screen_stall_stops_the_screen(
    screen_world, config, monkeypatch
):
    """No kernel is blamed for the stall; the kernels that ran on every
    problem before it (0 and 5) pass. Once the device answers again, the
    screen goes on where it stopped: the kernels left run on the second
    problem alone, the one the process stopped at after the others, and it
    is quarantined when it hangs alone."""
    seen = []

    def stuck():
        seen.append(len(_bench_calls(screen_world)))
        return [(4242, "bench")] if len(seen) > 1 else []

    monkeypatch.setattr(hz, "stuck_processes", stuck)
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"6@700x500x300x2": "hang"}))
    cfg = _screened(config)
    summary = hz.run_canary(_spec(screen_world, cfg))
    assert summary["rc"] == 3
    assert (
        "uninterruptible sleep after a screen bench that hung on 700x500x300x2"
        in summary["stopped"]
    )
    (stopped,) = summary["screen"]["stops"]
    assert stopped["suspect"]["solution"] == 6
    assert summary["stopped"].endswith(
        f"; the first kernel its line lists after its last row, solution 6 "
        f"({stopped['suspect']['kernel']}), runs alone after the problem's other "
        f"kernels next time (its log is {stopped['log']})"
    )
    screen = [_listed(c) for c in _bench_calls(screen_world)][5:]
    assert screen == [[0], SCREENED, SCREENED]
    store = hz.CanaryStore(screen_world[0] / "store")
    assert store.quarantined(ARCH) == {}
    passed = {r["name"] for r in store.records() if r.get("verdict") == "pass"}
    names = _names()
    assert passed == {names[s] for s in ("1", "2", "3", "4", "0", "5")}
    monkeypatch.setattr(hz, "stuck_processes", lambda: [])
    calls = len(_bench_calls(screen_world))
    again = hz.run_canary(_spec(screen_world, cfg))
    assert again["rc"] == 0
    assert [_listed(c) for c in _bench_calls(screen_world)[calls:]] == [
        [0],
        [7, 8],
        [6],
        [0],
    ]
    sc = again["screen"]
    assert (sc["passed_before"], sc["swept_before"], sc["screened"]) == (2, 1, 3)
    assert (sc["passed"], sc["quarantined"], sc["stops"]) == (2, 1, [])
    assert [(s["solution"], s["alone"][0]["verdict"]) for s in sc["suspects"]] == [
        (6, "hang")
    ]
    held = hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH)
    assert list(held) == [names["6"]] and held[names["6"]]["verdict"] == "hang"


def test_a_screen_bench_stuck_in_the_driver_is_left_running_and_stops_the_screen(
    screen_world, config, monkeypatch
):
    from lib import procs

    left = []

    def stuck(proc):
        """The first bench killed stays stuck in the driver."""
        if not left:
            left.append(proc)
        return [proc.pid] if proc is left[0] else []

    monkeypatch.setattr(procs, "stuck_members", stuck)
    monkeypatch.setattr(procs, "STUCK_WAIT_S", 0.3)
    monkeypatch.setenv("FAKE_CANARY", json.dumps({"6@700x500x300x2": "hang"}))
    try:
        summary = hz.run_canary(_spec(screen_world, _screened(config)))
        assert summary["rc"] == 3 and left[0].poll() is None
        (stopped,) = summary["screen"]["stops"]
        suspect = stopped["suspect"]
        assert (suspect["solution"], suspect["name"]) == (6, _names()["6"])
        assert stopped["log"] == str(screen_world[0] / "out/logs/screen1_p0.log")
        assert summary["stopped"] == (
            "device 0: a killed bench of the hang screen on 700x500x300x2 did not "
            "end; it stayed stuck in the GPU driver and is left running, not "
            "killed; the first kernel its line lists after its last row, solution "
            f"6 ({suspect['kernel']}), runs alone after the problem's other "
            f"kernels next time (its log is {stopped['log']})"
        )
        assert hz.CanaryStore(screen_world[0] / "store").quarantined(ARCH) == {}
    finally:
        for p in left:
            p.kill()
            p.wait()


def test_the_stage_reports_the_screen(screen_world, config, capsys):
    import yaml

    import stage00a_hazard_canary as s00a

    path = screen_world[0] / "cfg.yaml"
    path.write_text(yaml.safe_dump(_screened(config)))
    argv = [
        "--config-yaml",
        str(path),
        "--dat-dir",
        str(screen_world[1]),
        "--bench-binary",
        str(screen_world[2]),
        "--startup-grace-s",
        "0.5",
        "--out-dir",
        str(screen_world[0] / "a"),
    ]
    assert s00a.main(argv) == 0
    assert "5 other kernels on 2 problems" in capsys.readouterr().out


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


def test_the_screen_runs_the_canary_for_a_pool_without_hazard_classes(
    tmp_path, stubs, env
):
    """A pool of no hazard class runs no canary, unless the hang screen is
    on: then stage00a runs first, and the invocation names its blocklist."""
    import yaml

    import run_orchestrator as ro
    from test_run_orchestrator import STEM, arg, calls, make_config, run

    def config(**more):
        over = {"full.total_rounds": 1, "planner.enabled": True, **more}
        path = make_config(tmp_path, **over)
        cfg, _ = ro.load_config(path)
        paths = ro.derive_paths(cfg)
        paths.library_dir.mkdir(parents=True, exist_ok=True)
        bf.write_logic(paths.library_dir / f"{STEM}.dat", [bf.solution(0, 0)], [0])
        return path, ro.canary_needed(cfg, paths)

    path, needed = config()
    assert not needed
    assert run(stubs, path, tmp_path / "off", "--mode", "initial") == 0
    assert calls(env)[0]["stage"] == "stage01"
    seen = len(calls(env))
    path, needed = config(**{"hazards.screen": {"enabled": True, "per_cell": 2}})
    assert needed
    assert run(stubs, path, tmp_path / "on", "--mode", "initial") == 0
    canary = calls(env)[seen]
    assert canary["stage"] == "stage00a"
    snapshot = yaml.safe_load(Path(arg(canary, "--config-yaml")).read_text())
    assert snapshot["hazards"]["screen"] == {"enabled": True, "per_cell": 2}
    assert snapshot["correctness"]["blocklist"]


@pytest.mark.parametrize("hazard_listed", [False, True])
def test_a_store_quarantine_runs_the_canary_with_the_screen_off(
    tmp_path, stubs, env, monkeypatch, hazard_listed
):
    """A kernel of no hazard class that a hang screen quarantined in the
    run's store stays out of an invocation with the screen off whose pool
    has no kernel of a hazard class, or one the lists name: stage00a runs
    first and puts it into the invocation's blocklist, which the stages'
    pool applies."""
    import yaml

    import run_orchestrator as ro
    from test_run_orchestrator import STEM, arg, calls, make_config, run

    risky = _name((128, 64, 64), cd=(2, 4))
    sols = [bf.solution(0, 0), bf.solution(1, 1, mt=(256, 64, 64))]
    if hazard_listed:
        sols.append(bf.solution(2, 2, mt=(128, 64, 64)))
        sols[2]["name"] = risky
    lst = tmp_path / "known.tsv"
    lst.write_text(f"X\t{risky}\n")
    over = {"full.total_rounds": 1, "planner.enabled": True}
    path = make_config(tmp_path, **over, **{"correctness.lists": [str(lst)]})
    cfg, _ = ro.load_config(path)
    paths = ro.derive_paths(cfg)
    paths.library_dir.mkdir(parents=True, exist_ok=True)
    bf.write_logic(paths.library_dir / f"{STEM}.dat", sols, [s["index"] for s in sols])
    plain = [
        k["kernel_name"]
        for k in ro.load_library_kernels(paths.library_dir, STEM)
        if not hz.hazard_classes(k)
    ]
    run_dir = tmp_path / "run"
    assert not ro.canary_needed(cfg, paths, run_dir)
    found = _rec("end", "t", plain[1], "hang", screen={"shape": "8192x8192x8192x1"})
    found["arch"] = hz.config_arch(cfg)
    hz.CanaryStore(ro.hazard_store_dir(run_dir)).add([found])
    assert ro.canary_needed(cfg, paths, run_dir)
    monkeypatch.setenv("STUB_QUARANTINED", json.dumps([plain[1]]))
    assert run(stubs, path, run_dir, "--mode", "initial") == 0
    seen = calls(env)
    assert seen[0]["stage"] == "stage00a"
    snapshot = yaml.safe_load(Path(arg(seen[0], "--config-yaml")).read_text())
    assert cx.read_list(snapshot["correctness"]["blocklist"]).names == {plain[1]}
    pool = ro.bank_pool.config_pool(snapshot, paths.library_dir, STEM, 0)
    assert [k["kernel_name"] for k in pool.kernels] == [plain[0]]


def test_an_unreadable_store_runs_the_canary(tmp_path, stubs, env, capsys):
    """A store that cannot be read (`hazards.cache_dir`) may hold kernels
    that only stage00a puts into the blocklist, so the invocation runs the
    canary, with a warning, even when its pool has no kernel of a hazard
    class, instead of benching them or failing before its pin is settled."""
    import run_orchestrator as ro
    from test_run_orchestrator import STEM, calls, make_config, run

    store = tmp_path / "store"
    store.mkdir()
    store.chmod(0)
    try:
        if os.access(store, os.R_OK | os.X_OK):
            pytest.skip("the test's user reads a directory without permissions")
        over = {"full.total_rounds": 1, "hazards.cache_dir": str(store)}
        path = make_config(tmp_path, **over)
        cfg, _ = ro.load_config(path)
        paths = ro.derive_paths(cfg)
        paths.library_dir.mkdir(parents=True, exist_ok=True)
        bf.write_logic(paths.library_dir / f"{STEM}.dat", [bf.solution(0, 0)], [0])
        warned = f"cannot read the hazard canary's store {store}"
        assert ro.canary_needed(cfg, paths)
        assert warned in capsys.readouterr().err
        assert run(stubs, path, tmp_path / "run", "--mode", "initial") == 0
        assert warned in capsys.readouterr().err
        assert calls(env)[0]["stage"] == "stage00a"
    finally:
        store.chmod(0o700)


def test_the_canary_stops_on_a_store_it_cannot_read(world, config):
    """stage00a names a store it cannot read and stops, blaming no kernel
    and benching nothing."""
    store = world[0] / "store"
    store.mkdir()
    store.chmod(0)
    try:
        if os.access(store, os.R_OK | os.X_OK):
            pytest.skip("the test's user reads a directory without permissions")
        with pytest.raises(hz.CanaryError, match="cannot read the canary's store"):
            hz.run_canary(_spec(world, config))
        assert _bench_calls(world) == []
    finally:
        store.chmod(0o700)


def test_the_screen_is_pinned_only_while_it_is_on(tmp_path):
    """Off, the screen's keys change nothing, which a pin written before
    they existed means; on, they decide which kernels leave the pool and
    are pinned, but its patience is not, nor any of them in an audit
    directory."""
    import yaml

    import run_orchestrator as ro
    from test_run_orchestrator import make_config

    cfg = yaml.safe_load(make_config(tmp_path).read_text())
    plain = ro.pinned_config(cfg)
    assert not [k for k in plain if k.startswith("hazards.screen")]
    off = {"enabled": False, "shapes": [[64, 64, 64]], "per_cell": 3, "seed": 5}
    assert ro.pinned_config(dict(cfg, hazards={"screen": off})) == plain
    on = dict(cfg, hazards={"screen": {"enabled": True, "stall_s": 30.0}})
    pinned = ro.pinned_config(on)
    assert {k: v for k, v in pinned.items() if k.startswith("hazards.screen")} == {
        "hazards.screen.enabled": True,
        "hazards.screen.shapes": [],
        "hazards.screen.per_cell": 1,
        "hazards.screen.seed": 0,
    }
    patient = dict(cfg, hazards={"screen": {"enabled": True, "stall_s": 600.0}})
    assert ro.pinned_config(patient) == pinned
    assert not [k for k in ro.pinned_config(on, audit=True) if k.startswith("hazards")]


def test_the_config_check_refuses_a_screen_that_cannot_run(tmp_path):
    import yaml

    import run_orchestrator as ro
    from test_run_orchestrator import make_config

    base = yaml.safe_load(make_config(tmp_path).read_text())

    def errors(**hazards):
        return ro.validate_config(dict(base, hazards=hazards))[0]

    assert errors(screen={"enabled": True}) == []
    assert (
        errors(screen={"enabled": True, "shapes": [[7, 194, 895, 8]], "per_cell": 0})
        == []
    )
    assert errors(enabled=False, screen={"enabled": True}) == [
        "hazards.screen.enabled: the hang screen runs in the hazard canary's stage, "
        "which hazards.enabled false turns off"
    ]
    assert errors(screen={"enabled": True, "per_cell": 0}) == [
        "hazards.screen: no problem to run the screen on; give shapes or a "
        "per_cell above 0"
    ]
    assert errors(screen={"shapes": [[1, 2, 3, 4, 5]]}) == [
        "hazards.screen.shapes[0]: expected [M, N, K] or [M, N, K, batch], got "
        "[1, 2, 3, 4, 5]"
    ]
    assert errors(screen={"shapes": [[1, 2]]}) == [
        "hazards.screen.shapes[0]: needs at least 3 entries"
    ]
    assert errors(screen={"stall_s": 0}) == [
        "hazards.screen.stall_s: must be > 0, got 0"
    ]
