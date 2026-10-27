# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
import threading
import time

import pytest

import gpu_fakes as gf
from lib import gpu_health as gh
from lib import kfd
from lib.bench_yaml import shape_of_line


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    root = gf.write_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    return root


def _dev(root, i):
    return root / "class" / "drm" / f"renderD{gf.GPUS[i].render_minor}" / "device"


def _health_files(root, i, *, features="enabled", ue=(0, 0)):
    d = _dev(root, i)
    (d / "pp_features").write_text(
        "features high: 0x0 low: 0x1\nNo. Feature  Bit : State\n"
        f"00. DATA_CALCULATIONS ( 0) : {features}\n01. DPM_FCLK ( 1) : disabled\n"
    )
    (d / "ras").mkdir(exist_ok=True)
    (d / "ras" / "aca_umc").write_text(f"ue: {ue[0]}\nce: 7\nde: 0\n")
    (d / "ras" / "aca_gfx").write_text(f"ue: {ue[1]}\nce: 0\n")
    (d / "ras" / "features").write_text("feature mask: 0x1\n")


def test_readings_come_from_the_gpus_sysfs(sysfs):
    _health_files(sysfs, 0, ue=(2, 3))
    g = kfd.gpus()[0]
    assert gh.read_gpu(g) == {
        "power_management": True,
        "perf_level": "auto",
        "power_cap_w": 100.0,
        "ecc_uncorrectable": 5,
    }
    _health_files(sysfs, 0, features="disabled")
    assert gh.read_gpu(g)["power_management"] is False
    bare = gh.read_gpu(kfd.gpus()[1])
    assert bare["power_management"] is None and bare["ecc_uncorrectable"] is None


def test_a_reading_that_fails_says_so(sysfs, monkeypatch):
    real = gh._value

    def busy(path):
        if path.name == "power_dpm_force_performance_level":
            return None, "unreadable (EBUSY)"
        return real(path)

    monkeypatch.setattr(gh, "_value", busy)
    assert gh.read_gpu(kfd.gpus()[0])["perf_level"] == "unreadable (EBUSY)"


def test_departures_from_the_baseline():
    base = {
        "power_management": True,
        "perf_level": "auto",
        "power_cap_w": 100.0,
        "ecc_uncorrectable": 0,
    }
    assert gh.changes(base, dict(base)) == []
    now = {
        "power_management": False,
        "perf_level": "manual",
        "power_cap_w": None,
        "ecc_uncorrectable": 4,
    }
    got = dict(gh.changes(base, now))
    assert set(got) == set(gh.CHECKS)
    assert got["power_management"].startswith("power management is off")
    assert "performance level 'manual' ('auto' when" in got["perf_level"]
    assert got["power_cap_w"].startswith("power cap None (100 W when")
    assert "uncorrectable ECC errors 4 (0 when" in got["ecc_uncorrectable"]
    unknown = {k: None for k in gh.CHECKS}
    assert gh.changes(unknown, dict(now, power_management=None)) == []
    assert (
        gh.changes(dict(base, power_cap_w=100.0), dict(base, power_cap_w=100.3)) == []
    )


def test_power_management_off_departs_whatever_the_baseline_holds():
    off = {"power_management": False}
    for base in (off, {"power_management": None}, {}):
        ((check, what),) = gh.changes(base, off)
        assert check == "power_management"
        assert what.startswith("power management is off (off too when the run first")
    assert gh.changes({"power_management": True}, {"power_management": True}) == []
    assert gh.changes(off, {"power_management": True}) == []
    assert gh.changes(off, {"power_management": None}) == []


def test_the_baseline_is_recorded_once_per_gpu(sysfs, tmp_path):
    g0, g1, _ = kfd.gpus()
    guard = gh.Guard(tmp_path / "guard", "s1")
    first = guard.baseline([g0])
    (_dev(sysfs, 0) / "power_dpm_force_performance_level").write_text("manual\n")
    again = gh.Guard(tmp_path / "guard", "s2").baseline([g0, g1])
    assert again[gh.gpu_key(g0)] == first[gh.gpu_key(g0)]
    assert again[gh.gpu_key(g0)]["perf_level"] == "auto"
    assert gh.gpu_key(g1) in again and gh.gpu_key(g0) == g0.bdf


def test_a_check_confirms_a_departure_before_it_counts(sysfs, tmp_path):
    g = kfd.gpus()[0]
    guard = gh.Guard(tmp_path / "guard", "s")
    guard.baseline([g])
    level = _dev(sysfs, 0) / "power_dpm_force_performance_level"
    level.write_text("manual\n")

    def back(_s):
        level.write_text("auto\n")

    assert gh.check(guard, [g], ttm=lambda: [], sleep=back).problems == []
    level.write_text("manual\n")
    got = gh.check(guard, [g], ttm=lambda: [], sleep=lambda _s: None)
    assert got.keys == [g.bdf, gh.HOST]
    assert got.problems == [
        f"GPU 0 ({g.bdf}): performance level 'manual' ('auto' when the run started)"
    ]
    assert got.found[g.bdf][0][0] == "perf_level"
    level.write_text("auto\n")
    stuck = gh.check(guard, [g], ttm=lambda: ["kworker/u1:2+ttm"] * 5)
    assert stuck.problems == [
        "5 kernel worker(s) of the ttm memory manager stuck in D state "
        "(kworker/u1:2+ttm, kworker/u1:2+ttm, kworker/u1:2+ttm and 2 more), as "
        "after a GPU reset"
    ]
    assert list(stuck.found) == [gh.HOST]
    calls = []
    off = gh.check(guard, [g], host=False, ttm=lambda: calls.append(1) or ["x"])
    assert off.problems == [] and calls == [] and gh.HOST in off.keys


def _kworker(root, pid, comm, state):
    d = root / str(pid)
    d.mkdir(exist_ok=True)
    (d / "stat").write_text(f"{pid} ({comm}) {state} 2 0 0 0 0 0 0 0\n")


def test_only_ttm_workers_in_d_state_in_every_reading_count(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setenv("TILEWRIGHT_PROC_ROOT", str(root))
    _kworker(root, 5, "kworker/u9:1+ttm", "D")
    _kworker(root, 6, "kworker/u9:2+events", "D")
    _kworker(root, 7, "kworker/u9:3+ttm", "S")
    assert gh.ttm_workers_in_d_state() == ["kworker/u9:1+ttm"]
    assert gh.stuck_ttm_workers(sleep=lambda _s: None) == ["kworker/u9:1+ttm"]
    seen = []

    def clear_after_three(_s):
        seen.append(1)
        if len(seen) == 3:
            _kworker(root, 5, "kworker/u9:1+ttm", "S")

    assert gh.stuck_ttm_workers(sleep=clear_after_three) == []
    assert len(seen) == 3


def _found(keys, **found):
    return gh.Check(
        list(keys), {k: [("perf_level", t) for t in v] for k, v in found.items()}
    )


def test_the_hold_keeps_each_gpus_problems_until_a_check_of_that_gpu_passes(
    tmp_path,
):
    a = gh.Guard(tmp_path, "s", by="a")
    b = gh.Guard(tmp_path, "s", by="b")
    assert a.hold() is None and a.release(None, "x") == ("", None)
    assert a.settle(_found(["g0", gh.HOST], g0=["x"])) == ("raised", a.hold())
    assert b.settle(_found(["g1", gh.HOST])) == ("held", b.hold())
    what, hold = b.settle(_found(["g1", gh.HOST], g1=["y"]))
    assert what == "added" and hold["problems"] == ["x", "y"]
    assert set(hold["entries"]) == {"g0", "g1"} and hold["by"] == "a"
    assert hold["entries"]["g1"]["by"] == "b"
    assert a.settle(_found(["g0", gh.HOST], g0=["x2"]))[0] == "held"
    assert a.hold()["problems"] == ["x2", "y"]
    assert a.settle(_found(["g0", gh.HOST]))[0] == "held"
    what, done = b.settle(_found(["g1", gh.HOST]))
    assert what == "cleared" and done["cleared_by"] == "b" and a.hold() is None
    doc = a.read()
    assert doc["holds"][0]["raised"] == done["raised"]
    assert doc["last_cleared"] == done["cleared"]


def test_a_departure_holds_its_gpu_alone_unless_every_gpu_read_departs(tmp_path):
    a = gh.Guard(tmp_path, "s", by="a")
    what, hold = a.settle(_found(["g0", "g1", gh.HOST], g0=["x"]))
    assert what == "raised" and not hold["host_wide"] and not gh.run_wide(hold)
    assert gh.holds(hold, "g0") and not gh.holds(hold, "g1")
    assert gh.holds(hold, None) and gh.held_keys(hold) == ["g0"]
    assert "This holds g0 alone" in gh.hold_text(hold, tmp_path / "guard.json")
    what, hold = a.settle(_found(["g0", "g1", gh.HOST], g0=["x"], g1=["y"]))
    assert what == "added" and hold["host_wide"] and gh.run_wide(hold)
    assert gh.holds(hold, "g7")
    text = gh.hold_text(hold, tmp_path / "guard.json")
    assert "This holds every GPU of the run" in text
    one = a.settle(_found(["g1", "g0", gh.HOST], g1=["y"]))[1]
    assert not gh.run_wide(one) and gh.held_keys(one) == ["g1"]
    alone = a.settle(_found(["g1", gh.HOST], g1=["y"]))[1]
    assert not alone["host_wide"], "one GPU read is never every GPU"
    host = a.settle(gh.Check(["g1", gh.HOST], {gh.HOST: [("ttm", "stuck")]}))[1]
    assert gh.run_wide(host) and gh.holds(host, "g5")
    assert not gh.run_wide(None) and not gh.holds(None, "g0")
    assert gh.run_wide({"raised": "t", "problems": ["old"]})


def test_a_gpu_reads_its_canary_again_once_the_run_stops_holding_it(tmp_path):
    a = gh.Guard(tmp_path, "s", by="a")
    for key in ("g0", "g1"):
        a.record_canary(key, {"passed": True})
    a.settle(_found(["g0", "g1", gh.HOST], g0=["x"], g1=["y"]))
    assert not a.canary_due("g0") and not a.canary_due("g1")
    time.sleep(1.1)
    a.settle(_found(["g0", "g1", gh.HOST], g1=["y"]))
    released = a.read()["released"]
    assert set(released) == {gh.EVERY_GPU, "g0"} and a.hold() is not None
    assert a.canary_due("g0") and a.canary_due("g1")
    a.record_canary("g0", {"passed": True})
    assert not a.canary_due("g0")
    time.sleep(1.1)
    assert a.settle(_found(["g0", "g1", gh.HOST]))[0] == "cleared"
    assert not a.canary_due("g0"), "g0 was not held when the hold was cleared"
    assert a.canary_due("g1")
    old = gh.Guard(tmp_path / "old", "s")
    old.record_canary("g0", {"passed": True})
    time.sleep(1.1)
    with old.update() as doc:
        doc["last_cleared"] = gh.utc()
    assert old.canary_due("g0"), "a record without release times"


def test_released_lines_stay_apart_from_the_times_holds_end(sysfs, tmp_path):
    g0, g1, _ = kfd.gpus()
    for i in (0, 1):
        _health_files(sysfs, i)
    guard = gh.Guard(tmp_path, "s", by="a")
    guard.baseline([g0, g1])
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)
    guard.quarantine_lines([{"shape": [1, 2, 3, 1]}, {"shape": [4, 5, 6, 1]}])
    gone, kept = guard.release_lines([(1, 2, 3, 1)])
    assert [e["shape"] for e in gone] == [[1, 2, 3, 1]] and kept == 1
    doc = guard.read()
    assert "released" not in doc and doc[gh.RELEASED_LINES][0]["released"]
    _health_files(sysfs, 0, features="disabled")
    assert guard.settle(gh.check(guard, [g0, g1], **quick))[0] == "raised"
    _health_files(sysfs, 0)
    assert guard.settle(gh.check(guard, [g0, g1], **quick))[0] == "cleared"
    assert set(guard.read()["released"]) == {g0.bdf}
    gone, kept = guard.release_lines(None)
    assert [e["shape"] for e in gone] == [[4, 5, 6, 1]] and kept == 0
    doc = guard.read()
    assert set(doc["released"]) == {g0.bdf} and guard.quarantined_shapes() == set()
    assert [e["shape"] for e in gh.released_lines(doc)] == [[1, 2, 3, 1], [4, 5, 6, 1]]


def test_released_lines_kept_under_released_are_read_as_lines(tmp_path):
    import json

    guard = gh.Guard(tmp_path, "s", by="a")
    guard.record_canary("g0", {"passed": True})
    doc = guard.read()
    doc["released"] = [{"shape": [1, 2, 3, 1], "released": "t"}, "g1"]
    guard.path.write_text(json.dumps(doc))
    assert not guard.canary_due("g0"), "no time a hold of g0 ended"
    assert gh.released_lines(guard.read()) == [{"shape": [1, 2, 3, 1], "released": "t"}]
    assert guard.settle(_found(["g0", gh.HOST], g0=["x"]))[0] == "raised"
    doc = guard.read()
    assert "released" not in doc
    assert doc[gh.RELEASED_LINES] == [{"shape": [1, 2, 3, 1], "released": "t"}]
    time.sleep(1.1)
    assert guard.settle(_found(["g0", gh.HOST]))[0] == "cleared"
    assert set(guard.read()["released"]) == {"g0"} and guard.canary_due("g0")
    assert guard.release(None, "x") == ("", None)


def test_any_check_settles_the_hosts_problems_and_an_unkeyed_hold(tmp_path):
    a = gh.Guard(tmp_path, "s", by="a")
    a.settle(gh.Check(["g0", gh.HOST], {gh.HOST: [("ttm", "stuck")]}))
    text = gh.hold_text(a.hold(), tmp_path / "guard.json")
    assert "the ttm workers are the host's" in text and "bench.health.ttm" in text
    assert a.settle(_found(["g5", gh.HOST]))[0] == "cleared"
    with a.update() as doc:
        doc["hold"] = {"raised": "t", "seen": "t", "by": "old", "problems": ["p"]}
    assert gh.hold_entries(a.hold()) == {
        gh.UNKEYED: {
            "problems": ["p"],
            "checks": [],
            "raised": "t",
            "seen": "t",
            "by": "old",
        }
    }
    assert a.settle(_found(["g0", gh.HOST], g0=["q"]))[0] == "added"
    assert a.settle(_found(["g0", gh.HOST]))[0] == "cleared"


def test_a_check_reads_the_gpus_the_hold_names_and_lifts_it_for_them(sysfs, tmp_path):
    g0, g1, _ = kfd.gpus()
    for i in (0, 1):
        _health_files(sysfs, i)
    a = gh.Guard(tmp_path, "s", by="a")
    b = gh.Guard(tmp_path, "s", by="b")
    a.baseline([g0, g1])
    _health_files(sysfs, 0, features="disabled")
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)
    assert a.settle(gh.check(a, [g0], **quick))[0] == "raised"
    got = gh.check(b, [g1], **quick)
    assert got.keys == [g1.bdf, g0.bdf, gh.HOST]
    what, hold = b.settle(got)
    assert what == "held" and set(hold["entries"]) == {g0.bdf}
    text = gh.hold_text(hold, tmp_path / "guard.json")
    assert "power management is off (on when the run started)" in text
    assert "ends only when it is on again" in text
    _health_files(sysfs, 0)
    assert b.settle(gh.check(b, [g1], **quick))[0] == "cleared"


B_CHECKS_GPU_1 = """
import json, sys
sys.path.insert(0, sys.argv[1])
from lib import gpu_health as gh, kfd
guard = gh.Guard(sys.argv[2], "s", by="b")
g1 = kfd.gpus()[1]
what, _ = guard.settle(gh.check(guard, [g1], ttm=lambda: [], sleep=lambda s: None))
print(json.dumps([what, sorted(gh.hold_entries(guard.hold()))]))
"""


def test_a_process_that_checks_another_gpu_keeps_the_hold_of_this_one(
    sysfs, tmp_path, pipeline_dir
):
    import json
    import subprocess
    import sys

    g0, g1, _ = kfd.gpus()
    for i in (0, 1):
        _health_files(sysfs, i)
    a = gh.Guard(tmp_path / "guard", "s", by="a")
    a.baseline([g0, g1])
    _health_files(sysfs, 0, features="disabled")
    assert a.settle(gh.check(a, [g0], ttm=lambda: [], sleep=lambda _s: None))[0]

    def b():
        done = subprocess.run(
            [sys.executable, "-c", B_CHECKS_GPU_1, str(pipeline_dir), str(a.dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(done.stdout)

    assert b() == ["held", [g0.bdf]]
    assert a.hold()["entries"][g0.bdf]["by"] == "a"
    _health_files(sysfs, 0)
    assert b() == ["cleared", []]
    assert a.hold() is None and a.read()["holds"][-1]["cleared_by"] == "b"


def test_the_hold_of_an_excluded_gpu_is_dropped(sysfs, tmp_path):
    g0, g1, _ = kfd.gpus()
    guard = gh.Guard(tmp_path, "s")
    guard.settle(_found([g0.bdf, g1.bdf, gh.HOST], **{g0.bdf: ["x"], g1.bdf: ["y"]}))
    assert gh.release_excluded(guard, []) == []
    assert gh.release_excluded(guard, ["0b:00.0"]) == [g0.bdf]
    assert set(guard.hold()["entries"]) == {g1.bdf}
    assert gh.release_excluded(guard, [1]) == [g1.bdf] and guard.hold() is None
    guard.settle(_found(["0000:4c:00.0", gh.HOST], **{"0000:4c:00.0": ["gone"]}))
    assert gh.release_excluded(guard, ["4C:00.0"]) == ["0000:4c:00.0"]
    assert guard.hold() is None


def test_wait_healthy_holds_until_the_checks_pass(tmp_path):
    guard = gh.Guard(tmp_path, "s")
    cap = _found(["g0", gh.HOST], g0=["GPU 0: power cap 750 W"])
    answers = [cap, cap, _found(["g0", gh.HOST])]
    said = []
    ok = gh.wait_healthy(
        guard,
        [],
        gh.Settings(poll_s=1.0),
        report=said.append,
        sleep=lambda _s: None,
        checker=lambda: answers.pop(0),
    )
    assert ok and guard.hold() is None
    assert said[0].startswith("the run's GPUs are not as they were")
    assert "scripts/run_guard.py accept" in said[0]
    assert said[-1].startswith("the run's GPUs pass their checks again")
    assert len(guard.read()["holds"]) == 1

    clock = iter(range(0, 10_000, 10))
    assert not gh.wait_healthy(
        guard,
        [],
        gh.Settings(max_wait_s=25.0),
        report=lambda _m: None,
        sleep=lambda _s: None,
        clock=lambda: float(next(clock)),
        checker=lambda: _found(["g0", gh.HOST], g0=["still"]),
    )
    assert guard.hold()["problems"] == ["still"]
    other = gh.Guard(tmp_path, "s", by="other")
    assert not gh.wait_healthy(
        other,
        [],
        gh.Settings(max_wait_s=25.0),
        report=lambda _m: None,
        sleep=lambda _s: None,
        clock=lambda: float(next(clock)),
        checker=lambda: _found(["g1", gh.HOST]),
    )
    assert guard.hold()["problems"] == ["still"]


def test_wait_healthy_goes_on_beside_a_hold_that_does_not_hold_its_gpus(tmp_path):
    guard = gh.Guard(tmp_path, "s")
    said = []

    def waits(checker, key):
        return gh.wait_healthy(
            guard,
            [],
            gh.Settings(max_wait_s=25.0),
            report=said.append,
            sleep=lambda _s: None,
            clock=iter(range(0, 10_000, 10)).__next__,
            checker=checker,
            blocks=lambda hold: gh.holds(hold, key),
        )

    other = _found(["g0", "g1", gh.HOST], g1=["GPU 1: power cap 750 W"])
    assert waits(lambda: other, "g0")
    assert len(said) == 1 and "This holds g1 alone" in said[0]
    assert guard.hold() is not None
    assert not waits(lambda: other, "g1")
    every = _found(["g0", "g1", gh.HOST], g0=["a"], g1=["b"])
    assert not waits(lambda: every, "g0")
    assert waits(lambda: _found(["g0", "g1", gh.HOST]), "g0")
    assert guard.hold() is None


def test_the_canary_setup_and_references_are_first_come(tmp_path):
    a = gh.Guard(tmp_path, "s1", by="a")
    b = gh.Guard(tmp_path, "s1", by="b")
    setup = a.set_canary_setup(["- {M: 1}\n"], gh.CanarySettings(), "x")
    assert setup["rule"] == gh.CANARY_RULE and setup["tolerance"] == 0.05
    assert (
        b.set_canary_setup(["- {M: 2}"], gh.CanarySettings(tolerance=0.5), "y") == setup
    )
    assert a.canary_due("k")
    rec = a.record_canary(
        "k", {"reference": True, "passed": True, "readings_us": [1.0], "taken": "t1"}
    )
    assert rec["reference"] and a.reference("k")["readings_us"] == [1.0]
    late = b.record_canary(
        "k", {"reference": True, "passed": True, "readings_us": [9.0]}
    )
    assert late["reference"] is False and a.reference("k")["readings_us"] == [1.0]
    assert not a.canary_due("k")
    assert gh.Guard(tmp_path, "s2").canary_due("k")
    a.settle(_found(["k", gh.HOST], k=["p"]))
    time.sleep(1.1)
    assert a.release(None, "test")[0] == "cleared"
    assert a.canary_due("k")
    stale = "t0 a"
    again = a.record_canary(
        "k",
        {"reference": True, "passed": True, "readings_us": [5.0], "replaces": stale},
    )
    assert again["reference"] is False and a.reference("k")["readings_us"] == [1.0]
    assert gh.record_id(a.reference("k")) == "t1 a"
    anew = a.record_canary(
        "k",
        {"reference": True, "passed": True, "readings_us": [5.0], "replaces": "t1 a"},
    )
    assert anew["reference"] and a.reference("k")["readings_us"] == [5.0]


def test_a_canary_setup_the_library_cannot_time_is_chosen_anew(tmp_path):
    a = gh.Guard(tmp_path, "s1", by="a")
    first = a.set_canary_setup(["l1", "l2"], gh.CanarySettings(), "x")
    a.record_canary("k", {"reference": True, "readings_us": [1.0], "taken": "t1"})
    old = gh.record_id(first)
    assert a.set_canary_setup(["l3"], gh.CanarySettings(), "y", replaces="other") == (
        first
    )
    new = a.set_canary_setup(["l3"], gh.CanarySettings(), "y", replaces=old)
    assert new["lines"] == ["l3"] and new["replaces"] == old
    assert gh.record_id(new) != old
    assert a.read()["canary"]["setups"] == [first] and a.reference("k") is None
    again = gh.Guard(tmp_path, "s1", by="b").set_canary_setup(
        ["l4"], gh.CanarySettings(), "z", replaces=old
    )
    assert again == new


def test_quarantined_lines_are_kept_for_the_run(tmp_path):
    guard = gh.Guard(tmp_path, "s")
    assert guard.quarantined_shapes() == set()
    guard.quarantine_lines([{"shape": [1, 2, 3, 4], "in_flight": "kernel X"}])
    guard.quarantine_lines([])
    assert gh.Guard(tmp_path, "t").quarantined_shapes() == {(1, 2, 3, 4)}
    (entry,) = guard.read()["quarantined_lines"]
    assert entry["in_flight"] == "kernel X" and entry["at"]


def test_canary_lines_run_from_the_least_work_to_the_median():
    lines = [f"- {{M: {m}, N: 8, K: 8, batch_count: 1}}\n" for m in range(1, 11)]
    lines.append(lines[0])
    got = gh.choose_canary_lines(
        lines, 3, lambda ln: ln.rstrip("\n") + " #c", shape_of_line
    )
    assert [shape_of_line(g)[0] for g in got] == [1, 3, 5]
    assert all(g.endswith(" #c") for g in got)
    odd = gh.choose_canary_lines(
        lines, 2, lambda ln: None if "M: 1," in ln else ln, shape_of_line
    )
    assert [shape_of_line(g)[0] for g in odd] == [2, 6]
    assert gh.choose_canary_lines(lines[:2], 4, lambda ln: ln, shape_of_line) != []


def test_one_canary_line_is_the_shape_halfway_to_the_median():
    lines = [f"- {{M: {m}, N: 8, K: 8, batch_count: 1}}\n" for m in range(1, 11)]
    (one,) = gh.choose_canary_lines(lines, 1, lambda ln: ln, shape_of_line)
    assert shape_of_line(one)[0] == 3
    (alone,) = gh.choose_canary_lines(lines[:1], 1, lambda ln: ln, shape_of_line)
    assert shape_of_line(alone)[0] == 1
    assert gh.choose_canary_lines(lines, 0, lambda ln: ln, shape_of_line) == []


def test_canary_entries_readings_and_verdict():
    named = gh.canary_entries(["a", "b"], 3, lambda ln, n: f"{ln}:{n}")
    assert named == [
        "a:hcanary0_warm",
        "b:hcanary1_warm",
        "a:hcanary0_t0",
        "b:hcanary1_t0",
        "a:hcanary0_t1",
        "b:hcanary1_t1",
        "a:hcanary0_t2",
        "b:hcanary1_t2",
    ]
    lat = [99.0, 99.0, 10.0, 20.0, 11.0, None, 12.0, 22.0]
    assert gh.canary_readings(lat, 2, 3) == [11.0, 21.0]
    v = gh.judge([10.0, 20.0], [10.4, 20.8], 0.05)
    assert v["passed"] and v["ratio"] == pytest.approx(1.04)
    v = gh.judge([10.0, 20.0], [11.0, 22.0], 0.05)
    assert not v["passed"] and "1.100x its reference" in v["why"]
    v = gh.judge([10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 60.0, None], 0.05)
    assert v["passed"] and v["lines_read"] == 3
    assert gh.judge([10.0, 20.0, 30.0], [None, None, 30.0], 0.05)["passed"] is False
    assert math.isclose(gh.judge([1.0], [1.0 / 1.05], 0.05)["shift"], -math.log(1.05))


def test_settings_come_from_the_config():
    s = gh.Settings.of(
        {"bench": {"health": {"poll_s": 5, "canary": {"lines": 2, "enabled": False}}}}
    )
    assert (s.poll_s, s.canary.lines, s.canary.enabled) == (5.0, 2, False)
    assert gh.Settings.of({}) == gh.Settings() and gh.Settings().ttm
    assert gh.Settings().canary_on and not s.canary_on
    off = gh.Settings.of({"bench": {"health": {"enabled": False, "ttm": False}}})
    assert off.canary.enabled and not off.canary_on and not off.ttm


def test_the_guard_dir_and_session_come_from_the_orchestrator(tmp_path, monkeypatch):
    assert gh.guard_dir(tmp_path / "x") == tmp_path / "x"
    assert gh.session().startswith("pid")
    monkeypatch.setenv(gh.GUARD_DIR_VAR, str(tmp_path / "run" / "guard"))
    monkeypatch.setenv(gh.SESSION_VAR, "20000101_full")
    assert gh.guard_dir(tmp_path / "x") == tmp_path / "run" / "guard"
    assert gh.session() == "20000101_full"


def test_concurrent_writers_keep_every_line(tmp_path):
    def add(i):
        gh.Guard(tmp_path, "s").quarantine_lines([{"shape": [i, 1, 1, 1]}])

    threads = [threading.Thread(target=add, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(gh.Guard(tmp_path, "s").quarantined_shapes()) == 12


def _identity(rows):
    """A LibraryIdentity of library files given as {stem: {index: name}},
    every kernel of signature MT128x128x64."""
    from lib import kernel_identity as ki

    files = []
    for stem, kernels in rows.items():
        entries = [
            {
                "sol_idx_global": i,
                "kernel_name": name,
                "mt_m": 128,
                "mt_n": 128,
                "mt_k": 64,
                "mi_m": 16,
                "mi_n": 16,
                "mi_k": 32,
                "cache_hints_a": 0,
                "cache_hints_b": 0,
            }
            for i, name in kernels.items()
        ]
        files.append((stem, entries, None))
    return ki.identity_of_files(files)


def test_canary_lines_keep_their_kernel_when_the_library_renumbers():
    from lib.bench_yaml import line_solution_list

    lines = [f"- {{M: {m}, solution_index_list: [{i}]}}" for m, i in ((1, 10), (2, 11))]
    lines.append("- {M: 3, algo_method: 1}")
    then = _identity({"pool": {10: "ka", 11: "kb"}, "bank": {12: "kc"}})
    kernels = gh.canary_kernels(lines, line_solution_list, then)
    assert kernels[0]["stem"] == "pool" and kernels[0]["index"] == 10
    assert kernels[0]["signature"] == "MT128x128x64 MI16x16x32 NTA0 NTB0"
    assert kernels[2] is None
    setup = {"lines": lines, "kernels": kernels}

    def restamp(ln, i):
        return ln.replace(str(line_solution_list(ln)[0]), str(i))

    assert gh.canary_lines_now(setup, then, restamp) == (lines, [])
    assert gh.canary_lines_now(setup, None, restamp) == (lines, [])
    moved = _identity({"pool": {20: "ka", 10: "kz"}, "bank": {12: "kc"}})
    got, notes = gh.canary_lines_now(setup, moved, restamp)
    assert got == [lines[0].replace("10", "20"), None, lines[2]]
    assert notes == [
        "MT128x128x64 MI16x16x32 NTA0 NTB0: solution index 10 is 20 now",
        "MT128x128x64 MI16x16x32 NTA0 NTB0: no single solution index of the "
        "library now",
    ]
    bank_only = _identity({"bank": {10: "kz"}})
    assert gh.canary_lines_now(setup, bank_only, restamp) == (lines, [])


def test_a_canary_line_that_timed_another_kernel_is_not_judged():
    names = ["w0", "w1", "a", "b", "a", "b", "a", "x"]
    assert gh.timed_kernels(names, 2, 3) == ["a", None]
    assert gh.timed_kernels([None] * 4, 2, 1) == [None, None]
    v = gh.judge([10.0, 20.0], [10.0, 40.0], 0.05, ["a", "b"], ["a", "c"])
    assert v["passed"] and v["lines_read"] == 1 and v["other_kernels"] == 1
    v = gh.judge([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], 0.05, ["a"] * 3, ["b"] * 3)
    assert not v["passed"] and v["other_kernels"] == 3 and v["shift"] is None
    assert "3 timing another kernel than the reference" in v["why"]
    assert gh.judge([10.0], [10.0], 0.05, None, ["b"])["passed"]


# ── wedges ───────────────────────────────────────────────────────────────────


def _boot(monkeypatch, tmp_path, boot):
    """The host's boot id as gpu_health reads it (a fake /proc)."""
    root = tmp_path / "proc"
    path = root / "sys" / "kernel" / "random" / "boot_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(boot + "\n")
    monkeypatch.setenv("TILEWRIGHT_PROC_ROOT", str(root))


def test_a_wedge_holds_its_gpu_whatever_checks_find_until_the_host_reboots(
    tmp_path, monkeypatch
):
    _boot(monkeypatch, tmp_path, "boot-a")
    guard = gh.Guard(tmp_path / "guard", "s")
    what, hold = guard.hold_wedged({"g0": "GPU 0 (g0) is wedged: pid 7 stuck"})
    assert what == "raised" and gh.wedged_keys(hold) == ["g0"]
    assert hold["entries"]["g0"]["boot_id"] == "boot-a"
    assert not gh.run_wide(hold) and gh.holds(hold, "g0") and not gh.holds(hold, "g1")
    what, hold = guard.settle(_found(["g0", "g1", gh.HOST], g1=["GPU 1: cap"]))
    assert what == "added" and gh.wedged_keys(hold) == ["g0"]
    what, hold = guard.settle(_found(["g0", "g1", gh.HOST]))
    assert what == "held" and gh.held_keys(hold) == ["g0"]
    assert gh.wedge_holds(hold, ["g0"]) and gh.wedge_holds(hold, ["g0", "g1"])
    assert not gh.wedge_holds(hold, ["g0", "g1"], every=True)
    assert not gh.wedge_holds(hold, ["g1"])
    text = gh.hold_text(hold, guard.path)
    assert "until the host reboots" in text and "restore the setting" not in text
    assert text.endswith(
        "when no process stays stuck in the GPU driver, lift the hold with "
        "scripts/run_guard.py accept"
    )
    _boot(monkeypatch, tmp_path, "boot-b")
    what, hold = guard.settle(_found(["g1", gh.HOST]))
    assert what == "cleared" and guard.hold() is None
    assert "g0" in guard.read()["released"]


def test_a_wedged_host_holds_every_gpu_and_keeps_a_gpus_problems(tmp_path):
    guard = gh.Guard(tmp_path, "s")
    guard.settle(_found(["g0", gh.HOST], g0=["GPU 0: power cap 750 W"]))
    _, hold = guard.hold_wedged({"g0": "GPU 0 is wedged", gh.HOST: "the host is"})
    assert gh.run_wide(hold) and gh.wedged_keys(hold) == ["g0", gh.HOST]
    assert hold["entries"]["g0"]["problems"] == [
        "GPU 0: power cap 750 W",
        "GPU 0 is wedged",
    ]
    assert gh.wedge_holds(hold, ["g1"], every=True)
    _, hold = guard.settle(_found(["g0", gh.HOST]))
    assert gh.wedged_keys(hold) == ["g0", gh.HOST]
    _, hold = guard.release(["g0", gh.HOST], by="accept")
    assert hold is not None and hold.get("cleared")


def test_wait_healthy_does_not_wait_for_a_wedge(tmp_path):
    guard = gh.Guard(tmp_path, "s")
    guard.hold_wedged({"g0": "GPU 0 is wedged"})
    slept = []
    ok = gh.wait_healthy(
        guard,
        [],
        gh.Settings(max_wait_s=0.0),
        report=lambda _m: None,
        sleep=slept.append,
        checker=lambda: _found(["g0", gh.HOST]),
        blocks=lambda hold: gh.holds(hold, "g0"),
        wedged=lambda hold: gh.wedge_holds(hold, ["g0"]),
    )
    assert not ok and slept == []


def test_a_canary_floor_passes_a_shift_of_a_few_microseconds():
    v = gh.judge([20.0, 30.0], [25.0, 35.0], 0.05)
    assert not v["passed"] and v["offset_us"] == pytest.approx(5.0)
    assert "beyond the tolerance 0.05" in v["why"] and " us" not in v["why"]
    assert gh.judge([20.0, 30.0], [25.0, 35.0], 0.05, tolerance_us=6.0)["passed"]
    v = gh.judge([20.0, 30.0], [40.0, 50.0], 0.05, tolerance_us=6.0)
    assert not v["passed"] and "and 6 us (+20 us)" in v["why"]
    assert gh.judge([100.0], [104.0], 0.05, tolerance_us=1.0)["passed"]
    s = gh.Settings.of({"bench": {"health": {"canary": {"tolerance_us": 6}}}})
    assert s.canary.tolerance_us == 6.0 and gh.Settings().canary.tolerance_us == 0.0
