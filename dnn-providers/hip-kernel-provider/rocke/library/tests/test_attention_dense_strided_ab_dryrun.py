# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of ``benchmarks/common/attention_dense_strided_ab.py``: ``--list`` and
``--dry-run`` through real worker processes (one per arm tree, no torch, no GPU, no
compiler), the control rule, the exclusion of refused and failed cases, the bar check
in both directions, the exit codes, the checks on a pre-built export, and the layout
model.

The tests that run the driver in a subprocess (``_run``: ``--list``, ``--dry-run``,
``export``) export this rocKE source tree (``platform/python`` beside ``library``),
which the installed CI tree does not have, so they skip there with that reason; the
pure-function tests (layouts, schedule, seeds, reference rows, shape files, spec
identity, kernarg patching, root resolution, export checks) run in the CPU lane. Run
the whole file from the source tree (pytest or ``run_all.py``) after a change to the
harness.

The dry-run hooks the tests use (all addressed to one arm label):
``--dry-run-effect B=0.05`` (synthetic times +5%), ``--dry-run-refuse B=bhsd`` (arm B
refuses a layout or a case id) and ``--dry-run-wrong B=bhsd`` (arm B's output checks
fail). A dry run's bar never reads PASS (its times are made up), so a passing dry run
exits with ``DRY_EXIT``, not 0.
"""

from __future__ import annotations

import base64
import importlib.util
import inspect
import itertools
import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rocke.helpers.split_k import ARCH_NUM_CUS

_LIBROOT = Path(__file__).resolve().parents[1]
_ROCKE = _LIBROOT.parent
_AB_FILE = _LIBROOT / "benchmarks" / "common" / "attention_dense_strided_ab.py"
_SHAPES = _LIBROOT / "benchmarks" / "common" / "attention_dense_strided_ab_shapes.yaml"


def _load_ab():
    spec = importlib.util.spec_from_file_location(
        "attention_dense_strided_ab_under_test", _AB_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load_ab()
stats = ab.load_stats()
DRY_EXIT = stats.EXIT_SYNTHETIC  # a dry run that passes its bar: not a measurement

THREE_SHAPES = """\
shapes:
  - {id: g01, model: m, batch: 1, hq: 32, hkv: 8, d: 128, sq: 1024, sk: 1024,
     mask: causal_tl, dtype: bf16, select: gfx942_dense,
     override: {persistent: false}, layouts: [bshd, bhsd]}
  - {id: g03, model: m, batch: 1, hq: 32, hkv: 8, d: 64, sq: 1024, sk: 1024,
     mask: causal_tl, dtype: bf16, select: gfx942_dense, layouts: [bshd, bhsd]}
  - {id: g05, model: m, batch: 2, hq: 32, hkv: 32, d: 64, sq: 1024, sk: 1024,
     mask: none, dtype: fp16, select: gfx942_dense, layouts: [bshd, bhsd]}
"""


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROCKE / "platform" / "python"), str(_LIBROOT)]
    )
    env["ROCKE_CPP_QUIET_FALLBACK"] = "1"
    env["GIT_NO_LAZY_FETCH"] = "1"
    return env


def _run(args, check=None):
    """The driver in a subprocess; it exports this rocKE source tree, so it skips in
    the installed CI tree (no ``platform/python`` beside ``library``)."""
    if not (_ROCKE / "platform" / "python").is_dir():
        pytest.skip("the driver exports a rocKE source tree; this is an installed one")
    run = subprocess.run(
        [sys.executable, str(_AB_FILE), *args],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=600,
    )
    if check is not None:
        assert run.returncode == check, run.stdout[-3000:] + run.stderr[-3000:]
    return run


def _arms(*labels):
    return [x for label in labels for x in ("--arm", f"{label}={_ROCKE}")]


@pytest.fixture(scope="module")
def shapes3(tmp_path_factory):
    """The three-shape file (two layouts each) the dry runs read."""
    path = tmp_path_factory.mktemp("ab_shapes") / "three.yaml"
    path.write_text(THREE_SHAPES)
    return path


_RUNS: dict = {}


def _dry(shapes3, tmp_path_factory, name, *extra):
    """One dry run (cached per flag set): arms A, B and control A2, all the current
    tree."""
    key = (name, extra)
    if key not in _RUNS:
        out = tmp_path_factory.mktemp(f"ab_{name}") / "r.yaml"
        run = _run(
            [
                "--arch",
                "gfx942",
                "--shapes",
                str(shapes3),
                *_arms("A", "B"),
                "--baseline",
                "A",
                "--control",
                "A2",
                "--dry-run",
                "--rounds",
                "24",
                "--n-boot",
                "300",
                "--out",
                str(out),
                *extra,
            ]
        )
        _RUNS[key] = (run, out)
    return _RUNS[key]


# ---- --list ------------------------------------------------------------------------ #


def test_list_shipped_shapes_with_two_arm_trees_has_equal_spec_identity():
    run = _run(
        [
            "--arch",
            "gfx950",
            "--shapes",
            str(_SHAPES),
            *_arms("A", "B"),
            "--baseline",
            "A",
            "--list",
        ],
        check=0,
    )
    out = run.stdout
    assert out.count("spec identity across arms: equal") == 13
    assert (
        "DIFFERS" not in out
        and "EXPECT MISMATCH" not in out
        and "FACTORY ERROR" not in out
    )
    assert (
        "13 cases on layouts ['bshd']; expectation mismatches/errors/identity "
        "differences: 0" in out
    )
    assert (
        "g01/bshd" in out
        and "k03/bshd" in out
        and "variant=grid_kvseg" in out
        and "variant=grid_qbminor" in out
    )


# ---- dry run ----------------------------------------------------------------------- #


def test_dry_run_result_is_read_by_bar_and_by_the_analysis(shapes3, tmp_path_factory):
    run, out = _dry(shapes3, tmp_path_factory, "plain")
    # Every case passes its checks and the ratios are clean, but the times are made up:
    # the bar says so and the exit code is not 0.
    assert run.returncode == DRY_EXIT, run.stdout[-2000:] + run.stderr[-2000:]
    assert run.stdout.count("AB_ROW ") == 6 and "AB_DONE " in run.stdout
    assert "BAR: NOT A MEASUREMENT (dry run)" in run.stdout
    assert "BAR: PASS" not in run.stdout
    result = stats.load_structured(str(out))
    assert result["meta"]["baseline"] == "A" and result["meta"]["control"] == "A2"
    assert result["meta"]["timing_method"] == stats.DRY_RUN_TIMING
    assert all(a["torch_imported"] is False for a in result["meta"]["arms"].values())
    assert [c["case"] for c in result["cases"]] == [
        f"{s}/{lay}" for s in ("g01", "g03", "g05") for lay in ("bshd", "bhsd")
    ]
    # The stats file carries the bar's lines too: read alone, it says the times are
    # synthetic and whether every expected case is in.
    md = Path(str(out)[:-5] + ".stats.md").read_text()
    assert "| grid/bshd | 3 of 3 |" in md
    assert "run: 6 of 6 expected cases recorded; finished" in md
    assert "not a measurement" in md
    assert md.rstrip().endswith("BAR: NOT A MEASUREMENT (dry run)")
    assert {p.name for p in out.parent.glob("*.export.yaml")} == {
        "A.export.yaml",
        "B.export.yaml",
    }
    for c in result["cases"]:
        assert c["status"] == "ok" and c["spec_identity_equal"] is True
        assert {a["status"] for a in c["arms"].values()} == {"ok"}
        assert len(c["measurements"]) == 24 * 3 and c["rounds"] == 24
        assert c["o_hash_equal"] == {"pre": True, "post": True}
        assert set(c["check_pre"]) == set(c["check_post"]) == {"A", "B", "A2"}
        if c["case"] == "g01/bshd":
            assert c["strides"]["q"] == [
                1024 * 32 * 128,
                128,
                32 * 128,
            ]  # batch, head, token
        if c["case"] == "g01/bhsd":
            assert c["strides"]["q"] == [32 * 1024 * 128, 1024 * 128, 128]
    an = stats.analyze(result, n_boot=300)
    assert sorted(an["strata"]) == ["grid/bhsd", "grid/bshd"]
    assert {k: v["arms"]["B"]["verdict"] for k, v in an["strata"].items()} == {
        "grid/bshd": "pass",
        "grid/bhsd": "pass",
    }
    bar = _run(["bar", str(out), "--n-boot", "300"])
    assert bar.returncode == DRY_EXIT and "BAR: NOT A MEASUREMENT" in bar.stdout
    log = out.parent / "run.log"
    log.write_text(run.stdout)  # a log of AB_ROW lines is as good as the result file
    bar_log = _run(["bar", str(log), "--n-boot", "300"])
    assert bar_log.returncode == DRY_EXIT
    assert "run: 6 of 6 expected cases recorded; finished" in bar_log.stdout
    # the other --baseline: against the control's twin the same data still reads (same
    # tree)
    other = _run(
        ["bar", str(out), "--n-boot", "300", "--baseline", "A2", "--control", "A"]
    )
    assert other.returncode == DRY_EXIT and "baseline=A2 control=A" in other.stdout


def test_a_finished_result_lists_its_cases_and_says_it_is_complete(
    shapes3, tmp_path_factory
):
    """What ``bar`` needs to tell a finished run from a crashed one: the cases the run
    promised at its start, a ``complete`` flag, and the same in the log."""
    run, out = _dry(shapes3, tmp_path_factory, "plain")
    meta = stats.load_structured(str(out))["meta"]
    assert meta["expected_cases"] == [
        f"{s}/{lay}" for s in ("g01", "g03", "g05") for lay in ("bshd", "bhsd")
    ]
    assert meta["complete"] is True
    (done,) = [ln for ln in run.stdout.splitlines() if ln.startswith("AB_DONE ")]
    assert json.loads(done.split(" ", 1)[1])["complete"] is True
    (head,) = [ln for ln in run.stdout.splitlines() if ln.startswith("AB_META ")]
    assert json.loads(head.split(" ", 1)[1])["complete"] is False  # until it ends


def test_dry_run_schedule_is_mirrored_and_balanced(shapes3, tmp_path_factory):
    _, out = _dry(shapes3, tmp_path_factory, "plain")
    result = stats.load_structured(str(out))
    for c in result["cases"]:
        sched = [o.split(",") for o in c["order"]]
        assert len(sched) == 24
        for r in range(0, 24, 2):
            assert sched[r + 1] == sched[r][::-1]
        for arm in ("A", "B", "A2"):
            counts = [sum(1 for o in sched if o[p] == arm) for p in range(3)]
            assert counts == [8, 8, 8], (arm, counts)
        by_round = {}
        for m in c["measurements"]:
            by_round.setdefault(m["round"], []).append(m["arm"])
        assert [
            by_round[r] for r in range(24)
        ] == sched  # measured in the recorded order


def test_dry_run_bias_fails_the_bar_in_one_direction_only(shapes3, tmp_path_factory):
    slow, out_slow = _dry(
        shapes3, tmp_path_factory, "slow", "--dry-run-effect", "B=0.05"
    )
    fast, out_fast = _dry(
        shapes3, tmp_path_factory, "fast", "--dry-run-effect", "B=-0.05"
    )
    # The run's exit code is its own bar's: a +5% arm fails it (1); the -5% arm passes
    # every stratum, and only the made-up times keep the run from exiting 0.
    assert slow.returncode == 1 and fast.returncode == DRY_EXIT
    assert "BAR: FAIL" in slow.stdout and "BAR: NOT A MEASUREMENT" in fast.stdout
    assert _run(["bar", str(out_slow), "--n-boot", "300"]).returncode == 1
    assert _run(["bar", str(out_fast), "--n-boot", "300"]).returncode == DRY_EXIT
    fast_an = stats.analyze(stats.load_structured(str(out_fast)), n_boot=300)
    assert {e["arms"]["B"]["verdict"] for e in fast_an["strata"].values()} == {"pass"}
    an = stats.analyze(stats.load_structured(str(out_slow)), n_boot=300)
    for e in an["strata"].values():
        assert e["arms"]["B"]["geomean"] == pytest.approx(1.05, abs=0.01)
        assert len(e["arms"]["B"]["cases_wholly_above_band"]) == 3


def test_dry_run_refused_layout_is_excluded_from_that_arms_ratios_and_lost(
    shapes3, tmp_path_factory
):
    """B refuses a layout the baseline serves: the run is valid (no case failed to
    run), B's ratios skip those cases, and the bar fails them as lost, so the run exits
    with the bar's 1."""
    run, out = _dry(shapes3, tmp_path_factory, "refuse", "--dry-run-refuse", "B=bhsd")
    assert run.returncode == 1, run.stdout[-2000:] + run.stderr[-2000:]
    result = stats.load_structured(str(out))
    for c in result["cases"]:
        b = c["arms"]["B"]
        if c["layout"] == "bhsd":
            assert b["status"] == "refused" and "refusal hook" in b["reason"]
            assert {m["arm"] for m in c["measurements"]} == {
                "A",
                "A2",
            } and "B" not in c["check_pre"]
            assert c["status"] == "ok"  # a refusal is not a failure
        else:
            assert b["status"] == "ok"
    an = stats.analyze(result, n_boot=300)
    bhsd, bshd = an["strata"]["grid/bhsd"], an["strata"]["grid/bshd"]
    assert (
        bhsd["arms"]["B"]["verdict"] == "fail (refused)"
        and bhsd["arms"]["B"]["n_refused"] == 3
        and len(bhsd["arms"]["B"]["lost_cases"]) == 3
    )
    assert bhsd["control"]["n_cases"] == 3  # the control still measured them
    assert bshd["arms"]["B"]["verdict"] == "pass" and bshd["arms"]["B"]["n_cases"] == 3
    assert an["all_output_checks_passed"] is True
    assert stats.bar_check(an)["verdict"] == "FAIL"
    assert "BAR: FAIL" in run.stdout


def test_dry_run_one_refused_case_leaves_the_rest_of_the_stratum(
    shapes3, tmp_path_factory
):
    _, out = _dry(
        shapes3, tmp_path_factory, "refuse_one", "--dry-run-refuse", "B=g03/bhsd"
    )
    an = stats.analyze(stats.load_structured(str(out)), n_boot=300)
    b = an["strata"]["grid/bhsd"]["arms"]["B"]
    assert (
        b["refused_cases"] == ["g03/bhsd"]
        and b["lost_cases"] == ["g03/bhsd"]
        and b["n_cases"] == 2
        and b["verdict"] == "fail (refused)"
    )


def test_an_export_for_another_cu_count_is_refused(shapes3, tmp_path_factory):
    """An arm given as an export file must be built for this run's CU count: another
    count, or none recorded (an older export), is refused before any timing."""
    d = tmp_path_factory.mktemp("ab_cus")
    exp = d / "e228.yaml"
    _run(
        [
            "export",
            "--arch",
            "gfx942",
            "--shapes",
            str(shapes3),
            "--dry-run",
            "--num-cus",
            "228",
            "--out",
            str(exp),
        ],
        check=0,
    )
    run_args = [
        "--arch",
        "gfx942",
        "--shapes",
        str(shapes3),
        "--control",
        "A2",
        "--dry-run",
        "--n-boot",
        "100",
        "--out",
        str(d / "r.yaml"),
    ]
    run = _run([*run_args, "--arm", f"A={exp}"], check=stats.EXIT_NO_DATA)
    assert "built for 228 CUs, this run for 304" in run.stderr
    _run([*run_args, "--arm", f"A={exp}", "--num-cus", "228"], check=DRY_EXIT)
    doc = stats.load_structured(str(exp))
    del doc["meta"]["num_cus"]
    old = d / "old.yaml"
    old.write_text(stats.dump_flow_yaml(doc))
    run = _run([*run_args, "--arm", f"A={old}"], check=stats.EXIT_NO_DATA)
    assert "does not record the CU count" in run.stderr


def test_a_baseline_that_refuses_the_packed_layout_is_an_error_not_a_skip(
    shapes3, tmp_path_factory
):
    """Every tree serves packed BSHD. A baseline that declines it (a stub gap, a stride
    bug) must not drop the case from every stratum while the bar passes on the rest;
    declining another layout stays a neutral skip."""
    run, out = _dry(
        shapes3, tmp_path_factory, "base_refuses", "--dry-run-refuse", "A=bshd"
    )
    assert run.returncode == 1, run.stdout[-2000:] + run.stderr[-2000:]
    result = stats.load_structured(str(out))
    for c in result["cases"]:
        if c["layout"] == "bshd":
            assert c["status"] == "error"
            assert c["arms"]["A"]["status"] == "error"
            assert "the baseline refused the packed layout" in c["arms"]["A"]["reason"]
        else:
            assert c["status"] == "ok"
    an = stats.analyze(result, n_boot=300)
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "fail (error)"
    assert "BAR: FAIL" in run.stdout
    other, out2 = _dry(
        shapes3, tmp_path_factory, "base_refuses_bhsd", "--dry-run-refuse", "A=bhsd"
    )
    assert other.returncode == DRY_EXIT, other.stdout[-2000:] + other.stderr[-2000:]
    for c in stats.load_structured(str(out2))["cases"]:
        assert c["status"] == ("refused" if c["layout"] == "bhsd" else "ok")


def _driver_args(shapes, out):
    return ab.build_parser().parse_args(
        [
            "--arch",
            "gfx942",
            "--shapes",
            str(shapes),
            "--dry-run",
            "--n-boot",
            "100",
            "--control",
            "A2",
            "--out",
            str(out),
        ]
    )


def test_a_run_that_crashes_partway_leaves_a_result_that_is_not_complete(
    shapes3, tmp_path_factory, tmp_path, monkeypatch
):
    """The result is written after every case; the loop's end is what marks it
    complete. A fault on the third case leaves two cases, the six that were promised,
    and no ``complete``, so ``bar`` reads INCOMPLETE and not PASS."""
    _, plain = _dry(shapes3, tmp_path_factory, "plain")
    arms = [(x, str(plain.parent / f"{x}.export.yaml")) for x in ("A", "B")]
    real, done = ab.run_case, []

    def faulting(*args, **kwargs):
        if len(done) == 2:
            raise RuntimeError("device fault")
        done.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ab, "run_case", faulting)
    out = tmp_path / "crashed.yaml"
    with pytest.raises(RuntimeError, match="device fault"):
        ab.run_driver(_driver_args(shapes3, out), arms, "A", "A2")
    result = stats.load_structured(str(out))
    assert [c["case"] for c in result["cases"]] == ["g01/bshd", "g01/bhsd"]
    assert len(result["meta"]["expected_cases"]) == 6
    assert result["meta"]["complete"] is False  # the ``finally`` must not set it
    assert result["meta"]["finished_utc"]  # but it still stamps the end
    bar = stats.bar_check(stats.analyze(result, n_boot=100))
    assert bar["verdict"] == "INCOMPLETE" and bar["exit_code"] == stats.EXIT_INCOMPLETE
    assert any(line.startswith("  missing: g03/bshd") for line in bar["lines"])


def test_dry_run_check_failure_excludes_the_case_and_fails_the_arm(
    shapes3, tmp_path_factory
):
    run, out = _dry(shapes3, tmp_path_factory, "wrong", "--dry-run-wrong", "B=bhsd")
    assert run.returncode == 1  # the driver reports failed cases in its exit code
    result = stats.load_structured(str(out))
    for c in result["cases"]:
        if c["layout"] == "bhsd":
            assert c["status"] == "check_failed" and c["o_hash_equal"] == {
                "pre": False,
                "post": False,
            }
            assert (
                c["check_pre"]["B"]["pass"] is False
                and c["check_pre"]["A"]["pass"] is True
            )
        else:
            assert c["status"] == "ok"
    an = stats.analyze(result, n_boot=300)
    e = an["strata"]["grid/bhsd"]
    assert e["n_judged"] == 0 and set(e["excluded"].values()) == {
        "check"
    }  # never in a geomean
    assert e["arms"]["B"]["verdict"] == "fail (check)"
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"
    assert an["all_output_checks_passed"] is False
    assert _run(["bar", str(out), "--n-boot", "300"]).returncode == 1


def test_dry_run_packed_qkv_needs_equal_lengths(tmp_path):
    shapes = tmp_path / "s.yaml"
    shapes.write_text(
        "shapes:\n"
        "  - {id: c01, batch: 1, hq: 32, hkv: 8, d: 128, sq: 1024, sk: 2048,\n"
        "     mask: none, dtype: fp16, select: auto, layouts: [bshd, packed_qkv]}\n"
    )
    out = tmp_path / "r.yaml"
    run = _run(
        [
            "--arch",
            "gfx950",
            "--shapes",
            str(shapes),
            *_arms("A", "B"),
            "--control",
            "A2",
            "--dry-run",
            "--out",
            str(out),
            "--n-boot",
            "200",
        ],
        check=DRY_EXIT,
    )
    result = stats.load_structured(str(out))
    by = {c["case"]: c for c in result["cases"]}
    assert (
        by["c01/packed_qkv"]["status"] == "not_applicable"
        and not by["c01/packed_qkv"]["measurements"]
    )
    assert by["c01/bshd"]["status"] == "ok"
    an = stats.analyze(result, n_boot=200)
    assert an["strata"]["grid/bshd"]["n_judged"] == 1
    assert an["strata"]["-/packed_qkv"]["excluded"] == {"c01/packed_qkv": "skipped"}
    assert "BAR:" in run.stdout


def test_layouts_flag_and_rounds_rule(shapes3):
    base = [
        "--arch",
        "gfx942",
        "--shapes",
        str(shapes3),
        *_arms("A", "B"),
        "--control",
        "A2",
        "--list",
    ]
    run = _run(base + ["--layouts", "bhsd"], check=0)
    assert (
        run.stdout.count("spec identity across arms: equal") == 3
        and "/bshd" not in run.stdout
    )
    low = _run(
        [
            "--arch",
            "gfx942",
            "--shapes",
            str(shapes3),
            *_arms("A", "B"),
            "--control",
            "A2",
            "--dry-run",
            "--rounds",
            "23",
        ]
    )
    assert low.returncode != 0 and "--rounds must be >= 24" in low.stderr
    assert (
        ab.round_up_rounds(24, 3) == 24
        and ab.round_up_rounds(24, 5) == 30
        and ab.round_up_rounds(25, 3) == 30
    )


# ---- export: the recorded launches and the kernarg patching ------------------------ #

EXPORT_SHAPES = """\
shapes:
  - {id: g01, model: m, batch: 2, hq: 32, hkv: 8, d: 128, sq: 1024, sk: 1024,
     mask: causal_tl, dtype: bf16, select: gfx942_dense,
     override: {persistent: false}, layouts: [bshd, bhsd, packed_qkv, batch_slice]}
  - {id: k01, model: m, batch: 1, hq: 8, hkv: 1, d: 64, sq: 4096, sk: 4096,
     mask: causal_tl, dtype: fp16, select: gfx942_dense,
     override: {persistent: false, kv_segments: rule}}
"""


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """The bare dry-run export of case ``g01`` (all four layouts), read back."""
    d = tmp_path_factory.mktemp("ab_export")
    shapes = d / "s.yaml"
    shapes.write_text(EXPORT_SHAPES)
    out = d / "e.yaml"
    args = ["export", "--arch", "gfx942", "--shapes", str(shapes), "--dry-run"]
    _run([*args, "--only", "g01", "--out", str(out)], check=0)
    return stats.load_structured(str(out))


G01 = {"batch": 2, "hq": 32, "hkv": 8, "d": 128, "sq": 1024, "sk": 1024}


def _launch(doc, case):
    (rec,) = [c for c in doc["cases"] if c["case"] == case]
    assert rec["status"] == "ok" and len(rec["launches"]) == 1
    (ln,) = rec["launches"]
    return rec, ln, base64.b64decode(ln["kernarg"])


def test_export_records_the_runners_strides_and_pointer_slots(exported):
    """The kernarg bytes are the tree's own: the twelve trailing i32 strides of a
    served body equal the layout's, and every view's address is a recorded slot."""
    for layout in ab.LAYOUTS:
        rec, ln, ka = _launch(exported, f"g01/{layout}")
        plan = ab.layout_plan(layout, G01)
        got = struct.unpack_from("<12i", ka, len(ka) - 48)
        want = [x for n in "qkvo" for x in plan["strides"][n]]
        assert list(got) == want, layout
        slots = {s["offset"]: (s["buffer"], s["byte_offset"]) for s in ln["slots"]}
        assert slots == {
            8 * i: (v["buffer"], 2 * v["offset"])
            for i, v in enumerate(plan["views"][n] for n in "qkvo")
        }, layout


def test_bare_export_records_the_cu_count_it_was_built_for(exported):
    """The driver compares this field with its own CU count and refuses a mismatch."""
    assert exported["meta"]["num_cus"] == ARCH_NUM_CUS["gfx942"]


def test_every_supported_arch_has_its_cu_count_in_the_shared_table():
    """The harness keeps no CU table of its own: ``--arch`` offers ``ARCHES``, and
    each one's count is its entry of ``rocke.helpers.split_k.ARCH_NUM_CUS``."""
    assert ab.ARCHES and set(ab.ARCHES) <= set(ARCH_NUM_CUS)
    assert not hasattr(ab, "ARCH_CUS")
    parser = ab.build_parser()
    for arch in ab.ARCHES:
        assert parser.parse_args(["--arch", arch]).arch == arch
    with pytest.raises(SystemExit):
        parser.parse_args(["--arch", "gfx1100"])  # in the table, not supported here


@pytest.mark.parametrize("flag", ["dry_run", "list"])
@pytest.mark.parametrize("arch", ab.ARCHES)
def test_cu_count_without_a_device_is_read_from_the_shared_table(
    arch, flag, monkeypatch
):
    """The count is looked up in the shared table where it is used: a copy kept in the
    harness would not follow this one (the values are stand-ins, not the real counts,
    so the test does not pin any CU count)."""
    import rocke.helpers.split_k as split_k

    stand_in = {a: 1000 + i for i, a in enumerate(ab.ARCHES)}
    monkeypatch.setattr(split_k, "ARCH_NUM_CUS", stand_in)
    run = types.SimpleNamespace(arch=arch, num_cus=0, dry_run=False, list=False)
    setattr(run, flag, True)
    assert ab.resolve_num_cus(run) == stand_in[arch]


def test_a_rocke_without_the_cu_table_is_a_refusal_and_num_cus_still_works(
    monkeypatch,
):
    """An arm tree or a venv whose rocke predates the public table cannot give a CU
    count: that is a refusal that names ``--num-cus``, and a count that is passed never
    reads the table."""
    import rocke.helpers.split_k as split_k

    monkeypatch.delattr(split_k, "ARCH_NUM_CUS")
    run = types.SimpleNamespace(arch="gfx942", num_cus=0, dry_run=True, list=False)
    with pytest.raises(SystemExit, match="pass --num-cus"):
        ab.resolve_num_cus(run)
    run.num_cus = 77
    assert ab.resolve_num_cus(run) == 77


def test_cu_count_defaults_to_the_device_and_a_mismatched_export_is_refused(
    tmp_path, monkeypatch
):
    """A measured run builds its factory requests for the device's own CU count
    (``ARCH_NUM_CUS`` only without a device), and an arm exported for another count is
    refused before anything is timed. In-process (no driver subprocess, no source
    tree), so it runs in the installed CPU lane."""
    import json

    def args(**kw):
        base = dict(arch="gfx942", num_cus=0, dry_run=False, list=False)
        return types.SimpleNamespace(**{**base, **kw})

    device = types.SimpleNamespace(multi_processor_count=228)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True, get_device_properties=lambda i: device
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert ab.resolve_num_cus(args()) == 228  # a 228-CU gfx942, not the table's 304
    assert ab.resolve_num_cus(args(num_cus=80)) == 80
    assert ab.resolve_num_cus(args(dry_run=True)) == ARCH_NUM_CUS["gfx942"]
    assert ab.resolve_num_cus(args(list=True)) == ARCH_NUM_CUS["gfx942"]

    export = tmp_path / "a.export.yaml"
    export.write_text(json.dumps({"meta": {"num_cus": 304}, "cases": []}))
    run = args(
        dry_run=True,
        num_cus=228,
        shapes=str(_SHAPES),
        layouts="bshd",
        only="",
        out=str(tmp_path / "r.yaml"),
    )
    # The advice is to export for this run's count, never to time the other part's
    # specs.
    with pytest.raises(
        SystemExit,
        match="built for 304 CUs, this run for 228; export it again with "
        "--num-cus 228 ",
    ):
        ab.run_driver(run, [("A", str(export)), ("B", str(export))], "A", "A2")


def test_control_loads_the_baselines_code_object(tmp_path):
    shapes = tmp_path / "s.yaml"
    shapes.write_text(THREE_SHAPES)
    out = tmp_path / "r.yaml"
    _run(
        [
            "--arch",
            "gfx942",
            "--shapes",
            str(shapes),
            *_arms("A", "B"),
            "--control",
            "A2",
            "--dry-run",
            "--only",
            "g01",
            "--out",
            str(out),
            "--no-stats",
        ],
        check=stats.EXIT_NO_DATA,  # no analysis, so no verdict: not an exit 0
    )
    for c in stats.load_structured(str(out))["cases"]:
        a = c["arms"]
        assert a["A2"]["code_sha256"] == a["A"]["code_sha256"] == a["B"]["code_sha256"]


def test_kv_split_export_records_the_combine_and_the_workspace(tmp_path):
    shapes = tmp_path / "s.yaml"
    shapes.write_text(EXPORT_SHAPES)
    out = tmp_path / "e.yaml"
    _run(
        [
            "export",
            "--arch",
            "gfx942",
            "--shapes",
            str(shapes),
            "--dry-run",
            "--only",
            "k01",
            "--out",
            str(out),
        ],
        check=0,
    )
    doc = stats.load_structured(str(out))
    assert doc["meta"]["torch_imported"] is False  # the runner takes an allocator
    (rec,) = doc["cases"]
    main, combine = rec["launches"]
    assert rec["workspace_groups"] == 3 and "reduce" in combine["kernel"]
    # The recorded size is what the runner asked for: its largest buffer is the
    # [B * Sq, Hq, segments, D] FP32 accumulator, whatever the harness assumes.
    segments = rec["report"]["summary"]["kv_segments"]
    assert segments > 1
    assert rec["workspace_bytes"] == 4 * (1 * 4096) * 8 * segments * 64
    assert rec["signature"] == ab.case_signature(
        {**ab.load_shapes(str(shapes))[1], "layout": "bshd"}
    )

    def groups(launch):
        return [s["workspace"] for s in launch["slots"] if "workspace" in s]

    assert sorted(set(groups(main))) == [0, 1, 2]
    # seq_lens aliases the max buffer
    assert 1 in groups(combine) and groups(combine).count(1) == 2
    assert [s["buffer"] for s in combine["slots"] if "buffer" in s] == ["o"]


def test_export_asserts_that_the_launches_carry_every_view(tmp_path):
    """A runner whose launches do not address the views the harness handed it (here:
    none at all) is an error, not an empty export."""
    shapes = tmp_path / "s.yaml"
    shapes.write_text(EXPORT_SHAPES)
    (case,) = ab.expand_cases(ab.load_shapes(str(shapes)), ["bshd"], ["g01"])
    kmod = types.SimpleNamespace(
        supports_attention_dense=lambda spec, arch: (True, ""),
        run_attention_dense_torch=lambda **_: None,
    )
    args = types.SimpleNamespace(arch="gfx942", num_cus=0, list=False, dry_run=True)
    with pytest.raises(AssertionError, match="the launches address"):
        ab.export_case(case, args, kmod)


def _signature():
    return [
        {"name": "q_ptr", "type": "ptr<f16, global>"},
        {"name": "o_ptr", "type": "ptr<f16, global>"},
        {"name": "scale", "type": "f32"},
        {"name": "n", "type": "i32"},
        {"name": "ws_ptr", "type": "ptr<f32, global>"},
    ]


def test_locate_and_patch_pointer_slots():
    from rocke.runtime.packing import pack_args

    bases = ab.sentinel_bases(["q", "o"])
    values = {
        "q_ptr": bases["q"] + 64,
        "o_ptr": bases["o"],
        "scale": 0.5,
        "n": 7,
        "ws_ptr": 0x1234_0000,
    }
    ka = pack_args(_signature(), values)
    ws: dict = {}
    slots = ab.locate_pointer_slots(_signature(), values, ka, bases, ws)
    assert slots == [
        {"offset": 0, "buffer": "q", "byte_offset": 64},
        {"offset": 8, "buffer": "o", "byte_offset": 0},
        {"offset": 24, "workspace": 0},
    ]
    new = ab.patch_kernarg(ka, slots, {"q": 0x1000, "o": 0x2000}, [0x3000])
    assert struct.unpack_from("<QQ", new, 0) == (0x1040, 0x2000)
    assert struct.unpack_from("<Q", new, 24)[0] == 0x3000
    assert new[16:24] == ka[16:24]  # the scalars are untouched


def test_locate_refuses_a_stray_sentinel_and_a_wrong_slot():
    from rocke.runtime.packing import pack_args

    bases = ab.sentinel_bases(["q", "o"])
    values = {
        "q_ptr": bases["q"],
        "o_ptr": bases["o"],
        "scale": 0.5,
        "n": 7,
        "ws_ptr": 0x10,
    }
    ka = bytearray(pack_args(_signature(), values))
    ka[16:24] = struct.pack("<Q", bases["q"] + 8)  # a sentinel where no pointer is
    with pytest.raises(AssertionError, match="sentinel"):
        ab.locate_pointer_slots(_signature(), values, bytes(ka), bases, {})
    ka = bytearray(pack_args(_signature(), values))
    ka[0:8] = struct.pack(
        "<Q", bases["q"] + 8
    )  # the slot holds another address than given
    with pytest.raises(AssertionError, match="not the address given"):
        ab.locate_pointer_slots(_signature(), values, bytes(ka), bases, {})


# ---- a pre-built export is held to this run ---------------------------------------- #

G01_CASE = {
    "case": "g01/bshd",
    "id": "g01",
    "batch": 1,
    "hq": 32,
    "hkv": 8,
    "d": 128,
    "sq": 1024,
    "sk": 1024,
    "mask": "causal_tl",
    "dtype": "bf16",
    "select": "gfx942_dense",
    "override": {"persistent": False},
    "layout": "bshd",
}


def _export_doc(case=G01_CASE, **meta):
    record = {"case": case["case"], "signature": ab.case_signature(case)}
    return {
        "meta": {"num_cus": 304, "arch": "gfx942", **meta},
        "by_case": {case["case"]: record},
    }


def test_case_signature_holds_what_decides_the_launches_and_nothing_else():
    named = {**G01_CASE, "model": "x", "expect": {"body": "grid"}, "case": "z"}
    sig = ab.case_signature(named)
    assert set(sig) == set(ab.SIGNATURE_FIELDS)
    assert sig == ab.case_signature(G01_CASE)  # a label or an expectation is not one
    assert sig["override"] == {"persistent": False}
    assert json.loads(json.dumps(sig)) == sig
    bare = {k: v for k, v in G01_CASE.items() if k != "override"}
    assert ab.case_signature(bare)["override"] is None


def test_an_export_built_for_this_run_is_accepted():
    ab.check_export("A", _export_doc(), [G01_CASE], "gfx942", 304)
    other_label = {**G01_CASE, "model": "renamed", "expect": {"body": "persistent"}}
    ab.check_export("A", _export_doc(), [other_label], "gfx942", 304)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"hq": 16}, r"hq: exported 32, now 16"),
        ({"sq": 2048, "mask": "none"}, r"sq: exported 1024, now 2048; mask: exported"),
        ({"override": {"persistent": True}}, r"override: exported \{'persistent'"),
        ({"layout": "bhsd"}, r"layout: exported 'bshd', now 'bhsd'"),
    ],
)
def test_an_export_of_another_shape_is_refused_naming_the_field(change, reason):
    """The shape file changed after the export (same ids, other numbers): launching the
    stale kernargs on buffers sized for the new shape could write out of bounds."""
    stale = _export_doc()
    now = {**G01_CASE, **change}
    stale["by_case"][now["case"]]["signature"] = ab.case_signature(G01_CASE)
    with pytest.raises(
        SystemExit,
        match=r"arm A: its export of case g01/bshd was built for another shape \("
        + reason,
    ):
        ab.check_export("A", stale, [now], "gfx942", 304)


def test_an_export_that_lacks_a_case_or_its_shape_or_the_arch_is_refused():
    missing = {**G01_CASE, "case": "g01/bhsd", "layout": "bhsd"}
    with pytest.raises(SystemExit, match="its export holds no case 'g01/bhsd'"):
        ab.check_export("A", _export_doc(), [missing], "gfx942", 304)
    old = _export_doc()
    del old["by_case"]["g01/bshd"]["signature"]
    with pytest.raises(SystemExit, match="does not record the shape it was built for"):
        ab.check_export("A", old, [G01_CASE], "gfx942", 304)
    with pytest.raises(
        SystemExit, match="built for arch 'gfx942', this run is for 'gfx950'"
    ):
        ab.check_export("A", _export_doc(), [G01_CASE], "gfx950", 304)
    with pytest.raises(SystemExit, match="built for arch None"):
        ab.check_export("A", _export_doc(arch=None), [G01_CASE], "gfx942", 304)
    with pytest.raises(SystemExit, match="built for 228 CUs, this run for 304"):
        ab.check_export("A", _export_doc(num_cus=228), [G01_CASE], "gfx942", 304)


def test_a_stale_or_short_export_file_is_refused_before_anything_is_timed(
    shapes3, tmp_path_factory, tmp_path
):
    _, plain = _dry(shapes3, tmp_path_factory, "plain")
    arms = [
        x
        for lab in ("A", "B")
        for x in ("--arm", f"{lab}={plain.parent / lab}.export.yaml")
    ]
    base = ["--arch", "gfx942", "--baseline", "A", "--control", "A2", "--dry-run"]
    out = tmp_path / "r.yaml"
    edited = tmp_path / "edited.yaml"
    edited.write_text(
        THREE_SHAPES.replace("hq: 32, hkv: 8, d: 128", "hq: 16, hkv: 8, d: 128")
    )
    refused = stats.EXIT_NO_DATA  # no verdict: not the 1 of a failed case
    run = _run(
        [*base, "--shapes", str(edited), *arms, "--out", str(out)], check=refused
    )
    assert (
        "its export of case g01/bshd was built for another shape "
        "(hq: exported 32, now 16)"
    ) in run.stderr
    longer = tmp_path / "longer.yaml"
    longer.write_text(
        THREE_SHAPES
        + "  - {id: g07, batch: 1, hq: 8, hkv: 8, d: 64, sq: 1024, sk: 1024,\n"
        "     mask: none, dtype: fp16, select: gfx942_dense}\n"
    )
    run = _run(
        [*base, "--shapes", str(longer), *arms, "--out", str(out)], check=refused
    )
    assert "its export holds no case 'g07/bshd'" in run.stderr
    assert not out.exists()  # nothing ran


def test_arms_built_by_different_compilers_are_refused_before_anything_is_timed(
    shapes3, tmp_path_factory, tmp_path
):
    """A compiler change reads as a code change, and the control cannot show it: the
    arms of one run must come from one compiler."""
    _, plain = _dry(shapes3, tmp_path_factory, "plain")
    compiler = {"llvm_version": [22, 0, 0], "comgr_path": "/rocm/lib/libamd_comgr.so"}
    files = {}
    for label in ("A", "B"):
        doc = stats.load_structured(str(plain.parent / f"{label}.export.yaml"))
        doc["meta"]["compiler"] = compiler if label == "A" else None
        files[label] = tmp_path / f"{label}.export.yaml"
        files[label].write_text(stats.dump_flow_yaml(doc))
    args = [
        "--arch",
        "gfx942",
        "--shapes",
        str(shapes3),
        "--arm",
        f"A={files['A']}",
        "--arm",
        f"B={files['B']}",
        "--control",
        "A2",
        "--dry-run",
        "--n-boot",
        "100",
        "--out",
        str(tmp_path / "r.yaml"),
    ]
    run = _run(args, check=stats.EXIT_NO_DATA)
    assert "arm 'B' was compiled with None, baseline 'A' with ((22, 0, 0)" in run.stderr
    assert "export every arm in the environment that will time them" in run.stderr
    doc = stats.load_structured(str(files["B"]))
    doc["meta"]["compiler"] = dict(compiler)
    files["B"].write_text(stats.dump_flow_yaml(doc))
    _run(args, check=DRY_EXIT)  # the same compiler in both: timed


def test_a_refusal_exits_with_no_verdict_and_its_reason_not_as_a_failure(
    tmp_path, capsys
):
    """The harness refuses input by raising ``SystemExit`` with the reason; ``main``
    prints it and exits ``EXIT_NO_DATA``, so a script never reads a stale export or an
    unreadable file as the ``EXIT_FAIL`` of a slower or wrong arm. In-process (no
    driver subprocess, no source tree), so it runs in the installed CPU lane."""
    export = tmp_path / "a.export.yaml"
    export.write_text(json.dumps({"meta": {"num_cus": 304, "arch": "gfx942"}}))
    run = ["--arch", "gfx942", "--control", "A2", "--dry-run"]
    run += ["--out", str(tmp_path / "r.yaml")]
    junk_rows = tmp_path / "rows.yaml"
    junk_rows.write_text("shapes: [1, 2]\n")
    refusals = [
        ([f"A={export}"], str(_SHAPES), "not an export"),
        ([f"A={tmp_path}"], str(_SHAPES), "not a rocKE root"),
        ([f"A={export}"], str(tmp_path / "nope.yaml"), "cannot read the shape file"),
        ([f"A={export}"], str(junk_rows), "no shapes"),
    ]
    for arm, shapes, why in refusals:
        argv = [*run, "--arm", *arm, "--shapes", shapes]
        assert ab.main(argv) == stats.EXIT_NO_DATA != stats.EXIT_FAIL, why
        assert why in capsys.readouterr().err
    # An export of this run's CU count and arch that lacks the cases: stale.
    stale = {"meta": _export_doc()["meta"], "cases": []}
    export.write_text(stats.dump_flow_yaml(stale))
    argv = [*run, "--arm", f"A={export}", "--shapes", str(_SHAPES)]
    assert ab.main(argv) == stats.EXIT_NO_DATA
    assert "its export holds no case" in capsys.readouterr().err
    junk_log = tmp_path / "run.log"
    junk_log.write_text("no rows here\n")
    assert ab.main(["bar", str(junk_log)]) == stats.EXIT_NO_DATA
    assert "refused: " in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:  # argparse keeps its own exit code
        ab.main(["--arch", "gfx1"])
    assert usage.value.code == 2


def test_the_workspace_allocator_keeps_the_largest_size_it_was_asked_for():
    alloc = ab._WorkspaceAddresses()
    big, small = alloc((4, 3, 2, 128)), alloc((4, 3, 2))
    assert alloc.count == 2 and alloc.max_bytes == 4 * 4 * 3 * 2 * 128
    assert big.data_ptr() != small.data_ptr()


def test_segment_workspace_bytes_uses_the_recorded_size_or_the_old_runners_rule():
    case = {"batch": 2, "sq": 1024, "hq": 8, "d": 64}
    formula = 4 * 2 * 1024 * 8 * 3 * 64  # [B * Sq, Hq, segments, D] in FP32
    huge = {"workspace_groups": 3, "workspace_bytes": 10**9}
    tiny = {"workspace_groups": 3, "workspace_bytes": 5}
    older = {"workspace_groups": 3, "workspace_bytes": None}  # allocates its own
    assert ab.segment_workspace_bytes([huge, tiny], case, 3) == 10**9
    assert ab.segment_workspace_bytes([older], case, 3) == formula
    assert ab.segment_workspace_bytes([tiny, older], case, 3) == formula
    assert ab.segment_workspace_bytes([{"workspace_groups": 0}], case, 1) == 0
    assert ab.segment_workspace_bytes([], case, 1) == 0


def test_the_public_surface_is_listed_and_documented():
    """``__all__`` names every public function and class the module defines, each name
    resolves, and each callable and public method carries a docstring."""
    defined = {
        name
        for name, obj in vars(ab).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == ab.__name__
    }
    assert defined <= set(ab.__all__), sorted(defined - set(ab.__all__))
    assert [n for n in ab.__all__ if not hasattr(ab, n)] == []
    undocumented = []
    for name in ab.__all__:
        obj = getattr(ab, name)
        if not callable(obj):
            continue
        if not inspect.getdoc(obj):
            undocumented.append(name)
        if inspect.isclass(obj):
            undocumented += [
                f"{name}.{meth}"
                for meth, fn in vars(obj).items()
                if inspect.isfunction(fn)
                and not meth.startswith("_")
                and not inspect.getdoc(fn)
            ]
    assert undocumented == []


# ---- the control rule and the arguments -------------------------------------------- #


def test_control_is_a_new_label_and_the_arguments_are_checked(shapes3, tmp_path):
    common = ["--arch", "gfx942", "--shapes", str(shapes3), "--dry-run"]
    run = _run(common + _arms("A", "B"))
    assert run.returncode != 0 and "--control is required" in run.stderr
    run = _run(common + [*_arms("A", "B"), "--control", "B"])
    assert run.returncode != 0 and "must be a new label" in run.stderr
    run = _run(common + [*_arms("A", "B"), "--baseline", "Z", "--control", "A2"])
    assert run.returncode != 0 and "not an --arm label" in run.stderr
    run = _run(common + ["--control", "A2"])
    assert run.returncode != 0 and "at least one --arm" in run.stderr
    run = _run(common + [*_arms("A", "A"), "--control", "A2"])
    assert run.returncode != 0 and "duplicate label" in run.stderr
    run = _run(common + ["--arm", f"A={tmp_path}", "--control", "A2"])
    assert run.returncode != 0 and "not a rocKE root" in run.stderr


def test_resolve_rocke_root_accepts_the_root_and_the_repository_root(tmp_path):
    root = tmp_path / "repo" / "dnn-providers" / "hip-kernel-provider" / "rocke"
    (root / "platform" / "python").mkdir(parents=True)
    (root / "library").mkdir()
    assert ab.resolve_rocke_root(str(root)) == str(root.resolve())
    assert ab.resolve_rocke_root(str(tmp_path / "repo")) == str(root.resolve())
    link = tmp_path / "link"
    link.symlink_to(root)
    assert ab.resolve_rocke_root(str(link)) == str(
        root.resolve()
    )  # a symlink is the same tree
    with pytest.raises(SystemExit, match="not a rocKE root"):
        ab.resolve_rocke_root(str(tmp_path))
    assert (
        ab.parse_arms([f"A={root}", f"A2={tmp_path / 'repo'}"])[0][1]
        == ab.parse_arms([f"A2={tmp_path / 'repo'}"])[0][1]
    )
    with pytest.raises(SystemExit, match="label must match"):
        ab.parse_arms([f"bad label={root}"])


# ---- the layout model and the schedule (pure functions) ---------------------------- #


def _elements(view):
    """Element indices of a view (small shapes only)."""
    return {
        view["offset"] + sum(i * s for i, s in zip(idx, view["stride"]))
        for idx in itertools.product(*(range(n) for n in view["size"]))
    }


def _shape(**kw):
    base = {"batch": 2, "sq": 4, "sk": 4, "hq": 4, "hkv": 2, "d": 8}
    base.update(kw)
    return base


@pytest.mark.parametrize("layout", ab.LAYOUTS)
def test_layout_views_are_inside_their_buffers_and_do_not_overlap(layout):
    plan = ab.layout_plan(layout, _shape())
    owners = {}
    for name, v in plan["views"].items():
        els = _elements(v)
        assert (
            len(els) == v["size"][0] * v["size"][1] * v["size"][2] * v["size"][3]
        )  # no self overlap
        assert min(els) >= 0 and max(els) < plan["buffers"][v["buffer"]]
        owners.setdefault(v["buffer"], []).append((name, els))
    for buf, views in owners.items():
        for (n1, e1), (n2, e2) in itertools.combinations(views, 2):
            assert not (e1 & e2), (layout, buf, n1, n2)
    assert plan["views"]["q"]["size"] == [2, 4, 4, 8] and plan["views"]["k"][
        "size"
    ] == [2, 4, 2, 8]


def test_layout_strides_by_layout():
    s = _shape()
    bshd = ab.layout_plan("bshd", s)["strides"]
    assert bshd["q"] == [4 * 4 * 8, 8, 4 * 8]  # batch, head, token
    assert bshd["k"] == [4 * 2 * 8, 8, 2 * 8]
    bhsd = ab.layout_plan("bhsd", s)["strides"]
    assert (
        bhsd["q"] == [4 * 4 * 8, 4 * 8, 8]
        and bhsd["k"] == [2 * 4 * 8, 4 * 8, 8]
        and bhsd["o"] == bhsd["q"]
    )
    pk = ab.layout_plan("packed_qkv", s)
    width = 4 + 2 * 2
    assert (
        pk["strides"]["q"]
        == pk["strides"]["k"]
        == pk["strides"]["v"]
        == [4 * width * 8, 8, width * 8]
    )
    assert [pk["views"][n]["offset"] for n in "qkv"] == [0, 4 * 8, (4 + 2) * 8]
    assert pk["strides"]["o"] == bshd["q"] and pk["views"]["o"]["buffer"] == "o"
    sl = ab.layout_plan("batch_slice", s)
    assert (
        sl["strides"]["q"][0] == 2 * bshd["q"][0]
        and sl["views"]["q"]["offset"] == bshd["q"][0]
    )
    assert sl["buffers"]["q"] == 2 * 2 * 4 * 4 * 8 and sl["strides"]["o"] == bshd["q"]
    assert all(
        sl["strides"][n][1:] == bshd[n][1:] for n in "qkv"
    )  # only the batch stride differs


def test_layout_packed_qkv_needs_equal_lengths():
    with pytest.raises(ab.LayoutNotApplicable, match="packed_qkv needs sq == sk"):
        ab.layout_plan("packed_qkv", _shape(sq=4, sk=8))
    ab.layout_plan("batch_slice", _shape(sq=4, sk=8))  # the others take any lengths
    with pytest.raises(ValueError, match="unknown layout 'nope'"):
        ab.layout_plan("nope", _shape())


def test_order_schedule_is_balanced_and_mirrored():
    for arms in (["A", "B", "A2"], ["A", "B", "C", "A2"], ["A", "A2"]):
        k = len(arms)
        rounds = ab.round_up_rounds(24, k)
        assert rounds >= 24 and rounds % (2 * k) == 0
        sched = ab.order_schedule(arms, rounds)
        for r in range(0, rounds, 2):
            assert sched[r + 1] == sched[r][::-1]  # ABBA pair
        for arm in arms:
            counts = [sum(1 for o in sched if o[p] == arm) for p in range(k)]
            assert len(set(counts)) == 1, (arm, counts)  # every position equally often


def test_case_seed_is_per_shape_and_stable():
    assert (
        ab.case_seed(0, "g01")
        == ab.case_seed(0, "g01")
        != ab.case_seed(0, "g02")
        != ab.case_seed(1, "g02")
    )
    assert 0 <= ab.case_seed(7, "k03") < 2**31


def test_ref_rows_cover_first_middle_and_last_rows():
    pairs, rows = ab.ref_rows_plan({"batch": 4, "hq": 32, "sq": 1024})
    assert pairs == [(0, 0), (2, 16), (3, 31)]
    assert rows[0] == 0 and rows[-1] == 1023 and 512 in rows and len(rows) == 3 * 64
    assert ab.ref_rows_plan({"batch": 1, "hq": 8, "sq": 16})[1] == list(range(16))


def test_expand_cases_filters():
    shapes = ab.load_shapes(str(_SHAPES))
    assert len(ab.expand_cases(shapes)) == 13
    one = ab.expand_cases(shapes, only=["g01", "k02"])
    assert [c["case"] for c in one] == ["g01/bshd", "k02/bshd"]
    with pytest.raises(SystemExit, match="no case left after --only / --layouts"):
        ab.expand_cases(shapes, layouts=["bhsd"])  # no shape lists it
    with pytest.raises(SystemExit, match=r"--only: unknown shape ids \['nope'\]"):
        ab.expand_cases(shapes, only=["nope"])


def test_shipped_shapes_cover_the_grid_family_and_default_to_bshd():
    shapes = ab.load_shapes(str(_SHAPES))
    assert [s["id"] for s in shapes] == [
        "g01",
        "g02",
        "g03",
        "g04",
        "g05",
        "g06",
        "k01",
        "k02",
        "k03",
        "h01",
        "h02",
        "h03",
        "h04",
    ]
    assert all(
        s["layouts"] == ["bshd"] and s["expect"]["body"] == "grid" for s in shapes
    )
    assert {s["expect"]["variant"] for s in shapes} == {
        "grid",
        "grid_kvseg",
        "grid_qbminor",
    }
    assert {s["dtype"] for s in shapes} == {"fp16", "bf16"} and {
        s["d"] for s in shapes
    } == {64, 128}


def _identity(kernel="k", tuning="t1", **fields):
    return {
        "candidate": "c",
        "kernel_name": kernel,
        "tuning_id": tuning,
        "summary": {"block_m": 256, **fields},
    }


def test_a_tuning_id_difference_alone_is_not_a_spec_mismatch():
    """The tuning id hashes the kernel ABI version: arms on either side of an ABI
    change build the same kernel under different ids."""
    idents = {"A": _identity(tuning="t1"), "B": _identity(tuning="t2")}
    assert ab.identities_equal(idents) is True
    assert ab.tuning_ids_equal(idents) is False


def test_a_kernel_name_or_spec_field_difference_is_a_spec_mismatch():
    base = _identity(block_n=64)
    assert not ab.identities_equal(
        {"A": base, "B": _identity(kernel="other", block_n=64)}
    )
    assert not ab.identities_equal({"A": base, "B": _identity(block_n=128)})
    assert ab.identities_equal({"A": base, "B": _identity(block_n=64)})
    assert ab.tuning_ids_equal({"A": base, "B": _identity(block_n=64)}) is True


# ---- the strided numeric helper: a part that needs no device ----------------------- #


def test_the_numeric_helpers_compiler_label_survives_a_compiler_it_cannot_query(
    monkeypatch,
):
    """The label is read after every launch of a cohort has passed: a COMGR that failed
    to load, or that cannot say its version, must not turn that into an error."""
    from rocke.runtime import comgr

    from . import _attention_dense_strided as helper

    for info in (None, types.SimpleNamespace(llvm_version=None)):
        monkeypatch.setattr(comgr, "loaded_compiler_info", lambda info=info: info)
        assert helper.HipBackend.compiler_label() == "llvm unknown"
    monkeypatch.setattr(
        comgr,
        "loaded_compiler_info",
        lambda: types.SimpleNamespace(llvm_version=(22, 1, 0)),
    )
    assert helper.HipBackend.compiler_label() == "llvm 22.1.0"
