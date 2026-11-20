# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of ``benchmarks/common/attention_dense_strided_ab.py``: ``--list`` and
``--dry-run`` through real worker processes (one per arm tree, no torch, no GPU, no
compiler), the control rule, the exclusion of refused and failed cases, the bar check
in both directions, and the layout model.

The tests that run the driver in a subprocess (``_run``: ``--list``, ``--dry-run``,
``export``) export this rocKE source tree (``platform/python`` beside ``library``), which
the installed CI tree does not have, so they skip there with that reason; the
pure-function tests (layouts, schedule, seeds, reference rows, shape files, spec
identity, kernarg patching, root resolution) run in the CPU lane. Run the whole file
from the source tree (pytest or ``run_all.py``) after a change to the harness.

The dry-run hooks the tests use (all addressed to one arm label):
``--dry-run-effect B=0.05`` (synthetic times +5%), ``--dry-run-refuse B=bhsd`` (arm B
refuses a layout or a case id) and ``--dry-run-wrong B=bhsd`` (arm B's output checks fail).
"""

from __future__ import annotations

import base64
import importlib.util
import itertools
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

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

THREE_SHAPES = """\
shapes:
  - {id: g01, model: m, batch: 1, hq: 32, hkv: 8, d: 128, sq: 1024, sk: 1024, mask: causal_tl, dtype: bf16, select: gfx942_dense,
     override: {persistent: false}, layouts: [bshd, bhsd]}
  - {id: g03, model: m, batch: 1, hq: 32, hkv: 8, d: 64, sq: 1024, sk: 1024, mask: causal_tl, dtype: bf16, select: gfx942_dense,
     layouts: [bshd, bhsd]}
  - {id: g05, model: m, batch: 2, hq: 32, hkv: 32, d: 64, sq: 1024, sk: 1024, mask: none, dtype: fp16, select: gfx942_dense,
     layouts: [bshd, bhsd]}
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
    return [x for l in labels for x in ("--arm", f"{l}={_ROCKE}")]


@pytest.fixture(scope="module")
def shapes3(tmp_path_factory):
    path = tmp_path_factory.mktemp("ab_shapes") / "three.yaml"
    path.write_text(THREE_SHAPES)
    return path


_RUNS: dict = {}


def _dry(shapes3, tmp_path_factory, name, *extra):
    """One dry run (cached per flag set): arms A, B and control A2, all the current tree."""
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


# ---- --list ---------------------------------------------------------------------- #


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
        "13 cases on layouts ['bshd']; expectation mismatches/errors/identity differences: 0"
        in out
    )
    assert (
        "g01/bshd" in out
        and "k03/bshd" in out
        and "variant=grid_kvseg" in out
        and "variant=grid_qbminor" in out
    )


# ---- dry run ---------------------------------------------------------------------- #


def test_dry_run_result_is_read_by_bar_and_by_the_analysis(shapes3, tmp_path_factory):
    run, out = _dry(shapes3, tmp_path_factory, "plain")
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert (
        run.stdout.count("AB_ROW ") == 6
        and "AB_DONE " in run.stdout
        and "BAR: PASS" in run.stdout
    )
    result = stats.load_structured(str(out))
    assert result["meta"]["baseline"] == "A" and result["meta"]["control"] == "A2"
    assert all(a["torch_imported"] is False for a in result["meta"]["arms"].values())
    assert [c["case"] for c in result["cases"]] == [
        f"{s}/{l}" for s in ("g01", "g03", "g05") for l in ("bshd", "bhsd")
    ]
    assert Path(str(out)[:-5] + ".stats.md").exists()
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
    assert bar.returncode == 0 and "BAR: PASS" in bar.stdout
    log = out.parent / "run.log"
    log.write_text(run.stdout)  # a log of AB_ROW lines is as good as the result file
    bar_log = _run(["bar", str(log), "--n-boot", "300"])
    assert bar_log.returncode == 0 and "BAR: PASS" in bar_log.stdout
    # the other --baseline: against the control's twin the same data still reads (same tree)
    other = _run(
        ["bar", str(out), "--n-boot", "300", "--baseline", "A2", "--control", "A"]
    )
    assert other.returncode == 0 and "baseline=A2 control=A" in other.stdout


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
    assert (
        slow.returncode == 0 and fast.returncode == 0
    )  # the run itself is fine; the bar judges
    assert "BAR: FAIL" in slow.stdout and "BAR: PASS" in fast.stdout
    assert _run(["bar", str(out_slow), "--n-boot", "300"]).returncode == 1
    assert _run(["bar", str(out_fast), "--n-boot", "300"]).returncode == 0
    an = stats.analyze(stats.load_structured(str(out_slow)), n_boot=300)
    for e in an["strata"].values():
        assert e["arms"]["B"]["geomean"] == pytest.approx(1.05, abs=0.01)
        assert len(e["arms"]["B"]["cases_wholly_above_band"]) == 3


def test_dry_run_refused_layout_is_excluded_from_that_arms_ratios_and_lost(
    shapes3, tmp_path_factory
):
    """B refuses a layout the baseline serves: the run is valid (exit 0, nothing
    failed to run), B's ratios skip those cases, and the bar fails them as lost."""
    run, out = _dry(shapes3, tmp_path_factory, "refuse", "--dry-run-refuse", "B=bhsd")
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
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
    run = _run([*run_args, "--arm", f"A={exp}"], check=1)
    assert "built for 228 CUs, this run for 304" in run.stderr
    _run([*run_args, "--arm", f"A={exp}", "--num-cus", "228"], check=0)
    doc = stats.load_structured(str(exp))
    del doc["meta"]["num_cus"]
    old = d / "old.yaml"
    old.write_text(stats.dump_flow_yaml(doc))
    run = _run([*run_args, "--arm", f"A={old}"], check=1)
    assert "does not record the CU count" in run.stderr


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
        "  - {id: c01, batch: 1, hq: 32, hkv: 8, d: 128, sq: 1024, sk: 2048, mask: none, dtype: fp16, select: auto,\n"
        "     layouts: [bshd, packed_qkv]}\n"
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
        check=0,
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


# ---- export: the recorded launches and the kernarg patching ------------------------------ #

EXPORT_SHAPES = """\
shapes:
  - {id: g01, model: m, batch: 2, hq: 32, hkv: 8, d: 128, sq: 1024, sk: 1024, mask: causal_tl, dtype: bf16, select: gfx942_dense,
     override: {persistent: false}, layouts: [bshd, bhsd, packed_qkv, batch_slice]}
  - {id: k01, model: m, batch: 1, hq: 8, hkv: 1, d: 64, sq: 4096, sk: 4096, mask: causal_tl, dtype: fp16, select: gfx942_dense,
     override: {persistent: false, kv_segments: rule}}
"""


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
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
    assert exported["meta"]["num_cus"] == ab.ARCH_CUS["gfx942"]


def test_cu_count_defaults_to_the_device_and_a_mismatched_export_is_refused(
    tmp_path, monkeypatch
):
    """A measured run builds its factory requests for the device's own CU count
    (ARCH_CUS only without a device), and an arm exported for another count is refused
    before anything is timed. In-process (no driver subprocess, no source tree), so it
    runs in the installed CPU lane."""
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
    assert ab.resolve_num_cus(args(dry_run=True)) == ab.ARCH_CUS["gfx942"]
    assert ab.resolve_num_cus(args(list=True)) == ab.ARCH_CUS["gfx942"]

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
    # The advice is to export for this run's count, never to time the other part's specs.
    with pytest.raises(
        SystemExit,
        match="built for 304 CUs, this run for 228; export it again with --num-cus 228 ",
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
        check=0,
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
    groups = lambda ln: [s["workspace"] for s in ln["slots"] if "workspace" in s]
    assert sorted(set(groups(main))) == [0, 1, 2]
    assert (
        1 in groups(combine) and groups(combine).count(1) == 2
    )  # seq_lens aliases the max buffer
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


# ---- the control rule and the arguments ------------------------------------------------- #


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
    with pytest.raises(SystemExit):
        ab.resolve_rocke_root(str(tmp_path))
    assert (
        ab.parse_arms([f"A={root}", f"A2={tmp_path / 'repo'}"])[0][1]
        == ab.parse_arms([f"A2={tmp_path / 'repo'}"])[0][1]
    )
    with pytest.raises(SystemExit):
        ab.parse_arms([f"bad label={root}"])


# ---- the layout model and the schedule (pure functions) ---------------------------------- #


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
    with pytest.raises(ab.LayoutNotApplicable):
        ab.layout_plan("packed_qkv", _shape(sq=4, sk=8))
    ab.layout_plan("batch_slice", _shape(sq=4, sk=8))  # the others take any lengths
    with pytest.raises(ValueError):
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
    with pytest.raises(SystemExit):
        ab.expand_cases(shapes, layouts=["bhsd"])  # no shape lists it
    with pytest.raises(SystemExit):
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
