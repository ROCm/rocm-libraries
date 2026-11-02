# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host tests of the gfx950 GDN/KDA decode defaults report: the selection
rule's every step, scoring over shapes, and the printed ``AUTO_DEFAULTS`` row."""

from __future__ import annotations

import argparse
import os
from types import MappingProxyType

import pytest

from benchmarks.gfx950.gdn import decode_defaults_report as report
from benchmarks.gfx950.gdn import decode_table_sweep as sweep
from dispatch.gdn import dispatch_gdn_decode
from dispatch.gdn.axes import AUTO_DEFAULTS, AutoDefault, _auto

SHAPES = [
    sweep.Shape("kda", "plain", "bf16", h, h, 128, b)
    for h, b in ((8, 1), (8, 64), (32, 16))
]


def _cand(name, score, worst, nknobs):
    return report.Candidate(
        tuning_id=name,
        tile=(1, 16, 8),
        knobs={f"k{i}": True for i in range(nknobs)},
        score=score,
        worst=worst,
        worst_shape=SHAPES[0],
        ratios={},
    )


@pytest.mark.parametrize(
    "candidates,pick,step",
    [
        # Clear score winner.
        ([_cand("a", 1.000, 1.30, 5), _cand("b", 1.020, 1.00, 0)], "a", "score"),
        # Scores tied within 1%: the better worst shape wins.
        ([_cand("a", 1.000, 1.20, 0), _cand("b", 1.005, 1.10, 5)], "b", "worst shape"),
        # Scores and worst shapes tied: fewer non-default knobs.
        (
            [_cand("a", 1.000, 1.100, 3), _cand("b", 1.004, 1.105, 1)],
            "b",
            "fewer knobs",
        ),
        # Everything tied: the lower score.
        (
            [_cand("b", 1.004, 1.10, 2), _cand("a", 1.000, 1.10, 2)],
            "a",
            "score (last step)",
        ),
        # A worst shape just outside the band breaks the tie although the
        # other has fewer knobs.
        (
            [_cand("a", 1.000, 1.100, 4), _cand("b", 1.000, 1.112, 0)],
            "a",
            "worst shape",
        ),
    ],
)
def test_rule_each_step_decides_where_the_previous_ties(candidates, pick, step):
    chosen, decided_by = report.apply_rule(candidates, tie=0.01)
    assert (chosen.tuning_id, decided_by) == (pick, step)


# ``tie`` 0.25 and the scores below are exact in binary, so ``best * (1 + tie)``
# is exactly 1.25 and the boundary is hit exactly, not approximately.
@pytest.mark.parametrize(
    "candidates,pick,step",
    [
        # A score EXACTLY at the band edge is tied (``<=``): the worst shape decides.
        ([_cand("a", 1.0, 1.5, 0), _cand("b", 1.25, 1.0, 5)], "b", "worst shape"),
        # Just above the edge it is not: the better score wins outright.
        ([_cand("a", 1.0, 1.5, 0), _cand("b", 1.2500001, 1.0, 5)], "a", "score"),
        # The same edge for the worst-shape band: exactly on it is tied, so the
        # candidate with fewer knobs wins...
        ([_cand("a", 1.0, 1.0, 5), _cand("b", 1.0, 1.25, 0)], "b", "fewer knobs"),
        # ...and just above it the better worst shape wins.
        ([_cand("a", 1.0, 1.0, 5), _cand("b", 1.0, 1.2500001, 0)], "a", "worst shape"),
    ],
)
def test_the_tie_band_edge_counts_as_tied_and_just_above_does_not(
    candidates, pick, step
):
    chosen, decided_by = report.apply_rule(candidates, tie=0.25)
    assert (chosen.tuning_id, decided_by) == (pick, step)


def test_the_fastest_configuration_of_a_shape_breaks_a_time_tie_by_tuning_id():
    """Two configurations at the same time: the smaller ``tuning_id`` is named,
    whichever the dict lists first."""
    assert report.fastest_id({"b": 1.0, "a": 1.0, "c": 2.0}) == "a"
    assert report.fastest_id({"a": 1.0, "b": 1.0}) == "a"
    assert report.fastest_id({"a": 2.0, "b": 1.0}) == "b"


def test_rule_needs_a_candidate():
    with pytest.raises(ValueError, match="every shape"):
        report.apply_rule([])


def _row(
    shape, tid, us, *, stage=1, auto=False, status="ok", knobs=None, seed=0,
    bases_only=False,
):  # fmt: skip
    return dict(
        shape.as_dict(),
        stage=stage,
        status=status,
        us=us,
        tuning_id=tid,
        tile=[1, 16, 8],
        spec_id="spec",
        knobs=knobs or {},
        auto=auto,
        rotate_mb=1024,
        cold_target_mb=1024,
        sha="abc",
        device="gfx950",
        index=0,
        reason="" if status == "ok" else status,
        sweep_level="full",
        tuning_sample=40,
        seed=seed,
        bases_only=bases_only,
    )


def _robust_case():
    """x and y are each fastest somewhere and bad elsewhere; z is never fastest
    but close everywhere; w misses a shape; auto is a middling config."""
    times = {
        "x": (1.00, 1.50, 1.50),
        "y": (1.50, 1.00, 1.50),
        "z": (1.05, 1.05, 1.05),
        "auto": (1.20, 1.20, 1.00),
    }
    rows = [
        _row(shape, tid, us[i], auto=tid == "auto")
        for tid, us in times.items()
        for i, shape in enumerate(SHAPES)
    ]
    rows.append(_row(SHAPES[0], "w", 0.50, knobs={"dpp_reduce": True}))
    rows.append(_row(SHAPES[1], "w", 0.90, status="fail"))
    return rows


def test_scores_use_complete_configs_and_the_rule_can_differ_from_every_shape_best():
    cases = report.collect(_robust_case(), stage=1)
    (case,) = cases
    assert case == ("kda", False, "bf16")
    scored = {c.tuning_id: c for c in report.score_candidates(cases[case])}
    # w is fastest at one shape but untimed at the others: not a candidate.
    assert set(scored) == {"x", "y", "z", "auto"}
    # Ratios are against each shape's fastest, w included where it ran.
    assert scored["x"].ratios[SHAPES[0]] == pytest.approx(2.0)
    assert (scored["z"].worst, scored["z"].worst_shape) == (
        pytest.approx(2.1),
        SHAPES[0],
    )
    pick, _ = report.apply_rule(list(scored.values()))
    fastest = {
        min(("x", "y", "z", "auto"), key=lambda t: cases[case]["times"][s][t])
        for s in SHAPES
    }
    assert pick.tuning_id == "z" and "z" not in fastest


def test_repeated_rows_merge_by_median_and_stages_stay_apart():
    rows = [_row(SHAPES[0], "a", us) for us in (1.0, 9.0, 2.0)]
    rows += [_row(SHAPES[0], "a", 5.0, stage=2)]
    assert report.collect(rows, 1)[("kda", False, "bf16")]["times"][SHAPES[0]] == {
        "a": 2.0
    }
    assert report.collect(rows, 2)[("kda", False, "bf16")]["times"][SHAPES[0]] == {
        "a": 5.0
    }


def test_report_prints_the_proposal_and_fails_without_auto(capsys):
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    assert report.report(_robust_case(), args) == 0
    out = capsys.readouterr().out
    assert out.startswith("ADVISORY")
    assert "proposal z (decided by score)" in out
    assert (
        '("kda", False, "bf16"): AutoDefault((1, 16, 8), MappingProxyType({})),' in out
    )
    # The shortlist: the top 3 by score, the pick, and every shape's fastest
    # (w, y, auto), with the stage-1 walk flags the ids came from.
    assert (
        "re-time: --sweep-level full --tuning-sample 40 --seed 0 "
        "--tuning-ids z,auto,x,w,y --top-k 5" in out
    )
    rows = [r for r in _robust_case() if not (r["auto"] and r["batch"] == 64)]
    assert report.report(rows, args) == 1
    assert "auto: not timed at 1 shapes" in capsys.readouterr().out
    # Rows from two different walks cannot form one re-time command, and no
    # proposal or paste-ready row is printed for them.
    mixed = _robust_case() + [_row(SHAPES[0], "v", 9.0, seed=1)]
    assert report.report(mixed, args) == 1
    out = capsys.readouterr().out
    assert "WITHHELD: rows come from 2 walks" in out
    assert "proposal " not in out
    assert "AutoDefault(" not in out
    # bases_only is part of the walk too: the sweep refuses to re-time rows that
    # mix it, so the report must not propose from them either.
    mixed = _robust_case() + [_row(SHAPES[0], "v", 9.0, bases_only=True)]
    assert report.report(mixed, args) == 1
    out = capsys.readouterr().out
    assert "WITHHELD: rows come from 2 walks" in out and "bases_only" in out
    assert "proposal " not in out and "re-time:" not in out


def test_the_retime_line_and_walk_label_carry_bases_only(capsys):
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    rows = [dict(r, bases_only=True) for r in _robust_case()]
    assert report.report(rows, args) == 0
    out = capsys.readouterr().out
    assert "--seed 0 --bases-only)" in out
    assert (
        "re-time: --sweep-level full --tuning-sample 40 --seed 0 --bases-only "
        "--tuning-ids" in out
    )


def test_the_proposal_names_the_walk_it_was_chosen_from(capsys):
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    assert report.report(_robust_case(), args) == 0
    out = capsys.readouterr().out
    assert (
        "best of the walk the rows were swept with "
        "(--sweep-level full --tuning-sample 40 --seed 0), not of every configuration"
    ) in out


def test_warm_or_mixed_rotation_rows_withhold_the_proposal(capsys):
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    # A rotation below the cold target is warm, even though it is not 0.
    warm = _robust_case()
    warm[0]["rotate_mb"] = 8
    assert report.report(warm, args) == 1
    out = capsys.readouterr().out
    assert "WITHHELD" in out and "not timed in cold memory" in out
    assert "proposal z" not in out
    # Rows timed at two different rotations are not comparable, even if cold.
    mixed = _robust_case()
    mixed[0]["rotate_mb"] = 2048
    assert report.report(mixed, args) == 1
    out = capsys.readouterr().out
    assert "mix rotate_mb values [1024, 2048]" in out and "proposal z" not in out
    # Warm rows of another stage do not block this stage's report.
    other = _robust_case() + [_row(SHAPES[0], "q", 1.0, stage=2)]
    other[-1]["rotate_mb"] = 0
    assert report.report(other, args) == 0
    # A row that never recorded its cold target is not cold.
    unknown = _robust_case()
    del unknown[0]["cold_target_mb"]
    assert report.report(unknown, args) == 1
    assert "WITHHELD" in capsys.readouterr().out


def test_a_shape_without_a_timed_row_or_with_auto_failing_is_never_dropped(capsys):
    """B16: every configuration failed (no timed row); B64: auto failed the
    gate. Neither may silently shrink the case the score covers."""
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    rows = [_row(SHAPES[0], t, us, auto=t == "a") for t, us in (("a", 1.0), ("b", 1.1))]
    rows += [
        _row(SHAPES[2], "a", None, auto=True, status="fail"),
        _row(SHAPES[2], "b", None, status="error"),
    ]
    assert report.report(rows, args) == 1
    out = capsys.readouterr().out
    assert "2 shapes, 1 with a timed row" in out
    assert f"UNSCORED {SHAPES[2].label}: FAIL (auto) a: fail, ERROR b: error" in out
    rows = [
        _row(s, t, us, auto=t == "a")
        for s in SHAPES[:2]
        for t, us in (("a", 1.0), ("b", 1.1))
    ]
    rows = [r for r in rows if not (r["auto"] and r["batch"] == 64)]
    rows.append(_row(SHAPES[1], "a", None, auto=True, status="fail"))
    assert report.report(rows, args) == 1
    assert f"UNSCORED {SHAPES[1].label}: FAIL (auto) a: fail" in capsys.readouterr().out


def test_cli_refuses_a_tie_band_or_top_that_cannot_work(capsys):
    for bad in (("--tie", "-0.01"), ("--tie", "nan"), ("--top", "0")):
        with pytest.raises(SystemExit) as err:
            report.main(["rows.json", *bad])
        assert err.value.code == 2, bad
        assert bad[0] in capsys.readouterr().err, bad


def test_unreadable_or_non_sweep_input_is_a_usage_error(tmp_path, capsys):
    """An unreadable file, invalid JSON and another tool's JSON exit 2 naming
    the path, never a traceback."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    other = tmp_path / "other.json"
    other.write_text('{"schema": "something else", "rows": []}')
    for path in (tmp_path / "missing.json", bad_json, other):
        with pytest.raises(SystemExit) as err:
            report.main([str(path)])
        assert err.value.code == 2, path
        assert str(path) in capsys.readouterr().err, path


def test_a_missing_path_with_backslashes_is_named_as_typed(tmp_path, capsys):
    """``str(OSError)`` doubles each backslash of a Windows path; the usage
    error must print the path as given (a backslash is an ordinary file-name
    character on POSIX, so this fails there too on ``str(OSError)``)."""
    missing = str(tmp_path / "no\\such\\file.json")
    with pytest.raises(SystemExit) as err:
        report.main([missing])
    assert err.value.code == 2
    assert missing in capsys.readouterr().err


def test_rows_that_lack_a_required_key_are_a_usage_error(tmp_path, capsys):
    rows = _robust_case()
    del rows[3]["sha"]
    path = str(tmp_path / "rows.json")
    sweep.write_rows(path, rows)
    with pytest.raises(SystemExit) as err:
        report.main([path])
    assert err.value.code == 2
    message = capsys.readouterr().err
    assert path in message and "row 3 lacks sha" in message


def test_a_row_missing_any_key_the_report_reads_is_a_usage_error(tmp_path, capsys):
    """Every key the report reads from a saved row (``tile`` included), spelled
    out here rather than taken from ``ROW_KEYS``: a file missing one exits 2
    naming the row and the key, never a ``KeyError`` traceback."""
    read = (
        *sweep.SHAPE_FIELDS, "stage", "index", "status", "auto", "us", "reason",
        "tuning_id", "tile", "knobs", "rotate_mb", "sha", "device", *sweep.WALK_FIELDS,
    )  # fmt: skip
    assert set(read) <= set(sweep.ROW_KEYS)
    for key in read:
        rows = _robust_case()
        del rows[1][key]
        path = str(tmp_path / f"no_{key}.json")
        sweep.write_rows(path, rows)
        with pytest.raises(SystemExit) as err:
            report.main([path])
        assert err.value.code == 2, key
        message = capsys.readouterr().err
        assert path in message and f"row 1 lacks {key}" in message, key


@pytest.mark.parametrize(
    "field,value",
    [
        ("gate_kind", "x"),
        ("batch", 0),
        ("sweep_level", "x"),
        ("tuning_sample", "40"),
        ("seed", 0.5),
        ("bases_only", "no"),
        ("rotate_mb", "big"),
    ],
)
def test_a_row_with_a_bad_value_is_a_usage_error_in_the_report(
    tmp_path, capsys, field, value
):
    rows = _robust_case()
    rows[2][field] = value
    path = str(tmp_path / "rows.json")
    sweep.write_rows(path, rows)
    with pytest.raises(SystemExit) as err:
        report.main([path])
    assert err.value.code == 2
    message = capsys.readouterr().err
    assert path in message and f"row 2: {field}" in message


def test_the_walk_label_comes_from_the_sweeps_walk_fields():
    walk = ("full", 40, 7, True)
    assert sweep.walk_flags(walk) == (
        "--sweep-level full --tuning-sample 40 --seed 7 --bases-only"
    )
    assert sweep.walk_flags(("production", 0, 0, False)) == (
        "--sweep-level production --tuning-sample 0 --seed 0"
    )
    assert sweep.walk_of(dict(zip(sweep.WALK_FIELDS, walk))) == walk


def test_report_prints_the_toolchain_of_the_rows_and_every_distinct_set(capsys):
    args = argparse.Namespace(stage=1, tie=0.01, top=3)
    tool = dict(
        rocm_version="7.1.0",
        llvm_version="21.0.0",
        torch_version="2.9",
        backend="cpp",
        llvm_flavor="llvm22",
    )
    rows = [dict(r, **tool) for r in _robust_case()]
    assert report.report(rows, args) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("toolchain:")]
    assert lines == [
        "toolchain: rocm 7.1.0 llvm 21.0.0 torch 2.9 backend cpp flavor llvm22"
    ]
    assert out.index("rows from sha") < out.index(lines[0])
    # Rows that disagree: every set is printed and nothing is withheld.
    rows[0] = dict(rows[0], backend="python", llvm_flavor="llvm20")
    assert report.report(rows, args) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "toolchain:" in ln]
    assert len(lines) == 2 and any("backend python flavor llvm20" in ln for ln in lines)
    # Rows of an older file record none of them: unknown, still reported.
    assert report.report(_robust_case(), args) == 0
    assert "toolchain: rocm unknown llvm unknown torch unknown backend unknown" in (
        capsys.readouterr().out
    )


def test_printed_rows_rebuild_every_shipped_default():
    """The printed row of each case's current ``auto`` configuration, as the
    sweep records it, evaluates to that case's ``AUTO_DEFAULTS`` entry."""
    path = os.path.join(os.path.dirname(sweep.__file__), "decode_shapes.json")
    seen = set()
    for shape in sweep.load_shapes(path):
        case = shape.auto_key
        if case in seen:
            continue
        seen.add(case)
        auto = dispatch_gdn_decode(shape.request())
        kernel = auto.spec.kernel_spec
        cand = report.Candidate(
            tuning_id=auto.spec.tuning_id,
            tile=(kernel.num_warps, kernel.warp_threads_k, kernel.blocks_per_v_dim),
            knobs=dict(auto.spec.knobs),
            score=1.0,
            worst=1.0,
            worst_shape=shape,
            ratios={},
        )
        text = report.auto_defaults_row(case, cand)
        key, value = text.rstrip(",").split(": ", 1)
        namespace = {
            "_auto": _auto,
            "AutoDefault": AutoDefault,
            "MappingProxyType": MappingProxyType,
        }
        assert eval(key) == case
        assert eval(value, namespace) == AUTO_DEFAULTS[case], text
    assert seen == set(AUTO_DEFAULTS)
