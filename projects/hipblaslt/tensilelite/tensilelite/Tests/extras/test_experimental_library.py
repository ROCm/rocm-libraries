# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the pure logic of Tensile.ExperimentalLibrary.

Lives under ``Tests/extras`` (not ``Tests/unit``) on purpose: the ``unit``
conftest imports ``streamk5_test_helpers`` -> ``rocisa.code``, so collecting any
test there requires a built rocisa. The pure-logic tests below
(``coerce_value``, ``parse_set_arg``, ``augment_config`` round-trip) need no
toolchain because ``Tensile.ExperimentalLibrary`` keeps its rocisa-dependent
imports lazy. The ``validate_sets`` tests genuinely need ``validParameters``
(which pulls in rocisa) and are guarded so they skip gracefully without a build.
"""

import os
from pathlib import Path

import pytest
import yaml

from Tensile.ExperimentalLibrary import (
    ExperimentalLibraryError,
    augment_config,
    coerce_value,
    merge_configs,
    parse_set_arg,
    select_indices,
    solution_matches,
    summarize_solution,
    validate_sets,
    _apply_overrides,
    _dedup_keys,
    _find_solutions_element,
    _indexed_out_path,
    _library_type,
    _placeholder_problem_size_groups,
    _resolve_logic_sources,
    _unique_staged_name,
    cmd_build_lib,
    cmd_extract,
    cmd_list_solutions,
    cmd_patch_logic,
)


def _require_validparameters():
    """Skip cleanly when the rocisa-backed validParameters registry is absent."""
    pytest.importorskip("rocisa")
    try:
        from Tensile.Common.ValidParameters import validParameters  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Tensile.Common.ValidParameters unavailable: {e}")


def test_coerce_value():
    assert coerce_value("1") == 1 and isinstance(coerce_value("1"), int)
    assert coerce_value("0") == 0
    assert coerce_value("True") is True
    assert coerce_value("false") is False
    assert coerce_value("1.5") == 1.5
    assert coerce_value("MultipleBuffer") == "MultipleBuffer"
    assert coerce_value("[16, 16]") == [16, 16]


def test_parse_set_arg_good():
    assert parse_set_arg("StreamKFixupTreeReduction=1") == (
        "StreamKFixupTreeReduction",
        [1],
    )
    assert parse_set_arg("StreamK=0,1") == ("StreamK", [0, 1])


def test_parse_set_arg_bracketed_list_is_one_value():
    # A bracketed list value must stay a single token despite its commas.
    assert parse_set_arg("MatrixInstruction=[16,16,16,1]") == (
        "MatrixInstruction",
        [[16, 16, 16, 1]],
    )


@pytest.mark.parametrize("bad", ["NoEquals", "=5", "Name="])
def test_parse_set_arg_bad(bad):
    with pytest.raises(ExperimentalLibraryError):
        parse_set_arg(bad)


def test_validate_sets_good():
    _require_validparameters()
    # Both a [0,1] feature param and a multi-value enum param.
    validate_sets([("StreamKFixupTreeReduction", [0, 1])])
    validate_sets([("GlobalSplitUAlgorithm", ["MultipleBuffer"])])


def test_validate_sets_unknown_name_suggests():
    _require_validparameters()
    with pytest.raises(ExperimentalLibraryError) as ei:
        validate_sets([("StreamKFixupTreeReductn", [1])])
    msg = str(ei.value)
    assert "Unknown solution parameter" in msg
    # Close-match suggestion should surface the real name.
    assert "StreamKFixupTreeReduction" in msg


def test_validate_sets_bad_value_lists_allowed():
    _require_validparameters()
    with pytest.raises(ExperimentalLibraryError) as ei:
        validate_sets([("StreamKFixupTreeReduction", [5])])
    msg = str(ei.value)
    assert "Invalid value" in msg
    assert "Allowed values" in msg


def _base_config():
    return {
        "GlobalParameters": {"NumElementsToValidate": 0},
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM", "DataType": "s"},
                {
                    "InitialSolutionParameters": None,
                    "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                    "ForkParameters": [
                        {"PrefetchGlobalRead": [2]},
                        {"StreamK": [1]},
                        {"Groups": [[{"MatrixInstruction": [16, 16, 16, 1]}]]},
                    ],
                    "BenchmarkFinalParameters": [
                        {"ProblemSizes": [{"Exact": [256, 256, 1, 256]}]}
                    ],
                },
            ]
        ],
        "LibraryLogic": {"ArchitectureName": "gfx950"},
    }


def _fork(config):
    return config["BenchmarkProblems"][0][1]["ForkParameters"]


def test_augment_injects_new_param_before_groups():
    config = _base_config()
    augment_config(config, [("StreamKFixupTreeReduction", [1])])
    fork = _fork(config)
    # New entry present.
    entries = {k: v for d in fork for k, v in d.items()}
    assert entries["StreamKFixupTreeReduction"] == [1]
    # Groups stays the last entry.
    assert "Groups" in fork[-1]


def test_augment_overrides_existing_param_in_place():
    config = _base_config()
    augment_config(config, [("StreamK", [3])])
    fork = _fork(config)
    streamk_entries = [d for d in fork if "StreamK" in d]
    # Overridden in place, not duplicated.
    assert len(streamk_entries) == 1
    assert streamk_entries[0]["StreamK"] == [3]


def test_augment_preserves_structure_and_other_keys():
    config = _base_config()
    before_size_group_keys = set(config["BenchmarkProblems"][0][1].keys())
    augment_config(config, [("StreamKFixupTreeReduction", [1])])
    after = config["BenchmarkProblems"][0][1]
    assert set(after.keys()) == before_size_group_keys
    assert after["BenchmarkCommonParameters"] == [{"KernelLanguage": ["Assembly"]}]
    assert after["BenchmarkFinalParameters"][0]["ProblemSizes"] == [
        {"Exact": [256, 256, 1, 256]}
    ]
    # Untouched top-level sections survive.
    assert config["LibraryLogic"]["ArchitectureName"] == "gfx950"


def test_augment_missing_benchmark_problems_raises():
    with pytest.raises(ExperimentalLibraryError):
        augment_config({"GlobalParameters": {}}, [("StreamK", [1])])


# ---------------------------------------------------------------------------
# Solution selection (list-solutions)
# ---------------------------------------------------------------------------


def _states():
    return [
        {"StreamK": 0, "DepthU": 32, "MatrixInstruction": [16, 16, 16, 1]},
        {"StreamK": 5, "DepthU": 64, "MatrixInstruction": [16, 16, 16, 1]},
        {"StreamK": 5, "DepthU": 32, "MatrixInstruction": [32, 32, 8, 1]},
        {"StreamK": 3, "DepthU": 64},
    ]


def test_select_indices_filters_by_value():
    assert select_indices(_states(), [("StreamK", [5])]) == [1, 2]


def test_select_indices_or_within_values():
    assert select_indices(_states(), [("StreamK", [3, 5])]) == [1, 2, 3]


def test_select_indices_and_across_keys():
    assert select_indices(_states(), [("StreamK", [5]), ("DepthU", [32])]) == [2]


def test_select_indices_no_predicate_returns_all():
    assert select_indices(_states(), []) == [0, 1, 2, 3]


def test_select_indices_missing_key_excluded():
    # No solution carries WorkGroupMapping -> nothing matches.
    assert select_indices(_states(), [("WorkGroupMapping", [1])]) == []


def test_solution_matches_list_value():
    s = {"MatrixInstruction": [16, 16, 16, 1]}
    assert solution_matches(s, [("MatrixInstruction", [[16, 16, 16, 1]])]) is True
    assert solution_matches(s, [("MatrixInstruction", [[32, 32, 8, 1]])]) is False


def test_solution_matches_bool_int_distinct():
    # Python treats True == 1; matching must keep bool and int parameters apart.
    assert solution_matches({"Flag": True}, [("Flag", [1])]) is False
    assert solution_matches({"Flag": True}, [("Flag", [True])]) is True
    assert solution_matches({"N": 1}, [("N", [True])]) is False
    assert solution_matches({"N": 1}, [("N", [1])]) is True


def test_summarize_solution_lists_present_keys():
    summary = summarize_solution(_states()[1])
    assert "StreamK=5" in summary and "DepthU=64" in summary
    assert summarize_solution({}) == "(no summary keys)"


# ---------------------------------------------------------------------------
# Config merge (merge)
# ---------------------------------------------------------------------------


def _single_solution_config(streamk, depthu, arch="gfx950"):
    return {
        "GlobalParameters": {"NumElementsToValidate": 0},
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM", "DataType": "s"},
                {
                    "ForkParameters": [{"StreamK": [streamk]}, {"DepthU": [depthu]}],
                    "BenchmarkFinalParameters": [
                        {"ProblemSizes": [{"Exact": [256, 256, 1, 256]}]}
                    ],
                },
            ]
        ],
        "LibraryLogic": {"ArchitectureName": arch, "ScheduleName": "gfx950"},
    }


def test_merge_configs_concatenates_problems():
    merged = merge_configs(
        [_single_solution_config(5, 64), _single_solution_config(5, 32)]
    )
    bp = merged["BenchmarkProblems"]
    assert len(bp) == 2
    assert merged["LibraryLogic"]["ArchitectureName"] == "gfx950"
    # Each group keeps its own distinct fork values.
    forks = [g[1]["ForkParameters"][1]["DepthU"] for g in bp]
    assert forks == [[64], [32]]


def test_merge_configs_rejects_arch_mismatch():
    with pytest.raises(ExperimentalLibraryError):
        merge_configs(
            [
                _single_solution_config(5, 64, "gfx950"),
                _single_solution_config(5, 64, "gfx942"),
            ]
        )


def test_merge_configs_empty_raises():
    with pytest.raises(ExperimentalLibraryError):
        merge_configs([])


def test_merge_configs_returns_independent_copy():
    a = _single_solution_config(5, 64)
    merged = merge_configs([a])
    merged["BenchmarkProblems"].append("sentinel")
    # Mutating the merged result must not bleed back into the input.
    assert len(a["BenchmarkProblems"]) == 1


def test_merge_configs_allows_multiple_problem_types():
    # Different DataTypes warn (GlobalParameters come from the first config) but
    # must not abort the merge.
    a = _single_solution_config(5, 64)
    b = _single_solution_config(5, 64)
    b["BenchmarkProblems"][0][0]["DataType"] = "I8"
    merged = merge_configs([a, b])
    assert len(merged["BenchmarkProblems"]) == 2


# ---------------------------------------------------------------------------
# gen-logic host-arch guard (benchmark-by-default; hard-fail on arch mismatch)
#
# These need rocisa because they import Tensile.Common.Architectures (which
# imports rocisa at module load); they skip cleanly without a build.
# ---------------------------------------------------------------------------

import argparse


def _gen_logic_ns(tmp_path, arch, dry_run=False, config=None):
    return argparse.Namespace(
        config=str(config) if config else str(tmp_path / "c.yaml"),
        arch=arch,
        out=str(tmp_path / "work"),
        cu=40,
        feature_name="f",
        python="python",
        dry_run=dry_run,
        verbose=False,
    )


def test_detect_host_gfx_archs_normalizes_and_filters(monkeypatch):
    """detectHostGfxArchs de-dups, drops CPU gfx000, and normalizes :xnack± variants."""
    pytest.importorskip("rocisa")
    import Tensile.Common.Architectures as Arch

    class _Proc:
        returncode = 0
        stdout = b"gfx950\ngfx950:xnack-\ngfx000\n"

    monkeypatch.setattr(
        "Tensile.Toolchain.Validators.validateToolchain", lambda tool: "/fake/enum"
    )
    monkeypatch.setattr(Arch, "run", lambda *a, **k: _Proc())

    assert Arch.detectHostGfxArchs() == ["gfx950"]
    assert Arch.hostHasArch("gfx950") is True
    assert Arch.hostHasArch("gfx950:xnack-") is True  # variant normalized
    assert Arch.hostHasArch("gfx1151") is False


def test_gen_logic_rejects_arch_absent_on_host(monkeypatch, tmp_path):
    """Target arch not present -> ExperimentalLibraryError naming the arch and detected set."""
    pytest.importorskip("rocisa")
    import Tensile.Common.Architectures as Arch
    from Tensile.ExperimentalLibrary import cmd_gen_logic

    monkeypatch.setattr(Arch, "hostHasArch", lambda a: False)
    monkeypatch.setattr(Arch, "detectHostGfxArchs", lambda: ["gfx950"])

    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")

    with pytest.raises(ExperimentalLibraryError) as ei:
        cmd_gen_logic(_gen_logic_ns(tmp_path, "gfx1151", config=cfg))
    msg = str(ei.value)
    assert "gfx1151" in msg
    assert "not present on this host" in msg
    assert "gfx950" in msg  # detected archs surfaced in the message


def test_gen_logic_arch_mismatch_maps_to_nonzero_exit(monkeypatch, tmp_path):
    """main() maps the guard's ExperimentalLibraryError to a non-zero exit code."""
    pytest.importorskip("rocisa")
    import Tensile.Common.Architectures as Arch
    from Tensile.ExperimentalLibrary import main

    monkeypatch.setattr(Arch, "hostHasArch", lambda a: False)
    monkeypatch.setattr(Arch, "detectHostGfxArchs", lambda: ["gfx950"])

    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")

    rc = main(
        [
            "gen-logic",
            "--config", str(cfg),
            "--arch", "gfx1151",
            "--out", str(tmp_path / "work"),
            "--cu", "40",
        ]
    )
    assert rc != 0


def test_gen_logic_matching_arch_passes_guard_and_omits_cpu_only(monkeypatch, tmp_path):
    """Matching arch clears the guard; the constructed Tensile cmd has no --cpu-only."""
    pytest.importorskip("rocisa")
    import Tensile.Common.Architectures as Arch
    import Tensile.ExperimentalLibrary as E

    monkeypatch.setattr(Arch, "hostHasArch", lambda a: True)

    captured = {}

    class _Stop(Exception):
        pass

    def _fake_run_command(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        raise _Stop()  # stop right after cmd construction; guard already passed

    monkeypatch.setattr(E, "run_command", _fake_run_command)

    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}\n")

    with pytest.raises(_Stop):
        E.cmd_gen_logic(_gen_logic_ns(tmp_path, "gfx950", config=cfg))

    cmd = captured["cmd"]
    assert "--cpu-only" not in cmd  # regression guard: benchmarking is now default
    assert "--gpu-targets" in cmd
    assert "gfx950" in cmd


def test_gen_logic_dry_run_bypasses_guard(monkeypatch, tmp_path):
    """--dry-run must stay hardware-independent: the guard is never consulted."""
    pytest.importorskip("rocisa")
    import Tensile.Common.Architectures as Arch
    import Tensile.ExperimentalLibrary as E

    def _boom(*a, **k):
        raise AssertionError("host-arch detection called under --dry-run")

    monkeypatch.setattr(Arch, "hostHasArch", _boom)
    monkeypatch.setattr(Arch, "detectHostGfxArchs", _boom)

    # dry-run does not require the config to exist.
    assert E.cmd_gen_logic(_gen_logic_ns(tmp_path, "gfx1151", dry_run=True)) == 0


def test_gen_logic_rejects_placeholder_problem_sizes(tmp_path):
    """A config still carrying extract's Prediction-source placeholder must be
    rejected before benchmarking is attempted."""
    from Tensile.ExperimentalLibrary import cmd_gen_logic

    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(_config_with_problem_sizes([{"Exact": [1, 1, 1, 1]}])))

    with pytest.raises(ExperimentalLibraryError, match="placeholder"):
        cmd_gen_logic(_gen_logic_ns(tmp_path, "gfx950", config=cfg))


def test_gen_logic_placeholder_guard_checked_even_under_dry_run(tmp_path):
    """Unlike the arch guard, the placeholder check runs whenever the config
    file exists, dry-run or not."""
    from Tensile.ExperimentalLibrary import cmd_gen_logic

    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(_config_with_problem_sizes([{"Exact": [1, 1, 1, 1]}])))

    with pytest.raises(ExperimentalLibraryError, match="placeholder"):
        cmd_gen_logic(_gen_logic_ns(tmp_path, "gfx1151", dry_run=True, config=cfg))


# ---------------------------------------------------------------------------
# build-lib: --jobs plumbing
# ---------------------------------------------------------------------------


def test_build_lib_forwards_jobs_to_tensile_create_library(monkeypatch, tmp_path):
    import Tensile.ExperimentalLibrary as E

    logic_dir = tmp_path / "logic"
    logic_dir.mkdir()
    captured = {}

    class _Stop(Exception):
        pass

    def _fake_run_command(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        raise _Stop()

    monkeypatch.setattr(E, "run_command", _fake_run_command)

    ns = argparse.Namespace(
        logic_dir=str(logic_dir), arch="gfx950", out=str(tmp_path / "out"),
        experimental=True, python="python", dry_run=False, verbose=False, jobs=4,
    )
    with pytest.raises(_Stop):
        E.cmd_build_lib(ns)

    cmd = captured["cmd"]
    assert "--jobs" in cmd and "4" in cmd


# ---------------------------------------------------------------------------
# Library-logic structural helpers (_library_type, _find_solutions_element,
# _placeholder_problem_size_groups, _dedup_keys, summarize_solution extras)
# ---------------------------------------------------------------------------


def test_library_type_reads_index_11_else_none():
    raw = [None] * 12
    raw[11] = "Prediction"
    assert _library_type(raw) == "Prediction"
    assert _library_type([None] * 5) is None  # too short to carry index 11
    assert _library_type([None] * 12) is None  # present but falsy


def test_find_solutions_element_raises_when_not_a_list():
    with pytest.raises(ExperimentalLibraryError, match="did not parse"):
        _find_solutions_element({"not": "a list"}, "x.yaml")


def test_find_solutions_element_raises_when_no_solutions_list_found():
    with pytest.raises(ExperimentalLibraryError, match="could not find"):
        _find_solutions_element([1, 2, 3], "x.yaml")


def test_dedup_keys_preserves_first_occurrence_order():
    assert _dedup_keys(["StreamK", "DepthU"], ["DepthU", "MacroTile0"]) == [
        "StreamK",
        "DepthU",
        "MacroTile0",
    ]


def test_summarize_solution_extra_keys_shown_first_and_not_duplicated():
    # "StreamK" is already in _SUMMARY_KEYS; passing it as an extra key must not
    # print it twice.
    summary = summarize_solution({"StreamK": 5, "DepthU": 64}, extra_keys=["StreamK"])
    assert summary.startswith("StreamK=5")
    assert summary.count("StreamK=") == 1


def _config_with_problem_sizes(*problem_sizes_list):
    return {
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM"},
                {"BenchmarkFinalParameters": [{"ProblemSizes": ps}]},
            ]
            for ps in problem_sizes_list
        ]
    }


def test_placeholder_problem_size_groups_detects_only_unedited_groups():
    config = _config_with_problem_sizes(
        [{"Exact": [1, 1, 1, 1]}],  # unedited placeholder -> group 0
        [{"Exact": [256, 256, 1, 256]}],  # real sizes -> not flagged
    )
    assert _placeholder_problem_size_groups(config) == [0]


def test_indexed_out_path_single_vs_multi_index():
    assert _indexed_out_path("base.yaml", 3, single=True) == "base.yaml"
    assert _indexed_out_path("base.yaml", 3, single=False) == "base_3.yaml"
    assert _indexed_out_path("base", 3, single=False) == "base_3.yaml"


def test_apply_overrides_writes_scalar_in_place():
    state = {"PrefetchGlobalRead": 0}
    _apply_overrides(state, [("PrefetchGlobalRead", [1]), ("NewParam", ["x"])])
    assert state == {"PrefetchGlobalRead": 1, "NewParam": "x"}


def test_unique_staged_name_disambiguates_collisions():
    assigned = set()
    assert _unique_staged_name("a.yaml", assigned) == "a.yaml"
    assert _unique_staged_name("a.yaml", assigned) == "a_1.yaml"
    assert _unique_staged_name("a.yaml", assigned) == "a_2.yaml"


# ---------------------------------------------------------------------------
# _resolve_logic_sources
# ---------------------------------------------------------------------------


def _logic_states():
    return [
        {"SolutionIndex": 0, "StreamK": 0, "DepthU": 32, "MacroTile0": 128, "PrefetchGlobalRead": 0},
        {"SolutionIndex": 1, "StreamK": 5, "DepthU": 64, "MacroTile0": 128, "PrefetchGlobalRead": 0},
        {"SolutionIndex": 2, "StreamK": 5, "DepthU": 32, "MacroTile0": 256, "PrefetchGlobalRead": 0},
    ]


def _write_logic_yaml(path, states, library_type="Equality"):
    # Mirrors the shape LibraryIO expects: a raw list with the solution states
    # at index 5 and the LibraryType discriminator at index 11.
    raw = [None] * 12
    raw[5] = states
    raw[11] = library_type
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))


def test_resolve_logic_sources_expands_dirs_and_dedups(tmp_path):
    d = tmp_path / "logic"
    f1 = d / "a.yaml"
    f2 = d / "b.yaml"
    _write_logic_yaml(f1, _logic_states())
    _write_logic_yaml(f2, _logic_states())

    # Passing the dir and one of its files explicitly must not duplicate it.
    resolved = _resolve_logic_sources([str(d), str(f1)])
    assert len(resolved) == 2
    assert os.path.realpath(f1) in resolved
    assert os.path.realpath(f2) in resolved


def test_resolve_logic_sources_missing_path_raises():
    with pytest.raises(ExperimentalLibraryError):
        _resolve_logic_sources(["/no/such/path.yaml"])


# ---------------------------------------------------------------------------
# list-solutions (now fully in-process: no subprocess, no --python/--dry-run)
# ---------------------------------------------------------------------------


def _list_ns(logic_src, where=None, indices_only=False):
    return argparse.Namespace(
        logic_src=logic_src, where=where or [], indices_only=indices_only
    )


def test_list_solutions_prints_matches_with_where_name_first(tmp_path, capsys):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states(), library_type="Equality")

    rc = cmd_list_solutions(_list_ns([str(f)], where=["StreamK=5"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "[Equality]" in out
    assert "2/3 match" in out
    assert "1\tStreamK=5" in out


def test_list_solutions_dir_source_expands_yaml_files(tmp_path, capsys):
    d = tmp_path / "logic"
    _write_logic_yaml(d / "a.yaml", _logic_states())
    _write_logic_yaml(d / "b.yaml", _logic_states())

    cmd_list_solutions(_list_ns([str(d)]))
    out = capsys.readouterr().out
    assert out.count("match) ==") == 2


def test_list_solutions_indices_only_requires_single_resolved_file(tmp_path):
    f1 = tmp_path / "a.yaml"
    f2 = tmp_path / "b.yaml"
    _write_logic_yaml(f1, _logic_states())
    _write_logic_yaml(f2, _logic_states())

    with pytest.raises(ExperimentalLibraryError, match="requires exactly one"):
        cmd_list_solutions(_list_ns([str(f1), str(f2)], indices_only=True))


def test_list_solutions_indices_only_prints_bare_index_list(tmp_path, capsys):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states())

    cmd_list_solutions(_list_ns([str(f)], where=["StreamK=5"], indices_only=True))
    assert capsys.readouterr().out.strip() == "1,2"


# ---------------------------------------------------------------------------
# extract: Prediction-type source warning
# ---------------------------------------------------------------------------


def _extract_ns(logic, out):
    return argparse.Namespace(
        logic=logic, out=out, indices="0", skip_mi=False,
        python="python", dry_run=True, verbose=False,
    )


@pytest.mark.parametrize(
    "library_type,expect_warning", [("Prediction", True), ("Equality", False)]
)
def test_extract_warns_only_for_prediction_type_source(
    tmp_path, capsys, library_type, expect_warning
):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states(), library_type=library_type)

    cmd_extract(_extract_ns(str(f), str(tmp_path / "out.yaml")))
    err = capsys.readouterr().err
    assert ("placeholder" in err) == expect_warning


# ---------------------------------------------------------------------------
# patch-logic
# ---------------------------------------------------------------------------


def _patch_ns(tmp_path, logic_src, **overrides):
    defaults = dict(
        logic_src=logic_src,
        where=[],
        set=["PrefetchGlobalRead=1"],
        matched_pair=False,
        skip_unbuildable=False,
        arch="gfx950",
        out=str(tmp_path / "out"),
        feature_name="feat",
        jobs=None,
        skip_validation=True,
        python="python",
        dry_run=False,
        verbose=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"feature_name": ""}, "feature-name"),
        ({"set": []}, "--set"),
        ({"set": ["StreamK=3,5"]}, "exactly one value"),
    ],
)
def test_patch_logic_argument_validation(tmp_path, overrides, match):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states())
    ns = _patch_ns(tmp_path, [str(f)], **overrides)
    with pytest.raises(ExperimentalLibraryError, match=match):
        cmd_patch_logic(ns)


def test_patch_logic_no_where_warns_and_selects_every_solution(tmp_path, capsys):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states())
    cmd_patch_logic(_patch_ns(tmp_path, [str(f)], where=[], dry_run=True))

    captured = capsys.readouterr()
    assert "no --where given" in captured.err
    assert "3/3 solution(s)" in captured.out


def test_patch_logic_zero_matches_warns(tmp_path, capsys):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states())
    cmd_patch_logic(_patch_ns(tmp_path, [str(f)], where=["StreamK=99"], dry_run=True))
    assert "no solutions matched" in capsys.readouterr().err


def test_patch_logic_dry_run_writes_nothing_but_prints_exports(tmp_path, capsys):
    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states())
    ns = _patch_ns(tmp_path, [str(f)], where=["StreamK=5"], dry_run=True)
    cmd_patch_logic(ns)

    out_root = Path(ns.out)
    assert not (out_root / "patched_logic").exists()
    assert not (out_root / "patch_manifest.csv").exists()
    assert "PATCHED " in capsys.readouterr().out


def test_patch_logic_skip_unbuildable_dry_run_never_probes(tmp_path, monkeypatch):
    import Tensile.ExperimentalLibrary as E

    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states())

    def _boom(*a, **k):
        raise AssertionError("_probe_override called under --dry-run")

    monkeypatch.setattr(E, "_probe_override", _boom)
    ns = _patch_ns(
        tmp_path, [str(f)], where=["StreamK=5"], skip_unbuildable=True, dry_run=True
    )
    assert E.cmd_patch_logic(ns) == 0


def test_patch_logic_applies_override_and_writes_manifest(tmp_path, monkeypatch):
    """End-to-end pass 1-3 + manifest, with the actual TensileCreateLibrary
    build stubbed out (that part is exercised by build-lib's own tests)."""
    import Tensile.ExperimentalLibrary as E

    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states(), library_type="Equality")

    monkeypatch.setattr(E, "cmd_build_lib", lambda ns: 0)
    monkeypatch.setattr(E, "_resolve_lib_dir", lambda lib, arch, must_exist: lib)

    ns = _patch_ns(
        tmp_path, [str(f)], where=["StreamK=5"], set=["PrefetchGlobalRead=1"],
        matched_pair=True,
    )
    assert E.cmd_patch_logic(ns) == 0

    out_root = Path(ns.out)
    patched_files = list((out_root / "patched_logic").rglob("*.yaml"))
    assert len(patched_files) == 1
    patched_states = yaml.safe_load(patched_files[0].read_text())[5]
    assert patched_states[0]["PrefetchGlobalRead"] == 0  # StreamK=0: not matched
    assert patched_states[1]["PrefetchGlobalRead"] == 1  # StreamK=5: matched
    assert patched_states[2]["PrefetchGlobalRead"] == 1  # StreamK=5: matched

    assert list((out_root / "baseline_logic").rglob("*.yaml"))  # --matched-pair

    import csv

    with open(out_root / "patch_manifest.csv") as fh:
        rows = {r[1]: r[2] for r in csv.reader(fh) if r[0] != "logic_file"}
    assert rows == {"1": "applied", "2": "applied"}


def test_patch_logic_skip_unbuildable_keeps_failed_solution_as_baseline(
    tmp_path, monkeypatch
):
    import Tensile.ExperimentalLibrary as E

    f = tmp_path / "logic.yaml"
    _write_logic_yaml(f, _logic_states(), library_type="Equality")

    monkeypatch.setattr(E, "cmd_build_lib", lambda ns: 0)
    monkeypatch.setattr(E, "_resolve_lib_dir", lambda lib, arch, must_exist: lib)

    def _fake_probe(args, raw, sol_pos, override_state, probe_dir):
        # Reject the override on the DepthU=64 solution only.
        if override_state.get("DepthU") == 64:
            return False, "reject"
        return True, ""

    monkeypatch.setattr(E, "_probe_override", _fake_probe)

    ns = _patch_ns(
        tmp_path, [str(f)], where=["StreamK=5"], set=["PrefetchGlobalRead=1"],
        skip_unbuildable=True,
    )
    assert E.cmd_patch_logic(ns) == 0

    out_root = Path(ns.out)
    patched_file = next((out_root / "patched_logic").rglob("*.yaml"))
    patched_states = yaml.safe_load(patched_file.read_text())[5]
    assert patched_states[1]["PrefetchGlobalRead"] == 0  # DepthU=64: probe failed
    assert patched_states[2]["PrefetchGlobalRead"] == 1  # DepthU=32: probe passed

    import csv

    with open(out_root / "patch_manifest.csv") as fh:
        rows = {r[1]: (r[2], r[3]) for r in csv.reader(fh) if r[0] != "logic_file"}
    assert rows["1"] == ("skipped", "reject")
    assert rows["2"] == ("applied", "")
