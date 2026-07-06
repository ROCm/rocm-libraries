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

import pytest

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
