"""Unit tests for ``shared/ctest/parse_test_categories.py`` (GTest parser).

Covers:
  * Validator helpers (``validate_identifier``, ``validate_gtest_pattern``,
    ``validate_config``).
  * Hierarchical GPU-arch matching (``gpu_arch_matches``).
  * YAML loader error paths (``load_yaml``).
  * End-to-end CLI behaviour via ``subprocess.run`` against representative
    fixtures, asserting on:
      - generated build-tree CMake (stdout)
      - generated install-tree CTestTestfile.cmake (4th positional arg)
      - exit codes for validation / YAML errors

Run with::

    pytest -q shared/ctest/tests/test_parse_test_categories.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import parse_test_categories as ptc
from conftest import extract_add_test_blocks, parse_install_file


# ---------------------------------------------------------------------------
# Validator helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "quick",
        "ex_gpu_gfx1150",
        "pre-commit",
        "v1.2.3",
        "category_with_under_scores",
        "abc-def.ghi_jkl",
    ],
)
def test_validate_identifier_accepts_safe_strings(value):
    assert ptc.validate_identifier(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "bad label",
        "has space",
        "has/slash",
        "has\\backslash",
        "has*wildcard",
        "has:colon",
        "has;semi",
        "",
    ],
)
def test_validate_identifier_rejects_unsafe_strings(value):
    err = ptc.validate_identifier(value)
    assert err is not None
    assert "unsafe characters" in err or "Identifier" in err


@pytest.mark.parametrize("value", [123, None, ["a"], {"a": 1}, 1.5])
def test_validate_identifier_rejects_non_strings(value):
    err = ptc.validate_identifier(value)
    assert err is not None
    assert "must be a string" in err


@pytest.mark.parametrize(
    "pattern",
    [
        "Smoke.*",
        "*KnownBroken*",
        "Suite/Group.TestName",
        "*",
        "abc-def_ghi.jkl",
        "Group/*",
        "exact.test_name",
    ],
)
def test_validate_gtest_pattern_accepts_safe_patterns(pattern):
    assert ptc.validate_gtest_pattern(pattern) is None


@pytest.mark.parametrize(
    "pattern",
    [
        "has space",
        "bad:colon",
        "bad;semi",
        "bad|pipe",
        "bad`tick",
        "bad$dollar",
        "bad(paren)",
    ],
)
def test_validate_gtest_pattern_rejects_unsafe_patterns(pattern):
    err = ptc.validate_gtest_pattern(pattern)
    assert err is not None
    assert "Invalid gtest pattern" in err


def test_validate_config_accepts_minimal_valid_config():
    categories = {"quick": {"test_patterns": ["Smoke.*"], "labels": ["quick"]}}
    assert ptc.validate_config(categories, None, False, True) == []


def test_validate_config_rejects_non_mapping_categories():
    errors = ptc.validate_config(["not", "a", "mapping"], None, False, True)
    assert any("must be a mapping" in e for e in errors)


def test_validate_config_rejects_non_mapping_category_entry():
    categories = {"quick": "not a dict"}
    errors = ptc.validate_config(categories, None, False, True)
    assert any("entry must be a mapping" in e for e in errors)


def test_validate_config_collects_multiple_errors():
    categories = {
        "bad name": {  # invalid category name (space)
            "test_patterns": ["bad pattern"],  # invalid pattern (space)
            "labels": ["bad label"],  # invalid label (space)
        }
    }
    errors = ptc.validate_config(categories, None, False, True)
    # We expect at least one error per offending field: identifier (name),
    # pattern, identifier (label).
    assert len(errors) >= 3
    assert any("category name" in e for e in errors)
    assert any("test_patterns" in e for e in errors)
    assert any("label" in e for e in errors)


def test_validate_config_applies_os_specific_excludes_linux():
    categories = {
        "quick": {
            "test_patterns": ["Good.*"],
            "exclude_linux": ["bad pattern"],  # only validated on Linux
            "labels": ["quick"],
        }
    }
    win_errors = ptc.validate_config(categories, None, True, False)
    lin_errors = ptc.validate_config(categories, None, False, True)
    # On Windows the bad linux-only pattern is ignored; on Linux it errors.
    assert win_errors == []
    assert any("exclude" in e for e in lin_errors)


def test_validate_config_handles_exclude_gpu_list_of_lists():
    categories = {"quick": {"test_patterns": ["Smoke.*"], "labels": ["quick"]}}
    exclude_gpu = {
        "exclude_gpu_gfx11X": {
            # YAML anchor expansion in fixtures can produce list-of-lists; the
            # validator should descend into nested lists.
            "test_patterns": [["*Pat1*", "*Pat2*"], "*Pat3*"],
            "labels": ["quick", "ex_gpu_gfx11X"],
        }
    }
    assert ptc.validate_config(categories, exclude_gpu, False, True) == []


def test_validate_config_rejects_bad_exclude_gpu_pattern():
    categories = {"quick": {"test_patterns": ["Smoke.*"], "labels": ["quick"]}}
    exclude_gpu = {
        "exclude_gpu_gfx11X": {
            "test_patterns": ["bad pattern"],
            "labels": ["quick", "ex_gpu_gfx11X"],
        }
    }
    errors = ptc.validate_config(categories, exclude_gpu, False, True)
    assert any("exclude_gpu" in e and "test_patterns" in e for e in errors)


# ---------------------------------------------------------------------------
# gpu_arch_matches: hierarchical GPU pattern matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "specific,pattern,expected",
    [
        # Exact matches
        ("gfx1150", "gfx1150", True),
        ("gfx942", "gfx942", True),
        # Wildcard family matches
        ("gfx1150", "gfx11X", True),
        ("gfx1151", "gfx11X", True),
        ("gfx1100", "gfx11X", True),
        ("gfx1150", "gfx115X", True),
        ("gfx1151", "gfx115X", True),
        # Non-matches across families
        ("gfx950", "gfx11X", False),
        ("gfx1150", "gfx9X", False),
        ("gfx942", "gfx95X", False),
        # No-wildcard non-match falls through to False
        ("gfx1150", "gfx1151", False),
    ],
)
def test_gpu_arch_matches(specific, pattern, expected):
    assert ptc.gpu_arch_matches(specific, pattern) is expected


# ---------------------------------------------------------------------------
# YAML loader: success + error paths
# ---------------------------------------------------------------------------


def test_load_yaml_loads_valid_file(fixtures_dir):
    data = ptc.load_yaml(fixtures_dir / "gtest_minimal.yaml")
    assert "test_categories" in data
    assert "quick" in data["test_categories"]


def test_load_yaml_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        ptc.load_yaml(tmp_path / "does_not_exist.yaml")
    assert excinfo.value.code == 1


def test_load_yaml_malformed_file_exits(fixtures_dir):
    with pytest.raises(SystemExit) as excinfo:
        ptc.load_yaml(fixtures_dir / "malformed.yaml")
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# End-to-end CLI: parse_test_categories.py invoked via subprocess
# ---------------------------------------------------------------------------


def _run_parser(
    parser_dir: Path,
    yaml_path: Path,
    target: str,
    workdir: Path,
    install_file: Path | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(parser_dir / "parse_test_categories.py"),
        str(yaml_path),
        target,
        str(workdir),
    ]
    if install_file is not None:
        cmd.append(str(install_file))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_cli_minimal_yaml_generates_expected_test(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(
        parser_dir, fixtures_dir / "gtest_minimal.yaml", "my_target", tmp_path
    )
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert "my_target_quick_suite" in tests
    quick = tests["my_target_quick_suite"]
    # Patterns joined with ':' and no excludes -> no trailing '-'.
    assert "--gtest_filter=Smoke.*:Unit.*" in quick["command"]
    assert quick["labels"] == ["quick", "pre-commit"]
    assert quick["timeout"] == 300  # default (no per-category timeout in YAML)
    assert quick["environment"] is None
    assert quick["working_directory"] == str(tmp_path)


def test_cli_full_yaml_generates_all_categories_and_gpu_excludes(
    parser_dir, fixtures_dir, tmp_path
):
    res = _run_parser(parser_dir, fixtures_dir / "gtest_full.yaml", "target", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)

    # All four tiered category suites must be emitted.
    for cat in ("quick", "standard", "comprehensive", "full"):
        assert f"target_{cat}_suite" in tests, f"missing {cat} category"

    # Per-GPU variants should exist for any (category, arch) pair where the
    # arch's exclude_gpu entry lists that category as a label.
    # exclude_gpu_gfx950 only lists "quick" -> only quick_gfx950 expected.
    assert "target_quick_gfx950_suite" in tests
    assert "target_standard_gfx950_suite" not in tests
    # exclude_gpu_gfx1150 lists quick + standard -> both variants.
    assert "target_quick_gfx1150_suite" in tests
    assert "target_standard_gfx1150_suite" in tests
    # gfx11X applies to all four categories.
    for cat in ("quick", "standard", "comprehensive", "full"):
        assert f"target_{cat}_gfx11X_suite" in tests


def test_cli_full_yaml_applies_timeout_multiplier(parser_dir, fixtures_dir, tmp_path):
    # gtest_full.yaml: multiplier=2, category_timeouts: quick=300, standard=1800.
    res = _run_parser(parser_dir, fixtures_dir / "gtest_full.yaml", "target", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert tests["target_quick_suite"]["timeout"] == 600
    assert tests["target_standard_suite"]["timeout"] == 3600


def test_cli_full_yaml_propagates_environment(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(parser_dir, fixtures_dir / "gtest_full.yaml", "target", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    env = tests["target_quick_suite"]["environment"]
    assert env is not None
    # Order isn't guaranteed by dict iteration in older Pythons, so check parts.
    assert "OPENBLAS_NUM_THREADS=1" in env
    assert "OMP_NUM_THREADS=4" in env


def test_cli_full_yaml_emits_os_specific_excludes(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(parser_dir, fixtures_dir / "gtest_full.yaml", "target", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    cmd = tests["target_quick_suite"]["command"]
    # Common excludes (anchor-shared) appear in the negative filter portion.
    assert "*KnownBroken*" in cmd
    assert "*Flaky*" in cmd
    # On Linux: exclude_linux additions present; exclude_windows additions absent.
    import platform

    if platform.system() == "Linux":
        assert "*WindowsOnly*" in cmd
        assert "*LinuxOnly*" not in cmd
    elif platform.system() == "Windows":
        assert "*LinuxOnly*" in cmd
        assert "*WindowsOnly*" not in cmd


def test_cli_full_yaml_gpu_exclude_combines_with_category_exclude(
    parser_dir, fixtures_dir, tmp_path
):
    """For a GPU-specific variant of a category that already has category
    excludes, the resulting --gtest_filter must contain BOTH sets of excludes
    after the '-' separator.
    """
    res = _run_parser(parser_dir, fixtures_dir / "gtest_full.yaml", "target", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    cmd = tests["target_quick_gfx1150_suite"]["command"]
    # Category excludes (from `quick.exclude` + linux/windows-only).
    assert "*KnownBroken*" in cmd
    assert "*Flaky*" in cmd
    # GPU-family excludes (gfx11X).
    assert "*UnsupportedOnGfx11*" in cmd
    assert "*KnownGPUFailure*" in cmd
    # GPU-specific extras (gfx1150).
    assert "*Gfx1150SpecificFailure*" in cmd
    # ex_gpu_<arch> label must be appended to the category labels.
    assert "ex_gpu_gfx1150" in tests["target_quick_gfx1150_suite"]["labels"]


def test_cli_no_gpu_yaml_emits_no_gpu_suites(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(
        parser_dir, fixtures_dir / "gtest_no_gpu.yaml", "target", tmp_path
    )
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert {"target_quick_suite", "target_standard_suite"} <= set(tests)
    assert not any("gfx" in name for name in tests)


def test_cli_empty_patterns_emits_warning_and_skips_category(
    parser_dir, fixtures_dir, tmp_path
):
    res = _run_parser(
        parser_dir, fixtures_dir / "gtest_empty_patterns.yaml", "target", tmp_path
    )
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert "target_quick_suite" in tests
    assert "target_empty_suite" not in tests
    assert "has no test_patterns" in res.stderr


def test_cli_invalid_gtest_pattern_yaml_fails_with_clear_error(
    parser_dir, fixtures_dir, tmp_path
):
    res = _run_parser(
        parser_dir, fixtures_dir / "gtest_invalid_pattern.yaml", "target", tmp_path
    )
    assert res.returncode == 1
    assert "Invalid gtest pattern" in res.stderr
    assert "Bad Pattern With Spaces" in res.stderr
    # No CMake output should be emitted on validation failure (atomicity).
    assert "add_test(" not in res.stdout


def test_cli_invalid_identifier_yaml_fails_with_clear_error(
    parser_dir, fixtures_dir, tmp_path
):
    res = _run_parser(
        parser_dir, fixtures_dir / "gtest_invalid_identifier.yaml", "target", tmp_path
    )
    assert res.returncode == 1
    assert "unsafe characters" in res.stderr or "Identifier" in res.stderr
    assert "bad label" in res.stderr
    assert "add_test(" not in res.stdout


def test_cli_missing_yaml_file_fails(parser_dir, tmp_path):
    res = _run_parser(parser_dir, tmp_path / "missing.yaml", "target", tmp_path)
    assert res.returncode == 1
    assert "YAML file not found" in res.stderr


def test_cli_install_file_argument_writes_relative_path_tests(
    parser_dir, fixtures_dir, tmp_path
):
    install_file = tmp_path / "install_CTestTestfile.cmake"
    res = _run_parser(
        parser_dir,
        fixtures_dir / "gtest_minimal.yaml",
        "my_target",
        tmp_path,
        install_file=install_file,
    )
    assert res.returncode == 0, res.stderr
    assert install_file.exists()

    install_tests = parse_install_file(install_file)
    assert "my_target_quick_suite" in install_tests
    block = install_tests["my_target_quick_suite"]
    # Install-tree command line MUST use a relative path "../<target>" so the
    # generated file works when ctest is invoked from bin/<component>/.
    assert block["command_line"].startswith('"../my_target"')
    assert "--gtest_filter=Smoke.*:Unit.*" in block["command_line"]
    assert block["labels"] == ["quick", "pre-commit"]
    assert block["timeout"] == 300


def test_cli_install_file_appends_on_repeat_invocation(
    parser_dir, fixtures_dir, tmp_path
):
    """The parser opens the install file in append mode; back-to-back runs
    against the same file (e.g. for multiple targets in one project) must not
    overwrite previous content. We assert each target's tests survive.
    """
    install_file = tmp_path / "install_CTestTestfile.cmake"
    for target in ("target_a", "target_b"):
        res = _run_parser(
            parser_dir,
            fixtures_dir / "gtest_minimal.yaml",
            target,
            tmp_path,
            install_file=install_file,
        )
        assert res.returncode == 0, res.stderr

    install_tests = parse_install_file(install_file)
    assert "target_a_quick_suite" in install_tests
    assert "target_b_quick_suite" in install_tests
