"""Unit tests for ``shared/ctest/parse_catch2_categories.py`` (Catch2 parser).

Covers:
  * Validator helpers (``validate_tag``, ``validate_identifier``,
    ``validate_categories``).
  * ``build_catch2_tag_expression`` (the heart of the Catch2 logic; the
    operator-precedence comment in the parser is the spec).
  * YAML loader error paths.
  * End-to-end CLI behaviour against representative fixtures.

Run with::

    pytest -q shared/ctest/tests/test_parse_catch2_categories.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import parse_catch2_categories as pcc
from conftest import extract_add_test_blocks, parse_install_file


# ---------------------------------------------------------------------------
# validate_tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag",
    [
        "[smoke]",
        "[unit]",
        "[]",  # the "all tests" sentinel
        "~[slow]",
        "[a-b.c_d]",
        "[abc*]",
        "[abc/def]",
        "~[a-b.c_d]",
    ],
)
def test_validate_tag_accepts_valid_tags(tag):
    assert pcc.validate_tag(tag) is None


@pytest.mark.parametrize(
    "tag",
    [
        "smoke",  # no brackets
        "[smoke",  # unclosed bracket
        "smoke]",  # unopened bracket
        "[smoke][unit]",  # two tags concatenated -- must be in a list, not a string
        "[smoke ]",  # space inside brackets
        "[smoke;]",
        "[smoke|]",
        "",
        "~smoke",  # negation without brackets
    ],
)
def test_validate_tag_rejects_invalid_tags(tag):
    assert pcc.validate_tag(tag) is not None


@pytest.mark.parametrize("tag", [123, None, ["[smoke]"], {"a": 1}])
def test_validate_tag_rejects_non_strings(tag):
    err = pcc.validate_tag(tag)
    assert err is not None
    assert "must be a string" in err


# ---------------------------------------------------------------------------
# validate_identifier (Catch2 parser has its own copy; assert it behaves the
# same as the GTest parser's copy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["quick", "ex_gpu_gfx1150", "pre-commit", "v1.2.3"])
def test_validate_identifier_accepts_safe_strings(value):
    assert pcc.validate_identifier(value) is None


@pytest.mark.parametrize("value", ["bad name", "has/slash", "has*", ""])
def test_validate_identifier_rejects_unsafe_strings(value):
    assert pcc.validate_identifier(value) is not None


# ---------------------------------------------------------------------------
# validate_categories
# ---------------------------------------------------------------------------


def test_validate_categories_accepts_minimal_config():
    categories = {"quick": {"test_tags": ["[smoke]"], "labels": ["quick"]}}
    assert pcc.validate_categories(categories, False, True) == []


def test_validate_categories_rejects_non_mapping():
    errors = pcc.validate_categories(["not", "a", "dict"], False, True)
    assert any("must be a mapping" in e for e in errors)


def test_validate_categories_rejects_invalid_tag_in_test_tags():
    categories = {"quick": {"test_tags": ["[smoke"], "labels": ["quick"]}}
    errors = pcc.validate_categories(categories, False, True)
    assert any("test_tags" in e and "Invalid tag syntax" in e for e in errors)


def test_validate_categories_rejects_invalid_tag_in_exclude_tags():
    categories = {
        "quick": {
            "test_tags": ["[smoke]"],
            "exclude_tags": ["not a tag"],
            "labels": ["quick"],
        }
    }
    errors = pcc.validate_categories(categories, False, True)
    assert any("exclude_tags" in e for e in errors)


def test_validate_categories_applies_os_specific_excludes_linux():
    categories = {
        "quick": {
            "test_tags": ["[smoke]"],
            "exclude_tags_linux": ["not a tag"],  # only inspected on Linux
            "labels": ["quick"],
        }
    }
    win_errors = pcc.validate_categories(categories, True, False)
    lin_errors = pcc.validate_categories(categories, False, True)
    assert win_errors == []
    assert any("exclude_tags" in e for e in lin_errors)


def test_validate_categories_rejects_unsafe_label():
    categories = {"quick": {"test_tags": ["[smoke]"], "labels": ["bad label"]}}
    errors = pcc.validate_categories(categories, False, True)
    assert any("label" in e for e in errors)


def test_validate_categories_collects_multiple_errors():
    categories = {
        "bad name": {
            "test_tags": ["[smoke", "not_a_tag"],
            "exclude_tags": ["also not a tag"],
            "labels": ["bad label"],
        }
    }
    errors = pcc.validate_categories(categories, False, True)
    # Expect at least: category name + 2 bad test_tags + 1 bad exclude + 1 bad label.
    assert len(errors) >= 4


# ---------------------------------------------------------------------------
# build_catch2_tag_expression
# ---------------------------------------------------------------------------


def test_build_expression_only_includes():
    assert (
        pcc.build_catch2_tag_expression(["[smoke]", "[unit]"], []) == "[smoke],[unit]"
    )


def test_build_expression_only_excludes():
    # Excludes without includes => single AND-joined exclude clause.
    assert (
        pcc.build_catch2_tag_expression([], ["[slow]", "[flaky]"]) == "~[slow] ~[flaky]"
    )


def test_build_expression_include_and_exclude_distributes_excludes():
    """Catch2's grammar: ',' is OR, space is AND, '~' negates.  Because ',' binds
    looser than ' ', excludes must be repeated per include clause to get the
    intended (a OR b) AND NOT c semantics.
    """
    result = pcc.build_catch2_tag_expression(["[smoke]", "[unit]"], ["[slow]"])
    assert result == "[smoke] ~[slow],[unit] ~[slow]"


def test_build_expression_handles_multiple_excludes():
    result = pcc.build_catch2_tag_expression(["[a]", "[b]"], ["[x]", "[y]"])
    assert result == "[a] ~[x] ~[y],[b] ~[x] ~[y]"


def test_build_expression_skips_all_tests_sentinel():
    # "[]" stands for "run everything"; it must NOT appear as an include clause.
    assert pcc.build_catch2_tag_expression(["[]"], []) == ""


def test_build_expression_empty_inputs_returns_empty_string():
    assert pcc.build_catch2_tag_expression([], []) == ""
    assert pcc.build_catch2_tag_expression(None, None) == ""


def test_build_expression_sentinel_with_excludes_returns_excludes_only():
    # If the only "include" is the [] sentinel, the expression collapses to
    # the bare exclude clause.
    assert pcc.build_catch2_tag_expression(["[]"], ["[slow]"]) == "~[slow]"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_load_yaml_loads_valid_file(fixtures_dir):
    data = pcc.load_yaml(fixtures_dir / "catch2_minimal.yaml")
    assert "test_categories" in data
    assert "quick" in data["test_categories"]


def test_load_yaml_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        pcc.load_yaml(tmp_path / "missing.yaml")
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# End-to-end CLI
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
        str(parser_dir / "parse_catch2_categories.py"),
        str(yaml_path),
        target,
        str(workdir),
    ]
    if install_file is not None:
        cmd.append(str(install_file))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_cli_minimal_yaml_generates_expected_test(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(
        parser_dir, fixtures_dir / "catch2_minimal.yaml", "rr_tests", tmp_path
    )
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert "rr_tests_quick_suite" in tests
    quick = tests["rr_tests_quick_suite"]
    # Tag expression is the COMMAND's quoted argument: [smoke],[unit]
    assert '"[smoke],[unit]"' in quick["command"]
    assert quick["labels"] == ["quick", "pre-commit"]
    assert quick["timeout"] == 300  # default


def test_cli_full_yaml_generates_all_categories(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(parser_dir, fixtures_dir / "catch2_full.yaml", "t", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert {
        "t_quick_suite",
        "t_standard_suite",
        "t_all_tests_suite",
        "t_excludes_only_suite",
    } <= set(tests)


def test_cli_full_yaml_applies_timeout_multiplier(parser_dir, fixtures_dir, tmp_path):
    # catch2_full.yaml: multiplier=2, quick=60, standard=600, comprehensive=1800
    res = _run_parser(parser_dir, fixtures_dir / "catch2_full.yaml", "t", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert tests["t_quick_suite"]["timeout"] == 120
    assert tests["t_standard_suite"]["timeout"] == 1200
    assert tests["t_excludes_only_suite"]["timeout"] == 600


def test_cli_full_yaml_distributes_excludes_per_include(
    parser_dir, fixtures_dir, tmp_path
):
    """The 'quick' category in catch2_full.yaml has [smoke]+[unit] includes and
    [slow]+(linux: [windows-only], windows: [linux-only]) excludes.  The Catch2
    expression must repeat the excludes across each include clause.
    """
    res = _run_parser(parser_dir, fixtures_dir / "catch2_full.yaml", "t", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    cmd = tests["t_quick_suite"]["command"]
    # Each include clause must contain "~[slow]".
    assert cmd.count("~[slow]") == 2


def test_cli_full_yaml_all_tests_category_uses_bare_command(
    parser_dir, fixtures_dir, tmp_path
):
    """When the only include tag is the [] sentinel and there are no
    excludes, no filter argument should be passed -- the COMMAND must run the
    binary bare so Catch2 picks up every registered test.
    """
    res = _run_parser(parser_dir, fixtures_dir / "catch2_full.yaml", "t", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    cmd = tests["t_all_tests_suite"]["command"]
    # No quoted tag expression; just the bare target name.
    assert cmd == "t"


def test_cli_full_yaml_excludes_only_category(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(parser_dir, fixtures_dir / "catch2_full.yaml", "t", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    cmd = tests["t_excludes_only_suite"]["command"]
    assert '"~[slow] ~[flaky]"' in cmd


def test_cli_full_yaml_propagates_environment(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(parser_dir, fixtures_dir / "catch2_full.yaml", "t", tmp_path)
    assert res.returncode == 0, res.stderr
    tests = extract_add_test_blocks(res.stdout)
    assert tests["t_quick_suite"]["environment"] == "CATCH2_LOG_LEVEL=info"
    assert tests["t_all_tests_suite"]["environment"] == "CATCH2_LOG_LEVEL=info"


def test_cli_invalid_tag_yaml_fails(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(
        parser_dir, fixtures_dir / "catch2_invalid_tag.yaml", "t", tmp_path
    )
    assert res.returncode == 1
    assert "Invalid tag syntax" in res.stderr
    assert "add_test(" not in res.stdout


def test_cli_invalid_identifier_yaml_fails(parser_dir, fixtures_dir, tmp_path):
    res = _run_parser(
        parser_dir, fixtures_dir / "catch2_invalid_identifier.yaml", "t", tmp_path
    )
    assert res.returncode == 1
    assert "bad/name" in res.stderr or "unsafe characters" in res.stderr


def test_cli_install_file_writes_relative_path_tests(
    parser_dir, fixtures_dir, tmp_path
):
    install_file = tmp_path / "install_CTestTestfile.cmake"
    res = _run_parser(
        parser_dir,
        fixtures_dir / "catch2_minimal.yaml",
        "rr_tests",
        tmp_path,
        install_file=install_file,
    )
    assert res.returncode == 0, res.stderr
    assert install_file.exists()

    install_tests = parse_install_file(install_file)
    assert "rr_tests_quick_suite" in install_tests
    block = install_tests["rr_tests_quick_suite"]
    # Install-tree binary path is relative ("../<target>").
    assert block["command_line"].startswith('"../rr_tests"')
    # Tag expression carried through as quoted positional arg.
    assert '"[smoke],[unit]"' in block["command_line"]
    assert block["labels"] == ["quick", "pre-commit"]


def test_cli_install_file_for_all_tests_category_has_no_tag_arg(
    parser_dir, fixtures_dir, tmp_path
):
    """The "[]" sentinel category collapses to a bare binary invocation; the
    install file must mirror that and emit just ``"../<target>"``.
    """
    install_file = tmp_path / "install_CTestTestfile.cmake"
    res = _run_parser(
        parser_dir,
        fixtures_dir / "catch2_full.yaml",
        "t",
        tmp_path,
        install_file=install_file,
    )
    assert res.returncode == 0, res.stderr

    install_tests = parse_install_file(install_file)
    assert "t_all_tests_suite" in install_tests
    cmd_line = install_tests["t_all_tests_suite"]["command_line"]
    assert cmd_line == '"../t"'


def test_cli_missing_yaml_file_fails(parser_dir, tmp_path):
    res = _run_parser(parser_dir, tmp_path / "missing.yaml", "t", tmp_path)
    assert res.returncode == 1
    assert "YAML file not found" in res.stderr
