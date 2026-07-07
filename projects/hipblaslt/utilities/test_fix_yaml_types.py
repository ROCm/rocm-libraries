#!/usr/bin/env python3
"""Tests for fix_yaml_types.py."""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from fix_yaml_types import (
    BOOL_TO_INT_PARAMS,
    INT_TO_BOOL_PARAMS,
    INT_TO_FLOAT_PARAMS,
    count_mismatches,
    fix_content,
    fix_file,
    find_yaml_files,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_yaml(tmpdir, filename, content):
    """Write a YAML file in tmpdir and return its path."""
    path = os.path.join(tmpdir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))
    return path


# ── Tests for fix_content ────────────────────────────────────────────────────

class TestFixContentGroupA:
    """Group A: Bool -> Int (false->0, true->1)."""

    @pytest.mark.parametrize("param", BOOL_TO_INT_PARAMS)
    def test_false_to_zero(self, param):
        content = f"    {param}: false\n"
        result = fix_content(content)
        assert result == f"    {param}: 0\n"

    @pytest.mark.parametrize("param", BOOL_TO_INT_PARAMS)
    def test_true_to_one(self, param):
        content = f"    {param}: true\n"
        result = fix_content(content)
        assert result == f"    {param}: 1\n"

    @pytest.mark.parametrize("param", BOOL_TO_INT_PARAMS)
    def test_int_value_unchanged(self, param):
        """Already-correct int values should not be modified."""
        content = f"    {param}: 0\n"
        assert fix_content(content) == content
        content = f"    {param}: 1\n"
        assert fix_content(content) == content

    def test_unrelated_param_unchanged(self):
        content = "    SomeOtherParam: false\n"
        assert fix_content(content) == content


class TestFixContentGroupB:
    """Group B: Int -> Bool (0->false, 1->true)."""

    @pytest.mark.parametrize("param", INT_TO_BOOL_PARAMS)
    def test_zero_to_false(self, param):
        content = f"    {param}: 0\n"
        result = fix_content(content)
        assert result == f"    {param}: false\n"

    @pytest.mark.parametrize("param", INT_TO_BOOL_PARAMS)
    def test_one_to_true(self, param):
        content = f"    {param}: 1\n"
        result = fix_content(content)
        assert result == f"    {param}: true\n"

    @pytest.mark.parametrize("param", INT_TO_BOOL_PARAMS)
    def test_bool_value_unchanged(self, param):
        """Already-correct bool values should not be modified."""
        content = f"    {param}: false\n"
        assert fix_content(content) == content
        content = f"    {param}: true\n"
        assert fix_content(content) == content

    def test_unrelated_int_param_unchanged(self):
        content = "    DepthU: 1\n"
        assert fix_content(content) == content


class TestFixContentGroupC:
    """Group C: Int -> Float (1 -> 1.0)."""

    def test_int_to_float(self):
        content = "    GlobalReadPerMfma: 1\n"
        result = fix_content(content)
        assert result == "    GlobalReadPerMfma: 1.0\n"

    def test_already_float_unchanged(self):
        content = "    GlobalReadPerMfma: 1.0\n"
        assert fix_content(content) == content

    def test_other_int_unchanged(self):
        """Only value 1 should be converted, not other ints."""
        content = "    GlobalReadPerMfma: 2\n"
        assert fix_content(content) == content

    def test_unrelated_param_unchanged(self):
        content = "    OtherFloat: 1\n"
        assert fix_content(content) == content


# ── Tests for idempotency ────────────────────────────────────────────────────

class TestIdempotency:
    """Running fix_content twice should produce the same result."""

    def test_all_groups_idempotent(self):
        content = textwrap.dedent("""\
            UseCustomMainLoopSchedule: false
            DirectToLds: true
            ExpandPointerSwap: 0
            SourceSwap: 1
            GlobalReadPerMfma: 1
        """)
        first = fix_content(content)
        second = fix_content(first)
        assert first == second

    def test_already_correct_idempotent(self):
        content = textwrap.dedent("""\
            UseCustomMainLoopSchedule: 0
            ExpandPointerSwap: true
            GlobalReadPerMfma: 1.0
        """)
        assert fix_content(content) == content


# ── Tests for end-of-line anchoring ──────────────────────────────────────────

class TestEndOfLineAnchoring:
    """Values not at end-of-line should not be touched."""

    def test_value_in_list_not_touched(self):
        content = "    SomeList: [DirectToLds: false, Other: 1]\n"
        assert fix_content(content) == content

    def test_value_in_comment_not_touched(self):
        content = '    Comment: "DirectToLds: false means disabled"\n'
        assert fix_content(content) == content

    def test_bare_eol_value_is_fixed(self):
        content = "    DirectToLds: false\n"
        result = fix_content(content)
        assert result == "    DirectToLds: 0\n"

    def test_multi_digit_int_not_matched(self):
        """ExpandPointerSwap: 10 should NOT become ExpandPointerSwap: true0."""
        content = "    ExpandPointerSwap: 10\n"
        assert fix_content(content) == content


# ── Tests for count_mismatches ───────────────────────────────────────────────

class TestCountMismatches:

    def test_no_mismatches(self):
        content = "    UseCustomMainLoopSchedule: 0\n    ExpandPointerSwap: true\n"
        assert count_mismatches(content) == (0, 0, 0, 0)

    def test_group_a_counts(self):
        content = "    DirectToLds: false\n    DirectToLds: true\n"
        a, b, c, d = count_mismatches(content)
        assert a == 2
        assert b == 0
        assert c == 0
        assert d == 0

    def test_group_b_counts(self):
        content = "    ExpandPointerSwap: 0\n    SourceSwap: 1\n"
        a, b, c, d = count_mismatches(content)
        assert a == 0
        assert b == 2
        assert c == 0
        assert d == 0

    def test_group_c_counts(self):
        content = "    GlobalReadPerMfma: 1\n"
        a, b, c, d = count_mismatches(content)
        assert a == 0
        assert b == 0
        assert c == 1
        assert d == 0

    def test_group_d_counts(self):
        content = "    CodeObjectVersion: 4\n"
        a, b, c, d = count_mismatches(content)
        assert a == 0
        assert b == 0
        assert c == 0
        assert d == 1

    def test_mixed_groups(self):
        content = textwrap.dedent("""\
            DirectToLds: false
            ExpandPointerSwap: 0
            GlobalReadPerMfma: 1
            CodeObjectVersion: 4
        """)
        assert count_mismatches(content) == (1, 1, 1, 1)


# ── Tests for fix_file ───────────────────────────────────────────────────────

class TestFixFile:

    def test_modifies_file_in_place(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("    DirectToLds: false\n")
        assert fix_file(str(yaml_file)) is True
        assert yaml_file.read_text() == "    DirectToLds: 0\n"

    def test_returns_false_when_no_changes(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("    DirectToLds: 0\n")
        assert fix_file(str(yaml_file)) is False

    def test_preserves_unrelated_content(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        original = "    DepthU: 32\n    DirectToLds: false\n    ThreadTile: [4, 4]\n"
        yaml_file.write_text(original)
        fix_file(str(yaml_file))
        result = yaml_file.read_text()
        assert "DepthU: 32" in result
        assert "ThreadTile: [4, 4]" in result
        assert "DirectToLds: 0" in result


# ── Tests for find_yaml_files ────────────────────────────────────────────────

class TestFindYamlFiles:

    def test_finds_nested_yaml(self, tmp_path):
        subdir = tmp_path / "arch" / "Equality"
        subdir.mkdir(parents=True)
        (subdir / "test.yaml").write_text("content\n")
        (tmp_path / "top.yaml").write_text("content\n")
        result = find_yaml_files(str(tmp_path))
        assert len(result) == 2

    def test_ignores_non_yaml(self, tmp_path):
        (tmp_path / "test.txt").write_text("content\n")
        (tmp_path / "test.yaml").write_text("content\n")
        result = find_yaml_files(str(tmp_path))
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = find_yaml_files(str(tmp_path))
        assert result == []


# ── Tests for mixed correct/incorrect values ─────────────────────────────────

class TestMixedContent:

    def test_mixed_correct_and_incorrect(self):
        content = textwrap.dedent("""\
            - solution1:
                UseCustomMainLoopSchedule: 0
                DirectToLds: false
                ExpandPointerSwap: true
                SourceSwap: 1
            - solution2:
                UseCustomMainLoopSchedule: true
                DirectToLds: 1
                ExpandPointerSwap: 0
                SourceSwap: false
        """)
        result = fix_content(content)
        # Fixed
        assert "DirectToLds: false" not in result
        assert "SourceSwap: 1\n" not in result
        assert "UseCustomMainLoopSchedule: true" not in result
        assert "ExpandPointerSwap: 0\n" not in result
        # Already correct — preserved
        assert "UseCustomMainLoopSchedule: 0" in result
        assert "DirectToLds: 1" in result
        assert "ExpandPointerSwap: true" in result
        assert "SourceSwap: false" in result


# ── Tests for CLI ────────────────────────────────────────────────────────────

class TestCLI:
    """Test the script as a command-line tool."""

    SCRIPT = os.path.join(os.path.dirname(__file__), "fix_yaml_types.py")

    def test_no_args_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT],
            capture_output=True, text=True)
        assert result.returncode == 1
        assert "Usage" in result.stdout

    def test_nonexistent_dir_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "/nonexistent/path"],
            capture_output=True, text=True)
        assert result.returncode == 1
        assert "not a directory" in result.stdout

    def test_clean_dir_exits_zero(self, tmp_path):
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text("    DirectToLds: 0\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "No mismatches to fix" in result.stdout

    def test_fixes_and_reports_success(self, tmp_path):
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text("    DirectToLds: false\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "SUCCESS" in result.stdout
        assert (subdir / "test.yaml").read_text() == "    DirectToLds: 0\n"

    def test_reports_counts(self, tmp_path):
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text(
            "    DirectToLds: false\n    ExpandPointerSwap: 0\n    GlobalReadPerMfma: 1\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, str(tmp_path)],
            capture_output=True, text=True)
        assert "Group A (bool->int):   1" in result.stdout
        assert "Group B (int->bool):   1" in result.stdout
        assert "Group C (int->float):  1" in result.stdout

    def test_mode_flag_input(self, tmp_path):
        """--mode input is reported in the output."""
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text("    DirectToLds: 0\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--mode", "input", str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "Mode: input" in result.stdout

    def test_mode_flag_logic(self, tmp_path):
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text("    DirectToLds: 0\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--mode", "logic", str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "Mode: logic" in result.stdout

    def test_mode_defaults_to_both(self, tmp_path):
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text("    DirectToLds: 0\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "Mode: both" in result.stdout

    def test_mode_invalid_rejected(self, tmp_path):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--mode", "bogus", str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode != 0

    def test_multiple_directories(self, tmp_path):
        """Multiple directories on the CLI are all scanned."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "x.yaml").write_text("    DirectToLds: false\n")
        (b / "y.yaml").write_text("    ExpandPointerSwap: 0\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, str(a), str(b)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert (a / "x.yaml").read_text() == "    DirectToLds: 0\n"
        assert (b / "y.yaml").read_text() == "    ExpandPointerSwap: false\n"

    def test_duplicate_directories_not_double_processed(self, tmp_path):
        """Passing the same directory twice does not double-count files."""
        subdir = tmp_path / "arch"
        subdir.mkdir()
        (subdir / "test.yaml").write_text("    DirectToLds: false\n")
        result = subprocess.run(
            [sys.executable, self.SCRIPT, str(tmp_path), str(tmp_path)],
            capture_output=True, text=True)
        assert result.returncode == 0
        assert "YAML files found: 1" in result.stdout
