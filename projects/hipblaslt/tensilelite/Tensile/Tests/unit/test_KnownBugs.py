################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################

import importlib.util
from pathlib import Path

import pytest

# Load KnownBugs.py without importing TensileLogic/__init__.py (avoids rocisa in CI).
def _known_bugs_mod():
    kb_path = Path(__file__).resolve().parents[2] / "TensileLogic" / "KnownBugs.py"
    spec = importlib.util.spec_from_file_location("KnownBugs_under_test", kb_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kb = _known_bugs_mod()
is_known_bug = _kb.is_known_bug
load_bundled_known_bugs = _kb.load_bundled_known_bugs
load_known_bugs = _kb.load_known_bugs
normalize_logic_relative_path = _kb.normalize_logic_relative_path


def test_normalize_logic_relative_path():
    assert normalize_logic_relative_path(Path("a/b")) == "a/b"


def test_load_known_bugs_missing_file(tmp_path):
    assert load_known_bugs(tmp_path / "none.yaml") == frozenset()


def test_load_known_bugs_none_does_not_read_bundled_resource(monkeypatch):
    def fail_if_called():
        raise AssertionError("bundled resource should require an explicit opt-in")

    monkeypatch.setattr(_kb, "known_bugs_text", fail_if_called)

    assert load_known_bugs(None) == frozenset()


def test_load_bundled_known_bugs_uses_resource(monkeypatch):
    monkeypatch.setattr(
        _kb,
        "known_bugs_text",
        lambda: "skips:\n  - path: bundled/logic.yaml\n    solution_name: NameA\n",
    )

    assert load_bundled_known_bugs() == frozenset({("bundled/logic.yaml", "NameA")})


def test_load_known_bugs_roundtrip(tmp_path):
    p = tmp_path / "kb.yaml"
    p.write_text(
        """
version: 1
# ROCM-9999: example
skips:
  - path: foo/bar.yaml
    solution_name: Cijk_Ailk_Bljk_ExampleName
    ticket: ROCM-9999
""",
        encoding="utf-8",
    )
    kb = load_known_bugs(p)
    assert ("foo/bar.yaml", "Cijk_Ailk_Bljk_ExampleName") in kb
    assert is_known_bug(kb, Path("foo/bar.yaml"), "Cijk_Ailk_Bljk_ExampleName")
    assert not is_known_bug(kb, Path("foo/bar.yaml"), "Cijk_Ailk_Bljk_OtherName")
    # A None solution name (solution has no SolutionNameMin) never matches.
    assert not is_known_bug(kb, Path("foo/bar.yaml"), None)


def test_load_known_bugs_requires_solution_name(tmp_path):
    # An entry keyed only on the old solution_index is no longer accepted.
    p = tmp_path / "legacy.yaml"
    p.write_text(
        "version: 1\nskips:\n  - path: foo/bar.yaml\n    solution_index: 3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires string 'solution_name'"):
        load_known_bugs(p)


def test_load_known_bugs_invalid(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "version: 1\nskips: not-a-list\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_known_bugs(p)


def test_load_known_bugs_requires_pyyaml(tmp_path, monkeypatch):
    p = tmp_path / "kb.yaml"
    p.write_text("version: 1\nskips: []\n", encoding="utf-8")
    monkeypatch.setattr(_kb, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML"):
        load_known_bugs(p)
