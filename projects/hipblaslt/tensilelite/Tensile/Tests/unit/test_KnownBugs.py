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
load_known_bugs = _kb.load_known_bugs
normalize_logic_relative_path = _kb.normalize_logic_relative_path


def test_normalize_logic_relative_path():
    assert normalize_logic_relative_path(Path("a/b")) == "a/b"


def test_load_known_bugs_missing_file(tmp_path):
    assert load_known_bugs(tmp_path / "none.yaml") == frozenset()


def test_load_known_bugs_roundtrip(tmp_path):
    p = tmp_path / "kb.yaml"
    p.write_text(
        """
version: 1
# ROCM-9999: example
skips:
  - path: foo/bar.yaml
    solution_index: 3
    ticket: ROCM-9999
""",
        encoding="utf-8",
    )
    kb = load_known_bugs(p)
    assert ("foo/bar.yaml", 3) in kb
    assert is_known_bug(kb, Path("foo/bar.yaml"), 3)
    assert not is_known_bug(kb, Path("foo/bar.yaml"), 4)


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
