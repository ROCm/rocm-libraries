################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``TensileLogic.KnownBugs``.

The existing ``test_KnownBugs.py`` (loaded via importlib) covers the happy
round-trip, the missing-file return, the ``skips: not-a-list`` error, and the
missing-PyYAML guard. This suite pins the remaining branches the baseline left
uncovered — every input-validation path of ``load_known_bugs`` — plus the two
pure helpers, using syrupy snapshots of the structured result.

Determinism: ``load_known_bugs`` returns a ``frozenset`` (unordered); every
snapshot sorts it first. YAML inputs are written to ``tmp_path``.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.KnownBugs import (
    is_known_bug,
    load_known_bugs,
    normalize_logic_relative_path,
)

pytestmark = pytest.mark.unit


def _sorted(known):
    """Stable, snapshot-friendly view of an unordered frozenset of keys."""
    return sorted(known)


# --- pure helpers -----------------------------------------------------------

@pytest.mark.parametrize("name,raw", [
    ("posix_kept", "a/b/c.yaml"),
    ("backslashes_split", "a\\b\\c.yaml"),
    ("nested", "gfx942/aquavanjaram/Equality/logic.yaml"),
    ("single", "logic.yaml"),
])
def test_normalize_logic_relative_path(name, raw, snapshot):
    assert normalize_logic_relative_path(Path(raw)) == snapshot(name=name)


def test_is_known_bug_hit_and_miss(snapshot):
    # Build the known set explicitly to pin is_known_bug independently of the
    # YAML loader.
    known = frozenset({(normalize_logic_relative_path(Path("foo/bar.yaml")), 3)})
    result = {
        "hit_same": is_known_bug(known, Path("foo/bar.yaml"), 3),
        "hit_str_index": is_known_bug(known, Path("foo/bar.yaml"), "3"),
        "miss_index": is_known_bug(known, Path("foo/bar.yaml"), 4),
        "miss_path": is_known_bug(known, Path("foo/other.yaml"), 3),
    }
    assert result == snapshot


# --- load_known_bugs: frozenset-returning branches --------------------------

def test_load_none_returns_empty(snapshot):
    # config_path is None -> early empty frozenset (L57-58).
    assert _sorted(load_known_bugs(None)) == snapshot


def test_load_empty_yaml_returns_empty(tmp_path, snapshot):
    # File parses to None (raw is None) -> empty frozenset (L70-71).
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert _sorted(load_known_bugs(p)) == snapshot


def test_load_no_skips_key_returns_empty(tmp_path, snapshot):
    # Mapping without a "skips" key -> empty frozenset (L79-80).
    p = tmp_path / "noskips.yaml"
    p.write_text("version: 1\n", encoding="utf-8")
    assert _sorted(load_known_bugs(p)) == snapshot


def test_load_null_skips_returns_empty(tmp_path, snapshot):
    # "skips:" with no value parses to None -> empty frozenset (L79-80).
    p = tmp_path / "nullskips.yaml"
    p.write_text("skips:\n", encoding="utf-8")
    assert _sorted(load_known_bugs(p)) == snapshot


def test_load_roundtrip_multi(tmp_path, snapshot):
    # Multiple entries; extra keys (ticket) ignored. Snapshot sorted keys.
    p = tmp_path / "kb.yaml"
    p.write_text(
        "skips:\n"
        "  - path: foo/bar.yaml\n"
        "    solution_index: 3\n"
        "    ticket: ROCM-9999\n"
        "  - path: baz/qux.yaml\n"
        "    solution_index: 12\n",
        encoding="utf-8",
    )
    assert _sorted(load_known_bugs(p)) == snapshot


# --- load_known_bugs: ValueError branches -----------------------------------

def test_load_non_mapping_top_level_raises(tmp_path):
    # Top-level is a list, not a mapping (L73-74).
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at the top level"):
        load_known_bugs(p)


def test_load_entry_not_mapping_raises(tmp_path):
    # A skips entry is a scalar, not a mapping (L87-88).
    p = tmp_path / "scalar_entry.yaml"
    p.write_text("skips:\n  - just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"skips\[0\] must be a mapping"):
        load_known_bugs(p)


def test_load_entry_missing_path_raises(tmp_path):
    # Entry without a "path" key (L90-91).
    p = tmp_path / "nopath.yaml"
    p.write_text("skips:\n  - solution_index: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires string 'path'"):
        load_known_bugs(p)


def test_load_entry_empty_path_raises(tmp_path):
    # Entry with an empty "path" string (falsy) (L90-91).
    p = tmp_path / "emptypath.yaml"
    p.write_text("skips:\n  - path: ''\n    solution_index: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires string 'path'"):
        load_known_bugs(p)


def test_load_entry_missing_solution_index_raises(tmp_path):
    # Entry with a path but no solution_index (L95-96).
    p = tmp_path / "noidx.yaml"
    p.write_text("skips:\n  - path: foo/bar.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires integer 'solution_index'"):
        load_known_bugs(p)


def test_load_entry_non_int_solution_index_raises(tmp_path):
    # solution_index present but not an int (L99-100).
    p = tmp_path / "stridx.yaml"
    p.write_text(
        "skips:\n  - path: foo/bar.yaml\n    solution_index: three\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="solution_index must be int"):
        load_known_bugs(p)
