# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The differential golden file's rules (golden_store.py).

Each rule here closes a way the file went stale before: entries without an LLVM
flavor (hosts of different flavors blessed into one section), a filtered
re-bless that replaced the whole section, a filtered check that reported every
other family missing, and duplicate JSON keys that silently kept the older copy.

Host-only and pure: no emitter runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_store import (  # noqa: E402 -- after the sys.path line above
    IR_CANONICAL,
    LL,
    GoldenFormatError,
    GoldenStore,
    compare,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_duplicate_keys_are_refused_by_name(tmp_path):
    golden = _write(
        tmp_path / "g.json",
        '{"schema": 2, "ll": {"llvm20": {"conv": {"0": "a"}, "conv": {"0": "b"}}}}',
    )
    with pytest.raises(GoldenFormatError, match=r"duplicate keys \['conv'\]"):
        GoldenStore.load(golden)


def test_the_flavorless_format_is_refused_with_the_way_out(tmp_path):
    golden = _write(tmp_path / "g.json", '{"ll": {"conv": {"0": "a"}}}')
    with pytest.raises(
        GoldenFormatError, match=r"schema None, expected 2.*--record-golden"
    ):
        GoldenStore.load(golden)


def test_a_missing_file_is_an_empty_store(tmp_path):
    store = GoldenStore.load(tmp_path / "absent.json")
    assert store.section(LL, "llvm20") == {}
    assert store.section(IR_CANONICAL, None) == {}


def test_flavors_are_separate_sections(tmp_path):
    store = GoldenStore()
    store.record(LL, "llvm20", {"conv": {"0": "a20"}})
    store.record(LL, "llvm22", {"conv": {"0": "a22"}})
    store.save(tmp_path / "g.json")
    loaded = GoldenStore.load(tmp_path / "g.json")
    assert loaded.section(LL, "llvm20") == {"conv": {"0": "a20"}}
    assert loaded.section(LL, "llvm22") == {"conv": {"0": "a22"}}
    assert json.loads((tmp_path / "g.json").read_text())["schema"] == 2


def test_the_ll_section_needs_a_flavor():
    with pytest.raises(ValueError, match="needs an LLVM flavor"):
        GoldenStore().section(LL, None)
    with pytest.raises(ValueError, match="needs an LLVM flavor"):
        GoldenStore().record(LL, "", {"conv": {"0": "a"}})


def test_a_full_record_replaces_the_section():
    store = GoldenStore()
    store.record(LL, "llvm20", {"conv": {"0": "a"}, "gemm": {"0": "b"}})
    removed = store.record(LL, "llvm20", {"conv": {"0": "c"}})
    assert store.section(LL, "llvm20") == {"conv": {"0": "c"}}
    assert removed == ["gemm"]


def test_a_filtered_record_replaces_only_its_families():
    store = GoldenStore()
    store.record(
        IR_CANONICAL, None, {"conv": {"0": "a"}, "gemm": {"0": "b"}, "norm": {"0": "n"}}
    )
    # gemm re-blessed, norm ran but produced nothing, conv was not selected.
    removed = store.record(
        IR_CANONICAL, None, {"gemm": {"0": "B", "1": "B1"}}, families=["gemm", "norm"]
    )
    assert store.section(IR_CANONICAL, None) == {
        "conv": {"0": "a"},
        "gemm": {"0": "B", "1": "B1"},
    }
    # The caller is told which anchors went, so a record that silently
    # produced nothing for a family cannot pass unnoticed.
    assert removed == ["norm"]


def test_a_check_reports_changed_missing_and_new_configs():
    blessed = {"conv": {"0": "a", "1": "b"}, "gemm": {"0": "g"}}
    current = {"conv": {"0": "a", "1": "B", "2": "c"}}
    result = compare(blessed, current)
    assert result.changed == (("conv", "1"),)
    assert result.missing == (("gemm", "0"),)
    assert result.new == (("conv", "2"),)
    assert result.blessed == 3
    assert not result.ok


def test_a_filtered_check_ignores_unselected_families():
    blessed = {"conv": {"0": "a"}, "gemm": {"0": "g"}}
    result = compare(blessed, {"conv": {"0": "a"}}, families=["conv"])
    assert result.ok
    assert result.blessed == 1


def test_new_configs_alone_pass():
    result = compare({"conv": {"0": "a"}}, {"conv": {"0": "a", "1": "b"}})
    assert result.ok
    assert result.new == (("conv", "1"),)


def test_configs_are_reported_in_numeric_order():
    blessed = {"conv": {str(i): "x" for i in (10, 2, 1)}}
    result = compare(blessed, {"conv": {}})
    assert [idx for _, idx in result.missing] == ["1", "2", "10"]
