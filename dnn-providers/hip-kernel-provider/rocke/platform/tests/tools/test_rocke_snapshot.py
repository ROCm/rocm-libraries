# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for debugger-independent stopped-wave snapshots and rendering."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rocke.debug import (
    ValueSnapshot,
    collect_selected_wave,
    dump_snapshot,
    load_snapshot,
    logical_snapshot,
)

_RENDER_TOOL = Path(__file__).resolve().parents[2] / "tools/rocke_render_snapshot.py"
_RENDER_SPEC = importlib.util.spec_from_file_location(
    "rocke_render_snapshot", _RENDER_TOOL
)
rocke_render_snapshot = importlib.util.module_from_spec(_RENDER_SPEC)
assert _RENDER_SPEC.loader is not None
_RENDER_SPEC.loader.exec_module(rocke_render_snapshot)


def _manifest() -> dict:
    return {
        "schema": "rocke-debug-manifest/v1",
        "values": [
            {
                "logical": {
                    "name": "acc",
                    "dtype": "f32",
                    "shape": [2, 2],
                    "layout": {
                        "name": "test.acc",
                        "role": "acc",
                        "wave_size": 2,
                        "fragment_length": 2,
                        "replication_factor": 1,
                        "packing": {"kind": "scalar", "elements_per_slot": 1},
                        "coordinates": [
                            {"lane": 0, "slot": 0, "index": [0, 0]},
                            {"lane": 0, "slot": 1, "index": [0, 1]},
                            {"lane": 1, "slot": 0, "index": [1, 0]},
                            {"lane": 1, "slot": 1, "index": [1, 1]},
                        ],
                    },
                },
                "binding": {
                    "kind": "amdgpu_registers",
                    "storage_dtype": "f32",
                    "locations": ["$v2", "$v3"],
                },
            }
        ],
    }


def _snapshot() -> ValueSnapshot:
    words = {
        "$v2": [0x3F800000, 0x40400000],
        "$v3": [0x40000000, 0x40800000],
    }
    return collect_selected_wave(
        _manifest(),
        ["acc"],
        read_words=words.__getitem__,
        thread_id="7",
        pc=0x1234,
        exec_mask=0b01,
        architecture="amdgcn:gfx90a",
        kernel="demo_kernel",
        stop_mode="all-stop",
    )


def test_snapshot_round_trip_is_strict_and_deterministic(tmp_path: Path):
    snapshot = _snapshot()
    output = tmp_path / "acc.snapshot.json"
    dump_snapshot(snapshot, output)

    text = output.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "value_text" not in text
    assert load_snapshot(output) == snapshot
    assert json.loads(text)["waves"][0]["values"][0]["locations"][1] == {
        "expression": "$v3",
        "raw_words": [0x40000000, 0x40800000],
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        dump_snapshot(snapshot, output)


def test_offline_reconstruction_has_no_presentation_fields():
    logical = logical_snapshot(_snapshot().to_dict())

    assert logical["schema"] == "rocke-logical-snapshot/v1"
    assert logical["waves"][0]["values"][0]["status"] == "available"
    assert [
        [cell["value"] for cell in row]
        for row in logical["waves"][0]["values"][0]["tile"]
    ] == [[1.0, 2.0], [3.0, 4.0]]
    assert "value_text" not in json.dumps(logical)
    assert "raw_hex" not in json.dumps(logical)


def test_human_renderer_uses_derived_logical_snapshot():
    logical = logical_snapshot(_snapshot().to_dict())
    rendered = rocke_render_snapshot.render_human(logical)

    assert "capture scope=wave complete=true" in rendered
    assert "wave thread=7 status=available pc=0x1234 exec=0x1" in rendered
    assert "acc f32 [2x2] layout=test.acc status=available" in rendered
    assert "1.0 2.0" in rendered
    assert "~3.0 ~4.0" in rendered


def test_offline_renderer_cli_reads_saved_snapshot(tmp_path: Path, capsys):
    output = tmp_path / "acc.snapshot.json"
    dump_snapshot(_snapshot(), output)

    assert rocke_render_snapshot.main([str(output), "--format", "json"]) == 0
    rendered = capsys.readouterr().out
    parsed = json.loads(rendered)
    assert parsed["schema"] == "rocke-logical-snapshot/v1"
    assert parsed["waves"][0]["values"][0]["tile"][0][1]["value"] == 2.0
    assert "value_text" not in rendered


def test_failed_location_is_explicit_and_never_becomes_zero():
    calls = []

    def fail_on_second(expression: str):
        calls.append(expression)
        if expression == "$v3":
            raise RuntimeError("register is optimized out")
        return [0, 0]

    snapshot = collect_selected_wave(
        _manifest(),
        ["acc"],
        read_words=fail_on_second,
        thread_id="7",
        pc=0x1234,
        exec_mask=0b11,
        architecture="amdgcn:gfx90a",
        kernel="demo_kernel",
        stop_mode="all-stop",
    )

    assert calls == ["$v2", "$v3"]
    assert snapshot.capture["complete"] is False
    captured = snapshot.waves[0].values[0]
    assert captured.status == "optimized_out"
    assert captured.locations == ()
    logical = logical_snapshot(snapshot.to_dict())
    assert logical["waves"][0]["values"][0]["tile"] is None


@pytest.mark.parametrize("word", [-1, 1 << 32, True])
def test_snapshot_rejects_non_uint32_words(word: int):
    record = _snapshot().to_dict()
    record["waves"][0]["values"][0]["locations"][0]["raw_words"][0] = word
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        ValueSnapshot.from_dict(record)


def test_snapshot_rejects_lane_count_mismatch():
    record = _snapshot().to_dict()
    record["waves"][0]["values"][0]["locations"][0]["raw_words"].pop()
    with pytest.raises(ValueError, match="do not match wave size"):
        ValueSnapshot.from_dict(record)
