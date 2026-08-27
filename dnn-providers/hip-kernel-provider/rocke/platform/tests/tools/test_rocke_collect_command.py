# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fake-rocGDB test for the selected-wave collection command."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


class FakeGdbError(Exception):
    pass


class FakeType:
    def range(self):
        return 0, 1


class FakeVector:
    def __init__(self, words):
        self.type = FakeType()
        self.words = words

    def __getitem__(self, index):
        return self.words[index]


class FakeArchitecture:
    def name(self):
        return "amdgcn:gfx90a"


class FakeFrame:
    def architecture(self):
        return FakeArchitecture()

    def name(self):
        return "demo_kernel"


class FakeThread:
    global_num = 17

    def is_stopped(self):
        return True


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


def _load_with_fake_gdb(monkeypatch):
    commands = {}
    writes = []
    reads = []
    fake = types.ModuleType("gdb")
    fake.COMMAND_USER = 0
    fake.COMMAND_DATA = 1
    fake.error = FakeGdbError
    fake.GdbError = FakeGdbError

    class Command:
        def __init__(self, name, command_class, prefix=False):
            del command_class, prefix
            commands[name] = self

    values = {
        "$pc": 0x1234,
        "$exec": 0b01,
        "$v2": FakeVector([0x3F800000, 0x40400000]),
        "$v3": FakeVector([0x40000000, 0x40800000]),
    }

    def parse_and_eval(expression):
        reads.append(expression)
        if expression not in values:
            raise FakeGdbError(f"unknown expression {expression}")
        return values[expression]

    fake.Command = Command
    fake.parse_and_eval = parse_and_eval
    fake.selected_thread = lambda: FakeThread()
    fake.selected_frame = lambda: FakeFrame()
    fake.parameter = lambda name: False if name == "non-stop" else None
    fake.write = writes.append
    monkeypatch.setitem(sys.modules, "gdb", fake)
    tool = Path(__file__).resolve().parents[2] / "tools/rocke_debug.py"
    spec = importlib.util.spec_from_file_location("rocke_debug_with_fake_gdb", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return commands, writes, reads


def test_collect_command_writes_raw_snapshot_without_rendering(tmp_path, monkeypatch):
    commands, writes, reads = _load_with_fake_gdb(monkeypatch)
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "snapshot.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")

    commands["rocke collect"].invoke(
        f"acc --manifest {manifest} --scope wave --output {output}", False
    )

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["schema"] == "rocke-value-snapshot/v1"
    assert snapshot["capture"]["complete"] is True
    assert snapshot["waves"][0]["thread_id"] == "17"
    assert snapshot["waves"][0]["pc"] == "0x1234"
    assert snapshot["waves"][0]["exec"] == "0x1"
    assert snapshot["waves"][0]["values"][0]["locations"][1] == {
        "expression": "$v3",
        "raw_words": [0x40000000, 0x40800000],
    }
    assert "value_text" not in output.read_text(encoding="utf-8")
    assert reads == ["$pc", "$exec", "$v2", "$v3"]
    assert writes == [f"wrote {output} (complete=true)\n"]


def test_print_command_collects_and_renders_without_snapshot_file(
    tmp_path, monkeypatch
):
    commands, writes, reads = _load_with_fake_gdb(monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")

    commands["rocke print"].invoke(f"acc --manifest {manifest}", False)

    assert reads == ["$pc", "$exec", "$v2", "$v3"]
    assert len(writes) == 1
    assert "acc f32 [2x2] layout=test.acc status=available" in writes[0]
    assert "1.0 2.0" in writes[0]
    assert "~3.0 ~4.0" in writes[0]
    assert list(tmp_path.iterdir()) == [manifest]
