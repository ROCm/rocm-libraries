# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the layout-aware rocGDB manifest generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools/make_rocke_debug_manifest.py"
_SPEC = importlib.util.spec_from_file_location("make_rocke_debug_manifest", _TOOL)
make_manifest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_manifest)


def test_builds_mfma_accumulator_manifest_from_arch_layout():
    manifest = make_manifest.build_manifest(
        arch="gfx942",
        op_id="mfma_f32_16x16x16_f16",
        role="acc",
        name="acc",
        dtype="f32",
        storage_dtype="f32",
        locations=["$v40", "$v41", "$v42", "$v43"],
    )

    value = manifest["values"][0]
    assert value["logical"]["shape"] == [16, 16]
    assert value["logical"]["layout"]["name"] == "mfma_f32_16x16x16_f16.acc"
    assert value["logical"]["layout"]["role"] == "acc"
    assert value["logical"]["layout"]["coordinates"][0] == {
        "lane": 0,
        "slot": 0,
        "index": [0, 0],
    }


def test_main_writes_deterministic_json(tmp_path):
    output = tmp_path / "acc.json"
    result = make_manifest.main(
        [
            "--arch",
            "gfx942",
            "--op-id",
            "mfma_f32_16x16x16_f16",
            "--role",
            "acc",
            "--name",
            "acc",
            "--dtype",
            "f32",
            "--storage-dtype",
            "f32",
            "--location",
            "$v40",
            "--location",
            "$v41",
            "--location",
            "$v42",
            "--location",
            "$v43",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema"] == "rocke-debug-manifest/v1"
    assert manifest["values"][0]["binding"]["locations"] == [
        "$v40",
        "$v41",
        "$v42",
        "$v43",
    ]


def test_builds_replicated_gfx11_operand_manifest():
    manifest = make_manifest.build_manifest(
        arch="gfx1151",
        op_id="wmma_f32_16x16x16_f16",
        role="a",
        name="a",
        dtype="f16",
        storage_dtype="f16x2",
        locations=[f"$v{index}" for index in range(8)],
    )

    layout = manifest["values"][0]["logical"]["layout"]
    assert layout["replication_factor"] == 2
    assert len(layout["coordinates"]) == 512
    assert [
        (entry["lane"], entry["slot"])
        for entry in layout["coordinates"]
        if entry["index"] == [0, 0]
    ] == [(0, 0), (16, 0)]


def test_rejects_unknown_operation(capsys):
    result = make_manifest.main(
        [
            "--arch",
            "gfx942",
            "--op-id",
            "missing",
            "--role",
            "acc",
            "--name",
            "acc",
            "--dtype",
            "f32",
            "--storage-dtype",
            "f32",
            "--location",
            "$v40",
        ]
    )

    assert result == 2
    assert "has no MMA operation" in capsys.readouterr().err
