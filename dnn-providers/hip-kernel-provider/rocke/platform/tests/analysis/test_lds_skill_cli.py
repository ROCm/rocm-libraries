# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subprocess coverage for the public LDS conflict expert CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PLATFORM = Path(__file__).resolve().parents[2]
PREDICT = (
    PLATFORM
    / "dsl_docs"
    / "optimization"
    / "utilities"
    / "skills"
    / "lds-bank-conflict-expert"
    / "scripts"
    / "predict.py"
)


def _run(request: dict[str, object]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PLATFORM / "python")
    return subprocess.run(
        [sys.executable, str(PREDICT), "-"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_skill_cli_emits_canonical_conflict_json():
    completed = _run(
        {
            "target": "gfx90a",
            "opcode": "ds_read_b32",
            "wave_size": 64,
            "accesses": [
                {
                    "access_id": 0,
                    "lane": 0,
                    "lds_byte_address": 0,
                    "access_width_bytes": 4,
                },
                {
                    "access_id": 1,
                    "lane": 1,
                    "lds_byte_address": 128,
                    "access_width_bytes": 4,
                },
            ],
        }
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["profile"] == {"profile_version": 1, "target": "gfx90a"}
    assert result["conflict_groups"] == [
        {
            "access_ids": [0, 1],
            "group_id": 0,
            "kind": "distinct-address-conflict",
            "multiplicity": 2,
        }
    ]
    assert completed.stdout == completed.stdout.strip() + "\n"


def test_skill_cli_rejects_unregistered_target_without_fallback():
    completed = _run(
        {
            "target": "gfx9999",
            "opcode": "ds_read_b32",
            "wave_size": 64,
            "accesses": [],
        }
    )

    assert completed.returncode == 2
    assert "unsupported LDS target 'gfx9999'" in completed.stderr
    assert "registered targets: gfx90a" in completed.stderr
    assert completed.stdout == ""
