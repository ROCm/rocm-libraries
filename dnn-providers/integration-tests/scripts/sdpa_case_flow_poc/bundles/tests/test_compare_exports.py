# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""compare_exports reports a graph whose uids do not remap and goes on."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_exports  # noqa: E402


def write_case(root, key, link):
    case_dir = root / key
    case_dir.mkdir(parents=True)
    tensors = [
        {"name": "q", "uid": 1, "ragged_offset_tensor_uid": link},
        {"name": "ragged_offset_q", "uid": 2},
    ]
    (case_dir / f"{key}.json").write_text(json.dumps({"tensors": tensors}))
    (case_dir / f"{key}.meta.json").write_text("{}")


def test_a_link_to_an_unknown_uid_is_reported(tmp_path, monkeypatch):
    write_case(tmp_path / "a", "k1", 2)
    write_case(tmp_path / "b", "k1", 7)
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_exports.py",
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            "--report",
            str(report),
        ],
    )
    assert compare_exports.main() == 1
    out = json.loads(report.read_text())
    assert out["canonical_different"] == 1
    assert len(out["remap_errors"]) == 1
    assert out["remap_errors"][0].startswith("k1: ")
