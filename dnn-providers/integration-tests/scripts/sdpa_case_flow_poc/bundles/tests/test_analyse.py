# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stats layouts that analyse.py reads from a bundle tree with absent tensors."""
import json
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyse  # noqa: E402

STATS_UID = 5


def stats_tensor(dims, strides):
    return {"uid": STATS_UID, "dims": dims, "strides": strides}


def write_sweep(root, cases, stats_uid=STATS_UID):
    sweep_dir = root / "quick" / "SdpaFwdPlan" / "Superset"
    sweep_dir.mkdir(parents=True)
    template = {
        "nodes": [
            {"type": "SdpaAttributes", "outputs": {"stats_tensor_uid": stats_uid}}
        ]
    }
    (sweep_dir / "graph.template.json").write_text(json.dumps(template))
    (sweep_dir / "sweep.json").write_text(json.dumps({"version": 2, "cases": cases}))


def case(key, tensors, absent=None):
    c = {"id": key, "metadata": {"case_key": key}, "values": {"tensors": tensors}}
    if absent is not None:
        c["absent_tensors"] = absent
    return c


@pytest.fixture
def bundle_root(tmp_path, monkeypatch):
    monkeypatch.setattr(analyse, "BUNDLES", tmp_path)
    monkeypatch.setattr(analyse, "_STATS_LAYOUT", None)
    return tmp_path


def test_case_with_stats_absent_has_no_layout(bundle_root):
    present = stats_tensor([2, 4, 8, 1], [32, 8, 1, 1])
    write_sweep(
        bundle_root,
        [
            case("with_stats", [present]),
            case("no_stats", [], absent=[STATS_UID]),
        ],
    )

    assert analyse.stats_layouts() == {"with_stats": ([2, 4, 8, 1], [32, 8, 1, 1])}


def test_absent_tensor_other_than_stats_keeps_stats_layout(bundle_root):
    present = stats_tensor([1, 2, 16, 1], [32, 16, 1, 1])
    write_sweep(bundle_root, [case("no_sink", [present], absent=[9])])

    assert analyse.stats_layouts() == {"no_sink": ([1, 2, 16, 1], [32, 16, 1, 1])}


def test_present_stats_without_values_entry_raises(bundle_root):
    write_sweep(bundle_root, [case("broken", [], absent=[9])])

    with pytest.raises(ValueError, match="case 'broken' has no values for Stats"):
        analyse.stats_layouts()


def test_template_without_stats_port_is_skipped(bundle_root):
    write_sweep(bundle_root, [case("plain", [])], stats_uid=None)

    assert analyse.stats_layouts() == {}
