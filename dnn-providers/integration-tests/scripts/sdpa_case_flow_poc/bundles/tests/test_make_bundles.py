# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""make_bundles checks that run before the imports."""
import json
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_bundles  # noqa: E402

BASE = [{"name": n} for n in ("q", "k", "v", "sdpa_fwd::O")]
DEVICE_SEED = [{"name": "dropout_seed"}, {"name": "dropout_offset"}]
BY_VALUE_SEED = [{"name": n, "value": 1} for n in ("dropout_seed", "dropout_offset")]
RNG_DUMP = [{"name": "rng_dump"}]


def export(export_dir, case_key, tensors):
    case_dir = export_dir / case_key
    case_dir.mkdir(parents=True)
    graph = {"tensors": [dict(t, uid=i) for i, t in enumerate(tensors)]}
    (case_dir / f"{case_key}.json").write_text(json.dumps(graph))


def row(case_key, *roles):
    return {
        "case_key": case_key,
        "level": "full",
        "_signature": tuple(sorted(("q", "k", "v", "o") + roles)),
        "_case_id": f"bf16_{case_key}",
    }


def test_signature_mismatches_compare_roles_and_value_forms(tmp_path):
    export(tmp_path, "device", BASE + DEVICE_SEED + RNG_DUMP)
    export(tmp_path, "by_value", BASE + BY_VALUE_SEED + RNG_DUMP)
    export(tmp_path, "unlisted", BASE + DEVICE_SEED + RNG_DUMP)
    export(tmp_path, "stats", BASE + [{"name": "sdpa_fwd::STATS"}])
    export(
        tmp_path,
        "runtime",
        BASE + [{"name": "attn_scale", "is_runtime_pass_by_value": True}],
    )
    rows = [
        row("device", "dropout_seed", "dropout_offset", "rng_dump"),
        row("by_value", "dropout_seed", "dropout_offset", "rng_dump"),
        row("unlisted", "rng_dump"),
        row("stats", "stats"),
        row("runtime", "attn_scale[runtime]"),
    ]
    assert make_bundles.signature_mismatches(rows, tmp_path) == [
        (
            "by_value",
            ["dropout_offset", "dropout_seed"],
            ["dropout_offset[by_value]", "dropout_seed[by_value]"],
        ),
        ("unlisted", [], ["dropout_offset", "dropout_seed"]),
    ]


def test_main_stops_before_any_import_when_a_graph_differs(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export(export_dir, "unlisted", BASE + DEVICE_SEED + RNG_DUMP)
    rows = [row("unlisted", "rng_dump")]
    monkeypatch.setattr(make_bundles, "load_rows", lambda *_: [dict(r) for r in rows])
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_bundles.py",
            "--selected",
            str(tmp_path / "selected.csv"),
            "--class-values",
            str(tmp_path / "class-to-value.json"),
            "--export-dir",
            str(export_dir),
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as stop:
        make_bundles.main()
    assert "unlisted: graph lacks [], signature lacks" in str(stop.value.code)
    assert not work_dir.exists()


def test_check_tree_reports_a_link_to_an_unknown_uid(tmp_path):
    export_dir = tmp_path / "export"
    tensors = BASE + [{"name": "ragged_offset_q"}]
    export(export_dir, "linked", tensors)
    graph_path = export_dir / "linked" / "linked.json"
    graph = json.loads(graph_path.read_text())
    graph["tensors"][0]["ragged_offset_tensor_uid"] = 99
    graph_path.write_text(json.dumps(graph))
    (export_dir / "linked" / "linked.meta.json").write_text("{}")

    placed = dict(row("linked", "ragged_offset_q"), _topology="RgQ")
    sweep_dir = tmp_path / "tree" / "full" / "op" / "RgQ"
    sweep_dir.mkdir(parents=True)
    template = {"tensors": [dict(t, uid=i) for i, t in enumerate(tensors)]}
    (sweep_dir / "graph.template.json").write_text(json.dumps(template))
    sweep = {"import_mode": "grouped", "cases": [{"id": placed["_case_id"]}]}
    (sweep_dir / "sweep.json").write_text(json.dumps(sweep))

    problems = make_bundles.check_tree(tmp_path / "tree", "op", [placed], export_dir)
    assert any("does not remap" in p for p in problems), problems
