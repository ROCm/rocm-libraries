# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A feature semantics revision is re-stamped only on proof the model's inputs did not move."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from uhd_gen.__main__ import main
from uhd_gen.provenance import validate_provenance
from uhd_gen.restamp_semantics import (
    LATEST_REVISION,
    REVISION_CHANGES,
    restamp_semantics,
)

FEATURE_SEMANTICS_HPP = (
    Path(__file__).resolve().parents[3]
    / "plugin_sdk/include/hipdnn_plugin_sdk/heuristics/FeatureSemantics.hpp"
)
UED = {"id": "f7ea7021-d005-538d-8db5-22a9ea6d9cd3", "revision": "1.0"}
KMD = {"id": "331def97-6b65-540e-abda-13651841e429", "revision": "1.0"}
UMD = [{"id": "c0dc7bc9-320c-5d76-845a-7ca30d9ce116", "revision": "1.0"}]


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _model(tmp_path, signature, *, metric="time", revision=1):
    trained = {"ued": UED, "kmd": KMD, "umd": UMD}
    if revision is not None:
        trained["feature_semantics_revision"] = revision
    return _write(
        tmp_path / "model.uhd.json",
        {
            "version": "1.0",
            "id": "727e5401-3b99-49ff-a2fc-68fd4eedbb54",
            "name": "l1",
            "adapter": "mlp_data",
            "features_signature": signature,
            "trained_against": trained,
            "objective": "min" if metric == "time" else "max",
            "score": {"metric": metric, "calibrated": True, "transform": "log"},
            "mlp_data": {"artifact": "model.bin"},
        },
    )


def _rows(tmp_path, name, features, graphs=30):
    """Rows of graphs `g0`.. publishing `features(i)` in the `features` map, two devices
    each, as a collection holds them."""
    rows = [
        {"benchmark": f"g{i}", "device": device, "features": features(i)}
        for i in range(graphs)
        for device in ("0", "1")
    ]
    return _write(tmp_path / f"{name}.json", rows)


def _forward(i):
    return {"graph.flops": 1e9 * (i + 1), "graph.logical_bytes": 1e6 * (i + 1)}


def test_the_change_table_follows_feature_semantics_hpp():
    # Every revision after 1 says what it changed, and the newest is the build's.
    built = int(
        re.search(
            r"FEATURE_SEMANTICS_REVISION\s*=\s*(\d+)",
            FEATURE_SEMANTICS_HPP.read_text(encoding="utf-8"),
        ).group(1)
    )
    assert LATEST_REVISION == built
    assert sorted(REVISION_CHANGES) == list(range(2, built + 1))


def test_a_model_reading_no_changed_feature_is_restamped_without_rows(tmp_path):
    path = _model(tmp_path, ["$rmsnorm.rows", "$graph.nodes[0].x.dims[1]"])
    before = path.read_text(encoding="utf-8")
    report = restamp_semantics(path, write=True)
    assert report["passed"] and report["written"] and report["proof"] == "unread"
    trained = json.loads(path.read_text(encoding="utf-8"))["trained_against"]
    assert trained["feature_semantics_revision"] == LATEST_REVISION
    assert trained["x-restamps"] == [
        {
            "member": "feature_semantics_revision",
            "from": 1,
            "to": LATEST_REVISION,
            "proof": "unread",
            "compared": [],
        }
    ]
    # The proof is extension data: provenance reads the model as before plus the revision.
    assert validate_provenance(trained)["feature_semantics_revision"] == LATEST_REVISION
    assert "x-restamps" not in validate_provenance(trained)
    # Only trained_against changed, in the file's own layout.
    original = json.loads(before)
    rewritten = json.loads(path.read_text(encoding="utf-8"))
    del original["trained_against"], rewritten["trained_against"]
    assert original == rewritten


def test_an_absent_revision_is_revision_one(tmp_path):
    path = _model(tmp_path, ["$rmsnorm.rows"], revision=None)
    report = restamp_semantics(path, write=False)
    assert report["passed"] and report["from"] == 1 and not report["written"]
    assert (
        "feature_semantics_revision"
        not in json.loads(path.read_text(encoding="utf-8"))["trained_against"]
    )


def test_a_tflops_label_reads_graph_flops(tmp_path):
    # No changed feature in the signature, but the label is graph.flops over time.
    path = _model(tmp_path, ["$rmsnorm.rows"], metric="tflops")
    report = restamp_semantics(path, write=True)
    assert not report["passed"] and not report["written"]
    assert report["reads_changed"] == ["graph.flops"]
    assert "--rows-before" in report["problems"][0]


def test_changed_features_are_found_inside_expressions(tmp_path):
    signature = [
        "$graph.nodes[0].flops",
        {"/": ["$graph.flops_by_type.SdpaBackwardAttributes", "$graph.logical_bytes"]},
        "$graph.flops_by_type.SdpaAttributes",
    ]
    report = restamp_semantics(_model(tmp_path, signature), write=False)
    assert report["reads_changed"] == [
        "graph.flops_by_type.SdpaBackwardAttributes",
        "graph.logical_bytes",
        "graph.nodes[0].flops",
    ]


def test_identical_features_at_both_revisions_restamp(tmp_path):
    path = _model(tmp_path, ["$graph.flops", "$graph.logical_bytes"], metric="tflops")
    before = _rows(tmp_path, "before", _forward)
    after = _rows(tmp_path, "after", _forward)
    report = restamp_semantics(path, rows_before=before, rows_after=after, write=True)
    assert report["passed"] and report["proof"] == "measured-identity"
    assert report["identity"]["graphs"] == 30 and report["identity"]["moved"] == 0
    entry = json.loads(path.read_text(encoding="utf-8"))["trained_against"][
        "x-restamps"
    ][0]
    assert entry["compared"] == ["graph.flops", "graph.logical_bytes"]
    assert entry["identity"]["graphs"] == 30


def test_flattened_rows_are_read_by_column(tmp_path):
    path = _model(tmp_path, ["$graph.flops"])
    rows = [{"benchmark": f"g{i}", "graph.flops": 1e9 * (i + 1)} for i in range(30)]
    before = _write(tmp_path / "before.json", rows)
    after = _write(tmp_path / "after.json", rows)
    assert restamp_semantics(path, rows_before=before, rows_after=after, write=False)[
        "passed"
    ]


def test_a_moved_value_refuses(tmp_path):
    # An MLA backward graph: revision 2 counts its work on each head dimension.
    path = _model(tmp_path, ["$graph.flops"])
    before = _rows(tmp_path, "before", _forward)
    after = _rows(
        tmp_path,
        "after",
        lambda i: {**_forward(i), "graph.flops": 1.04e9} if i == 7 else _forward(i),
    )
    original = path.read_text(encoding="utf-8")
    report = restamp_semantics(path, rows_before=before, rows_after=after, write=True)
    assert not report["passed"] and not report["written"]
    assert report["identity"]["moved"] == 1
    assert "graph.flops of graph g7" in report["problems"][0]
    assert "retrain" in report["problems"][0]
    assert path.read_text(encoding="utf-8") == original


def test_a_feature_published_at_one_revision_only_refuses(tmp_path):
    path = _model(tmp_path, ["$graph.logical_bytes"])
    before = _rows(tmp_path, "before", _forward)
    after = _rows(
        tmp_path,
        "after",
        lambda i: (
            {"graph.flops": _forward(i)["graph.flops"]} if i == 0 else _forward(i)
        ),
    )
    report = restamp_semantics(path, rows_before=before, rows_after=after, write=False)
    assert not report["passed"] and report["identity"]["moved"] == 1


def test_too_few_shared_graphs_refuse(tmp_path):
    path = _model(tmp_path, ["$graph.flops"])
    before = _rows(tmp_path, "before", _forward)
    after = _rows(tmp_path, "after", _forward, graphs=29)
    report = restamp_semantics(path, rows_before=before, rows_after=after, write=False)
    assert not report["passed"]
    assert report["identity"] == {
        "graphs": 29,
        "unmatched": 1,
        "moved": 0,
        "relative_tolerance": 1e-12,
        "min_graphs": 30,
    }


def test_rows_disagreeing_within_one_corpus_cannot_be_judged(tmp_path):
    path = _model(tmp_path, ["$graph.flops"])
    rows = [
        {"benchmark": "g0", "features": {"graph.flops": 1.0}},
        {"benchmark": "g0", "features": {"graph.flops": 2.0}},
    ]
    before = _write(tmp_path / "before.json", rows)
    report = restamp_semantics(path, rows_before=before, rows_after=before, write=False)
    assert (
        not report["passed"] and "not collected by one build" in report["problems"][0]
    )


def test_current_newer_and_unknown_revisions(tmp_path):
    current = restamp_semantics(
        _model(tmp_path, ["$graph.flops"], revision=LATEST_REVISION), write=True
    )
    assert current["passed"] and current["already_current"] and not current["written"]

    newer = restamp_semantics(
        _model(tmp_path, [], revision=LATEST_REVISION + 1), write=True
    )
    assert not newer["passed"] and "never re-stamped backwards" in newer["problems"][0]

    unknown = restamp_semantics(
        _model(tmp_path, []), target=LATEST_REVISION + 1, write=True
    )
    assert not unknown["passed"]
    assert f"revision(s) [{LATEST_REVISION + 1}]" in unknown["problems"][0]


def test_the_command_writes_its_report_and_exit_code(tmp_path, capsys):
    path = _model(tmp_path, ["$graph.flops"])
    before = _rows(tmp_path, "before", _forward)
    report_path = tmp_path / "out" / "report.json"
    assert (
        main(
            [
                "restamp-semantics",
                str(path),
                "--rows-before",
                str(before),
                "--rows-after",
                str(before),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    assert "would be re-stamped (no --write)" in capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "uhd_gen.semantics_restamp_report/1" and report["passed"]

    assert main(["restamp-semantics", str(path), "--write"]) == 1
    assert "REFUSED" in capsys.readouterr().out
