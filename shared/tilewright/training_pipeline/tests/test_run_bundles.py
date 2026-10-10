# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The bundles of a run in the order it deploys them (lib/run_bundles.py)."""

import json
import shutil

from lib import run_bundles as rb


def complete(rd, finished):
    rd.mkdir(parents=True, exist_ok=True)
    (rd / rb.ROUND_MARKER).write_text(json.dumps({"finished": finished}))


def final_retrain(run, rounds, accepted=True, deployed=True):
    d = run / rb.FINAL_RETRAIN_DIR
    (d / "stage06").mkdir(parents=True)
    stamps = rb.round_stamps(run / name for name in rounds)
    (d / rb.RECORD).write_text(json.dumps({"rounds": stamps}))
    (d / rb.DECISION).write_text(json.dumps({"accepted": accepted}))
    if deployed:
        (d / "stage06" / rb.STAGE_MARKER).write_text("{}")
        (d / "stage06" / rb.DEPLOY_RECORD).write_text("{}")
    return d


def test_the_final_retrain_deploys_after_its_rounds_and_before_later_ones(tmp_path):
    run = tmp_path / "run"
    r0, r1, r2 = (run / f"round_{i}" for i in range(3))
    complete(r0, "2026-01-01T00:00:00")
    complete(r1, "2026-01-01T01:00:00")
    final = final_retrain(run, ["round_0", "round_1"])
    complete(r2, "2026-01-01T03:00:00")
    assert rb.final_retrain(run) == final
    assert rb.bundle_key(final) == (1, 1)
    assert rb.bundles(run, [r2, r0, r1]) == [r0, r1, final, r2]
    assert rb.bundles(run, [r0]) == [r0, final]


def test_the_final_retrain_is_a_bundle_only_while_its_rounds_are_unchanged(
    tmp_path,
):
    run = tmp_path / "run"
    r0, r1 = run / "round_0", run / "round_1"
    complete(r0, "2026-01-01T00:00:00")
    complete(r1, "2026-01-01T01:00:00")
    final = final_retrain(run, ["round_0", "round_1"])
    assert rb.changed_rounds(run, rb.trained_rounds(final)) == []
    copy = tmp_path / "copy"
    shutil.copytree(run, copy, copy_function=shutil.copy)
    assert rb.final_retrain(copy) == copy / rb.FINAL_RETRAIN_DIR
    complete(r1, "2026-01-02T00:00:00")
    assert rb.final_retrain(run) is None
    assert rb.changed_rounds(run, rb.trained_rounds(final)) == ["round_1"]
    (r1 / rb.ROUND_MARKER).unlink()
    assert rb.changed_rounds(run, rb.trained_rounds(final)) == ["round_1"]
    assert rb.bundles(run, [r0]) == [r0]


def test_a_final_retrain_that_was_not_accepted_or_deployed_is_no_bundle(tmp_path):
    for accepted, deployed in ((False, False), (True, False)):
        run = tmp_path / f"run_{accepted}_{deployed}"
        complete(run / "round_0", "2026-01-01T00:00:00")
        final_retrain(run, ["round_0"], accepted=accepted, deployed=deployed)
        assert rb.final_retrain(run) is None
    assert rb.final_retrain(tmp_path / "nowhere") is None


def test_audit_results_lists_what_the_run_wrote_outside_its_rounds(tmp_path):
    run = tmp_path / "run"
    assert rb.audit_result_files(run) == []
    (run / rb.AUDIT_RESULTS_DIR).mkdir(parents=True)
    for name in ("round_1.json", "round_0.json"):
        (run / rb.AUDIT_RESULTS_DIR / name).write_text("{}")
    final = run / rb.FINAL_RETRAIN_DIR
    (final / "stage06").mkdir(parents=True)
    (final / rb.FINAL_AUDIT).write_text("{}")
    assert rb.audit_result_files(run) == [
        run / rb.AUDIT_RESULTS_DIR / "round_0.json",
        run / rb.AUDIT_RESULTS_DIR / "round_1.json",
        final / rb.FINAL_AUDIT,
    ]
    (final / "stage06" / rb.RESULT_FILE).write_text("{}")
    assert rb.audit_result_files(run)[2] == final / "stage06" / rb.RESULT_FILE


def cells(rd, trained, labels, carried_forward=False, origins=None, finished="t0"):
    s5 = rd / "stage05"
    s5.mkdir(parents=True, exist_ok=True)
    (s5 / rb.STAGE_MARKER).write_text(json.dumps({"finished": finished}))
    entries = [{"label": c, "round": rd.name} for c in trained]
    (s5 / "cells.json").write_text(
        json.dumps({"cells": entries, "model_labels": labels})
    )
    if carried_forward:
        (s5 / rb.CARRIED_FORWARD).write_text("{}")
    if origins is not None:
        (s5 / rb.CELL_ORIGINS).write_text(json.dumps(origins))


def test_a_cell_comes_from_the_latest_bundle_that_trained_it_or_recorded_it(
    tmp_path,
):
    """`cell_origins`: from the latest bundle, the cells its stage05
    trained, then those its cell_origins.json records it carried, from the
    bundle the record names while that is one of the bundles with the same
    training, else as the record kept them; a bundle without that record,
    or one that carried the bundle before it forward, leaves the cells it
    did not train to the bundles before it."""
    r0, r1, r2, r3 = (tmp_path / f"round_{i}" for i in range(4))
    cells(r0, ["A", "B"], ["A", "B"])
    cells(r1, ["A"], ["A", "B"])
    cells(r2, ["A"], ["A", "B"], carried_forward=True)
    out = rb.cell_origins([r0, r1, r2])
    assert {c: (o["trained_in"], o["bundle"]) for c, o in out.items()} == {
        "A": ("round_1", r1),
        "B": ("round_0", r0),
    }
    assert out["A"]["entry"] == {"label": "A", "round": "round_1"}
    assert out["A"]["model_options"] is None and out["A"]["finished"] == "t0"
    kept = {"label": "B", "round": "then"}
    record = {
        "source": "round_2",
        "origins": {
            "round_0@t0": {
                "bundle": "round_0",
                "finished": "t0",
                "model_options": {"lam": 0.4},
            },
            "gone@t9": {"bundle": "gone", "finished": "t9", "model_options": None},
        },
        "cells": {
            "B": {"origin": "round_0@t0", "entry": kept},
            "C": {"origin": "gone@t9", "entry": {"label": "C"}},
        },
    }
    cells(r3, [], ["A", "B", "C"], origins=record)
    out = rb.cell_origins([r0, r1, r2, r3])
    assert out["A"]["trained_in"] == "round_1"
    # round_0 is in the run with the training the record names: its own
    # records tell the cell
    assert out["B"] == {
        "trained_in": "round_0",
        "bundle": r0,
        "finished": "t0",
        "entry": {"label": "B", "round": "round_0"},
        "model_options": None,
    }
    assert out["C"] == {
        "trained_in": "gone",
        "bundle": None,
        "finished": "t9",
        "entry": {"label": "C"},
        "model_options": None,
    }
    # round_0 trained again since: what the record kept tells the cell
    cells(r0, ["A", "B"], ["A", "B"], finished="t1")
    assert rb.cell_origins([r0, r1, r2, r3])["B"] == {
        "trained_in": "round_0",
        "bundle": None,
        "finished": "t0",
        "entry": kept,
        "model_options": {"lam": 0.4},
    }
    # and so it does when round_0 is not among the bundles
    assert rb.cell_origins([r3])["B"]["bundle"] is None
