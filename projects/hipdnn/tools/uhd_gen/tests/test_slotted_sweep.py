# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A slotted engine's sweep as training data (RFC 0021 §7).

`fixtures/slotted_sweep.csv` is the output `hipdnn_bench --sweep` makes for a synthetic 3-slot
engine (prep: 2 kernels, dkdv: 3, dq: 2) over two problems, written by hand with separable
timings, so no GPU is needed. TestCandidatePageJson.cpp marks the same kernel ids with the same
rule. Each slot's training table keeps the rows where the other slots hold their reference pick
(first kernel in kernel-id order).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from uhd_gen import addressing  # noqa: E402
from uhd_gen.dataset import store  # noqa: E402
from uhd_gen.dataset.training_table import (  # noqa: E402
    slot_kernels,
    slot_training_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"
SWEEP = FIXTURES / "slotted_sweep.csv"

#: slot -> kernel ids in kernel-id order; the first is the training reference pick.
KERNELS = {
    "prep": [f"11111111-0000-4000-8000-00000000000{i}" for i in (1, 2)],
    "dkdv": [f"22222222-0000-4000-8000-00000000000{i}" for i in (1, 2, 3)],
    "dq": [f"33333333-0000-4000-8000-00000000000{i}" for i in (1, 2)],
}
PROBLEMS = ["sweep_512", "sweep_1024"]


def _dataset(tmp_path, csv=SWEEP):
    directory = tmp_path / "ds"
    store.add(
        directory,
        *store.from_csv(
            [csv],
            engine="SLOTTED_FIXTURE",
            engine_id=7,
            role="sort_kernel_catalog",
            arch="gfx942",
            selector_revision="fixture/1",
            collected_at="2026-10-18T00:00:00+00:00",
            label="slotted-sweep",
        ),
    )
    return directory


def _tables(directory, **options):
    return {
        slot: store.export_rows(directory, purposes=["training"], slot=slot, **options)
        for slot in KERNELS
    }


#: slot -> {(problem seqlen, position in KERNELS[slot]): (kernel.tile, avgTimeMs)}. The run
#: that failed has no time.
TILE_AND_TIME = {
    "prep": {
        (512, 0): (64, 0.8),
        (512, 1): (128, 0.84),
        (1024, 0): (64, 1.6),
        (1024, 1): (128, 1.52),
    },
    "dkdv": {
        (512, 0): (32, 0.8),
        (512, 1): (64, 0.7),
        (512, 2): (128, 0.9),
        (1024, 0): (32, 1.6),
        (1024, 1): (64, 1.8),
        (1024, 2): (128, None),
    },
    "dq": {
        (512, 0): (16, 0.8),
        (512, 1): (32, 0.9),
        (1024, 0): (16, 1.6),
        (1024, 1): (32, 1.45),
    },
}


def test_each_slots_table_carries_the_slots_own_tile_and_time(tmp_path):
    for slot, table in _tables(_dataset(tmp_path)).items():
        for row in table:
            assert {"q.seqlen", "device.cu_count", "avgTimeMs"} <= set(row)
        assert {
            (row["q.seqlen"], KERNELS[slot].index(row["kernel"])): (
                row["kernel.tile"],
                row["avgTimeMs"],
            )
            for row in table
        } == TILE_AND_TIME[slot]


def test_a_slots_table_holds_each_of_its_kernels_once_per_problem_and_no_other_slots(
    tmp_path,
):
    for slot, table in _tables(_dataset(tmp_path)).items():
        for problem in PROBLEMS:
            rows = [row for row in table if row["benchmark"] == problem]
            assert sorted(row["kernel"] for row in rows) == KERNELS[slot]
        assert len(table) == len(PROBLEMS) * len(KERNELS[slot])
        for row in table:
            assert "slot_reference" not in row
            others = [s for s in KERNELS if s != slot]
            assert not [
                c for c in row if c.startswith(tuple(f"kernel.{o}." for o in others))
            ]
            assert not [c for c in row if c.startswith(f"kernel.{slot}.")]
            # The ranker's signature: `$kernel.<field>`.
            assert "kernel.tile" in row


def test_a_slots_rows_vary_only_that_slot(tmp_path):
    """Whole-program time moves with the varied slot alone, so a slot's table ranks its own
    kernels: here the fixture's costs are separable, and the reference program is shared.
    """
    tables = _tables(_dataset(tmp_path))
    reference = {
        problem: next(
            row["avgTimeMs"]
            for row in tables["prep"]
            if row["benchmark"] == problem and row["kernel"] == KERNELS["prep"][0]
        )
        for problem in PROBLEMS
    }
    for slot, table in tables.items():
        first = [row for row in table if row["kernel"] == KERNELS[slot][0]]
        assert {row["benchmark"]: row["avgTimeMs"] for row in first} == reference


def test_the_training_view_drops_a_failed_kernel_from_its_slots_table(tmp_path):
    directory = _dataset(tmp_path)
    stored = store.export_rows(directory, purposes=["training"], slot="dkdv")
    kept = slot_training_rows(store.read_training_rows(directory), "dkdv")
    failed = KERNELS["dkdv"][2]
    # The 1024 problem's third dkdv kernel failed to run: stored, but not a training row.
    assert (failed, "sweep_1024") in {(r["kernel"], r["benchmark"]) for r in stored}
    assert (failed, "sweep_1024") not in {(r["kernel"], r["benchmark"]) for r in kept}
    assert len(kept) == len(stored) - 1


def test_workspace_tokens_never_reach_a_training_row(tmp_path):
    frame = pd.read_csv(SWEEP, dtype={"device": str})
    frame["ws.scratch_offset"] = 4096
    csv = tmp_path / "with_ws.csv"
    frame.to_csv(csv, index=False)
    directory = _dataset(tmp_path, csv)
    manifest = store.read_manifest(directory)
    assert not [
        n for n in manifest["contributions"][0]["published"] if n.startswith("ws.")
    ]
    for table in _tables(directory).values():
        assert not [c for row in table for c in row if c.startswith("ws.")]
    # ... and a view of rows that still carry them drops them too.
    rows = frame.to_dict(orient="records")
    assert not [
        c for r in slot_training_rows(rows, "dq") for c in r if c.startswith("ws.")
    ]


def test_knob_settings_keep_the_slots_knobs_by_field():
    rows = [
        {
            "kernel": f"prep={KERNELS['prep'][0]}|dq={KERNELS['dq'][0]}",
            "slot_reference": "prep|dq",
            "knob_settings": json.dumps({"prep.tile": 1, "dq.tile": 0}),
        }
    ]
    assert slot_training_rows(rows, "dq")[0]["knob_settings"] == '{"tile": 0}'
    assert slot_training_rows(rows, "prep")[0]["knob_settings"] == '{"tile": 1}'


def test_a_slotted_sweep_without_the_marker_is_refused():
    rows = [{"kernel": f"prep={KERNELS['prep'][0]}|dq={KERNELS['dq'][0]}"}]
    with pytest.raises(ValueError, match="slot_reference"):
        slot_training_rows(rows, "prep")


def test_an_unknown_slot_and_an_unslotted_corpus_are_refused(tmp_path):
    rows = pd.read_csv(SWEEP, dtype={"device": str}).to_dict(orient="records")
    with pytest.raises(ValueError, match="no slot 'nope'.*prep.*dkdv.*dq"):
        slot_training_rows(rows, "nope")
    with pytest.raises(ValueError, match="slotted engine's sweep"):
        slot_training_rows([{"kernel": KERNELS["prep"][0]}], "prep")


def test_an_unslotted_export_is_unchanged(tmp_path):
    directory = _dataset(tmp_path)
    everything = store.export_rows(directory, purposes=["training"])
    assert len(everything) == 24
    # Without --slot the rows are the sweep exactly as it was stored: composite ids, marker.
    assert set(slot_kernels(everything[0]["kernel"])) == set(KERNELS)
    assert "slot_reference" in everything[0]


def test_composite_ids_split_in_slot_order_and_a_bare_id_has_no_slots():
    ids = f"prep={KERNELS['prep'][1]}|dkdv={KERNELS['dkdv'][0]}|dq={KERNELS['dq'][1]}"
    assert slot_kernels(ids) == {
        "prep": KERNELS["prep"][1],
        "dkdv": KERNELS["dkdv"][0],
        "dq": KERNELS["dq"][1],
    }
    assert list(slot_kernels(ids)) == ["prep", "dkdv", "dq"]
    assert slot_kernels(KERNELS["prep"][0]) == {}
    with pytest.raises(ValueError, match="composite"):
        slot_kernels("prep=abc|prep=def")


def test_addressing_tolerates_dotted_slot_knobs():
    candidates = [
        {
            "knob_settings": {"prep.mode": 1, "dq.mode": 0},
            "kernel_features": {"kernel.prep.mode": "fast", "kernel.dq.mode": "slow"},
        },
    ]
    table = addressing.observe(candidates)
    assert addressing.decode(table, "prep.mode", 1) == "fast"
    assert addressing.for_slot(table, "dq") == {"mode": {0: "slow"}}
    assert addressing.unaddressable(["prep.mode", "dq.other"], table) == ["dq.other"]


def test_knobs_name_the_problem_column_that_twins_a_slotted_field():
    from uhd_gen.knobs import analyse_knobs

    rows = []
    for seq in (64, 128, 256, 512):
        for tile in (seq, seq * 2):
            rows.append(
                {
                    "benchmark": f"g{seq}",
                    "device": "d",
                    "attention.seqlen": seq,
                    "kernel": f"prep=aa|dq={tile}",
                    "kernel.prep.seqlen": seq,
                    "kernel.prep.tile": tile,
                    "avgTimeMs": 1.0 + (tile == seq),
                    "robustMeanMs": 1.0 + (tile == seq),
                    "minTimeMs": 1.0,
                    "succeeded": True,
                    "is_valid": True,
                }
            )
    report = analyse_knobs(pd.DataFrame(rows))
    twin = next(k for k in report["knobs"] if k["name"] == "kernel.prep.seqlen")
    assert twin["graph_bound"] and twin["bound_as"] == "attention.seqlen"


def _slotted_page():
    ids = [f"prep={p}|dq={q}" for p in KERNELS["prep"] for q in KERNELS["dq"]]
    candidates = [
        {
            "id": identity,
            "knob_settings": {"prep.tile": index // 2, "dq.tile": index % 2},
            "kernel_features": {"kernel.prep.tile": 64, "kernel.dq.tile": 16},
        }
        for index, identity in enumerate(ids)
    ]
    return {
        "engine_id": 7,
        "graph_key": "graph",
        "graph_id": "graph",
        "device_id": "board",
        "device_arch": "gfx942",
        "problem_features": {"graph.flops": 4e12, "ws.scratch_offset": 512},
        "device_features": {"device.cu_count": 120},
        "engine_descriptor_id": "13ab344f-4818-4772-bb8e-8e1441fec82c",
        "engine_name": "hipkernel:slotted",
        "candidates": candidates,
        "total_count": len(candidates),
        "next_offset": None,
    }


def _slotted_sweep(page, marks):
    response = copy.deepcopy(page)
    response["results"] = [
        {
            "candidate_id": candidate["id"],
            "knob_settings": candidate["knob_settings"],
            "kernel_features": candidate["kernel_features"],
            "succeeded": True,
            "is_valid": True,
            "numerically_valid": True,
            "validation": "agrees_with_catalog",
            "robust_time_ms": 2.5,
            "min_time_ms": 2.0,
            "avg_time_ms": 2.6,
            "stddev_ms": 0.01,
            "iterations": 40,
            **({"slot_reference": mark} if mark is not None else {}),
        }
        for candidate, mark in zip(page["candidates"], marks)
    ]
    return response


def _collect(monkeypatch, tmp_path, response):
    from uhd_gen.generate import collect_graph

    monkeypatch.setattr("uhd_gen.generate._run_json", lambda *args, **kwargs: response)
    return collect_graph(
        ["hipdnn_bench", "--graph", "graph.json", "--engine-id", "7"],
        {},
        tmp_path,
        [],
        engine_descriptor_id="13ab344f-4818-4772-bb8e-8e1441fec82c",
    )


def test_collection_keeps_the_marker_and_the_composite_id_and_drops_ws_tokens(
    monkeypatch, tmp_path
):
    page = _slotted_page()
    marks = ["prep|dq", "dq", "prep", ""]
    rows, published = _collect(monkeypatch, tmp_path, _slotted_sweep(page, marks))
    assert [row["slot_reference"] for row in rows] == marks
    assert [row["kernel"] for row in rows] == [c["id"] for c in page["candidates"]]
    assert not [name for row in rows for name in row if name.startswith("ws.")]
    assert not [name for name in published if name.startswith("ws.")]
    assert "kernel.prep.tile" in published
    # ... and the rows split into per-slot training rows.
    assert [r["kernel"] for r in slot_training_rows(rows, "dq")] == [
        slot_kernels(page["candidates"][0]["id"])["dq"],
        slot_kernels(page["candidates"][1]["id"])["dq"],
    ]


def test_a_slotted_result_without_the_marker_is_refused(monkeypatch, tmp_path):
    page = _slotted_page()
    marks = ["prep|dq", "dq", None, ""]
    with pytest.raises(ValueError, match="slot_reference"):
        _collect(monkeypatch, tmp_path, _slotted_sweep(page, marks))


def test_slot_requires_a_dataset():
    import argparse

    from uhd_gen.generate import _check_generate_arguments, add_generate_arguments

    parser = argparse.ArgumentParser()
    add_generate_arguments(parser)
    args = parser.parse_args(
        [
            "--graphs",
            "g",
            "--descriptor-tree",
            "t",
            "--engine-id",
            "7",
            "--output-dir",
            "o",
            "--slot",
            "dq",
        ]
    )
    with pytest.raises(ValueError, match="--slot.*--dataset"):
        _check_generate_arguments(args, False, Path("o"), Path("t"))


def test_the_export_command_reports_an_unknown_slot_instead_of_a_traceback(
    tmp_path, capsys
):
    out = tmp_path / "rows.json"
    status = store.export_main(
        [
            "--from",
            str(_dataset(tmp_path)),
            "--out",
            str(out),
            "--purpose",
            "training",
            "--slot",
            "nope",
        ]
    )
    assert status == 1
    assert "no slot 'nope'" in capsys.readouterr().err
    assert not out.exists()


def test_knobs_judges_one_slots_knobs_on_its_reference_rows(tmp_path, capsys):
    from uhd_gen.__main__ import main

    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(store.export_rows(_dataset(tmp_path), purposes=["training"])),
        encoding="utf-8",
    )
    assert main(["knobs", "--input", str(corpus), "--slot", "dkdv"]) == 0
    report = capsys.readouterr().out
    # The slot's reference rows only (two problems, minus the failed run), and the slot's own
    # knob under its bare field name.
    assert "5 measurement(s)" in report
    assert "dkdv.tile" not in report and "dq.tile" not in report
