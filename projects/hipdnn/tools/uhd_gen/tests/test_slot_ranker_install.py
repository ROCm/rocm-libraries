# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A slot ranker is installable end to end with the shipped tools (RFC 0021 §9).

The synthetic slotted sweep (`fixtures/slotted_sweep.csv`) is added to a dataset, one slot's
ranker is generated from it, and `promote` installs it into that slot of the engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("lightgbm")
pytest.importorskip("flatbuffers")

from uhd_gen.__main__ import main  # noqa: E402
from uhd_gen.dataset import store  # noqa: E402

SWEEP = Path(__file__).parent / "fixtures" / "slotted_sweep.csv"
UED = "1f2a3b4c-5d6e-4f70-8192-a3b4c5d6e7f8"
KMDS = {
    "prep": "a10c7e1d-5b3f-4c22-8d61-0e9f4a7b2c31",
    "dkdv": "b27e90a4-1d6c-4f58-9a03-7c2e5d81f640",
    "dq": "d48a31bc-6e07-4d95-b1f2-3a8c0e5792d6",
}
MATCH = "8d4f3144-02a7-4ec2-b79c-e849951f8c24"


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _tree(root: Path) -> Path:
    """The fixture's engine: three slots, each KMD declaring and each slot exposing `tile`."""
    _write(
        root / "engine.ued.json",
        {
            "version": "1.1",
            "id": UED,
            "name": "test:slotted",
            "slots": [
                {"name": slot, "metadata": kmd, "knobs": ["tile"]}
                for slot, kmd in KMDS.items()
            ],
        },
    )
    for number, (slot, kmd) in enumerate(KMDS.items()):
        _write(
            root / f"{slot}.kmd.json",
            {
                "version": "1.0",
                "id": kmd,
                "fields": [{"name": "tile", "type": "int"}],
            },
        )
        _write(
            root / f"{slot}.kdp.json",
            {
                "version": "1.1",
                "id": f"{number:02d}be5fe7-02a7-4ec2-b79c-e849951f8c24",
                "engine": UED,
                "slot": slot,
                "arch": ["gfx942"],
                "matchers": [MATCH],
            },
        )
    _write(
        root / "match.umd.json",
        {"version": "1.0", "id": MATCH, "scope": "kernel", "match_symbol": "t.match"},
    )
    return root


def _many_problems(tmp_path: Path, count: int = 12) -> Path:
    """The fixture's two problems repeated at `count` problem sizes, enough to hold some out."""
    sweep = pd.read_csv(SWEEP)
    frames = []
    for number in range(count):
        copy = sweep[sweep["benchmark"] == f"sweep_{512 * (1 + number % 2)}"].copy()
        size = 512 + 64 * number
        copy["benchmark"] = f"sweep_{size}"
        copy["q.seqlen"] = size
        frames.append(copy)
    path = tmp_path / "sweep.csv"
    pd.concat(frames).to_csv(path, index=False)
    return path


def _add_sweep(tmp_path: Path, sweep: Path) -> Path:
    dataset = tmp_path / "ds"
    store.add(
        dataset,
        *store.from_csv(
            [sweep],
            engine="SLOTTED_FIXTURE",
            engine_id=7,
            role="sort_kernel_catalog",
            arch="gfx942",
            selector_revision="fixture/1",
            collected_at="2026-10-18T00:00:00+00:00",
            label="slotted-sweep",
        ),
    )
    return dataset


def _generate(dataset, tree, evaluator, output, *objective) -> int:
    """`generate --slot dkdv` over the sweep; `objective` is the target or the metric."""
    return main(
        [
            "generate",
            "--dataset",
            str(dataset),
            "--slot",
            "dkdv",
            "--descriptor-tree",
            str(tree),
            "--engine",
            "test:slotted",
            "--arch",
            "gfx942",
            "--role",
            "sort_kernel_catalog",
            "--no-promote",
            *objective,
            "--features",
            "q.seqlen",
            "kernel.tile",
            "--num-boost-round",
            "10",
            "--early-stopping",
            "3",
            "--eval-fraction",
            "0.25",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(output),
        ]
    )


def test_a_slot_ranker_trained_from_a_sweep_installs_into_its_slot(tmp_path, evaluator):
    dataset = _add_sweep(tmp_path, _many_problems(tmp_path))
    tree = _tree(tmp_path / "tree")
    output = tmp_path / "out"
    assert (
        _generate(
            dataset,
            tree,
            evaluator,
            output,
            "--target",
            "avgTimeMs",
            "--objective",
            "min",
        )
        == 0
    )
    trained = json.loads((output / "model" / "heuristic.uhd.json").read_text("utf-8"))
    assert trained["version"] == "1.1"
    assert trained["trained_against"]["slot"] == "dkdv"
    # Only this slot's KMD: another slot's revision cannot invalidate the ranker.
    assert trained["trained_against"]["kmd"]["id"] == KMDS["dkdv"]

    assert (
        main(
            [
                "promote",
                "--model-dir",
                str(output / "model"),
                "--descriptor-tree",
                str(tree),
                "--engine",
                "test:slotted",
                "--role",
                "sort_kernel_catalog",
                "--arch",
                "gfx942",
                "--corpus",
                str(output / "corpus.json"),
                "--feature-evaluator",
                evaluator,
            ]
        )
        == 0
    )
    slots = json.loads((tree / "engine.ued.json").read_text("utf-8"))["slots"]
    installed_id = slots[1]["sort_kernel_catalog"]["gfx942"][0]
    assert (
        "sort_kernel_catalog" not in slots[0] and "sort_kernel_catalog" not in slots[2]
    )
    installed = next(
        json.loads(path.read_text("utf-8"))
        for path in (tree / "heuristics" / UED / "dkdv").rglob("*.uhd.json")
    )
    assert installed["id"] == installed_id
    assert installed["version"] == "1.1"
    assert installed["trained_against"]["slot"] == "dkdv"


def test_a_slot_ranker_that_declares_a_metric_is_refused(tmp_path, evaluator, caplog):
    """The loader drops a slotted engine whose slot ranker declares `score.metric`
    (RFC 0021 §7), so `generate --slot` refuses to produce one."""
    sweep = _many_problems(tmp_path)
    # `--metric tflops` trains on the tflops the dataset derives from reported flops.
    frame = pd.read_csv(sweep)
    frame["graph.flops"] = 4e12
    frame.to_csv(sweep, index=False)
    dataset = _add_sweep(tmp_path, sweep)
    output = tmp_path / "out"
    assert (
        _generate(
            dataset, _tree(tmp_path / "tree"), evaluator, output, "--metric", "tflops"
        )
        != 0
    )
    assert "a slot ranker must be metric-less" in caplog.text


def test_generate_refuses_to_measure_a_slotted_engine(
    monkeypatch, tmp_path, evaluator, caplog
):
    """A slotted engine is swept with `hipdnn_bench --sweep`, not measured by `generate`."""
    from uhd_gen.tests.bench_stub import find_any_bench

    find_any_bench(monkeypatch)
    graph = tmp_path / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    assert (
        main(
            [
                "generate",
                "--graphs",
                str(graph),
                "--descriptor-tree",
                str(_tree(tmp_path / "tree")),
                "--engine",
                "test:slotted",
                "--engine-id",
                "7",
                "--arch",
                "gfx942",
                "--role",
                "sort_kernel_catalog",
                "--feature-evaluator",
                evaluator,
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        != 0
    )
    assert "generate measures an engine that declares no slots" in caplog.text
