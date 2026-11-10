# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`generate --bench-batch`: many graphs per hipdnn_bench process, against a stand-in bench
executable that answers `--graph` and `--graph-list` as hipdnn_bench does."""
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

# One immediate response per graph, as hipdnn_bench --collect-immediate writes it. A graph
# whose id names "crash" kills a --graph-list process when reached; one naming "stall" holds
# a --graph-list process for STALL_SECONDS before it answers; one naming "poison" is measured
# but fails every graph a --graph-list process measures after it, as a broken device or
# handle would; one naming "bad" fails on its own. Every process start is appended to
# calls.txt beside the script.
_BENCH = r"""
import json, os, sys, time
from pathlib import Path

STALL_SECONDS = 6
argv = sys.argv[1:]
here = Path(__file__).parent
with open(here / "calls.txt", "a", encoding="utf-8") as calls:
    calls.write(("list" if "--graph-list" in argv else "single") + "\n")
metric = argv[argv.index("--ranking-metric") + 1]
poisoned = False


def answer(path, listed):
    global poisoned
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if "crash" in graph["id"] and listed:
        sys.stdout.flush()
        os._exit(3)
    if "stall" in graph["id"] and listed:
        time.sleep(STALL_SECONDS)
    reason = "hipErrorIllegalAddress" if poisoned else None
    poisoned = poisoned or ("poison" in graph["id"] and listed)
    if "bad" in graph["id"]:
        reason = "no plan for this graph"
    if reason:
        line = {"engine_id": 7, "is_valid": False, "skip_reason": reason}
        if listed:
            line["graph_path"] = path
        print(json.dumps(line), flush=True)
        print(f"Engine query/collection failed: {reason}", file=sys.stderr)
        return False
    average = 1.0 + graph["size"] / 10
    line = {
        "engine_id": 7, "engine_name": "provider:engine7", "graph_id": graph["id"],
        "graph_key": graph["id"], "device_id": "board", "device_uuid": "uuid-of-gpu-0",
        "arch": "gfx942", "metric": metric,
        "binding": {"engine": "provider:engine7", "role": "predict_engine", "arch": "gfx942",
                    "metric": metric, "selector_revision": "provider-1",
                    "trained_against": {"selector_revision": "provider-1"}},
        "features": {"graph.flops": 2e9 * (graph["size"] + 1), "device.cu_count": 120},
        "avgTimeMs": average, "robustMeanMs": average * 0.9, "stddevMs": 0.01, "iters": 30,
        "is_valid": True, "selection_mode": "immediate", "timing_statistic": "robustMeanMs",
    }
    if listed:
        line["graph_path"] = path
    print(json.dumps(line), flush=True)
    return True


if "--graph-list" in argv:
    listed = Path(argv[argv.index("--graph-list") + 1]).read_text(encoding="utf-8").split()
    sys.exit(0 if all([answer(path, True) for path in listed]) else 1)
sys.exit(0 if answer(argv[argv.index("--graph") + 1], False) else 1)
"""


def _bench(directory: Path) -> str:
    script = directory / "bench.py"
    script.write_text(_BENCH, encoding="utf-8")
    if os.name == "nt":
        wrapper = directory / "bench.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        wrapper = directory / "bench"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        wrapper.chmod(0o755)
    return str(wrapper)


def _world(tmp_path: Path, ids: list[str]) -> dict:
    tree = tmp_path / "descriptors"
    tree.mkdir()
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for size, graph_id in enumerate(ids):
        (graphs / f"{size:02d}.json").write_text(
            json.dumps({"id": graph_id, "size": size}), encoding="utf-8"
        )
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    return {
        "tree": tree,
        "graphs": graphs,
        "bench": _bench(bench_dir),
        "calls": bench_dir / "calls.txt",
    }


def _collect(world: dict, output: Path, batch: int, *extra: str) -> int:
    from uhd_gen.__main__ import main

    if world["calls"].exists():
        world["calls"].unlink()
    return main(
        [
            "generate",
            "--collect-only",
            "--graphs",
            str(world["graphs"]),
            "--descriptor-tree",
            str(world["tree"]),
            "--engine-id",
            "7",
            "--role",
            "predict_engine",
            "--bench",
            world["bench"],
            "--bench-batch",
            str(batch),
            "--output-dir",
            str(output),
            *extra,
        ]
    )


def _rows(output: Path) -> dict:
    corpus = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    return {row["benchmark"]: (row["avgTimeMs"], row["robustMeanMs"]) for row in corpus}


def test_a_batched_collection_measures_what_one_process_per_graph_does(
    tmp_path, monkeypatch
):
    """Eight graphs in batches of four take two processes, and give the same rows."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, [f"g{i}" for i in range(8)])

    assert _collect(world, tmp_path / "single", 1) == 0
    single_calls = world["calls"].read_text(encoding="utf-8").split()
    assert _collect(world, tmp_path / "batched", 4) == 0
    batched_calls = world["calls"].read_text(encoding="utf-8").split()

    assert single_calls == ["single"] * 8
    assert batched_calls == ["list"] * 2
    assert _rows(tmp_path / "batched") == _rows(tmp_path / "single")
    assert len(_rows(tmp_path / "batched")) == 8


def test_a_batch_that_crashes_is_finished_one_graph_per_process(tmp_path, monkeypatch):
    """The graphs before the crash keep their batch answers; the crashing graph and the rest
    are measured alone, where only the graph that fails on its own is recorded as failed.
    """
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "crash1", "bad2", "g3"])

    assert _collect(world, tmp_path / "out", 4, "--max-graph-failures", "0.5") == 0

    assert (
        world["calls"].read_text(encoding="utf-8").split() == ["list"] + ["single"] * 3
    )
    assert set(_rows(tmp_path / "out")) == {"g0", "crash1", "g3"}
    manifest = json.loads(
        (tmp_path / "out" / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert [Path(failure["source"]).name for failure in manifest["failed_graphs"]] == [
        "02.json"
    ]
    assert "no plan for this graph" in manifest["failed_graphs"][0]["error"]


def test_graphs_a_poisoned_batch_fails_are_measured_alone(tmp_path, monkeypatch):
    """A batch whose device or handle breaks fails every later graph; measured alone, they
    succeed, so none of them is recorded as failed."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "poison1", "g2", "g3"])

    assert _collect(world, tmp_path / "out", 4) == 0

    assert (
        world["calls"].read_text(encoding="utf-8").split() == ["list"] + ["single"] * 2
    )
    assert set(_rows(tmp_path / "out")) == {"g0", "poison1", "g2", "g3"}
    manifest = json.loads(
        (tmp_path / "out" / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["failed_graphs"] == []


def test_a_batch_that_stalls_on_a_graph_is_cut_at_the_per_graph_timeout(
    tmp_path, monkeypatch
):
    """--bench-timeout bounds each graph of a batch, not the batch: a graph that stalls for
    3x the limit (well inside 8x, the whole batch's share) gets the batch killed, and the
    graphs it left are measured alone."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "stall1", *(f"g{i}" for i in range(2, 8))])

    assert _collect(world, tmp_path / "out", 8, "--bench-timeout", "2") == 0

    calls = world["calls"].read_text(encoding="utf-8").split()
    assert calls[0] == "list"
    assert calls[1:] and set(calls[1:]) == {"single"}
    assert len(_rows(tmp_path / "out")) == 8


def test_a_catalog_sweep_is_not_batched(tmp_path, caplog):
    from uhd_gen.__main__ import main

    world = _world(tmp_path, ["g0"])
    code = main(
        [
            "generate",
            "--collect-only",
            "--graphs",
            str(world["graphs"]),
            "--descriptor-tree",
            str(world["tree"]),
            "--engine-id",
            "7",
            "--role",
            "sort_kernel_catalog",
            "--bench",
            world["bench"],
            "--bench-batch",
            "4",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "--bench-batch batches engine-level" in caplog.text
