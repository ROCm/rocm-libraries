#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run held-out graphs with a candidate catalog ranker (L2) installed, and record whether the
runtime read it.

Standard library only, plus `uhd_gen promote` (through candidate_tree.py).

How good the ranking is gets scored offline (`uhd_gen evaluate` on the held-out graphs'
catalog sweeps); this asks whether the runtime READS the candidate at all. A ranker the loader
refuses still lets the engine run (it ranks by priority, then id, and logs why), so a plan that
builds proves nothing. For every graph the engine runs twice with `--collect-immediate`:

  incumbent  the root (`--root`) as it stands;
  candidate  a copy of it with the model promoted into the engine's UED (candidate_tree.py).

Both sides read one tree, the same one but for the candidate, so the comparison is the
model's alone, whichever root the engine is staged in.

Each run reads only the tree named for it and gets its own empty HIPDNN_CACHE_DIR
(engine_run.py), because a benchmark winner record takes precedence over any ranker and a run
writes one: shared, the record graph i's run left could serve graph j. The runtime says (at
INFO) when a record ordered the catalog instead of the ranker, and every such run is counted.
Every message the runtime prints that it is not using the model is recorded per graph.

Writes the rows; `decide.py l2` turns them into the verdict.

    validate_l2.py --bench <hipdnn_bench> --plugin-dir <dir> --engine-id <id> --engine <UED name>
        --root <per-arch descriptor root> --role sort_kernel_catalog --arch-key <gfx...|default>
        --candidate-dir <model dir> (--sample sample.json | --graphs ...) --work-dir <new dir>
        --out rows.json [--ranking-metric tflops|time] [--timeout 300]
        [--feature-evaluator <hipdnn_uhd_features>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_tree  # noqa: E402  (beside this script)
import engine_run  # noqa: E402

#: What the runtime says when it declines a model it was given: the descriptor loader
#: (DescriptorLoader.hpp), the ranker (UhdKernelHeuristic.hpp) and the UHD adapters. The plan
#: still builds, so this text is the only evidence.
REFUSAL = re.compile(
    r"the model is not used|ranking degrades to static_order|cannot be brought up"
    r"|declared-order fallback|disabling model, preserving engine|is not loaded"
    r"|reason=no_model_for_arch"
)
#: The leading `[...]` fields of a runtime log line.
LOG_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)*")
#: KernelIngestorStateManager.hpp, at INFO: a benchmarked record ordered the catalog, not the ranker.
FROM_RECORD = re.compile(r"catalog entries from a benchmarked\s*record")


def run(
    bench: str,
    graph: Path,
    plugin_dir: str,
    engine_id: str,
    env: dict,
    ranking_metric: str | None,
    timeout: int,
    scratch: Path,
) -> dict:
    """One `--collect-immediate` run: whether it ran, its timing, and what the runtime said
    about the model. `ok` is the plan running (exit 0), whatever it printed."""
    command = [
        bench,
        "--graph",
        str(graph),
        "--plugin-dir",
        plugin_dir,
        "--engine-id",
        engine_id,
        "--collect-immediate",
        "--json",
    ]
    if ranking_metric:
        command += ["--ranking-metric", ranking_metric]
    result = engine_run.call(command, env, timeout, scratch)
    if result["timed_out"]:
        return {
            "ok": False,
            "error": result["error"],
            "refusals": [],
            "timed_out": True,
        }
    output = result["stderr"] + "\n" + result["stdout"]
    lines = output.splitlines()
    # Without the log prefix (time, logger, level), so one refusal compares equal across the
    # two runs: decide.py counts only what the candidate run alone reports.
    refusals = sorted(
        {
            LOG_PREFIX.sub("", line).strip()[:400]
            for line in lines
            if REFUSAL.search(line)
        }
    )
    row = {
        "ok": result["returncode"] == 0,
        "refusals": refusals[:20],
        "from_record": bool(FROM_RECORD.search(output)),
    }
    if not row["ok"]:
        # At INFO the tail is teardown chatter; the runtime's own error lines say what failed.
        errors = [
            line.strip()
            for line in lines
            if "[error]" in line or line.lstrip().startswith("Error")
        ]
        row["error"] = (
            "\n".join(errors[-4:])[-1200:] if errors else result["error"][-600:]
        )
        return row
    response = result.get("response")
    if response is None:
        row["error"] = "hipdnn_bench emitted no JSON timing"
        return row
    timing = response.get("avgTimeMs", response.get("avg_time_ms"))
    if not isinstance(timing, (int, float)) or timing <= 0:
        row["error"] = "hipdnn_bench emitted no positive avgTimeMs"
    else:
        row["avgTimeMs"] = timing
    row["benchmark"] = response.get("graph_key")
    row["device"] = response.get("device") or response.get("device_id")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", required=True)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--engine", required=True, help="The engine (UED) name")
    parser.add_argument(
        "--root", required=True, help="The build's per-arch descriptor root"
    )
    parser.add_argument("--role", required=True)
    parser.add_argument("--arch-key", required=True)
    parser.add_argument("--candidate-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", help="eval_slice.py sample output")
    source.add_argument("--graphs", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ranking-metric")
    parser.add_argument(
        "--timeout", type=int, default=300, help="seconds per hipdnn_bench call"
    )
    parser.add_argument(
        "--feature-evaluator", help="The build's hipdnn_uhd_features, for promote"
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    farm = work / "descriptor_tree"
    candidate_tree.build_farm(Path(args.root), farm)
    install = candidate_tree.install_promoted(
        farm,
        Path(args.candidate_dir),
        args.engine,
        args.role,
        args.arch_key,
        args.feature_evaluator,
    )

    graphs = engine_run.graphs_of(args.sample, args.graphs or [], args.limit)
    # INFO, not warn: the record-served line is INFO; refusals are errors and print either way.
    sides = {
        "incumbent": dict(
            engine_run.isolated_env(Path(args.root)), HIPDNN_LOG_LEVEL="info"
        ),
        "candidate": dict(engine_run.isolated_env(farm), HIPDNN_LOG_LEVEL="info"),
    }
    rows = []
    for graph in graphs:
        row: dict = {"graph": str(graph)}
        for side, env in sides.items():
            row[side] = run(
                args.bench,
                graph,
                args.plugin_dir,
                args.engine_id,
                env,
                args.ranking_metric,
                args.timeout,
                work / "caches",
            )
        rows.append(row)
    document = {
        "schema": "hipdnn_uhd.catalog_validation_rows/1",
        "install": install,
        "graphs": len(graphs),
        "rows": rows,
        "winner_records": {
            side: sum(1 for r in rows if r[side].get("from_record")) for side in sides
        },
    }
    Path(args.out).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in document.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
