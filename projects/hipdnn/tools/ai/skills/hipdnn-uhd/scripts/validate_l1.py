#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ask the engine about held-out graphs with a candidate engine predictor (L1) installed.

Standard library only, plus `uhd_gen.immediate` (so a measured throughput is derived by the
rule the model was trained against) and `uhd_gen promote` for a descriptor engine.

For every graph it asks the engine three questions:

  measured   `--collect-immediate`: the engine runs its own selection and is timed.
  incumbent  `--predict-engine` against the root (`--descriptor-root`) as it stands: the
             model it binds now.
  candidate  `--predict-engine` with HIPDNN_DESCRIPTOR_DIR on a copy of that tree in which the
             candidate is the only document carrying the engine's declared id.

The copy is the point. An opaque engine binds its model by an id declared in provider code,
and the incumbent model already carries that id. The runtime's additive directory cannot
install a second document under it -- a file redefining an id the build defines is refused
and the build's definition stands (DescriptorLoader.hpp) -- so a candidate dropped there is
silently never read and "validation" would score the incumbent model. HIPDNN_DESCRIPTOR_DIR replaces the tree
instead; the copy dereferences links so kernels and packs stay inside the tree the runtime
reads (candidate_tree.py).

A descriptor engine's L1 (`--promote`, in order: the L2 its rows were measured under, then
the candidate) predicts the selection its L2 makes, so its truth is measured under the
candidate tree; the incumbent model's truth is measured under the root as it stands.

Every call reads only the tree named for it and gets an empty cache of its own, so neither a
descriptor root in the caller's environment nor a benchmark winner record answers in a
model's place (engine_run.py).

Writes the rows; `decide.py l1` turns them into the verdict.

    validate_l1.py --bench <hipdnn_bench> --plugin-dir <dir> --engine-id <id>
        --descriptor-root <dir> --candidate-dir <model dir> --declared-id <uuid>
        (--sample sample.json | --graphs <dir or file> ...) --work-dir <new dir> --out rows.json
        [--metric tflops|time] [--promote ENGINE ROLE ARCH DIR ...] [--timeout 300]
        [--feature-evaluator <hipdnn_uhd_features>]
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_tree  # noqa: E402  (beside this script)
import engine_run  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def opaque_tree(
    descriptor_root: Path, farm: Path, candidate_dir: Path, declared_id: str
) -> dict:
    """A contained copy whose only `declared_id` document is the candidate."""
    candidate_tree.build_farm(descriptor_root, farm)
    install = candidate_tree.install_opaque(farm, candidate_dir, declared_id)
    source = sorted(candidate_dir.glob("*.uhd.json"))[0]
    installed = farm / install["installed"]
    found = candidate_tree.carrying(farm, declared_id)
    return {
        "farm": str(farm),
        "removed": install["removed"],
        "candidate_id": str(_load(source).get("id", "")).lower(),
        "carrying_declared_id": [str(p.relative_to(farm)) for p in found],
        "unique": found == [installed],
        "byte_equal": filecmp.cmp(source, installed, shallow=False),
    }


def promoted_tree(
    descriptor_root: Path,
    farm: Path,
    promotes: list,
    candidate_dir: Path,
    declared_id: str,
    feature_evaluator: str | None = None,
) -> dict:
    """A descriptor engine's tree: each model promoted into its UED, in order."""
    candidate_tree.build_farm(descriptor_root, farm)
    installs = [
        candidate_tree.install_promoted(farm, Path(d), e, r, a, feature_evaluator)
        for e, r, a, d in promotes
    ]
    found = candidate_tree.carrying(farm, declared_id)
    # Promote rewrites the descriptor it installs, so the check that the model under test is
    # the one installed is on the artifact's bytes.
    source = sorted(candidate_dir.glob("*.uhd.json"))[0]
    artifact = candidate_tree.artifact_of(_load(source))
    equal = bool(found) and (
        not artifact
        or filecmp.cmp(
            candidate_dir / artifact,
            found[0].parent / candidate_tree.artifact_of(_load(found[0])),
            shallow=False,
        )
    )
    return {
        "farm": str(farm),
        "removed": [],
        "installs": installs,
        "candidate_id": str(_load(source).get("id", "")).lower(),
        "carrying_declared_id": [str(p.relative_to(farm)) for p in found],
        "unique": len(found) == 1,
        "byte_equal": equal,
    }


def prediction(answer: dict) -> dict:
    if not answer["ok"]:
        return {
            "status": "error",
            "error": answer["error"],
            "timed_out": answer.get("timed_out", False),
        }
    response = answer["response"]
    binding = (
        response.get("binding") if isinstance(response.get("binding"), dict) else {}
    )
    value = response.get("value", response.get("tflops"))
    metric = response.get("metric") or ("tflops" if "tflops" in response else None)
    return {
        "status": response.get("status"),
        "model": response.get("model"),
        "reason": response.get("reason"),
        "value": value,
        "metric": metric,
        "selector_revision": binding.get("selector_revision"),
        "trained_against": binding.get("trained_against"),
    }


#: The measured label each metric's prediction is compared with (RFC 0019 §13.4).
LABELS = {"tflops": "tflops", "time": "avgTimeMs"}


def measurement(answer: dict, metric: str) -> dict:
    if not answer["ok"]:
        return {
            "ok": False,
            "error": answer["error"],
            "timed_out": answer.get("timed_out", False),
        }
    from uhd_gen.immediate import normalize_row

    try:
        row = normalize_row(answer["response"])
    except ValueError as error:
        # The engine declined or failed this graph: not a model failure, and not scored.
        return {"ok": False, "error": str(error)}
    return {
        "ok": True,
        "value": row[LABELS[metric]],
        "metric": metric,
        "tflops": row.get("tflops"),
        "avgTimeMs": row["avgTimeMs"],
        "graph_key": row["benchmark"],
        "device": row["device"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", required=True)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--declared-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", help="eval_slice.py sample output")
    source.add_argument("--graphs", nargs="+")
    parser.add_argument(
        "--feature-evaluator", help="The build's hipdnn_uhd_features, for --promote"
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-measure", action="store_true")
    parser.add_argument("--metric", default="tflops", choices=sorted(LABELS))
    parser.add_argument(
        "--timeout", type=int, default=300, help="seconds per hipdnn_bench call"
    )
    parser.add_argument(
        "--promote",
        nargs=4,
        action="append",
        default=[],
        metavar=("ENGINE", "ROLE", "ARCH", "DIR"),
        help="A descriptor engine: promote DIR into ENGINE's UED (its L2 first, "
        "then the L1 candidate) instead of the declared-id swap",
    )
    args = parser.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    declared = args.declared_id.lower()
    if args.promote:
        tree = promoted_tree(
            Path(args.descriptor_root),
            work / "descriptor_tree",
            args.promote,
            Path(args.candidate_dir),
            declared,
            args.feature_evaluator,
        )
    else:
        tree = opaque_tree(
            Path(args.descriptor_root),
            work / "descriptor_tree",
            Path(args.candidate_dir),
            declared,
        )

    base_env = engine_run.isolated_env(Path(args.descriptor_root))
    candidate_env = engine_run.isolated_env(Path(tree["farm"]))
    caches = work / "caches"

    def ask(graph: Path, mode: str, env: dict) -> dict:
        command = [
            args.bench,
            "--graph",
            str(graph),
            "--engine-id",
            str(args.engine_id),
            "--plugin-dir",
            args.plugin_dir,
            mode,
            "--json",
            "--ranking-metric",
            args.metric,
        ]
        return engine_run.call(command, env, args.timeout, caches)

    rows = []
    graphs = engine_run.graphs_of(args.sample, args.graphs or [], args.limit)
    for index, graph in enumerate(graphs):
        row: dict = {"graph": str(graph)}
        if not args.no_measure:
            truth_env = candidate_env if args.promote else base_env
            row["measured"] = measurement(
                ask(graph, "--collect-immediate", truth_env), args.metric
            )
            if args.promote:
                row["measured_incumbent"] = measurement(
                    ask(graph, "--collect-immediate", base_env), args.metric
                )
        for side, env in (("incumbent", base_env), ("candidate", candidate_env)):
            row[side] = prediction(ask(graph, "--predict-engine", env))
        rows.append(row)
        if index % 25 == 0:
            print(
                f"[{index + 1}/{len(graphs)}] {graph.name}: candidate "
                f"{row['candidate'].get('status')}",
                flush=True,
            )

    Path(args.out).write_text(
        json.dumps(
            {
                "schema": "hipdnn_uhd.validation_rows/1",
                "declared_id": declared,
                "engine_id": str(args.engine_id),
                "measured": not args.no_measure,
                "metric": args.metric,
                "tree": tree,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"validated {len(rows)} graph(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
