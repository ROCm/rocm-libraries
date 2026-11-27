# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-stamp a model's `feature_semantics_revision`, but only with proof its inputs did not move.

The loader refuses any model whose recorded `trained_against.feature_semantics_revision`
(absent: 1) is not the build's `FEATURE_SEMANTICS_REVISION` exactly, so a revision bump
expires every shipped model -- including the many that read none of the features the bump
changed, or read them only on graphs where their values stayed the same. Retraining those
wastes a collection and a training run; accepting them softly at runtime would hide the
models the bump does break. This tool is the middle road, offline and auditable. It rewrites
the recorded revision only on one of two proofs, and otherwise refuses with "retrain":

- `unread`: the model reads none of the features the revisions in between changed, through
  its `features_signature` or its label (a `tflops` label is `graph.flops` over time).
- `measured-identity`: the model reads changed features, and the same graphs' published
  features, collected at the old revision and at the new one, agree on every one of them.
  Since a published feature is a function of the graph alone, an engine whose graphs never
  hold what a revision changed (an RMSNorm engine under an SDPA-backward fix) passes.

`REVISION_CHANGES` is the history of `FeatureSemantics.hpp`: what each revision may have
moved. A revision step this table does not describe is refused, never assumed harmless.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

from .corpus_io import read_corpus_frame, write_json
from .features import signature_references
from .provenance import FEATURE_SEMANTICS_REVISION, validate_provenance

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "uhd_gen.semantics_restamp_report/1"
#: The ignored `trained_against` member every re-stamp appends its proof to.
RECORD_MEMBER = "x-restamps"

#: Published features (full names, without `$`) whose values each revision may have changed,
#: as patterns. Must follow `FEATURE_SEMANTICS_REVISION` in FeatureSemantics.hpp: the newest
#: key is the build's revision (test_restamp_semantics checks it).
REVISION_CHANGES: dict[int, tuple[str, ...]] = {
    # SDPA backward counts each product on its own head dimension, max-pool backward counts
    # dy.numel, and a paged KV cache's bytes are its declared keys.
    2: (
        r"graph\.flops",
        r"graph\.nodes\[\d+\]\.flops",
        r"graph\.flops_by_type\.(SdpaBackwardAttributes|ResampleBwdAttributes)",
        r"graph\.logical_bytes",
        r"graph\.arithmetic_intensity",
    ),
}
LATEST_REVISION = max(REVISION_CHANGES)

#: The published features a metric's label is derived from (`immediate.normalize_row`).
METRIC_INPUTS = {"tflops": ("graph.flops",)}

#: Relative tolerance of `measured-identity`: the same double through two serializations.
RELATIVE_TOLERANCE = 1e-12

#: The columns a row names its graph by, in the immediate and the §8.3 spelling.
_GRAPH_COLUMNS = ("benchmark", "graph_key")


class RestampError(ValueError):
    """An input that cannot be judged at all (as opposed to a model that is refused)."""


def recorded_revision(trained_against: dict) -> int:
    return validate_provenance(trained_against).get(FEATURE_SEMANTICS_REVISION, 1)


def changed_patterns(recorded: int, target: int) -> list[re.Pattern]:
    """The features the revisions after `recorded` up to `target` may have changed."""
    unknown = [
        step for step in range(recorded + 1, target + 1) if step not in REVISION_CHANGES
    ]
    if unknown:
        raise RestampError(
            f"this uhd_gen does not know what feature semantics revision(s) {unknown} "
            "changed; update REVISION_CHANGES from FeatureSemantics.hpp"
        )
    return [
        re.compile(pattern)
        for step in range(recorded + 1, target + 1)
        for pattern in REVISION_CHANGES[step]
    ]


def read_changed(model: dict, patterns: list[re.Pattern]) -> list[str]:
    """The changed features `model` reads (full names, sorted), through its signature or its
    label."""
    read = {
        reference[1:]
        for reference in signature_references(model.get("features_signature", []))
    }
    metric = (
        model.get("score", {}).get("metric")
        if isinstance(model.get("score"), dict)
        else None
    )
    read.update(METRIC_INPUTS.get(metric, ()))
    return sorted(
        name for name in read if any(pattern.fullmatch(name) for pattern in patterns)
    )


# --------------------------------------------------------------------------------------
# Measured identity
# --------------------------------------------------------------------------------------


def _published(record: dict, names: list[str]) -> dict:
    """`names` as the row publishes them (None: absent), from its `features` map or, for a
    flattened row, its columns."""
    features = record.get("features")
    if isinstance(features, str):
        features = json.loads(features)
    if not isinstance(features, dict):
        features = record
    values = {}
    for name in names:
        value = features.get(name)
        if isinstance(value, float) and math.isnan(value):
            value = None
        values[name] = value
    return values


def graph_features(path: Path, names: list[str], which: str) -> dict:
    """{graph: {name: value or None}} over a corpus. A published feature is a function of the
    graph, so the rows of one graph (engines, devices, kernels) must agree."""
    frame = read_corpus_frame(Path(path))
    column = next((name for name in _GRAPH_COLUMNS if name in frame.columns), None)
    if column is None or frame.empty:
        raise RestampError(
            f"the {which} rows name no graph ({' or '.join(_GRAPH_COLUMNS)})"
        )
    graphs: dict = {}
    for record in frame.to_dict(orient="records"):
        graph = record[column]
        values = _published(record, names)
        if graphs.setdefault(graph, values) != values:
            raise RestampError(
                f"the {which} rows publish different features for graph {graph}; they "
                "were not collected by one build"
            )
    return graphs


def _same(before, after) -> bool:
    if before is None or after is None:
        return before is after
    if isinstance(before, bool) or isinstance(after, bool):
        return before == after
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return math.isclose(before, after, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0)
    return before == after


def measure_identity(
    rows_before: Path, rows_after: Path, names: list[str], min_graphs: int
) -> tuple[dict, list[str]]:
    """The identity summary of `names` over the graphs both corpora hold, and what fails it."""
    before = graph_features(rows_before, names, "before")
    after = graph_features(rows_after, names, "after")
    matched = sorted(before.keys() & after.keys())
    moved = [
        (graph, name, before[graph][name], after[graph][name])
        for graph in matched
        for name in names
        if not _same(before[graph][name], after[graph][name])
    ]
    summary = {
        "graphs": len(matched),
        "unmatched": len(before.keys() ^ after.keys()),
        "moved": len(moved),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "min_graphs": min_graphs,
    }
    problems = []
    if len(matched) < min_graphs:
        problems.append(
            f"only {len(matched)} graphs are in both corpora; identity needs at least "
            f"{min_graphs}"
        )
    if moved:
        graph, name, old, new = moved[0]
        problems.append(
            f"{len(moved)} published value(s) the model reads differ between the revisions "
            f"(e.g. {name} of graph {graph}: {old!r} -> {new!r}); retrain"
        )
    return summary, problems


# --------------------------------------------------------------------------------------
# The model file
# --------------------------------------------------------------------------------------


def _rewritten(text: str, document: dict) -> str:
    """`document` serialized the way `text` was, so the diff is the restamp and nothing else."""
    indent_match = re.search(r"^( +|\t+)\S", text, re.MULTILINE)
    indent = None
    if indent_match:
        width = indent_match.group(1)
        indent = width if "\t" in width else len(width)
    newline = "\r\n" if "\r\n" in text else "\n"
    body = json.dumps(
        document, indent=indent, ensure_ascii="\\u" in text, allow_nan=False
    )
    return body.replace("\n", newline) + (newline if text.endswith("\n") else "")


def _write_model(path: Path, text: str, document: dict) -> None:
    temporary = path.with_name(path.name + ".restamp.tmp")
    # newline="" because `_rewritten` already chose the file's own line ending.
    with open(temporary, "w", encoding="utf-8", newline="") as stream:
        stream.write(_rewritten(text, document))
    temporary.replace(path)


# --------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------


def restamp_semantics(
    model_path: Path,
    *,
    target: int = LATEST_REVISION,
    rows_before: Path | None = None,
    rows_after: Path | None = None,
    min_graphs: int = 30,
    write: bool,
) -> dict:
    """Judge a re-stamp to `target` and, if it passes and `write`, perform it. Returns the
    report."""
    report = {
        "schema": REPORT_SCHEMA,
        "passed": False,
        "already_current": False,
        "written": False,
        "model": str(model_path),
        "from": None,
        "to": target,
        "reads_changed": [],
        "proof": None,
        "identity": None,
        "problems": [],
    }
    try:
        _judge(
            report, Path(model_path), target, rows_before, rows_after, min_graphs, write
        )
    except (RestampError, ValueError, OSError) as error:
        report["problems"].append(str(error))
    report["passed"] = not report["problems"]
    return report


def _judge(
    report, model_path, target, rows_before, rows_after, min_graphs, write
) -> None:
    text = model_path.read_text(encoding="utf-8")
    model = json.loads(text)
    trained = model.get("trained_against")
    recorded = recorded_revision(trained)
    report["from"] = recorded
    if recorded == target:
        report["already_current"] = True
        return
    if recorded > target:
        raise RestampError(
            f"the model records feature semantics revision {recorded}, newer than {target}; "
            "a model is never re-stamped backwards"
        )
    changed = read_changed(model, changed_patterns(recorded, target))
    report["reads_changed"] = changed
    if not changed:
        report["proof"] = "unread"
    else:
        if rows_before is None or rows_after is None:
            raise RestampError(
                f"the model reads {len(changed)} feature(s) revision(s) {recorded + 1}-{target} "
                f"changed ({', '.join(changed)}); pass --rows-before (rows of its graphs "
                f"collected at revision {recorded}) and --rows-after (the same graphs at "
                f"revision {target}), or retrain"
            )
        summary, problems = measure_identity(
            rows_before, rows_after, changed, min_graphs
        )
        report["identity"] = summary
        if problems:
            report["problems"] += problems
            return
        report["proof"] = "measured-identity"

    if not write:
        return
    entry = {
        "member": FEATURE_SEMANTICS_REVISION,
        "from": recorded,
        "to": target,
        "proof": report["proof"],
        "compared": changed,
    }
    if report["identity"] is not None:
        entry["identity"] = report["identity"]
    history = trained.setdefault(RECORD_MEMBER, [])
    if not isinstance(history, list):
        raise RestampError(f"trained_against.{RECORD_MEMBER} is not an array")
    history.append(entry)
    trained[FEATURE_SEMANTICS_REVISION] = target
    _write_model(model_path, text, model)
    report["written"] = True


def add_restamp_semantics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="the model's .uhd.json")
    parser.add_argument(
        "--to",
        type=int,
        default=LATEST_REVISION,
        metavar="REVISION",
        help=f"the feature semantics revision to bind to (default {LATEST_REVISION}, this "
        "uhd_gen's newest)",
    )
    parser.add_argument(
        "--rows-before",
        metavar="CORPUS",
        help="rows of the model's graphs collected at its recorded revision",
    )
    parser.add_argument(
        "--rows-after",
        metavar="CORPUS",
        help="the same graphs' rows collected at the target revision",
    )
    parser.add_argument(
        "--min-graphs",
        type=int,
        default=30,
        help="Fewest graphs in both corpora that identity may rest on (default 30)",
    )
    parser.add_argument("--report", metavar="JSON", help="write the report here")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the model file; without it, only judge",
    )


def run_restamp_semantics(args: argparse.Namespace) -> int:
    report = restamp_semantics(
        Path(args.model),
        target=args.to,
        rows_before=Path(args.rows_before) if args.rows_before else None,
        rows_after=Path(args.rows_after) if args.rows_after else None,
        min_graphs=args.min_graphs,
        write=args.write,
    )
    if args.report:
        write_json(Path(args.report), report)
    for problem in report["problems"]:
        logger.error("%s", problem)
    if report["problems"]:
        print(f"REFUSED: {args.model} is not re-stamped.")
        return 1
    if report["already_current"]:
        print(
            f"{args.model} already records feature semantics revision {report['to']}."
        )
        return 0
    verb = "re-stamped" if report["written"] else "would be re-stamped (no --write)"
    detail = (
        "reads no changed feature"
        if report["proof"] == "unread"
        else f"{len(report['reads_changed'])} feature(s) identical over "
        f"{report['identity']['graphs']} graphs"
    )
    print(
        f"{args.model} {verb}: feature semantics revision {report['from']} -> "
        f"{report['to']} ({report['proof']}: {detail})"
    )
    return 0
