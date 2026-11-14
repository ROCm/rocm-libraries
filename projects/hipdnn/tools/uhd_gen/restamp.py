# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-stamp an L1 model's `selector_revision`, but only with proof it still describes the engine.

The loader refuses any L1 model whose recorded `trained_against.selector_revision` is not the
build's exactly, so every edit to an engine's selector inputs expires its shipped models --
including edits that cannot change a prediction (a rebased kernel path, an unrelated cache
added to a shared source). Retraining for those wastes a collection; accepting them softly at
runtime would hide the edits that do matter. This tool is the middle road, offline and
auditable: it compares the selector description the model was trained under with the one this
build produces, and rewrites the recorded revision only when every change is provably (path
moved, bytes pinned by a hash) or measurably (the engine's own L1 labels before and after)
irrelevant. Anything else is a refusal that says to retrain. The runtime rule stays exact.

Selector dumps are written by the build (`HIPDNN_UHD_SELECTOR_DUMP_DIR`), one JSON object per
engine: `{"engine", "selector_revision", "selector": <description>}`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
from pathlib import Path

from .immediate import read_corpus, validate_binding, validate_model
from .ranking_metrics import is_valid_metric_value, ranking_metric

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "uhd_gen.restamp_report/1"
RECORD_MEMBER = "x-restamps"

GENERIC_KIND = "generic-untuned-v1"
NATIVE_KIND = "hkp-source-digest-v1"

#: A generic engine's kernel moved on disk: `sha256` still pins its bytes, so the revision
#: change carries no behavior. RFC 6901 pointers; `slots` is present for slotted engines.
_GENERIC_SOFT = re.compile(r"^(/slots/\d+)?/packs/[^/]+/kernels/[^/]+/source_file$")
#: A native input that is a code object, a table or kernel source decides what the engine
#: runs; parity on a corpus cannot vouch for the shapes the corpus did not cover.
_NATIVE_HARD = re.compile(r"\.(co|csv|hip)$|(^|/)codegen\.py$|(^|/)kernels/")


class RestampError(ValueError):
    """An input that cannot be judged at all (as opposed to a change that is refused)."""


# --------------------------------------------------------------------------------------
# Selector dumps
# --------------------------------------------------------------------------------------


def load_dump(path: Path) -> dict:
    try:
        dump = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RestampError(f"cannot read selector dump {path}: {error}") from error
    if (
        not isinstance(dump, dict)
        or not isinstance(dump.get("engine"), str)
        or not dump["engine"]
        or not isinstance(dump.get("selector_revision"), str)
        or not dump["selector_revision"]
        or not isinstance(dump.get("selector"), dict)
    ):
        raise RestampError(
            f"{path} is not a selector dump: expected an object with string `engine`, "
            "string `selector_revision` and object `selector`"
        )
    return dump


def _digest_lines(inputs: dict) -> str:
    # hkp_source_digest sorts the input paths, not the finished lines; the two orders differ
    # when one path is a prefix of another (`a` vs `a.b`).
    return "".join(f"{name}:{inputs[name]}\n" for name in sorted(inputs))


def dump_is_consistent(dump: dict) -> bool:
    """Whether `selector_revision` is what its own description hashes to.

    The C++ serializer is authoritative, so a mismatch is a warning about the dump (a stale or
    hand-edited file, or a serializer this tool has drifted from), never a verdict on the model.
    """
    description, revision = dump["selector"], dump["selector_revision"]
    kind = description.get("selector")
    if kind == GENERIC_KIND:
        text = json.dumps(
            description, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        )
        return revision == f"{GENERIC_KIND}/{hashlib.sha256(text.encode()).hexdigest()}"
    if kind == NATIVE_KIND:
        inputs = description.get("inputs")
        if not isinstance(inputs, dict) or not all(
            isinstance(value, str) for value in inputs.values()
        ):
            return False
        digest = hashlib.sha256(_digest_lines(inputs).encode()).hexdigest()[:16]
        return revision.endswith(f"/{digest}")
    return False


# --------------------------------------------------------------------------------------
# What changed, and whether a change can be vouched for
# --------------------------------------------------------------------------------------


def _pointer_token(token) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _diff_paths(before, after, path: str, out: list[str]) -> None:
    """JSON pointers of the smallest differing nodes of two JSON values."""
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            child = f"{path}/{_pointer_token(key)}"
            if key in before and key in after:
                _diff_paths(before[key], after[key], child, out)
            else:
                out.append(child)
    elif isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child = f"{path}/{index}"
            if index < len(before) and index < len(after):
                _diff_paths(before[index], after[index], child, out)
            else:
                out.append(child)
    elif type(before) is not type(after) or before != after:
        out.append(path)


def classify_changes(before: dict, after: dict) -> tuple[list[str], list[str], str]:
    """(hard, soft, kind) between two selector descriptions.

    A hard change cannot be re-stamped; a soft one can, with the proof `kind` calls for.
    """
    kind_before, kind_after = before.get("selector"), after.get("selector")
    if kind_before != kind_after:
        return (
            [f"selector kind changed: {kind_before!r} -> {kind_after!r}"],
            [],
            "unknown",
        )
    if kind_before == GENERIC_KIND:
        paths: list[str] = []
        _diff_paths(before, after, "", paths)
        soft = [path for path in paths if _GENERIC_SOFT.fullmatch(path)]
        return [path for path in paths if path not in soft], soft, GENERIC_KIND
    if kind_before == NATIVE_KIND:
        inputs_before, inputs_after = before.get("inputs"), after.get("inputs")
        if not isinstance(inputs_before, dict) or not isinstance(inputs_after, dict):
            return ["`inputs` is not an object"], [], NATIVE_KIND
        changed = sorted(
            name
            for name in inputs_before.keys() | inputs_after.keys()
            if inputs_before.get(name) != inputs_after.get(name)
        )
        hard = [name for name in changed if _NATIVE_HARD.search(name)]
        # Anything beside `inputs` has no soft reading.
        others: list[str] = []
        _diff_paths(
            {k: v for k, v in before.items() if k != "inputs"},
            {k: v for k, v in after.items() if k != "inputs"},
            "",
            others,
        )
        hard += others
        return hard, [name for name in changed if name not in hard], NATIVE_KIND
    return [f"unknown selector kind {kind_before!r}"], [], "unknown"


# --------------------------------------------------------------------------------------
# Measured parity
# --------------------------------------------------------------------------------------


def _engine_labels(
    frame,
    engine: str,
    arch: str | None,
    revision: str,
    label: str,
    metric: str,
    which: str,
) -> tuple[dict, str]:
    """{(graph, device): label or None} for one engine, and the arch the rows are of.

    None marks a row that is present but carries no usable label (a numerically wrong
    result, or a decline the corpus recorded).
    """
    frame = frame[frame["engine_name"].eq(engine)]
    if frame.empty:
        raise RestampError(f"the {which} corpus has no rows of engine {engine}")
    arches = sorted(frame["arch"].unique())
    if arch is not None:
        frame = frame[frame["arch"].eq(arch)]
        if frame.empty:
            raise RestampError(
                f"the {which} corpus has no rows of {engine} on {arch} (has {arches})"
            )
    elif len(arches) > 1:
        raise RestampError(
            f"the {which} corpus holds {engine} rows of several architectures {arches}; "
            "pass --arch"
        )
    for value in frame["binding"].unique():
        recorded = validate_binding(value)["selector_revision"]
        if recorded != revision:
            raise RestampError(
                f"the {which} corpus was collected at selector revision {recorded}, "
                f"not {revision}; rows at any other revision prove nothing about it"
            )
    labels = {}
    for graph, device, value in zip(frame["benchmark"], frame["device"], frame[label]):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        usable = math.isfinite(number) and is_valid_metric_value(metric, number)
        labels[(graph, device)] = number if usable else None
    return labels, str(frame["arch"].iloc[0])


def measure_parity(
    before_rows,
    after_rows,
    *,
    engine: str,
    arch: str | None,
    revision_before: str,
    revision_after: str,
    metric_name: str,
    thresholds: dict,
) -> tuple[dict, list[str]]:
    """The parity summary of an engine's labels across two revisions, and what fails it."""
    metric = ranking_metric(metric_name)
    before, arch_before = _engine_labels(
        before_rows, engine, arch, revision_before, metric.label, metric.name, "before"
    )
    after, arch_after = _engine_labels(
        after_rows, engine, arch, revision_after, metric.label, metric.name, "after"
    )
    if arch is None and arch_before != arch_after:
        raise RestampError(
            f"the corpora are of different architectures ({arch_before}, {arch_after})"
        )
    valid_before = {key for key, value in before.items() if value is not None}
    valid_after = {key for key, value in after.items() if value is not None}
    matched = sorted(valid_before & valid_after)
    # Only a graph the old corpus recorded without a label was declined then; one it never
    # measured says nothing either way, so it is reported, not judged.
    newly_served = sorted(valid_after & (before.keys() - valid_before))
    changes = [abs(math.log(after[key] / before[key])) for key in matched]
    median = statistics.median(changes) if changes else None
    outliers = (
        sum(change > thresholds["outlier_change"] for change in changes) / len(changes)
        if changes
        else None
    )
    summary = {
        "graphs": len(matched),
        "median_change": median,
        "outlier_fraction": outliers,
        "newly_declined": len(valid_before & (after.keys() - valid_after)),
        "newly_served": len(newly_served),
        "unmatched": len(after.keys() - before.keys()),
    }

    problems = []
    if len(matched) < thresholds["min_graphs"]:
        problems.append(
            f"only {len(matched)} graphs have a valid {metric.label} at both revisions; "
            f"parity needs at least {thresholds['min_graphs']}"
        )
    if median is not None and median > thresholds["max_median_change"]:
        problems.append(
            f"median {metric.label} change is {median:.4f} (|ln after/before|), above "
            f"{thresholds['max_median_change']}"
        )
    if outliers is not None and outliers > thresholds["max_outlier_fraction"]:
        problems.append(
            f"{outliers:.1%} of graphs changed by more than {thresholds['outlier_change']} "
            f"(|ln after/before|); at most {thresholds['max_outlier_fraction']:.1%} allowed"
        )
    if newly_served:
        problems.append(
            f"{len(newly_served)} graph(s) are served at the new revision that were not "
            f"served at the old one (e.g. {newly_served[0][0]}); the model never saw "
            "them as this engine's, so retrain"
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


def restamp(
    model_path: Path,
    selector_before: Path,
    selector_after: Path,
    *,
    rows_before: Path | None = None,
    rows_after: Path | None = None,
    arch: str | None = None,
    thresholds: dict,
    write: bool,
) -> dict:
    """Judge a restamp and, if it passes and `write`, perform it. Returns the report."""
    report = {
        "schema": REPORT_SCHEMA,
        "passed": False,
        "already_current": False,
        "written": False,
        "model": str(model_path),
        "from": None,
        "to": None,
        "engine": None,
        "changes": {"hard": [], "soft": []},
        "proof": None,
        "parity": None,
        "verified": {"before": None, "after": None},
        "problems": [],
    }
    try:
        _judge(
            report,
            model_path,
            selector_before,
            selector_after,
            rows_before,
            rows_after,
            arch,
            thresholds,
            write,
        )
    except (RestampError, ValueError, OSError) as error:
        report["problems"].append(str(error))
    report["passed"] = not report["problems"]
    return report


def _judge(
    report,
    model_path,
    selector_before,
    selector_after,
    rows_before,
    rows_after,
    arch,
    thresholds,
    write,
) -> None:
    text = Path(model_path).read_text(encoding="utf-8")
    model = json.loads(text)
    metric = validate_model(model)
    trained = model["trained_against"]
    recorded = trained.get("selector_revision")
    if recorded is None:
        raise RestampError(
            "the model records no trained_against.selector_revision, so it is not bound to "
            "a selector; there is nothing to re-stamp"
        )
    before, after = load_dump(selector_before), load_dump(selector_after)
    report.update({"from": before["selector_revision"], "engine": before["engine"]})
    report["to"] = after["selector_revision"]
    report["verified"] = {
        "before": dump_is_consistent(before),
        "after": dump_is_consistent(after),
    }
    for name, verified in report["verified"].items():
        if not verified:
            logger.warning(
                "the %s selector dump's revision is not the hash of its own description "
                "(a stale dump, or a serializer this tool has drifted from)",
                name,
            )

    if before["engine"] != after["engine"]:
        raise RestampError(
            f"the dumps are of different engines ({before['engine']}, {after['engine']})"
        )
    if recorded == after["selector_revision"]:
        report["already_current"] = True
        return
    if before["selector_revision"] != recorded:
        raise RestampError(
            f"the model was trained against {recorded}, but --selector-before is "
            f"{before['selector_revision']}; pass the dump from the build that trained it"
        )
    if before["selector_revision"] == after["selector_revision"]:
        report["already_current"] = True
        return

    hard, soft, kind = classify_changes(before["selector"], after["selector"])
    report["changes"] = {"hard": hard, "soft": soft}
    if hard:
        raise RestampError(
            f"{len(hard)} change(s) between the revisions cannot be re-stamped; retrain: "
            + "; ".join(hard)
        )
    if not soft:
        raise RestampError(
            "the two selector descriptions are identical, yet their revisions differ; one "
            "dump does not describe the revision it names"
        )

    if kind == GENERIC_KIND:
        report["proof"] = "path-only"
    else:
        if rows_before is None or rows_after is None:
            raise RestampError(
                f"{len(soft)} native source change(s) need measured parity: pass "
                "--rows-before (L1 corpus at the old revision) and --rows-after "
                "(at the new one)"
            )
        summary, problems = measure_parity(
            read_corpus(Path(rows_before)),
            read_corpus(Path(rows_after)),
            engine=before["engine"],
            arch=arch,
            revision_before=before["selector_revision"],
            revision_after=after["selector_revision"],
            metric_name=metric.name,
            thresholds=thresholds,
        )
        report["parity"] = summary
        if problems:
            report["problems"] += problems
            return
        report["proof"] = "measured-parity"

    if not write:
        return
    entry = {
        "from": before["selector_revision"],
        "to": after["selector_revision"],
        "proof": report["proof"],
        "changes": soft,
    }
    if report["parity"] is not None:
        entry["parity"] = report["parity"]
    history = trained.setdefault(RECORD_MEMBER, [])
    if not isinstance(history, list):
        raise RestampError(f"trained_against.{RECORD_MEMBER} is not an array")
    history.append(entry)
    trained["selector_revision"] = after["selector_revision"]
    _write_model(Path(model_path), text, model)
    report["written"] = True


def add_restamp_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="the L1 model's .uhd.json")
    parser.add_argument(
        "--selector-before",
        required=True,
        metavar="DUMP",
        help="selector dump from the build the model was trained at",
    )
    parser.add_argument(
        "--selector-after",
        required=True,
        metavar="DUMP",
        help="selector dump from the build to bind the model to",
    )
    parser.add_argument(
        "--rows-before",
        metavar="CORPUS",
        help="the engine's L1 corpus at the old revision (native source changes)",
    )
    parser.add_argument(
        "--rows-after",
        metavar="CORPUS",
        help="the engine's L1 corpus at the new revision, over the same graphs",
    )
    parser.add_argument(
        "--arch",
        help="the architecture to compare on (default: the only one in the corpora)",
    )
    parser.add_argument(
        "--max-median-change",
        type=float,
        default=0.03,
        help="Largest median |ln(after/before)| of the label (default 0.03)",
    )
    parser.add_argument(
        "--outlier-change",
        type=float,
        default=0.10,
        help="A graph whose |ln(after/before)| exceeds this is an outlier (default 0.10)",
    )
    parser.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=0.05,
        help="Largest fraction of matched graphs that may be outliers (default 0.05)",
    )
    parser.add_argument(
        "--min-graphs",
        type=int,
        default=30,
        help="Fewest graphs valid at both revisions that parity may rest on (default 30)",
    )
    parser.add_argument("--report", metavar="JSON", help="write the report here")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the model file; without it, only judge",
    )


def run_restamp(args: argparse.Namespace) -> int:
    report = restamp(
        Path(args.model),
        Path(args.selector_before),
        Path(args.selector_after),
        rows_before=Path(args.rows_before) if args.rows_before else None,
        rows_after=Path(args.rows_after) if args.rows_after else None,
        arch=args.arch,
        thresholds={
            "max_median_change": args.max_median_change,
            "outlier_change": args.outlier_change,
            "max_outlier_fraction": args.max_outlier_fraction,
            "min_graphs": args.min_graphs,
        },
        write=args.write,
    )
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    for problem in report["problems"]:
        logger.error("%s", problem)
    if report["problems"]:
        print(f"REFUSED: {args.model} is not re-stamped.")
        return 1
    if report["already_current"]:
        print(f"{args.model} already records the current revision; nothing to do.")
        return 0
    verb = "re-stamped" if report["written"] else "would be re-stamped (no --write)"
    print(
        f"{args.model} {verb}: {report['from']} -> {report['to']} "
        f"({report['proof']}; {len(report['changes']['soft'])} change(s))"
    )
    if report["parity"]:
        parity = report["parity"]
        print(
            f"  parity over {parity['graphs']} graphs: median change "
            f"{parity['median_change']:.4f}, outliers {parity['outlier_fraction']:.1%}, "
            f"{parity['newly_declined']} newly declined"
        )
    return 0
