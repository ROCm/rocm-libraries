#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn an ingestor benchmark log into the RFC 0019.13 §8.3 training CSV.

The ingestor emits one JSON object per benchmarked kernel (see
`BenchmarkPlan::logCandidateTiming`). This reads those out of a log file and
writes the dataset envelope a UHD is trained on.

Why the log and not the winner cache: the cache keeps the winner alone, because
it is also read back at runtime to replay a decision, and it drops candidates
that failed to run so a broken kernel can never be served from it. Training needs
the losers and the failures. Only the log has both.

Five §8.3 columns describe the *sweep* rather than any single dispatch --
`collection_mode`, `problem_complete`, `shard_id`, `config_set_hash`,
`applicability_id`. The runtime has no notion of them, so they are supplied by
whoever drove the corpus and stamped onto every row here.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

CANDIDATE_EVENT = "ingestor.benchmark.candidate"

# Envelope columns, in the order they are written. Feature columns (`q.*`,
# `kernel.*`, `device.*`) are appended after these; RFC 0019.13 §8.3 reads columns
# by name and leaves order arbitrary, so this is only for readability.
ENVELOPE_COLUMNS = (
    "benchmark",
    # The device half of the problem identity, beside the graph half it is only half of.
    # The runtime keys its winner cache on (graph, device) because the same graph on two
    # GPUs is two problems with two different best kernels; a corpus merged from several
    # machines, or one sweep spanning two devices, therefore has to be grouped on both.
    # Grouping on `benchmark` alone would take RFC 0019.13 §11.2's per-problem oracle
    # `v*(p)` across devices, making every regret figure derived from it too small --
    # silently, and in the flattering direction.
    #
    # Dotless like every other envelope column, so FEATURE_KEY_MARKER below still sorts
    # it as envelope rather than discovering it as a feature. It sits beside the `device.*`
    # feature columns exactly as `kernel` sits beside `kernel.*`: identity, then properties.
    "device",
    "kernel",
    "pack",
    "dispatch",
    "minTimeMs",
    "avgTimeMs",
    "stddevMs",
    "robustMeanMs",
    "iters",
    "is_valid",
    "skip_reason",
    "collection_mode",
    "problem_complete",
    "shard_id",
    "config_set_hash",
    "applicability_id",
)

# What tells a feature key apart from an envelope key in a record. The runtime
# namespaces every feature it logs (`q.seqlen`, `kernel.tile_m`) and every envelope
# key is a bare word, so the dot is the whole test. Discovered rather than listed:
# a sweep over a different operation binds different problem tokens and different KMD
# fields, and a hardcoded set would silently drop the columns of any op but the one
# it was written for.
FEATURE_KEY_MARKER = "."


def feature_items(record: dict) -> Iterator[tuple[str, Any]]:
    """The (key, value) pairs of @p record that name a feature."""
    for key, value in record.items():
        if FEATURE_KEY_MARKER in key:
            yield key, value


@dataclass
class SweepProvenance:
    """The §8.7/§8.8 fields the runtime cannot know.

    Defaults describe the simplest honest case: one unsharded process that
    measured whatever the engine offered, making no claim to have covered a
    configuration set it never enumerated.
    """

    collection_mode: str = "exhaustive"
    problem_complete: bool = False
    shard_id: int = 0
    config_set_hash: str = ""
    applicability_id: str = ""


@dataclass
class ParseStats:
    lines: int = 0
    records: int = 0
    valid: int = 0
    failed: int = 0
    malformed: int = 0
    problems: set = field(default_factory=set)
    #: Every feature column the sweep produced. Reported so an operator sees at a
    #: glance that a log carried nothing to train on, rather than discovering it when
    #: `train --features` cannot find a column.
    feature_keys: set = field(default_factory=set)


def iter_candidate_records(lines: Iterable[str], stats: ParseStats) -> Iterator[dict]:
    """Yield the candidate records in a log stream, skipping everything else.

    A log file interleaves these with ordinary prose from every other component,
    and a line may carry a timestamp or severity prefix, so each is located by its
    first `{` and parsed. A line that looks like JSON but is not is counted rather
    than raised on: truncation at the tail of a killed run is normal, and losing a
    whole corpus to it would not be.
    """
    for line in lines:
        stats.lines += 1
        start = line.find("{")
        if start < 0:
            continue
        try:
            record = json.loads(line[start:])
        except json.JSONDecodeError:
            stats.malformed += 1
            continue
        if not isinstance(record, dict) or record.get("event") != CANDIDATE_EVENT:
            continue
        stats.records += 1
        yield record


def row_from_record(record: dict, provenance: SweepProvenance) -> dict[str, Any]:
    """One CSV row from one candidate record.

    A failed candidate keeps its identity and its reason and carries no timings:
    §8.3 rule 6 wants the timing columns empty when `is_valid` is False, so a
    zero can never be read as a measurement of zero. It keeps its features, though:
    a pair that could not run is a row the corpus needs placed in feature space.

    Identity includes the device: a failed candidate still belongs to a specific
    problem on a specific device, so `device` is written on every row exactly like
    `benchmark`. Empty when the record carries none, which is what a log collected
    before the runtime emitted the field looks like -- distinguishable from a real
    identity, so a consumer can degrade deliberately rather than silently grouping
    every machine's rows together (RFC 0019.13 §11.2).
    """
    succeeded = record.get("status") == "ok"
    row: dict[str, Any] = {
        "benchmark": record.get("benchmark", ""),
        "device": record.get("device", ""),
        "kernel": record.get("kernel", ""),
        "pack": record.get("pack", ""),
        "dispatch": record.get("dispatch", ""),
        "is_valid": "True" if succeeded else "False",
        "skip_reason": "" if succeeded else record.get("reason", "unknown"),
        "collection_mode": provenance.collection_mode,
        "problem_complete": "True" if provenance.problem_complete else "False",
        "shard_id": provenance.shard_id,
        "config_set_hash": provenance.config_set_hash,
        "applicability_id": provenance.applicability_id,
    }

    if succeeded:
        row["minTimeMs"] = record.get("min_ms", "")
        row["avgTimeMs"] = record.get("avg_ms", "")
        row["stddevMs"] = record.get("stddev_ms", "")
        row["robustMeanMs"] = record.get("robust_mean_ms", "")
        row["iters"] = record.get("iters", "")
    else:
        row["minTimeMs"] = ""
        row["avgTimeMs"] = ""
        row["stddevMs"] = ""
        row["robustMeanMs"] = ""
        row["iters"] = ""

    # Copied through untouched. The runtime logs raw values -- a string feature stays
    # a string -- and encoding one here would put a second, invisible encoding between
    # the sweep and the feature extractor that owns the real one (RFC 0019 §7).
    row.update(feature_items(record))

    return row


def convert(
    log_paths: list[Path],
    output_path: Path,
    provenance: SweepProvenance | None = None,
) -> ParseStats:
    """Write the §8.3 CSV for every candidate record across @p log_paths.

    Several logs because a sharded campaign produces one per worker and they
    concatenate: rows are independent, and `shard_id` distinguishes them.
    """
    provenance = provenance or SweepProvenance()
    stats = ParseStats()
    rows: list[dict[str, Any]] = []

    for log_path in log_paths:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for record in iter_candidate_records(handle, stats):
                row = row_from_record(record, provenance)
                # The pair, not the graph alone: a problem is (graph, device), so a
                # corpus merged from two machines has twice the problems, not the same
                # ones twice. Counting on `benchmark` alone would under-report exactly
                # where the RFC 0019.13 §11.2 oracle would be conflated.
                stats.problems.add((row["benchmark"], row["device"]))
                stats.feature_keys.update(key for key, _ in feature_items(record))
                if row["is_valid"] == "True":
                    stats.valid += 1
                else:
                    stats.failed += 1
                rows.append(row)

    # The header is the union across every row, sorted: two kernels of one sweep can
    # carry different KMD fields, and a row that lacks a key another row has must still
    # line up under the columns it does have rather than shifting every cell after it.
    # DictWriter fills the gap with its empty restval.
    fieldnames = list(ENVELOPE_COLUMNS) + sorted(stats.feature_keys)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Wrote %d row(s) to %s (%d measured, %d failed, %d problem(s), "
        "%d feature column(s), %d malformed line(s) skipped)",
        len(rows),
        output_path,
        stats.valid,
        stats.failed,
        len(stats.problems),
        len(stats.feature_keys),
        stats.malformed,
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uhd_gen.benchmark_log",
        description="Convert an ingestor benchmark log into the RFC 0019.13 §8.3 CSV.",
        epilog=(
            "Produce a log with:\n"
            "  HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=sweep.log <your sweep>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("logs", nargs="+", type=Path, help="benchmark log file(s)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="CSV to write")
    parser.add_argument(
        "--collection-mode",
        choices=("targeted", "exhaustive"),
        default="exhaustive",
        help="how this sweep's pairs were requested (RFC 0019.13 §8.7)",
    )
    parser.add_argument(
        "--problem-complete",
        action="store_true",
        help=(
            "assert every applicable configuration for each problem is present. "
            "Only true if the sweep enumerated the configuration set; the runtime "
            "cannot know it and does not claim it."
        ),
    )
    parser.add_argument("--shard-id", type=int, default=0, help="shard that produced these logs")
    parser.add_argument("--config-set-hash", default="", help="hash of the enumerated config set")
    parser.add_argument("--applicability-id", default="", help="identity of the applicability predicate")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stats = convert(
        args.logs,
        args.output,
        SweepProvenance(
            collection_mode=args.collection_mode,
            problem_complete=args.problem_complete,
            shard_id=args.shard_id,
            config_set_hash=args.config_set_hash,
            applicability_id=args.applicability_id,
        ),
    )

    if stats.records == 0:
        logger.error(
            "No '%s' records in %d line(s). The sweep must run with HIPDNN_LOG_LEVEL=info; "
            "at the default level the ingestor emits nothing.",
            CANDIDATE_EVENT,
            stats.lines,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
