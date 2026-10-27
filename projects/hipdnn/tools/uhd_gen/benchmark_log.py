#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn an ingestor benchmark log into the RFC 0019.13 §8.3 training CSV.

Reads the log, not the winner cache, because only the log keeps losers and failures.
The five sweep-level columns (collection_mode .. applicability_id) are unknown to the
runtime and are supplied by whoever drove the corpus.
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

# Envelope columns, in write order; feature columns follow. §8.3 reads by name.
ENVELOPE_COLUMNS = (
    "benchmark",
    # A problem is (graph, device): grouping on `benchmark` alone would take the §11.2
    # oracle across devices and understate regret. Dotless, so it stays envelope.
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

# Feature keys are namespaced (`kernel.tile_m`); envelope keys are bare words. Discovered,
# not listed, so sweeps of other operations keep their columns.
FEATURE_KEY_MARKER = "."


def feature_items(record: dict) -> Iterator[tuple[str, Any]]:
    """The (key, value) pairs of @p record that name a feature."""
    for key, value in record.items():
        if FEATURE_KEY_MARKER in key:
            yield key, value


@dataclass
class SweepProvenance:
    """The §8.7/§8.8 fields the runtime cannot know; defaults claim no completeness."""

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
    #: Every feature column the sweep produced, so an empty log is reported up front.
    feature_keys: set = field(default_factory=set)


def iter_candidate_records(lines: Iterable[str], stats: ParseStats) -> Iterator[dict]:
    """Yield the candidate records in a log stream, skipping everything else.

    Malformed JSON is counted, not raised: a killed run normally truncates its tail.
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

    A failed candidate keeps identity, reason and features but empty timings (§8.3 rule
    6). `device` is empty when the record has none. `is_valid`/`skip_reason` are the
    collection spelling; `uhd_gen.dataset` translates them to the published encoding.
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

    # Raw values: the feature extractor owns encoding (RFC 0019 §7).
    row.update(feature_items(record))

    return row


def convert(
    log_paths: list[Path],
    output_path: Path,
    provenance: SweepProvenance | None = None,
) -> ParseStats:
    """Write the §8.3 CSV for every candidate record across @p log_paths (e.g. shards)."""
    provenance = provenance or SweepProvenance()
    stats = ParseStats()
    rows: list[dict[str, Any]] = []

    for log_path in log_paths:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            for record in iter_candidate_records(handle, stats):
                row = row_from_record(record, provenance)
                # A problem is (graph, device).
                stats.problems.add((row["benchmark"], row["device"]))
                stats.feature_keys.update(key for key, _ in feature_items(record))
                if row["is_valid"] == "True":
                    stats.valid += 1
                else:
                    stats.failed += 1
                rows.append(row)

    # Header is the union across rows: kernels can carry different KMD fields, and
    # DictWriter leaves a missing key empty.
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
    parser.add_argument(
        "--shard-id", type=int, default=0, help="shard that produced these logs"
    )
    parser.add_argument(
        "--config-set-hash", default="", help="hash of the enumerated config set"
    )
    parser.add_argument(
        "--applicability-id", default="", help="identity of the applicability predicate"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
