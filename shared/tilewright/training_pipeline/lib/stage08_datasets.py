# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How preflight reads the problems of a stage08 dataset, one at a time as
they are consumed: the entries of a bench yaml, the rows of an enriched
directory's chunk_*.csv files or the first measured row of every problem of
a log directory's block_*.log files. run_orchestrator.py's preflight reads
them to find the config's data types and layout in each dataset
(`_dataset_errors`).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Tuple

import yaml


def _yaml_problems(path: Path) -> Iterator[Mapping[str, Any]]:
    """The entries of a bench yaml: its `- {...}` lines one at a time, as
    benches are written, or the whole file when it has none."""
    flow = False
    with path.open() as f:
        for line in f:
            text = line.strip()
            if text.startswith("- {"):
                flow = True
                entry = yaml.safe_load(text[2:])
                if isinstance(entry, dict):
                    yield entry
    if not flow:
        entries = yaml.safe_load(path.read_text()) or []
        for entry in [entries] if isinstance(entries, dict) else entries:
            if isinstance(entry, dict):
                yield entry


def _csv_rows(chunks: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    for chunk in chunks:
        with chunk.open(newline="") as f:
            yield from csv.DictReader(f)


def _log_rows(logs: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    """The first measured row of every problem of the logs, in file order."""
    from .bench_log import iter_problems

    for log in logs:
        with log.open(errors="replace") as f:
            for prob in iter_problems(f):
                if prob.tested:
                    yield prob.tested[0].fields


def dataset_problems(path: Path) -> Tuple[str, Iterator[Mapping[str, Any]]]:
    """(kind, its problems, read as they are consumed) of a stage08 dataset:
    a bench yaml's entries, the rows of an enriched directory's chunk_*.csv
    files or the problems of a log directory's block_*.log files, file
    after file; kind `missing` or `unknown` without problems otherwise."""
    if not path.exists():
        return "missing", iter(())
    if path.is_file():
        if path.suffix.lower() not in (".yaml", ".yml"):
            return "unknown", iter(())
        return "yaml", _yaml_problems(path)
    chunks = sorted(path.glob("chunk_*.csv"))
    if chunks:
        return "enriched", _csv_rows(chunks)
    logs = sorted(path.glob("block_*.log"))
    if logs:
        return "logs", _log_rows(logs)
    return "unknown", iter(())
