# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The per-case result row of an AOT conv sweep, and the merge across engines.

A ``--run-from-cache`` sweep leaves text: a block per case and a ``Best:``
line. That is readable one case at a time and useless for a few thousand of
them spread over eight cards, which is what
:mod:`benchmarks.common.run_conv_shards` produces. (The ``--csv`` of the
implicit-GEMM sweep does not cover this: it is wired into the JIT sweep's
per-case ranking, not into ``_run_from_cache``.)

So each sweep writes one JSON object per case as the case finishes --
:class:`ResultsWriter`, one call site per script, next to the ``Best:`` print --
and :func:`merge` joins the two engines' rows into one table.

The join key is :func:`benchmarks.common.early_stop.case_key`: the normalised
geometry plus dtype and direction, the key ``--early-stop-seed`` already uses to
carry a direct-conv time into the implicit-GEMM sweep. The two sweeps describe a
shape with different types (``DirectConvProblem`` / ``ConvProblem``) whose
``short()`` strings do not even have the same fields, so that key is the only
thing the two sides can be matched on -- and reusing it means a row cannot be
keyed differently from the seed that primed it.

Rows are appended and flushed per case: a shard that is killed after four hours
keeps the four hours.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: The two sweeps, by the name their rows carry.
ENGINES = ("direct", "implicit")

#: How many kernels below the winner each engine records, per case.
#:
#: Names only, no times: when the top few kernels are within the noise of the
#: measurement the winner is not stable between runs -- two runs of the same
#: case pick different instances at the same TFLOPS to three digits. What that
#: costs you is the *identity* of the kernel, so that is what the runner-up
#: columns carry; their times would only repeat the winner's.
RUNNERS_UP = 2


def _engine_fields(engine: str) -> tuple:
    """The merged table's columns for one engine, in order."""
    return (
        f"{engine}_tflops",
        f"{engine}_ms",
        f"{engine}_kernel",
        *(f"{engine}_kernel_{rank}" for rank in range(2, RUNNERS_UP + 2)),
    )


#: Columns of the merged table, in order.
MERGED_FIELDS = (
    "key",
    "shape",
    "dtype",
    "direction",
    *_engine_fields("direct"),
    *_engine_fields("implicit"),
    "winner",
    "speedup",
)


def add_results_arg(parser, *, merged: bool) -> None:
    """Add ``--results FILE`` to a sweep's parser.

    ``merged`` distinguishes the comparison driver (which writes one row per
    case with both engines on it) from a single sweep (one row per case, its
    own engine).
    """
    if merged:
        what = (
            "carrying the best direct-conv and implicit-GEMM kernel side by "
            f"side (plus the next {RUNNERS_UP} kernels of each by name, which "
            "is what a near-tie costs you), and a .csv of the same rows next "
            "to it. Written once both sweeps have finished, from the rows "
            "each of them recorded."
        )
    else:
        what = (
            f"carrying the best kernel, its time, the next {RUNNERS_UP} "
            "kernels by name, and the key benchmark_conv_compare.py joins the "
            "two sweeps on. Rows are appended as each case finishes, so an "
            "interrupted run keeps what it measured."
        )
    parser.add_argument(
        "--results",
        default=None,
        metavar="FILE",
        help=f"AOT (--run-from-cache) only: write FILE (JSON Lines), one row "
        f"per case {what}",
    )


def add_skip_measured_arg(parser) -> None:
    """Add ``--skip-measured FILE`` to a sweep's parser."""
    parser.add_argument(
        "--skip-measured",
        default=None,
        dest="skip_measured",
        metavar="FILE",
        help="AOT (--run-from-cache) only: skip every case this sweep already "
        "recorded in FILE (rows written by --results, this engine's only). "
        "Resumes a run that was interrupted without re-measuring what it "
        "finished.",
    )


def measured_keys(path, engine: str) -> set:
    """Case keys ``engine`` already has a row for in ``path``.

    One engine's rows only: a run killed between the direct sweep and the
    implicit one has the case measured on one side and not the other, and
    resuming has to finish the second rather than skip it.
    """
    if not path:
        return set()
    return {row["key"] for row in read_rows(path) if row.get("engine") == engine}


class ResultsWriter:
    """Appends one row per case to a JSON Lines file.

    A sweep builds one of these for the whole run (``None`` path: a writer that
    does nothing, so the call site needs no branch) and calls :meth:`row` where
    it prints the case's ``Best:`` line. A case that produced no result writes
    no row -- its absence is the finding, and the log says why.
    """

    def __init__(self, path: Optional[str], engine: str) -> None:
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        self._engine = engine
        self._path = Path(path) if path else None
        self._fh = None
        self._n = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("w", encoding="utf-8")

    @property
    def n_rows(self) -> int:
        return self._n

    def row(
        self,
        *,
        key: str,
        case: int,
        shape: str,
        dtype: str,
        direction: str,
        tflops: float,
        ms: float,
        gbps: float,
        kernel: str,
        candidates: int,
        passed: Optional[bool] = None,
        runners_up: Sequence[str] = (),
    ) -> None:
        """Record one case's best kernel. No-op without a path.

        ``runners_up`` are the names of the kernels ranked below it, best
        first; at most :data:`RUNNERS_UP` are kept. The caller passes the tail
        of the same ranking it prints, so the row says what the winner beat.
        """
        if self._fh is None:
            return
        record: Dict[str, Any] = {
            "key": key,
            "engine": self._engine,
            "case": int(case),
            "shape": shape,
            "dtype": dtype,
            "direction": direction,
            "tflops": round(float(tflops), 3),
            "ms": round(float(ms), 6),
            "gbps": round(float(gbps), 3),
            "kernel": kernel,
            "candidates": int(candidates),
        }
        if passed is not None:
            record["passed"] = bool(passed)
        if runners_up:
            record["runners_up"] = [str(name) for name in runners_up][:RUNNERS_UP]
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        # Per case, not per run: a shard killed mid-sweep keeps its rows.
        self._fh.flush()
        self._n += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ResultsWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_rows(path) -> List[dict]:
    """Rows of a JSON Lines results file; ``[]`` when there is none.

    A line that does not parse is skipped rather than fatal: the file is
    append-as-you-go, so a run killed mid-write can leave a partial last line.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: List[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "key" in row:
            out.append(row)
    return out


def merge(rows: Iterable[dict]) -> List[dict]:
    """Join per-engine rows into one row per case.

    Rows may come from both engines and from several shards. A key only one
    engine reached still gets a row with the other side left ``None`` -- that
    is the common and interesting case (a shape the direct kernels do not
    cover), not an error. Within one engine and key, the faster row wins, so
    merging a case measured twice is idempotent.

    A row's runner-up names travel with it: a slower row is dropped whole
    rather than having its ranking blended into the faster one's. A ranking
    only means something within the run that measured it -- two runs of one
    case can order the same near-tied kernels differently, and a mix of the
    two would be a ranking that never happened.

    Sorted by key, so two runs of the same case list produce comparable files.
    """
    by_key: Dict[str, dict] = {}
    for row in rows:
        key = row.get("key")
        engine = row.get("engine")
        if not key or engine not in ENGINES:
            continue
        merged = by_key.setdefault(
            key,
            {
                "key": key,
                "shape": row.get("shape"),
                "dtype": row.get("dtype"),
                "direction": row.get("direction"),
            },
        )
        ms = row.get("ms")
        prev = merged.get(f"{engine}_ms")
        if prev is not None and (ms is None or ms >= prev):
            continue
        merged[f"{engine}_tflops"] = row.get("tflops")
        merged[f"{engine}_ms"] = ms
        merged[f"{engine}_kernel"] = row.get("kernel")
        runners_up = row.get("runners_up") or ()
        for rank in range(2, RUNNERS_UP + 2):
            # Assigned even when absent: a faster row with a shorter ranking
            # (or none) has to clear the columns the row it replaced left.
            index = rank - 2
            merged[f"{engine}_kernel_{rank}"] = (
                runners_up[index] if index < len(runners_up) else None
            )

    out = []
    for key in sorted(by_key):
        merged = by_key[key]
        for field in MERGED_FIELDS:
            merged.setdefault(field, None)
        merged["winner"], merged["speedup"] = _verdict(
            merged["direct_ms"], merged["implicit_ms"]
        )
        out.append({field: merged[field] for field in MERGED_FIELDS})
    return out


def _verdict(direct_ms, implicit_ms):
    """``(winner, speedup)``: which engine was faster and by how much.

    ``speedup`` is the loser's time over the winner's, so it reads as "the
    winner is N times faster" whichever side won. Both ``None`` when only one
    engine reached the case -- a one-sided row is a fact about coverage, not a
    comparison.
    """
    if direct_ms is None or implicit_ms is None:
        return None, None
    if direct_ms <= implicit_ms:
        return "direct", round(implicit_ms / direct_ms, 3) if direct_ms else None
    return "implicit", round(direct_ms / implicit_ms, 3) if implicit_ms else None


def write(rows: Sequence[dict], path) -> Path:
    """Write merged ``rows`` as JSON Lines, plus a ``.csv`` beside it.

    Returns the CSV's path. Both are written whole (the merge has every row by
    then), unlike :class:`ResultsWriter`, which appends as a sweep goes.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = p.with_suffix(p.suffix + ".csv") if p.suffix else p.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MERGED_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def summary(rows: Sequence[dict]) -> str:
    """One line: how many cases, and how the two engines split them."""
    both = sum(1 for r in rows if r["direct_ms"] is not None and r["implicit_ms"])
    direct_only = sum(
        1 for r in rows if r["direct_ms"] is not None and r["implicit_ms"] is None
    )
    implicit_only = sum(
        1 for r in rows if r["direct_ms"] is None and r["implicit_ms"] is not None
    )
    wins = sum(1 for r in rows if r["winner"] == "direct")
    return (
        f"{len(rows)} case(s): {both} measured by both "
        f"(direct faster on {wins}), {direct_only} direct-only, "
        f"{implicit_only} implicit-only"
    )
