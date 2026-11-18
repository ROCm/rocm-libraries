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
and :func:`merge` joins the engines' rows into one table. The engines are the
direct-conv and implicit-GEMM sweeps, and the AOT Winograd pipeline
(``benchmark_winograd_aot.py``) on the 3x3 stride-1 forward cases it runs.

The join key is :func:`benchmarks.common.early_stop.case_key`: the normalised
geometry plus dtype and direction, the key ``--early-stop-seed`` already uses to
carry a direct-conv time into the implicit-GEMM sweep. The two sweeps describe a
shape with different types (``DirectConvProblem`` / ``ConvProblem``) whose
``short()`` strings do not even have the same fields, so that key is the only
thing the two sides can be matched on -- and reusing it means a row cannot be
keyed differently from the seed that primed it.

The merge writes two files: the JSON keeps every field, and the ``.csv`` beside
it is the view you read -- the MIOpenDriver command of the case, the instance
each engine picked, what they cost, and the winner. The command is not in the
rows (they are keyed by geometry, which is what joins the two engines), so the
merge puts it back from the run's own input, or writes one from the key when
the run was given a shape rather than a command.

Rows are appended and flushed per case: a shard that is killed after four hours
keeps the four hours.

Every row says how its time was measured (``timing``, one of
:data:`benchmarks.common.batched_timing.TIMINGS`; a row without the field is
from before it existed, when every sweep timed block). The protocols differ by a fixed
per-call cost, so rows of two protocols are never compared with each other:
resuming skips only cases measured the same way, and a merge takes one
protocol's rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from benchmarks.common.batched_timing import BLOCK_TIMING, TIMINGS

#: The sweeps, by the name their rows carry.
ENGINES = ("direct", "implicit", "winograd")

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


#: Fields of a merged row, in order. The JSON file carries all of them.
MERGED_FIELDS = (
    "key",
    "miopen_cmd",
    "shape",
    "dtype",
    "direction",
    *_engine_fields("direct"),
    *_engine_fields("implicit"),
    *_engine_fields("winograd"),
    "winner",
    "winner_ms",
    "winner_kernel",
    "speedup",
    "timing",
)

#: Columns of the ``.csv`` beside it, in order.
#:
#: A narrower view of the same rows, in the order a comparison is read: what
#: was run, which instance each engine picked, what they cost, who won. The
#: JSON keeps the rest (tflops, the key, the shape and its dtype/direction)
#: for anything that reads these files by field rather than by eye.
#:
#: The implicit side lists its whole ranking because that is where near-ties
#: live -- the three names are what re-picking the winner would cost you. The
#: direct and Winograd sides list only their winner: the tables are joined per
#: case, and a row wide enough to hold every ranking is one nobody reads.
CSV_FIELDS = (
    "miopen_cmd",
    "direct_kernel",
    "implicit_kernel",
    *(f"implicit_kernel_{rank}" for rank in range(2, RUNNERS_UP + 2)),
    "winograd_kernel",
    "direct_ms",
    "implicit_ms",
    "winograd_ms",
    "winner_ms",
    "winner_kernel",
    "timing",
)


def add_results_arg(parser, *, merged: bool) -> None:
    """Add ``--results FILE`` to a sweep's parser.

    ``merged`` distinguishes the comparison driver (which writes one row per
    case with both engines on it) from a single sweep (one row per case, its
    own engine).
    """
    if merged:
        what = (
            "carrying the best direct-conv, implicit-GEMM and Winograd kernel "
            f"side by side (plus the next {RUNNERS_UP} kernels of each by name, "
            "which is what a near-tie costs you), and a .csv of the same rows "
            "next to it -- the .csv narrowed to the reading columns: the "
            "MIOpenDriver command, each engine's pick, their times, and the "
            "winner. Written once every sweep has finished, from the rows "
            "each of them recorded."
        )
    else:
        what = (
            f"carrying the best kernel, its time, the next {RUNNERS_UP} "
            "kernels by name, and the key benchmark_conv_compare.py joins the "
            "sweeps on. Rows are appended as each case finishes, so an "
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


def same_results_file(results_path, skip_measured_path) -> bool:
    """Whether ``--results`` and ``--skip-measured`` name the same file.

    Call before constructing :class:`ResultsWriter`: a sweep resuming into its
    own results file must open it for append, not truncate it out from under
    the skip set it was just read for.
    """
    if not results_path or not skip_measured_path:
        return False
    try:
        return Path(results_path).resolve() == Path(skip_measured_path).resolve()
    except OSError:
        return Path(results_path) == Path(skip_measured_path)


def row_timing(row: dict) -> str:
    """How ``row``'s time was measured; rows older than the field are block."""
    return row.get("timing") or BLOCK_TIMING


def measured_keys(path, engine: str, timing: str = BLOCK_TIMING) -> set:
    """Case keys ``engine`` already has a row for in ``path``, timed by ``timing``.

    One engine's rows only: a run killed between the direct sweep and the
    implicit one has the case measured on one side and not the other, and
    resuming has to finish the second rather than skip it. One protocol's rows
    only, for the same reason: a case measured the other way is not measured
    for this run.
    """
    if not path:
        return set()
    return {
        row["key"]
        for row in read_rows(path)
        if row.get("engine") == engine and row_timing(row) == timing
    }


def expand_merged_for_skip(rows: Iterable[dict]) -> List[dict]:
    """Per-engine skip rows from already-merged (:data:`MERGED_FIELDS`) rows.

    A merged row carries every engine's time under ``{engine}_ms`` and no
    ``engine`` field of its own, so it is not what :func:`measured_keys` (and
    the sweeps' own ``--skip-measured`` reader) looks for. One output row per
    engine side the merged row actually has a time for -- :func:`merge` leaves
    a side ``None`` when that engine never reached the case, and that side is
    not "measured" to skip.
    """
    out: List[dict] = []
    for row in rows:
        key = row.get("key")
        if not key:
            continue
        for engine in ENGINES:
            ms = row.get(f"{engine}_ms")
            if ms is None:
                continue
            out.append(
                {"key": key, "engine": engine, "ms": ms, "timing": row.get("timing")}
            )
    return out


class ResultsWriter:
    """Appends one row per case to a JSON Lines file.

    A sweep builds one of these for the whole run (``None`` path: a writer that
    does nothing, so the call site needs no branch) and calls :meth:`row` where
    it prints the case's ``Best:`` line. A case that produced no result writes
    no row -- its absence is the finding, and the log says why.
    """

    def __init__(
        self,
        path: Optional[str],
        engine: str,
        timing: str = BLOCK_TIMING,
        *,
        append: bool = False,
    ) -> None:
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        if timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, got {timing!r}")
        self._engine = engine
        self._timing = timing
        self._path = Path(path) if path else None
        self._fh = None
        self._n = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # ``append``: ``--results`` and ``--skip-measured`` name the same
            # file, so the rows ``measured_keys`` already read for this run's
            # skip set are the run's own history, not scratch to discard.
            self._fh = self._path.open("a" if append else "w", encoding="utf-8")

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
            "timing": self._timing,
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


#: MIOpenDriver keyword per dtype, and ``-F`` value per direction.
_MIOPEN_DRIVER = {"fp32": "conv", "fp16": "convfp16", "bf16": "convbfp16"}
_MIOPEN_FORW = {"fwd": 1, "dgrad": 2, "wgrad": 4}

#: ``(MIOpenDriver flag, geometry field)`` of a 2-D command, in driver order.
_MIOPEN_2D_FLAGS = (
    ("-n", "N"),
    ("-c", "C"),
    ("-H", "Hi"),
    ("-W", "Wi"),
    ("-k", "K"),
    ("-y", "Y"),
    ("-x", "X"),
    ("-p", "pH"),
    ("-q", "pW"),
    ("-u", "sH"),
    ("-v", "sW"),
    ("-l", "dH"),
    ("-j", "dW"),
    ("-g", "groups"),
)

#: The depth flags a 3-D command adds, long-form.
_MIOPEN_3D_FLAGS = (
    ("--in_d", "Di"),
    ("--fil_d", "Z"),
    ("--conv_stride_d", "sD"),
    ("--pad_d", "pD"),
    ("--dilation_d", "dD"),
)


def miopen_cmd_from_key(key: str) -> str:
    """A MIOpenDriver command for ``key``, synthesised from its geometry.

    A case measured from explicit shape flags (or a JSON case file) has no
    command behind it, and a row identified only by its kernel names is one
    you cannot re-run. The key carries every field
    :func:`benchmarks.common.early_stop.case_key` put in it, so the command is
    a rewrite of the key rather than new information: 2-D commands round-trip
    back through ``parse_miopen_cmd``.

    3-D keys get the depth extents as long-form flags -- the sweeps' MIOpen
    parsers are 2-D only, so nothing here reads them back; they are there so
    the row still says what was run. Returns ``key`` unchanged when it does not
    parse (a file written by a future version, say).
    """
    try:
        direction, dtype, fields = key.split("/", 2)
        geom = dict(field.split("=", 1) for field in fields.split(","))
        driver = _MIOPEN_DRIVER[dtype]
        forw = _MIOPEN_FORW[direction]
        flags = [f"{flag} {geom[name]}" for flag, name in _MIOPEN_2D_FLAGS]
        if geom.get("is_3d") == "True":
            flags += ["--spatial_dim 3"]
            flags += [f"{flag} {geom[name]}" for flag, name in _MIOPEN_3D_FLAGS]
    except (KeyError, ValueError):
        return key
    return f"MIOpenDriver {driver} {' '.join(flags)} -F {forw} -t 1 -in_layout=NHWC"


def miopen_cmds_by_key(
    *, miopen_file=None, miopen_cmd: Optional[str] = None
) -> Dict[str, str]:
    """Map each case key back to the MIOpenDriver line it was read from.

    The sweeps key their rows by geometry, not by input line, so the line is
    not in them; re-parsing the input the run was given is what puts it back.
    One line can carry several cases (``-F 0`` is three directions), and each
    of them maps to that same line.

    Best effort by design: a line this does not parse is simply absent from the
    map, and its row falls back to :func:`miopen_cmd_from_key`. Writing the
    results of a finished sweep must not fail over the label on a row.
    """
    try:
        from benchmarks.common.benchmark_implicit_gemm_conv import (
            _FORW_TO_DIRECTIONS,
            parse_miopen_cmd,
        )
        from benchmarks.common.early_stop import case_key
    except ImportError:
        return {}

    lines: List[str] = []
    if miopen_file:
        path = Path(miopen_file)
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    elif miopen_cmd:
        lines = [miopen_cmd]

    out: Dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            problem, dtype, forw = parse_miopen_cmd(line)
            for direction in _FORW_TO_DIRECTIONS.get(forw, ["fwd"]):
                # setdefault: two lines describing one case are one case, and
                # it is the first that the sweeps measured.
                out.setdefault(case_key(problem, dtype, direction), line)
        except (ValueError, KeyError, SystemExit):
            # SystemExit among them: the MIOpen parsers hand the flags to an
            # argparse of their own, which exits rather than raises on a
            # malformed value -- and a line nobody can read is a line to pass
            # over, not the end of writing the results.
            continue
    return out


def merge(
    rows: Iterable[dict],
    miopen_cmds: Optional[Dict[str, str]] = None,
    timing: Optional[str] = None,
) -> List[dict]:
    """Join per-engine rows into one row per case.

    Rows may come from every engine and from several shards. A key only one
    engine reached still gets a row with the other sides left ``None`` -- that
    is the common and interesting case (a shape the direct kernels do not
    cover), not an error. Within one engine and key, the faster row wins, so
    merging a case measured twice is idempotent.

    ``miopen_cmds`` is :func:`miopen_cmds_by_key` of the input the run was
    given, when it was given one; a key it does not cover is labelled with
    :func:`miopen_cmd_from_key`, so every row says what to re-run.

    A row's runner-up names travel with it: a slower row is dropped whole
    rather than having its ranking blended into the faster one's. A ranking
    only means something within the run that measured it -- two runs of one
    case can order the same near-tied kernels differently, and a mix of the
    two would be a ranking that never happened.

    ``timing`` keeps only the rows measured that way (:func:`row_timing`).
    Without it the rows must all have been measured one way, and a mix raises
    ``ValueError``: the faster-row rule would otherwise pick whichever protocol
    is cheaper rather than whichever kernel is faster.

    Sorted by key, so two runs of the same case list produce comparable files.
    """
    rows = list(rows)
    if timing is None:
        seen = {row_timing(row) for row in rows if row.get("engine") in ENGINES}
        if len(seen) > 1:
            raise ValueError(
                f"rows measured with different timings {sorted(seen)}; "
                f"merge one protocol at a time"
            )
        timing = seen.pop() if seen else BLOCK_TIMING
    by_key: Dict[str, dict] = {}
    for row in rows:
        key = row.get("key")
        engine = row.get("engine")
        if (
            not key
            or engine not in ENGINES
            or row_timing(row) != timing
            or row.get("passed") is False
        ):
            continue
        merged = by_key.setdefault(
            key,
            {
                "key": key,
                "shape": row.get("shape"),
                "dtype": row.get("dtype"),
                "direction": row.get("direction"),
                "timing": timing,
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
            {engine: merged[f"{engine}_ms"] for engine in ENGINES}
        )
        if merged["winner"] is not None:
            merged["winner_ms"] = merged[f"{merged['winner']}_ms"]
            merged["winner_kernel"] = merged[f"{merged['winner']}_kernel"]
        # A row only one engine reached keeps both empty, as ``winner`` does:
        # one time is not a comparison, and the engine that did run it is
        # already on the row under its own name.
        merged["miopen_cmd"] = (miopen_cmds or {}).get(key) or miopen_cmd_from_key(key)
        out.append({field: merged[field] for field in MERGED_FIELDS})
    return out


def _verdict(times: Dict[str, Optional[float]]):
    """``(winner, speedup)``: which engine was fastest and by how much.

    ``times`` is each engine's time, ``None`` where it did not reach the case.
    ``speedup`` is the runner-up's time over the winner's, so it reads as "the
    winner is N times faster than the next best" whichever side won; a tie goes
    to the engine listed first in :data:`ENGINES`. Both ``None`` when only one
    engine reached the case -- a one-sided row is a fact about coverage, not a
    comparison.
    """
    timed = sorted(
        (ms, ENGINES.index(engine), engine)
        for engine, ms in times.items()
        if ms is not None
    )
    if len(timed) < 2:
        return None, None
    (best_ms, _, winner), (next_ms, _, _) = timed[:2]
    return winner, round(next_ms / best_ms, 3) if best_ms else None


def write(rows: Sequence[dict], path) -> Path:
    """Write merged ``rows`` as JSON Lines, plus a ``.csv`` beside it.

    The JSON carries every field of :data:`MERGED_FIELDS`; the CSV is the
    reading view, :data:`CSV_FIELDS` of the same rows. Nothing is lost by the
    narrower one -- the two files are written from the same list.

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
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def summary(rows: Sequence[dict]) -> str:
    """One line: how many cases, and how the engines split them."""
    both = sum(1 for r in rows if r["direct_ms"] is not None and r["implicit_ms"])
    direct_only = sum(
        1 for r in rows if r["direct_ms"] is not None and r["implicit_ms"] is None
    )
    implicit_only = sum(
        1 for r in rows if r["direct_ms"] is None and r["implicit_ms"] is not None
    )
    wins = sum(
        1
        for r in rows
        if r["direct_ms"] is not None
        and r["implicit_ms"] is not None
        and r["direct_ms"] <= r["implicit_ms"]
    )
    timing = f" ({rows[0].get('timing')} timing)" if rows else ""
    line = (
        f"{len(rows)} case(s){timing}: {both} measured by both "
        f"(direct faster on {wins}), {direct_only} direct-only, "
        f"{implicit_only} implicit-only"
    )
    winograd = sum(1 for r in rows if r.get("winograd_ms") is not None)
    if winograd:
        fastest = sum(1 for r in rows if r["winner"] == "winograd")
        line += f"; Winograd measured {winograd} (fastest on {fastest})"
    return line
