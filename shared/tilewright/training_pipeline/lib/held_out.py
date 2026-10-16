# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Held-out GEMM lists: sets of (M, N, K, batch) that no training round may
draw or re-use, whichever tool wrote them.

A list is one of

  - a directory holding `stage01/shapes.yaml` (an audit set, a fresh audit
    or any other stage01 output's parent directory);
  - a bench yaml: `- {..., M: m, N: n, K: k, batch_count: b, ...}` entries,
    batch_count 1 when absent, as in the bench (`lib.bench_yaml`);
  - plain rows `- [m, n, k, b]`;
  - a CSV (`.csv`) with `m`, `n` and `k` columns and `batch` or
    `batch_count`.

A GEMM is held out by its exact (M, N, K, batch); a GEMM near one is not.
A list that is an external tuning workflow's evaluation set is refused
(lib/external_sets.py `ExternalSet`).
Reading is strict, since a list read short holds out less than it names: an
entry line (one starting with `-`, other than a yaml document marker) that
names no GEMM in either form, a CSV row without the columns or with a value
that is not a positive integer, and a list without any GEMM are errors
(ValueError). Lines that are not entries, such as comments and the keys of
a yaml mapping, are skipped.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from . import external_sets as xs
from .bench_yaml import shape_of_line

ShapeKey = Tuple[int, int, int, int]
# The list of a directory entry, below the directory.
DIR_LIST = Path("stage01") / "shapes.yaml"
FORMATS = (
    "a directory holding stage01/shapes.yaml, a bench yaml, '- [M, N, K, B]' "
    "rows, or a CSV with m, n, k and batch (or batch_count) columns"
)
_ROW = re.compile(r"-\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*(#.*)?")
_BATCH_COLUMNS = ("batch_count", "batch")


@dataclass(frozen=True)
class HeldOutList:
    """A list as read: the entry that named it, the file read, that file's
    sha256 and the GEMMs it holds out."""

    entry: str
    file: Path
    sha256: str
    gemms: FrozenSet[ShapeKey]

    @property
    def gemms_sha256(self) -> str:
        return gemms_digest(self.gemms)

    def record(self, **extra: Any) -> Dict[str, Any]:
        """What a stage records of the list: the entry, the file read with
        its sha256, the number of GEMMs and their digest (`gemms_digest`),
        and `extra`."""
        return {
            "entry": self.entry,
            "file": str(self.file),
            "sha256": self.sha256,
            "n_gemms": len(self.gemms),
            "gemms_sha256": self.gemms_sha256,
            **extra,
        }


def gemms_digest(gemms: Iterable[Sequence[int]]) -> str:
    """sha256 of the distinct GEMMs, sorted, one `M N K batch` line each:
    lists that hold out the same GEMMs have one digest, whatever their
    format."""
    keys = sorted({tuple(int(v) for v in g) for g in gemms})
    text = "".join(" ".join(str(v) for v in g) + "\n" for g in keys)
    return hashlib.sha256(text.encode()).hexdigest()


def list_file(entry: Any) -> Path:
    """The file an entry names: the entry itself, or a directory's
    `stage01/shapes.yaml`. ValueError when there is none."""
    p = Path(str(entry))
    if p.is_dir():
        f = p / DIR_LIST
        if not f.is_file():
            raise ValueError(f"{entry} is a directory without {DIR_LIST}")
        return f
    if not p.is_file():
        raise ValueError(f"{entry} does not exist")
    return p


def _positive(value: str, where: str) -> int:
    try:
        v = int(str(value).strip())
    except ValueError:
        raise ValueError(f"{where}: {value!r} is not an integer") from None
    if v < 1:
        raise ValueError(f"{where}: {value!r} is not a positive integer")
    return v


def _yaml_gemms(path: Path, text: str) -> List[ShapeKey]:
    out: List[ShapeKey] = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s.startswith("-") or s.split()[0] == "---":
            continue
        where = f"{path}:{i}"
        values: Optional[Sequence[Any]]
        row = _ROW.fullmatch(s)
        values = row.groups()[:4] if row is not None else shape_of_line(s)
        if values is None:
            raise ValueError(
                f"{where}: the entry {s[:80]!r} names no GEMM (a bench entry "
                f"with M, N and K, or a [M, N, K, B] row)"
            )
        m, n, k, b = (_positive(v, where) for v in values)
        out.append((m, n, k, b))
    return out


def _csv_gemms(path: Path, text: str) -> List[ShapeKey]:
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    xs.check_csv(reader.fieldnames, rows, path)
    names = {str(c).strip().lower(): c for c in reader.fieldnames or []}
    batch = [c for c in _BATCH_COLUMNS if c in names]
    missing = [c for c in ("m", "n", "k") if c not in names]
    if missing or not batch:
        raise ValueError(
            f"{path}: a CSV list needs the columns m, n, k and batch (or "
            f"batch_count); it has {sorted(names)}"
        )
    out: List[ShapeKey] = []
    for i, row in enumerate(rows, 2):
        where = f"{path}:{i}"
        dims = [_positive(row.get(names[c]) or "", where) for c in ("m", "n", "k")]
        counts = {_positive(row.get(names[c]) or "", where) for c in batch}
        if len(counts) > 1:
            raise ValueError(f"{where}: batch and batch_count differ")
        out.append((dims[0], dims[1], dims[2], counts.pop()))
    return out


def read(entry: Any) -> HeldOutList:
    """The list `entry` names (a path, see the module docstring).
    ValueError, naming the entry and the line, when it cannot be read
    whole or holds no GEMM."""
    f = list_file(entry)
    try:
        data = f.read_bytes()
    except OSError as e:
        raise ValueError(f"cannot read {f}: {e}") from None
    text = data.decode("utf-8-sig", errors="replace")
    gemms = _csv_gemms(f, text) if f.suffix.lower() == ".csv" else _yaml_gemms(f, text)
    if not gemms:
        raise ValueError(f"{entry} holds no GEMM; a held-out list is {FORMATS}")
    xs.check(gemms, entry)
    return HeldOutList(
        entry=str(entry),
        file=f.resolve(),
        sha256=hashlib.sha256(data).hexdigest(),
        gemms=frozenset(gemms),
    )


def read_all(entries: Iterable[Any]) -> Tuple[List[HeldOutList], List[str]]:
    """(the lists `entries` name, one error per entry that cannot be read)."""
    lists: List[HeldOutList] = []
    errors: List[str] = []
    for e in entries:
        try:
            lists.append(read(e))
        except ValueError as err:
            errors.append(str(err))
    return lists, errors


def digests(entries: Iterable[Any]) -> Dict[str, str]:
    """{entry: `gemms_digest` of its list}, or the reason it cannot be read:
    what a run pins of its held-out lists, whose GEMMs its rounds may not
    draw."""
    out: Dict[str, str] = {}
    for e in entries:
        try:
            out[str(e)] = read(e).gemms_sha256
        except ValueError as err:
            out[str(e)] = f"unreadable ({err})"
    return out
