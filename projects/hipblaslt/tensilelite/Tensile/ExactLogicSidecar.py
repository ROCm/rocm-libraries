# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compressed sidecar storage for GridBased ``ExactLogic`` tables.

A GridBased library logic file ``<name>.yaml`` may keep its ``ExactLogic`` table
in ``<name>.yaml.csv.gz`` instead of inline. The YAML then carries
``ExactLogic: null`` (dict format) or ``- null`` as list element 7 (list format);
everything else in it is unchanged. :func:`attachSidecar` puts the table back
into freshly loaded YAML data, so every reader sees the table it would have
seen inline.

The sidecar is a gzip-compressed (level 9) CSV written with :mod:`csv` default
dialect (``\\r\\n`` line endings), the encoding of the MeshBased tables of
PR 12199 plus a sixth column for the speed::

    M,N,batch,K,solutionIdx,speed
    k0,k1,k2,k3,idx,speed
    ...

Rows keep the order of the original YAML table. The ``speed`` field is the
speed value of the YAML row ``[idx, speed]`` written as text in one of three
forms, each with one meaning that PyYAML and :func:`parseSpeed` agree on:

* int: ``0``, ``-3``, ``10331`` (decimal, no leading zeros) -> ``int``
* float: ``7.1``, ``0.0``, ``6.57484e-05`` (shortest ``repr``, with a ``.``
  and an optional signed exponent) -> ``float``
* str: ``'.inf'`` (single-quoted, printable ASCII without ``'`` or ``,``)
  -> ``str``

A table with a speed that has no such form (null, bool, inf, nan, other
strings) stays inline. Decoded rows are ``[[k0, k1, k2, k3], [solutionIdx,
speed]]`` with the same speed type and value as the inline table, so tools
that compare speeds (TensileMergeLibrary, Utilities/merge.py) see the
original numbers.

This module uses only the standard library plus PyYAML (imported lazily, for
the eligibility check in ``split``) so it can run as a script without the rest
of Tensile::

    python ExactLogicSidecar.py split <files|dirs>...
    python ExactLogicSidecar.py dump <sidecar>
    python ExactLogicSidecar.py verify [--against REV] <files|dirs>...
"""

import argparse
import csv
import gzip
import io
import os
import re
import struct
import subprocess
import sys
import zlib
from typing import Any, Iterable, List, Optional, Sequence, Tuple

SIDECAR_SUFFIX = ".csv.gz"
CSV_HEADER = ["M", "N", "batch", "K", "solutionIdx", "speed"]
GZIP_LEVEL = 9
NUM_KEYS = 4

# Element index of the table / library type in list-format logic files.
LIST_TABLE_INDEX = 7
LIST_TYPE_INDEX = 11

Row = Tuple[int, int, int, int, int, str]

_SPEED_INT = r"-?(?:0|[1-9][0-9]*)"
_SPEED_FLOAT = r"-?[0-9]+\.[0-9]*(?:[eE][-+][0-9]+)?"
_SPEED_STR = r"'[\x20-\x26\x28-\x2b\x2d-\x7e]*'"
_SPEED_RE = re.compile("({})|({})|({})".format(_SPEED_INT, _SPEED_FLOAT, _SPEED_STR))


class ExactLogicSidecarError(RuntimeError):
    """A sidecar is corrupt, or conflicts with the YAML it belongs to."""


###############################################################################
# Codec
###############################################################################
def parseSpeed(text: str) -> Any:
    """Return the value of speed *text* (see the module docstring for the forms)."""
    m = _SPEED_RE.fullmatch(text)
    if m is None:
        raise ValueError(f"bad speed text {text!r}")
    if m.group(1):
        return int(text)
    if m.group(2):
        return float(text)
    return text[1:-1]


def formatSpeed(value: Any) -> str:
    """Return speed text that :func:`parseSpeed` and PyYAML both read back as *value*."""
    if isinstance(value, bool):
        raise ValueError(f"speed {value!r} is a bool")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if "e" in text and "." not in text:
            text = text.replace("e", ".0e")  # PyYAML reads "1e-05" as a str
    elif isinstance(value, str):
        text = "'" + value + "'"
    else:
        raise ValueError(f"speed {value!r} is not an int, float or str")
    if _SPEED_RE.fullmatch(text) is None:
        raise ValueError(f"speed {value!r} has no sidecar text form")
    return text


def speedKey(v: Any) -> Tuple[str, Any]:
    """Exact identity of a speed value: type, and the bits of a float (so NaN == NaN)."""
    if isinstance(v, float):
        return "float", struct.pack("<d", v)
    return type(v).__name__, v


def rowsKey(table: Iterable[Any]) -> List[Tuple[Any, ...]]:
    """Comparable form of ExactLogic rows that tells ``0`` from ``0.0``."""
    return [(tuple(k), v[0], speedKey(v[1])) for k, v in table]


def _checkRows(rows: Iterable[Sequence[Any]]) -> List[Row]:
    out = []
    for r in rows:
        if len(r) != NUM_KEYS + 2:
            raise ValueError(f"expected {NUM_KEYS + 2} values per row, got {r!r}")
        speed = r[5]
        if not isinstance(speed, str) or _SPEED_RE.fullmatch(speed) is None:
            raise ValueError(f"bad speed text in row {r!r}")
        out.append((int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), speed))
    return out


def encodeRows(rows: Iterable[Sequence[Any]]) -> bytes:
    """Encode ``(k0, k1, k2, k3, idx, speedText)`` rows, in the given order, into sidecar bytes.

    The gzip header carries no file name and mtime 0, so equal tables give
    equal bytes.
    """
    rows = _checkRows(rows)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADER)
    writer.writerows(rows)
    return gzip.compress(buf.getvalue().encode("ascii"), compresslevel=GZIP_LEVEL, mtime=0)


def tableRows(table: Iterable[Any]) -> List[Row]:
    """Convert ExactLogic rows ``[[k0,k1,k2,k3],[idx,speed]]`` to sidecar rows."""
    rows = []
    for key, value in table:
        if len(key) != NUM_KEYS:
            raise ValueError(f"expected a {NUM_KEYS}-element key, got {key!r}")
        rows.append((key[0], key[1], key[2], key[3], value[0], formatSpeed(value[1])))
    return rows


def encodeTable(table: Iterable[Any]) -> bytes:
    """Encode an ExactLogic table (``[[k0,k1,k2,k3],[idx,speed]]`` rows)."""
    return encodeRows(tableRows(table))


def decodeText(data: bytes, source: str = "<bytes>") -> str:
    try:
        return gzip.decompress(data).decode("ascii")
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as e:
        raise ExactLogicSidecarError(f"{source}: cannot decompress sidecar: {e}") from e


def decodeRows(data: bytes, source: str = "<bytes>") -> List[Row]:
    """Decode sidecar bytes into ``(k0, k1, k2, k3, idx, speedText)`` tuples."""
    reader = csv.reader(io.StringIO(decodeText(data, source), newline=""))
    header = next(reader, None)
    if header != CSV_HEADER:
        raise ExactLogicSidecarError(f"{source}: bad sidecar header {header!r}")
    rows = []
    known = set()
    for row in reader:
        try:
            a, b, c, d, e, f = row
            if f not in known:
                parseSpeed(f)
                known.add(f)
            rows.append((int(a), int(b), int(c), int(d), int(e), f))
        except ValueError as err:
            raise ExactLogicSidecarError(
                f"{source}: bad sidecar row {reader.line_num}: {row!r} ({err})") from err
    return rows


def decodeTable(data: bytes, source: str = "<bytes>") -> List[list]:
    """Decode sidecar bytes into ExactLogic rows ``[[k0,k1,k2,k3],[idx,speed]]``."""
    rows = decodeRows(data, source)
    # Speeds repeat a lot; parse each distinct text once.
    values = {t: parseSpeed(t) for t in {r[5] for r in rows}}
    return [[[a, b, c, d], [e, values[f]]] for a, b, c, d, e, f in rows]


def readSidecar(path: str) -> List[list]:
    with open(path, "rb") as f:
        return decodeTable(f.read(), path)


###############################################################################
# Lookup / attach
###############################################################################
def sidecarPath(yamlPath) -> str:
    return os.fspath(yamlPath) + SIDECAR_SUFFIX


def findSidecar(yamlPath) -> Optional[str]:
    p = sidecarPath(yamlPath)
    return p if os.path.isfile(p) else None


def attachSidecar(raw: Any, yamlPath) -> Any:
    """Fill the ExactLogic table of freshly loaded logic data from its sidecar.

    ``raw`` is the unprocessed YAML data (list or dict format) of ``yamlPath``.
    No sidecar: ``raw`` is returned untouched. Sidecar present: the inline
    table must be null, else :class:`ExactLogicSidecarError` (the two would be
    ambiguous). Mutates and returns ``raw``.
    """
    sc = findSidecar(yamlPath)
    if sc is None:
        return raw
    if isinstance(raw, list):
        if len(raw) <= LIST_TABLE_INDEX:
            raise ExactLogicSidecarError(f"{yamlPath}: has sidecar {sc} but is not a library logic list")
        if raw[LIST_TABLE_INDEX] is not None:
            raise ExactLogicSidecarError(
                f"{yamlPath}: ExactLogic is inline and a sidecar {sc} also exists; "
                "remove one (re-run 'ExactLogicSidecar.py split' to regenerate the sidecar)")
        raw[LIST_TABLE_INDEX] = readSidecar(sc)
    elif isinstance(raw, dict):
        if raw.get("ExactLogic") is not None:
            raise ExactLogicSidecarError(
                f"{yamlPath}: ExactLogic is inline and a sidecar {sc} also exists; "
                "remove one (re-run 'ExactLogicSidecar.py split' to regenerate the sidecar)")
        raw["ExactLogic"] = readSidecar(sc)
    else:
        raise ExactLogicSidecarError(f"{yamlPath}: has sidecar {sc} but is not library logic data")
    return raw


###############################################################################
# Eligibility and YAML text surgery
###############################################################################
def _tableSlot(raw: Any) -> Tuple[Any, Any, Any]:
    """Return ``(container, key, libraryType)`` for the ExactLogic slot of raw logic data."""
    if isinstance(raw, dict):
        return raw, "ExactLogic", raw.get("LibraryType")
    if isinstance(raw, list) and len(raw) > LIST_TABLE_INDEX:
        libType = raw[LIST_TYPE_INDEX] if len(raw) > LIST_TYPE_INDEX else None
        return raw, LIST_TABLE_INDEX, libType
    return None, None, None


def _isInt(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def isEligible(raw: Any) -> Tuple[bool, str]:
    """Return ``(eligible, reason)`` for moving the table of *raw* into a sidecar.

    Eligible: GridBased, non-empty table, every key 4 ints, every value
    ``[non-negative int, speed]`` where the speed has a sidecar text form
    (:func:`formatSpeed`).
    """
    container, key, libType = _tableSlot(raw)
    if container is None:
        return False, "not a library logic file"
    if libType != "GridBased":
        return False, "LibraryType {!r}".format(libType)
    table = container.get(key) if isinstance(container, dict) else container[key]
    if not isinstance(table, list) or not table:
        return False, "empty table"
    for row in table:
        if not (isinstance(row, list) and len(row) == 2):
            return False, "malformed row"
        k, v = row
        if not (isinstance(k, list) and len(k) == NUM_KEYS and all(_isInt(x) for x in k)):
            return False, "key is not {} ints".format(NUM_KEYS)
        if not (isinstance(v, list) and len(v) == 2 and _isInt(v[0]) and v[0] >= 0):
            return False, "value is not [index, speed]"
        try:
            formatSpeed(v[1])
        except ValueError:
            return False, "speed {!r} has no sidecar text form".format(v[1])
    return True, ""


def _findTableSpan(lines: List[bytes], isDict: bool) -> Tuple[int, int]:
    """Return the ``[start, end)`` line span holding the table (including its key line)."""
    if isDict:
        starts = [i for i, l in enumerate(lines) if l.rstrip(b"\r\n") == b"ExactLogic:"]
        if len(starts) != 1:
            raise ExactLogicSidecarError("expected one 'ExactLogic:' line, found {}".format(len(starts)))
        i = starts[0]
        j = i + 1
        while j < len(lines) and lines[j][:1] in (b"-", b" "):
            j += 1
        return i, j
    # list format: top-level items are lines starting with '-' at column 0
    tops = [i for i, l in enumerate(lines) if l[:1] == b"-"]
    if len(tops) <= LIST_TABLE_INDEX + 1:
        raise ExactLogicSidecarError("too few top-level list items")
    return tops[LIST_TABLE_INDEX], tops[LIST_TABLE_INDEX + 1]


def stripTableText(text: bytes, isDict: bool) -> bytes:
    """Replace the table text with ``ExactLogic: null`` / ``- null``; other bytes untouched."""
    lines = text.splitlines(keepends=True)
    i, j = _findTableSpan(lines, isDict)
    eol = lines[i][len(lines[i].rstrip(b"\r\n")):] or b"\n"
    repl = (b"ExactLogic: null" if isDict else b"- null") + eol
    return b"".join(lines[:i]) + repl + b"".join(lines[j:])


def _loadYaml(text: bytes) -> Any:
    import yaml
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(text, Loader=loader)  # nosec B506


class SplitResult:
    """Outcome of :func:`splitText`. ``rows`` is None when ineligible."""

    def __init__(self, status: str, rows=None, yamlText: Optional[bytes] = None, fmt: str = "",
                 table=None):
        self.status = status
        self.rows = rows
        self.table = table
        self.yamlText = yamlText
        self.fmt = fmt


def splitText(data: bytes) -> SplitResult:
    """Split a logic YAML (bytes) into table rows and the YAML with a null table.

    Status is ``split`` on success; otherwise ``skip: <reason>`` and no rows.
    Eligibility is decided on the parsed YAML (:func:`isEligible`); the text
    surgery is checked to change nothing but the table. Rows keep file order.
    """
    raw = _loadYaml(data)
    container, key, libType = _tableSlot(raw)
    fmt = "dict" if isinstance(raw, dict) else "list"
    if container is not None and libType == "GridBased":
        if fmt == "dict" and key not in container:
            return SplitResult("skip: no ExactLogic key", fmt=fmt)
        if container[key] is None:
            return SplitResult("skip: table already null", fmt=fmt)
    ok, reason = isEligible(raw)
    if not ok:
        return SplitResult("skip: " + reason, fmt=fmt)
    table = container[key]
    newText = stripTableText(data, fmt == "dict")
    newRaw = _loadYaml(newText)
    container[key] = None
    if newRaw != raw:
        raise ExactLogicSidecarError("text surgery changed more than the ExactLogic table")
    return SplitResult("split", tableRows(table), newText, fmt, table)


def _atomicWrite(path: str, data: bytes) -> None:
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def splitFile(path: str, check: bool = True, dryRun: bool = False) -> Tuple[str, str, int, int]:
    """Move the table of ``path`` into its sidecar. Returns (path, status, rows, sidecarBytes).

    A YAML that still has its table inline is (re)split even if a sidecar
    already exists; the inline table wins and the sidecar is overwritten.
    """
    with open(path, "rb") as f:
        data = f.read()
    try:
        res = splitText(data)
    except ExactLogicSidecarError as e:
        raise ExactLogicSidecarError(f"{path}: {e}") from e
    if res.status != "split":
        return (path, res.status, 0, 0)
    blob = encodeRows(res.rows)
    if check and decodeRows(blob, path) != res.rows:
        raise ExactLogicSidecarError(f"{path}: sidecar round trip mismatch")
    # The decoded table must equal the inline table, with each speed of the
    # same type and value.
    if check and rowsKey(decodeTable(blob, path)) != rowsKey(res.table):
        raise ExactLogicSidecarError(f"{path}: sidecar table differs from the YAML table")
    if not dryRun:
        # Sidecar first: an interruption leaves "inline table + sidecar", which
        # readers reject loudly and a re-run of split repairs.
        _atomicWrite(sidecarPath(path), blob)
        _atomicWrite(path, res.yamlText)
    return (path, "split", len(res.rows), len(blob))


###############################################################################
# Verify
###############################################################################
def _gitShow(rev: str, path: str) -> bytes:
    absPath = os.path.abspath(path)
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=os.path.dirname(absPath),
                         check=True, capture_output=True, text=True).stdout.strip()
    rel = os.path.relpath(absPath, top)
    return subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=top, check=True,
                          capture_output=True).stdout


def verifyFile(path: str, against: Optional[str] = None) -> List[str]:
    """Check one converted YAML + sidecar pair. Returns a list of problems."""
    problems = []
    sc = findSidecar(path)
    if sc is None:
        return [f"{path}: no sidecar"]
    with open(path, "rb") as f:
        data = f.read()
    raw = _loadYaml(data)
    container, key, libType = _tableSlot(raw)
    if container is None or (isinstance(container, dict) and key not in container) \
            or container[key] is not None:
        problems.append(f"{path}: YAML table is not null")
    if libType != "GridBased":
        problems.append(f"{path}: LibraryType is {libType!r}, not GridBased")
    with open(sc, "rb") as f:
        try:
            rows = decodeRows(f.read(), sc)
        except ExactLogicSidecarError as e:
            return problems + [str(e)]
    if not rows:
        problems.append(f"{sc}: sidecar has no rows")
    if against:
        orig = splitText(_gitShow(against, path))
        if orig.status != "split":
            problems.append(f"{path}: original at {against} is not splittable ({orig.status})")
        else:
            if orig.yamlText != data:
                problems.append(f"{path}: YAML differs from {against} outside the table")
            if orig.rows != rows:
                problems.append(f"{sc}: rows differ from the table at {against}")
    return problems


###############################################################################
# CLI
###############################################################################
def _collect(paths: Sequence[str], suffix: str) -> List[str]:
    """Files under *paths* ending in *suffix*; explicit file arguments are filtered too."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files if f.endswith(suffix))
        elif p.endswith(suffix):
            out.append(p)
    return sorted(out)


def _pmap(fn, items, jobs):
    if jobs <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items, chunksize=1))


def _splitOne(args):
    try:
        return splitFile(*args)
    except Exception as e:  # report every failing file, not just the first
        return (args[0], "error: " + str(e), 0, 0)


def _verifyOne(args):
    try:
        return verifyFile(*args)
    except Exception as e:
        return [f"{args[0]}: {e}"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="move GridBased ExactLogic tables into sidecars")
    s.add_argument("paths", nargs="+")
    s.add_argument("-j", "--jobs", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-check", action="store_true", help="skip the sidecar round-trip check")
    s.add_argument("-v", "--verbose", action="store_true", help="print the reason for every skipped file")
    d = sub.add_parser("dump", help="print a sidecar as CSV text")
    d.add_argument("sidecar")
    v = sub.add_parser("verify", help="check YAML + sidecar pairs")
    v.add_argument("paths", nargs="+")
    v.add_argument("-j", "--jobs", type=int, default=1)
    v.add_argument("--against", metavar="REV",
                   help="also check against the unconverted file at git revision REV")
    a = ap.parse_args(argv)

    if a.cmd == "dump":
        with open(a.sidecar, "rb") as f:
            sys.stdout.write(decodeText(f.read(), a.sidecar))
        return 0

    files = _collect(a.paths, ".yaml")
    if a.cmd == "split":
        results = _pmap(_splitOne, [(f, not a.no_check, a.dry_run) for f in files], a.jobs)
        counts = {}
        nRows = nBytes = 0
        for path, status, rows, size in results:
            key = "error" if status.startswith("error") else status
            counts[key] = counts.get(key, 0) + 1
            nRows += rows
            nBytes += size
            if status == "split" or key == "error" or a.verbose:
                print(f"{status}: {path}" + (f" ({rows} rows, {size} bytes)" if rows else ""))
        for status, n in sorted(counts.items()):
            print(f"# {n:6d} {status}")
        print(f"# total rows {nRows}, sidecar bytes {nBytes}")
        return 1 if counts.get("error") else 0

    targets = [f for f in files if findSidecar(f)]
    problems = [p for ps in _pmap(_verifyOne, [(f, a.against) for f in targets], a.jobs) for p in ps]
    # Orphan sidecars (no YAML next to them) are problems too.
    problems += [f"{s}: orphan sidecar (no YAML)" for s in _collect(a.paths, SIDECAR_SUFFIX)
                 if not os.path.isfile(s[:-len(SIDECAR_SUFFIX)])]
    for p in problems:
        print(p)
    print(f"# verified {len(targets)} file(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
