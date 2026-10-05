# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compressed sidecar storage for GridBased ``ExactLogic`` tables.

A GridBased library logic file ``<name>.yaml`` may keep its ``ExactLogic`` table
in ``<name>.yaml.exactlogic.csv.xz`` instead of inline. The YAML then carries
``ExactLogic: null`` (dict format) or ``- null`` as list element 7 (list format);
everything else in it is unchanged. :func:`attachSidecar` puts the table back
into freshly loaded YAML data, so every reader sees the same rows it would have
seen with the table inline.

Sidecar payload (xz-compressed ASCII, ``\\n`` line endings)::

    M,N,batch,K,solutionIdx
    k0,k1,k2,k3,idx
    ...

Rows are stably sorted by the raw key ``(k0, k1, k2, k3)`` as stored in the
YAML. The second value of an inline row (``speed``) is not stored: for
GridBased libraries ``MatchingLibrary.FromOriginalState`` drops it. Decoded
rows are ``[[k0, k1, k2, k3], [solutionIdx, 0.0]]``.

This module uses only the standard library (plus PyYAML, imported lazily, for
the optional post-split self check) so it can run as a script without the rest
of Tensile::

    python ExactLogicSidecar.py split <files|dirs>...
    python ExactLogicSidecar.py dump <sidecar>
    python ExactLogicSidecar.py verify [--against REV] <files|dirs>...
"""

import argparse
import lzma
import os
import re
import subprocess
import sys
from typing import Any, Iterable, List, Optional, Sequence, Tuple

SIDECAR_SUFFIX = ".exactlogic.csv.xz"
CSV_HEADER = "M,N,batch,K,solutionIdx"
XZ_PRESET = 9 | lzma.PRESET_EXTREME

Row = Tuple[int, int, int, int, int]


class ExactLogicSidecarError(RuntimeError):
    """A sidecar is corrupt, or conflicts with the YAML it belongs to."""


###############################################################################
# Codec
###############################################################################
def sortRows(rows: Iterable[Sequence[int]]) -> List[Row]:
    """Return ``(k0, k1, k2, k3, idx)`` rows stably sorted by key."""
    out = [tuple(int(v) for v in r) for r in rows]
    for r in out:
        if len(r) != 5:
            raise ValueError(f"expected 5 values per row, got {r!r}")
    out.sort(key=lambda r: r[:4])
    return out


def encodeRows(rows: Iterable[Sequence[int]]) -> bytes:
    """Encode ``(k0, k1, k2, k3, idx)`` rows into sidecar bytes (sorts them)."""
    rows = sortRows(rows)
    text = CSV_HEADER + "\n" + "".join("%d,%d,%d,%d,%d\n" % r for r in rows)
    return lzma.compress(text.encode("ascii"), format=lzma.FORMAT_XZ, preset=XZ_PRESET)


def encodeTable(table: Iterable[Any]) -> bytes:
    """Encode an ExactLogic table (``[[k0,k1,k2,k3],[idx,speed]]`` rows)."""
    rows = []
    for key, value in table:
        if len(key) != 4:
            raise ValueError(f"expected a 4-element key, got {key!r}")
        rows.append((key[0], key[1], key[2], key[3], value[0]))
    return encodeRows(rows)


def decodeText(data: bytes, source: str = "<bytes>") -> str:
    try:
        return lzma.decompress(data, format=lzma.FORMAT_XZ).decode("ascii")
    except (lzma.LZMAError, UnicodeDecodeError, EOFError) as e:
        raise ExactLogicSidecarError(f"{source}: cannot decompress sidecar: {e}") from e


def decodeRows(data: bytes, source: str = "<bytes>") -> List[Row]:
    """Decode sidecar bytes into ``(k0, k1, k2, k3, idx)`` tuples."""
    text = decodeText(data, source)
    lines = text.split("\n")
    if lines[0] != CSV_HEADER:
        raise ExactLogicSidecarError(f"{source}: bad sidecar header {lines[0]!r}")
    if lines[-1] != "":
        raise ExactLogicSidecarError(f"{source}: sidecar does not end with a newline")
    rows = []
    try:
        for n, line in enumerate(lines[1:-1], start=2):
            a, b, c, d, e = line.split(",")
            rows.append((int(a), int(b), int(c), int(d), int(e)))
    except ValueError as err:
        raise ExactLogicSidecarError(f"{source}: bad sidecar row {n}: {line!r} ({err})") from err
    return rows


def decodeTable(data: bytes, source: str = "<bytes>") -> List[list]:
    """Decode sidecar bytes into ExactLogic rows ``[[k0,k1,k2,k3],[idx,0.0]]``."""
    return [[[a, b, c, d], [e, 0.0]] for a, b, c, d, e in decodeRows(data, source)]


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
        if len(raw) <= 7:
            raise ExactLogicSidecarError(f"{yamlPath}: has sidecar {sc} but is not a library logic list")
        if raw[7] is not None:
            raise ExactLogicSidecarError(
                f"{yamlPath}: ExactLogic is inline and a sidecar {sc} also exists; "
                "remove one (re-run 'ExactLogicSidecar.py split' to regenerate the sidecar)")
        raw[7] = readSidecar(sc)
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
# YAML text surgery
###############################################################################
# Non-negative decimal ints without leading zeros (YAML 1.1 would read "017" as octal).
_INT = rb"(0|[1-9][0-9]*)"
_KEY = rb"\[" + rb", ".join([_INT] * 4) + rb"\]\n"
_VAL = rb"\[" + _INT + rb", [^\]\n]+\]\n"
# dict format:  "- - [k0, k1, k2, k3]\n  - [idx, speed]\n"
_DICT_ROW = re.compile(rb"- - " + _KEY + rb"  - " + _VAL)
# list format, element 7: first row "- - - [...]\n    - [...]\n", then "  - - [...]\n    - [...]\n"
_LIST_ROW0 = re.compile(rb"- - - " + _KEY + rb"    - " + _VAL)
_LIST_ROW = re.compile(rb"  - - " + _KEY + rb"    - " + _VAL)
_DICT_TABLE_START = re.compile(rb"^ExactLogic:\n", re.M)
_DICT_NULL = re.compile(rb"^ExactLogic: null\n", re.M)
_DICT_GRID = re.compile(rb"^LibraryType: GridBased\n", re.M)
_TOP_ITEM = re.compile(rb"^-", re.M)
_PREAMBLE = re.compile(rb"(?:[ \t]*(?:#[^\n]*)?\n|---[ \t]*\n)*")


class SplitResult:
    """Outcome of :func:`splitText`. ``rows`` is None when ineligible."""

    def __init__(self, status: str, rows=None, yamlText: Optional[bytes] = None, fmt: str = ""):
        self.status = status
        self.rows = rows
        self.yamlText = yamlText
        self.fmt = fmt


def _scanRows(data: bytes, pos: int, first, rest) -> Tuple[List[Row], int]:
    rows = []
    m = first.match(data, pos)
    while m:
        rows.append(tuple(int(g) for g in m.groups()))
        pos = m.end()
        m = rest.match(data, pos)
    return rows, pos


def splitText(data: bytes) -> SplitResult:
    """Split a logic YAML (bytes) into table rows and the YAML with a null table.

    Status is ``split`` on success; otherwise a short reason and no rows.
    Only GridBased files whose table is entirely ``[int x4] -> [int, number]``
    rows in the standard block layout are split.
    """
    if b"\r" in data:
        return SplitResult("skip: CR line endings")
    body = _PREAMBLE.match(data).end()
    if data.startswith(b"- ", body):
        tops = []
        for m in _TOP_ITEM.finditer(data, body):
            tops.append(m.start())
            if len(tops) == 13:
                break
        if len(tops) < 12:
            return SplitResult("skip: list too short", fmt="list")
        lt = data[tops[11]:tops[12] if len(tops) > 12 else len(data)]
        if lt.rstrip(b"\n") != b"- GridBased":
            return SplitResult("skip: not GridBased", fmt="list")
        start, stop = tops[7], tops[8]
        if data[start:stop] == b"- null\n":
            return SplitResult("skip: table already null", fmt="list")
        rows, end = _scanRows(data, start, _LIST_ROW0, _LIST_ROW)
        if not rows:
            return SplitResult("skip: table empty or not 4-int keys", fmt="list")
        if end != stop:
            return SplitResult("skip: unrecognized table layout", fmt="list")
        return SplitResult("split", rows, data[:start] + b"- null\n" + data[stop:], "list")

    if not _DICT_GRID.search(data):
        return SplitResult("skip: not GridBased", fmt="dict")
    if _DICT_NULL.search(data):
        return SplitResult("skip: table already null", fmt="dict")
    m = _DICT_TABLE_START.search(data)
    if not m:
        return SplitResult("skip: table not a block sequence (null/empty?)", fmt="dict")
    rows, end = _scanRows(data, m.end(), _DICT_ROW, _DICT_ROW)
    if not rows:
        return SplitResult("skip: table empty or not 4-int keys", fmt="dict")
    nxt = data[end:end + 1]
    if nxt in (b"-", b" ", b"\t", b"#"):
        return SplitResult("skip: unrecognized table layout", fmt="dict")
    return SplitResult("split", rows, data[:m.start()] + b"ExactLogic: null\n" + data[end:], "dict")


def _atomicWrite(path: str, data: bytes) -> None:
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _selfCheck(path: str, res: SplitResult, sidecarBytes: bytes) -> None:
    if decodeRows(sidecarBytes, path) != sortRows(res.rows):
        raise ExactLogicSidecarError(f"{path}: sidecar round trip mismatch")
    try:
        import yaml
    except ImportError:
        return
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    data = yaml.load(res.yamlText, Loader=loader)  # nosec B506
    if isinstance(data, list):
        ok = data[7] is None and data[11] == "GridBased"
    else:
        ok = data.get("ExactLogic", 1) is None and data.get("LibraryType") == "GridBased"
    if not ok:
        raise ExactLogicSidecarError(f"{path}: stripped YAML does not parse as expected")


def splitFile(path: str, check: bool = True, dryRun: bool = False) -> Tuple[str, str, int, int]:
    """Move the table of ``path`` into its sidecar. Returns (path, status, rows, sidecarBytes).

    A YAML that still has its table inline is (re)split even if a sidecar
    already exists; the inline table wins and the sidecar is overwritten.
    """
    with open(path, "rb") as f:
        data = f.read()
    res = splitText(data)
    if res.status != "split":
        return (path, res.status, 0, 0)
    blob = encodeRows(res.rows)
    if check:
        _selfCheck(path, res, blob)
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
    res = splitText(data)
    if res.status != "skip: table already null":
        problems.append(f"{path}: YAML table is not null ({res.status})")
    with open(sc, "rb") as f:
        try:
            rows = decodeRows(f.read(), sc)
        except ExactLogicSidecarError as e:
            return problems + [str(e)]
    if rows != sortRows(rows):
        problems.append(f"{sc}: rows are not sorted by key")
    if against:
        orig = splitText(_gitShow(against, path))
        if orig.status != "split":
            problems.append(f"{path}: original at {against} is not splittable ({orig.status})")
        else:
            if orig.yamlText != data:
                problems.append(f"{path}: YAML differs from {against} outside the table")
            if sortRows(orig.rows) != rows:
                problems.append(f"{sc}: rows differ from the table at {against}")
    return problems


###############################################################################
# CLI
###############################################################################
def _expand(paths: Sequence[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files if f.endswith(".yaml"))
        else:
            out.append(p)
    return sorted(out)


def _pmap(fn, items, jobs):
    if jobs <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items, chunksize=1))


def _splitOne(args):
    return splitFile(*args)


def _verifyOne(args):
    return verifyFile(*args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="move GridBased ExactLogic tables into sidecars")
    s.add_argument("paths", nargs="+")
    s.add_argument("-j", "--jobs", type=int, default=1)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-check", action="store_true", help="skip the post-split self check")
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

    files = _expand(a.paths)
    if a.cmd == "split":
        results = _pmap(_splitOne, [(f, not a.no_check, a.dry_run) for f in files], a.jobs)
        counts = {}
        nRows = nBytes = 0
        for path, status, rows, size in results:
            counts[status] = counts.get(status, 0) + 1
            nRows += rows
            nBytes += size
            if status == "split" or a.verbose:
                print(f"{status}: {path}" + (f" ({rows} rows, {size} bytes)" if rows else ""))
        for status, n in sorted(counts.items()):
            print(f"# {n:6d} {status}")
        print(f"# total rows {nRows}, sidecar bytes {nBytes}")
        return 0

    targets = [f for f in files if findSidecar(f)]
    problems = [p for ps in _pmap(_verifyOne, [(f, a.against) for f in targets], a.jobs) for p in ps]
    for p in problems:
        print(p)
    print(f"# verified {len(targets)} file(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
