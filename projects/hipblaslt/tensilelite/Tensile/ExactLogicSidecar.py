# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Columnar xz sidecars for GridBased ``ExactLogic`` tables.

The ``ExactLogic`` table of a GridBased library logic file is the bulk of the
file. For eligible files the table is moved out of the YAML into a sidecar
``<name>.yaml.exactlogic.bin.xz`` next to it, and the YAML keeps
``ExactLogic: null`` (dict format) or ``- null`` (list format, element 7).
Readers call :func:`attachSidecar` on the raw loaded YAML to put the table back.

Sidecar payload (before xz compression, preset 9e)::

    b"TLXL"                magic
    0x02                   version
    uvarint N              number of rows
    uvarint L              byte length of the next two columns
    4 x N zigzag varints   key columns k0..k3, each delta-coded against the
                           previous row of the same column (row -1 is 0)
    N uvarints             solution index column
    N bytes                speed type column: 0 float, 1 int, 2 str, 3 null
    F x 8 bytes            float speeds (IEEE 754 binary64, little endian),
                           one for each row of type 0, in row order
    I zigzag varints       int speeds, one for each row of type 1
    S x (uvarint + bytes)  str speeds (byte length, then UTF-8), one for each
                           row of type 2

Rows are stably sorted by the raw key ``(k0, k1, k2, k3)``. A decoded row is
``[[k0, k1, k2, k3], [solutionIndex, speed]]``. The speed comes back with the
same Python type and the same value that the YAML loader gave (``0`` stays an
int, ``0.0`` a float, ``'.inf'`` a str), so tools that compare speeds
(TensileMergeLibrary, Utilities/merge.py) see the original numbers.

This module only depends on the standard library and PyYAML so it can run
without the rest of Tensile (``python ExactLogicSidecar.py split <paths>``).
"""

import argparse
import gc
import itertools
import lzma
import os
import re
import struct
import sys
from typing import Any, Iterable, List, Optional, Sequence, Tuple

SIDECAR_SUFFIX = ".exactlogic.bin.xz"
MAGIC = b"TLXL"
VERSION = 2
NUM_KEYS = 4
XZ_PRESET = 9 | lzma.PRESET_EXTREME

# Element index of the table / library type in list-format logic files.
LIST_TABLE_INDEX = 7
LIST_TYPE_INDEX = 11

_VARINT_RE = re.compile(rb"[\x80-\xff]*[\x00-\x7f]", re.DOTALL)

# Speed type tags.
_FLOAT, _INT, _STR, _NULL = 0, 1, 2, 3


class ExactLogicSidecarError(RuntimeError):
    """Raised for corrupt sidecars or an ambiguous YAML/sidecar pair."""


###############################################################################
# Codec
###############################################################################
def _putUvarint(out: bytearray, v: int) -> None:
    if v < 0:
        raise ValueError("uvarint cannot encode negative value {}".format(v))
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


def _zigzag(v: int) -> int:
    return (v << 1) if v >= 0 else ((-v << 1) - 1)


def _unzigzag(u: int) -> int:
    return (u >> 1) if not (u & 1) else -((u + 1) >> 1)


def _uvarintValue(chunk: bytes) -> int:
    v = 0
    for shift, b in enumerate(chunk):
        v |= (b & 0x7F) << (7 * shift)
    return v


def sortRows(rows: Iterable[Sequence[Any]]) -> List[Sequence[Any]]:
    """Stable sort of table rows by the raw key ``(k0, k1, k2, k3)``."""
    return sorted(rows, key=lambda r: tuple(r[0]))


def _speedTag(v: Any) -> int:
    # bool is an int subclass, but YAML true/false is not a speed.
    if isinstance(v, bool):
        raise ValueError("speed {!r} is a bool".format(v))
    if isinstance(v, float):
        return _FLOAT
    if isinstance(v, int):
        return _INT
    if isinstance(v, str):
        return _STR
    if v is None:
        return _NULL
    raise ValueError("speed {!r} is not a float, int, str or null".format(v))


def encodeTable(rows: Sequence[Sequence[Any]]) -> bytes:
    """Encode table rows ``[[k0..k3], [solutionIndex, speed]]`` into a sidecar blob.

    Rows are stably sorted by key first.
    """
    rows = sortRows(rows)
    cols = bytearray()
    for col in range(NUM_KEYS):
        prev = 0
        for r in rows:
            v = r[0][col]
            _putUvarint(cols, _zigzag(v - prev))
            prev = v
    for r in rows:
        _putUvarint(cols, r[1][0])

    speeds = [r[1][1] for r in rows]
    tags = bytes(_speedTag(v) for v in speeds)
    floats = [v for v, t in zip(speeds, tags) if t == _FLOAT]
    tail = bytearray(struct.pack("<{}d".format(len(floats)), *floats))
    for v, t in zip(speeds, tags):
        if t == _INT:
            _putUvarint(tail, _zigzag(v))
    for v, t in zip(speeds, tags):
        if t == _STR:
            b = v.encode("utf-8")
            _putUvarint(tail, len(b))
            tail += b

    out = bytearray(MAGIC)
    out.append(VERSION)
    _putUvarint(out, len(rows))
    _putUvarint(out, len(cols))
    out += cols
    out += tags
    out += tail
    return lzma.compress(bytes(out), format=lzma.FORMAT_XZ, preset=XZ_PRESET)


def _readUvarint(payload: bytes, pos: int, path: str, what: str) -> Tuple[int, int]:
    m = _VARINT_RE.match(payload, pos)
    if m is None:
        raise ExactLogicSidecarError("{}: truncated {}".format(path, what))
    return _uvarintValue(m.group()), m.end()


def _decodeSpeeds(payload: bytes, pos: int, n: int, path: str) -> List[Any]:
    tags = payload[pos:pos + n]
    if len(tags) != n:
        raise ExactLogicSidecarError("{}: truncated speed type column".format(path))
    pos += n
    nFloat = tags.count(_FLOAT)
    nInt = tags.count(_INT)
    nStr = tags.count(_STR)
    if nFloat + nInt + nStr + tags.count(_NULL) != n:
        raise ExactLogicSidecarError("{}: bad speed type in sidecar".format(path))
    if pos + 8 * nFloat > len(payload):
        raise ExactLogicSidecarError("{}: truncated float speeds".format(path))
    floats = list(struct.unpack_from("<{}d".format(nFloat), payload, pos))
    pos += 8 * nFloat
    ints = []
    for _ in range(nInt):
        u, pos = _readUvarint(payload, pos, path, "int speeds")
        ints.append(_unzigzag(u))
    strs = []
    for _ in range(nStr):
        size, pos = _readUvarint(payload, pos, path, "str speeds")
        if pos + size > len(payload):
            raise ExactLogicSidecarError("{}: truncated str speeds".format(path))
        try:
            strs.append(payload[pos:pos + size].decode("utf-8"))
        except UnicodeDecodeError as e:
            raise ExactLogicSidecarError("{}: bad str speed ({})".format(path, e))
        pos += size
    if pos != len(payload):
        raise ExactLogicSidecarError("{}: trailing bytes after speeds".format(path))
    if nFloat == n:
        return floats
    sources = (iter(floats), iter(ints), iter(strs), itertools.repeat(None))
    return [next(sources[t]) for t in tags]


def decodeTable(blob: bytes, path: str = "<sidecar>") -> List[List[List[Any]]]:
    """Decode a sidecar blob into ``[[[k0, k1, k2, k3], [solutionIndex, speed]], ...]``."""
    try:
        payload = lzma.decompress(blob, format=lzma.FORMAT_XZ)
    except lzma.LZMAError as e:
        raise ExactLogicSidecarError("{}: not a valid xz stream ({})".format(path, e))
    if payload[:4] != MAGIC:
        raise ExactLogicSidecarError("{}: bad magic {!r}".format(path, payload[:4]))
    if len(payload) < 7 or payload[4] != VERSION:
        raise ExactLogicSidecarError(
            "{}: unsupported version {}".format(path, payload[4] if len(payload) > 4 else None))

    n, pos = _readUvarint(payload, 5, path, "row count")
    size, pos = _readUvarint(payload, pos, path, "column length")
    body = payload[pos:pos + size]
    if len(body) != size:
        raise ExactLogicSidecarError("{}: truncated varint data".format(path))
    if body and body[-1] & 0x80:
        raise ExactLogicSidecarError("{}: truncated varint data".format(path))

    # Split into varints in C; decode each distinct byte string once.
    chunks = _VARINT_RE.findall(body)
    if len(chunks) != (NUM_KEYS + 1) * n:
        raise ExactLogicSidecarError(
            "{}: expected {} varints for {} rows, found {}".format(
                path, (NUM_KEYS + 1) * n, n, len(chunks)))
    speeds = _decodeSpeeds(payload, pos + size, n, path)
    if n == 0:
        return []

    keyChunks = chunks[:NUM_KEYS * n]
    idxChunks = chunks[NUM_KEYS * n:]
    zz = {c: _unzigzag(_uvarintValue(c)) for c in set(keyChunks)}
    uv = {c: _uvarintValue(c) for c in set(idxChunks)}

    cols = []
    for col in range(NUM_KEYS):
        deltas = map(zz.__getitem__, keyChunks[col * n:(col + 1) * n])
        cols.append(itertools.accumulate(deltas))
    idx = map(uv.__getitem__, idxChunks)
    # Building ~1e6 small lists repeatedly triggers the cyclic GC, which
    # otherwise dominates decode time; none of these objects form cycles.
    gcWasEnabled = gc.isenabled()
    gc.disable()
    try:
        return [[[a, b, c, d], [s, sp]] for a, b, c, d, s, sp in zip(*cols, idx, speeds)]
    finally:
        if gcWasEnabled:
            gc.enable()


def speedKey(v: Any) -> Tuple[str, Any]:
    """Exact identity of a speed value: type, and the bits of a float (so NaN == NaN)."""
    if isinstance(v, float):
        return "float", struct.pack("<d", v)
    return type(v).__name__, v


def rowsKey(rows: Iterable[Sequence[Any]]) -> List[Tuple[Any, ...]]:
    """Comparable form of table rows that tells ``0`` from ``0.0`` and keeps NaN equal."""
    return [(tuple(r[0]), r[1][0], speedKey(r[1][1])) for r in rows]


def _atomicWrite(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def writeSidecar(path: str, rows: Sequence[Sequence[Any]]) -> None:
    _atomicWrite(path, encodeTable(rows))


def readSidecar(path: str) -> List[List[List[Any]]]:
    with open(path, "rb") as f:
        return decodeTable(f.read(), path)


###############################################################################
# Logic-file integration
###############################################################################
def sidecarPath(yamlPath: str) -> str:
    """Return the sidecar path for a logic YAML (whether or not it exists)."""
    return str(yamlPath) + SIDECAR_SUFFIX


def findSidecar(yamlPath: Optional[str]) -> Optional[str]:
    """Return the sidecar path for *yamlPath* if the sidecar exists, else None."""
    if not yamlPath:
        return None
    p = sidecarPath(yamlPath)
    return p if os.path.isfile(p) else None


def _tableSlot(raw: Any) -> Tuple[Any, Any, Any]:
    """Return ``(container, key, libraryType)`` for the ExactLogic slot of raw logic data."""
    if isinstance(raw, dict):
        return raw, "ExactLogic", raw.get("LibraryType")
    if isinstance(raw, list) and len(raw) > LIST_TABLE_INDEX:
        libType = raw[LIST_TYPE_INDEX] if len(raw) > LIST_TYPE_INDEX else None
        return raw, LIST_TABLE_INDEX, libType
    return None, None, None


def attachSidecar(raw: Any, yamlPath: Optional[str]) -> Any:
    """Fill the ExactLogic table of raw (unparsed) logic data from its sidecar.

    *raw* is the object loaded from the YAML (list or dict format); it is
    mutated in place and returned. When no sidecar exists, *raw* is returned
    unchanged. A sidecar next to a YAML whose table is not null is an error,
    as is a sidecar next to a non-GridBased file.
    """
    sc = findSidecar(yamlPath)
    if sc is None:
        return raw
    container, key, libType = _tableSlot(raw)
    if container is None:
        raise ExactLogicSidecarError(
            "{}: sidecar {} exists but the YAML is not a library logic file".format(yamlPath, sc))
    if container[key] is not None:
        raise ExactLogicSidecarError(
            "{}: ExactLogic table is present in the YAML and in sidecar {}; "
            "remove one of them (re-run `ExactLogicSidecar.py split` after editing "
            "the table)".format(yamlPath, sc))
    if libType != "GridBased":
        raise ExactLogicSidecarError(
            "{}: sidecar {} exists but LibraryType is {!r}, not GridBased".format(yamlPath, sc, libType))
    container[key] = readSidecar(sc)
    return raw


def _isInt(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def isEligible(raw: Any) -> Tuple[bool, str]:
    """Return ``(eligible, reason)`` for moving the table of *raw* into a sidecar."""
    container, key, libType = _tableSlot(raw)
    if container is None:
        return False, "not a library logic file"
    if libType != "GridBased":
        return False, "LibraryType {!r}".format(libType)
    table = container[key]
    if not isinstance(table, list) or not table:
        return False, "empty table"
    for row in table:
        if not (isinstance(row, list) and len(row) == 2):
            return False, "malformed row"
        k, v = row
        if not (isinstance(k, list) and len(k) == NUM_KEYS and all(_isInt(x) for x in k)):
            return False, "key is not {} ints".format(NUM_KEYS)
        # Speed is stored with its type; some logic files carry YAML-quoted strings
        # such as '.inf' or '.nan'.
        if not (isinstance(v, list) and len(v) == 2 and _isInt(v[0]) and v[0] >= 0
                and (v[1] is None or (isinstance(v[1], (int, float, str))
                                      and not isinstance(v[1], bool)))):
            return False, "value is not [index, scalar speed]"
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


def splitFile(yamlPath: str, dryRun: bool = False) -> Tuple[str, str]:
    """Move the table of one logic YAML into a sidecar.

    Returns ``(status, detail)`` with status one of ``converted``, ``skipped``,
    ``already``. Raises :class:`ExactLogicSidecarError` on any inconsistency.
    """
    with open(yamlPath, "rb") as f:
        text = f.read()
    raw = _loadYaml(text)
    sc = sidecarPath(yamlPath)
    container, key, _ = _tableSlot(raw)
    if os.path.exists(sc):
        if container is not None and container[key] is None:
            return "already", sc
        raise ExactLogicSidecarError("{}: sidecar exists and table is not null".format(yamlPath))
    ok, reason = isEligible(raw)
    if not ok:
        return "skipped", reason

    isDict = isinstance(raw, dict)
    table = container[key]
    newText = stripTableText(text, isDict)

    # Verify: only the table changed, and the sidecar round-trips.
    newRaw = _loadYaml(newText)
    newContainer, _, _ = _tableSlot(newRaw)
    if newContainer is None or newContainer[key] is not None:
        raise ExactLogicSidecarError("{}: surgery did not null the table".format(yamlPath))
    container[key] = None
    if newRaw != raw:
        raise ExactLogicSidecarError("{}: surgery changed non-table content".format(yamlPath))
    blob = encodeTable(table)
    if rowsKey(decodeTable(blob, sc)) != rowsKey(sortRows(table)):
        raise ExactLogicSidecarError("{}: sidecar round-trip mismatch".format(yamlPath))

    if not dryRun:
        _atomicWrite(sc, blob)
        _atomicWrite(yamlPath, newText)
    return "converted", "{} rows, {} -> {} + {} bytes".format(
        len(table), len(text), len(newText), len(blob))


def verifyFile(yamlPath: str) -> Tuple[str, str]:
    """Check a YAML/sidecar pair: sidecar decodes and the YAML table is null."""
    sc = findSidecar(yamlPath)
    if sc is None:
        return "nosidecar", ""
    with open(yamlPath, "rb") as f:
        raw = _loadYaml(f.read())
    attachSidecar(raw, yamlPath)
    container, key, _ = _tableSlot(raw)
    return "ok", "{} rows".format(len(container[key]))


###############################################################################
# CLI
###############################################################################
def _dumpSpeed(v: Any) -> str:
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return "" if v is None else repr(v)


def _collect(paths: Iterable[str], suffix: str) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files if f.endswith(suffix))
        elif p.endswith(suffix):
            out.append(p)
    return sorted(out)


def _splitWorker(args):
    path, dryRun = args
    try:
        return (path,) + splitFile(path, dryRun)
    except Exception as e:  # report every failing file, not just the first
        return path, "error", str(e)


def _verifyWorker(path):
    try:
        return (path,) + verifyFile(path)
    except Exception as e:
        return path, "error", str(e)


def _runParallel(fn, items, jobs):
    if jobs <= 1:
        return [fn(i) for i in items]
    import multiprocessing
    with multiprocessing.Pool(jobs) as pool:
        return list(pool.imap_unordered(fn, items, chunksize=1))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="move eligible tables of logic YAMLs into sidecars")
    sp.add_argument("paths", nargs="+", help="logic YAML files or directories")
    sp.add_argument("-j", "--jobs", type=int, default=1)
    sp.add_argument("-n", "--dry-run", action="store_true")
    sp.add_argument("-v", "--verbose", action="store_true")
    dp = sub.add_parser("dump", help="print a sidecar as CSV")
    dp.add_argument("sidecar")
    vp = sub.add_parser("verify", help="check YAML/sidecar pairs")
    vp.add_argument("paths", nargs="+", help="logic YAML files or directories")
    vp.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    if args.cmd == "dump":
        out = sys.stdout
        out.write("k0,k1,k2,k3,solutionIdx,speed\n")
        for k, v in readSidecar(args.sidecar):
            out.write("{},{},{},{},{},{}\n".format(k[0], k[1], k[2], k[3], v[0], _dumpSpeed(v[1])))
        return 0

    if args.cmd == "split":
        files = _collect(args.paths, ".yaml")
        results = _runParallel(_splitWorker, [(f, args.dry_run) for f in files], args.jobs)
    else:
        files = _collect(args.paths, ".yaml")
        # Orphan sidecars (no YAML next to them) are errors too.
        sidecars = _collect(args.paths, SIDECAR_SUFFIX)
        results = _runParallel(_verifyWorker, files, args.jobs)
        results += [(s, "error", "orphan sidecar (no YAML)") for s in sidecars
                    if not os.path.isfile(s[:-len(SIDECAR_SUFFIX)])]

    counts = {}
    for path, status, detail in sorted(results):
        counts[status] = counts.get(status, 0) + 1
        if status == "error" or getattr(args, "verbose", False):
            print("{}: {}: {}".format(status, path, detail), file=sys.stderr)
    print(" ".join("{}={}".format(k, v) for k, v in sorted(counts.items())))
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
