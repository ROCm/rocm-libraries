# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write an ``inline_frames.json`` sidecar into a decoded ATT trace folder.

A rocke kernel is one flat GPU function assembled by many layers of Python: an
``instances/`` builder calling ``helpers/`` emitters calling closures. When the
kernel is built with source-location capture (``ROCKE_DEBUG_LOC=1``), the
lowering records that whole authoring stack as DWARF inlining scopes, so the
code object knows the full Python call stack behind every program counter.

``rocprofv3`` flattens that to a bare ``file:line`` in ``code.json``'s Source
column -- the innermost frame only. That is why a one-line helper such as
``return b.global_load_f16(self.base, off)`` shows up owning a large share of a
kernel's stalls with no indication of which phase asked for the load.

This recovers the rest by joining the code object's ``DW_TAG_inlined_subroutine``
tree, which carries a PC range per frame, to ``code.json``'s Vaddr column, and
writes the result beside the trace. WaveScope picks the sidecar up
automatically; without it the Source tab behaves exactly as before.

Entries are keyed ``"<codeobj>:<vaddr>"``. Virtual addresses are per code object
and collide across objects, so a trace that loaded more than one needs both
columns to identify an instruction.

Re-running over a folder that already has sidecars is safe and is the expected
way to use this: every sidecar under the trace directory goes first, before this
run looks for a code object or for llvm-dwarfdump, either of which can end the
run -- so a dispatch this run does not rewrite is left with no sidecar rather
than the previous run's answer. The capture sentinel is preserved: capture owns
whether trace bytes are complete, and this producer refuses a running, truncated,
or changed trace rather than promoting it to complete.

Sidecar version 3 binds each file to the exact ``code.json`` bytes and code
object whose DWARF produced the stacks, via ``wavescope-trace.json``.

    python emit_inline_frames.py <capture-generation-dir>
    python emit_inline_frames.py <capture-generation-dir> --code-object k.hsaco
    python emit_inline_frames.py <capture-generation-dir> --invalidate-only
    python emit_inline_frames.py <direct-dispatch-dir> --code-object k.hsaco
    python emit_inline_frames.py <legacy-output-dir> --assume-complete
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "trace_provenance", _HERE / "trace_provenance.py"
)
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)

CAPTURE_COMPLETE = _tp.CAPTURE_COMPLETE
DISPATCH_GLOB = _tp.DISPATCH_GLOB
SIDECAR = _tp.SIDECAR
SIDECAR_VERSION = _tp.SIDECAR_VERSION
TMP_SUFFIX = _tp.SIDECAR_TMP_SUFFIX
TRACE_SENTINEL = _tp.TRACE_SENTINEL
dispatch_dirs = _tp.dispatch_dirs
enrich_trace_sentinel = _tp.enrich_trace_sentinel
invalidate_sidecars = _tp.invalidate_sidecars
new_trace_id = _tp.new_trace_id
read_trace_sentinel = _tp.read_trace_sentinel
sha256_file = _tp.sha256_file
write_trace_sentinel = _tp.write_trace_sentinel

# rocprofv3 dumps each loaded code object next to the raw trace.
CODE_OBJECT_GLOB = "*code_object_id_*.out"
CODE_OBJECT_ID_RE = re.compile(r"code_object_id_(\d+)")

# Virtual addresses are per code object, so an address alone does not identify an
# instruction in a trace that loaded more than one. Both columns form the join key.
CODEOBJ_COL = 4
VADDR_COL = 5

# Frames shallower than this are the enclosing GPU function itself, not a call.
_DIE_RE = re.compile(r"^(0x[0-9a-f]+):(\s+)DW_TAG_(\w+)")
_RANGE_RE = re.compile(r"^\s+\[(0x[0-9a-f]+), (0x[0-9a-f]+)\)")
_ATTR_RE = re.compile(r"^\s+DW_AT_(\w+)\s+\((.*)\)\s*$")
_QUOTED = re.compile(r'"([^"]*)"')


def find_dwarfdump() -> str:
    """Locate llvm-dwarfdump, preferring the ROCm LLVM that built the object."""
    for cand in ("/opt/rocm/llvm/bin/llvm-dwarfdump", "llvm-dwarfdump"):
        found = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if found:
            return found
    raise SystemExit(
        "llvm-dwarfdump not found. It ships with ROCm at "
        "/opt/rocm/llvm/bin/llvm-dwarfdump; install it or put it on PATH."
    )


def parse_inline_frames(code_object: Path, dwarfdump: str) -> list[dict]:
    """Return one entry per subprogram / inlined subroutine that has PC ranges.

    ``depth`` is the DIE nesting depth, so a smaller number is an outer frame.
    """
    proc = subprocess.run(
        [dwarfdump, "--debug-info", str(code_object)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"llvm-dwarfdump failed on {code_object}:\n{proc.stderr[-800:]}"
        )

    dies: list[dict] = []
    cur: dict | None = None
    for line in proc.stdout.splitlines():
        die = _DIE_RE.match(line)
        if die:
            if cur is not None:
                dies.append(cur)
            cur = {
                "depth": len(die.group(2)),
                "tag": die.group(3),
                "ranges": [],
                "name": None,
                "file": None,
                "line": 0,
                "col": 0,
            }
            continue
        if cur is None:
            continue
        rng = _RANGE_RE.match(line)
        if rng:
            cur["ranges"].append((int(rng.group(1), 16), int(rng.group(2), 16)))
            continue
        attr = _ATTR_RE.match(line)
        if not attr:
            continue
        key, val = attr.group(1), attr.group(2)
        if key in ("abstract_origin", "name") and cur["name"] is None:
            quoted = _QUOTED.search(val)
            if quoted:
                cur["name"] = quoted.group(1)
        elif key == "low_pc":
            cur["lo"] = int(val, 16)
        elif key == "high_pc":
            cur["hi"] = int(val, 16)
        elif key == "decl_file" and cur["file"] is None:
            quoted = _QUOTED.search(val)
            cur["file"] = quoted.group(1) if quoted else None
        elif key == "call_file":
            quoted = _QUOTED.search(val)
            if quoted:
                cur["call_file"] = quoted.group(1)
        elif key == "call_line":
            cur["call_line"] = int(val)
        elif key == "call_column":
            cur["call_column"] = int(val)
    if cur is not None:
        dies.append(cur)

    frames = []
    for die in dies:
        if die["tag"] not in ("subprogram", "inlined_subroutine"):
            continue
        ranges = die["ranges"]
        if not ranges and die.get("lo") is not None and die.get("hi"):
            ranges = [(die["lo"], die["hi"])]
        if not ranges or not die["name"]:
            continue
        frames.append(
            {
                "depth": die["depth"],
                "name": die["name"],
                "ranges": ranges,
                "call_file": die.get("call_file"),
                "call_line": die.get("call_line", 0),
                "call_col": die.get("call_column", 0),
            }
        )
    return frames


def stack_for(frames: list[dict], addr: int) -> list[dict]:
    """The frames covering ``addr``, outermost first."""
    hits = [f for f in frames if any(lo <= addr < hi for lo, hi in f["ranges"])]
    hits.sort(key=lambda f: f["depth"])
    return hits


def build_sidecar(
    rows: list,
    frames: list[dict],
    code_object_id: str | None,
    *,
    trace_id: str,
    code_json_hash: str,
    code_object_hash: str | None,
) -> dict:
    """Map each instruction to its authoring call stack, keyed by code object and address."""
    files: dict[str, int] = {}
    funcs: dict[str, int] = {}

    def intern(table: dict, value: str) -> int:
        if value not in table:
            table[value] = len(table)
        return table[value]

    stacks: dict[str, list] = {}
    resolved = 0
    skipped_other_object = 0
    for row in rows:
        isa = row[0] if row else ""
        if not isa or isa.startswith(";"):
            continue
        codeobj = row[CODEOBJ_COL] if len(row) > CODEOBJ_COL else None
        if code_object_id is not None and str(codeobj) != code_object_id:
            skipped_other_object += 1
            continue
        addr = row[VADDR_COL]
        stack = stack_for(frames, addr)
        if not stack:
            continue
        encoded = []
        for frame in stack:
            call_file = frame["call_file"]
            encoded.append(
                [
                    intern(funcs, frame["name"]),
                    intern(files, call_file) if call_file else -1,
                    frame["call_line"] or 0,
                    frame["call_col"] or 0,
                ]
            )
        if encoded:
            stacks[f"{codeobj}:{addr}"] = encoded
            resolved += 1

    return {
        "version": SIDECAR_VERSION,
        "schema": '"codeobj:addr" -> [[func, call_file, call_line, call_col], ...]',
        "traceId": trace_id,
        "codeJsonHash": code_json_hash,
        "codeObjectHash": code_object_hash,
        "code_object_id": code_object_id,
        "functions": list(funcs),
        "files": list(files),
        "stacks": stacks,
        "resolved": resolved,
        "skipped_other_object": skipped_other_object,
    }


@contextlib.contextmanager
def sidecar_write(d: Path) -> Iterator[Path]:
    """Yield the temporary path to write ``d``'s sidecar to."""
    out = d / SIDECAR
    tmp = out.with_name(out.name + TMP_SUFFIX)
    out.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    try:
        yield tmp
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def find_code_objects(root: Path) -> list[Path]:
    candidates = [p for p in root.rglob(CODE_OBJECT_GLOB) if p.is_file()]
    candidates += [p for p in root.rglob("*.hsaco") if p.is_file()]
    return sorted(candidates, key=lambda p: (-p.stat().st_size, p.name))


def code_object_id_of(path: Path) -> str | None:
    m = CODE_OBJECT_ID_RE.search(path.name)
    return m.group(1) if m else None


def select_code_object(
    candidates: list[Path], present: set[str], explicit: Path | None
) -> tuple[Path | None, str | None, str | None]:
    matches = [(p, cid) for p in candidates if (cid := code_object_id_of(p)) in present]
    if len(matches) == 1:
        return matches[0][0], matches[0][1], None
    if len(matches) > 1:
        names = ", ".join(sorted(p.name for p, _ in matches))
        return (
            None,
            None,
            f"ran several dumped code objects ({names}); pass --code-object to "
            "choose which one to read DWARF from",
        )
    if explicit is not None and code_object_id_of(explicit) is None:
        if len(present) == 1:
            return explicit, next(iter(present)), None
        return (
            None,
            None,
            f"{explicit.name} carries no code object id and this dispatch ran "
            f"{len(present) or 'no'} objects, so which rows it produced is unknown",
        )
    labelled = sorted(i for p in candidates if (i := code_object_id_of(p)))
    have = f"ids {', '.join(labelled)}" if labelled else "no id in its name"
    return (
        None,
        None,
        f"ran code objects {sorted(present)}, and the dumped DWARF carries "
        f"{have}; pass --code-object to point at this dispatch's",
    )


def row_code_objects(rows: list) -> set[str]:
    return {
        str(r[CODEOBJ_COL])
        for r in rows
        if r and r[0] and not r[0].startswith(";") and len(r) > CODEOBJ_COL
    }


def trace_identity_for(
    dispatch: Path,
    code_json: Path,
    *,
    assume_complete: bool,
) -> tuple[str | None, str, str | None]:
    """Validate capture provenance before a dispatch receives a sidecar."""
    code_json_hash = sha256_file(code_json)
    sentinel = read_trace_sentinel(dispatch)
    if sentinel is None:
        if not assume_complete:
            return (
                None,
                code_json_hash,
                (
                    "has no capture sentinel; pass --assume-complete only for a "
                    "known-complete legacy trace"
                ),
            )
        trace_id = new_trace_id()
        write_trace_sentinel(
            dispatch,
            trace_id=trace_id,
            code_json_hash=code_json_hash,
            capture=CAPTURE_COMPLETE,
        )
        return trace_id, code_json_hash, None

    capture = sentinel.get("capture")
    if capture != CAPTURE_COMPLETE:
        return None, code_json_hash, f"capture is {capture!r}, not complete"
    if sentinel.get("codeJsonHash") != code_json_hash:
        return None, code_json_hash, "code.json no longer matches its capture sentinel"
    trace_id = sentinel.get("traceId")
    if not isinstance(trace_id, str) or not trace_id:
        return None, code_json_hash, "capture sentinel has no valid traceId"
    return trace_id, code_json_hash, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_dir", type=Path, help="rocprofv3 --att output directory")
    ap.add_argument(
        "--code-object",
        type=Path,
        default=None,
        help="code object with DWARF; defaults to the one rocprofv3 dumped",
    )
    ap.add_argument(
        "--invalidate-only",
        action="store_true",
        help="drop existing sidecars, preserve capture sentinels, and write none",
    )
    ap.add_argument(
        "--assume-complete",
        action="store_true",
        help="allow a known-complete legacy trace with no capture sentinel",
    )
    args = ap.parse_args(argv)

    root: Path = args.trace_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    generations = sorted(p for p in root.glob("capture-*") if p.is_dir())
    if generations:
        available = "\n".join(f"  {path}" for path in generations)
        raise SystemExit(
            f"{root} is a capture output root, not one capture generation.\n"
            f"Available capture generations:\n{available}\n"
            "Run against exactly one generation, for example:\n"
            f"  python {Path(__file__).name} {generations[0]}"
        )

    dirs = dispatch_dirs(root)
    dropped = invalidate_sidecars(root)

    if args.invalidate_only:
        print(f"  {dropped} sidecar file(s) removed under {root}")
        return 0

    if not dirs:
        raise SystemExit(f"no decoded dispatch folder under {root}")

    identities: dict[Path, tuple[str, str]] = {}
    skipped = 0
    for d in dirs:
        code_json = d / "code.json"
        if not code_json.is_file():
            continue
        trace_id, code_json_hash, problem = trace_identity_for(
            d, code_json, assume_complete=args.assume_complete
        )
        if problem is not None:
            print(f"  {d.name}: skipped: {problem}")
            skipped += 1
            continue
        assert trace_id is not None
        identities[d] = (trace_id, code_json_hash)

    if not identities:
        raise SystemExit(f"no sidecar written ({skipped} dispatch(es) skipped)")

    candidates = [args.code_object] if args.code_object else find_code_objects(root)
    if not candidates:
        raise SystemExit(
            f"no code object found under {root}. rocprofv3 writes "
            f"'{CODE_OBJECT_GLOB}' beside the raw trace; pass --code-object "
            "to point at it (or at the .hsaco the kernel was built from)."
        )

    dwarfdump = find_dwarfdump()
    parsed: dict[Path, list[dict]] = {}
    written = 0
    for d, (trace_id, code_json_hash) in identities.items():
        code_json = d / "code.json"
        rows = json.loads(code_json.read_text())["code"]
        present = row_code_objects(rows)

        code_object, code_object_id, problem = select_code_object(
            candidates, present, args.code_object
        )
        if problem is not None:
            print(f"  {d.name}: skipped: {problem}")
            skipped += 1
            continue

        if code_object not in parsed:
            parsed[code_object] = parse_inline_frames(code_object, dwarfdump)
            print(
                f"  {code_object.name}: {len(parsed[code_object])} inline frames "
                "with PC ranges"
            )
        frames = parsed[code_object]
        if not frames:
            print(
                f"  {d.name}: skipped: {code_object.name} carries no inlining "
                "info. Build the kernel with ROCKE_DEBUG_LOC=1 (or "
                "IRBuilder(capture_loc=True)) so the lowering emits DWARF "
                "inlining scopes, then re-capture."
            )
            skipped += 1
            continue

        code_object_hash = sha256_file(code_object)
        sidecar = build_sidecar(
            rows,
            frames,
            code_object_id,
            trace_id=trace_id,
            code_json_hash=code_json_hash,
            code_object_hash=code_object_hash,
        )
        total = len([r for r in rows if r and r[0] and not r[0].startswith(";")])
        with sidecar_write(d) as tmp:
            tmp.write_text(json.dumps(sidecar))
        enrich_trace_sentinel(
            d,
            traceId=trace_id,
            codeJsonHash=code_json_hash,
            codeObjectHash=code_object_hash,
        )
        out = d / SIDECAR
        written += 1
        print(
            f"  {d.name}: {sidecar['resolved']}/{total} instructions resolved, "
            f"{len(sidecar['functions'])} functions -> {SIDECAR} "
            f"({out.stat().st_size / 1024:.1f} KiB)"
        )
    if written == 0:
        raise SystemExit(f"no sidecar written ({skipped} dispatch(es) skipped)")
    if skipped:
        print(
            f"  {written} dispatch(es) resolved, {skipped} left without a "
            "sidecar (innermost frame only)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
