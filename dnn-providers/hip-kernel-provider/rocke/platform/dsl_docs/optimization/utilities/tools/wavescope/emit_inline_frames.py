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
way to use this: every dispatch's old sidecar goes away as soon as the dispatch
folders are known -- before this run looks for a code object or for
llvm-dwarfdump, either of which can end the run -- so a dispatch this run does
not rewrite is left with no sidecar rather than the previous run's answer.

    python emit_inline_frames.py <att-output-dir>
    python emit_inline_frames.py <att-output-dir> --code-object k.hsaco
    python emit_inline_frames.py <att-output-dir> --invalidate-only
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

SIDECAR = "inline_frames.json"

# Written first and renamed over the destination, so a viewer never opens a
# half-written file.
TMP_SUFFIX = ".tmp"

# The dispatch folders rocprofv3 writes, each a self-contained trace.
DISPATCH_GLOB = "ui_output_*_dispatch_*"

# rocprofv3 dumps each loaded code object next to the raw trace.
CODE_OBJECT_GLOB = "*code_object_id_*.out"
CODE_OBJECT_ID_RE = re.compile(r"code_object_id_(\d+)")

# Virtual addresses are per code object, so an address alone does not identify an
# instruction in a trace that loaded more than one. Both columns form the join key.
CODEOBJ_COL = 4
VADDR_COL = 5

# Bumped whenever the on-disk shape changes. The viewer refuses versions it does
# not know rather than guessing at a layout and mis-attributing cost.
SIDECAR_VERSION = 2

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
                # The call site is recorded on the *callee*, so it describes where
                # this frame was entered from -- which is the line a reader wants.
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


def build_sidecar(rows: list, frames: list[dict], code_object_id: str | None) -> dict:
    """Map each instruction to its authoring call stack, keyed by code object and address.

    The DWARF came from exactly one code object, so rows belonging to any other
    are skipped: virtual addresses repeat across objects, and matching on address
    alone would confidently attach this object's call stacks to another's
    instructions wherever the two happen to collide.

    ``code_object_id`` of ``None`` means the caller could not identify which
    object the DWARF came from; every row is then a candidate, but the key still
    carries the row's own code object so the viewer's join stays exact.

    Files and function names are interned: the same handful repeat across
    hundreds of instructions, and the sidecar crosses a network hop to the
    viewer on a remote workspace.
    """
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
        # [funcIndex, callFileIndex, callLine, callColumn], outermost frame first.
        # The call site describes where the frame was entered, so the innermost
        # frame's own line stays in code.json's Source column.
        "schema": '"codeobj:addr" -> [[func, call_file, call_line, call_col], ...]',
        "code_object_id": code_object_id,
        "functions": list(funcs),
        "files": list(files),
        "stacks": stacks,
        "resolved": resolved,
        "skipped_other_object": skipped_other_object,
    }


def dispatch_dirs(root: Path) -> list[Path]:
    if (root / "code.json").is_file():
        return [root]
    return sorted(root.glob(DISPATCH_GLOB))


def invalidate_sidecars(dirs: list[Path]) -> None:
    """Drop every dispatch's sidecar before this run decides anything.

    This runs ahead of code-object discovery and of looking for
    llvm-dwarfdump, both of which end the run, and ahead of the loop, which an
    unreadable object ends part way through. A trace folder gets re-decoded and
    re-run over, and a sidecar left beside a dispatch this run does not rewrite
    is worse than no sidecar at all: the viewer loads it and attributes the
    dispatch to whichever code object the previous run picked, which is the
    address-overlap mis-attribution the per-dispatch selection exists to stop.
    No sidecar just falls back to innermost-frame attribution, which is how the
    Source tab behaved before this file existed.

    Staleness cannot be detected by reading the file. Its keys are bare
    addresses with nothing in them tying the file to a build, and a rebuild that
    moved only some addresses still joins on the rest. The file's presence is
    therefore the only signal there is, so this run has to earn it back.
    """
    for d in dirs:
        for path in (d / SIDECAR, d / f"{SIDECAR}{TMP_SUFFIX}"):
            if path.exists():
                path.unlink()
                print(f"  {d.name}: removed {path.name} from an earlier run")


@contextlib.contextmanager
def sidecar_write(d: Path) -> Iterator[Path]:
    """Yield the temporary path to write ``d``'s sidecar to.

    Leaving the block normally renames that file over the destination, so a
    viewer only ever opens a whole one: a partial write, from a full disk or an
    interrupt, is indistinguishable from a valid sidecar until it fails to
    parse. An exception removes the temporary instead and leaves no destination
    -- in particular it does not put back what was there before, which is the
    one outcome that would be read as this run's answer.
    """
    out = d / SIDECAR
    tmp = out.with_name(out.name + TMP_SUFFIX)
    out.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    try:
        yield tmp
        # Inside the guard: a full disk can fail the rename as easily as the
        # write, and that would otherwise leave the temporary behind.
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def find_code_objects(root: Path) -> list[Path]:
    """Every code object rocprofv3 dumped under ``root``, largest first.

    All of them are kept rather than just the biggest: a trace with more than
    one dispatch can carry a different object per dispatch, and the biggest one
    is not necessarily the one any particular dispatch ran.
    """
    candidates = [p for p in root.rglob(CODE_OBJECT_GLOB) if p.is_file()]
    candidates += [p for p in root.rglob("*.hsaco") if p.is_file()]
    return sorted(candidates, key=lambda p: (-p.stat().st_size, p.name))


def code_object_id_of(path: Path) -> str | None:
    """The id rocprofv3 put in the dump's filename, if it named one that way."""
    m = CODE_OBJECT_ID_RE.search(path.name)
    return m.group(1) if m else None


def select_code_object(
    candidates: list[Path], present: set[str], explicit: Path | None
) -> tuple[Path | None, str | None, str | None]:
    """Pick the code object a single dispatch ran, as ``(path, id, problem)``.

    Matching is by the id rocprofv3 named the dump with against the ids the
    dispatch's own rows carry, so a trace with several objects attributes each
    dispatch to the one it actually executed. Guessing is what makes this
    dangerous -- addresses repeat across objects, so a wrong pick still joins
    and silently reports another kernel's source -- and every case that cannot
    be decided returns a ``problem`` instead.
    """

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
        # A file rocprofv3 did not label with an id -- an .hsaco from the build,
        # say. The caller named it, so trust that over a filename convention,
        # but only where there is a single object for it to be.
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
        help="drop existing sidecars and write none, for a capture that has none",
    )
    args = ap.parse_args(argv)

    root: Path = args.trace_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    # Identify the dispatches, then invalidate, then everything that can fail.
    dirs = dispatch_dirs(root)
    invalidate_sidecars(dirs)

    if args.invalidate_only:
        print(f"  {len(dirs)} dispatch(es) left without a sidecar")
        return 0

    if not dirs:
        raise SystemExit(f"no decoded dispatch folder under {root}")

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
    skipped = 0
    for d in dirs:
        code_json = d / "code.json"
        if not code_json.is_file():
            continue
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

        sidecar = build_sidecar(rows, frames, code_object_id)
        total = len([r for r in rows if r and r[0] and not r[0].startswith(";")])
        with sidecar_write(d) as tmp:
            tmp.write_text(json.dumps(sidecar))
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
        # Success: the dispatches named above have a sidecar each. The skipped
        # ones have none, having had any stale file removed, so they fall back
        # to innermost-frame attribution. Reporting failure here would have the
        # caller announce that nothing was written at all.
        print(
            f"  {written} dispatch(es) resolved, {skipped} left without a "
            "sidecar (innermost frame only)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
