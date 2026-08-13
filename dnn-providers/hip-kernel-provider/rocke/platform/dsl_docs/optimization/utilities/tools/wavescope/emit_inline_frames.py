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

    python emit_inline_frames.py <att-output-dir>
    python emit_inline_frames.py <att-output-dir> --code-object k.hsaco
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SIDECAR = "inline_frames.json"

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


def find_code_object(root: Path) -> Path | None:
    """Largest code object under ``root`` -- the kernel's, not a stub."""
    candidates = [p for p in root.rglob(CODE_OBJECT_GLOB) if p.is_file()]
    candidates += [p for p in root.rglob("*.hsaco") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def code_object_id_of(path: Path) -> str | None:
    """The id rocprofv3 put in the dump's filename, if it named one that way."""
    m = CODE_OBJECT_ID_RE.search(path.name)
    return m.group(1) if m else None


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
    args = ap.parse_args(argv)

    root: Path = args.trace_dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    code_object = args.code_object or find_code_object(root)
    if code_object is None:
        raise SystemExit(
            f"no code object found under {root}. rocprofv3 writes "
            f"'{CODE_OBJECT_GLOB}' beside the raw trace; pass --code-object "
            "to point at it (or at the .hsaco the kernel was built from)."
        )

    dwarfdump = find_dwarfdump()
    frames = parse_inline_frames(code_object, dwarfdump)
    if not frames:
        raise SystemExit(
            f"{code_object.name} carries no inlining info. Build the kernel with "
            "ROCKE_DEBUG_LOC=1 (or IRBuilder(capture_loc=True)) so the lowering "
            "emits DWARF inlining scopes, then re-capture."
        )

    dirs = dispatch_dirs(root)
    if not dirs:
        raise SystemExit(f"no decoded dispatch folder under {root}")

    dumped_id = code_object_id_of(code_object)
    print(f"code object: {code_object}")
    print(f"inline frames with PC ranges: {len(frames)}")
    for d in dirs:
        code_json = d / "code.json"
        if not code_json.is_file():
            continue
        rows = json.loads(code_json.read_text())["code"]

        # Prefer the id rocprofv3 named the dump with. Fall back to the trace's
        # own value when it loaded exactly one object, which is the common case
        # and unambiguous. Anything else is left unfiltered rather than guessed
        # at, and the key still carries each row's code object.
        present = row_code_objects(rows)
        if dumped_id in present:
            code_object_id = dumped_id
        elif len(present) == 1:
            code_object_id = next(iter(present))
        else:
            code_object_id = None
            print(
                f"  {d.name}: warning: cannot tell which of {sorted(present)} "
                f"{code_object.name} is; matching on address across all of them"
            )

        sidecar = build_sidecar(rows, frames, code_object_id)
        total = len([r for r in rows if r and r[0] and not r[0].startswith(";")])
        out = d / SIDECAR
        out.write_text(json.dumps(sidecar))
        print(
            f"  {d.name}: {sidecar['resolved']}/{total} instructions resolved, "
            f"{len(sidecar['functions'])} functions -> {SIDECAR} "
            f"({out.stat().st_size / 1024:.1f} KiB)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
