# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lever-scope gate of the Rust direct compiler (``dlevers``, ``rir::lever``).

    python lever_scope_check.py stamp SRC_DIR DST_DIR SCOPE
        writes every ``SRC_DIR/*.ir`` to DST_DIR with ``dlevers = s:"SCOPE"`` on every op of its body
    python lever_scope_check.py lift SRC_DIR DST_DIR
        writes every ``SRC_DIR/*.ir`` to DST_DIR with its inline asm lifted as ``compile_kernel`` does (PYTHONPATH=python)
    python lever_scope_check.py compare O_DIR OUT_FILE WANT:GOT[:WHAT] ...
        compares two ``target_compile`` runs (``O_DIR/<name>.status`` and ``O_DIR/<name>/<stem>.hsaco``):
        every kernel the same status and, when compiled, the same object; exit 1 on any difference

A scope over every op with levers L must build what ``ROCKE_DIRECT_LEVERS=L`` builds, and a scope turning every
lever of L off (``-name``) under ``ROCKE_DIRECT_LEVERS=L`` must build what no levers build.
"""

from __future__ import annotations

import hashlib
import os
import sys


def stamp(text: str, scope: str) -> str:
    """``text`` with lever scope ``scope`` on every op line of the kernel body."""
    out, body = [], False
    for line in text.split("\n"):
        tr = line.lstrip()
        body = body or tr.startswith("region @entry")
        if body and " ( " in tr and not tr.startswith("region"):
            loc = ""
            if " @loc " in line:
                line, loc = line.split(" @loc ", 1)
                loc = " @loc " + loc
            at = line.find(" ) { ")
            if at >= 0:
                line = line[:at] + f' ) {{ dlevers = s:"{scope}", ' + line[at + 5 :]
            else:
                line = line + f' {{ dlevers = s:"{scope}" }}'
            line += loc
        out.append(line)
    return "\n".join(out)


def _status(path: str) -> dict:
    st = {}
    with open(path) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2:
                st[parts[0]] = parts[1]
    return st


def _sha(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def compare(o: str, out_file: str, pairs: list) -> int:
    lines, bad = [], 0
    for pair in pairs:
        want, got, *what = pair.split(":", 2)
        what = what[0] if what else f"{want} == {got}"
        a, b = _status(f"{o}/{want}.status"), _status(f"{o}/{got}.status")
        same = diff = 0
        for stem in sorted(set(a) | set(b)):
            if a.get(stem) != b.get(stem):
                lines.append(f"{what}: {stem} STATUS {a.get(stem)} -> {b.get(stem)}")
                diff += 1
            elif a.get(stem) == "ok" and _sha(f"{o}/{want}/{stem}.hsaco") != _sha(f"{o}/{got}/{stem}.hsaco"):
                lines.append(f"{what}: {stem} OBJECT DIFFERS")
                diff += 1
            else:
                same += 1
        bad += diff
        lines.append(f"{what}: {same} same, {diff} differ")
    lines.append(f"SCOPES {'OK' if not bad else 'FAIL'} ({bad} differences)")
    with open(out_file, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-30:]))
    return 1 if bad else 0


def main(argv: list) -> int:
    if len(argv) >= 4 and argv[0] == "stamp":
        src, dst, scope = argv[1:4]
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            if f.endswith(".ir"):
                with open(os.path.join(src, f)) as fh:
                    text = fh.read()
                with open(os.path.join(dst, f), "w", newline="\n") as fh:
                    fh.write(stamp(text, scope))
        return 0
    if len(argv) >= 3 and argv[0] == "lift":
        # The inline-asm lifting compile_kernel applies before a direct compile (rocke.runtime.direct_lift).
        from rocke.core.ir_serialize import parse, serialize
        from rocke.runtime.direct_lift import lift

        src, dst = argv[1:3]
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            if f.endswith(".ir"):
                with open(os.path.join(src, f)) as fh:
                    text = fh.read()
                try:
                    text = serialize(lift(parse(text)))
                except Exception as e:  # noqa: BLE001 - kept as is: the compiles then compare its refusal
                    print(f"{f}: not lifted ({type(e).__name__}: {e})"[:300])
                with open(os.path.join(dst, f), "w", newline="\n") as fh:
                    fh.write(text)
        return 0
    if len(argv) >= 4 and argv[0] == "compare":
        return compare(argv[1], argv[2], argv[3:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
