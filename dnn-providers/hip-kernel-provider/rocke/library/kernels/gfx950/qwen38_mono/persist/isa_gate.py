# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP0a's ISA gate for rocKE builds (D0 section 5, WP1).

WP0a (``probes/WP0a.md`` section 3) replaced the variant-set ISA gate by two
gates:

* host-only edits: build-key identity is decisive. For a rocKE build the key
  is the serialized rocKE IR (``ck.dsl.ir/v1``) plus the lowered LLVM IR; the
  WP1 identity census (``hackathon-prep/qwen38/truemono/WP1/wp1_identity.py``)
  compares both, byte for byte, over every build family;
* kernel-side knobs: the pooled ISA distribution gate (:mod:`.isa_dist`,
  vendored verbatim with its self-test). The compile of one build need not be
  one binary (the alola memory note: ``iglp_opt(1)`` code depends on earlier
  compiles in the process), so TEST's variants must lie in REF's pooled
  distribution.

This module writes rocKE compiles in the record format :mod:`.isa_dist`
reads (one JSON per ``<tree>__<arm>__<ctx>__<rep>``; per build: ``ok``,
``kind``, ``keyed_sha`` = sha256 of the serialized IR, ``objects`` =
``[{text_sha}]`` = sha256 of the code object's ``.text``), compiling each
build in a fresh process (``ctx="solo-<build>"``) or all builds in one
process (``ctx="fix-0"``)::

    python -m kernels.gfx950.qwen38_mono.persist.isa_gate --builds MOD:FN --tree H --arm base --reps 8 --out DIR
    python -m kernels.gfx950.qwen38_mono.persist.isa_dist DIR --gate H:T:base:differ --out gate.json

``--builds MOD:FN``: ``FN()`` returns ``{name: (kind, zero-arg builder)}``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys

__all__ = ["elf_sections", "record", "text_sha", "write_pool"]


def elf_sections(elf: bytes) -> dict:
    """{name: bytes} of an ELF64 little-endian object's sections (WP0a keys_isa)."""
    assert elf[:4] == b"\x7fELF" and elf[4] == 2 and elf[5] == 1, "not an ELF64 LE object"
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    hdrs = []
    for i in range(shnum):
        o = shoff + i * shentsize
        name, typ, _flags, _addr, off, size = struct.unpack_from("<IIQQQQ", elf, o)
        hdrs.append((name, typ, off, size))
    stro, strs = hdrs[shstrndx][2], hdrs[shstrndx][3]
    strtab = elf[stro : stro + strs]
    out = {}
    for name, typ, off, size in hdrs:
        n = strtab[name : strtab.index(b"\0", name)].decode()
        out[n] = b"" if typ == 8 else elf[off : off + size]  # SHT_NOBITS
    return out


def text_sha(hsaco: bytes) -> str:
    return hashlib.sha256(elf_sections(hsaco).get(".text", b"")).hexdigest()


def record(kind: str, kernel) -> dict:
    """One build's record: compile ``kernel`` (Python engine) in this process."""
    from rocke.core.ir_serialize import serialize
    from rocke.helpers.compile import compile_kernel

    try:
        art = compile_kernel(kernel, arch="gfx950", backend="python", capture_ir_text=False)
        return {
            "ok": True,
            "kind": kind,
            "keyed_sha": hashlib.sha256(serialize(kernel).encode()).hexdigest(),
            "objects": [{"text_sha": text_sha(bytes(art.hsaco)), "name": kernel.name}],
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "kind": kind, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _builds(spec: str) -> dict:
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)()


_ERRORS: list[str] = []


def write_pool(spec: str, tree: str, arm: str, reps: int, out: str, *, solo: bool = True) -> int:
    """``reps`` compiles of every build; ``solo``: each build in its own
    process (``ctx = solo-<name>``), else all builds in this process."""
    os.makedirs(out, exist_ok=True)
    builds = _builds(spec)
    n = 0
    for rep in range(reps):
        if solo:
            for name in builds:
                cmd = [sys.executable, "-m", "kernels.gfx950.qwen38_mono.persist.isa_gate", "--one", name, "--builds", spec]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                rec = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {
                    "ok": False, "kind": builds[name][0], "error": r.stderr[-300:]}  # fmt: skip
                _dump(out, tree, arm, f"solo-{name}", rep, {name: rec})
                n += 1
        else:
            recs = {name: record(kind, b()) for name, (kind, b) in builds.items()}
            _dump(out, tree, arm, "fix-0", rep, recs)
            n += 1
    return n


def _dump(out, tree, arm, ctx, rep, builds):
    for name, r in builds.items():
        if not r.get("ok"):
            _ERRORS.append(f"{name}: {r.get('error', '')[:300]}")
    data = {"env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset")}, "personality": None,
            "builds": builds}  # fmt: skip
    with open(os.path.join(out, f"{tree}__{arm}__{ctx}__{rep}.json"), "w") as fh:
        json.dump(data, fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", required=True, help="MOD:FN returning {name: (kind, builder)}")
    ap.add_argument("--one", default="", help="(internal) compile one build, print its record")
    ap.add_argument("--tree", default="H")
    ap.add_argument("--arm", default="base")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--fix", action="store_true", help="all builds in one process per rep (ctx fix-0)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.one:
        kind, b = _builds(a.builds)[a.one]
        print(json.dumps(record(kind, b())))
        return 0
    n = write_pool(a.builds, a.tree, a.arm, a.reps, a.out, solo=not a.fix)
    print(f"[isa_gate] {n} records -> {a.out}; failed compiles {len(_ERRORS)}"
          + (f"; first: {_ERRORS[0]}" if _ERRORS else ""))
    return 1 if _ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
