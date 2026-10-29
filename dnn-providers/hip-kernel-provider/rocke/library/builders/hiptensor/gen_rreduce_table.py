#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write (or check) platform/python/rocke/dispatch/families/hiptensor_rreduce.json, the route
table of hipTensor's tuned reduction classes (kernels.common.hiptensor_aot.rreduce): per target
the palette (the classes of the kernel pack), the tuned problems (canonical problem -> measured
knobs, with the class those knobs run it with), and the prototypes of the untuned choice
(measured problems with the palette knobs that ran them fastest and their wave count,
``rreduce_host.untuned_choices``).

  python gen_rreduce_table.py --merge picks.jsonl      # add / replace tuned problems
  python gen_rreduce_table.py --palette palette.json   # replace the palette of the targets it lists
  python gen_rreduce_table.py --protos protos.json     # replace the prototypes
  python gen_rreduce_table.py --check                  # exit 1 when the table is invalid

Run it from library/builders/hiptensor (or with library/ and platform/python on PYTHONPATH). A
pick is one JSON object per line: ``{"arch", "a", "d", "op", "kept", "red", "knobs"}`` with the
problem in hipTensor's canonical form (``kept`` = (extent, A stride, D stride), ``red`` = (extent,
A stride), each innermost first, extent-1 modes dropped, contiguous modes merged) and the knobs
of ``rreduce_host.Knobs``; the class is derived (``rreduce_host.entry_class``). A palette file is
a JSON list of classes (``rreduce_host.class_dict``). The knobs come from measurements on the
target (a tuning flow outside this tree); this tool only validates and orders them.

``--check`` verifies: the ABI, every palette class (a valid class, streaming loads only on
gfx942 / gfx950), every entry (canonical problem, knobs valid for it, class equal to the derived
one, one entry per problem and target), every prototype (fields, knobs of a palette class of every
target). library/tests/test_hiptensor_aot_rreduce_table.py runs the same check.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parents[1]
PLATFORM_PY = LIBRARY.parent / "platform" / "python"
sys.path.insert(0, str(PLATFORM_PY))
sys.path.insert(0, str(LIBRARY))

from kernels.common.hiptensor_aot import rreduce_host as H  # noqa: E402
from kernels.common.hiptensor_aot.host import canonicalize  # noqa: E402
from kernels.common.hiptensor_aot.rreduce import (  # noqa: E402
    OPS,
    RREDUCE_ABI,
    STREAMING_ARCHS,
)

OUT = PLATFORM_PY / "rocke" / "dispatch" / "families" / H.TABLE_NAME
ARCHES = ("gfx90a", "gfx942", "gfx950")


def canonical(kept, red):
    """The problem in hipTensor's canonical form (rreduce_host conventions)."""
    kd = canonicalize([(e, [sa, sd]) for e, sa, sd in kept], 0)
    rd = canonicalize([(e, [sa]) for e, sa in red], 0)
    return [[e, s[0], s[1]] for e, s in kd], [[e, s[0]] for e, s in rd]


def entry_of(pick: dict) -> dict:
    kept = [[int(x) for x in k] for k in pick["kept"]]
    red = [[int(x) for x in r] for r in pick["red"]]
    k = H.knobs_of(pick["knobs"])
    spec = H.entry_class(pick["arch"], pick["a"], pick["d"], pick["op"], kept, red, k)
    return dict(
        arch=pick["arch"],
        a=pick["a"],
        d=pick["d"],
        op=pick["op"],
        kept=kept,
        red=red,
        knobs=dict(k.as_dict()),
        **{"class": H.class_dict(spec, pick["arch"])},
    )


def entry_key(e: dict):
    return (e["arch"], H.problem_key(e["a"], e["d"], e["op"], e["kept"], e["red"]))


def ordered(table: dict) -> dict:
    """Palette by target in file order; entries sorted by target and problem key."""
    pal = sorted(table["palette"], key=lambda c: ARCHES.index(c["arch"]))
    ent = sorted(table["entries"], key=entry_key)
    return dict(
        abi=RREDUCE_ABI,
        version=1,
        palette=pal,
        entries=ent,
        prototypes=list(table.get("prototypes", [])),
    )


def dump(table: dict) -> str:
    """The table with one palette class / entry per line."""

    def line(x):
        return json.dumps(x, separators=(", ", ": "))

    pal = ",\n".join("  " + line(c) for c in table["palette"])
    ent = ",\n".join("  " + line(e) for e in table["entries"])
    pro = ",\n".join("  " + line(p) for p in table.get("prototypes", []))
    return (
        "{\n"
        f' "abi": {json.dumps(table["abi"])},\n'
        f' "version": {table["version"]},\n'
        f' "palette": [\n{pal}\n ],\n'
        f' "entries": [\n{ent}\n ],\n'
        f' "prototypes": [\n{pro}\n ]\n'
        "}\n"
    )


def check(table: dict) -> list:
    errs = []
    if table.get("abi") != RREDUCE_ABI:
        errs.append(f"abi {table.get('abi')!r} is not {RREDUCE_ABI}")
    seen_cls = set()
    for i, c in enumerate(table.get("palette", [])):
        try:
            s = H.spec_of(c)
        except (KeyError, ValueError) as e:
            errs.append(f"palette {i}: {e}")
            continue
        if c.get("arch") not in ARCHES:
            errs.append(f"palette {i}: arch {c.get('arch')!r}")
        if s.nt and c.get("arch") not in STREAMING_ARCHS:
            errs.append(f"palette {i}: streaming loads on {c.get('arch')}")
        if s.a != s.d:
            errs.append(f"palette {i}: a != d (the API's rows have one type)")
        name = (c.get("arch"), s.kernel_name())
        if name in seen_cls:
            errs.append(f"palette {i}: duplicate class {s.kernel_name()}")
        seen_cls.add(name)
    seen = set()
    for i, e in enumerate(table.get("entries", [])):
        try:
            if e["arch"] not in ARCHES or e["a"] != e["d"] or e["op"] not in OPS:
                errs.append(f"entry {i}: arch / types / op")
                continue
            kept, red = canonical(e["kept"], e["red"])
            if kept != e["kept"] or red != e["red"]:
                errs.append(f"entry {i}: the problem is not in canonical form")
            k = H.knobs_of(e["knobs"])
            why = H.check_knobs(e["kept"], e["red"], e["a"], k, arch=e["arch"])
            if why is not None:
                errs.append(f"entry {i}: knobs do not apply ({why})")
                continue
            want = entry_of(e)
            if want["class"] != e["class"] or want["knobs"] != e["knobs"]:
                errs.append(f"entry {i}: class / knobs differ from the derived ones")
            key = entry_key(e)
            if key in seen:
                errs.append(f"entry {i}: duplicate problem {key}")
            seen.add(key)
        except (KeyError, TypeError, ValueError) as ex:
            errs.append(f"entry {i}: {ex!r}")
    # prototypes: valid fields, and knobs some palette class of every target runs (with the nt
    # bit the target allows)
    knobsets = {}
    for c in table.get("palette", []):
        key = (c["arch"], c["mode"], H.ELEM_BYTES[c["a"]])
        knobsets.setdefault(key, set()).add(
            (
                c["block"],
                c["vec"],
                c["lanes"],
                c["unroll"],
                c.get("rows", 0),
                int(bool(c.get("nt", 0))),
            )
        )
    arches = sorted({c["arch"] for c in table.get("palette", [])})
    for i, p in enumerate(table.get("prototypes", [])):
        try:
            k = p["knobs"]
            if p["mode"] not in ("row", "col") or int(p["eb"]) not in (2, 4, 8):
                errs.append(f"prototype {i}: mode / eb")
                continue
            if len(p["f"]) != 5 or any(int(x) != x or x < 0 for x in p["f"]):
                errs.append(f"prototype {i}: features")
            if int(p["waves"]) < 1 or int(p["cus"]) < 1:
                errs.append(f"prototype {i}: waves / cus")
            for arch in arches:
                nt = int(bool(k.get("nt", 0))) if arch in STREAMING_ARCHS else 0
                want = (
                    k["block"],
                    k["vec"],
                    k["lanes"],
                    k["unroll"],
                    k.get("rows", 0),
                    nt,
                )
                if want not in knobsets.get((arch, p["mode"], int(p["eb"])), set()):
                    errs.append(
                        f"prototype {i}: knobs {want} not in the {arch} palette"
                    )
        except (KeyError, TypeError, ValueError) as ex:
            errs.append(f"prototype {i}: {ex!r}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--merge", help="JSONL of measured picks to add or replace")
    ap.add_argument(
        "--palette", help="JSON list of palette classes (replaces their targets')"
    )
    ap.add_argument(
        "--protos", help="JSON list of prototypes of the untuned choice (replaces them)"
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    path = Path(args.out)
    table = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else dict(abi=RREDUCE_ABI, version=1, palette=[], entries=[])
    )
    if args.palette:
        pal = json.loads(Path(args.palette).read_text(encoding="utf-8"))
        arches = {c["arch"] for c in pal}
        table["palette"] = [
            c for c in table["palette"] if c["arch"] not in arches
        ] + pal
    if args.merge:
        by_key = {entry_key(e): e for e in table["entries"]}
        for line in Path(args.merge).read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = entry_of(json.loads(line))
                by_key[entry_key(e)] = e
        table["entries"] = list(by_key.values())
    if args.protos:
        table["prototypes"] = json.loads(Path(args.protos).read_text(encoding="utf-8"))
    if args.palette or args.merge or args.protos:
        table = ordered(table)
        errs = check(table)
        if errs:
            print("\n".join(errs[:50]), file=sys.stderr)
            return 1
        path.write_text(dump(table), encoding="utf-8", newline="\n")
        print(
            f"wrote {path}: {len(table['palette'])} palette classes, {len(table['entries'])} "
            f"entries, {len(table['prototypes'])} prototypes"
        )
        return 0
    errs = check(table)
    if errs:
        print("\n".join(errs[:50]), file=sys.stderr)
        return 1
    print(
        f"{path}: {len(table['palette'])} palette classes, {len(table['entries'])} entries, "
        f"{len(table.get('prototypes', []))} prototypes: OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
