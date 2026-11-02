# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build a multi-mode conv pin table (``dispatch/conv_pins.py``) from pins files.

usage::

    python conv_pin_table.py --out dispatch/conv_pin_tables/gfx942_unet3d_bf16.json \\
        --arch gfx942 --mode deterministic=pins_det.json --mode fastest=pins_fast.json \\
        [--deterministic-mode deterministic] [--comment "..."]

Every pinned label must decode to a kernel identity of ``--arch`` whose label
round-trips. Every launch is classified with ``det_classify`` (``f32``: no
partial sums rounded below f32) and the class is recorded per case
(``deterministic``); the mode named by ``--deterministic-mode`` is refused if
any of its launches is not deterministic. Per case the table keeps the pins
file's command, call count, pinned launch and the number of judging sessions.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.common.det_classify import classify, is_deterministic
from kernels.conv.kernel_identity import KernelIdentity

from dispatch.conv_pins import _SPLIT_RE, parse_miopen_cmd

# Selection only: the measurements behind a pick stay with the run that made it
# (no absolute timings in committed artifacts).
_KEEP = ("cmd", "calls", "kernel", "sessions")


def build_table(arch: str, modes: dict, det_mode: str | None, comment: str = "", llvm_flavor: str = "llvm22") -> dict:
    out = {"arch": arch, "llvm_flavor": llvm_flavor, "comment": comment, "modes": {}}
    for mode, path in modes.items():
        doc = json.load(open(path))
        cases = {}
        for case, row in sorted(doc["cases"].items()):
            label = str(row["kernel"]).strip()
            inst = label.split()[0]
            ident = KernelIdentity.from_label(inst)
            if ident.label() != inst:
                raise SystemExit(f"{mode}/{case}: label does not round-trip: {inst}")
            if ident.arch != arch:
                raise SystemExit(f"{mode}/{case}: {ident.arch} kernel in a {arch} table")
            direction, _, _ = parse_miopen_cmd(row["cmd"])
            if direction != ident.direction:
                raise SystemExit(f"{mode}/{case}: {ident.direction} kernel pinned to a {direction} problem")
            m = _SPLIT_RE.search(label)
            sk = int(m.group(1)) if m else None
            det = is_deterministic(direction, ident, sk, "f32")
            if det_mode == mode and not det:
                raise SystemExit(f"{mode}/{case}: {label} is not deterministic ({classify(direction, ident, sk)[1]})")
            cases[case] = {k: row[k] for k in _KEEP if k in row}
            cases[case]["deterministic"] = det
        out["modes"][mode] = {"deterministic": mode == det_mode, "cases": cases}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--llvm-flavor", default="llvm22")
    ap.add_argument("--mode", action="append", required=True, metavar="NAME=PINS")
    ap.add_argument("--deterministic-mode", default="deterministic",
                    help="the mode whose every launch must be deterministic ('' = none)")
    ap.add_argument("--comment", default="")
    a = ap.parse_args(argv)
    modes = dict(m.split("=", 1) for m in a.mode)
    table = build_table(a.arch, modes, a.deterministic_mode or None, a.comment, a.llvm_flavor)
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1, sort_keys=False)
        f.write("\n")
    for mode, body in table["modes"].items():
        n = len(body["cases"])
        d = sum(1 for c in body["cases"].values() if c["deterministic"])
        print(f"{mode}: {n} cases, {d} deterministic launches", file=sys.stderr)
    print(f"-> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
