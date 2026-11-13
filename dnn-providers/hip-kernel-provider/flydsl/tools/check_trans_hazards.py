# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Check the CDNA objects for a transcendental result read by the very next VALU op.

    python tools/check_trans_hazards.py --content-dir <descriptors>/FlyDSL \\
        [--arches gfx942 gfx950] [--objdump $ROCM_PATH/lib/llvm/bin/llvm-objdump]

On gfx942 and gfx950 a VALU instruction that reads the result of a transcendental
(``v_exp_f32``, ``v_log_f32``, ``v_rcp_f32``, ...) issued just before it needs a wait
state, which the backend inserts (an ``s_nop 0``) for the instructions it emits itself.
It cannot for inline assembly: a ``v_cvt_pk_bf16_f32`` written as inline asm right after
the ``v_exp_f32`` that produced its operand read a stale value, and gfx950's head_dim 64
bias objects returned inf and NaN (vendored CDNA kernel modification 46). Any such
adjacency in a shipped object therefore comes from inline assembly, and is reported.
Run it after regenerating CDNA objects (REGEN.md section 3); exit status 1 on a finding.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_TRANS = re.compile(
    r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)_(f32|f16|legacy_f32)\w*\s+(v\d+)"
)
_REG = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


def _registers(operands: str) -> set[int]:
    found: set[int] = set()
    for single, lo, hi in _REG.findall(operands):
        if single:
            found.add(int(single))
        else:
            found.update(range(int(lo), int(hi) + 1))
    return found


def findings(path: Path, objdump: str) -> list[str]:
    """Each place in @p path where a VALU op reads the transcendental just before it."""
    listing = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = []
    for raw in listing.splitlines():
        text = raw.split("//")[0].strip()
        if text and not text.endswith(">:") and not text.startswith("Disassembly"):
            lines.append(text)
    out = []
    for first, second in zip(lines, lines[1:]):
        match = _TRANS.match(first)
        if not match or not second.startswith("v_"):
            continue
        written = int(match.group(3)[1:])
        parts = second.split(None, 1)
        if len(parts) < 2:
            continue
        # The first operand is the destination; the rest are read.
        sources = parts[1].split(",", 1)[1] if "," in parts[1] else ""
        if written in _registers(sources):
            out.append(f"{path.name}: '{first}' then '{second}'")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--content-dir", type=Path, required=True)
    parser.add_argument("--arches", nargs="+", default=["gfx942", "gfx950"])
    parser.add_argument(
        "--objdump",
        default=os.path.join(
            os.environ.get("ROCM_PATH", "/opt/rocm"),
            "lib",
            "llvm",
            "bin",
            "llvm-objdump",
        ),
    )
    args = parser.parse_args()
    problems = []
    checked = 0
    for arch in args.arches:
        for path in sorted(args.content_dir.glob(f"*/{arch}/*.hsaco")):
            checked += 1
            problems += findings(path, args.objdump)
    for problem in problems:
        print(problem)
    print(
        f"checked {checked} object(s) for {' '.join(args.arches)}: {len(problems)} finding(s)"
    )
    return 1 if problems or not checked else 0


if __name__ == "__main__":
    sys.exit(main())
