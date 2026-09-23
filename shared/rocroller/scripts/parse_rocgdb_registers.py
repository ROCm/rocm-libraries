#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Parse rocgdb output. For each instruction line followed by 4 thread arrays,
prints the instruction, register, and the concatenated 256-element array.

Reads from stdin or a file argument.
"""

import re
import sys

INSTR_RE = re.compile(r"=>\s+0x[0-9a-f]+\s+<[^>]+>:\s+(.+)")
REG_RE = re.compile(r"^# (\$\w+)", re.MULTILINE)
ARRAY_RE = re.compile(r"\$\d+\s*=\s*\{([^}]*)\}", re.DOTALL)


def format_array(values):
    return ", ".join(str(v) for v in values)


def main():  # noqa: C901
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()

    # Collect (position, kind, value) events in document order
    events = []
    for m in INSTR_RE.finditer(text):
        events.append((m.start(), "instr", m.group(1).strip()))
    for m in REG_RE.finditer(text):
        events.append((m.start(), "reg", m.group(1)))
    for m in ARRAY_RE.finditer(text):
        nums = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        events.append((m.start(), "array", nums))
    events.sort()

    # Walk events: each instruction starts a new group; flush when group has 4 arrays
    groups = []  # list of (instruction, reg, joined_array)
    current_instr = None
    current_reg = None
    current_arrays = []

    for _, kind, value in events:
        if kind == "reg":
            current_reg = value
            current_instr = None
            current_arrays = []
        elif kind == "instr":
            if current_instr is None:
                current_instr = value
        elif kind == "array":
            if current_reg is not None:
                current_arrays.append(value)
                if len(current_arrays) == 4:
                    joined = [v for a in current_arrays for v in a]
                    groups.append((current_instr, current_reg, joined))
                    current_reg = None
                    current_instr = None
                    current_arrays = []

    # Group instructions that share identical address arrays
    seen = {}  # tuple(values) -> index in ordered_groups
    ordered_groups = []  # list of (values, [label, ...])
    for instr, reg, values in groups:
        key = tuple(values)
        label = f"{instr}  ({reg})"
        if key in seen:
            ordered_groups[seen[key]][1].append(label)
        else:
            seen[key] = len(ordered_groups)
            ordered_groups.append((values, [label]))

    for values, labels in ordered_groups:
        for label in labels:
            print(f"// {label}")
        print("{" + format_array(values) + "}")
        print()


if __name__ == "__main__":
    main()
