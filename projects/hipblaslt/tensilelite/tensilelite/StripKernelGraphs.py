# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Remove rocRoller's debug kernel-graph dumps from checked-in custom kernels.

rocRoller emits its internal kernel representation alongside the kernel it
generates, and the capture procedure that turns a generated kernel into a
checked-in ``.s`` file keeps that dump.  Three YAML keys inside the
``.amdgpu_metadata`` section carry it:

  ``.kernel_graph``      the control and coordinate graphs, as nested YAML
  ``.kernel_graph_dot``  the same two graphs again, as one Graphviz string
  ``.command``           the command-level operation list the graphs lower from

None of the three is read by anything.  ``CustomKernels._readEmbeddedYaml``
deserialises the whole YAML block and its callers take only ``custom.config``
(the registration record that drives kernel launch) and ``amdhsa.kernels``
(the kernel descriptor and argument layout the GPU runtime needs), so the dump
costs parse time and repository size and returns nothing.

This script deletes those three keys and the lines they own.  It never touches
instructions, assembler directives such as ``.amdgcn_target``, the
``.amdhsa_kernel`` descriptor block, ``custom.config``, or ``amdhsa.kernels``:
it only edits lines between ``.amdgpu_metadata`` and ``.end_amdgpu_metadata``,
and within that region only a matched key and the more-deeply-indented lines
that follow it.

Usage:
    python -m tensilelite.StripKernelGraphs [--check] FILE_OR_DIR [...]

``--check`` reports what would be removed and exits non-zero if anything
would be, which lets continuous integration fail a kernel that reintroduces a
dump.
"""

import argparse
import os
import re
import sys

# The keys to delete.  Each is a mapping key inside the per-kernel entry of
# amdhsa.kernels, written with a leading dot like its neighbours.
GRAPH_KEYS = ("kernel_graph", "kernel_graph_dot", "command")

_KEY_RE = re.compile(r"^(\s*)\.(?:" + "|".join(GRAPH_KEYS) + r"):")
_METADATA_BEGIN = ".amdgpu_metadata"
_METADATA_END = ".end_amdgpu_metadata"


def _indent(line):
    """Indentation width of a line, or None when the line holds no content."""
    stripped = line.rstrip("\n")
    if not stripped.strip():
        return None
    return len(stripped) - len(stripped.lstrip(" "))


def _metadataRegion(lines):
    """Index range [begin, end) of the lines inside .amdgpu_metadata.

    Returns (None, None) when the file has no such section.
    """
    begin = end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin is None and stripped == _METADATA_BEGIN:
            begin = i + 1
        elif begin is not None and stripped == _METADATA_END:
            end = i
            break
    if begin is None or end is None:
        return None, None
    return begin, end


def stripLines(lines):
    """Return (newLines, removedByKey) with the graph dumps deleted.

    ``removedByKey`` maps each key found to the number of lines it owned.
    """
    begin, end = _metadataRegion(lines)
    if begin is None:
        return lines, {}

    removed = {}
    keep = [True] * len(lines)
    i = begin
    while i < end:
        match = _KEY_RE.match(lines[i])
        if not match:
            i += 1
            continue
        keyIndent = len(match.group(1))
        key = lines[i].strip().split(":", 1)[0].lstrip(".")
        j = i + 1
        while j < end:
            lineIndent = _indent(lines[j])
            if lineIndent is None:
                # A blank line continues the block only when indented content
                # follows it; otherwise it belongs to whatever comes next.
                k = j
                while k < end and _indent(lines[k]) is None:
                    k += 1
                if k < end and _indent(lines[k]) > keyIndent:
                    j = k
                    continue
                break
            if lineIndent <= keyIndent:
                break
            j += 1
        for k in range(i, j):
            keep[k] = False
        removed[key] = removed.get(key, 0) + (j - i)
        i = j

    if not removed:
        return lines, {}
    return [line for line, k in zip(lines, keep) if k], removed


def processFile(path, check=False):
    """Strip one file.  Returns (removedByKey, bytesRemoved)."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    before = sum(len(line.encode("utf-8")) for line in lines)
    newLines, removed = stripLines(lines)
    if not removed:
        return {}, 0
    after = sum(len(line.encode("utf-8")) for line in newLines)
    if not check:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(newLines)
    return removed, before - after


def gatherFiles(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files.extend(os.path.join(root, n) for n in sorted(names)
                             if n.endswith(".s"))
        else:
            files.append(path)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+",
                        help="Assembly files, or directories searched for *.s")
    parser.add_argument("--check", action="store_true",
                        help="Report without writing; exit 1 if a dump is present")
    args = parser.parse_args(argv)

    files = gatherFiles(args.paths)
    totals = {}
    totalBytes = 0
    touched = 0
    for path in files:
        removed, bytesRemoved = processFile(path, check=args.check)
        if not removed:
            continue
        touched += 1
        totalBytes += bytesRemoved
        for key, count in removed.items():
            totals[key] = totals.get(key, 0) + count
        print(f"{'would strip' if args.check else 'stripped'} {path}: "
              + ", ".join(f".{k} ({v} lines)" for k, v in sorted(removed.items())))

    print(f"{touched} of {len(files)} file(s) carried a kernel-graph dump; "
          f"{totalBytes} bytes removed")
    for key, count in sorted(totals.items()):
        print(f"  .{key}: {count} lines")
    return 1 if (args.check and touched) else 0


if __name__ == "__main__":
    sys.exit(main())
