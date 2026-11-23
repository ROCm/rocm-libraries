# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Remove rocRoller's codegen provenance from checked-in custom kernels.

rocRoller emits its internal view of the kernel alongside the kernel it
generates, and the capture procedure that turns a generated kernel into a
checked-in ``.s`` file keeps all of it.  It arrives in two forms.

**Kernel-graph dumps.**  Three YAML keys inside the ``.amdgpu_metadata``
section:

  ``.kernel_graph``      the control and coordinate graphs, as nested YAML
  ``.kernel_graph_dot``  the same two graphs again, as one Graphviz string
  ``.command``           the command-level operation list the graphs lower from

None of the three is read by anything.  ``CustomKernels._readEmbeddedYaml``
deserialises the whole YAML block and its callers take only ``custom.config``
(the registration record that drives kernel launch) and ``amdhsa.kernels``
(the kernel descriptor and argument layout the GPU runtime needs), so the dump
costs parse time and repository size and returns nothing.

**Codegen-trace comments.**  Whole-line ``//`` comments outside the metadata
section, which narrate how rocRoller lowered the kernel: register allocation
(``// Allocated : 1 VGPR ...``), expression lowering (``// (op 1122) Assign
stride expression``), liveness (``// Deleting tag 84``), and so on.  They are
comments, so the assembler discards them; they outnumber the instructions they
annotate by better than ten to one.

This script deletes both.  What it never touches:

  * anything between ``.amdgpu_metadata`` and ``.end_amdgpu_metadata`` other
    than the three graph keys -- so ``custom.config`` and ``amdhsa.kernels``
    come through byte for byte;
  * any line that carries something besides a comment, so instructions keep
    their trailing ``// (op N)`` tags and directives such as
    ``.amdgcn_target`` are left alone;
  * comments that carry a licence or an annotation marker (see
    ``_KEEP_COMMENT_RE``), so the SPDX header and any hand-written note
    survive;
  * every comment in a kernel that did not come from rocRoller.  The other
    custom kernels in the tree are hand-written and their comments are load
    bearing, so the trace strip runs only where ``custom.config`` says
    ``Origin: rocroller``.

Usage:
    python -m tensilelite.StripKernelGraphs [--check] FILE_OR_DIR [...]

``--check`` reports what would be removed and exits non-zero if anything
would be, which lets continuous integration fail a kernel that reintroduces
provenance.
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

# The label under which trace-comment removals are counted and reported.  It is
# not a YAML key, so it is printed without the leading dot the graph keys get.
COMMENT_LABEL = "trace comments"

# A line whose first content is "//" carries nothing but a comment.  A line
# with code before the "//" is never a candidate: removing it would delete an
# instruction, and rewriting it would mean parsing the instruction.
_COMMENT_ONLY_RE = re.compile(r"^\s*//")

# Comments worth keeping.  The licence words are matched case-insensitively
# because headers vary; the annotation markers are matched as written, because
# that is how a human writes them and rocRoller never does.
_KEEP_COMMENT_RE = re.compile(
    r"(?i:copyright|spdx|licen[cs]e)|\b(?:TODO|FIXME|XXX|HACK|NOTE|WARNING)\b"
)

# Only rocRoller writes trace comments.  Every other custom kernel in the tree
# is hand-written, and its comments are the only documentation it has, so the
# trace strip runs solely on kernels whose custom.config declares this origin.
_ROCROLLER_ORIGIN_RE = re.compile(r"^\s*Origin:\s*rocroller\s*$", re.IGNORECASE)


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


def stripGraphDumps(lines):
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


def isRocRollerKernel(lines):
    """True when custom.config declares the kernel came out of rocRoller."""
    begin, end = _metadataRegion(lines)
    if begin is None:
        return False
    return any(_ROCROLLER_ORIGIN_RE.match(line) for line in lines[begin:end])


def stripTraceComments(lines):
    """Return (newLines, removedCount) with rocRoller's trace comments deleted.

    A line goes only when all four hold: the kernel is rocRoller's, the line
    lies outside the metadata section, its first content is ``//``, and it
    carries no licence or annotation marker.  Everything else -- instructions
    and their trailing comments, directives, blank lines, the whole metadata
    section -- is returned as it came in.
    """
    if not isRocRollerKernel(lines):
        return lines, 0

    begin, end = _metadataRegion(lines)
    if begin is None:
        begin = end = len(lines)

    keep = []
    removed = 0
    for i, line in enumerate(lines):
        protected = begin <= i < end
        if (not protected and _COMMENT_ONLY_RE.match(line)
                and not _KEEP_COMMENT_RE.search(line)):
            removed += 1
            continue
        keep.append(line)
    if not removed:
        return lines, 0
    return keep, removed


def stripLines(lines):
    """Return (newLines, removedByLabel) with all rocRoller provenance gone.

    ``removedByLabel`` maps each graph key found to the number of lines it
    owned, plus ``COMMENT_LABEL`` to the number of trace comments removed.
    Labels that removed nothing are absent.
    """
    lines, removed = stripGraphDumps(lines)
    lines, comments = stripTraceComments(lines)
    if comments:
        removed = dict(removed)
        removed[COMMENT_LABEL] = comments
    return lines, removed


def _label(key):
    """How a removal label is printed: graph keys are YAML keys, dot and all."""
    return key if key == COMMENT_LABEL else "." + key


def processFile(path, check=False):
    """Strip one file.  Returns (removedByLabel, bytesRemoved)."""
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
              + ", ".join(f"{_label(k)} ({v} lines)"
                          for k, v in sorted(removed.items())))

    print(f"{touched} of {len(files)} file(s) carried rocRoller provenance; "
          f"{totalBytes} bytes removed")
    for key, count in sorted(totals.items()):
        print(f"  {_label(key)}: {count} lines")
    return 1 if (args.check and touched) else 0


if __name__ == "__main__":
    sys.exit(main())
