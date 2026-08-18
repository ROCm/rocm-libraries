#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structural checks on a ReuseAcrossPersistent kernel's generated assembly.

RAP is developed against a functional simulator that does not model timing, so a
kernel with missing s_waitcnt or missing s_barrier still validates PASSED. Those
are not hypothetical: an earlier attempt at the iter0/iterN peel produced an
iterN region with zero backend-inserted waitcnts (the region had no CFG
predecessor) and another with four missing barriers, both of which a functional
run cannot see. This script is therefore the gate that "PASSED" cannot be.

Two independent checks:

1. Region equivalence. iter0 and iterN are two emissions of the same compute
   section, so their instruction histograms must match except for the transfers
   RAP deliberately drops. --mode peel expects no differences at all; --mode drop
   additionally expects iterN to have no A/MXSA local reads.

2. Resident-range integrity. Nothing anywhere in the kernel may write into the
   VGPRs that hold A/MXSA across persistent iterations. The scan resolves both
   `v[N]` and bare `vN` destinations -- the store address math uses the bare form,
   and missing it once let a wrong conclusion stand for several rounds.

Exit status is non-zero if any check fails.
"""

import argparse
import re
import sys
from collections import Counter

# "/* RAP: ValuA vgpr [176...370) held resident across persistent iterations */"
RESIDENT_RE = re.compile(r"RAP: (Valu\w+) vgpr \[(\d+)\.\.\.(\d+)\) held resident")
LABEL_RE = re.compile(r"^\s*(label_\w+):")
SET_RE = re.compile(r"^\s*\.set\s+(\w+)\s*,\s*(.+?)\s*$")
# A destination is the first operand; capture both v[..] and bare vN forms.
DST_RE = re.compile(r"^\s*([a-z][\w.]*)\s+(v\[[^\]]+\]|v\d+)\s*,")
BARE_RE = re.compile(r"^v(\d+)$")
BRACKETED_RE = re.compile(r"^v\[([^\]]+)\]$")
# gfx1250 encodes only 8 bits of VGPR index and selects the bank with
# s_set_vgpr_msb, so operands are printed as "<logical> - <bank base>", e.g.
# v[512-512] or v[vgprValuA_X0_I0+7-256]. Subtracting the bank base would make a
# high register look like a low one, so drop it and keep the logical index.
BANK_OFFSET_RE = re.compile(r"-\s*(\d+)\b")

ITER0_LABEL = "label_PersistentLoopStart"
ITERN_LABEL = "label_RAP_IterN"
JOIN_LABEL = "label_RAP_StoreJoin"


def readInstructions(path):
    """Return [(lineNo, label|None, text)] with comments and blanks dropped."""
    out = []
    for lineNo, raw in enumerate(open(path, errors="replace"), 1):
        text = re.sub(r"/\*.*?\*/", "", raw)
        text = re.sub(r"//.*$", "", text).rstrip()
        if not text.strip():
            continue
        label = LABEL_RE.match(text)
        out.append((lineNo, label.group(1) if label else None, text))
    return out


def parseSymbols(path):
    """Resolve `.set` symbols to integers; expressions may reference other symbols."""
    raw = {}
    for line in open(path, errors="replace"):
        m = SET_RE.match(re.sub(r"//.*$", "", line))
        if m:
            raw.setdefault(m.group(1), m.group(2))

    resolved = {}

    def evaluate(name, seen):
        if name in resolved:
            return resolved[name]
        if name not in raw or name in seen:
            return None
        expr = raw[name]
        # Only plain integer arithmetic over other symbols is expected here.
        if not re.fullmatch(r"[\w\s+\-*/()]+", expr):
            return None
        substituted = expr
        for ref in set(re.findall(r"[A-Za-z_]\w*", expr)):
            value = evaluate(ref, seen | {name})
            if value is None:
                return None
            substituted = re.sub(r"\b%s\b" % re.escape(ref), str(value), substituted)
        try:
            value = int(eval(substituted, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception:
            return None
        resolved[name] = value
        return value

    for name in raw:
        evaluate(name, set())
    return resolved


def residentRanges(path):
    ranges = {}
    for line in open(path, errors="replace"):
        m = RESIDENT_RE.search(line)
        if m:
            ranges[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return ranges


def evaluateOperand(expr, symbols):
    """Logical register index of one operand expression, or None if unresolvable."""
    expr = BANK_OFFSET_RE.sub(lambda m: "" if int(m.group(1)) % 256 == 0 and int(m.group(1)) else m.group(0), expr)
    substituted = expr
    for ref in set(re.findall(r"[A-Za-z_]\w*", expr)):
        if ref not in symbols:
            return None
        substituted = re.sub(r"\b%s\b" % re.escape(ref), str(symbols[ref]), substituted)
    if not re.fullmatch(r"[\d\s+\-*/()]+", substituted):
        return None
    try:
        return int(eval(substituted, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


def destinationRegisters(text, symbols):
    """Logical [lo, hi] of an instruction's destination, or None if not resolvable."""
    m = DST_RE.match(text)
    if not m:
        return None
    operand = m.group(2)
    bare = BARE_RE.match(operand)
    if bare:
        return int(bare.group(1)), int(bare.group(1))
    bracketed = BRACKETED_RE.match(operand)
    if not bracketed:
        return None
    parts = bracketed.group(1).split(":")
    lo = evaluateOperand(parts[0], symbols)
    if lo is None:
        return None
    hi = evaluateOperand(parts[1], symbols) if len(parts) > 1 else lo
    return lo, hi if hi is not None else lo


def histogram(instructions):
    counts = Counter()
    for _, _, text in instructions:
        stripped = text.strip()
        mnemonic = stripped.split()[0] if stripped else ""
        if mnemonic.startswith("ds_load"):
            for tensor in ("ValuA", "ValuMXSA", "ValuB", "ValuMXSB"):
                if re.search(r"vgpr%s_X" % tensor, stripped):
                    counts["ds_load_%s" % tensor] += 1
                    break
            else:
                counts["ds_load_other"] += 1
        elif mnemonic.startswith("s_wait_dscnt"):
            counts["s_wait_dscnt"] += 1
        elif mnemonic.startswith("s_wait_tensorcnt"):
            counts["s_wait_tensorcnt"] += 1
        elif mnemonic.startswith("s_barrier"):
            counts["s_barrier"] += 1
        elif mnemonic == "tensor_load_to_lds":
            counts["tensor_load_to_lds"] += 1
        elif mnemonic.startswith("v_wmma"):
            counts["wmma"] += 1
            operands = stripped.split(None, 1)[1] if " " in stripped else ""
            # Splitting on commas outside brackets is enough to spot a literal 0
            # accumulator source, which is how InitCIterWmma zeroes C.
            depth = 0
            current = ""
            for ch in operands:
                if ch in "[(":
                    depth += 1
                elif ch in "])":
                    depth -= 1
                if ch == "," and depth == 0:
                    if current.strip() == "0":
                        counts["wmma_zero_c"] += 1
                        break
                    current = ""
                else:
                    current += ch
    return counts


def splitRegions(instructions):
    """Return (iter0, iterN) instruction slices, or (None, None) when unpeeled."""
    starts = {}
    for index, (_, label, _) in enumerate(instructions):
        if label in (ITER0_LABEL, ITERN_LABEL, JOIN_LABEL) and label not in starts:
            starts[label] = index
    if ITERN_LABEL not in starts or JOIN_LABEL not in starts:
        return None, None
    return (instructions[starts[ITER0_LABEL]:starts[ITERN_LABEL]],
            instructions[starts[ITERN_LABEL]:starts[JOIN_LABEL]])


# Counters that must be identical between the two emissions, and the reason each
# one is here. Every entry corresponds to a failure mode seen in practice.
EQUAL_KEYS = {
    "s_barrier": "barrier rebuild started iterN from iter0's memory-token state",
    "wmma": "the two emissions diverged",
    "wmma_zero_c": "iterN did not get the InitCIterWmma clone, so C is never zeroed",
    "tensor_load_to_lds": "TDM descriptors are nulled, not removed, so the count is fixed",
    "ds_load_ValuB": "B is reloaded every tile in both emissions",
    "ds_load_ValuMXSB": "B scales are reloaded every tile in both emissions",
}


def checkRegions(path, mode, iter0, iterN):
    failures = []
    h0, hN = histogram(iter0), histogram(iterN)

    print("  %-22s %8s %8s" % ("counter", "iter0", "iterN"))
    for key in sorted(set(h0) | set(hN)):
        print("  %-22s %8d %8d" % (key, h0[key], hN[key]))

    for key, reason in EQUAL_KEYS.items():
        if h0[key] != hN[key]:
            failures.append("%s: iter0=%d iterN=%d (%s)" % (key, h0[key], hN[key], reason))
    if h0["wmma_zero_c"] == 0:
        failures.append("wmma_zero_c is 0 in both emissions; C is never zeroed")

    # Waits are inserted by the backend from real dataflow, so once iterN stops
    # loading A the two emissions legitimately need different numbers of them --
    # they must match while the copies are identical, but after the drop the only
    # sound invariant is that the backend produced any at all. Zero means it never
    # looked at the region, which is the failure a functional run cannot see.
    for key in ("s_wait_dscnt", "s_wait_tensorcnt"):
        if mode == "peel" and h0[key] != hN[key]:
            failures.append("%s: iter0=%d iterN=%d (the two copies are identical here, so these must match)"
                            % (key, h0[key], hN[key]))
        for region, counts in (("iter0", h0), ("iterN", hN)):
            if counts[key] == 0:
                failures.append("%s: %s has none; the backend never processed that region"
                                % (key, region))

    for tensor in ("ValuA", "ValuMXSA"):
        key = "ds_load_%s" % tensor
        if mode == "peel":
            if h0[key] != hN[key]:
                failures.append("%s: iter0=%d iterN=%d (peel alone must not drop loads)"
                                % (key, h0[key], hN[key]))
        else:
            if h0[key] == 0:
                failures.append("%s: iter0 has no loads, so nothing fills the resident block" % key)
            if hN[key] != 0:
                failures.append("%s: iterN still has %d loads, which RAP should have dropped"
                                % (key, hN[key]))
    return failures


def checkResidentRanges(path, instructions, symbols, ranges):
    failures = []
    # The block only has to survive from the first fill onwards, so start at the
    # persistent loop. Everything before it runs once, before any tile is loaded.
    start = next((i for i, (_, label, _) in enumerate(instructions) if label == ITER0_LABEL), 0)
    for name, (lo, hi) in sorted(ranges.items()):
        writers = []
        for lineNo, label, text in instructions[start:]:
            if label:
                continue
            # Accesses written through the block's own symbols are the fill path;
            # the hazard is code that reaches the same registers by number, which
            # is what the store's address math does.
            if "vgprValu" in text.split(",")[0]:
                continue
            dst = destinationRegisters(text, symbols)
            if dst and dst[0] < hi and dst[1] >= lo:
                writers.append((lineNo, text.strip()))
        print("  %-10s [%d...%d)  writers outside the fill path: %d" % (name, lo, hi, len(writers)))
        for lineNo, text in writers[:5]:
            print("      %d: %s" % (lineNo, text))
        if writers:
            failures.append("%s [%d...%d) is written by %d instruction(s), e.g. line %d"
                            % (name, lo, hi, len(writers), writers[0][0]))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assembly", nargs="+")
    parser.add_argument("--mode", choices=("peel", "drop"), default="peel",
                        help="peel: the two emissions must be identical. "
                             "drop: iterN must additionally have no A/MXSA local reads.")
    parser.add_argument("--skip-resident-scan", action="store_true",
                        help="only run the region equivalence check")
    args = parser.parse_args()

    allFailures = []
    for path in args.assembly:
        print("== %s" % path)
        instructions = readInstructions(path)
        ranges = residentRanges(path)
        if not ranges:
            print("  not a ReuseAcrossPersistent kernel; skipped")
            continue

        iter0, iterN = splitRegions(instructions)
        if iter0 is None:
            print("  peel labels absent; region equivalence not applicable yet")
        else:
            allFailures += ["%s: %s" % (path, f) for f in checkRegions(path, args.mode, iter0, iterN)]

        if not args.skip_resident_scan:
            symbols = parseSymbols(path)
            allFailures += ["%s: %s" % (path, f)
                            for f in checkResidentRanges(path, instructions, symbols, ranges)]

    if allFailures:
        print("\nFAILED")
        for failure in allFailures:
            print("  " + failure)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
