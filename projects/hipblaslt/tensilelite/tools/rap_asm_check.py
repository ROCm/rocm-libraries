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

Three independent checks:

1. Region equivalence. iter0 and iterN are two emissions of the same compute
   section, so their instruction histograms must match except for the transfers
   RAP deliberately drops. --mode peel expects no differences at all; --mode drop
   additionally expects iterN to have no A/MXSA local reads.

2. Resident-range integrity. Nothing anywhere in the kernel may write into the
   VGPRs that hold A/MXSA across persistent iterations. The scan resolves both
   `v[N]` and bare `vN` destinations -- the store address math uses the bare form,
   and missing it once let a wrong conclusion stand for several rounds.

3. Wait sufficiency. Every ds_load must be waited for before anything reads what
   it wrote. Counting waits cannot answer this: dropping iterN's A loads legitimately
   removed 12 of its waits, and a count alone cannot distinguish that from a
   backend that lost a wait it still needed.

Run --mutate after touching check 3. It removes each wait in turn, and weakens
each by one, and reports how many of those the check notices. Without that number
"0 violations" might only mean the check never looked.

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
# KernelWriter.RAP_ITERN_SUFFIX. Labels the reuse copy renames to avoid colliding
# with the fill copy's.
ITERN_SUFFIX = "_RAPIterN"
# Any operand naming a label: branch targets, and the s_add_i32 anchor of the
# long-branch idiom.
LABEL_OPERAND_RE = re.compile(r"\b(label_\w+)")
# Every VGPR operand of an instruction, in printed order.
OPERAND_V_RE = re.compile(r"v\[[^\]]*\]|\bv\d+\b")
WAIT_DSCNT_RE = re.compile(r"^\s*s_wait_dscnt\s+(\S+)")
# The unroll's head label; the reuse copy's twin carries the suffix.
LOOP_HEAD_LABELS = ("label_LoopBeginL", "label_LoopBeginL" + ITERN_SUFFIX)


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
    "tensor_load_to_lds": "the two emissions drop the same sections' prefetch, so whatever "
                          "survives must survive in both",
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
    # they must match while the copies are identical, but after the drop a count
    # says nothing either way, and whether the surviving waits are enough is
    # checkWaitSufficiency's job. Zero still means the backend never looked at the
    # region, which is the failure a functional run cannot see.
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


def checkIterNLabelTargets(labels, iterN):
    """Every label the reuse copy names must be its own copy's.

    Missing a suffix does not fail at assembly time when the fill copy defines the
    same base name: the reference silently resolves to the other copy. That is how
    the "do not enter LoopL" escape came to jump from the reuse copy into the fill
    copy's drain, which reloads A over the resident registers. The escape is
    unreachable while K is pinned, so nothing caught it.

    No exemption list is needed, which is the point: a label defined outside the
    compute section has no suffixed twin, so referring to it cannot trip this.
    """
    failures = []
    for lineNo, label, text in iterN:
        if label:
            continue
        for target in LABEL_OPERAND_RE.findall(text):
            if target.endswith(ITERN_SUFFIX):
                continue
            if target + ITERN_SUFFIX in labels:
                failures.append(
                    "line %d names %s, but %s%s exists -- the reuse copy is "
                    "pointing into the fill copy: %s"
                    % (lineNo, target, target, ITERN_SUFFIX, text.strip()))
    return failures


def operandRanges(operandText, symbols):
    """Logical [lo, hi] of every VGPR operand, in printed order."""
    out = []
    for match in OPERAND_V_RE.finditer(operandText):
        token = match.group(0)
        bare = BARE_RE.match(token)
        if bare:
            out.append((int(bare.group(1)), int(bare.group(1))))
            continue
        bracketed = BRACKETED_RE.match(token)
        if not bracketed:
            continue
        parts = bracketed.group(1).split(":")
        lo = evaluateOperand(parts[0], symbols)
        if lo is None:
            continue
        hi = evaluateOperand(parts[1], symbols) if len(parts) > 1 else lo
        out.append((lo, hi if hi is not None else lo))
    return out


def branchTargets(instructions):
    """Labels control can arrive at from elsewhere.

    A label nothing names is a marker that control only falls into, so the set of
    outstanding loads carries straight through it. Clearing state at those too
    would throw away most of this check's reach for no reason.
    """
    targets = set()
    for _, label, text in instructions:
        if not label:
            targets.update(LABEL_OPERAND_RE.findall(text))
    return targets


def checkWaitSufficiency(instructions, symbols, targets, reportFrom=0):
    """Registers a ds_load writes must not be touched until a wait covers it.

    `s_wait_dscnt N` retires all but the newest N DS ops, and gfx1250 gives DS its
    own counter with in-order LDS completion, so a load with M later DS ops issued
    before the wait is covered exactly when N <= M. Walking the region while
    tracking the outstanding queue therefore decides coverage without a second
    build to compare against.

    Where control flow is not straight-line the outstanding set is cleared: at any
    label something branches to, and after any unconditional transfer. A
    conditional branch is not a cut, because its fall-through continues with this
    exact history. Clearing only ever forgets loads, so it cannot invent a
    violation -- it costs reach, which is why the caller prints how much was
    covered and why --mutate measures what the check still notices.
    """
    outstanding = []
    violations = []
    covered = 0
    cut = 0

    for index, (lineNo, label, text) in enumerate(instructions):
        stripped = text.strip()
        if label:
            if label in targets:
                cut += sum(1 for e in outstanding if e["dst"])
                outstanding = []
            continue
        if not stripped or stripped.startswith("."):
            continue
        mnemonic = stripped.split()[0]
        operands = stripped.split(None, 1)[1] if " " in stripped else ""

        wait = WAIT_DSCNT_RE.match(text)
        if wait:
            try:
                n = int(wait.group(1), 0)
            except ValueError:
                continue
            retired = outstanding[:max(0, len(outstanding) - n)]
            covered += sum(1 for e in retired if e["dst"])
            outstanding = outstanding[len(retired):]
            continue

        ranges = operandRanges(operands, symbols)
        isLoad = mnemonic.startswith("ds_") and "load" in mnemonic
        # A DS load writes its first operand and reads the rest; anything else
        # only reads, as far as a pending load is concerned.
        uses = ranges[1:] if isLoad else ranges
        if index >= reportFrom:
            for use in uses:
                for entry in outstanding:
                    if entry["dst"] and overlapsRange(use, entry["dst"]):
                        violations.append((lineNo, stripped, entry))

        if mnemonic.startswith("ds_"):
            outstanding.append({"dst": ranges[0] if (isLoad and ranges) else None,
                                "line": lineNo, "text": stripped})
            continue

        if mnemonic == "s_branch" or mnemonic.startswith("s_setpc") \
                or mnemonic == "s_endpgm":
            cut += sum(1 for e in outstanding if e["dst"])
            outstanding = []

    return violations, covered, cut


def overlapsRange(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def checkWaitsAcrossBackEdge(instructions, symbols, targets, headLabel):
    """The hazards that cross the unroll's back edge.

    The unroll is software-pipelined: a trip loads the X1 buffers and the next
    trip's WMMAs read them. A straight-line walk clears state at the loop head and
    so never checks those, which is most of the waits in the body -- when this was
    first measured, adding the back edge took the check from noticing 213 of 245
    wait mutations to 229.

    Walking the body twice puts the back-edge state at the second copy's entry, so
    uses there are checked against the first copy's loads. Only the second copy
    reports; the first is there to build the state up.
    """
    head = next((i for i, (_, label, _) in enumerate(instructions)
                 if label == headLabel), None)
    if head is None:
        return []
    back = max((i for i, (_, label, text) in enumerate(instructions)
                if not label and headLabel in LABEL_OPERAND_RE.findall(text)),
               default=None)
    if back is None or back <= head:
        return []
    body = instructions[head:back + 1]
    violations, _, _ = checkWaitSufficiency(
        body + body, symbols, targets - {headLabel}, reportFrom=len(body))
    return violations


def reportWaits(instructions, symbols, iter0, iterN):
    failures = []
    targets = branchTargets(instructions)
    regions = [("whole", instructions)]
    if iter0 is not None:
        regions += [("iter0", iter0), ("iterN", iterN)]
    for name, region in regions:
        violations, covered, cut = checkWaitSufficiency(region, symbols, targets)
        print("  waits %-6s loads covered before use: %-4d  at a control-flow cut: %-4d"
              "  unwaited uses: %d" % (name, covered, cut, len(violations)))
        for lineNo, text, entry in violations[:5]:
            print("      line %d reads what the load at line %d writes"
                  % (lineNo, entry["line"]))
            print("          load: %s" % entry["text"][:110])
            print("          use : %s" % text[:110])
        failures += ["%s: line %d reads what the ds_load at line %d writes with no "
                     "covering s_wait_dscnt" % (name, lineNo, entry["line"])
                     for lineNo, _, entry in violations]
    for headLabel in LOOP_HEAD_LABELS:
        violations = checkWaitsAcrossBackEdge(instructions, symbols, targets, headLabel)
        if violations:
            print("  waits %-6s cross-iteration unwaited uses: %d"
                  % (headLabel.replace("label_", "")[:6], len(violations)))
        failures += ["%s: line %d reads across the back edge what the ds_load at "
                     "line %d writes with no covering s_wait_dscnt"
                     % (headLabel, lineNo, entry["line"])
                     for lineNo, _, entry in violations]
    return failures


def mutateWaits(path):
    """How many wait mutations does check 3 actually notice?

    Removing a wait, or weakening it by one, must turn into a violation. Anything
    missed is a blind spot, and the blind spots are not evenly spread: they cluster
    at join points where state is cleared, so they are worth printing rather than
    summarising.

    Measured on MT64x256 at the time this was written: 229 of 245 noticed, for both
    mutations. The 16 misses are the two InitCIterWmma descending staircases and the
    PGR2 priming waits. Both are reached by an unconditional branch, and the loads
    they cover are issued before it, so propagating state along that one edge is
    what would close them -- they are not redundant waits.
    """
    instructions = readInstructions(path)
    symbols = parseSymbols(path)
    targets = branchTargets(instructions)
    waitIndices = [i for i, (_, label, text) in enumerate(instructions)
                   if not label and WAIT_DSCNT_RE.match(text)]

    def notices(mutated):
        violations, _, _ = checkWaitSufficiency(mutated, symbols, targets)
        if violations:
            return True
        return any(checkWaitsAcrossBackEdge(mutated, symbols, targets, h)
                   for h in LOOP_HEAD_LABELS)

    for name, delta in (("removed", None), ("weakened by one", 1)):
        missed = []
        for i in waitIndices:
            mutated = list(instructions)
            if delta is None:
                del mutated[i]
            else:
                lineNo, label, text = mutated[i]
                n = int(WAIT_DSCNT_RE.match(text).group(1), 0)
                mutated[i] = (lineNo, label,
                              WAIT_DSCNT_RE.sub("  s_wait_dscnt %d" % (n + delta), text))
            if not notices(mutated):
                missed.append(instructions[i][0])
        print("  wait %-16s mutations: %-4d noticed: %-4d missed: %d"
              % (name, len(waitIndices), len(waitIndices) - len(missed), len(missed)))
        if missed:
            print("      missed at lines: %s"
                  % ", ".join(str(x) for x in missed[:16]))


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
    parser.add_argument("--mutate", action="store_true",
                        help="report how many wait mutations the wait-sufficiency "
                             "check notices; run after changing that check")
    args = parser.parse_args()

    if args.mutate:
        for path in args.assembly:
            print("== mutation sensitivity: %s" % path)
            mutateWaits(path)
        return 0

    allFailures = []
    for path in args.assembly:
        print("== %s" % path)
        instructions = readInstructions(path)
        symbols = parseSymbols(path)
        ranges = residentRanges(path)
        iter0, iterN = splitRegions(instructions)

        # Wait sufficiency does not depend on RAP, so the non-RAP kernel in the
        # same build is worth checking too: it is untouched and validates PASSED,
        # so a violation reported there is a false positive in this check.
        allFailures += ["%s: %s" % (path, f)
                        for f in reportWaits(instructions, symbols, iter0, iterN)]

        if not ranges:
            print("  not a ReuseAcrossPersistent kernel; remaining checks skipped")
            continue

        if iter0 is None:
            print("  peel labels absent; region equivalence not applicable yet")
        else:
            allFailures += ["%s: %s" % (path, f) for f in checkRegions(path, args.mode, iter0, iterN)]
            labels = {label for _, label, _ in instructions if label}
            targetFailures = checkIterNLabelTargets(labels, iterN)
            print("  iterN label targets pointing into iter0: %d" % len(targetFailures))
            allFailures += ["%s: %s" % (path, f) for f in targetFailures]

        if not args.skip_resident_scan:
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
