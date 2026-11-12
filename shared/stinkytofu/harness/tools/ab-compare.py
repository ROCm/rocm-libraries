#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare the asm of two Tensile runs (baseline and candidate), ignoring only label names.

Usage:  python3 tools/ab-compare.py <baseline dir> <candidate dir>

For every *.s file under the two directories (matched by path relative to each directory):
  - raw compare
  - label-normalized compare: every distinct token matching label_[A-Za-z0-9_]+ is renamed LBL0, LBL1, ...
    in order of first appearance in that file. Nothing else is changed. A control-flow change still shows,
    because the numbering follows where each label first appears.
  - proof that only labels changed: masking every label token to one fixed token must give the same text
    before and after normalizing.
Exit code: 0 = identical after normalization (same set of files, no other difference),
           1 = a difference, a file missing on one side, or the label proof failed,
           2 = an error (no files found).
The label pattern is the one in use when this was written. If asm label names change, the baseline-against-
itself check (run this on two runs of the same build) will fail, and the pattern needs updating.
"""
import pathlib
import re
import sys

LABEL = re.compile(r"label_[A-Za-z0-9_]+")


def normalize(text):
    seen = {}
    return LABEL.sub(lambda m: seen.setdefault(m.group(0), "LBL%d" % len(seen)), text)


def mask(text):
    return LABEL.sub("LBL", text)


def mask_normalized(text):
    return re.sub(r"LBL\d+", "LBL", text)


def asm_files(root):
    root = pathlib.Path(root)
    return {str(p.relative_to(root)): p for p in root.rglob("*.s")}


if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)
a, b = asm_files(sys.argv[1]), asm_files(sys.argv[2])
if not a or not b:
    print("ERROR: zero files compared (no .s file under one of the directories)")
    sys.exit(2)

only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
for name in only_a:
    print("only in baseline: ", name)
for name in only_b:
    print("only in candidate:", name)

compared = raw_diff = norm_diff = proof_failed = 0
for name in sorted(set(a) & set(b)):
    ta, tb = a[name].read_text(), b[name].read_text()
    na, nb = normalize(ta), normalize(tb)
    proof = mask(ta) == mask_normalized(na) and mask(tb) == mask_normalized(nb)
    compared += 1
    raw_diff += ta != tb
    norm_diff += na != nb
    proof_failed += not proof
    print(
        f"{name[-70:]}: raw_differs={ta != tb} normalized_differs={na != nb} only_labels_changed_by_normalizing={proof}"
        f" labels={len(set(LABEL.findall(ta)))}/{len(set(LABEL.findall(tb)))}"
    )

missing = len(only_a) + len(only_b)
print(
    f"RESULT files_compared={compared} raw_diff={raw_diff} normalized_diff={norm_diff} "
    f"missing_on_one_side={missing} label_proof_failed={proof_failed}"
)
sys.exit(0 if (norm_diff == 0 and missing == 0 and proof_failed == 0) else 1)
