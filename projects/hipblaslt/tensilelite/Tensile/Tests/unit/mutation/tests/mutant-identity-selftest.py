#!/usr/bin/env python3
"""Selftest for mutant-identity.py (Issue 15). Pure: no mutmut, no source edits."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUT = os.path.join(HERE, "..", "mutant-identity.py")
REAL_DIFF = os.path.join(HERE, "fixtures", "inc_7.diff")

spec = importlib.util.spec_from_file_location("mutant_identity", SUT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

fail = 0


def ok(m):
    print("ok   - " + m)


def bad(m):
    global fail
    print("BAD  - " + m)
    fail = 1


def check(cond, label):
    ok(label) if cond else bad(label)


CHANGE = (
    "-        self.index = (self.index + 1) % len(self.chars)\n"
    "+        self.index = (self.index - 1) % len(self.chars)\n"
)

# variant A: canonical git diff
DIFF_A = (
    "diff --git a/Tensile/Common/Utilities.py b/Tensile/Common/Utilities.py\n"
    "index 685ce17c5e5..b9eb22740c3 100644\n"
    "--- a/Tensile/Common/Utilities.py\n"
    "+++ b/Tensile/Common/Utilities.py\n"
    "@@ -218,7 +218,7 @@ class SpinnyThing:\n"
    "         sys.stdout.flush()\n" + CHANGE
)
# variant B: SAME change, DIFFERENT mutmut ordinal header + index blob hashes +
# hunk line numbers (line move) + different surrounding context line
DIFF_B = (
    "# mutant xǁSpinnyThingǁincrement__mutmut_7\n"
    "diff --git a/Tensile/Common/Utilities.py b/Tensile/Common/Utilities.py\n"
    "index deadbeef000..cafef00d111 100644\n"
    "--- a/Tensile/Common/Utilities.py\n"
    "+++ b/Tensile/Common/Utilities.py\n"
    "@@ -404,7 +404,7 @@ class Relocated:\n"
    "         some_other_context_line()\n" + CHANGE
)

check(
    mod.diff_hash(DIFF_A) == mod.diff_hash(DIFF_B),
    "ordinal header + index hashes + hunk numbers + nearby context do NOT change the hash",
)

# semantic change differs
DIFF_SEM = DIFF_A.replace("self.index - 1", "self.index - 2")
check(
    mod.diff_hash(DIFF_A) != mod.diff_hash(DIFF_SEM),
    "a real semantic change (-1 vs -2) DOES change the hash",
)

# leading indentation is semantic -> changes the hash
DIFF_INDENT = (
    "@@ -1,1 +1,1 @@\n"
    "-        self.index = (self.index + 1) % len(self.chars)\n"
    "+            self.index = (self.index - 1) % len(self.chars)\n"
)
DIFF_INDENT_BASE = "@@ -1,1 +1,1 @@\n" + CHANGE
check(
    mod.diff_hash(DIFF_INDENT) != mod.diff_hash(DIFF_INDENT_BASE),
    "a leading-indentation change DOES change the hash (indentation is semantic)",
)

# trailing whitespace + CRLF normalized -> same hash
DIFF_WS = (
    "@@ -1,1 +1,1 @@\r\n"
    "-        self.index = (self.index + 1) % len(self.chars)   \r\n"
    "+        self.index = (self.index - 1) % len(self.chars)\t\r\n"
)
check(
    mod.diff_hash(DIFF_WS) == mod.diff_hash(DIFF_INDENT_BASE),
    "trailing whitespace and CRLF are normalized (same hash)",
)

# canonical body contains only the changed lines (no headers/context)
canon = mod.canonicalize_diff(DIFF_B)
check("__mutmut_" not in canon, "canonical body drops the ordinal header")
_cl = canon.splitlines()
check(
    all(
        not l.startswith("diff --git")
        and not l.startswith("index ")
        and not l.startswith("--- ")
        and not l.startswith("+++ ")
        and not l.startswith("@@")
        for l in _cl
    ),
    "canonical body drops git/index/file/hunk header lines",
)
check("some_other_context_line" not in canon, "canonical body excludes context lines")
check(
    canon == "-        self.index = (self.index + 1) % len(self.chars)\n"
    "+        self.index = (self.index - 1) % len(self.chars)",
    "canonical body is exactly the two changed lines (indentation preserved)",
)

# original_line + identity
check(
    mod.original_line(DIFF_A) == "self.index = (self.index + 1) % len(self.chars)",
    "original_line returns the stripped mutated line",
)
ident = mod.identity("Tensile/Common/Utilities.py", DIFF_A, line_number=220)
check(ident["file"] == "Tensile/Common/Utilities.py", "identity carries the file path")
check(
    ident["line_anchor"] == "self.index = (self.index + 1) % len(self.chars)",
    "identity line_anchor is the stripped original line (survives line moves)",
)
check(len(ident["diff_hash"]) == 16, "identity diff_hash is a 16-char hex prefix")
# key_tuple excludes the (unstable) line number
check(
    "line_number"
    not in dict(zip(("file", "line_anchor", "diff_hash"), mod.key_tuple(ident))),
    "CI key_tuple excludes line_number",
)
check(
    mod.key_tuple(ident)
    == (
        "Tensile/Common/Utilities.py",
        "self.index = (self.index + 1) % len(self.chars)",
        ident["diff_hash"],
    ),
    "key_tuple = (file, line_anchor, diff_hash)",
)

# BUG-1 regression: a hunk change line whose CONTENT starts with -- / ++ must be
# KEPT (not mistaken for a git file header). Only pre-hunk ---/+++ are headers.
DIFF_DASHDASH = (
    "diff --git a/f.py b/f.py\n"
    "index aaa..bbb 100644\n"
    "--- a/f.py\n"
    "+++ b/f.py\n"
    "@@ -1,1 +1,1 @@\n"
    "--- header comment removed\n"
    "+++ header comment added\n"
)
canon_dd = mod.canonicalize_diff(DIFF_DASHDASH)
check(
    canon_dd == "--- header comment removed\n+++ header comment added",
    "hunk change lines starting with --/++ are kept, not dropped as headers (BUG-1)",
)
check(
    mod.diff_hash(DIFF_DASHDASH) != mod.diff_hash("@@ -1,1 +1,1 @@\n"),
    "such a diff does not collapse to the empty-string hash",
)

# pure-insertion mutant: original_line falls back to the first added line
DIFF_INS = "@@ -1,0 +2,1 @@\n+    logging.info('added')\n"
check(
    mod.original_line(DIFF_INS) == "logging.info('added')",
    "pure-insertion mutant: original_line falls back to the inserted line",
)

# real slice-1 artifact
if os.path.isfile(REAL_DIFF):
    with open(REAL_DIFF) as fh:
        real = fh.read()
    h1 = mod.diff_hash(real)
    # prepend a mutmut ordinal header + change index hashes -> hash must be identical
    real2 = "# mutant xǁSpinnyThingǁincrement__mutmut_99\n" + real.replace(
        "685ce17c5e5", "0000000"
    )
    check(
        mod.diff_hash(real) == mod.diff_hash(real2),
        "real slice-1 inc_7.diff hash is stable under ordinal/index changes",
    )
    check(len(h1) == 16, "real diff hashes to a 16-char id (%s)" % h1)
else:
    print("ok   - (real inc_7.diff not present; skipped real-artifact check)")

print("")
if fail == 0:
    print("ALL SELFTESTS PASSED")
    sys.exit(0)
print("SELFTESTS FAILED")
sys.exit(1)
