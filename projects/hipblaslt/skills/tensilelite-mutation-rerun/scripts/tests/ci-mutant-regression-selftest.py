#!/usr/bin/env python3
"""Selftest for ci-mutant-regression.py (Issue 16). Pure: no mutmut, no source edits."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUT = os.path.join(HERE, "..", "ci-mutant-regression.py")

spec = importlib.util.spec_from_file_location("ci_mutant_regression", SUT)
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


def M(file, anchor, dhash, status, diff=None):
    e = {"file": file, "status": status}
    if diff is not None:
        e["diff"] = diff
    else:
        e["line_anchor"] = anchor
        e["diff_hash"] = dhash
    return e


def art(*mutants):
    return {"mutants": list(mutants)}


F = "Tensile/Common/Utilities.py"

# --- compare() classification ---
base = art(
    M(F, "a = 1", "h_a", "killed"),
    M(F, "b = 2", "h_b", "killed"),
    M(F, "c = 3", "h_c", "killed"),
    M(F, "d = 4", "h_d", "killed"),
    M(F, "e = 5", "h_e", "equivalent"),
    M(F, "g = 7", "h_g", "killed"),  # will be absent in current
)
cur = art(
    M(F, "a = 1", "h_a", "killed"),  # killed -> killed
    M(F, "b = 2", "h_b", "survived"),  # killed -> survived
    M(F, "c = 3", "h_c", "no_test"),  # killed -> no_test
    M(F, "d = 4", "h_d", "inconclusive"),  # killed -> inconclusive
    M(F, "e = 5", "h_e", "killed"),  # equivalent -> killed (re-audit)
    M(F, "z = 9", "h_z", "survived"),  # new mutant
)
res = mod.compare(base, cur)
keys = lambda cat: set(k[1] for (k, b, c) in res[cat])  # by anchor for readability

check("a = 1" in keys("ok"), "killed -> killed classified ok")
check("b = 2" in keys("regression"), "killed -> survived classified regression")
check("c = 3" in keys("regression"), "killed -> no_test classified regression")
check("d = 4" in keys("regression"), "killed -> inconclusive classified regression")
check("e = 5" in keys("reaudit"), "equivalent -> killed classified reaudit")
check("z = 9" in keys("new"), "unbaselined mutant classified new")
# g=7 (killed, absent) + z=9 (new survivor, same file) -> reformat_suspect, not silent absent
check(
    "g = 7" in keys("reformat_suspect"),
    "absent killed + new same-file survivor -> reformat_suspect",
)
check(len(res["regression"]) == 3, "exactly 3 regressions")

# killed -> timeout / suspicious also regressions
res2 = mod.compare(
    art(M(F, "t=1", "h_t", "killed"), M(F, "s=1", "h_s", "killed")),
    art(M(F, "t=1", "h_t", "timeout"), M(F, "s=1", "h_s", "suspicious")),
)
check(
    len(res2["regression"]) == 2,
    "killed -> timeout and killed -> suspicious are regressions",
)

# raw-diff derivation: an entry with only a diff keys the same as a precomputed one
DIFF = (
    "@@ -1,1 +1,1 @@\n"
    "-        self.index = (self.index + 1) % len(self.chars)\n"
    "+        self.index = (self.index - 1) % len(self.chars)\n"
)
import importlib.util as _u  # reuse mutant-identity for the expected hash

_mi_spec = _u.spec_from_file_location(
    "mutant_identity", os.path.join(HERE, "..", "mutant-identity.py")
)
_mi = _u.module_from_spec(_mi_spec)
_mi_spec.loader.exec_module(_mi)
expect_hash = _mi.diff_hash(DIFF)
expect_anchor = _mi.original_line(DIFF)
b_raw = art({"file": F, "status": "killed", "diff": DIFF})
c_pre = art(M(F, expect_anchor, expect_hash, "survived"))
res3 = mod.compare(b_raw, c_pre)
check(
    len(res3["regression"]) == 1,
    "raw-diff baseline entry keys identically to a precomputed current entry (regression matched)",
)

# --- fail-closed: killed -> anything-not-killed (incl reclassify/unknown/empty/missing) ---
for st in ("equivalent", "pragma", "survivedd", "SURVIVED", "", None):
    be = M(F, "k = 1", "h_k", "killed")
    ce = {"file": F, "line_anchor": "k = 1", "diff_hash": "h_k"}
    if st is not None:
        ce["status"] = st
    r = mod.compare(art(be), {"mutants": [ce]})
    check(
        len(r["regression"]) == 1, "killed -> %r is a regression (fail-closed)" % (st,)
    )

# --- duplicate identity keys: worst-status wins, order-independent ---
dupbase = art(M(F, "x = 1", "h_x", "killed"))
dup_ks = {"mutants": [M(F, "x = 1", "h_x", "killed"), M(F, "x = 1", "h_x", "survived")]}
dup_sk = {"mutants": [M(F, "x = 1", "h_x", "survived"), M(F, "x = 1", "h_x", "killed")]}
check(
    len(mod.compare(dupbase, dup_ks)["regression"]) == 1
    and len(mod.compare(dupbase, dup_sk)["regression"]) == 1,
    "duplicate identity keys aggregate to worst status (order-independent regression)",
)

# --- reformat-masked regression: absent killed + new non-killed in SAME file ---
rb = art(M(F, "self.index = (self.index + 1) % len(self.chars)", "h_r", "killed"))
rc = art(
    M(F, "self.index = ( self.index + 1 ) % len(self.chars)", "h_r2", "survived")
)  # respaced -> new key
res_r = mod.compare(rb, rc)
check(
    len(res_r["reformat_suspect"]) == 1 and len(res_r["absent"]) == 0,
    "absent killed + new same-file survivor -> reformat_suspect (not silent absent)",
)
res_a = mod.compare(rb, art(M("Tensile/Other.py", "n = 1", "h_n", "survived")))
check(
    len(res_a["absent"]) == 1 and len(res_a["reformat_suspect"]) == 0,
    "absent killed with no same-file new survivor stays 'absent' (report)",
)

# --- exit codes via the CLI ---
_TMPDIR = tempfile.mkdtemp()


def write(doc):
    import uuid  # unique name; cleaned with _TMPDIR

    p = os.path.join(_TMPDIR, "a-%s.json" % os.urandom(6).hex())
    with open(p, "w") as f:
        json.dump(doc, f)
    return p


def cli(baseline, current, *extra):
    r = subprocess.run(
        [
            sys.executable,
            SUT,
            "--baseline",
            write(baseline),
            "--current",
            current if isinstance(current, str) else write(current),
        ]
        + list(extra),
        capture_output=True,
        text=True,
    )
    return r.returncode


clean_base = art(M(F, "a = 1", "h_a", "killed"))
clean_cur = art(M(F, "a = 1", "h_a", "killed"))
reg_cur = art(M(F, "a = 1", "h_a", "survived"))
eqbase = art(M(F, "e = 5", "h_e", "equivalent"))
eqcur = art(M(F, "e = 5", "h_e", "killed"))
newbase = art(M(F, "a = 1", "h_a", "killed"))
newcur = art(M(F, "a = 1", "h_a", "killed"), M(F, "n = 9", "h_n", "survived"))

check(
    cli(clean_base, clean_cur, "--stage", "fail-on-regression") == 0,
    "CLI: killed->killed exit 0",
)
check(
    cli(clean_base, reg_cur, "--stage", "fail-on-regression") == 1,
    "CLI: killed->survived exit 1",
)
check(
    cli(clean_base, reg_cur, "--stage", "report-only") == 0,
    "CLI: report-only never fails (exit 0) even with a regression",
)
check(
    cli(eqbase, eqcur, "--stage", "fail-on-regression") == 0,
    "CLI: equivalent->killed is non-fatal by default (exit 0)",
)
check(
    cli(eqbase, eqcur, "--stage", "fail-on-regression", "--strict") == 1,
    "CLI: --strict fails on re-audit (exit 1)",
)
check(
    cli(newbase, newcur, "--stage", "fail-on-regression") == 0,
    "CLI: new mutant reported, not failed by default (exit 0)",
)
check(
    cli(newbase, newcur, "--stage", "fail-on-regression", "--fail-on-new") == 1,
    "CLI: --fail-on-new fails on new mutant (exit 1)",
)
check(
    cli(
        art(M(F, "a = 1", "h_a", "killed")),
        art(M(F, "a = 1", "h_a", "equivalent")),
        "--stage",
        "fail-on-regression",
    )
    == 1,
    "CLI: killed->equivalent declassification exit 1",
)
check(
    cli(rb, rc, "--stage", "fail-on-regression") == 1,
    "CLI: reformat-masked regression exit 1",
)

import shutil

shutil.rmtree(_TMPDIR, ignore_errors=True)

print("")
if fail == 0:
    print("ALL SELFTESTS PASSED")
    sys.exit(0)
print("SELFTESTS FAILED")
sys.exit(1)
