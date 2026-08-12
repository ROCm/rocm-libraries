#!/usr/bin/env python3
"""ci-mutant-regression.py - Issue 16: CI ladder regression gate.

Compares a killed-set BASELINE artifact to a CURRENT mutation-result artifact using
the stable mutant identity `(file, line_anchor, diff_hash)` from Issue 15
(mutant-identity.py) - NOT mutmut ordinal IDs. Implements the fail-on-regression
rung of the CI ladder (report-only -> fail-on-regression -> per-slice SLICE_FLOOR
-> allowlist ratchet); see coverage/mutprod/ci/mutation-ci-ladder.md.

Artifact schema (baseline and current), JSON:
  {"mutants": [ {"file": "...", "line_anchor": "...", "diff_hash": "...",
                 "status": "killed"} , ... ]}
Each entry may instead carry a raw "diff" (and omit line_anchor/diff_hash); this
tool then derives them via mutant-identity.py so both forms are accepted.

Statuses: killed | survived | no_test | suspicious | timeout | inconclusive |
          equivalent | pragma

Rules:
  - baseline killed, current == killed        -> OK
  - baseline killed, current in NON_KILLED    -> REGRESSION (fail-on-regression)
  - baseline killed, current absent           -> ABSENT (line changed/reformatted; report)
  - baseline equivalent, current == killed    -> REAUDIT (equivalence refuted; flag)
  - current key not in baseline               -> NEW (report; fail only with --fail-on-new)

Exit codes: 0 pass; 1 gate failure (regression, or --strict re-audit, or --fail-on-new).
No mutmut is run; no source edited. Python 3.8 compatible.
"""

import argparse
import importlib.util
import json
import os
import sys

NON_KILLED = ("survived", "no_test", "suspicious", "timeout", "inconclusive")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MI_PATH = os.path.join(_HERE, "mutant-identity.py")
_spec = importlib.util.spec_from_file_location("mutant_identity", _MI_PATH)
mi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mi)


def _load(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _entries(doc, which):
    if not isinstance(doc, dict) or not isinstance(doc.get("mutants"), list):
        raise ValueError("%s artifact must be an object with a 'mutants' array" % which)
    return doc["mutants"]


def entry_key(e):
    """Stable identity tuple, deriving from a raw diff if needed."""
    file_ = mi.normalize_path(e["file"])
    anchor = e.get("line_anchor")
    dhash = e.get("diff_hash")
    if (anchor is None or dhash is None) and e.get("diff"):
        if anchor is None:
            anchor = mi.original_line(e["diff"])
        if dhash is None:
            dhash = mi.diff_hash(e["diff"])
    if anchor is None or dhash is None:
        raise ValueError(
            "entry missing line_anchor/diff_hash (and no diff to derive them): %r" % e
        )
    return (file_, anchor.strip(), dhash)


def _severity(status):
    """Fail-closed ordering for duplicate-key aggregation and gating.

    killed=0 (best); equivalent/pragma=1; unknown/None/empty=2; NON_KILLED=3 (worst).
    """
    if status == "killed":
        return 0
    if status in ("equivalent", "pragma"):
        return 1
    if status in NON_KILLED:
        return 3
    return 2  # unknown / None / typo / empty -> treated as suspect, never benign


def _merge(dst, key, status):
    """Keep the WORST status per identity, so duplicate keys are order-independent
    and can never mask a survival with a later 'killed'."""
    if key not in dst or _severity(status) > _severity(dst[key]):
        dst[key] = status


def compare(baseline, current):
    """Return a classification dict of category -> list of (key, bstatus, cstatus).

    Rule (Issue 15): a baseline `killed` at the SAME identity is a REGRESSION if its
    current status is ANYTHING other than `killed` - survived/no_test/suspicious/
    timeout/inconclusive, but ALSO equivalent/pragma reclassification and any
    unknown/missing status (fail-closed; a kill must not be dropped by a typo or a
    silent relabel). An absent baseline-killed identity whose file also gains a new
    non-killed mutant is a suspected reformat-masked regression.
    """
    b = {}
    for e in _entries(baseline, "baseline"):
        _merge(b, entry_key(e), e.get("status"))
    c = {}
    for e in _entries(current, "current"):
        _merge(c, entry_key(e), e.get("status"))

    result = {
        "ok": [],
        "regression": [],
        "reaudit": [],
        "absent": [],
        "reformat_suspect": [],
        "new": [],
        "other": [],
    }
    new_nonkilled_files = set()
    for key, cstatus in c.items():
        if key not in b:
            result["new"].append((key, None, cstatus))
            if cstatus != "killed":
                new_nonkilled_files.add(key[0])
    for key, bstatus in b.items():
        cstatus = c.get(key)
        if key not in c:
            if bstatus == "killed" and key[0] in new_nonkilled_files:
                result["reformat_suspect"].append((key, bstatus, None))
            else:
                result["absent"].append((key, bstatus, None))
            continue
        if bstatus == "killed":
            if cstatus == "killed":
                result["ok"].append((key, bstatus, cstatus))
            else:
                result["regression"].append((key, bstatus, cstatus))
        elif bstatus == "equivalent":
            if cstatus == "killed":
                result["reaudit"].append((key, bstatus, cstatus))
            else:
                result["other"].append((key, bstatus, cstatus))
        else:
            result["other"].append((key, bstatus, cstatus))
    return result


def _fmt(key):
    return "%s :: %s :: %s" % key


def run(args):
    baseline = _load(args.baseline)
    current = _load(args.current)
    res = compare(baseline, current)

    def emit(cat, label):
        for key, b, c in res[cat]:
            print("%-11s %s (baseline=%s current=%s)" % (label, _fmt(key), b, c))

    emit("regression", "REGRESSION")
    emit("reformat_suspect", "REFORMAT?")
    emit("reaudit", "REAUDIT")
    emit("absent", "ABSENT")
    emit("new", "NEW")
    emit("other", "OTHER")
    print("")
    print(
        "summary: ok=%d regression=%d reformat_suspect=%d reaudit=%d absent=%d new=%d other=%d (stage=%s)"
        % (
            len(res["ok"]),
            len(res["regression"]),
            len(res["reformat_suspect"]),
            len(res["reaudit"]),
            len(res["absent"]),
            len(res["new"]),
            len(res["other"]),
            args.stage,
        )
    )

    if args.stage == "report-only":
        return 0
    # fail-on-regression rung: real regressions AND suspected reformat-masked
    # regressions fail (the latter forces an explicit re-baseline confirmation).
    fail = len(res["regression"]) > 0 or len(res["reformat_suspect"]) > 0
    if args.strict and res["reaudit"]:
        fail = True
    if args.fail_on_new and res["new"]:
        fail = True
    return 1 if fail else 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="CI mutation regression gate (stable identity)"
    )
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", required=True)
    p.add_argument(
        "--stage",
        choices=("report-only", "fail-on-regression"),
        default="fail-on-regression",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="also fail on equivalent->killed re-audit flags",
    )
    p.add_argument(
        "--fail-on-new",
        action="store_true",
        help="also fail on new (unbaselined) mutants",
    )
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
