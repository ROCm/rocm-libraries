#!/usr/bin/env python3
"""Selftest for solution-subslice.py (Issue 14). Pure: no mutmut, no source edits."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUT = os.path.join(HERE, "..", "solution-subslice.py")
REGIONS = os.path.join(HERE, "fixtures", "solution-regions.json")

spec = importlib.util.spec_from_file_location("solution_subslice", SUT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

fail = 0


def ok(m):
    print("ok   - " + m)


def bad(m):
    global fail
    print("BAD  - " + m)
    fail = 1


def base_doc():
    return {
        "module": "Tensile/SolutionStructs/Solution.py",
        "src_rel": "projects/hipblaslt/tensilelite",
        "regions": [
            {
                "id": "r01",
                "start_line": 61,
                "end_line": 261,
                "loc": 201,
                "same_level_fence": True,
                "covering_test_selection": ["t/a"],
            },
            {
                "id": "r02",
                "start_line": 262,
                "end_line": 396,
                "loc": 135,
                "same_level_fence": True,
                "covering_test_selection": ["t/b"],
            },
        ],
    }


def expect_error(doc, needle, label):
    errs = mod.validate_doc(doc)
    if any(needle in e for e in errs):
        ok(label)
    else:
        bad("%s: expected error containing %r, got %r" % (label, needle, errs))


def check(cond, label):
    if cond:
        ok(label)
    else:
        bad(label)


# --- the real regions.json validates clean ---
real = mod._load(REGIONS)
errs = mod.validate_doc(real)
(
    ok("real regions.json validates clean")
    if not errs
    else bad("real regions.json errors: %r" % errs)
)

# --- proof region [61..261] present ---
ids = {r["id"]: r for r in real["regions"]}
r01 = ids.get("r01")
check(
    bool(r01) and r01["start_line"] == 61 and r01["end_line"] == 261,
    "proof region r01 [61..261] present",
)

# --- a clean synthetic doc validates ---
errs = mod.validate_doc(base_doc())
ok("clean synthetic doc validates") if not errs else bad("clean doc errors: %r" % errs)

# --- each validation rule bites ---
d = base_doc()
d["regions"][1]["start_line"] = 60
d["regions"][1]["loc"] = 60 - 396  # unordered + will also flag loc
expect_error(d, "not ordered", "detects unordered regions")

d = base_doc()
d["regions"][1]["start_line"] = 200  # overlaps r01 (61..261)
d["regions"][1]["loc"] = d["regions"][1]["end_line"] - 200 + 1
expect_error(d, "overlaps", "detects overlapping regions")

d = base_doc()
d["regions"][0]["start_line"] = "61"
expect_error(d, "start_line must be an integer", "detects non-integer start_line")

d = base_doc()
d["regions"][0]["loc"] = 999
expect_error(d, "!= end-start+1", "detects loc inconsistent with span")

d = base_doc()
d["regions"][0]["loc"] = -1
expect_error(d, "loc must be a positive integer", "detects non-positive loc")

d = base_doc()
del d["regions"][0]["same_level_fence"]
expect_error(d, "same_level_fence", "detects missing same_level_fence")

d = base_doc()
d["regions"][0]["covering_test_selection"] = []
expect_error(
    d, "covering_test_selection is empty", "detects empty covering_test_selection"
)

d = base_doc()
d["regions"][1]["id"] = "r01"
expect_error(d, "duplicate id", "detects duplicate region id")

d = base_doc()
d["regions"][0]["end_line"] = 50  # end < start
d["regions"][0]["loc"] = 50 - 61 + 1
expect_error(d, "< start_line", "detects end_line < start_line")

# bool is a subclass of int - must NOT be accepted as start_line
d = base_doc()
d["regions"][0]["start_line"] = True
expect_error(
    d, "start_line must be an integer", "detects bool start_line (not just int)"
)

# negative / zero start
d = base_doc()
d["regions"][0]["start_line"] = 0
d["regions"][0]["loc"] = d["regions"][0]["end_line"] - 0 + 1
expect_error(d, "start_line must be >= 1", "detects start_line < 1")

# covering_test_selection not a list
d = base_doc()
d["regions"][0]["covering_test_selection"] = "t/a"
expect_error(d, "must be a list", "detects non-list covering_test_selection")

# covering_test_selection with non-string / empty members (would emit an invalid config)
d = base_doc()
d["regions"][0]["covering_test_selection"] = [1, ""]
expect_error(
    d,
    "non-string or empty member",
    "detects non-string/empty covering_test_selection member",
)

# structural: doc not a dict
check(bool(mod.validate_doc("nope")), "detects non-dict doc")
# structural: missing top-level key
d = base_doc()
del d["module"]
expect_error(d, "missing top-level key: module", "detects missing top-level module key")

# emit-config raises a clean KeyError (not a silent success) for an unknown region id
try:
    mod._find_region(real, "does-not-exist")
    bad("emit: should raise for unknown region id")
except KeyError:
    ok("emit: unknown region id raises KeyError (handled to clean message in CLI)")

# build_region_config rejects non-string covering members (no invalid config)
try:
    br = dict(r01)
    br["covering_test_selection"] = [1]
    mod.build_region_config(real, br)
    bad("emit: should raise on non-string covering member")
except ValueError:
    ok("emit: raises on non-string covering_test_selection member")

# --- build_region_config emits a valid mutation-rerun slice record ---
cfg = mod.build_region_config(real, r01)
check(
    bool(cfg.get("only_mutate")) and isinstance(cfg["only_mutate"], list),
    "emit: only_mutate non-empty list",
)
check(
    bool(cfg.get("test_selection")) and isinstance(cfg["test_selection"], list),
    "emit: test_selection non-empty list",
)
check(
    cfg["only_mutate"] == ["Tensile/SolutionStructs/Solution.py"],
    "emit: only_mutate is the whole file (mutmut has no sub-file targeting)",
)
check(
    "region" in cfg
    and cfg["region"]["start_line"] == 61
    and cfg["region"]["end_line"] == 261,
    "emit: region metadata carried",
)
check(
    "fence_gate" in cfg and "PragmaParseError" in cfg["fence_gate"],
    "emit: mandatory PragmaVisitor dry-parse gate hook present",
)
check(
    "source_safety" in cfg and "diff --quiet" in cfg["source_safety"],
    "emit: source-safety/restore requirement present",
)
check(cfg["slice_id"] == "10-r01", "emit: slice_id derived (10-r01)")

# --- emit-config raises on an empty covering set (would be an invalid mutmut config) ---
try:
    bad_reg = dict(r01)
    bad_reg["covering_test_selection"] = []
    mod.build_region_config(real, bad_reg)
    bad("emit: should raise on empty covering_test_selection")
except ValueError:
    ok("emit: raises on empty covering_test_selection (no invalid config)")

print("")
if fail == 0:
    print("ALL SELFTESTS PASSED")
    sys.exit(0)
print("SELFTESTS FAILED")
sys.exit(1)
