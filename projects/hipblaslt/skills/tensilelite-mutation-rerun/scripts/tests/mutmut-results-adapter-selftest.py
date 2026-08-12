#!/usr/bin/env python3
"""Pure-Python selftest for mutmut-results-adapter.py (no subprocess, no node).

Uses a real fixture source file so ast enclosing-function resolution is genuinely
exercised. Run: python3 mutmut-results-adapter-selftest.py
"""
import importlib.util
import json
import os
import sys
import time

sys.excepthook = sys.__excepthook__

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
SRC_ROOT = FIX  # sample_mod.py lives here; survivor file is "sample_mod.py"
ADAPTER = os.path.join(HERE, "..", "mutmut-results-adapter.py")

spec = importlib.util.spec_from_file_location("mutmut_results_adapter", ADAPTER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
build_groups = mod.build_groups

fail = 0


def ok(m):
    print("ok   - " + m)


def bad(m):
    global fail
    print("BAD  - " + m)
    fail = 1


with open(os.path.join(FIX, "slice-sample-results.json")) as fh:
    fixture = json.load(fh)
opts = {
    "srcRoot": SRC_ROOT,
    "testDirBase": fixture["testDirBase"],
    "charDirMap": fixture["charDirMap"],
    "defaultCharDir": fixture["defaultCharDir"],
}

res = build_groups(fixture["survivors"], opts)
by_fn = {g["function"]: g for g in res["groups"]}

# --- enclosing-function resolution (AST) ---
(
    ok("AST grouped survivors into function parse")
    if "parse" in by_fn
    else bad("missing group: parse")
)
(
    ok("AST grouped survivors into function parse_all")
    if "parse_all" in by_fn
    else bad("missing group: parse_all")
)
(
    ok("AST qualified class method as Reader.parse")
    if "Reader.parse" in by_fn
    else bad("missing group: Reader.parse")
)
(
    ok("module-level survivor grouped as <module>")
    if "<module>" in by_fn
    else bad("missing group: <module>")
)
(
    ok("decorator-line survivor maps to cached_double (not <module>)")
    if (
        "cached_double" in by_fn
        and "sample_mod.cached_double__mutmut_deco"
        in by_fn["cached_double"]["survivors"]
    )
    else bad("decorator-line survivor mis-grouped")
)

# --- collision-free test_file ---
tfs = [g["test_file"] for g in res["groups"]]
(
    ok("test_file is collision-free across all groups")
    if len(set(tfs)) == len(tfs)
    else bad("test_file collision")
)
(
    ok("parse vs Reader.parse get distinct test_file")
    if (by_fn["parse"]["test_file"] != by_fn["Reader.parse"]["test_file"])
    else bad("parse / Reader.parse test_file collided")
)
(
    ok("test_file uses the SRC_REL-relative test-dir base")
    if (
        by_fn["parse"]["test_file"].startswith(
            "Tensile/Tests/unit/characterization/_generated/"
        )
    )
    else bad("test_file base wrong")
)

# --- survivor conservation ---
placed = sorted(m for g in res["groups"] for m in g["survivors"])
expected = sorted(
    s["mutant_id"] for s in fixture["survivors"] if s.get("status") != "no-test"
)
(
    ok("every survived mutant mapped exactly once (conservation)")
    if placed == expected
    else bad("conservation mismatch")
)
(
    ok("parse group has its 2 survivors")
    if len(by_fn["parse"]["survivors"]) == 2
    else bad("parse survivor count wrong")
)

# --- no-test separated + excluded ---
(
    ok("no-test record reported separately")
    if (
        len(res["no_test"]) == 1
        and res["no_test"][0]["mutant_id"] == "sample_mod.read__mutmut_notest"
    )
    else bad("no-test not separated")
)
all_ids = [m for g in res["groups"] for m in g["survivors"]]
(
    ok("no-test record excluded from triage groups")
    if "sample_mod.read__mutmut_notest" not in all_ids
    else bad("no-test leaked")
)

# --- source_file semantics: SRC_REL-relative, not absolute/prefixed ---
g = by_fn["parse"]
(
    ok("source_file is SRC_REL-relative (works for verify manifest)")
    if g["source_file"] == "sample_mod.py"
    else bad("source_file wrong: " + g["source_file"])
)
(
    ok("source_file not absolute/worktree-prefixed")
    if (SRC_ROOT not in g["source_file"] and not g["source_file"].startswith("/"))
    else bad("source_file leaked a prefix")
)

# --- char_dir string-or-list; no null/empty ---
(
    ok("heterogeneous covering set emits char_dir as a list")
    if (
        isinstance(by_fn["parse_all"]["char_dir"], list)
        and len(by_fn["parse_all"]["char_dir"]) == 2
    )
    else bad("parse_all char_dir not a list")
)
(
    ok("default covering set emits char_dir as a string")
    if isinstance(by_fn["parse"]["char_dir"], str)
    else bad("parse char_dir not a string")
)
(
    ok("no group has a null/empty char_dir")
    if all(
        g["char_dir"]
        and (
            len(g["char_dir"])
            if isinstance(g["char_dir"], list)
            else len(g["char_dir"])
        )
        for g in res["groups"]
    )
    else bad("empty char_dir emitted")
)


# --- missing char_dir + no default => clear error ---
def expect_error(fn, pat, label):
    try:
        fn()
    except ValueError as e:
        ok(label) if pat in str(e) else bad("%s: wrong error %r" % (label, e))
        return
    bad("%s: no error raised" % label)


expect_error(
    lambda: build_groups(
        [
            {
                "module": "m",
                "mutant_id": "x",
                "file": "sample_mod.py",
                "line": 10,
                "status": "survived",
            }
        ],
        {"srcRoot": SRC_ROOT},
    ),
    "no char_dir",
    "missing char_dir + no default throws a clear error",
)

# --- empty [] / "" char_dir treated as absent ---
for empty in ([], ""):
    expect_error(
        lambda e=empty: build_groups(
            [
                {
                    "module": "m",
                    "mutant_id": "x",
                    "file": "sample_mod.py",
                    "line": 10,
                    "status": "survived",
                }
            ],
            {"srcRoot": SRC_ROOT, "charDirMap": {"m::parse": e}},
        ),
        "no char_dir",
        "empty char_dir %r rejected" % (empty,),
    )
r_fb = build_groups(
    [
        {
            "module": "m",
            "mutant_id": "x",
            "file": "sample_mod.py",
            "line": 10,
            "status": "survived",
        }
    ],
    {"srcRoot": SRC_ROOT, "charDirMap": {"m::parse": []}, "defaultCharDir": "D"},
)
(
    ok("empty char_dir mapping falls back to defaultCharDir")
    if r_fb["groups"][0]["char_dir"] == "D"
    else bad("empty char_dir did not fall back")
)

# --- non-integer line rejected clearly ---
expect_error(
    lambda: build_groups(
        [
            {
                "module": "m",
                "mutant_id": "x",
                "file": "sample_mod.py",
                "line": "abc",
                "status": "survived",
            }
        ],
        {"srcRoot": SRC_ROOT, "defaultCharDir": "D"},
    ),
    "non-integer line",
    "non-integer line rejected with a clear error",
)

# --- missing source file degrades to <module>, returns promptly ---
t0 = time.time()
r_missing = build_groups(
    [
        {
            "module": "ghost",
            "mutant_id": "g1",
            "file": "does_not_exist.py",
            "line": 5,
            "status": "survived",
        }
    ],
    {"srcRoot": SRC_ROOT, "defaultCharDir": "D"},
)
dt = (time.time() - t0) * 1000
(
    ok("missing source file degrades to <module> (no crash/hang)")
    if (
        len(r_missing["groups"]) == 1
        and r_missing["groups"][0]["function"] == "<module>"
    )
    else bad("missing source file mis-handled")
)
(
    ok("resolution returns promptly (%dms)" % dt)
    if dt < 5000
    else bad("too slow: %dms" % dt)
)

# --- determinism ---
(
    ok("output is deterministic")
    if json.dumps(build_groups(fixture["survivors"], opts))
    == json.dumps(build_groups(fixture["survivors"], opts))
    else bad("non-deterministic output")
)

print("")
if fail == 0:
    print("ALL SELFTESTS PASSED")
    sys.exit(0)
else:
    print("SELFTESTS FAILED")
    sys.exit(1)
