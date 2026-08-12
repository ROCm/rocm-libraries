#!/usr/bin/env python3
"""Selftest for rank-modules.py (Issue 17). Pure: no mutmut, no source edits."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUT = os.path.join(HERE, "..", "rank-modules.py")

spec = importlib.util.spec_from_file_location("rank_modules", SUT)
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


def approx(a, b, eps=1e-9):
    return abs(a - b) <= eps


# --- weights are exactly the plan's and sum to 1.0 ---
check(mod.W_IMPORTERS == 0.40, "importers weight 0.40")
check(mod.W_CYCLOMATIC == 0.25, "cyclomatic weight 0.25")
check(mod.W_LOC == 0.15, "LOC weight 0.15")
check(mod.W_NO_TEST == 0.20, "no_test_fraction weight 0.20")
check(
    approx(mod.W_IMPORTERS + mod.W_CYCLOMATIC + mod.W_LOC + mod.W_NO_TEST, 1.0),
    "weights sum to 1.0",
)

# --- score() ---
check(approx(mod.score(1, 1, 1, 1), 1.0), "score(1,1,1,1) == 1.0")
check(approx(mod.score(0, 0, 0, 0), 0.0), "score(0,0,0,0) == 0.0")
check(approx(mod.score(1, 0, 0, 0), 0.40), "score isolates importers weight (0.40)")
check(approx(mod.score(0, 1, 0, 0), 0.25), "score isolates cyclomatic weight (0.25)")
check(approx(mod.score(0, 0, 1, 0), 0.15), "score isolates LOC weight (0.15)")
check(approx(mod.score(0, 0, 0, 1), 0.20), "score isolates no_test weight (0.20)")
# LOC enters LINEARLY (no log10): score is linear in loc_norm
check(
    approx(mod.score(0, 0, 0.5, 0), 0.075),
    "LOC contributes linearly (0.15*0.5), no log10",
)
check(
    approx(mod.score(0.5, 0.5, 0.5, 0.5), 0.5),
    "score is a convex combination (all 0.5 -> 0.5)",
)

# --- minmax_normalize ---
check(
    mod.minmax_normalize([10, 20, 30]) == [0.0, 0.5, 1.0],
    "minmax [10,20,30] -> [0,0.5,1]",
)
check(mod.minmax_normalize([5]) == [0.0], "minmax single element -> [0.0]")
check(
    mod.minmax_normalize([7, 7, 7]) == [0.0, 0.0, 0.0],
    "minmax all-equal (zero range) -> all 0.0",
)
check(
    all(0.0 <= v <= 1.0 for v in mod.minmax_normalize([3, 1, 4, 1, 5, 9, 2, 6])),
    "minmax output stays within [0,1]",
)

# --- disposition coverage: every discovered dir gets exactly one allowed disposition ---
root = mod._root()
dirs = mod.discover_char_dirs(root)
rows = mod.build_rows(root)
allowed = {mod.CERTIFIED, mod.SCHEDULED, mod.DEF_HV, mod.DEF_CG, mod.OOS}
check(len(dirs) > 0, "char dirs discovered (%d)" % len(dirs))
check(len(rows) == len(dirs), "one row per discovered dir (no dir silently absent)")
row_dirs = [r["dir"] for r in rows]
check(sorted(row_dirs) == sorted(dirs), "rows cover exactly the discovered dirs")
check(len(set(row_dirs)) == len(row_dirs), "no duplicate dir rows")
check(
    all(r["disposition"] in allowed for r in rows),
    "every disposition is one of the 5 allowed values",
)

# --- score/cyclomatic/no_test are PENDING (None), not faked, in this environment ---
check(
    all(r["score"] is None for r in rows),
    "every score is PENDING (not faked) while inputs are pending",
)
check(
    all(r["cyclomatic"] is None and r["no_test_fraction"] is None for r in rows),
    "cyclomatic + no_test_fraction are PENDING (not faked)",
)
# --- LOC/importers ARE computed for scheduled/certified dirs with a known module ---
# Expected LOC is derived from the current LibraryIO.py source rather than a
# frozen literal, so this self-test does not go stale as the production file
# evolves on develop.
_libio_src = os.path.join(root, mod.SRC_REL, "Tensile", "LibraryIO.py")
with open(_libio_src, "r", errors="replace") as _fh:
    _expected_libio_loc = sum(1 for _ in _fh)
li = next((r for r in rows if r["dir"] == "LibraryIO"), None)
check(
    li is not None
    and li["loc"] == _expected_libio_loc
    and li["loc"] > 0
    and isinstance(li["importers"], int),
    "LibraryIO LOC computed (== current source line count, %d) and importers computed (real)"
    % _expected_libio_loc,
)

# --- certified + scheduled are the plan's; deferred-coverage-gap is reserved (0) ---
disp_of = {r["dir"]: r["disposition"] for r in rows}
check(
    disp_of.get("CommonUtilities") == mod.CERTIFIED
    and disp_of.get("TensileLogic") == mod.CERTIFIED,
    "slice-1 dirs certified",
)
check(disp_of.get("LibraryIO") == mod.SCHEDULED, "LibraryIO scheduled (slice 2)")
check(
    sum(1 for r in rows if r["disposition"] == mod.DEF_CG) == 0,
    "deferred-coverage-gap has 0 members (reserved; needs measured no_test_fraction)",
)
# docs-only dir (no Python) -> out-of-scope (data-driven, not forced high-value)
check(disp_of.get("adr") == mod.OOS, "docs-only dir 'adr' classified out-of-scope")

# --- --metrics ingestion computes real min-max-normalized scores ---
metrics = {
    "LibraryIO": {"cyclomatic": 120, "no_test_fraction": 0.30},
    "Configuration": {"cyclomatic": 90, "no_test_fraction": 0.20},
    "ValidParameters": {"cyclomatic": 200, "no_test_fraction": 0.10},
}
mrows = mod.build_rows(root, metrics=metrics)
mby = {r["dir"]: r for r in mrows}
# only the 3 dirs with all four inputs get a score; others stay PENDING
scored = [r for r in mrows if r["score"] is not None]
check(
    set(r["dir"] for r in scored) == {"LibraryIO", "Configuration", "ValidParameters"},
    "score computed only for dirs with all four inputs present",
)
# ValidParameters is injected as max cyclomatic / min no_test; its expected score
# is derived from the live min-max normalization, not a frozen literal (no drift).
_sorder = [r for r in mrows if r["score"] is not None]
_imp_n = mod.minmax_normalize([r["importers"] for r in _sorder])
_cyc_n = mod.minmax_normalize([r["cyclomatic"] for r in _sorder])
_loc_n = mod.minmax_normalize([r["loc"] for r in _sorder])
_nt_n = mod.minmax_normalize([r["no_test_fraction"] for r in _sorder])
_vp_i = next(i for i, r in enumerate(_sorder) if r["dir"] == "ValidParameters")
_expected_vp = round(mod.score(_imp_n[_vp_i], _cyc_n[_vp_i], _loc_n[_vp_i], _nt_n[_vp_i]), 4)
check(
    approx(mby["ValidParameters"]["score"], _expected_vp),
    "computed score matches min-max+weight pipeline (ValidParameters -> %.4f)" % _expected_vp,
)
check(
    mby["LibraryIO"]["score"] < mby["ValidParameters"]["score"],
    "ranking orders ValidParameters above LibraryIO",
)
check(
    mby.get("Naming") is not None and mby["Naming"]["score"] is None,
    "a dir without provided metrics stays score=PENDING",
)

print("")
if fail == 0:
    print("ALL SELFTESTS PASSED")
    sys.exit(0)
print("SELFTESTS FAILED")
sys.exit(1)
