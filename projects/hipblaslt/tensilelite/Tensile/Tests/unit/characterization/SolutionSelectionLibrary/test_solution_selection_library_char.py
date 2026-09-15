################################################################################
# Characterization tests for Tensile.SolutionSelectionLibrary
#
# ADD-ONLY: pins the pure selection-analysis helpers. The two Naming imports
# (getSolutionNameMin/getKernelNameMin) used by updateValidSolutions are stubbed
# at the module level — they require fully-derived Solution state which is out of
# scope here; we pin that updateValidSolutions *calls* them and stores results.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.SolutionSelectionLibrary")


class Sol:
    """Hashable (identity) mapping wrapper standing in for a Solution object."""

    def __init__(self, d):
        self._d = dict(d)

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v


def _mk_sol(mt0, mt1, lsu, gsu):
    return Sol({"MacroTile0": mt0, "MacroTile1": mt1, "GlobalSplitU": gsu, "WorkGroup": [0, 0, lsu]})


# ---------------------------------------------------------------------------
# getSummationKeys
# ---------------------------------------------------------------------------
def test_get_summation_keys():
    header = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "sum=10", "sum= 20 "]
    assert M.getSummationKeys(header) == [10, 20]


def test_get_summation_keys_empty_tail():
    header = ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]
    assert M.getSummationKeys(header) == []


# ---------------------------------------------------------------------------
# makeKey
# ---------------------------------------------------------------------------
def test_make_key():
    row = ["a", "b", "c", "base", " x ", " y ", " z "]
    assert M.makeKey(row) == "base_x_y_z"


# ---------------------------------------------------------------------------
# getSolutionBaseKey
# ---------------------------------------------------------------------------
def test_get_solution_base_key():
    sol = _mk_sol(64, 128, 2, 4)
    # "MacroTile0_MacroTile1_localSplitU_globalSplitU"
    assert M.getSolutionBaseKey(sol) == "64_128_2_4"


# ---------------------------------------------------------------------------
# updateIfGT
# ---------------------------------------------------------------------------
def test_update_if_gt_new_key():
    d = {}
    M.updateIfGT(d, "k", 5)
    assert d == {"k": 5}


def test_update_if_gt_greater_replaces():
    d = {"k": 5}
    M.updateIfGT(d, "k", 9)
    assert d["k"] == 9


def test_update_if_gt_not_greater_keeps():
    d = {"k": 5}
    M.updateIfGT(d, "k", 3)
    assert d["k"] == 5


# ---------------------------------------------------------------------------
# updateValidSolutions
# ---------------------------------------------------------------------------
def test_update_valid_solutions_included_branch():
    s = _mk_sol(64, 64, 1, 1)
    analyzer = [s]  # s already present -> "included" branch
    validSolutions = [(s, {"info": 1})]
    ids = M.updateValidSolutions(validSolutions, analyzer)
    # included solution gets its Ideals set and is not re-named
    assert s["Ideals"] == {"info": 1}
    assert 0 in ids


def test_update_valid_solutions_remainder_branch(monkeypatch):
    monkeypatch.setattr(M, "getSolutionNameMin", lambda sol, split: "SNAME")
    monkeypatch.setattr(M, "getKernelNameMin", lambda sol, split: "KNAME")
    s = _mk_sol(32, 32, 1, 1)
    analyzer = []  # empty -> s goes to remainder branch
    validSolutions = [(s, {"info": 2})]
    ids = M.updateValidSolutions(validSolutions, analyzer)
    assert s["SolutionNameMin"] == "SNAME"
    assert s["KernelNameMin"] == "KNAME"
    assert s["Ideals"] == {"info": 2}
    assert ids == [0]


# ---------------------------------------------------------------------------
# analyzeSolutionSelection (CSV-driven)
# ---------------------------------------------------------------------------
def test_analyze_solution_selection(tmp_path):
    problemType = {"TotalIndices": 2, "NumIndicesLD": 0}
    # summationIndex=2, numIndices=2, totalSizeIdx=3, solutionStartIdx=4
    # numSolutions=2 -> rowLength=6 -> data cols at indices 4 and 5
    s0 = _mk_sol(64, 64, 1, 1)
    s1 = _mk_sol(32, 32, 1, 2)
    solutions = [s0, s1]

    csv_path = tmp_path / "sel.csv"
    csv_path.write_text(
        "h0,h1,h2,h3,h4,h5\n"  # header (skipped)
        "x,y,SUM,z,1.0,2.0\n"  # row 1
        "x,y,SUM,z,3.0,0.5\n"  # row 2 -> s0 value rises to 3.0 (>valueOld branch)
    )

    result = M.analyzeSolutionSelection(
        problemType,
        [str(csv_path)],
        [2],            # numSolutionsPerGroup
        {},             # solutionGroupMap (unused)
        [solutions],    # solutionsList (one file)
    )

    by_sol = dict(result)
    # both solutions appear; s0's best for SUM is 3.0, s1's is 2.0
    assert by_sol[s0]["SUM"] == 3.0
    assert by_sol[s1]["SUM"] == 2.0
