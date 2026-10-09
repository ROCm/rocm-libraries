################################################################################
# Characterization tests for Tensile.Utilities.merge
#
# ADD-ONLY: pins the library-logic merge tooling. Logic files are list-indexed:
#   data[4]  = header ProblemType (dict)
#   data[5]  = solutions (list of dicts; each has SolutionIndex/SolutionNameMin)
#   data[7]  = size map: list of [size(list), [solutionIndex, efficiency]]
#   data[11] = attribute string ("Equality"/"GridBased")
# Other indices are metadata placeholders.
################################################################################
import importlib

import pytest
import yaml

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.Utilities.merge")


def _sol(idx, name="s", mfma=True, **extra):
    d = {"SolutionIndex": idx, "SolutionNameMin": name, "EnableMatrixInstruction": mfma}
    d.update(extra)
    return d


def _logic(solutions, sizemap, header=None, attr="Equality"):
    data = [None] * 12
    data[4] = header if header is not None else {"DataType": "s"}
    data[5] = solutions
    data[7] = sizemap
    data[11] = attr
    return data


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    # default verbosity is 1; keep tests quiet & deterministic
    monkeypatch.setattr(M, "verbosity", 1)


# ---------------------------------------------------------------------------
# ensurePath / allFiles
# ---------------------------------------------------------------------------
def test_ensure_path(tmp_path):
    p = tmp_path / "x" / "y"
    assert M.ensurePath(str(p)) == str(p)
    assert p.is_dir()
    # idempotent (exists -> no makedirs)
    assert M.ensurePath(str(p)) == str(p)


def test_all_files_recurses_into_yaml_named_dir(tmp_path):
    (tmp_path / "a.yaml").write_text("1")
    (tmp_path / "ignore.txt").write_text("x")
    # a directory whose name ends in .yaml triggers the recursion branch
    sub = tmp_path / "sub.yaml"
    sub.mkdir()
    (sub / "b.yaml").write_text("2")
    files = M.allFiles(str(tmp_path))
    names = sorted(f.split("/")[-1] for f in files)
    assert names == ["a.yaml", "b.yaml"]


# ---------------------------------------------------------------------------
# reindexSolutions
# ---------------------------------------------------------------------------
def test_reindex_solutions():
    data = _logic([_sol(9), _sol(4)], [])
    M.reindexSolutions(data)
    assert [s["SolutionIndex"] for s in data[5]] == [0, 1]


# ---------------------------------------------------------------------------
# fixSizeInconsistencies
# ---------------------------------------------------------------------------
def test_fix_size_short_format_unchanged():
    sizes = [[[128, 128, 1, 64], [0, 0.5]]]
    out, n = M.fixSizeInconsistencies(sizes, "base")
    assert n == 1
    assert out[0][0] == [128, 128, 1, 64]


def test_fix_size_trims_long_format():
    sizes = [[[1, 2, 3, 4, 5, 6, 7, 8], [0, 0.5]]]  # len 8 -> trim last 4
    out, n = M.fixSizeInconsistencies(sizes, "base")
    assert out[0][0] == [1, 2, 3, 4]


def test_fix_size_removes_duplicate_after_trim():
    # two len-8 sizes that trim to the same prefix -> second is a duplicate
    sizes = [
        [[1, 2, 3, 4, 9, 9, 9, 9], [0, 0.5]],
        [[1, 2, 3, 4, 8, 8, 8, 8], [1, 0.6]],
    ]
    out, n = M.fixSizeInconsistencies(sizes, "base")
    assert n == 1


# ---------------------------------------------------------------------------
# cmpHelper / addKernel
# ---------------------------------------------------------------------------
def test_cmp_helper_strips_index_and_name():
    out = M.cmpHelper({"SolutionIndex": 3, "SolutionNameMin": "x", "A": 1})
    assert out == {"A": 1}


def test_add_kernel_reuses_existing():
    pool = [_sol(0, "s0", A=1)]
    pool, idx = M.addKernel(pool, _sol(99, "different-name", A=1))
    assert idx == 0  # matched by cmpHelper (index/name ignored)
    assert len(pool) == 1


def test_add_kernel_adds_new():
    pool = [_sol(0, "s0", A=1)]
    pool, idx = M.addKernel(pool, _sol(0, "s1", A=2))
    assert idx == 1
    assert pool[1]["SolutionIndex"] == 1


# ---------------------------------------------------------------------------
# sanitizeSolutions
# ---------------------------------------------------------------------------
def test_sanitize_removes_problemtype_and_zeroes_stagger():
    sols = [{"ProblemType": {"x": 1}, "StaggerU": 0}]
    M.sanitizeSolutions(sols)
    assert "ProblemType" not in sols[0]
    assert sols[0]["StaggerUMapping"] == 0
    assert sols[0]["StaggerUStride"] == 0
    assert sols[0]["_staggerStrideShift"] == 0


def test_sanitize_nonzero_stagger_untouched():
    sols = [{"StaggerU": 4}]
    M.sanitizeSolutions(sols)
    assert "StaggerUMapping" not in sols[0]


# ---------------------------------------------------------------------------
# removeUnusedKernels
# ---------------------------------------------------------------------------
def test_remove_unused_kernels():
    sols = [_sol(0, "used"), _sol(1, "unused")]
    sizemap = [[[1, 2, 3, 4], [0, 0.5]]]  # only solution 0 in use
    data = _logic(sols, sizemap)
    out, removed = M.removeUnusedKernels(data)
    assert removed == 1
    assert len(out[5]) == 1
    assert out[5][0]["SolutionIndex"] == 0
    # size map reindexed to the surviving solution
    assert out[7][0][1][0] == 0


# ---------------------------------------------------------------------------
# loadData
# ---------------------------------------------------------------------------
def test_load_data_ok(tmp_path):
    f = tmp_path / "d.yaml"
    f.write_text(yaml.safe_dump([1, 2, 3]))
    assert M.loadData(str(f)) == [1, 2, 3]


def test_load_data_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        M.loadData(str(tmp_path / "nope.yaml"))


# ---------------------------------------------------------------------------
# compareDestFolderToYaml
# ---------------------------------------------------------------------------
def test_compare_dest_folder_ok():
    data = _logic([], [], attr="Equality")
    # destFolder "Equality" matches attribute -> no exit
    M.compareDestFolderToYaml("/some/path/Equality", "inc.yaml", data)


def test_compare_dest_folder_noncheck_folder_passes():
    data = _logic([], [], attr="GridBased")
    M.compareDestFolderToYaml("/some/path/Other", "inc.yaml", data)


def test_compare_dest_folder_mismatch_exits():
    data = _logic([], [], attr="GridBased")
    with pytest.raises(SystemExit):
        M.compareDestFolderToYaml("/some/path/Equality", "inc.yaml", data)


def test_compare_dest_folder_empty_attr_exits():
    data = _logic([], [], attr="")
    with pytest.raises(SystemExit):
        M.compareDestFolderToYaml("/some/path/Equality", "inc.yaml", data)


# ---------------------------------------------------------------------------
# compareProblemType
# ---------------------------------------------------------------------------
def test_compare_problem_type_waives_and_matches():
    ori = _logic([{"ProblemType": {"A": 1, "B": 2}, "SolutionIndex": 0}], [],
                 header={"A": 1, "B": 2})
    inc = _logic([{"ProblemType": {"A": 1}, "SolutionIndex": 0}], [], header={"A": 1})
    # B is waived (absent from inc header) -> popped, remainder matches -> no exit
    M.compareProblemType(ori, inc)
    assert "B" not in ori[4]


def test_compare_problem_type_mismatch_exits():
    ori = _logic([], [], header={"A": 1})
    inc = _logic([{"ProblemType": {"A": 2}}], [], header={"A": 2})
    with pytest.raises(SystemExit):
        M.compareProblemType(ori, inc)


# ---------------------------------------------------------------------------
# defaultForceMergePolicy
# ---------------------------------------------------------------------------
def test_default_force_merge_policy():
    assert M.defaultForceMergePolicy("/x/arcturus/foo.yaml") is False
    assert M.defaultForceMergePolicy("/x/gfx942/foo.yaml") is True


# ---------------------------------------------------------------------------
# msg / verbose / debug verbosity gating
# ---------------------------------------------------------------------------
def test_verbose_gated(monkeypatch, capsys):
    monkeypatch.setattr(M, "verbosity", 0)
    M.verbose("hidden")
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(M, "verbosity", 1)
    M.verbose("shown")
    assert "shown" in capsys.readouterr().out


def test_debug_gated(monkeypatch, capsys):
    monkeypatch.setattr(M, "verbosity", 1)
    M.debug("hidden")
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(M, "verbosity", 2)
    M.debug("shown")
    assert "shown" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tag enums
# ---------------------------------------------------------------------------
def test_tag_str_repr():
    assert str(M.MfmaTag.MFMA) == "MFMA"
    assert repr(M.MfmaTag.VALU) == "VALU"
    assert str(M.AlphaValueTag.NEG_ONE) == "Alpha=-1"
    assert repr(M.AlphaValueTag.ZERO) == "Alpha=0"
    assert str(M.BetaValueTag.ONE) == "Beta=1"
    assert repr(M.BetaValueTag.ANY) == "Beta=Any"
    assert str(M.CEqualsDTag.C_EQ_D) == "C=D"
    assert repr(M.CEqualsDTag.C_NEQ_D) == "C!=D"


def test_str_to_scalar_value_tag():
    assert M.strToScalarValueTag(M.AlphaValueTag, "Any") == M.AlphaValueTag.ANY
    assert M.strToScalarValueTag(M.AlphaValueTag, 1) == M.AlphaValueTag.ONE
    assert M.strToScalarValueTag(M.BetaValueTag, -1) == M.BetaValueTag.NEG_ONE
    assert M.strToScalarValueTag(M.BetaValueTag, 0) == M.BetaValueTag.ZERO
    with pytest.raises(RuntimeError):
        M.strToScalarValueTag(M.AlphaValueTag, 7)


# ---------------------------------------------------------------------------
# getSolutionTag / findSolutionWithIndex / tag key helpers
# ---------------------------------------------------------------------------
def test_get_solution_tag():
    assert M.getSolutionTag({"EnableMatrixInstruction": True}) == (M.MfmaTag.MFMA,)
    assert M.getSolutionTag({"MatrixInstruction": [1]}) == (M.MfmaTag.MFMA,)
    assert M.getSolutionTag({}) == (M.MfmaTag.VALU,)


def test_find_solution_with_index_direct_and_search():
    sols = [_sol(0), _sol(1)]
    assert M.findSolutionWithIndex(sols, 1)["SolutionIndex"] == 1
    # out-of-position: index 5 lives at a different list slot
    shuffled = [_sol(5), _sol(0)]
    assert M.findSolutionWithIndex(shuffled, 0)["SolutionIndex"] == 0


def test_add_and_remove_solution_tag_from_keys():
    pool = [_sol(0, mfma=True)]
    solmap = [[[128, 128, 1, 64], [0, 0.5]]]
    tagged = M.addSolutionTagToKeys(solmap, pool)
    # the tag is prepended as a single tuple element
    assert tagged[0][0][0] == (M.MfmaTag.MFMA,)
    assert tagged[0][0][1:] == [128, 128, 1, 64]
    untagged = M.removeSolutionTagFromKeys(tagged)
    assert untagged[0][0] == [128, 128, 1, 64]


# ---------------------------------------------------------------------------
# findFastestCompatibleSolution (standalone; full 4-tuple tag contract)
# ---------------------------------------------------------------------------
def test_find_fastest_compatible_solution():
    T = (M.MfmaTag.MFMA, M.AlphaValueTag.ONE, M.BetaValueTag.ONE, M.CEqualsDTag.C_EQ_D)
    sizeMapping = (T, 128, 128, 1, 64)
    # a compatible (Alpha=ANY) entry with higher eff should win
    compatTag = (M.MfmaTag.MFMA, M.AlphaValueTag.ANY, M.BetaValueTag.ONE, M.CEqualsDTag.C_EQ_D)
    origDict = {
        (T,) + sizeMapping[1:]: [0, 0.4],
        (compatTag,) + sizeMapping[1:]: [1, 0.9],
    }
    assert M.findFastestCompatibleSolution(origDict, sizeMapping) == 0.9


# ---------------------------------------------------------------------------
# mergeLogic — improve / reject / new-size / forceMerge
# ---------------------------------------------------------------------------
def _base_inc(ori_eff, inc_eff, same_size=True):
    ori = _logic([_sol(0, "s0")], [[[128, 128, 1, 64], [0, ori_eff]]])
    inc_size = [128, 128, 1, 64] if same_size else [256, 256, 1, 64]
    inc = _logic([_sol(0, "s1", mfma=True, Foo="bar")], [[inc_size, [0, inc_eff]]])
    M.reindexSolutions(ori)
    M.reindexSolutions(inc)
    return ori, inc


def test_merge_logic_improves():
    ori, inc = _base_inc(0.5, 0.9)
    merged, nSizes, nSols, nRemoved = M.mergeLogic(ori, inc, forceMerge=False)
    assert nSizes == 0          # same size, no new size
    assert nSols == 1           # improved solution added to pool
    assert merged[7][0][1][1] == 0.9


def test_merge_logic_rejects_when_not_better():
    ori, inc = _base_inc(0.9, 0.5)
    merged, nSizes, nSols, nRemoved = M.mergeLogic(ori, inc, forceMerge=False)
    assert nSizes == 0
    assert nSols == 0           # rejected: no new solution
    assert merged[7][0][1][1] == 0.9  # original kept


def test_merge_logic_force_merge_replaces_slower():
    ori, inc = _base_inc(0.9, 0.5)
    merged, nSizes, nSols, nRemoved = M.mergeLogic(ori, inc, forceMerge=True)
    assert merged[7][0][1][1] == 0.5  # forced even though slower


def test_merge_logic_adds_new_size():
    ori, inc = _base_inc(0.5, 0.7, same_size=False)
    merged, nSizes, nSols, nRemoved = M.mergeLogic(ori, inc, forceMerge=False)
    assert nSizes == 1          # new size appended
    assert nSols == 1


def test_merge_logic_add_solution_tags_verbose(monkeypatch):
    # addSolutionTags=True path: oriData[7]/incData[7] become tag-prefixed and the
    # final map is de-tagged via removeSolutionTagFromKeys. verbosity=2 also drives
    # the debug() prints in addKernel/removeUnusedKernels.
    monkeypatch.setattr(M, "verbosity", 2)
    ori, inc = _base_inc(0.5, 0.9)
    merged, nSizes, nSols, nRemoved = M.mergeLogic(
        ori, inc, forceMerge=False, addSolutionTags=True
    )
    # de-tagged: the surviving size key is the plain 4-tuple again
    assert merged[7][0][0] == [128, 128, 1, 64]


# ---------------------------------------------------------------------------
# mergePartialLogics / avoidRegressions (end-to-end, file IO)
# ---------------------------------------------------------------------------
def _write_logic(path, eff, size=None):
    sols = [_sol(0, "s0")]
    sizemap = [[size or [128, 128, 1, 64], [0, eff]]]
    path.write_text(yaml.safe_dump(_logic(sols, sizemap)))


def test_merge_partial_logics(tmp_path):
    d0 = tmp_path / "p0"
    d1 = tmp_path / "p1"
    out = tmp_path / "out"
    d0.mkdir()
    d1.mkdir()
    _write_logic(d0 / "logic.yaml", 0.5)
    _write_logic(d1 / "logic.yaml", 0.9)  # same size, better -> improves
    M.mergePartialLogics([str(d0 / "logic.yaml"), str(d1 / "logic.yaml")], str(out), forceMerge=False)
    result = yaml.safe_load((out / "logic.yaml").read_text())
    assert result[7][0][1][1] == 0.9


def test_avoid_regressions_merge_and_copy(tmp_path):
    orig = tmp_path / "Other"        # non-check folder -> skips dest/attr check
    inc = tmp_path / "inc"
    out = tmp_path / "out"
    orig.mkdir()
    inc.mkdir()
    # matching basename -> merged
    _write_logic(orig / "logic.yaml", 0.5)
    _write_logic(inc / "logic.yaml", 0.9)
    # non-matching basename -> copied straight through
    _write_logic(inc / "extra.yaml", 0.3)

    M.avoidRegressions(str(orig), str(inc), str(out), forceMerge=True)
    merged = yaml.safe_load((out / "logic.yaml").read_text())
    assert merged[7][0][1][1] == 0.9
    assert (out / "extra.yaml").is_file()  # copied
