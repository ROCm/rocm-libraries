################################################################################
# Characterization tests for Tensile.TensileMergeLibrary — pure helper layer.
#
# ADD-ONLY. The mergeLogic / avoidRegressions / main CLI driver (and the
# ProblemType-deriving compareProblemType / reNameSolutions) are resistance; this
# pins the pure logic-data helpers. Logic data is list-indexed:
#   data[4]=ProblemType, data[5]=solutions, data[7]=size map, data[11]=attribute.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileMergeLibrary")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(M, "verbosity", 1)


def _sol(idx, name="s", kname="k", **extra):
    d = {"SolutionIndex": idx, "SolutionNameMin": name, "KernelNameMin": kname}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# ensurePath / allFiles
# ---------------------------------------------------------------------------
def test_ensure_path(tmp_path):
    p = tmp_path / "a" / "b"
    assert M.ensurePath(str(p)) == str(p)
    assert p.is_dir()
    assert M.ensurePath(str(p)) == str(p)  # idempotent


def test_all_files(tmp_path):
    (tmp_path / "a.yaml").write_text("1")
    (tmp_path / "skip.txt").write_text("x")
    sub = tmp_path / "sub.yaml"
    sub.mkdir()
    (sub / "b.yaml").write_text("2")
    names = sorted(f.split("/")[-1] for f in M.allFiles(str(tmp_path)))
    assert names == ["a.yaml", "b.yaml"]


# ---------------------------------------------------------------------------
# fixSizeInconsistencies (pins the generator-key dedup BUG)
# ---------------------------------------------------------------------------
def test_fix_size_trims_long_format():
    sizes = [[[1, 2, 3, 4, 5, 6, 7, 8], [0, 0.5]]]
    out, n = M.fixSizeInconsistencies(sizes, "base")
    assert out[0][0] == [1, 2, 3, 4]
    assert n == 1


def test_fix_size_dedup_merges_trimmed_duplicates():
    """Dedup now merges entries that trim to the same prefix.

    The previous latent bug keyed the dedup dict by ``(value for value in size)``
    -- a fresh generator object per entry, never equal -- so duplicates were never
    merged. The fix materializes a tuple key (``tuple(value for value in size)``),
    so two long-format sizes that both trim to ``[1, 2, 3, 4]`` collapse to one.
    """
    sizes = [
        [[1, 2, 3, 4, 9, 9, 9, 9], [0, 0.5]],
        [[1, 2, 3, 4, 8, 8, 8, 8], [1, 0.6]],
    ]
    out, n = M.fixSizeInconsistencies(sizes, "base")
    assert n == 1
    assert len(out) == 1
    assert out[0][0] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# addKernel
# ---------------------------------------------------------------------------
def test_add_kernel_reuse():
    pool = [_sol(0, "s0")]
    solDict = {"s0": pool[0]}
    pool, solDict, idx = M.addKernel(pool, solDict, _sol(99, "s0"))
    assert idx == 0
    assert len(pool) == 1


def test_add_kernel_new():
    pool = [_sol(0, "s0")]
    solDict = {"s0": pool[0]}
    pool, solDict, idx = M.addKernel(pool, solDict, _sol(0, "s1"))
    assert idx == 1
    assert "s1" in solDict
    assert pool[1]["SolutionIndex"] == 1


# ---------------------------------------------------------------------------
# sanitizeSolutions
# ---------------------------------------------------------------------------
def test_sanitize_zero_stagger():
    sols = [{"StaggerU": 0}]
    M.sanitizeSolutions(sols)
    assert sols[0]["StaggerUMapping"] == 0
    assert sols[0]["_staggerStrideShift"] == 0


def test_sanitize_nonzero_stagger_untouched():
    sols = [{"StaggerU": 4}]
    M.sanitizeSolutions(sols)
    assert "StaggerUMapping" not in sols[0]


# ---------------------------------------------------------------------------
# removeUnusedSolutions
# ---------------------------------------------------------------------------
def test_remove_unused_solutions():
    data = [None] * 8
    data[5] = [_sol(0, "used"), _sol(1, "unused")]
    data[7] = [[[1, 2, 3, 4], [0, 0.5]]]  # only solution 0 in use
    out, removed = M.removeUnusedSolutions(data)
    assert removed == 1
    assert len(out[5]) == 1
    assert out[5][0]["SolutionIndex"] == 0
    assert out[7][0][1][0] == 0


# ---------------------------------------------------------------------------
# removeDuplicatedSolutions
# ---------------------------------------------------------------------------
def test_remove_duplicated_solutions():
    data = [None] * 8
    # two solutions share SolutionNameMin -> deduped to one
    data[5] = [_sol(0, "dup", "k0"), _sol(1, "dup", "k0"), _sol(2, "uniq", "k1")]
    data[7] = [[[1], [0, 0.5]], [[2], [1, 0.6]], [[3], [2, 0.7]]]
    out, numRemoved, numSols, numKernels = M.removeDuplicatedSolutions(data)
    assert numRemoved == 1
    assert numSols == 2
    assert numKernels == 2


# ---------------------------------------------------------------------------
# msg / verbose / debug
# ---------------------------------------------------------------------------
def test_verbose_gated(monkeypatch, capsys):
    monkeypatch.setattr(M, "verbosity", 0)
    M.verbose("hidden")
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(M, "verbosity", 1)
    M.verbose("shown")
    assert "shown" in capsys.readouterr().out


def test_debug_gated(monkeypatch, capsys):
    monkeypatch.setattr(M, "verbosity", 2)
    M.debug("shown")
    assert "shown" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# findSolutionWithIndex
# ---------------------------------------------------------------------------
def test_find_solution_with_index():
    sols = [_sol(0), _sol(1)]
    assert M.findSolutionWithIndex(sols, 1)["SolutionIndex"] == 1
    shuffled = [_sol(5), _sol(0)]
    assert M.findSolutionWithIndex(shuffled, 0)["SolutionIndex"] == 0


# ---------------------------------------------------------------------------
# compareDestFolderToYaml
# ---------------------------------------------------------------------------
def _logic_attr(attr):
    d = [None] * 12
    d[11] = attr
    return d


def test_compare_dest_folder_ok():
    M.compareDestFolderToYaml("/x/Equality", "inc.yaml", _logic_attr("Equality"))


def test_compare_dest_folder_noncheck_passes():
    M.compareDestFolderToYaml("/x/Other", "inc.yaml", _logic_attr("GridBased"))


def test_compare_dest_folder_mismatch_exits():
    with pytest.raises(SystemExit):
        M.compareDestFolderToYaml("/x/Equality", "inc.yaml", _logic_attr("GridBased"))


def test_compare_dest_folder_empty_attr_exits():
    with pytest.raises(SystemExit):
        M.compareDestFolderToYaml("/x/Equality", "inc.yaml", _logic_attr(""))
