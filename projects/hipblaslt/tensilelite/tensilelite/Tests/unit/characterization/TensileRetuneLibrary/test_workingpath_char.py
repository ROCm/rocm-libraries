################################################################################
# Characterization tests for Tensile.TensileRetuneLibrary — working-path helpers.
#
# ADD-ONLY. parseCurrentLibrary / runBenchmarking / TensileRetuneLibrary / main
# require derived ProblemType + benchmarking and are resistance. This pins the
# pure WorkingPath stack helpers: ensurePath, pushWorkingPath, popWorkingPath,
# setWorkingPath.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileRetuneLibrary")


@pytest.fixture
def _gp(monkeypatch, tmp_path):
    # isolate the global WorkingPath and the module-level stack
    monkeypatch.setitem(M.globalParameters, "WorkingPath", str(tmp_path))
    monkeypatch.setattr(M, "workingDirectoryStack", [])
    return tmp_path


def test_ensure_path(tmp_path):
    p = tmp_path / "x" / "y"
    assert M.ensurePath(str(p)) == str(p)
    assert p.is_dir()
    assert M.ensurePath(str(p)) == str(p)  # FileExistsError swallowed


def test_push_working_path(_gp):
    out = M.pushWorkingPath("sub")
    assert out.endswith("sub")
    assert M.globalParameters["WorkingPath"].endswith("sub")


def test_set_working_path_pushes_and_sets(_gp, tmp_path):
    target = tmp_path / "w"
    M.setWorkingPath(str(target))
    assert M.globalParameters["WorkingPath"] == str(target)
    assert target.is_dir()
    assert M.workingDirectoryStack == [str(_gp)]


def test_pop_working_path_from_stack(_gp, tmp_path):
    M.setWorkingPath(str(tmp_path / "w"))     # stack=[orig], WP=w
    M.popWorkingPath()                          # stack non-empty -> restore orig
    assert M.globalParameters["WorkingPath"] == str(_gp)


def test_pop_working_path_empty_stack_goes_to_parent(_gp):
    # empty stack -> WorkingPath becomes its parent dir
    M.popWorkingPath()
    assert M.globalParameters["WorkingPath"] == str(_gp.parent)
