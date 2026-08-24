################################################################################
# Characterization tests for Tensile.TensileCreateLibrary.Run — helper layer.
#
# ADD-ONLY. Run.py is the TensileCreateLibrary driver: most of it is asm codegen
# (processKernelSource / writeAssembly / writeSolutionsAndKernels / run()) using
# KernelWriterAssembly + the toolchain, which is out of scope. This suite pins
# the pure / stubbable helpers: libraryDir, the result NamedTuples,
# _stinky_asm_verify_wanted/_stinky_out, memCompress/memDecompress, the
# invalid-solution filters, passPostKernelInfoToSolution, and the fallback
# placeholder renamers.
################################################################################
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileCreateLibrary.Run")
SL = importlib.import_module("Tensile.SolutionLibrary")


# ---------------------------------------------------------------------------
# libraryRoot / libraryDir
# ---------------------------------------------------------------------------
def test_library_root():
    assert M.libraryRoot("/out") == Path("/out/library")


def test_library_dir_single_arch():
    assert M.libraryDir("/out", "gfx942") == Path("/out/library/gfx942")


def test_library_dir_strips_target_features():
    assert M.libraryDir("/out", "gfx942:sramecc+:xnack-") == Path("/out/library/gfx942")


# ---------------------------------------------------------------------------
# result NamedTuples
# ---------------------------------------------------------------------------
def test_kernel_codegen_result_fields():
    r = M.KernelCodeGenResult(0, "src", "hdr", "name", "obj", (9, 0, 10), 64, 1, 2, 3)
    assert r.err == 0 and r.cuoccupancy == 1 and r.pgr == 2 and r.mathclk == 3


def test_kernel_min_result_fields():
    r = M.KernelMinResult(1, 4, 5, 6)
    assert (r.err, r.cuoccupancy, r.pgr, r.mathclk) == (1, 4, 5, 6)


# ---------------------------------------------------------------------------
# _stinky_asm_verify_wanted / _stinky_out
# ---------------------------------------------------------------------------
def test_stinky_wanted_true(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "CheckASMCodeSize", True)
    monkeypatch.setattr(M, "isaToGfx", lambda isa: "gfx1250")
    assert M._stinky_asm_verify_wanted((12, 5, 0)) is True


def test_stinky_wanted_false_flag_off(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "CheckASMCodeSize", False)
    monkeypatch.setattr(M, "isaToGfx", lambda isa: "gfx1250")
    assert M._stinky_asm_verify_wanted((12, 5, 0)) is False


def test_stinky_wanted_false_other_arch(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "CheckASMCodeSize", True)
    monkeypatch.setattr(M, "isaToGfx", lambda isa: "gfx942")
    assert M._stinky_asm_verify_wanted((9, 0, 10)) is False


def test_stinky_out_does_not_raise(capsys):
    M._stinky_out("hello stinky")  # writes to fd 2


# ---------------------------------------------------------------------------
# memCompress / memDecompress
# ---------------------------------------------------------------------------
def test_mem_compress_roundtrip():
    obj = {"a": [1, 2, 3], "b": "x"}
    assert M.memDecompress(M.memCompress(obj)) == obj


# ---------------------------------------------------------------------------
# _checkInvalidSolutionsAndKernels
# ---------------------------------------------------------------------------
def test_check_invalid_ok():
    assert M._checkInvalidSolutionsAndKernels(False, SimpleNamespace(err=0), {}) is False


def test_check_invalid_err_not_tolerant_prints(capsys):
    kernel = {"SolutionIndex": 7, "SolutionNameMin": "bad"}
    assert M._checkInvalidSolutionsAndKernels(False, SimpleNamespace(err=1), kernel) is True
    assert "Kernel generation failed" in capsys.readouterr().out


def test_check_invalid_err_tolerant_no_print(capsys):
    assert M._checkInvalidSolutionsAndKernels(True, SimpleNamespace(err=1), {}) is True
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _checkInvalidSolutions
# ---------------------------------------------------------------------------
class _Sol:
    def __init__(self, kernels, state=None):
        self._kernels = kernels
        self._state = state if state is not None else {}

    def getKernels(self):
        return self._kernels


def test_check_invalid_solutions(monkeypatch):
    monkeypatch.setattr(M, "getKeyNoInternalArgs", lambda k, split: k["name"])
    sols = [_Sol([{"name": "k0"}]), _Sol([{"name": "kBAD"}])]
    out = M._checkInvalidSolutions(False, {"kBAD"}, sols)
    # one True appended for the matching solution, plus a trailing False per loop
    assert True in out


# ---------------------------------------------------------------------------
# removeInvalidSolutionsAndKernels (ParallelMap2 + Naming stubbed)
# ---------------------------------------------------------------------------
def test_remove_invalid_solutions_and_kernels(monkeypatch):
    monkeypatch.setattr(M, "getKeyNoInternalArgs", lambda k, split: k["name"])
    monkeypatch.setattr(
        M, "ParallelMap2",
        lambda fn, iterable, desc, return_as=None: [fn(*x) for x in iterable],
    )
    results = [SimpleNamespace(err=0), SimpleNamespace(err=1)]
    kernels = [{"name": "k0", "SolutionIndex": 0, "SolutionNameMin": "s0"},
               {"name": "k1", "SolutionIndex": 1, "SolutionNameMin": "s1"}]
    sols = [_Sol([{"name": "k0"}]), _Sol([{"name": "k1"}])]
    M.removeInvalidSolutionsAndKernels(
        results, kernels, sols, errorTolerant=True, printLevel=1, splitGSU=False
    )
    # k1 (err=1) removed from kernels/results; sol holding k1 removed
    assert [k["name"] for k in kernels] == ["k0"]
    assert len(results) == 1
    assert len(sols) == 1


# ---------------------------------------------------------------------------
# passPostKernelInfoToSolution
# ---------------------------------------------------------------------------
def test_pass_post_kernel_info_to_solution(monkeypatch):
    monkeypatch.setattr(M, "getKernelNameMin", lambda k, split: k["name"])
    results = [SimpleNamespace(cuoccupancy=11, pgr=22, mathclk=33)]
    kernels = [{"name": "k0"}]
    sol = _Sol([{"name": "k0"}], state={})
    M.passPostKernelInfoToSolution(results, kernels, [sol], splitGSU=False)
    assert sol._state["CUOccupancy"] == 11
    assert sol._state["PrefetchGlobalRead"] == 22
    assert sol._state["MathClocksUnrolledLoop"] == 33


# ---------------------------------------------------------------------------
# _renameFallbackPlaceholders / renameFallbacksPerArch
# ---------------------------------------------------------------------------
def test_rename_fallback_placeholder_leaf():
    ph = SL.PlaceholderLibrary("lib_fallback")
    M._renameFallbackPlaceholders(ph, "gfx942")
    assert ph.filenamePrefix == "lib_fallback_gfx942"


def test_rename_fallback_placeholder_idempotent():
    ph = SL.PlaceholderLibrary("lib_fallback_gfx942")
    M._renameFallbackPlaceholders(ph, "gfx942")
    assert ph.filenamePrefix == "lib_fallback_gfx942"  # not double-suffixed


def test_rename_fallback_placeholder_non_fallback_untouched():
    ph = SL.PlaceholderLibrary("plain")
    M._renameFallbackPlaceholders(ph, "gfx942")
    assert ph.filenamePrefix == "plain"


def test_rename_fallback_walks_rows_and_mapping():
    leaf1 = SL.PlaceholderLibrary("a_fallback")
    leaf2 = SL.PlaceholderLibrary("b_fallback")
    rows_node = SimpleNamespace(rows=[{"library": leaf1}])
    mapping_node = SimpleNamespace(rows=None, mapping={"k": leaf2})
    M._renameFallbackPlaceholders(rows_node, "gfx90a")
    M._renameFallbackPlaceholders(mapping_node, "gfx90a")
    assert leaf1.filenamePrefix == "a_fallback_gfx90a"
    assert leaf2.filenamePrefix == "b_fallback_gfx90a"


def test_rename_fallback_none_is_noop():
    M._renameFallbackPlaceholders(None, "gfx942")  # no raise


def test_rename_fallbacks_per_arch():
    ph = SL.PlaceholderLibrary("tree_fallback")
    libtree = SimpleNamespace(rows=[{"library": ph}], mapping=None)
    master = SL.MasterSolutionLibrary({}, libtree)
    master.lazyLibraries = {
        "x_fallback": SimpleNamespace(),
        "y_plain": SimpleNamespace(),
    }
    masters = {"gfx942": master}
    M.renameFallbacksPerArch(masters)
    m = masters["gfx942"]
    assert "x_fallback_gfx942" in m.lazyLibraries
    assert "y_plain" in m.lazyLibraries
    # the deep-copied tree's placeholder was arch-suffixed
    assert m.library.rows[0]["library"].filenamePrefix == "tree_fallback_gfx942"
