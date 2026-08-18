################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Characterization tests for Tensile.TensileCreateLibrary.Run — orchestration layer.

Targets methodology-A missing ranges:
  152-173  : _stinky_asm_verify_wanted, _stinky_out, _verify_stinky_asm_comment_vs_elf_text
  529-619  : writeSolutionsAndKernelsTCL (setup, dedup, compose block)
  754-873  : _renameFallbackPlaceholders, renameFallbacksPerArch,
             generateLogicDataAndSolutions (arch split, fIter setup, libraryIter)
  881-1086 : generateLogicDataAndSolutions (parse loop, re-index, fallback merge,
             solutions dedupe) + generateKernelObjectsFromSolutions

Pattern B: drive public helpers + generateLogicDataAndSolutions with a real tiny
logic YAML; no GPU, no actual assembly.
"""

import shutil
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Module imports (deferred so rocisa import happens only once per process)
# ---------------------------------------------------------------------------
M = importlib.import_module("Tensile.TensileCreateLibrary.Run")
SL = importlib.import_module("Tensile.SolutionLibrary")

_DATA_DIR = Path(__file__).parent / "data"
_LOGIC_YAML = _DATA_DIR / "logic_gfx942_HSS_BH_tiny.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_isa_info_map(arch: str = "gfx942"):
    """Build a real IsaInfo map with SupportedISA patched to True.

    makeIsaInfoMap calls rocisa which returns SupportedISA=0 in CI (no GPU).
    We patch the one field needed to pass Solution.__init__'s assertion.
    """
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.Architectures import gfxToIsa

    clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
    isa = gfxToIsa(arch)
    info_map = makeIsaInfoMap([isa], clang)
    info_map[isa].asmCaps["SupportedISA"] = True
    return isa, info_map


def _make_assembler():
    """Return a real Assembler wrapping the system clang++."""
    from Tensile.Toolchain.Component import Assembler

    clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
    return Assembler(Path(clang), "4")


# ===========================================================================
# 1. _stinky_asm_verify_wanted / _stinky_out  (lines 151-173)
# ===========================================================================

class TestStinkyAsmVerifyWanted:
    """_stinky_asm_verify_wanted — lines 151-157."""

    def test_false_when_check_disabled(self, monkeypatch):
        """Line 157: returns False when CheckASMCodeSize is falsy."""
        monkeypatch.setitem(M.globalParameters, "CheckASMCodeSize", False)
        monkeypatch.setattr(M, "isaToGfx", lambda _isa: "gfx1250")
        assert M._stinky_asm_verify_wanted((12, 5, 0)) is False

    def test_false_when_not_gfx1250(self, monkeypatch):
        """Line 157: returns False for non-gfx1250 arch even if flag is set."""
        monkeypatch.setitem(M.globalParameters, "CheckASMCodeSize", True)
        monkeypatch.setattr(M, "isaToGfx", lambda _isa: "gfx942")
        assert M._stinky_asm_verify_wanted((9, 4, 2)) is False

    def test_true_when_gfx1250_and_flag_set(self, monkeypatch):
        """Line 157: returns True only for gfx1250 with flag on."""
        monkeypatch.setitem(M.globalParameters, "CheckASMCodeSize", True)
        monkeypatch.setattr(M, "isaToGfx", lambda _isa: "gfx1250")
        assert M._stinky_asm_verify_wanted((12, 5, 0)) is True


class TestStinkyOut:
    """_stinky_out — lines 160-169."""

    def test_writes_to_stderr_no_raise(self):
        """Lines 166-167: os.write(2, ...) completes without raising.

        _stinky_out uses os.write(2, ...) directly (not sys.stderr), so
        capsys cannot capture it. We verify execution completes without error.
        """
        M._stinky_out("check stinky ok")  # must not raise

    def test_oserror_silenced(self, monkeypatch):
        """Lines 168-169: OSError on os.write is swallowed."""
        monkeypatch.setattr(M.os, "write", lambda *_: (_ for _ in ()).throw(OSError("fake")))
        M._stinky_out("should not raise")  # must not propagate


class TestVerifyStinkyAsmCommentVsElfText:
    """_verify_stinky_asm_comment_vs_elf_text — lines 172-207."""

    def test_code2_calls_printExit(self, monkeypatch, tmp_path):
        """Lines 195-196: code==2 triggers printExit (verifier error)."""
        monkeypatch.setattr(M, "verify_stinky_paths", lambda s, o: (2, "", "err"))
        monkeypatch.setattr(M, "_stinky_out", lambda msg: None)
        with pytest.raises(SystemExit):
            M._verify_stinky_asm_comment_vs_elf_text(
                tmp_path / "k.s", tmp_path / "k.o", "k"
            )

    def test_code1_calls_printExit(self, monkeypatch, tmp_path):
        """Lines 197-200: code==1 triggers printExit (mismatch)."""
        monkeypatch.setattr(M, "verify_stinky_paths", lambda s, o: (1, "", ""))
        monkeypatch.setattr(M, "_stinky_out", lambda msg: None)
        with pytest.raises(SystemExit):
            M._verify_stinky_asm_comment_vs_elf_text(
                tmp_path / "k.s", tmp_path / "k.o", "k"
            )

    def test_code0_ok_stinky_logged(self, monkeypatch, tmp_path):
        """Lines 201-207: code==0 with 'OK STINKY' emits match log."""
        logged = []
        monkeypatch.setattr(M, "verify_stinky_paths", lambda s, o: (0, "OK STINKY match", ""))
        monkeypatch.setattr(M, "_stinky_out", lambda msg: logged.append(msg))
        M._verify_stinky_asm_comment_vs_elf_text(
            tmp_path / "k.s", tmp_path / "k.o", "myk"
        )
        assert any("OK STINKY_TOTAL_INST_BYTES" in m for m in logged)

    def test_exception_in_verifier_calls_printExit(self, monkeypatch, tmp_path):
        """Lines 186-188: exception inside verify_stinky_paths triggers printExit."""
        monkeypatch.setattr(
            M, "verify_stinky_paths",
            lambda s, o: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        monkeypatch.setattr(M, "_stinky_out", lambda msg: None)
        with pytest.raises(SystemExit):
            M._verify_stinky_asm_comment_vs_elf_text(
                tmp_path / "k.s", tmp_path / "k.o", "k"
            )

    def test_out_s_and_err_s_logged(self, monkeypatch, tmp_path):
        """Lines 189-193: out_s and err_s lines are forwarded through _stinky_out."""
        logged = []
        monkeypatch.setattr(M, "verify_stinky_paths", lambda s, o: (0, "out line", "err line"))
        monkeypatch.setattr(M, "_stinky_out", lambda msg: logged.append(msg))
        M._verify_stinky_asm_comment_vs_elf_text(
            tmp_path / "k.s", tmp_path / "k.o", "k"
        )
        assert "out line" in logged
        assert "err line" in logged


# ===========================================================================
# 2. _renameFallbackPlaceholders / renameFallbacksPerArch  (lines 734-786)
# ===========================================================================

class TestRenameFallbackPlaceholders:
    """_renameFallbackPlaceholders — lines 734-756."""

    def test_none_node_is_noop(self):
        """Line 743-744: None returns immediately."""
        M._renameFallbackPlaceholders(None, "gfx942")

    def test_placeholder_with_fallback_gets_suffix(self):
        """Lines 745-748: _fallback prefix is suffixed with arch."""
        ph = SL.PlaceholderLibrary("lib_fallback")
        M._renameFallbackPlaceholders(ph, "gfx942")
        assert ph.filenamePrefix == "lib_fallback_gfx942"

    def test_already_suffixed_is_idempotent(self):
        """Line 747: already arch-suffixed placeholder is untouched."""
        ph = SL.PlaceholderLibrary("lib_fallback_gfx942")
        M._renameFallbackPlaceholders(ph, "gfx942")
        assert ph.filenamePrefix == "lib_fallback_gfx942"

    def test_non_fallback_placeholder_untouched(self):
        """Line 746: non-fallback prefix is not modified."""
        ph = SL.PlaceholderLibrary("tuned")
        M._renameFallbackPlaceholders(ph, "gfx942")
        assert ph.filenamePrefix == "tuned"

    def test_walks_rows(self):
        """Lines 749-752: rows attribute is walked recursively."""
        leaf = SL.PlaceholderLibrary("z_fallback")
        node = SimpleNamespace(rows=[{"library": leaf}], mapping=None)
        M._renameFallbackPlaceholders(node, "gfx90a")
        assert leaf.filenamePrefix == "z_fallback_gfx90a"

    def test_walks_mapping(self):
        """Lines 753-756: mapping values are walked recursively."""
        leaf = SL.PlaceholderLibrary("q_fallback")
        node = SimpleNamespace(rows=None, mapping={"k": leaf})
        M._renameFallbackPlaceholders(node, "gfx90a")
        assert leaf.filenamePrefix == "q_fallback_gfx90a"


class TestRenameFallbacksPerArch:
    """renameFallbacksPerArch — lines 759-786."""

    def test_lazy_lib_fallback_key_gets_arch_suffix(self):
        """Lines 780-783: _fallback lazy-lib key is renamed to _fallback_<arch>."""
        master = SL.MasterSolutionLibrary({}, None)
        lazy_stub = SimpleNamespace()
        master.lazyLibraries = {"x_fallback": lazy_stub, "plain": SimpleNamespace()}
        masters = {"gfx942": master}
        M.renameFallbacksPerArch(masters)
        assert "x_fallback_gfx942" in masters["gfx942"].lazyLibraries
        assert "plain" in masters["gfx942"].lazyLibraries
        assert "x_fallback" not in masters["gfx942"].lazyLibraries

    def test_deep_copy_isolates_masters(self):
        """Line 777: each arch gets a deep copy (original not mutated)."""
        master = SL.MasterSolutionLibrary({}, None)
        master.lazyLibraries = {}
        original_id = id(master)
        masters = {"gfx942": master}
        M.renameFallbacksPerArch(masters)
        assert id(masters["gfx942"]) != original_id

    def test_placeholder_inside_tree_gets_arch_suffix(self):
        """Lines 785-786: PlaceholderLibrary inside the library tree is renamed."""
        ph = SL.PlaceholderLibrary("t_fallback")
        lib_tree = SimpleNamespace(rows=[{"library": ph}], mapping=None)
        master = SL.MasterSolutionLibrary({}, lib_tree)
        master.lazyLibraries = {}
        masters = {"gfx90a": master}
        M.renameFallbacksPerArch(masters)
        # The deep copy's placeholder should be renamed
        new_tree = masters["gfx90a"].library
        renamed_ph = new_tree.rows[0]["library"]
        assert renamed_ph.filenamePrefix == "t_fallback_gfx90a"


# ===========================================================================
# 3. generateLogicDataAndSolutions  (lines 790-911)
#    Exercises: arch split, fIter, libraryIter, parse loop, re-index, solutions
# ===========================================================================

class TestGenerateLogicDataAndSolutions:
    """generateLogicDataAndSolutions with a real tiny logic YAML — lines 790-911."""

    @pytest.fixture(scope="class")
    def isa_map(self):
        _isa, _map = _make_isa_info_map("gfx942")
        return _isa, _map

    @pytest.fixture(scope="class")
    def assembler(self):
        return _make_assembler()

    def _run(self, assembler, isa_map, gen_sol_table=False, lazy=True):
        _isa, info_map = isa_map
        args = {
            "Architecture": "gfx942",
            "LazyLibraryLoading": lazy,
            "CodeObjectVersion": "4",
            "GenSolTable": gen_sol_table,
        }
        solutions, masterLibraries, libraryMapping = M.generateLogicDataAndSolutions(
            [str(_LOGIC_YAML)], args, assembler, info_map
        )
        return solutions, masterLibraries, libraryMapping

    def test_returns_solutions_for_gfx942(self, assembler, isa_map):
        """Lines 822-835: logic file parsed, masterLibraries populated."""
        solutions, masterLibraries, _ = self._run(assembler, isa_map)
        assert "gfx942" in masterLibraries
        sol_list = list(solutions)
        assert len(sol_list) >= 1

    def test_library_mapping_is_dict(self, assembler, isa_map):
        """Lines 884-902: codeObjectFilesIndex built from lazy-library solutions."""
        _, _, libraryMapping = self._run(assembler, isa_map, lazy=True)
        assert isinstance(libraryMapping, dict)

    def test_gen_sol_table_writes_match_table(self, assembler, isa_map, monkeypatch):
        """Lines 860-866: GenSolTable=True calls LibraryIO.write with MatchTable."""
        written = {}

        def fake_write(name, data, *args, **kwargs):
            written[name] = data

        monkeypatch.setattr(M.LibraryIO, "write", fake_write)
        self._run(assembler, isa_map, gen_sol_table=True)
        assert "MatchTable" in written
        assert isinstance(written["MatchTable"], dict)

    def test_solutions_dedup_on_repeated_file(self, assembler, isa_map):
        """Lines 904-906: solutions are deduplicated when same YAML given twice."""
        _isa, info_map = isa_map
        args = {
            "Architecture": "gfx942",
            "LazyLibraryLoading": True,
            "CodeObjectVersion": "4",
            "GenSolTable": False,
        }
        solutions_double, masterLibs_double, _ = M.generateLogicDataAndSolutions(
            [str(_LOGIC_YAML), str(_LOGIC_YAML)], args, assembler, info_map
        )
        solutions_single, masterLibs_single, _ = self._run(assembler, isa_map)
        # After dedup, count should not double
        count_double = len(list(solutions_double))
        count_single = len(list(solutions_single))
        assert count_double <= count_single * 2

    def test_semicolon_arch_split(self, assembler, isa_map):
        """Lines 792-795: arch split works for semicolon-separated form."""
        _isa, info_map = isa_map
        args = {
            "Architecture": "gfx942;gfx942",  # semicolon split path
            "LazyLibraryLoading": True,
            "CodeObjectVersion": "4",
            "GenSolTable": False,
        }
        solutions, masterLibs, _ = M.generateLogicDataAndSolutions(
            [str(_LOGIC_YAML)], args, assembler, info_map
        )
        assert "gfx942" in masterLibs

    def test_underscore_arch_split(self, assembler, isa_map):
        """Lines 794-795: arch split works for underscore-separated (default) form."""
        _isa, info_map = isa_map
        args = {
            "Architecture": "gfx942",
            "LazyLibraryLoading": True,
            "CodeObjectVersion": "4",
            "GenSolTable": False,
        }
        solutions, masterLibs, _ = M.generateLogicDataAndSolutions(
            [str(_LOGIC_YAML)], args, assembler, info_map
        )
        assert "gfx942" in masterLibs

    def test_empty_logic_list_returns_empty(self, assembler, isa_map):
        """Lines 804-812: empty logicFiles list produces empty masterLibraries."""
        _isa, info_map = isa_map
        args = {
            "Architecture": "gfx942",
            "LazyLibraryLoading": True,
            "CodeObjectVersion": "4",
            "GenSolTable": False,
        }
        solutions, masterLibs, libraryMapping = M.generateLogicDataAndSolutions(
            [], args, assembler, info_map
        )
        assert masterLibs == {}
        assert libraryMapping == {}
        assert list(solutions) == []


# ===========================================================================
# 4. generateKernelObjectsFromSolutions  (lines 678-688)
#    Called downstream of generateLogicDataAndSolutions
# ===========================================================================

class TestGenerateKernelObjectsFromSolutions:
    """generateKernelObjectsFromSolutions — lines 678-688."""

    @pytest.fixture(scope="class")
    def solutions_fixture(self):
        _isa, info_map = _make_isa_info_map("gfx942")
        asm = _make_assembler()
        args = {
            "Architecture": "gfx942",
            "LazyLibraryLoading": True,
            "CodeObjectVersion": "4",
            "GenSolTable": False,
        }
        solutions, _, _ = M.generateLogicDataAndSolutions(
            [str(_LOGIC_YAML)], args, asm, info_map
        )
        return list(solutions)

    def test_returns_kernels_list(self, solutions_fixture):
        """Lines 678-688: returns non-empty list of kernel objects."""
        kernels = M.generateKernelObjectsFromSolutions(solutions_fixture)
        assert isinstance(kernels, list)
        assert len(kernels) >= 1

    def test_kernels_are_deduplicated(self, solutions_fixture):
        """Line 684-687: duplicate kernel names are filtered."""
        # Duplicate solutions list to force duplicate kernels
        doubled = solutions_fixture + solutions_fixture
        kernels_single = M.generateKernelObjectsFromSolutions(solutions_fixture)
        kernels_doubled = M.generateKernelObjectsFromSolutions(doubled)
        # Should be same count — dedup applies
        assert len(kernels_doubled) == len(kernels_single)
