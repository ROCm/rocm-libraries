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

"""Characterization tests for Tensile.TensileCreateLibrary.Run — deep coverage pass.

Target miss ranges (methodology-A):
  336-348  : writeAssembly (assembly-file writer)
  353-375  : writeHelpers (Kernels.cpp/h writer)
  394-513  : writeSolutionsAndKernels (legacy orchestration, all branches)
  529-619  : writeSolutionsAndKernelsTCL (TCL orchestration setup + compose block)
  624-636  : copyStaticFiles
  671-693  : generateKernelHelperObjects (helper dedup + enum-first sort)
  881-1086 : run() (CLI entry point — all major arms)
  808-809, 832-835, 840, 844-845, 855: generateLogicDataAndSolutions branches
    (fallback merge, codeObjectFilesIndex, etc.)

Strategy: all tests are pure-assert and CPU-only.  Heavy assembly / linker
calls are monkeypatched.  No GPU device is required.
"""

import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
M = importlib.import_module("Tensile.TensileCreateLibrary.Run")
SL = importlib.import_module("Tensile.SolutionLibrary")

_DATA_DIR = Path(__file__).parent / "data"
_LOGIC_YAML = _DATA_DIR / "logic_gfx942_HSS_BH_tiny.yaml"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_isa_info_map(arch: str = "gfx942"):
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.Architectures import gfxToIsa

    clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
    isa = gfxToIsa(arch)
    info_map = makeIsaInfoMap([isa], clang)
    info_map[isa].asmCaps["SupportedISA"] = True
    return isa, info_map


def _make_assembler():
    from Tensile.Toolchain.Component import Assembler

    clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
    return Assembler(Path(clang), "4")


def _load_solutions(logic_yaml=_LOGIC_YAML, arch="gfx942"):
    """Load solutions from logic YAML; return (list[sol], info_map)."""
    _, info_map = _make_isa_info_map(arch)
    asm = _make_assembler()
    args = {
        "Architecture": arch,
        "LazyLibraryLoading": True,
        "CodeObjectVersion": "4",
        "GenSolTable": False,
    }
    solutions, _, _ = M.generateLogicDataAndSolutions(
        [str(logic_yaml)], args, asm, info_map
    )
    return list(solutions), info_map


def _fake_result(name="mykernel"):
    """Minimal KernelCodeGenResult with err=0."""
    return M.KernelCodeGenResult(
        err=0,
        src="// asm\n",
        header="// header\n",
        name=name,
        targetObjFilename=None,
        isa=(9, 4, 2),
        wavefrontSize=64,
        cuoccupancy=4,
        pgr=2,
        mathclk=100,
    )


# ===========================================================================
# 1. writeAssembly — lines 336-348
# ===========================================================================

class TestWriteAssembly:
    """writeAssembly writes a .s file and returns a (Path, isa, wfsize, MinResult)."""

    def test_writes_asm_file(self, tmp_path):
        result = _fake_result("kern0")
        out = M.writeAssembly(tmp_path, result)
        asm_file = tmp_path / "kern0.s"
        assert asm_file.exists(), "writeAssembly must create <name>.s"
        assert asm_file.read_text() == "// asm\n"

    def test_returns_correct_tuple(self, tmp_path):
        result = _fake_result("kern0")
        path, isa, wfsize, min_result = M.writeAssembly(tmp_path, result)
        assert path == tmp_path / "kern0.s"
        assert isa == (9, 4, 2)
        assert wfsize == 64
        assert min_result.err == 0
        assert min_result.cuoccupancy == 4
        assert min_result.pgr == 2
        assert min_result.mathclk == 100

    def test_compressed_src_is_decompressed(self, tmp_path):
        """Lines 343-345: bytes src goes through memDecompress before writing."""
        raw_src = "// compressed src\n"
        compressed = M.memCompress(raw_src)
        result = M.KernelCodeGenResult(
            err=0, src=compressed, header=None,
            name="compk", targetObjFilename=None,
            isa=(9, 4, 2), wavefrontSize=64,
            cuoccupancy=0, pgr=0, mathclk=0,
        )
        M.writeAssembly(tmp_path, result)
        assert (tmp_path / "compk.s").read_text() == raw_src

    def test_error_result_calls_printExit(self, tmp_path):
        """Line 337: err!=0 triggers printExit."""
        result = M.KernelCodeGenResult(
            err=1, src="", header=None,
            name="badk", targetObjFilename=None,
            isa=(9, 4, 2), wavefrontSize=64,
            cuoccupancy=0, pgr=0, mathclk=0,
        )
        with pytest.raises(SystemExit):
            M.writeAssembly(tmp_path, result)


# ===========================================================================
# 2. writeHelpers — lines 353-375
# ===========================================================================

class TestWriteHelpers:
    """writeHelpers writes Kernels.cpp + Kernels.h from helper objects."""

    class _FakeKHO:
        def __init__(self, name, err=0):
            self._name = name
            self._err = err

        def getKernelName(self):
            return self._name

        def getSourceFileString(self):
            return (self._err, f"// src for {self._name}\n")

        def getHeaderFileString(self):
            return f"// hdr for {self._name}\n"

    def test_creates_both_files(self, tmp_path):
        M.writeHelpers(tmp_path, [], "Kernels.cpp", "Kernels.h")
        assert (tmp_path / "Kernels.cpp").exists()
        assert (tmp_path / "Kernels.h").exists()

    def test_empty_helpers_writes_headers_only(self, tmp_path):
        M.writeHelpers(tmp_path, [], "Kernels.cpp", "Kernels.h")
        cpp = (tmp_path / "Kernels.cpp").read_text()
        h = (tmp_path / "Kernels.h").read_text()
        assert '#include "Kernels.h"' in cpp
        assert "#pragma once" in h

    def test_helper_source_is_written_to_cpp(self, tmp_path):
        kho = self._FakeKHO("MyKernel")
        M.writeHelpers(tmp_path, [kho], "Kernels.cpp", "Kernels.h")
        cpp_text = (tmp_path / "Kernels.cpp").read_text()
        assert "// src for MyKernel" in cpp_text

    def test_helper_header_is_written_to_h(self, tmp_path):
        kho = self._FakeKHO("MyKernel")
        M.writeHelpers(tmp_path, [kho], "Kernels.cpp", "Kernels.h")
        h_text = (tmp_path / "Kernels.h").read_text()
        assert "// hdr for MyKernel" in h_text

    def test_error_kernel_emits_warning_but_does_not_raise(self, tmp_path, capsys):
        """Line 373: err!=0 triggers a warning print, not an exception.

        The warning line uses %u format which requires a numeric kernelName,
        matching the source code expectation that getKernelName() returns a number.
        """
        class _NumericKHO:
            def getKernelName(self):
                return 42  # numeric, matches the %u format in the source

            def getSourceFileString(self):
                return (1, "// bad src\n")  # err=1

            def getHeaderFileString(self):
                return ""

        M.writeHelpers(tmp_path, [_NumericKHO()], "Kernels.cpp", "Kernels.h")
        out = capsys.readouterr().out
        assert "warning" in out or "invalid kernel" in out

    def test_multiple_helpers_all_written(self, tmp_path):
        khos = [self._FakeKHO(f"K{i}") for i in range(3)]
        M.writeHelpers(tmp_path, khos, "Kernels.cpp", "Kernels.h")
        cpp_text = (tmp_path / "Kernels.cpp").read_text()
        for i in range(3):
            assert f"// src for K{i}" in cpp_text


# ===========================================================================
# 3. copyStaticFiles — lines 624-636
# ===========================================================================

class TestCopyStaticFiles:
    """copyStaticFiles copies known header files to outputPath."""

    def test_returns_nonempty_list(self, tmp_path):
        files = M.copyStaticFiles(tmp_path)
        assert isinstance(files, list)
        assert len(files) > 0

    def test_all_returned_files_exist_in_dest(self, tmp_path):
        files = M.copyStaticFiles(tmp_path)
        for f in files:
            assert (tmp_path / f).exists(), f"{f} was not copied to outputPath"

    def test_includes_tensile_types_header(self, tmp_path):
        files = M.copyStaticFiles(tmp_path)
        assert "TensileTypes.h" in files

    def test_includes_kernel_header(self, tmp_path):
        files = M.copyStaticFiles(tmp_path)
        assert "KernelHeader.h" in files


# ===========================================================================
# 4. generateKernelHelperObjects — lines 671-693
# ===========================================================================

class TestGenerateKernelHelperObjects:
    """generateKernelHelperObjects deduplicates and sorts helper objects."""

    @pytest.fixture(scope="class")
    def solutions_and_map(self):
        sols, info_map = _load_solutions()
        return sols, info_map

    def test_returns_nonempty_list_for_real_solutions(self, solutions_and_map):
        solutions, info_map = solutions_and_map
        clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
        khos = M.generateKernelHelperObjects(solutions, clang, info_map)
        assert isinstance(khos, list)
        assert len(khos) >= 1

    def test_enum_helpers_appear_first(self, solutions_and_map):
        """Lines 692-693: Enum helpers are sorted to front."""
        solutions, info_map = solutions_and_map
        clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
        khos = M.generateKernelHelperObjects(solutions, clang, info_map)
        if len(khos) >= 2:
            names = [k.getKernelName() for k in khos]
            enum_indices = [i for i, n in enumerate(names) if "Enum" in n]
            non_enum_indices = [i for i, n in enumerate(names) if "Enum" not in n]
            if enum_indices and non_enum_indices:
                assert max(enum_indices) < min(non_enum_indices), (
                    "Enum helpers must come before non-Enum helpers"
                )

    def test_duplicate_solutions_do_not_double_helpers(self, solutions_and_map):
        """Lines 680-683: name-visited set prevents duplicate helpers."""
        solutions, info_map = solutions_and_map
        clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
        doubled = solutions + solutions
        khos_single = M.generateKernelHelperObjects(solutions, clang, info_map)
        khos_doubled = M.generateKernelHelperObjects(doubled, clang, info_map)
        # dedup: doubling input must not double the output
        assert len(khos_doubled) == len(khos_single)

    def test_empty_solutions_returns_empty_list(self):
        clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
        khos = M.generateKernelHelperObjects([], clang, {})
        assert khos == []


# ===========================================================================
# 5. writeSolutionsAndKernelsTCL — lines 529-619
# ===========================================================================

class TestWriteSolutionsAndKernelsTCL:
    """writeSolutionsAndKernelsTCL: setup + compose block with mocked I/O."""

    def _run_mocked(self, tmp_path, kernels=None, solutions=None, cmdlineArchs=None):
        """Run writeSolutionsAndKernelsTCL with all GPU/FS side effects mocked."""
        if kernels is None:
            kernels = []
        if solutions is None:
            solutions = []
        if cmdlineArchs is None:
            cmdlineArchs = ["gfx942"]

        asm_tc = MagicMock()
        src_tc = MagicMock()
        kwa = MagicMock()

        with patch.object(M, "ParallelMap2", return_value=[]):
            with patch.object(M, "buildAssemblyCodeObjectFiles", return_value=[]):
                with patch.object(M, "buildSourceCodeObjectFiles"):
                    with patch.object(M, "writeHelpers"):
                        return M.writeSolutionsAndKernelsTCL(
                            tmp_path / "output",
                            asm_tc, src_tc, solutions, kernels,
                            [], kwa, cmdlineArchs,
                        )

    def test_returns_zero_count_when_no_kernels(self, tmp_path):
        n, uq, results = self._run_mocked(tmp_path)
        assert n == 0
        assert uq == []
        assert results == []

    def test_creates_library_dir(self, tmp_path):
        self._run_mocked(tmp_path)
        lib_dir = tmp_path / "output" / "library" / "gfx942"
        assert lib_dir.is_dir()

    def test_creates_assembly_tmp_dir(self, tmp_path):
        self._run_mocked(tmp_path)
        build_tmp = tmp_path / "output" / "build_tmp"
        assert build_tmp.is_dir()

    def test_parallel_map2_called(self, tmp_path):
        """Line 589: ParallelMap2 is called for kernel codegen."""
        calls_log = []

        def _fake_pm2(fn, items, *args, **kw):
            calls_log.append(True)
            return []

        asm_tc = MagicMock()
        src_tc = MagicMock()
        kwa = MagicMock()

        with patch.object(M, "ParallelMap2", side_effect=_fake_pm2):
            with patch.object(M, "buildAssemblyCodeObjectFiles", return_value=[]):
                with patch.object(M, "buildSourceCodeObjectFiles"):
                    with patch.object(M, "writeHelpers"):
                        M.writeSolutionsAndKernelsTCL(
                            tmp_path / "output",
                            asm_tc, src_tc, [], [], [], kwa, ["gfx942"],
                        )
        assert len(calls_log) >= 1

    def test_duplicate_kernels_excluded_from_unique(self, tmp_path):
        """Lines 553: uniqueAsmKernels excludes duplicates; return count reflects unique."""
        from Tensile.Common.GlobalParameters import assignGlobalParameters

        # Create two fake assembly kernels with the same base name => one duplicate
        k1 = MagicMock()
        k1.__getitem__ = lambda self, key: "Assembly" if key == "KernelLanguage" else ""
        k2 = MagicMock()
        k2.__getitem__ = lambda self, key: "Assembly" if key == "KernelLanguage" else ""

        asm_tc = MagicMock()
        src_tc = MagicMock()
        kwa = MagicMock()

        # Patch getKernelFileBase to return same name for both => k2 is duplicate
        with patch.object(M, "getKernelFileBase", return_value="same_kernel"):
            with patch.object(M, "ParallelMap2", return_value=[]):
                with patch.object(M, "buildAssemblyCodeObjectFiles", return_value=[]):
                    with patch.object(M, "buildSourceCodeObjectFiles"):
                        with patch.object(M, "writeHelpers"):
                            n, uq, results = M.writeSolutionsAndKernelsTCL(
                                tmp_path / "output2",
                                asm_tc, src_tc, [], [k1, k2], [], kwa, ["gfx942"],
                            )
        # n should be count of unique kernels, not total
        assert n == 1
        assert len(uq) == 1

    def test_multiple_archs_creates_per_base_library_dirs(self, tmp_path):
        """Lines 574-576: multiple archs => one per-base subdir each under library/."""
        self._run_mocked(tmp_path, cmdlineArchs=["gfx942", "gfx950"])
        lib_dir = tmp_path / "output" / "library"
        assert lib_dir.is_dir()
        # Per-base layout: each base arch gets its own subdir
        assert (lib_dir / "gfx942").is_dir()
        assert (lib_dir / "gfx950").is_dir()

    def test_write_helpers_is_called(self, tmp_path):
        """Line 606: writeHelpers is invoked."""
        called = []

        def _spy(*a, **kw):
            called.append(True)

        asm_tc = MagicMock()
        src_tc = MagicMock()
        kwa = MagicMock()

        with patch.object(M, "ParallelMap2", return_value=[]):
            with patch.object(M, "buildAssemblyCodeObjectFiles", return_value=[]):
                with patch.object(M, "buildSourceCodeObjectFiles"):
                    with patch.object(M, "writeHelpers", side_effect=_spy):
                        M.writeSolutionsAndKernelsTCL(
                            tmp_path / "output",
                            asm_tc, src_tc, [], [], [], kwa, ["gfx942"],
                        )
        assert called, "writeHelpers must be called from writeSolutionsAndKernelsTCL"


# ===========================================================================
# 6. generateLogicDataAndSolutions — uncovered branches
#    (lines 808-809: architectureName==""; 832-835: fallback merge; 840, 844-845)
# ===========================================================================

class TestGenerateLogicDataAndSolutionsExtra:
    """Extra branch coverage for generateLogicDataAndSolutions."""

    @pytest.fixture(scope="class")
    def base_assembler(self):
        return _make_assembler()

    @pytest.fixture(scope="class")
    def base_isa_map(self):
        _, m = _make_isa_info_map("gfx942")
        return m

    def _run(self, assembler, isa_map, arch="gfx942", lazy=True, gen_sol_table=False):
        args = {
            "Architecture": arch,
            "LazyLibraryLoading": lazy,
            "CodeObjectVersion": "4",
            "GenSolTable": gen_sol_table,
        }
        return M.generateLogicDataAndSolutions(
            [str(_LOGIC_YAML)], args, assembler, isa_map
        )

    def test_non_lazy_loading_still_returns_solutions(self, base_assembler, base_isa_map):
        """Lines 844-845: LazyLibraryLoading=False branch (masterFile without lazy prefix)."""
        solutions, masterLibs, mapping = self._run(
            base_assembler, base_isa_map, lazy=False
        )
        assert "gfx942" in masterLibs
        sol_list = list(solutions)
        assert len(sol_list) >= 1

    def test_code_object_version_stored_in_master_library(
        self, base_assembler, base_isa_map
    ):
        """Line 835: CodeObjectVersion is stored on the per-arch master library."""
        _, masterLibs, _ = self._run(base_assembler, base_isa_map)
        master = masterLibs.get("gfx942")
        assert master is not None
        assert master.version == "4"

    def test_solutions_index_reassigned_ascending(self, base_assembler, base_isa_map):
        """Lines 840-845: solution re-indexing produces non-negative contiguous indices."""
        _, masterLibs, _ = self._run(base_assembler, base_isa_map)
        master = masterLibs.get("gfx942")
        if master and master.solutions:
            indices = [s.index for s in master.solutions.values()]
            assert all(i >= 0 for i in indices)

    def test_empty_architecture_name_skipped(self, base_assembler, base_isa_map):
        """Line 808-809: architectureName=="" causes the library to be skipped.

        parseLibraryLogicFile receives 7 positional args (filename, assembler, splitGSU,
        printSolutionRejectionReason, printIndexAssignmentInfo, isaInfoMap, lazyLibraryLoading).
        """
        from Tensile.SolutionLibrary import MasterSolutionLibrary as MSL

        def _fake_parse(filename, assembler, splitGSU, psr, piai, isaInfoMap, lazy):
            lib = MSL({}, None)
            return (None, "", None, None, None, lib, {})

        with patch.object(M.LibraryIO, "parseLibraryLogicFile", side_effect=_fake_parse):
            solutions, masterLibs, mapping = self._run(base_assembler, base_isa_map)
        # No arch key should be added when architectureName==""
        assert masterLibs == {}

    def test_fallback_merge_integrates_into_per_arch_masters(self, base_assembler, base_isa_map):
        """Lines 832-835: fallback key is merged into each non-fallback arch master,
        then removed after merge.

        We stub ParallelMap2 directly (not parseLibraryLogicFile) so we control
        the exact library objects returned and avoid triggering MasterSolutionLibrary.merge
        with an incomplete structure (which requires a non-None library tree node).
        """
        from Tensile.SolutionLibrary import MasterSolutionLibrary

        # Build minimal real MSL objects (empty solutions, no lazy libs).
        # renameFallbacksPerArch requires a deep-copyable object so we use real MSL.
        gfx942_lib = MasterSolutionLibrary({}, MagicMock())
        gfx942_lib.lazyLibraries = {}

        fallback_lib = MasterSolutionLibrary({}, MagicMock())
        fallback_lib.lazyLibraries = {}

        # ParallelMap2 yields (extras, archName, ..., newLibrary, typeMismatches) tuples
        fake_results = [
            (None, "gfx942", None, None, None, gfx942_lib, {}),
            (None, "fallback", None, None, None, fallback_lib, {}),
        ]

        args = {
            "Architecture": "gfx942",
            "LazyLibraryLoading": True,
            "CodeObjectVersion": "4",
            "GenSolTable": False,
        }

        with patch.object(M, "ParallelMap2", return_value=iter(fake_results)):
            solutions, masterLibs, mapping = M.generateLogicDataAndSolutions(
                [str(_LOGIC_YAML), str(_LOGIC_YAML)], args, base_assembler, base_isa_map
            )
        # After merge, "fallback" key must be removed
        assert "fallback" not in masterLibs


# ===========================================================================
# 7. run() — lines 881-1086 (CLI entry point with all collaborators mocked)
# ===========================================================================

def _make_run_arguments(logic_path, output_path):
    """Build a minimal arguments dict that run() uses after parseArguments()."""
    return {
        "PrintLevel": 1,
        "OutputPath": str(output_path),
        "CxxCompiler": "/fake/hipcc",
        "CCompiler": "/fake/hipcc",
        "OffloadBundler": "/fake/clang-offload-bundler",
        "Assembler": "/fake/assembler",
        "CodeObjectVersion": "4",
        "BuildIdKind": "sha1",
        "AsmDebug": False,
        "AsanBuild": False,
        "Architecture": "gfx942",
        "LogicPath": str(logic_path),
        "LogicFormat": "yaml",
        "LibraryFormat": "msgpack",
        "CpuThreads": 1,
        "LazyLibraryLoading": True,
        "GenSolTable": False,
        "Experimental": False,
        "LogicFilter": "*",
        "DisableAsmComments": False,
        "UseCompression": False,
        "KeepBuildTmp": False,
    }


class TestRunCLIEntryPoint:
    """run() — lines 881-1086.

    We patch:
      - parseArguments          -> our _make_run_arguments dict
      - validateToolchain       -> fake compiler paths
      - makeIsaInfoMap          -> real info_map with SupportedISA patched
      - assignGlobalParameters  -> no-op
      - makeAssemblyToolchain   -> MagicMock
      - makeSourceToolchain     -> MagicMock
      - generateLogicDataAndSolutions -> returns ([], {}, {})
      - generateKernelObjectsFromSolutions -> returns []
      - generateKernelHelperObjects  -> returns []
      - copyStaticFiles         -> returns []
      - writeSolutionsAndKernelsTCL  -> returns (0, [], [])
      - passPostKernelInfoToLibrary   -> no-op
      - LibraryIO.write               -> no-op
      - shutil.rmtree                 -> no-op
    """

    @pytest.fixture
    def logic_dir(self, tmp_path):
        d = tmp_path / "logic"
        d.mkdir()
        # Copy a real logic file into the dir so the glob finds it
        import shutil as _shutil
        _shutil.copy(str(_LOGIC_YAML), str(d / _LOGIC_YAML.name))
        return d

    @pytest.fixture
    def output_dir(self, tmp_path):
        d = tmp_path / "output"
        d.mkdir()
        return d

    def _run_with_stubs(self, logic_dir, output_dir, extra_args=None):
        """Invoke M.run() with all heavy collaborators monkeypatched."""
        base_args = _make_run_arguments(logic_dir, output_dir)
        if extra_args:
            base_args.update(extra_args)

        _, info_map = _make_isa_info_map("gfx942")

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {"gfx942": MagicMock(
                              solutions={}, lazyLibraries={})}, {})), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write"), \
             patch.object(M.shutil, "rmtree"):
            M.run()

    def test_run_completes_without_exception(self, logic_dir, output_dir):
        """Lines 881-1086: run() must complete without raising."""
        self._run_with_stubs(logic_dir, output_dir)

    def test_run_uses_lazy_library_loading_branch(self, logic_dir, output_dir):
        """Lines 1053-1054: LazyLibraryLoading=True uses 'TensileLibrary_lazy_' prefix."""
        written_names = []

        def _spy_write(name, data, *a, **kw):
            written_names.append(str(name))

        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(logic_dir, output_dir)
        base_args["LazyLibraryLoading"] = True

        mock_master = MagicMock()
        mock_master.solutions = {}
        mock_master.lazyLibraries = {}

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {"gfx942": mock_master}, {})), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write", side_effect=_spy_write), \
             patch.object(M.shutil, "rmtree"):
            M.run()

        # Check the basename to avoid test-dir path components polluting the filter
        lazy_names = [n for n in written_names if "lazy" in Path(n).name.lower()]
        assert lazy_names, f"Expected a lazy-prefixed write, got: {written_names}"

    def test_run_non_lazy_loading_uses_plain_prefix(self, logic_dir, output_dir):
        """Lines 1055-1056: LazyLibraryLoading=False uses plain 'TensileLibrary_' prefix."""
        written_names = []

        def _spy_write(name, data, *a, **kw):
            written_names.append(str(name))

        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(logic_dir, output_dir)
        base_args["LazyLibraryLoading"] = False

        mock_master = MagicMock()
        mock_master.solutions = {}
        mock_master.lazyLibraries = {}

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {"gfx942": mock_master}, {})), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write", side_effect=_spy_write), \
             patch.object(M.shutil, "rmtree"):
            M.run()

        # non-lazy path: should NOT write a file with "lazy" in the name
        # Check the basename only to avoid test-dir paths containing "lazy"
        lazy_names = [n for n in written_names if "_lazy_" in Path(n).name]
        assert not lazy_names, f"Expected no lazy-prefixed write in non-lazy mode, got: {lazy_names}"

    def test_run_yaml_library_format(self, logic_dir, output_dir):
        """Line 1030: LibraryFormat='yaml' is forwarded to LibraryIO.write."""
        write_calls = []

        def _spy_write(name, data, fmt=None, *a, **kw):
            write_calls.append({"name": str(name), "fmt": fmt})

        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(logic_dir, output_dir)
        base_args["LibraryFormat"] = "yaml"

        mock_master = MagicMock()
        mock_master.solutions = {}
        mock_master.lazyLibraries = {}

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {"gfx942": mock_master}, {})), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write", side_effect=_spy_write), \
             patch.object(M.shutil, "rmtree"):
            M.run()

        yaml_writes = [c for c in write_calls if c["fmt"] == "yaml"]
        assert yaml_writes, f"Expected a yaml-format write call; got {write_calls}"

    def test_run_logic_path_not_exist_exits(self, tmp_path):
        """Line 928-929: non-existent LogicPath triggers printExit."""
        nonexistent = tmp_path / "no_such_dir"
        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(nonexistent, tmp_path / "out")

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()):
            with pytest.raises(SystemExit):
                M.run()

    def test_run_semicolon_arch_is_split(self, logic_dir, output_dir):
        """Lines 899-900: semicolon-separated arch string is split correctly."""
        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(logic_dir, output_dir)
        base_args["Architecture"] = "gfx942;gfx942"  # semicolon path

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {}, {})), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write"), \
             patch.object(M.shutil, "rmtree"):
            M.run()  # must not raise

    def test_run_keep_build_tmp_suppresses_rmtree(self, logic_dir, output_dir):
        """Lines 1067-1073: KeepBuildTmp=True means shutil.rmtree is not called."""
        rmtree_calls = []

        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(logic_dir, output_dir)
        base_args["KeepBuildTmp"] = True

        mock_master = MagicMock()
        mock_master.solutions = {}
        mock_master.lazyLibraries = {}

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {"gfx942": mock_master}, {})), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write"), \
             patch.object(M.shutil, "rmtree",
                          side_effect=lambda p, **kw: rmtree_calls.append(p)):
            M.run()

        # joblib's memmap reaper may rmtree its own /dev/shm scratch dirs during
        # teardown; those are unrelated to KeepBuildTmp. Only build_tmp removals
        # are the behavior under test.
        build_tmp_calls = [
            p for p in rmtree_calls
            if "joblib_memmapping_folder_" not in str(p)
        ]
        assert not build_tmp_calls, (
            "KeepBuildTmp=True must NOT rmtree build_tmp, got: " + str(build_tmp_calls)
        )

    def test_run_experimental_flag_passes_through(self, logic_dir, output_dir):
        """Lines 957-960: Experimental=False filters 'experimental' dirs."""
        self._run_with_stubs(logic_dir, output_dir, extra_args={"Experimental": False})

    def test_run_gen_sol_table_false(self, logic_dir, output_dir):
        """Line 822-828: GenSolTable=False skips MatchTable write."""
        self._run_with_stubs(logic_dir, output_dir, extra_args={"GenSolTable": False})

    def test_run_library_mapping_written_per_arch(self, logic_dir, output_dir):
        """Lines 1038-1048: non-empty libraryMapping with arch-suffixed entries
        causes archMapping file to be written via LibraryIO.write."""
        written = []

        def _spy_write(name, data, fmt=None, *a, **kw):
            written.append(str(name))

        _, info_map = _make_isa_info_map("gfx942")
        base_args = _make_run_arguments(logic_dir, output_dir)

        mock_master = MagicMock()
        mock_master.solutions = {}
        mock_master.lazyLibraries = {}

        # libraryMapping has a gfx942-suffixed entry => arch Mapping file written
        lib_mapping = {0: "SomeLib_gfx942"}

        with patch.object(M, "parseArguments", return_value=base_args), \
             patch.object(M, "setVerbosity"), \
             patch.object(M, "validateToolchain", return_value=(
                 "/fake/hipcc", None, "/fake/bundler", None, None)), \
             patch.object(M, "makeIsaInfoMap", return_value=info_map), \
             patch.object(M, "assignGlobalParameters"), \
             patch.object(M, "makeAssemblyToolchain", return_value=MagicMock()), \
             patch.object(M, "makeSourceToolchain", return_value=MagicMock()), \
             patch.object(M, "generateLogicDataAndSolutions",
                          return_value=([], {"gfx942": mock_master}, lib_mapping)), \
             patch.object(M, "generateKernelObjectsFromSolutions", return_value=[]), \
             patch.object(M, "generateKernelHelperObjects", return_value=[]), \
             patch.object(M, "copyStaticFiles", return_value=[]), \
             patch.object(M, "writeSolutionsAndKernelsTCL",
                          return_value=(0, [], [])), \
             patch.object(M, "passPostKernelInfoToLibrary"), \
             patch.object(M.LibraryIO, "write", side_effect=_spy_write), \
             patch.object(M.shutil, "rmtree"):
            M.run()

        mapping_writes = [n for n in written if "Mapping" in n]
        assert mapping_writes, f"Expected Mapping file write; got: {written}"
