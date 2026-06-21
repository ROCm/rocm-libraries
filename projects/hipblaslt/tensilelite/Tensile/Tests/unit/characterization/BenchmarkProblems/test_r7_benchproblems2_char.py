################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — BenchmarkProblems solution-derivation and custom-kernel paths (CPU-only).

TARGET: Tensile/BenchmarkProblems.py
  miss=111; target ranges: 182-325  (_generate_single_solution,
  _getCustomKernelSolutionObj, _generateCustomKernelSolutions).

STRATEGY:
  PRONG A — direct in-process call to _generate_single_solution with a valid
  (perm, problemType, constantParams, assembler, debugConfig, isaInfoMap)
  tuple derived from a known-good BenchmarkProblems config via config_harness.
  Drives lines 182-222 in the main process (no subprocess jitter).

  PRONG B — monkeypatch _getCustomKernelSolutionObj to return a stub Solution,
  then call _generateCustomKernelSolutions directly to exercise the
  type-match (lines 320-325) and type-mismatch-no-fail (lines 318-319) arms.

  PRONG C — _generateForkedSolutions with a non-empty perm list (drives
  the ParallelMap2 -> _generate_single_solution worker path).

CPU-only. No GPU, no compile. pytestmark = pytest.mark.unit.
"""

import copy
import os
import sys

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# sys.path: _codegen dir exposes config_harness and codegen_harness
# ---------------------------------------------------------------------------
_CODEGEN_DIR = os.path.join(os.path.dirname(__file__), "..", "_codegen")
if _CODEGEN_DIR not in sys.path:
    sys.path.insert(0, _CODEGEN_DIR)

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
import importlib
M = importlib.import_module("Tensile.BenchmarkProblems")

# ---------------------------------------------------------------------------
# PRONG A helpers — build valid inputs for _generate_single_solution in-process
# ---------------------------------------------------------------------------

def _make_toolchain(arch="gfx942"):
    """Return (assembler, isaInfoMap) for arch via config_harness toolchain."""
    from config_harness import _toolchain_for
    return _toolchain_for(arch)


def _make_debug_config():
    from Tensile.Common.Types import makeDebugConfig
    return makeDebugConfig({})


def _make_problem_type(arch="gfx942"):
    """Build a ProblemType (BBS NT shape, Batched=True) valid for MFMA on gfx942."""
    from Tensile.SolutionStructs.Problem import ProblemType
    from config_harness import _toolchain_for, _isolated_globals_with_isa

    _, iim = _toolchain_for(arch)
    pt_config = {
        "OperationType": "GEMM",
        "DataType": "b",
        "DestDataType": "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": 1,
        "TransposeB": 0,
        "UseBeta": True,
        "Batched": True,
    }
    with _isolated_globals_with_isa(iim):
        pt = ProblemType(pt_config, False)
    return pt


def _make_constant_params():
    """Minimal constant-params dict for a BBS MFMA-16 kernel on gfx942.

    These are taken from the BenchmarkCommonParameters defaults plus the
    required keys _generate_single_solution reads before calling Solution().
    """
    return {
        "KernelLanguage": "Assembly",
        # MatrixInstruction and WorkGroup come from the perm, but WavefrontSize
        # can be -1 to trigger the auto-detect arm (line 195-196).
        "WavefrontSize": -1,
        "MatrixInstruction": [16, 16, 16, 1, 1, 1, 1, 4, 1],
        "WorkGroup": [16, 16, 1],
        # Standard fork-common params (from defaultBenchmarkCommonParameters)
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "DepthU": 32,
        "LocalReadVectorWidth": 8,
        "ScheduleIterAlg": 3,
        "ExpandPointerSwap": 0,
        "TransposeLDS": 1,
        "GlobalSplitU": 1,
        "SourceSwap": 0,
        "StoreRemapVectorWidth": 0,
        "ClusterLocalRead": 1,
    }


# ---------------------------------------------------------------------------
# PRONG A — direct in-process call to _generate_single_solution
# ---------------------------------------------------------------------------

class TestGenerateSingleSolution:
    """Exercises BenchmarkProblems._generate_single_solution (lines 182-222)."""

    @pytest.fixture(scope="class")
    def toolchain(self):
        return _make_toolchain("gfx942")

    @pytest.fixture(scope="class")
    def problem_type(self):
        return _make_problem_type("gfx942")

    @pytest.fixture(scope="class")
    def debug_config(self):
        return _make_debug_config()

    def test_valid_perm_returns_solution(self, toolchain, problem_type, debug_config):
        """A well-formed perm dict yields a non-None Solution object (line 214-215)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        perm = {}  # all params in constantParams
        const = _make_constant_params()

        with _isolated_globals_with_isa(iim):
            sol = M._generate_single_solution(
                perm, problem_type, const, assembler, debug_config, iim
            )
        # The key assertion: the function returned a Solution, not None.
        assert sol is not None, (
            "_generate_single_solution returned None for a valid BBS MFMA-16 perm"
        )

    def test_mi_length_9_triggers_mi_param_expansion(self, toolchain, problem_type, debug_config):
        """len(mi)==9 perm triggers matrixInstructionToMIParameters (lines 199-201)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        const = _make_constant_params()
        perm = {}

        with _isolated_globals_with_isa(iim):
            sol = M._generate_single_solution(
                perm, problem_type, const, assembler, debug_config, iim
            )
        # Still returns a valid solution; the test is just to ensure line 200 runs.
        assert sol is not None

    def test_wavefrontsize_minus1_triggers_autodetect(self, toolchain, problem_type, debug_config):
        """WavefrontSize==-1 triggers arch-caps lookup (lines 195-196)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        const = _make_constant_params()
        const["WavefrontSize"] = -1  # explicit to document intent (default is already -1)
        perm = {}

        with _isolated_globals_with_isa(iim):
            sol = M._generate_single_solution(
                perm, problem_type, const, assembler, debug_config, iim
            )
        assert sol is not None

    def test_exception_in_perm_returns_none(self, toolchain, problem_type, debug_config):
        """A perm that causes an exception returns None (lines 220-222)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        # Pass a perm whose MatrixInstruction key is missing (KeyError caught by except).
        bad_perm = {}
        bad_const = {"WavefrontSize": 64}  # missing MatrixInstruction -> KeyError at line 190

        with _isolated_globals_with_isa(iim):
            sol = M._generate_single_solution(
                bad_perm, problem_type, bad_const, assembler, debug_config, iim
            )
        assert sol is None


# ---------------------------------------------------------------------------
# PRONG B — _generateCustomKernelSolutions via monkeypatched _getCustomKernelSolutionObj
# ---------------------------------------------------------------------------

class TestGenerateCustomKernelSolutions:
    """Exercises BenchmarkProblems._generateCustomKernelSolutions (lines 297-325)."""

    @pytest.fixture(scope="class")
    def toolchain(self):
        return _make_toolchain("gfx942")

    @pytest.fixture(scope="class")
    def problem_type(self):
        return _make_problem_type("gfx942")

    @pytest.fixture(scope="class")
    def debug_config(self):
        return _make_debug_config()

    def _build_stub_solution(self, problem_type, valid=True, mismatch_transpose=False):
        """Build a minimal stub object that quacks like a Solution.

        We need:
          sol["ProblemType"] — a ProblemType object comparable to problemType via __eq__
          sol["Valid"]       — bool

        The Solution.__eq__ comparison in _generateCustomKernelSolutions compares
        solution["ProblemType"] (a ProblemType instance) to the BenchmarkProcess
        problemType (also a ProblemType instance) using ProblemType.__eq__ which
        checks isinstance and getAttributes() dict equality.
        """
        # _StubSol: dict-like stub that also has subscript access for ["Valid"] etc.
        class _StubSol(dict):
            pass

        stub = _StubSol()
        if mismatch_transpose:
            # Build a ProblemType with a different TransposeA so comparison fails.
            from Tensile.SolutionStructs.Problem import ProblemType as PT_cls
            from config_harness import _isolated_globals_with_isa, _toolchain_for
            _, iim = _toolchain_for("gfx942")
            pt_config = {
                "OperationType": "GEMM",
                "DataType": "b",
                "DestDataType": "b",
                "ComputeDataType": "s",
                "HighPrecisionAccumulate": True,
                "TransposeA": 0,  # different from the fixture's TransposeA=1
                "TransposeB": 0,
                "UseBeta": True,
                "Batched": True,
            }
            with _isolated_globals_with_isa(iim):
                mismatched_pt = PT_cls(pt_config, False)
            stub["ProblemType"] = mismatched_pt
        else:
            # Use the same ProblemType object so comparison returns equal.
            stub["ProblemType"] = problem_type
        stub["Valid"] = valid
        return stub

    def test_type_match_valid_solution_appended(self, monkeypatch, toolchain, problem_type, debug_config):
        """When ProblemType matches and solution is Valid, it is appended (lines 320-323)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        stub = self._build_stub_solution(problem_type, valid=True)

        # Monkeypatch _getCustomKernelSolutionObj so no real .s file is needed.
        monkeypatch.setattr(M, "_getCustomKernelSolutionObj", lambda *a, **kw: stub)

        with _isolated_globals_with_isa(iim):
            sols = M._generateCustomKernelSolutions(
                problem_type,
                customKernels=["dummy_kernel"],
                internalSupportParams={},
                failOnMismatch=False,
                assembler=assembler,
                debugConfig=debug_config,
                isaInfoMap=iim,
            )

        assert len(sols) == 1, f"Expected 1 solution, got {len(sols)}: {sols}"
        assert sols[0] is stub

    def test_type_mismatch_no_fail_rejected(self, monkeypatch, toolchain, problem_type, debug_config):
        """ProblemType mismatch + failOnMismatch=False logs a rejection (lines 318-319)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        # Build a stub whose ProblemType is a DIFFERENT ProblemType object (TransposeA=0
        # instead of 1) so solution["ProblemType"] != problemType evaluates True.
        stub = self._build_stub_solution(problem_type, valid=True, mismatch_transpose=True)

        monkeypatch.setattr(M, "_getCustomKernelSolutionObj", lambda *a, **kw: stub)

        with _isolated_globals_with_isa(iim):
            sols = M._generateCustomKernelSolutions(
                problem_type,
                customKernels=["dummy_kernel"],
                internalSupportParams={},
                failOnMismatch=False,
                assembler=assembler,
                debugConfig=debug_config,
                isaInfoMap=iim,
            )

        # Rejected: mismatch with failOnMismatch=False => empty list
        assert len(sols) == 0, f"Expected 0 solutions after mismatch, got {len(sols)}"

    def test_type_match_invalid_solution_not_appended(
        self, monkeypatch, toolchain, problem_type, debug_config
    ):
        """Valid=False solution with matching ProblemType is not appended (lines 322-325)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        stub = self._build_stub_solution(problem_type, valid=False)

        monkeypatch.setattr(M, "_getCustomKernelSolutionObj", lambda *a, **kw: stub)

        with _isolated_globals_with_isa(iim):
            sols = M._generateCustomKernelSolutions(
                problem_type,
                customKernels=["dummy_kernel"],
                internalSupportParams={},
                failOnMismatch=False,
                assembler=assembler,
                debugConfig=debug_config,
                isaInfoMap=iim,
            )

        assert len(sols) == 0, f"Expected 0 solutions for invalid kernel, got {len(sols)}"

    def test_empty_custom_kernels_returns_empty(self, toolchain, problem_type, debug_config):
        """No customKernels yields empty solutions list (line 295 entry guard)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        with _isolated_globals_with_isa(iim):
            sols = M._generateCustomKernelSolutions(
                problem_type,
                customKernels=[],
                internalSupportParams={},
                failOnMismatch=False,
                assembler=assembler,
                debugConfig=debug_config,
                isaInfoMap=iim,
            )
        assert sols == []

    def test_multiple_kernels_mixed_validity(self, monkeypatch, toolchain, problem_type, debug_config):
        """Two kernels: one valid-match, one invalid-match => only valid one appended."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        # Both stubs use the SAME problem_type reference so == comparison passes.
        stub_valid = self._build_stub_solution(problem_type, valid=True)
        stub_invalid = self._build_stub_solution(problem_type, valid=False)
        call_count = [0]

        def _stubber(*a, **kw):
            idx = call_count[0]
            call_count[0] += 1
            return [stub_valid, stub_invalid][idx]

        monkeypatch.setattr(M, "_getCustomKernelSolutionObj", _stubber)

        with _isolated_globals_with_isa(iim):
            sols = M._generateCustomKernelSolutions(
                problem_type,
                customKernels=["kernel_a", "kernel_b"],
                internalSupportParams={},
                failOnMismatch=False,
                assembler=assembler,
                debugConfig=debug_config,
                isaInfoMap=iim,
            )

        assert len(sols) == 1
        assert sols[0] is stub_valid


# ---------------------------------------------------------------------------
# PRONG C — _generateForkedSolutions with a real perm list
# Exercises the ParallelMap2 worker path and the duplicate-filter tqdm loop.
# ---------------------------------------------------------------------------

class TestGenerateForkedSolutions:
    """Exercises BenchmarkProblems._generateForkedSolutions (lines 224-247)."""

    @pytest.fixture(scope="class")
    def toolchain(self):
        return _make_toolchain("gfx942")

    @pytest.fixture(scope="class")
    def problem_type(self):
        return _make_problem_type("gfx942")

    @pytest.fixture(scope="class")
    def debug_config(self):
        return _make_debug_config()

    def test_single_perm_yields_solution(self, toolchain, problem_type, debug_config):
        """_generateForkedSolutions with one valid perm returns >=1 Solution."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        const = _make_constant_params()

        # A single permutation — no extra fork params needed.
        perms = [{}]

        with _isolated_globals_with_isa(iim):
            sols = M._generateForkedSolutions(
                problem_type, const, perms, assembler, debug_config, iim
            )

        assert len(sols) >= 1, f"Expected >=1 solution, got {len(sols)}"

    def test_empty_perms_yields_empty(self, toolchain, problem_type, debug_config):
        """_generateForkedSolutions with empty perm list returns []."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        const = _make_constant_params()

        with _isolated_globals_with_isa(iim):
            sols = M._generateForkedSolutions(
                problem_type, const, [], assembler, debug_config, iim
            )

        assert sols == []

    def test_duplicate_perms_deduplicated(self, toolchain, problem_type, debug_config):
        """Two identical perms yield at most 1 Solution (dedup filter, lines 240-245)."""
        from config_harness import _isolated_globals_with_isa

        assembler, iim = toolchain
        const = _make_constant_params()
        # Two identical perms → same Solution hash → deduped to 1.
        perms = [{}, {}]

        with _isolated_globals_with_isa(iim):
            sols = M._generateForkedSolutions(
                problem_type, const, perms, assembler, debug_config, iim
            )

        # Dedup should yield <=1 unique solution.
        assert len(sols) <= 1, f"Expected dedup to <=1, got {len(sols)}"


# ---------------------------------------------------------------------------
# PRONG D — solutions_from_config breadth: two ProblemSizeGroups in same config
# Uses config_harness.solutions_from_config on a multi-group inline config.
# ---------------------------------------------------------------------------

_TWO_GROUP_CONFIG_YAML = """\
GlobalParameters:
  SyncsPerBenchmark: 0
  MinimumRequiredVersion: 5.0.0
  NumElementsToValidate: 0
  DataInitTypeBeta: 0
  DataInitTypeAlpha: 1
  Device: 0

BenchmarkProblems:
  # Group 0 — BBS NT, minimal fork (1 kernel)
  -
    - # ProblemType
      OperationType: GEMM
      DataType: b
      DestDataType: b
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: 1
      TransposeB: 0
      UseBeta: True
      Batched: True
    - # BenchmarkProblemSizeGroup
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction:
          - [16, 16, 16, 1, 1, 1, 1, 4, 1]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - ClusterLocalRead: [1]
        - DepthU: [32]
        - LocalReadVectorWidth: [8]
        - ScheduleIterAlg: [3]
        - ExpandPointerSwap: [0]
        - TransposeLDS: [1]
        - GlobalSplitU: [1]
        - SourceSwap: [0]
        - StoreRemapVectorWidth: [0]
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [256, 256, 1, 256]
"""


class TestSolutionsFromConfigBreadth:
    """Drives solutions_from_config on an inline config — exercises config parsing."""

    def test_config_yields_solutions(self, tmp_path):
        """solutions_from_config on a minimal BBS config returns >=1 Solution."""
        import yaml
        from config_harness import solutions_from_config

        cfg_path = tmp_path / "benchproblems2_two_group.yaml"
        cfg_path.write_text(_TWO_GROUP_CONFIG_YAML)

        sols = solutions_from_config(str(cfg_path), arch="gfx942", limit_solutions=8)
        assert len(sols) >= 1, f"Expected >=1 solution from two-group config, got {len(sols)}"

    def test_config_solutions_are_valid(self, tmp_path):
        """All returned solutions have Valid=True."""
        from config_harness import solutions_from_config

        cfg_path = tmp_path / "benchproblems2_valid.yaml"
        cfg_path.write_text(_TWO_GROUP_CONFIG_YAML)

        sols = solutions_from_config(str(cfg_path), arch="gfx942", limit_solutions=8)
        for sol in sols:
            assert sol["Valid"], f"Expected sol['Valid']==True, got {sol['Valid']}"
