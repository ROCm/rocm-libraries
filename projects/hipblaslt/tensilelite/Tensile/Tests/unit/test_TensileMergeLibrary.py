################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for TensileMergeLibrary using embedded YAML fixture data.

This test module uses embedded YAML strings that mirror the actual
gfx950 (list format) and gfx1250 (dict format) YAML structures.
"""

import pytest
from unittest.mock import patch
import yaml
from copy import deepcopy

from Tensile.TensileMergeLibrary import (
    createAccessor,
    getArchitectureFromData,
    isGfx1250,
    fixSizeInconsistencies,
    sanitizeSolutions,
    removeUnusedSolutions,
    removeDuplicatedSolutions,
    mergeLogic,
    syncDefaultParams,
    removeDefaultInitParams,
    findSolutionWithIndex,
    addKernel,
    reNameSolutions,
    msg,
    verbose,
    debug,
)


# =============================================================================
# Embedded YAML Fixture Data
# =============================================================================

# Simplified gfx950 format matching actual merge folder structure
# Structure: [version, scheduleName, archName, devices, problemType, solutions, indexOrder, exactLogic, rangeLogic, null, perfMetric, libraryType]
GFX950_YAML = """
- {MinimumRequiredVersion: 5.0.0}
- gfx950
- gfx950
- [Device 75a8]
- Activation: true
  ActivationType: hipblaslt_all
  AssignedDerivedParameters: true
  Batched: true
  ComputeDataType: 0
  DataType: 15
  DataTypeA: 15
  DataTypeB: 15
  DestDataType: 7
  HighPrecisionAccumulate: true
  Index0: 0
  Index1: 1
  IndexAssignmentsA: [3, 0, 2]
  IndexAssignmentsB: [3, 1, 2]
  IndexUnroll: 3
  IndicesBatch: [2]
  IndicesFree: [0, 1]
  IndicesSummation: [3]
  NumIndicesBatch: 1
  NumIndicesC: 3
  NumIndicesFree: 2
  NumIndicesSummation: 1
  OperationType: GEMM
  StridedBatched: true
  SupportUserArgs: true
  TotalIndices: 4
  TransposeA: true
  TransposeB: false
  UseBeta: true
  UseBias: 1
  UseScaleAB: Scalar
- - 1LDSBuffer: 0
    ActivationFused: true
    AssignedDerivedParameters: true
    AssignedProblemIndependentDerivedParameters: true
    BaseName: Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgNMVfVa0LuB8lRTcyOpGO2YH741LGRrePzbeHygp1IY8=
    BufferLoad: true
    BufferStore: true
    DepthU: 128
    GlobalSplitU: 0
    ISA: [9, 5, 0]
    Kernel: true
    KernelLanguage: Assembly
    KernelNameMin: Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_MT160x192x128_Test0
    MacroTile0: 160
    MacroTile1: 192
    NumThreads: 256
    SolutionIndex: 0
    SolutionNameMin: Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_Test0
    StaggerU: 0
    StaggerUMapping: 0
    StaggerUStride: 0
    Valid: true
    WavefrontSize: 64
    WorkGroup: [16, 16, 1]
    _staggerStrideShift: 0
  - 1LDSBuffer: 0
    ActivationFused: true
    AssignedDerivedParameters: true
    AssignedProblemIndependentDerivedParameters: true
    BaseName: Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgXzDbxA8LsiKAenzoLoqFOTbCc3GwRQe0GwhODFVZXaI=
    BufferLoad: true
    BufferStore: true
    DepthU: 128
    GlobalSplitU: 0
    ISA: [9, 5, 0]
    Kernel: true
    KernelLanguage: Assembly
    KernelNameMin: Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_MT160x192x128_Test1
    MacroTile0: 160
    MacroTile1: 192
    NumThreads: 256
    SolutionIndex: 1
    SolutionNameMin: Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_Test1
    StaggerU: 0
    StaggerUMapping: 0
    StaggerUStride: 0
    Valid: true
    WavefrontSize: 64
    WorkGroup: [16, 16, 1]
    _staggerStrideShift: 0
- [2, 3, 0, 1]
- - - [10240, 384, 1, 8192]
    - [0, 0.0]
  - - [10240, 336, 1, 8192]
    - [1, 0.0]
  - - [10240, 272, 1, 8192]
    - [0, 0.0]
- null
- null
- DeviceEfficiency
- Equality
"""

GFX1250_YAML = """
MinimumRequiredVersion: 5.0.0
ScheduleName: gfx1250
ArchitectureName: gfx1250
CUCount: null
DeviceNames: [Device 73f0]
ProblemType:
  OperationType: GEMM
  DataType: 7
  UseBeta: true
  Batched: true
  StridedBatched: true
  TransposeA: 0
  TransposeB: 0
DefaultSolution:
  GlobalSplitU: 1
  StaggerU: 32
  StaggerUMapping: 0
  StaggerUStride: 256
  DepthU: -1
  WorkGroup: [16, 16, 1]
Solutions:
- SolutionIndex: 0
  SolutionNameMin: Sol_gfx1250_0
  KernelNameMin: Kernel_gfx1250_0
  BaseName: Base_gfx1250_0
  StaggerU: 32
  StaggerUMapping: 0
  StaggerUStride: 256
  _staggerStrideShift: 2
  DepthU: 32
  MacroTile0: 16
  MacroTile1: 16
  WorkGroup: [16, 2, 1]
- SolutionIndex: 1
  SolutionNameMin: Sol_gfx1250_1
  KernelNameMin: Kernel_gfx1250_1
  BaseName: Base_gfx1250_1
  StaggerU: 32
  StaggerUMapping: 0
  StaggerUStride: 256
  _staggerStrideShift: 2
  DepthU: 64
  MacroTile0: 32
  MacroTile1: 32
  WorkGroup: [32, 4, 1]
IndexOrder: [2, 3, 0, 1]
ExactLogic:
- - [129, 129, 1, 129]
  - [0, 0.0]
- - [128, 128, 1, 128]
  - [1, 0.0]
- - [256, 256, 1, 256]
  - [0, 0.0]
RangeLogic: null
PerfMetric: DeviceEfficiency
LibraryType: Matching
Library:
  indexOrder: [2, 3, 0, 1]
  table:
  - - [129, 129, 1, 129]
    - [0, 0.0]
  - - [128, 128, 1, 128]
    - [1, 0.0]
  - - [256, 256, 1, 256]
    - [0, 0.0]
  distance: GridBased
"""


# =============================================================================
# Helper Functions to Load Embedded YAML
# =============================================================================

def load_gfx950_data():
    """Load gfx950 test data (list format) from embedded YAML string.
    
    The gfx950 YAML is already a list, so safe_load returns the list directly.
    """
    return yaml.safe_load(GFX950_YAML)


def load_gfx1250_data():
    """Load gfx1250 test data (dict format) from embedded YAML string."""
    return yaml.safe_load(GFX1250_YAML)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def gfx950_data():
    """Load gfx950 test data (list format)."""
    return load_gfx950_data()


@pytest.fixture
def gfx1250_data():
    """Load gfx1250 test data (dict format)."""
    return load_gfx1250_data()


@pytest.fixture
def gfx950_accessor(gfx950_data):
    """Create DataAccessor for gfx950 data."""
    return createAccessor(gfx950_data)


@pytest.fixture
def gfx1250_accessor(gfx1250_data):
    """Create DataAccessor for gfx1250 data."""
    return createAccessor(gfx1250_data)


# =============================================================================
# Test Classes
# =============================================================================

class TestEmbeddedYamlLoading:
    """Tests to verify embedded YAML loads correctly."""

    def test_gfx950_loads_as_list(self, gfx950_data):
        """gfx950 data loads as a list."""
        assert isinstance(gfx950_data, list)
        assert len(gfx950_data) >= 8  # Minimum expected elements

    def test_gfx1250_loads_as_dict(self, gfx1250_data):
        """gfx1250 data loads as a dict."""
        assert isinstance(gfx1250_data, dict)
        assert "Solutions" in gfx1250_data
        assert "ExactLogic" in gfx1250_data
        assert "DefaultSolution" in gfx1250_data


class TestDataAccessorWithFixtures:
    """Tests for DataAccessor using embedded fixture data."""

    def test_gfx950_accessor_identifies_list_format(self, gfx950_accessor):
        """gfx950 accessor identifies list format."""
        assert gfx950_accessor.isList is True
        assert gfx950_accessor.isDict is False

    def test_gfx1250_accessor_identifies_dict_format(self, gfx1250_accessor):
        """gfx1250 accessor identifies dict format."""
        assert gfx1250_accessor.isDict is True
        assert gfx1250_accessor.isList is False

    def test_gfx950_get_solutions(self, gfx950_accessor):
        """gfx950 accessor can get solutions."""
        solutions = gfx950_accessor.getSolutions()
        assert len(solutions) == 2
        assert solutions[0]["SolutionIndex"] == 0
        assert solutions[1]["SolutionIndex"] == 1
        assert "Cijk_Alik_Bljk" in solutions[0]["SolutionNameMin"]

    def test_gfx1250_get_solutions(self, gfx1250_accessor):
        """gfx1250 accessor can get solutions."""
        solutions = gfx1250_accessor.getSolutions()
        assert len(solutions) == 2
        assert solutions[0]["SolutionNameMin"] == "Sol_gfx1250_0"
        assert solutions[1]["SolutionNameMin"] == "Sol_gfx1250_1"

    def test_gfx950_get_exact_logic(self, gfx950_accessor):
        """gfx950 accessor can get ExactLogic."""
        logic = gfx950_accessor.getExactLogic()
        assert len(logic) == 3
        # Check first size entry
        assert logic[0][0] == [10240, 384, 1, 8192]
        assert logic[0][1] == [0, 0.0]

    def test_gfx1250_get_exact_logic(self, gfx1250_accessor):
        """gfx1250 accessor can get ExactLogic."""
        logic = gfx1250_accessor.getExactLogic()
        assert len(logic) == 3
        # Check first size entry
        assert logic[0][0] == [129, 129, 1, 129]
        assert logic[0][1] == [0, 0.0]

    def test_gfx950_no_default_solution(self, gfx950_accessor):
        """gfx950 does not have DefaultSolution."""
        assert gfx950_accessor.hasDefaultSolution() is False
        assert gfx950_accessor.getDefaultSolution() is None

    def test_gfx1250_has_default_solution(self, gfx1250_accessor):
        """gfx1250 has DefaultSolution."""
        assert gfx1250_accessor.hasDefaultSolution() is True
        default = gfx1250_accessor.getDefaultSolution()
        assert default["GlobalSplitU"] == 1
        assert default["StaggerU"] == 32


class TestArchitectureDetectionWithFixtures:
    """Tests for architecture detection using embedded fixtures."""

    def test_gfx950_architecture(self, gfx950_data):
        """Detect architecture from gfx950 data."""
        arch = getArchitectureFromData(gfx950_data)
        assert arch == "gfx950"

    def test_gfx1250_architecture(self, gfx1250_data):
        """Detect architecture from gfx1250 data."""
        arch = getArchitectureFromData(gfx1250_data)
        assert arch == "gfx1250"

    def test_gfx950_is_not_gfx1250(self, gfx950_data):
        """gfx950 is not identified as gfx1250."""
        assert isGfx1250(gfx950_data) is False

    def test_gfx1250_is_gfx1250(self, gfx1250_data):
        """gfx1250 is identified as gfx1250."""
        assert isGfx1250(gfx1250_data) is True


class TestFixSizeInconsistenciesWithFixtures:
    """Tests for fixSizeInconsistencies using embedded fixture data."""

    def test_gfx950_sizes_preserved(self, gfx950_accessor):
        """gfx950 sizes are preserved (no duplicates in fixture)."""
        logic = gfx950_accessor.getExactLogic()
        result, count = fixSizeInconsistencies(deepcopy(logic), "gfx950")
        assert count == 3  # All three unique sizes preserved

    def test_gfx1250_sizes_preserved(self, gfx1250_accessor):
        """gfx1250 sizes are preserved (no duplicates in fixture)."""
        logic = gfx1250_accessor.getExactLogic()
        result, count = fixSizeInconsistencies(deepcopy(logic), "gfx1250")
        assert count == 3  # All three unique sizes preserved

    def test_deduplication_with_gfx950_format(self):
        """Verify deduplication works with gfx950-style sizes."""
        sizes = [
            [[10240, 384, 1, 8192], [0, 0.0]],
            [[10240, 336, 1, 8192], [1, 0.0]],
            [[10240, 384, 1, 8192], [2, 1.0]],  # Duplicate of first
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 2  # Only 2 unique sizes
        # Verify no duplicate sizes
        result_sizes = [tuple(r[0]) for r in result]
        assert len(set(result_sizes)) == 2

    def test_deduplication_with_gfx1250_format(self):
        """Verify deduplication works with gfx1250-style sizes."""
        sizes = [
            [[129, 129, 1, 129], [0, 0.0]],
            [[128, 128, 1, 128], [1, 0.0]],
            [[129, 129, 1, 129], [2, 1.0]],  # Duplicate of first
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 2  # Only 2 unique sizes


class TestSanitizeSolutionsWithFixtures:
    """Tests for sanitizeSolutions using embedded fixture data."""

    def test_gfx950_sanitize_preserves_stagger_zero(self, gfx950_data):
        """gfx950 solutions have StaggerU=0, sanitize should preserve this."""
        accessor = createAccessor(deepcopy(gfx950_data))
        solutions = accessor.getSolutions()
        
        # Verify initial state
        assert solutions[0]["StaggerU"] == 0
        
        sanitizeSolutions(accessor)
        
        # After sanitize, all stagger params should be 0
        assert solutions[0]["StaggerUMapping"] == 0
        assert solutions[0]["StaggerUStride"] == 0
        assert solutions[0]["_staggerStrideShift"] == 0

    def test_gfx1250_sanitize_preserves_nonzero_stagger(self, gfx1250_data):
        """gfx1250 solutions have StaggerU=32, sanitize should preserve dependent params."""
        accessor = createAccessor(deepcopy(gfx1250_data))
        solutions = accessor.getSolutions()
        
        # Verify initial state
        assert solutions[0]["StaggerU"] == 32
        orig_stride = solutions[0]["StaggerUStride"]
        
        sanitizeSolutions(accessor)
        
        # After sanitize, stagger params should be preserved
        assert solutions[0]["StaggerUStride"] == orig_stride


class TestRemoveUnusedSolutionsWithFixtures:
    """Tests for removeUnusedSolutions using embedded fixture data."""

    def test_gfx950_all_solutions_used(self, gfx950_data):
        """In gfx950 data, all solutions are used."""
        accessor = createAccessor(deepcopy(gfx950_data))
        _, num_removed = removeUnusedSolutions(accessor)
        # Solution 0 is used twice, solution 1 once - both are used
        assert num_removed == 0

    def test_gfx1250_all_solutions_used(self, gfx1250_data):
        """In gfx1250 data, all solutions are used."""
        accessor = createAccessor(deepcopy(gfx1250_data))
        _, num_removed = removeUnusedSolutions(accessor)
        # Solution 0 is used twice, solution 1 once - both are used
        assert num_removed == 0

    def test_gfx950_remove_unused(self, gfx950_data):
        """Add an unused solution to gfx950 and verify it's removed."""
        data = deepcopy(gfx950_data)
        solutions = data[5]  # Solutions list
        solutions.append({
            "SolutionIndex": 2,
            "SolutionNameMin": "Unused_Sol",
            "KernelNameMin": "Unused_Kernel",
            "StaggerU": 0,
        })
        accessor = createAccessor(data)
        
        _, num_removed = removeUnusedSolutions(accessor)
        
        assert num_removed == 1
        assert len(accessor.getSolutions()) == 2


class TestRemoveDuplicatedSolutionsWithFixtures:
    """Tests for removeDuplicatedSolutions using embedded fixture data."""

    def test_gfx950_no_duplicates(self, gfx950_data):
        """gfx950 data has no duplicate solutions."""
        accessor = createAccessor(deepcopy(gfx950_data))
        _, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(accessor)
        assert num_removed == 0
        assert num_solutions == 2

    def test_gfx1250_no_duplicates(self, gfx1250_data):
        """gfx1250 data has no duplicate solutions."""
        accessor = createAccessor(deepcopy(gfx1250_data))
        _, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(accessor)
        assert num_removed == 0
        assert num_solutions == 2


class TestMergeLogicWithFixtures:
    """Tests for mergeLogic using embedded fixture data."""

    def test_merge_gfx950_with_new_size(self, gfx950_data):
        """Merge adds new size to gfx950 data."""
        ori_accessor = createAccessor(deepcopy(gfx950_data))
        
        # Create incremental data with a new size
        inc_data = deepcopy(gfx950_data)
        inc_solutions = inc_data[5]
        inc_logic = inc_data[7]
        # Add a new size
        inc_logic.append([[9999, 999, 1, 9999], [0, 0.0]])
        inc_accessor = createAccessor(inc_data)
        
        merged_data, num_sizes_added, _, _ = mergeLogic(
            ori_accessor, inc_accessor, forceMerge=False
        )
        
        assert num_sizes_added == 1
        merged_accessor = createAccessor(merged_data)
        assert len(merged_accessor.getExactLogic()) == 4

    def test_merge_gfx1250_with_new_size(self, gfx1250_data):
        """Merge adds new size to gfx1250 data."""
        ori_accessor = createAccessor(deepcopy(gfx1250_data))
        
        # Create incremental data with a new size
        inc_data = deepcopy(gfx1250_data)
        inc_data["ExactLogic"].append([[512, 512, 1, 512], [0, 0.0]])
        inc_accessor = createAccessor(inc_data)
        
        merged_data, num_sizes_added, _, _ = mergeLogic(
            ori_accessor, inc_accessor, forceMerge=False
        )
        
        assert num_sizes_added == 1
        merged_accessor = createAccessor(merged_data)
        assert len(merged_accessor.getExactLogic()) == 4

    def test_merge_gfx950_better_efficiency_replaces(self, gfx950_data):
        """Better efficiency solution replaces original in gfx950."""
        ori_accessor = createAccessor(deepcopy(gfx950_data))
        
        # Create incremental with better efficiency for existing size
        inc_data = deepcopy(gfx950_data)
        inc_data[7][0][1][1] = 2.0  # Improve efficiency from 0.0 to 2.0
        inc_data[5][0]["SolutionNameMin"] = "Better_Sol"
        inc_accessor = createAccessor(inc_data)
        
        merged_data, num_sizes_added, num_solutions_added, _ = mergeLogic(
            ori_accessor, inc_accessor, forceMerge=False
        )
        
        # No new sizes, but solution should be replaced
        assert num_sizes_added == 0

    def test_merge_gfx1250_force_merge(self, gfx1250_data):
        """Force merge replaces even with worse efficiency."""
        ori_data = deepcopy(gfx1250_data)
        ori_data["ExactLogic"][0][1][1] = 5.0  # High efficiency
        ori_accessor = createAccessor(ori_data)
        
        inc_data = deepcopy(gfx1250_data)
        inc_data["ExactLogic"][0][1][1] = 0.0  # Lower efficiency
        inc_data["Solutions"][0]["SolutionNameMin"] = "Forced_Sol"
        inc_accessor = createAccessor(inc_data)
        
        merged_data, _, _, _ = mergeLogic(
            ori_accessor, inc_accessor, forceMerge=True
        )
        
        merged_accessor = createAccessor(merged_data)
        # Forced solution should be present
        solution_names = [s["SolutionNameMin"] for s in merged_accessor.getSolutions()]
        assert "Forced_Sol" in solution_names


class TestDefaultSolutionFunctionsWithFixtures:
    """Tests for DefaultSolution-related functions using gfx1250 data."""

    def test_sync_default_params(self, gfx1250_data):
        """syncDefaultParams handles default changes."""
        data = deepcopy(gfx1250_data)
        orig_defaults = {"StaggerU": 32, "TestParam": 100}
        inc_defaults = {"StaggerU": 64, "TestParam": 200}  # Changed
        
        syncDefaultParams(data, orig_defaults, inc_defaults)
        
        # Function should add old defaults to solutions when they change

    def test_remove_default_init_params(self, gfx1250_data):
        """removeDefaultInitParams removes params matching default."""
        data = deepcopy(gfx1250_data)
        # Add a parameter that matches default
        data["Solutions"][0]["GlobalSplitU"] = 1
        data["DefaultSolution"]["GlobalSplitU"] = 1
        
        removeDefaultInitParams(data)
        
        # GlobalSplitU should be removed from solution since it matches default
        assert "GlobalSplitU" not in data["Solutions"][0]

    def test_remove_cu_count_from_default(self, gfx1250_data):
        """CUCount is removed from DefaultSolution."""
        data = deepcopy(gfx1250_data)
        data["DefaultSolution"]["CUCount"] = 304
        
        removeDefaultInitParams(data)
        
        assert "CUCount" not in data["DefaultSolution"]


class TestFindSolutionWithIndexWithFixtures:
    """Tests for findSolutionWithIndex using embedded fixture data."""

    def test_find_gfx950_solution_by_index(self, gfx950_accessor):
        """Find solution by index in gfx950 data."""
        solutions = gfx950_accessor.getSolutions()
        
        result = findSolutionWithIndex(solutions, 0)
        assert result["SolutionIndex"] == 0
        assert "Test0" in result["SolutionNameMin"]
        
        result = findSolutionWithIndex(solutions, 1)
        assert result["SolutionIndex"] == 1
        assert "Test1" in result["SolutionNameMin"]

    def test_find_gfx1250_solution_by_index(self, gfx1250_accessor):
        """Find solution by index in gfx1250 data."""
        solutions = gfx1250_accessor.getSolutions()
        
        result = findSolutionWithIndex(solutions, 0)
        assert result["SolutionNameMin"] == "Sol_gfx1250_0"
        
        result = findSolutionWithIndex(solutions, 1)
        assert result["SolutionNameMin"] == "Sol_gfx1250_1"


class TestCrossFormatOperations:
    """Tests for operations that might span different formats."""

    def test_accessor_set_and_get_solutions(self, gfx950_data, gfx1250_data):
        """Test setting and getting solutions on both formats."""
        # gfx950
        gfx950_accessor = createAccessor(deepcopy(gfx950_data))
        new_sol = {"SolutionIndex": 99, "SolutionNameMin": "New_Sol", "KernelNameMin": "New_K"}
        solutions = gfx950_accessor.getSolutions()
        solutions.append(new_sol)
        gfx950_accessor.setSolutions(solutions)
        assert len(gfx950_accessor.getSolutions()) == 3
        
        # gfx1250
        gfx1250_accessor = createAccessor(deepcopy(gfx1250_data))
        solutions = gfx1250_accessor.getSolutions()
        solutions.append(new_sol)
        gfx1250_accessor.setSolutions(solutions)
        assert len(gfx1250_accessor.getSolutions()) == 3

    def test_accessor_set_and_get_exact_logic(self, gfx950_data, gfx1250_data):
        """Test setting and getting ExactLogic on both formats."""
        new_entry = [[1, 1, 1, 1], [0, 0.0]]
        
        # gfx950
        gfx950_accessor = createAccessor(deepcopy(gfx950_data))
        logic = gfx950_accessor.getExactLogic()
        logic.append(new_entry)
        gfx950_accessor.setExactLogic(logic)
        assert len(gfx950_accessor.getExactLogic()) == 4
        
        # gfx1250
        gfx1250_accessor = createAccessor(deepcopy(gfx1250_data))
        logic = gfx1250_accessor.getExactLogic()
        logic.append(new_entry)
        gfx1250_accessor.setExactLogic(logic)
        assert len(gfx1250_accessor.getExactLogic()) == 4


@pytest.mark.unit
class TestAddKernel:
    """Test addKernel function"""

    def test_add_new_kernel(self):
        """Test adding a new kernel"""
        solutionPool = []
        solDict = {}
        solution = {"SolutionNameMin": "kernel_1", "data": "test"}

        pool, sol_dict, index = addKernel(solutionPool, solDict, solution)

        assert len(pool) == 1
        assert index == 0
        assert pool[0]["SolutionIndex"] == 0
        assert "kernel_1" in sol_dict

    def test_reuse_existing_kernel(self):
        """Test reusing an existing kernel"""
        solution_existing = {"SolutionNameMin": "kernel_1", "SolutionIndex": 0, "data": "test"}
        solutionPool = [solution_existing]
        solDict = {"kernel_1": solution_existing}

        solution_new = {"SolutionNameMin": "kernel_1", "data": "new"}
        pool, sol_dict, index = addKernel(solutionPool, solDict, solution_new)

        assert len(pool) == 1  # Should not add duplicate
        assert index == 0

    def test_add_multiple_kernels(self):
        """Test adding multiple different kernels"""
        solutionPool = []
        solDict = {}

        sol1 = {"SolutionNameMin": "kernel_1"}
        sol2 = {"SolutionNameMin": "kernel_2"}

        pool, sol_dict, idx1 = addKernel(solutionPool, solDict, sol1)
        pool, sol_dict, idx2 = addKernel(pool, sol_dict, sol2)

        assert len(pool) == 2
        assert idx1 == 0
        assert idx2 == 1


@pytest.mark.unit
class TestMessageFunctions:
    """Test msg, verbose, and debug functions"""

    @patch('builtins.print')
    def test_msg_output(self, mock_print):
        """Test msg function"""
        msg("test", "message")
        assert mock_print.call_count >= 1

    @patch('builtins.print')
    @patch('Tensile.TensileMergeLibrary.verbosity', 1)
    def test_verbose_output_when_enabled(self, mock_print):
        """Test verbose function when verbosity >= 1"""
        verbose("test", "message")
        assert mock_print.call_count >= 1

    @patch('builtins.print')
    @patch('Tensile.TensileMergeLibrary.verbosity', 0)
    def test_verbose_no_output_when_disabled(self, mock_print):
        """Test verbose function when verbosity < 1"""
        verbose("test", "message")
        assert mock_print.call_count == 0

    @patch('builtins.print')
    @patch('Tensile.TensileMergeLibrary.verbosity', 2)
    def test_debug_output_when_enabled(self, mock_print):
        """Test debug function when verbosity >= 2"""
        debug("test", "message")
        assert mock_print.call_count >= 1

    @patch('builtins.print')
    @patch('Tensile.TensileMergeLibrary.verbosity', 1)
    def test_debug_no_output_when_disabled(self, mock_print):
        """Test debug function when verbosity < 2"""
        debug("test", "message")
        assert mock_print.call_count == 0


@pytest.mark.unit
class TestReNameSolutions:
    """Test reNameSolutions function"""

    @patch('Tensile.TensileMergeLibrary.getSolutionNameMin')
    @patch('Tensile.TensileMergeLibrary.getKernelNameMin')
    @patch('Tensile.TensileMergeLibrary.assignParameterWithDefault')
    def test_rename_solutions(self, mock_assign, mock_kernel_name, mock_solution_name):
        """Test renaming solutions using DataAccessor"""
        mock_solution_name.return_value = "sol_min"
        mock_kernel_name.return_value = "kernel_min"

        problem_type = {"OperationType": "GEMM"}
        solutions = [{"key": "value"}]
        data = [None, None, None, None, problem_type, solutions]
        accessor = createAccessor(data)

        reNameSolutions(accessor)

        assert solutions[0]["SolutionNameMin"] == "sol_min"
        assert solutions[0]["KernelNameMin"] == "kernel_min"
        assert "ProblemType" not in solutions[0]


@pytest.mark.unit
class TestMainFunction:
    """Test main function argument parsing"""

    @patch('Tensile.TensileMergeLibrary.avoidRegressions')
    @patch('sys.argv', ['script', '/orig', '/inc', '/out', '-v', '2'])
    def test_main_with_arguments(self, mock_avoid):
        """Test main function with command line arguments"""
        from Tensile.TensileMergeLibrary import main

        main()

        mock_avoid.assert_called_once()
        args = mock_avoid.call_args[0]
        assert args[0] == '/orig'
        assert args[1] == '/inc'
        assert args[2] == '/out'

    @patch('Tensile.TensileMergeLibrary.avoidRegressions')
    @patch('sys.argv', ['script', '/orig', '/inc', '/out', '--force_merge', 'true'])
    def test_main_with_force_merge_true(self, mock_avoid):
        """Test main with force_merge=true"""
        from Tensile.TensileMergeLibrary import main

        main()

        args = mock_avoid.call_args[0]
        assert args[3] == True  # forceMerge

    @patch('Tensile.TensileMergeLibrary.avoidRegressions')
    @patch('sys.argv', ['script', '/orig', '/inc', '/out', '--force_merge', 'false'])
    def test_main_with_force_merge_false(self, mock_avoid):
        """Test main with force_merge=false"""
        from Tensile.TensileMergeLibrary import main

        main()

        args = mock_avoid.call_args[0]
        assert args[3] == False  # forceMerge

    @patch('Tensile.TensileMergeLibrary.avoidRegressions')
    @patch('sys.argv', ['script', '/orig', '/inc', '/out', '--no_eff'])
    def test_main_with_no_eff_flag(self, mock_avoid):
        """Test main with --no_eff flag"""
        from Tensile.TensileMergeLibrary import main

        main()

        kwargs = mock_avoid.call_args[0]
        assert kwargs[4] == True  # no_eff
