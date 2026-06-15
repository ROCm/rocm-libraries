# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
from typing import Any
from Tensile import LibraryIO
from Tensile.CustomYamlLoader import DEFAULT_YAML_LOADER, load_yaml_stream

from Tensile.TensileMergeLibrary import (
    ensurePath,
    allFiles,
    fixSizeInconsistencies,
    addKernel,
    sanitizeSolutions,
    reNameSolutions,
    removeUnusedSolutions,
    removeDuplicatedSolutions,
    loadData,
    compareDestFolderToYaml,
    compareProblemType,
    msg,
    verbose,
    debug,
    findSolutionWithIndex,
    mergeLogic,
    convertToDict,
    createAccessor,
    getArchitectureFromData,
    isDictBasedArchitecture,
    removeDefaultInitParams,
    normalizeDictLibraryLayout,
    syncDefaultParams,
)


def _load_arch_data(arch: str) -> Any:
    """Load architecture fixture data from a test_data YAML file.

    Args:
        arch: Architecture tag (``"gfx950"`` or ``"gfx1250"``).

    Returns:
        Parsed Python object (list for gfx950, dict for gfx1250).

    Raises:
        FileNotFoundError: If no ``merge_logic_{arch}.yaml`` exists in test_data.
    """
    yaml_file = Path(__file__).parent / "test_data" / f"merge_logic_{arch}.yaml"
    return load_yaml_stream(yaml_file, DEFAULT_YAML_LOADER)


def _append_new_size(data: Any, arch: str) -> None:
    """Append one new exact-logic size in either format."""
    if arch == "gfx950":
        data[7].append([[100, 200, 1, 300], [0, 0.0]])
    else:
        data["ExactLogic"].append([[512, 512, 1, 512], [0, 0.0]])


def _minimal_gfx1250_dict_logic(
    *,
    library_type: str = "GridBased",
    library_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a tiny dict-format logic root for ``normalizeDictLibraryLayout`` tests.

    Args:
        library_type: Initial top-level ``LibraryType`` string.
        library_block: Optional ``Library`` sub-dict; when None, no ``Library`` key.

    Returns:
        A mutable dict with ``ArchitectureName`` ``gfx1250`` so
        ``isDictBasedArchitecture`` succeeds.

    Raises:
        None.
    """
    d: dict[str, Any] = {
        "ArchitectureName": "gfx1250",
        "LibraryType": library_type,
        "Solutions": [],
        "ExactLogic": [],
    }
    if library_block is not None:
        d["Library"] = library_block
    return d


@pytest.fixture(scope="module")
def gfx950_data() -> list[Any]:
    return _load_arch_data("gfx950")


@pytest.fixture(scope="module")
def gfx1250_data() -> dict[str, Any]:
    return _load_arch_data("gfx1250")


@pytest.fixture(scope="module")
def gfx950_accessor(gfx950_data: list[Any]) -> Any:
    return createAccessor(gfx950_data)


@pytest.fixture(scope="module")
def gfx1250_accessor(gfx1250_data: dict[str, Any]) -> Any:
    return createAccessor(gfx1250_data)


@pytest.fixture(scope="module", params=["gfx950", "gfx1250"])
def arch_data(request: pytest.FixtureRequest) -> tuple[str, Any]:
    return request.param, _load_arch_data(request.param)


@pytest.fixture(scope="module")
def arch_accessor(arch_data: tuple[str, Any]) -> tuple[str, Any]:
    name, data = arch_data
    return name, createAccessor(data)

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

    def test_accessor_identifies_format(self, arch_accessor):
        """Accessor correctly identifies list vs dict format for both architectures."""
        name, accessor = arch_accessor
        assert accessor.isList == (name == "gfx950")
        assert accessor.isDict == (name == "gfx1250")

    def test_get_solutions(self, arch_accessor):
        """Accessor can get solutions for both architectures."""
        name, accessor = arch_accessor
        solutions = accessor.getSolutions()
        assert len(solutions) == 2
        assert solutions[0]["SolutionIndex"] == 0
        assert solutions[1]["SolutionIndex"] == 1
        if name == "gfx950":
            assert "Cijk_Alik_Bljk" in solutions[0]["SolutionNameMin"]
        else:
            assert solutions[0]["SolutionNameMin"] == "Sol_gfx1250_0"
            assert solutions[1]["SolutionNameMin"] == "Sol_gfx1250_1"

    def test_get_exact_logic(self, arch_accessor):
        """Accessor can get ExactLogic for both architectures."""
        name, accessor = arch_accessor
        logic = accessor.getExactLogic()
        assert len(logic) == 3
        assert logic[0][1] == [0, 0.0]
        if name == "gfx950":
            assert logic[0][0] == [10240, 384, 1, 8192]
        else:
            assert logic[0][0] == [129, 129, 1, 129]

    def test_set_unknown_list_key_raises(self, gfx950_accessor):
        with pytest.raises(ValueError, match="Invalid key"):
            gfx950_accessor.set("InvalidKey", "value")

    @pytest.mark.parametrize(
        "fixture_name, expected",
        [("gfx950_accessor", False), ("gfx1250_accessor", True)],
    )
    def test_default_solution_presence(
        self, request: pytest.FixtureRequest, fixture_name: str, expected: bool
    ):
        """DefaultSolution is present only for gfx1250."""
        accessor = request.getfixturevalue(fixture_name)
        assert accessor.hasDefaultSolution() is expected
        if expected:
            default = accessor.getDefaultSolution()
            assert default["GlobalSplitU"] == 1
            assert default["StaggerU"] == 32
        else:
            assert accessor.getDefaultSolution() is None

    def test_set_default_solution_on_dict(self, gfx1250_data: dict[str, Any]) -> None:
        """``setDefaultSolution`` writes ``DefaultSolution`` for dict-format data."""
        data = deepcopy(gfx1250_data)
        accessor = createAccessor(data)
        new_default = {"DepthU": 99}
        accessor.setDefaultSolution(new_default)
        assert accessor.getDefaultSolution() == new_default
        assert data["DefaultSolution"] == new_default


class TestArchitectureDetectionWithFixtures:
    """Tests for architecture detection using embedded fixtures."""

    def test_architecture_name(self, arch_data):
        """Architecture name is correctly detected for both formats."""
        name, data = arch_data
        assert getArchitectureFromData(data) == name


    def test_is_dict_based_architecture(self, arch_data):
        """isDictBasedArchitecture returns True only for configured dict-based architectures."""
        name, data = arch_data
        assert isDictBasedArchitecture(data) == (name == "gfx1250")

    def test_is_dict_based_architecture_custom_list(self, gfx950_data: list[Any]) -> None:
        """Optional ``dictArchs`` overrides global ``dictBasedArchitectures``."""
        assert isDictBasedArchitecture(gfx950_data) is False
        assert isDictBasedArchitecture(gfx950_data, dictArchs=["gfx950"]) is True
        assert isDictBasedArchitecture(gfx950_data, dictArchs=["gfx942"]) is False


class TestStripDictBasedLibraryLayout:
    """Tests for ``normalizeDictLibraryLayout`` canonical dict layout."""

    def test_skips_non_dict_architecture(self) -> None:
        """Non dict-based architecture short-circuits without mutation."""
        data = {
            "ArchitectureName": "gfx942",
            "LibraryType": "Matching",
            "Library": {"distance": "GridBased"},
        }
        assert normalizeDictLibraryLayout(data) is False
        assert "Library" in data

    def test_strips_library_and_sets_distance_from_library_block(self) -> None:
        """``Library.distance`` becomes top-level ``LibraryType``; ``Library`` removed."""
        data = _minimal_gfx1250_dict_logic(
            library_type="Matching",
            library_block={"distance": "Equality"},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "Equality"
        assert "Library" not in data

    def test_matching_without_library_is_noop(self) -> None:
        """``LibraryType: Matching`` without ``Library`` is not rewritten (non-canonical)."""
        data = _minimal_gfx1250_dict_logic(library_type="Matching")
        assert normalizeDictLibraryLayout(data) is False
        assert data["LibraryType"] == "Matching"
        assert "Library" not in data

    def test_freesize_preserved(self) -> None:
        """``FreeSize`` stays when ``Library`` has no usable ``distance`` (only block removed)."""
        data = _minimal_gfx1250_dict_logic(
            library_type="FreeSize",
            library_block={},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "FreeSize"
        assert "Library" not in data

    def test_range_from_library_distance(self) -> None:
        """``Library.distance: Range`` is promoted to top-level ``LibraryType``."""
        data = _minimal_gfx1250_dict_logic(
            library_type="Matching",
            library_block={"distance": "Range"},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "Range"
        assert "Library" not in data

    def test_prediction_toplevel_empty_library(self) -> None:
        """``Prediction`` passes through the non-distance branch while dropping ``Library``."""
        data = _minimal_gfx1250_dict_logic(
            library_type="Prediction",
            library_block={},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "Prediction"
        assert "Library" not in data


class TestCompareProblemType:
    """Tests for ``compareProblemType`` (ProblemType equality gate).

    The real ``ProblemType`` constructor rejects the trimmed fixture
    ``ProblemType`` dict; tests patch it with a lightweight stand-in.
    """

    def test_matching_problem_types_do_not_exit(self) -> None:
        """Identical ``ProblemType`` dicts do not call ``sys.exit``."""
        pt = {"OperationType": "GEMM", "Batched": True}
        ori = createAccessor({"ProblemType": deepcopy(pt)})
        inc = createAccessor({"ProblemType": deepcopy(pt)})

        class _FakeProblemType:
            def __init__(self, state: Any, _assign_gpus: bool) -> None:
                self.state = deepcopy(state)

        with patch("Tensile.TensileMergeLibrary.ProblemType", _FakeProblemType), patch(
            "Tensile.TensileMergeLibrary.problemTypeToEnum", lambda _pt: None
        ):
            compareProblemType(ori, inc)

    def test_mismatch_exits(self) -> None:
        """Differing ``ProblemType`` after normalization triggers ``sys.exit``."""
        ori = createAccessor({"ProblemType": {"OperationType": "GEMM", "Batched": True}})
        inc = createAccessor({"ProblemType": {"OperationType": "GEMM", "Batched": False}})

        class _FakeProblemType:
            def __init__(self, state: Any, _assign_gpus: bool) -> None:
                self.state = deepcopy(state)

        with patch("Tensile.TensileMergeLibrary.ProblemType", _FakeProblemType), patch(
            "Tensile.TensileMergeLibrary.problemTypeToEnum", lambda _pt: None
        ):
            with pytest.raises(SystemExit, match="ProblemType"):
                compareProblemType(ori, inc)


class TestFixSizeInconsistenciesWithFixtures:
    """Tests for fixSizeInconsistencies using embedded fixture data."""

    def test_sizes_preserved(self, arch_accessor):
        """Sizes are preserved (no duplicates in fixture) for both architectures."""
        name, accessor = arch_accessor
        logic = accessor.getExactLogic()
        result, count = fixSizeInconsistencies(deepcopy(logic), name)
        assert count == 3  # All three unique sizes preserved

    @pytest.mark.parametrize("sizes", [
        [[[10240, 384, 1, 8192], [0, 0.0]], 
         [[10240, 336, 1, 8192], [1, 0.0]], 
         [[10240, 384, 1, 8192], [2, 1.0]]]
    ])
    def test_deduplication(self, sizes):
        """Duplicate sizes are collapsed to unique entries regardless of format."""
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 2
        assert len({tuple(r[0]) for r in result}) == 2


class TestSolutionCleanup:
    """Tests for removeUnusedSolutions and removeDuplicatedSolutions."""

    def test_all_solutions_used(self, arch_data):
        """All solutions are used for both architectures."""
        _, data = arch_data
        accessor = createAccessor(deepcopy(data))
        _, num_removed = removeUnusedSolutions(accessor)
        # Solution 0 is used twice, solution 1 once - both are used
        assert num_removed == 0

    def test_remove_unused(self, arch_data):
        """Add an unused solution and verify it is removed for both architectures."""
        _, data = arch_data
        accessor = createAccessor(deepcopy(data))
        solutions = accessor.getSolutions()
        solutions.append({
            "SolutionIndex": 99,
            "SolutionNameMin": "Unused_Sol",
            "KernelNameMin": "Unused_Kernel",
            "StaggerU": 0,
        })
        accessor.setSolutions(solutions)
        _, num_removed = removeUnusedSolutions(accessor)
        assert num_removed == 1
        assert len(accessor.getSolutions()) == 2


    def test_no_duplicates(self, arch_data):
        """Data has no duplicate solutions for both architectures."""
        _, data = arch_data
        accessor = createAccessor(deepcopy(data))
        _, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(accessor)
        assert num_removed == 0
        assert num_solutions == 2

    def test_remove_duplicate_solution_names_keeps_first(self, gfx1250_data: dict[str, Any]) -> None:
        """Duplicate ``SolutionNameMin`` entries collapse to the first solution."""
        data = deepcopy(gfx1250_data)
        dup = deepcopy(data["Solutions"][0])
        dup["SolutionIndex"] = 1
        data["Solutions"] = [data["Solutions"][0], dup]
        data["ExactLogic"][0][1][0] = 0
        data["ExactLogic"][1][1][0] = 1
        accessor = createAccessor(data)
        _, num_removed, num_solutions, _ = removeDuplicatedSolutions(accessor)
        assert num_removed == 1
        assert num_solutions == 1

    def test_sanitize_solutions_sets_stagger_dependent_params(self, arch_data):
        """sanitizeSolutions zeroes dependent stagger params when StaggerU is zero."""
        _, data = arch_data
        accessor = createAccessor(deepcopy(data))
        solutions = accessor.getSolutions()
        solutions[0]["StaggerU"] = 0
        solutions[0]["StaggerUMapping"] = 9
        solutions[0]["StaggerUStride"] = 123
        solutions[0]["_staggerStrideShift"] = 7
        accessor.setSolutions(solutions)

        sanitizeSolutions(accessor)

        sanitized = accessor.getSolutions()[0]
        assert sanitized["StaggerUMapping"] == 0
        assert sanitized["StaggerUStride"] == 0
        assert sanitized["_staggerStrideShift"] == 0


class TestMergeLogicWithFixtures:
    """Tests for mergeLogic using embedded fixture data."""

    @pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
    def test_merge_with_new_size(self, arch):
        """Merge adds one new size for both formats."""
        ori_data = _load_arch_data(arch)
        inc_data = deepcopy(ori_data)
        _append_new_size(inc_data, arch)
        ori_accessor = createAccessor(deepcopy(ori_data))
        inc_accessor = createAccessor(inc_data)
        merged_data, num_sizes_added, _, _ = mergeLogic(ori_accessor, inc_accessor, forceMerge=False)
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
        
        # No new sizes, but solution should be replaced with higher efficiency
        assert num_sizes_added == 0
        assert num_solutions_added >= 1
        merged_accessor = createAccessor(merged_data)
        first_logic = merged_accessor.getExactLogic()[0]
        assert first_logic[1][1] == 2.0
        sol = findSolutionWithIndex(merged_accessor.getSolutions(), first_logic[1][0])
        assert sol["SolutionNameMin"] == "Better_Sol"

    def test_merge_no_eff_zeros_stored_efficiency(self, gfx950_data: list[Any]) -> None:
        """``noEff=True`` forces stored efficiency to 0.0 on merged sizes."""
        ori_accessor = createAccessor(deepcopy(gfx950_data))
        inc_data = deepcopy(gfx950_data)
        _append_new_size(inc_data, "gfx950")
        inc_accessor = createAccessor(inc_data)
        merged_data, _, _, _ = mergeLogic(ori_accessor, inc_accessor, forceMerge=False, noEff=True)
        merged_accessor = createAccessor(merged_data)
        for _size, (_idx, eff) in merged_accessor.getExactLogic():
            assert eff == 0.0

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
        """syncDefaultParams runs without error when defaults change between libraries."""
        data = deepcopy(gfx1250_data)
        orig_defaults = {"StaggerU": 32, "TestParam": 100}
        inc_defaults = {"StaggerU": 64, "TestParam": 200}
        syncDefaultParams(data, orig_defaults, inc_defaults)
        # When a default changes, the old value should be pinned onto solutions
        # that previously relied on it. Verify solutions are still present.
        assert len(data["Solutions"]) == 2

    def test_sync_default_params_identical_defaults_no_op(self, gfx1250_data: dict[str, Any]) -> None:
        """When default maps are equal, ``syncDefaultParams`` returns immediately."""
        data = deepcopy(gfx1250_data)
        before = deepcopy(data["Solutions"])
        syncDefaultParams(data, {"StaggerU": 32}, {"StaggerU": 32})
        assert data["Solutions"] == before

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

    def test_find_solution_out_of_order_list(self) -> None:
        """Uses linear search when ``SolutionIndex`` does not match list position."""
        solutions = [
            {"SolutionIndex": 1, "SolutionNameMin": "B"},
            {"SolutionIndex": 0, "SolutionNameMin": "A"},
        ]
        assert findSolutionWithIndex(solutions, 0)["SolutionNameMin"] == "A"
        assert findSolutionWithIndex(solutions, 1)["SolutionNameMin"] == "B"

    def test_find_solution_by_index(self, arch_accessor):
        """Find solution by index for both architectures."""
        name, accessor = arch_accessor
        solutions = accessor.getSolutions()

        result0 = findSolutionWithIndex(solutions, 0)
        result1 = findSolutionWithIndex(solutions, 1)
        assert result0["SolutionIndex"] == 0
        assert result1["SolutionIndex"] == 1
        if name == "gfx950":
            assert "Test0" in result0["SolutionNameMin"]
            assert "Test1" in result1["SolutionNameMin"]
        else:
            assert result0["SolutionNameMin"] == "Sol_gfx1250_0"
            assert result1["SolutionNameMin"] == "Sol_gfx1250_1"


class TestLibraryTypeAccessor:
    """Tests for ``getLibraryType`` (Equality / GridBased vs list index 11)."""

    def test_get_library_type_dict_uses_top_level(self, gfx1250_data):
        """Dict-format fixture exposes GridBased via top-level ``LibraryType``."""
        accessor = createAccessor(gfx1250_data)
        assert accessor.getLibraryType() == "GridBased"

    def test_get_library_type_list_uses_index_eleven(self, gfx950_data):
        """List-format fixture keeps Equality at legacy index 11."""
        accessor = createAccessor(gfx950_data)
        assert accessor.getLibraryType() == "Equality"

    @pytest.mark.parametrize(
        "dest_dir,expect_exit",
        [
            ("/path/to/GridBased", False),
            ("/path/to/Equality", True),
        ],
    )
    def test_compare_dest_folder_to_yaml_library_type(
        self, gfx1250_data, dest_dir: str, expect_exit: bool
    ) -> None:
        """compareDestFolderToYaml matches dest folder to ``getLibraryType()`` (GridBased)."""
        accessor = createAccessor(gfx1250_data)
        if expect_exit:
            with pytest.raises(SystemExit):
                compareDestFolderToYaml(dest_dir, "logic.yaml", accessor)
        else:
            compareDestFolderToYaml(dest_dir, "logic.yaml", accessor)

    def test_compare_dest_folder_exits_when_library_type_unset(self) -> None:
        """``compareDestFolderToYaml`` exits when ``getLibraryType()`` is empty."""
        data = _minimal_gfx1250_dict_logic(library_type="FreeSize")
        accessor = createAccessor(data)
        assert accessor.getLibraryType() is None
        with pytest.raises(SystemExit, match="Empty YAML attribute"):
            compareDestFolderToYaml("/any/GridBased", "logic.yaml", accessor)


class TestCrossFormatOperations:
    """Tests for set/get round-trips on accessor for both formats."""

    def test_accessor_set_and_get_solutions(self, arch_data):
        """Setting and getting solutions round-trips correctly for both formats."""
        _, data = arch_data
        accessor = createAccessor(deepcopy(data))
        new_sol = {"SolutionIndex": 99, "SolutionNameMin": "New_Sol", "KernelNameMin": "New_K"}
        solutions = accessor.getSolutions()
        solutions.append(new_sol)
        accessor.setSolutions(solutions)
        assert len(accessor.getSolutions()) == 3

    def test_accessor_set_and_get_exact_logic(self, arch_data):
        """Setting and getting ExactLogic round-trips correctly for both formats."""
        _, data = arch_data
        accessor = createAccessor(deepcopy(data))
        new_entry = [[1, 1, 1, 1], [0, 0.0]]
        logic = accessor.getExactLogic()
        logic.append(new_entry)
        accessor.setExactLogic(logic)
        assert len(accessor.getExactLogic()) == 4



@pytest.mark.unit
class TestEnsurePath:
    """Test ensurePath function"""

    def test_create_new_directory(self):
        """Test creating a new directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "test_dir", "nested")
            result = ensurePath(new_path)
            assert os.path.exists(new_path)
            assert result == new_path

    def test_existing_directory(self):
        """Test with existing directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensurePath(tmpdir)
            assert os.path.exists(tmpdir)
            assert result == tmpdir


@pytest.mark.unit
class TestAllFiles:
    """Test allFiles function"""

    def test_find_yaml_files_in_directory(self):
        """Test finding YAML files in a directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(os.path.join(tmpdir, "file1.yaml")).touch()
            Path(os.path.join(tmpdir, "file2.YAML")).touch()
            Path(os.path.join(tmpdir, "file3.txt")).touch()

            files = allFiles(tmpdir)
            yaml_names = [os.path.basename(f) for f in files]

            assert len(files) == 2
            assert "file1.yaml" in yaml_names
            assert "file2.YAML" in yaml_names
            assert "file3.txt" not in yaml_names

    def test_empty_directory(self):
        """Test with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = allFiles(tmpdir)
            assert files == []

    def test_nested_directories(self):
        """Test finding YAML files in nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure - but not as a subdirectory with yaml extension
            # because allFiles recursively searches directories, not files ending in .yaml
            Path(os.path.join(tmpdir, "top.yaml")).touch()
            Path(os.path.join(tmpdir, "another.yaml")).touch()

            files = allFiles(tmpdir)
            # Should find both .yaml files
            assert len(files) == 2


@pytest.mark.unit
class TestFixSizeInconsistencies:
    """Test fixSizeInconsistencies function"""

    def test_remove_duplicates(self):
        """Test removing duplicate sizes"""
        sizes = [
            ([1, 2, 3, 4, 5, 6, 7, 8], 0),
            ([1, 2, 3, 4], 1),
            ([1, 2, 3, 4, 9, 10, 11, 12], 2),
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        # All 3 sizes truncate/normalize to [1, 2, 3, 4], so should deduplicate to 1 unique size
        # - [1,2,3,4,5,6,7,8] -> [1,2,3,4] (removes last 4)
        # - [1,2,3,4] -> [1,2,3,4] (no change)
        # - [1,2,3,4,9,10,11,12] -> [1,2,3,4] (removes last 4)
        assert count == 1
        assert len(result) == 1
        assert result[0][0] == [1, 2, 3, 4]

    @patch('Tensile.TensileMergeLibrary.verbose')
    def test_remove_duplicates_verbose_message(self, mock_verbose):
        """Test that verbose message is triggered with correct count when duplicates are removed"""
        sizes = [
            ([1, 2, 3, 4, 5, 6, 7, 8], 0),
            ([1, 2, 3, 4], 1),
            ([1, 2, 3, 4, 9, 10, 11, 12], 2),
        ]
        result, count = fixSizeInconsistencies(sizes, "test")

        # Verify verbose was called with correct count: origNumSizes - numSize = 3 - 1 = 2
        mock_verbose.assert_called_once_with(2, "duplicate size(s) removed from", "test", "logic file")

    def test_no_duplicates(self):
        """Test with no duplicates"""
        sizes = [
            ([1, 2, 3], 0),
            ([4, 5, 6], 1),
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 2


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
class TestFindSolutionWithIndex:
    """Test findSolutionWithIndex function"""

    def test_find_solution_at_correct_index(self):
        """Test finding solution when index matches position"""
        solutions = [
            {"SolutionIndex": 0, "name": "sol0"},
            {"SolutionIndex": 1, "name": "sol1"},
            {"SolutionIndex": 2, "name": "sol2"},
        ]

        result = findSolutionWithIndex(solutions, 1)
        assert result["name"] == "sol1"

    def test_find_solution_with_search(self):
        """Test finding solution when index doesn't match position"""
        solutions = [
            {"SolutionIndex": 5, "name": "sol5"},
            {"SolutionIndex": 10, "name": "sol10"},
            {"SolutionIndex": 15, "name": "sol15"},
        ]

        result = findSolutionWithIndex(solutions, 10)
        assert result["name"] == "sol10"

    def test_find_nonexistent_solution(self):
        """Test finding non-existent solution raises assertion"""
        solutions = [
            {"SolutionIndex": 0, "name": "sol0"},
        ]

        with pytest.raises(AssertionError):
            findSolutionWithIndex(solutions, 99)


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


class TestDataAccessorListEdges:
    """Branch coverage for list-format ``DataAccessor.get`` / ``set``."""

    def test_get_returns_none_when_index_beyond_list(self) -> None:
        """``get`` returns None when the mapped index is past the list end."""
        short: list[Any] = [None] * 5
        accessor = createAccessor(short)
        assert accessor.get("LibraryType") is None

    def test_set_extends_short_list(self) -> None:
        """``set`` pads a short list with ``None`` until the target index exists."""
        short: list[Any] = [{"MinimumRequiredVersion": "1.0"}]
        accessor = createAccessor(short)
        accessor.set("Solutions", [])
        assert len(short) > 5
        assert accessor.get("Solutions") == []


class TestConvertToDictAndLoadData:
    """``convertToDict`` / ``loadData`` integration with on-disk YAML."""

    def test_convert_list_fixture_to_dict(self, gfx950_data: list[Any]) -> None:
        """Legacy list fixture converts to dict via ``parseLibraryLogicList``."""
        out = convertToDict(deepcopy(gfx950_data), "fixture.yaml")
        assert isinstance(out, dict)
        assert "Solutions" in out
        assert isinstance(out["Solutions"], list)

    def test_convert_dict_is_noop(self, gfx1250_data: dict[str, Any]) -> None:
        """Dict input is returned unchanged (same object)."""
        d = deepcopy(gfx1250_data)
        assert convertToDict(d, "any.yaml") is d

    def test_load_data_list_gfx950_no_migration(self, tmp_path: Path, gfx950_data: list[Any]) -> None:
        """List-format non-dict arch: ``loadData`` returns data without dict migration.

        Also checks that ``LibraryIO.writeYAML`` → ``loadData`` preserves key
        structural fields (architecture, solution and exact-logic counts).
        """
        src = deepcopy(gfx950_data)
        accessor_src = createAccessor(src)
        out_file = tmp_path / "logic.yaml"
        LibraryIO.writeYAML(
            str(out_file),
            src,
            explicit_start=False,
            explicit_end=False,
            sort_keys=True,
        )
        fn, data, migrated = loadData(str(out_file))
        assert fn == str(out_file)
        assert isinstance(data, list)
        assert migrated is False
        accessor_dst = createAccessor(data)
        assert getArchitectureFromData(src) == getArchitectureFromData(data)
        assert len(accessor_src.getSolutions()) == len(accessor_dst.getSolutions())
        assert len(accessor_src.getExactLogic()) == len(accessor_dst.getExactLogic())
        assert (
            accessor_src.getSolutions()[0]["SolutionIndex"]
            == accessor_dst.getSolutions()[0]["SolutionIndex"]
        )

    def test_load_data_dict_gfx1250(self, tmp_path: Path, gfx1250_data: dict[str, Any]) -> None:
        """``loadData`` reads dict YAML for a dict-based architecture.

        Also checks that ``LibraryIO.writeYAML`` → ``loadData`` preserves key
        structural fields after any dict normalization.
        """
        src = deepcopy(gfx1250_data)
        accessor_src = createAccessor(src)
        out_file = tmp_path / "logic.yaml"
        LibraryIO.writeYAML(
            str(out_file),
            src,
            explicit_start=False,
            explicit_end=False,
            sort_keys=False,
        )
        fn, data, migrated = loadData(str(out_file))
        assert fn == str(out_file)
        assert isinstance(data, dict)
        assert "Solutions" in data
        assert isinstance(migrated, bool)
        accessor_dst = createAccessor(data)
        assert getArchitectureFromData(src) == getArchitectureFromData(data)
        assert len(accessor_src.getSolutions()) == len(accessor_dst.getSolutions())
        assert len(accessor_src.getExactLogic()) == len(accessor_dst.getExactLogic())
        assert (
            accessor_src.getSolutions()[0]["SolutionIndex"]
            == accessor_dst.getSolutions()[0]["SolutionIndex"]
        )
