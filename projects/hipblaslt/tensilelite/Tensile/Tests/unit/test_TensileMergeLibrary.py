# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from Tensile import LibraryIO
from Tensile.CustomYamlLoader import DEFAULT_YAML_LOADER, load_yaml_stream
from Tensile.TensileMergeLibrary import (
    addKernel,
    allFiles,
    avoidRegressions,
    compareDestFolderToYaml,
    compareProblemType,
    convertToDict,
    debug,
    ensurePath,
    findSolutionWithIndex,
    fixSizeInconsistencies,
    loadData,
    mergeLogic,
    msg,
    normalizeDictLibraryLayout,
    removeDefaultInitParams,
    removeDuplicatedSolutions,
    removeUnusedSolutions,
    reNameSolutions,
    sanitizeSolutions,
    syncDefaultParams,
    verbose,
)

_TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _load_fixture(name: str) -> Any:
    """Load raw YAML fixture data without dict conversion.

    Args:
        name: Base name of the fixture file (without ``.yaml``).

    Returns:
        Parsed Python object from the YAML file.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
    """
    yaml_file = _TEST_DATA_DIR / f"{name}.yaml"
    return load_yaml_stream(yaml_file, DEFAULT_YAML_LOADER)


def _append_new_exact_logic_size(data: dict[str, Any]) -> None:
    """Append one new exact-logic size entry to dict-format logic data.

    Args:
        data: Dict-format library logic root to mutate in place.

    Returns:
        None.

    Raises:
        KeyError: If ``ExactLogic`` is missing from ``data``.
    """
    data["ExactLogic"].append([[512, 512, 1, 512], [0, 0.0]])


def _minimal_dict_logic(
    *,
    library_type: str = "GridBased",
    library_block: dict[str, Any] | None = None,
    architecture_name: str = "gfx1250",
) -> dict[str, Any]:
    """Build a tiny dict-format logic root for layout normalization tests.

    Args:
        library_type: Initial top-level ``LibraryType`` string.
        library_block: Optional ``Library`` sub-dict; when None, no ``Library`` key.
        architecture_name: Value for ``ArchitectureName``.

    Returns:
        A mutable dict with empty ``Solutions`` and ``ExactLogic``.

    Raises:
        None.
    """
    d: dict[str, Any] = {
        "ArchitectureName": architecture_name,
        "LibraryType": library_type,
        "Solutions": [],
        "ExactLogic": [],
    }
    if library_block is not None:
        d["Library"] = library_block
    return d


@pytest.fixture(scope="module")
def dict_logic() -> dict[str, Any]:
    """On-disk dict-format library logic fixture."""
    return _load_fixture("merge_logic_dict")


@pytest.fixture(scope="module")
def list_logic() -> list[Any]:
    """On-disk list-format library logic fixture."""
    return _load_fixture("merge_logic_list")


@pytest.fixture(scope="module")
def dict_from_list_logic(list_logic: list[Any]) -> dict[str, Any]:
    """List-format fixture converted to dict layout."""
    return convertToDict(deepcopy(list_logic), "merge_logic_list.yaml")


class TestFixtureLoading:
    """Verify embedded YAML fixtures load in expected raw formats."""

    def test_list_logic_loads_as_list(self, list_logic: list[Any]) -> None:
        """List-format fixture loads as a YAML sequence from disk."""
        assert isinstance(list_logic, list)
        assert len(list_logic) >= 8

    def test_dict_logic_loads_as_dict(self, dict_logic: dict[str, Any]) -> None:
        """Dict-format fixture loads as a YAML mapping from disk."""
        assert isinstance(dict_logic, dict)
        assert "Solutions" in dict_logic
        assert "ExactLogic" in dict_logic
        assert "DefaultSolution" in dict_logic


class TestLoadData:
    """``loadData`` converts list YAML to dict and normalizes layout."""

    def test_load_converts_list_to_dict(
        self, tmp_path: Path, list_logic: list[Any]
    ) -> None:
        """``loadData`` migrates list-format YAML to dict in a temp dir."""
        out_file = tmp_path / "logic.yaml"
        LibraryIO.writeYAML(
            str(out_file),
            deepcopy(list_logic),
            explicit_start=False,
            explicit_end=False,
            sort_keys=True,
        )
        fn, data, normalized = loadData(str(out_file))
        assert fn == str(out_file)
        assert isinstance(data, dict)
        assert normalized is True
        assert "Solutions" in data
        assert "ExactLogic" in data
        assert len(data["Solutions"]) == 2
        assert len(data["ExactLogic"]) == 3

    def test_load_dict_fixture_no_list_normalization(
        self, tmp_path: Path, dict_logic: dict[str, Any]
    ) -> None:
        """Dict YAML without ``Library`` block is not list-normalized."""
        out_file = tmp_path / "logic.yaml"
        LibraryIO.writeYAML(
            str(out_file),
            deepcopy(dict_logic),
            explicit_start=False,
            explicit_end=False,
            sort_keys=False,
        )
        fn, data, normalized = loadData(str(out_file))
        assert fn == str(out_file)
        assert isinstance(data, dict)
        assert normalized is False

    def test_convert_list_fixture_to_dict(self, list_logic: list[Any]) -> None:
        """``convertToDict`` turns legacy list fixture into dict layout."""
        out = convertToDict(deepcopy(list_logic), "fixture.yaml")
        assert isinstance(out, dict)
        assert "Solutions" in out
        assert isinstance(out["Solutions"], list)

    def test_convert_dict_is_noop(self, dict_logic: dict[str, Any]) -> None:
        """Dict input is returned unchanged (same object)."""
        d = deepcopy(dict_logic)
        assert convertToDict(d, "any.yaml") is d

    def test_convert_list_has_tile_selection_indices(self, list_logic: list[Any]) -> None:
        """Converted dict carries ``TileSelectionIndices`` (null for matching-table lists)."""
        out = convertToDict(deepcopy(list_logic), "fixture.yaml")
        assert "TileSelectionIndices" in out
        assert out["TileSelectionIndices"] is None

    def test_convert_list_canonical_top_level_order(self, list_logic: list[Any]) -> None:
        """Converted dict keys appear in canonical order (before Library is stripped)."""
        out = convertToDict(deepcopy(list_logic), "fixture.yaml")
        keys = [k for k in out.keys() if k != "Library"]
        assert keys == [
            "MinimumRequiredVersion", "ScheduleName", "ArchitectureName", "CUCount",
            "DeviceNames", "ProblemType", "DefaultSolution", "Solutions", "IndexOrder",
            "ExactLogic", "RangeLogic", "TileSelectionIndices", "PerfMetric", "LibraryType",
        ]

    def test_convert_sorts_solution_params_regardless_of_input_order(
        self, list_logic: list[Any]
    ) -> None:
        """Solution keys are sorted (naming keys first) even when the input is shuffled."""
        shuffled = deepcopy(list_logic)
        sol = shuffled[5][0]
        # Rebuild the first solution dict in a deliberately non-alphabetical order.
        reordered = {}
        tail = ["WorkGroup", "BaseName", "MacroTile0", "DepthU", "SolutionNameMin",
                "SolutionIndex", "KernelNameMin", "_staggerStrideShift"]
        for k in tail:
            if k in sol:
                reordered[k] = sol[k]
        for k, v in sol.items():
            reordered.setdefault(k, v)
        shuffled[5][0] = reordered

        out = convertToDict(shuffled, "fixture.yaml")
        keys = list(out["Solutions"][0].keys())
        assert keys[:3] == ["SolutionIndex", "KernelNameMin", "SolutionNameMin"]
        assert keys[3:] == sorted(keys[3:])


class TestNormalizeDictLibraryLayout:
    """Tests for ``normalizeDictLibraryLayout`` (always runs on dict input)."""

    def test_normalizes_regardless_of_architecture(self) -> None:
        """``Library.distance`` is promoted for any architecture dict."""
        data = _minimal_dict_logic(
            library_type="Matching",
            library_block={"distance": "GridBased"},
            architecture_name="gfx942",
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "GridBased"
        assert "Library" not in data

    def test_strips_library_and_sets_distance_from_library_block(self) -> None:
        """``Library.distance`` becomes top-level ``LibraryType``; ``Library`` removed."""
        data = _minimal_dict_logic(
            library_type="Matching",
            library_block={"distance": "Equality"},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "Equality"
        assert "Library" not in data

    def test_matching_without_library_is_noop(self) -> None:
        """``LibraryType: Matching`` without ``Library`` is not rewritten."""
        data = _minimal_dict_logic(library_type="Matching")
        assert normalizeDictLibraryLayout(data) is False
        assert data["LibraryType"] == "Matching"
        assert "Library" not in data

    def test_freesize_preserved(self) -> None:
        """``FreeSize`` stays when ``Library`` has no usable ``distance``."""
        data = _minimal_dict_logic(
            library_type="FreeSize",
            library_block={},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "FreeSize"
        assert "Library" not in data

    def test_range_from_library_distance(self) -> None:
        """``Library.distance: Range`` is promoted to top-level ``LibraryType``."""
        data = _minimal_dict_logic(
            library_type="Matching",
            library_block={"distance": "Range"},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "Range"
        assert "Library" not in data

    def test_prediction_toplevel_empty_library(self) -> None:
        """``Prediction`` passes through while dropping empty ``Library``."""
        data = _minimal_dict_logic(
            library_type="Prediction",
            library_block={},
        )
        assert normalizeDictLibraryLayout(data) is True
        assert data["LibraryType"] == "Prediction"
        assert "Library" not in data


class TestCompareProblemType:
    """Tests for ``compareProblemType`` (ProblemType equality gate)."""

    def test_matching_problem_types_do_not_exit(self) -> None:
        """Identical ``ProblemType`` dicts do not call ``sys.exit``."""
        pt = {"OperationType": "GEMM", "Batched": True}
        ori = {"ProblemType": deepcopy(pt)}
        inc = {"ProblemType": deepcopy(pt)}

        class _FakeProblemType:
            def __init__(self, state: Any, _assign_gpus: bool) -> None:
                self.state = deepcopy(state)

        with patch("Tensile.TensileMergeLibrary.ProblemType", _FakeProblemType), patch(
            "Tensile.TensileMergeLibrary.problemTypeToEnum", lambda _pt: None
        ):
            compareProblemType(ori, inc)

    def test_mismatch_exits(self) -> None:
        """Differing ``ProblemType`` after normalization triggers ``sys.exit``."""
        ori = {"ProblemType": {"OperationType": "GEMM", "Batched": True}}
        inc = {"ProblemType": {"OperationType": "GEMM", "Batched": False}}

        class _FakeProblemType:
            def __init__(self, state: Any, _assign_gpus: bool) -> None:
                self.state = deepcopy(state)

        with patch("Tensile.TensileMergeLibrary.ProblemType", _FakeProblemType), patch(
            "Tensile.TensileMergeLibrary.problemTypeToEnum", lambda _pt: None
        ):
            with pytest.raises(SystemExit, match="ProblemType"):
                compareProblemType(ori, inc)


class TestFixSizeInconsistenciesWithFixtures:
    """Tests for ``fixSizeInconsistencies`` using fixture-derived exact logic."""

    def test_sizes_preserved(self, dict_logic: dict[str, Any]) -> None:
        """Unique sizes in fixture are all preserved."""
        logic = deepcopy(dict_logic["ExactLogic"])
        result, count = fixSizeInconsistencies(logic, "test")
        assert count == 3

    @pytest.mark.parametrize(
        "sizes",
        [
            [
                [[10240, 384, 1, 8192], [0, 0.0]],
                [[10240, 336, 1, 8192], [1, 0.0]],
                [[10240, 384, 1, 8192], [2, 1.0]],
            ]
        ],
    )
    def test_deduplication(self, sizes: list[Any]) -> None:
        """Duplicate sizes collapse to unique entries."""
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 2
        assert len({tuple(r[0]) for r in result}) == 2


class TestSolutionCleanup:
    """Tests for remove/sanitize helpers on dict-format logic."""

    def test_all_solutions_used(self, dict_logic: dict[str, Any]) -> None:
        """All solutions referenced in ExactLogic are kept."""
        data = deepcopy(dict_logic)
        _, num_removed = removeUnusedSolutions(data)
        assert num_removed == 0

    def test_remove_unused(self, dict_logic: dict[str, Any]) -> None:
        """Unused solutions are removed from dict data."""
        data = deepcopy(dict_logic)
        data["Solutions"].append(
            {
                "SolutionIndex": 99,
                "SolutionNameMin": "Unused_Sol",
                "KernelNameMin": "Unused_Kernel",
                "StaggerU": 0,
            }
        )
        _, num_removed = removeUnusedSolutions(data)
        assert num_removed == 1
        assert len(data["Solutions"]) == 2

    def test_no_duplicates(self, dict_logic: dict[str, Any]) -> None:
        """Fixture has no duplicate solution names."""
        data = deepcopy(dict_logic)
        _, num_removed, num_solutions, _ = removeDuplicatedSolutions(data)
        assert num_removed == 0
        assert num_solutions == 2

    def test_remove_duplicate_solution_names_keeps_first(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """Duplicate ``SolutionNameMin`` entries collapse to the first solution."""
        data = deepcopy(dict_logic)
        dup = deepcopy(data["Solutions"][0])
        dup["SolutionIndex"] = 1
        data["Solutions"] = [data["Solutions"][0], dup]
        data["ExactLogic"][0][1][0] = 0
        data["ExactLogic"][1][1][0] = 1
        _, num_removed, num_solutions, _ = removeDuplicatedSolutions(data)
        assert num_removed == 1
        assert num_solutions == 1

    def test_sanitize_solutions_sets_stagger_dependent_params(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """``sanitizeSolutions`` zeroes dependent stagger params when StaggerU is zero."""
        data = deepcopy(dict_logic)
        data["Solutions"][0]["StaggerU"] = 0
        data["Solutions"][0]["StaggerUMapping"] = 9
        data["Solutions"][0]["StaggerUStride"] = 123
        data["Solutions"][0]["_staggerStrideShift"] = 7

        sanitizeSolutions(data)

        sanitized = data["Solutions"][0]
        assert sanitized["StaggerUMapping"] == 0
        assert sanitized["StaggerUStride"] == 0
        assert sanitized["_staggerStrideShift"] == 0


class TestMergeLogic:
    """Tests for ``mergeLogic`` using dict-format data."""

    def test_merge_with_new_size(self, dict_logic: dict[str, Any]) -> None:
        """Merge adds one new size from incremental dict data."""
        ori_data = deepcopy(dict_logic)
        inc_data = deepcopy(dict_logic)
        _append_new_exact_logic_size(inc_data)
        merged_data, num_sizes_added, _, _ = mergeLogic(
            deepcopy(ori_data), inc_data, forceMerge=False
        )
        assert num_sizes_added == 1
        assert len(merged_data["ExactLogic"]) == 4

    def test_merge_better_efficiency_replaces(self, dict_logic: dict[str, Any]) -> None:
        """Better efficiency solution replaces original (winner policy)."""
        ori_data = deepcopy(dict_logic)
        inc_data = deepcopy(dict_logic)
        inc_data["ExactLogic"][0][1][1] = 2.0
        inc_data["Solutions"][0]["SolutionNameMin"] = "Better_Sol"

        merged_data, num_sizes_added, num_solutions_added, _ = mergeLogic(
            ori_data, inc_data, forceMerge=False
        )
        assert num_sizes_added == 0
        assert num_solutions_added >= 1
        first_logic = merged_data["ExactLogic"][0]
        assert first_logic[1][1] == 2.0
        sol = findSolutionWithIndex(merged_data["Solutions"], first_logic[1][0])
        assert sol["SolutionNameMin"] == "Better_Sol"

    def test_merge_no_eff_zeros_stored_efficiency(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """``noEff=True`` forces stored efficiency to 0.0 on merged sizes."""
        ori_data = deepcopy(dict_logic)
        inc_data = deepcopy(dict_logic)
        _append_new_exact_logic_size(inc_data)
        merged_data, _, _, _ = mergeLogic(
            ori_data, inc_data, forceMerge=False, noEff=True
        )
        for _size, (_idx, eff) in merged_data["ExactLogic"]:
            assert eff == 0.0

    def test_merge_force_merge(self, dict_logic: dict[str, Any]) -> None:
        """Force merge replaces even with worse efficiency."""
        ori_data = deepcopy(dict_logic)
        ori_data["ExactLogic"][0][1][1] = 5.0
        inc_data = deepcopy(dict_logic)
        inc_data["ExactLogic"][0][1][1] = 0.0
        inc_data["Solutions"][0]["SolutionNameMin"] = "Forced_Sol"

        merged_data, _, _, _ = mergeLogic(ori_data, inc_data, forceMerge=True)
        solution_names = [s["SolutionNameMin"] for s in merged_data["Solutions"]]
        assert "Forced_Sol" in solution_names

    def test_merge_converted_list_fixture(self, dict_from_list_logic: dict[str, Any]) -> None:
        """Merge works on dict converted from legacy list fixture."""
        ori_data = deepcopy(dict_from_list_logic)
        inc_data = deepcopy(dict_from_list_logic)
        inc_data["ExactLogic"].append([[100, 200, 1, 300], [0, 0.0]])
        merged_data, num_sizes_added, _, _ = mergeLogic(
            ori_data, inc_data, forceMerge=False
        )
        assert num_sizes_added == 1
        assert len(merged_data["ExactLogic"]) == 4


class TestDefaultSolutionFunctions:
    """Tests for DefaultSolution sync and cleanup on dict data."""

    def test_sync_default_params(self, dict_logic: dict[str, Any]) -> None:
        """``syncDefaultParams`` runs when defaults change between libraries."""
        data = deepcopy(dict_logic)
        orig_defaults = {"StaggerU": 32, "TestParam": 100}
        inc_defaults = {"StaggerU": 64, "TestParam": 200}
        syncDefaultParams(data, orig_defaults, inc_defaults)
        assert len(data["Solutions"]) == 2

    def test_sync_default_params_pins_base_only_stagger_stride(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """Base-only default ``StaggerUStride`` is pinned when inc default changes."""
        data = deepcopy(dict_logic)
        soln = data["Solutions"][0]
        del soln["StaggerUStride"]
        orig_defaults = deepcopy(data["DefaultSolution"])
        inc_defaults = deepcopy(data["DefaultSolution"])
        inc_defaults["StaggerUStride"] = 512

        syncDefaultParams(data, orig_defaults, inc_defaults)

        assert soln["StaggerUStride"] == 256

    def test_sync_default_params_identical_defaults_no_op(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """When default maps are equal, ``syncDefaultParams`` returns immediately."""
        data = deepcopy(dict_logic)
        before = deepcopy(data["Solutions"])
        syncDefaultParams(data, {"StaggerU": 32}, {"StaggerU": 32})
        assert data["Solutions"] == before

    def test_sync_default_params_removes_inc_default_match(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """Solution values equal to incremental defaults are stripped."""
        data = deepcopy(dict_logic)
        soln = data["Solutions"][0]
        soln["StaggerU"] = 64
        syncDefaultParams(data, {"StaggerU": 32}, {"StaggerU": 64})
        assert "StaggerU" not in soln

    def test_remove_default_init_params(self, dict_logic: dict[str, Any]) -> None:
        """``removeDefaultInitParams`` removes params matching default."""
        data = deepcopy(dict_logic)
        data["Solutions"][0]["GlobalSplitU"] = 1
        data["DefaultSolution"]["GlobalSplitU"] = 1

        removeDefaultInitParams(data)

        assert "GlobalSplitU" not in data["Solutions"][0]

    def test_remove_cu_count_from_default(self, dict_logic: dict[str, Any]) -> None:
        """``CUCount`` is removed from ``DefaultSolution``."""
        data = deepcopy(dict_logic)
        data["DefaultSolution"]["CUCount"] = 304

        removeDefaultInitParams(data)

        assert "CUCount" not in data["DefaultSolution"]


class TestCompareDestFolderToYaml:
    """Tests for destination-folder vs ``LibraryType`` validation."""

    @pytest.mark.parametrize(
        "dest_dir,expect_exit",
        [
            ("/path/to/GridBased", False),
            ("/path/to/Equality", True),
        ],
    )
    def test_compare_dest_folder_to_yaml_library_type(
        self, dict_logic: dict[str, Any], dest_dir: str, expect_exit: bool
    ) -> None:
        """``compareDestFolderToYaml`` matches dest folder to ``LibraryType``."""
        if expect_exit:
            with pytest.raises(SystemExit):
                compareDestFolderToYaml(dest_dir, "logic.yaml", dict_logic)
        else:
            compareDestFolderToYaml(dest_dir, "logic.yaml", dict_logic)

    def test_compare_dest_folder_exits_when_library_type_unset(self) -> None:
        """``compareDestFolderToYaml`` exits when ``LibraryType`` is missing."""
        data = _minimal_dict_logic(library_type="")
        with pytest.raises(SystemExit, match="Empty YAML attribute"):
            compareDestFolderToYaml("/any/GridBased", "logic.yaml", data)


@pytest.mark.unit
class TestEnsurePath:
    """Test ``ensurePath`` function."""

    def test_create_new_directory(self) -> None:
        """Creating a nested directory succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "test_dir", "nested")
            result = ensurePath(new_path)
            assert os.path.exists(new_path)
            assert result == new_path

    def test_existing_directory(self) -> None:
        """Existing directory is returned unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensurePath(tmpdir)
            assert os.path.exists(tmpdir)
            assert result == tmpdir


@pytest.mark.unit
class TestAllFiles:
    """Test ``allFiles`` function."""

    def test_find_yaml_files_in_directory(self) -> None:
        """YAML files are found case-insensitively by extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "file1.yaml")).touch()
            Path(os.path.join(tmpdir, "file2.YAML")).touch()
            Path(os.path.join(tmpdir, "file3.txt")).touch()

            files = allFiles(tmpdir)
            yaml_names = [os.path.basename(f) for f in files]

            assert len(files) == 2
            assert "file1.yaml" in yaml_names
            assert "file2.YAML" in yaml_names
            assert "file3.txt" not in yaml_names

    def test_empty_directory(self) -> None:
        """Empty directory returns no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = allFiles(tmpdir)
            assert files == []

    def test_nested_directories(self) -> None:
        """Top-level YAML files in a directory are collected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "top.yaml")).touch()
            Path(os.path.join(tmpdir, "another.yaml")).touch()

            files = allFiles(tmpdir)
            assert len(files) == 2


    def test_nested_yaml_directory_recurses(self) -> None:
        """A subdirectory named ``*.yaml`` is searched recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested.yaml")
            os.makedirs(nested)
            Path(os.path.join(nested, "inner.yaml")).touch()
            Path(os.path.join(tmpdir, "top.yaml")).touch()

            files = allFiles(tmpdir)
            yaml_names = sorted(os.path.basename(f) for f in files)

            assert yaml_names == ["inner.yaml", "top.yaml"]


@pytest.mark.unit
class TestFixSizeInconsistencies:
    """Test ``fixSizeInconsistencies`` function."""

    def test_remove_duplicates(self) -> None:
        """Duplicate truncated sizes collapse to one entry."""
        sizes = [
            ([1, 2, 3, 4, 5, 6, 7, 8], 0),
            ([1, 2, 3, 4], 1),
            ([1, 2, 3, 4, 9, 10, 11, 12], 2),
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 1
        assert len(result) == 1
        assert result[0][0] == [1, 2, 3, 4]

    @patch("Tensile.TensileMergeLibrary.verbose")
    def test_remove_duplicates_verbose_message(self, mock_verbose: Any) -> None:
        """Verbose message reports duplicate count when sizes are deduplicated."""
        sizes = [
            ([1, 2, 3, 4, 5, 6, 7, 8], 0),
            ([1, 2, 3, 4], 1),
            ([1, 2, 3, 4, 9, 10, 11, 12], 2),
        ]
        fixSizeInconsistencies(sizes, "test")
        mock_verbose.assert_called_once_with(
            2, "duplicate size(s) removed from", "test", "logic file"
        )

    def test_no_duplicates(self) -> None:
        """Distinct sizes are all retained."""
        sizes = [
            ([1, 2, 3], 0),
            ([4, 5, 6], 1),
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == 2
        assert len(result) == 2


@pytest.mark.unit
class TestAddKernel:
    """Test ``addKernel`` function."""

    def test_add_new_kernel(self) -> None:
        """A new kernel is appended to the pool."""
        solution_pool: list[dict[str, Any]] = []
        sol_dict: dict[str, Any] = {}
        solution = {"SolutionNameMin": "kernel_1", "data": "test"}

        pool, sol_dict, index = addKernel(solution_pool, sol_dict, solution)

        assert len(pool) == 1
        assert index == 0
        assert pool[0]["SolutionIndex"] == 0
        assert "kernel_1" in sol_dict

    def test_reuse_existing_kernel(self) -> None:
        """An existing kernel name reuses the same index."""
        solution_existing = {
            "SolutionNameMin": "kernel_1",
            "SolutionIndex": 0,
            "data": "test",
        }
        solution_pool = [solution_existing]
        sol_dict = {"kernel_1": solution_existing}

        solution_new = {"SolutionNameMin": "kernel_1", "data": "new"}
        pool, sol_dict, index = addKernel(solution_pool, sol_dict, solution_new)

        assert len(pool) == 1
        assert index == 0

    def test_add_multiple_kernels(self) -> None:
        """Multiple distinct kernels receive sequential indices."""
        solution_pool: list[dict[str, Any]] = []
        sol_dict: dict[str, Any] = {}

        sol1 = {"SolutionNameMin": "kernel_1"}
        sol2 = {"SolutionNameMin": "kernel_2"}

        pool, sol_dict, idx1 = addKernel(solution_pool, sol_dict, sol1)
        pool, sol_dict, idx2 = addKernel(pool, sol_dict, sol2)

        assert len(pool) == 2
        assert idx1 == 0
        assert idx2 == 1


@pytest.mark.unit
class TestMessageFunctions:
    """Test ``msg``, ``verbose``, and ``debug`` functions."""

    @patch("builtins.print")
    def test_msg_output(self, mock_print: Any) -> None:
        """``msg`` prints to stdout."""
        msg("test", "message")
        assert mock_print.call_count >= 1

    @patch("builtins.print")
    @patch("Tensile.TensileMergeLibrary.verbosity", 1)
    def test_verbose_output_when_enabled(self, mock_print: Any) -> None:
        """``verbose`` prints when verbosity >= 1."""
        verbose("test", "message")
        assert mock_print.call_count >= 1

    @patch("builtins.print")
    @patch("Tensile.TensileMergeLibrary.verbosity", 0)
    def test_verbose_no_output_when_disabled(self, mock_print: Any) -> None:
        """``verbose`` is silent when verbosity < 1."""
        verbose("test", "message")
        assert mock_print.call_count == 0

    @patch("builtins.print")
    @patch("Tensile.TensileMergeLibrary.verbosity", 2)
    def test_debug_output_when_enabled(self, mock_print: Any) -> None:
        """``debug`` prints when verbosity >= 2."""
        debug("test", "message")
        assert mock_print.call_count >= 1

    @patch("builtins.print")
    @patch("Tensile.TensileMergeLibrary.verbosity", 1)
    def test_debug_no_output_when_disabled(self, mock_print: Any) -> None:
        """``debug`` is silent when verbosity < 2."""
        debug("test", "message")
        assert mock_print.call_count == 0


@pytest.mark.unit
class TestFindSolutionWithIndex:
    """Test ``findSolutionWithIndex`` function."""

    def test_find_solution_at_correct_index(self) -> None:
        """Direct index lookup when position matches ``SolutionIndex``."""
        solutions = [
            {"SolutionIndex": 0, "name": "sol0"},
            {"SolutionIndex": 1, "name": "sol1"},
            {"SolutionIndex": 2, "name": "sol2"},
        ]

        result = findSolutionWithIndex(solutions, 1)
        assert result["name"] == "sol1"

    def test_find_solution_with_search(self) -> None:
        """Linear search when index does not match list position."""
        solutions = [
            {"SolutionIndex": 5, "name": "sol5"},
            {"SolutionIndex": 10, "name": "sol10"},
            {"SolutionIndex": 15, "name": "sol15"},
        ]

        result = findSolutionWithIndex(solutions, 10)
        assert result["name"] == "sol10"

    def test_find_nonexistent_solution(self) -> None:
        """Missing solution index raises assertion."""
        solutions = [{"SolutionIndex": 0, "name": "sol0"}]

        with pytest.raises(AssertionError):
            findSolutionWithIndex(solutions, 99)

    def test_find_solution_from_fixture(self, dict_logic: dict[str, Any]) -> None:
        """Fixture solutions are found by index."""
        solutions = dict_logic["Solutions"]
        result0 = findSolutionWithIndex(solutions, 0)
        result1 = findSolutionWithIndex(solutions, 1)
        assert result0["SolutionIndex"] == 0
        assert result1["SolutionIndex"] == 1
        assert result0["SolutionNameMin"] == "Sol_dict_0"
        assert result1["SolutionNameMin"] == "Sol_dict_1"


@pytest.mark.unit
class TestReNameSolutions:
    """Test ``reNameSolutions`` function."""

    @patch("Tensile.TensileMergeLibrary.getSolutionNameMin")
    @patch("Tensile.TensileMergeLibrary.getKernelNameMin")
    @patch("Tensile.TensileMergeLibrary.assignParameterWithDefault")
    def test_rename_solutions(
        self,
        mock_assign: Any,
        mock_kernel_name: Any,
        mock_solution_name: Any,
    ) -> None:
        """``reNameSolutions`` updates min names on dict-format data."""
        mock_solution_name.return_value = "sol_min"
        mock_kernel_name.return_value = "kernel_min"

        data: dict[str, Any] = {
            "ProblemType": {"OperationType": "GEMM"},
            "Solutions": [{"key": "value"}],
        }

        reNameSolutions(data)

        assert data["Solutions"][0]["SolutionNameMin"] == "sol_min"
        assert data["Solutions"][0]["KernelNameMin"] == "kernel_min"
        assert "ProblemType" not in data["Solutions"][0]

    @patch("Tensile.TensileMergeLibrary.getSolutionNameMin")
    @patch("Tensile.TensileMergeLibrary.getKernelNameMin")
    @patch("Tensile.TensileMergeLibrary.assignParameterWithDefault")
    def test_rename_applies_default_gsu_and_strips_match(
        self,
        mock_assign: Any,
        mock_kernel_name: Any,
        mock_solution_name: Any,
    ) -> None:
        """Default ``GlobalSplitU`` is applied then removed when it matches default."""
        mock_solution_name.return_value = "sol_min"
        mock_kernel_name.return_value = "kernel_min"

        data: dict[str, Any] = {
            "ProblemType": {"OperationType": "GEMM"},
            "DefaultSolution": {"GlobalSplitU": 1},
            "Solutions": [{"GlobalSplitU": 1}],
        }

        reNameSolutions(data)

        assert "GlobalSplitU" not in data["Solutions"][0]

    @patch("Tensile.TensileMergeLibrary.getSolutionNameMin")
    @patch("Tensile.TensileMergeLibrary.getKernelNameMin")
    @patch("Tensile.TensileMergeLibrary.assignParameterWithDefault")
    def test_rename_applies_missing_default_gsu(
        self,
        mock_assign: Any,
        mock_kernel_name: Any,
        mock_solution_name: Any,
    ) -> None:
        """Missing ``GlobalSplitU`` is filled from ``DefaultSolution`` before renaming."""
        mock_solution_name.return_value = "sol_min"
        mock_kernel_name.return_value = "kernel_min"

        data: dict[str, Any] = {
            "ProblemType": {"OperationType": "GEMM"},
            "DefaultSolution": {"GlobalSplitU": 1},
            "Solutions": [{}],
        }

        reNameSolutions(data)

        assert "GlobalSplitU" not in data["Solutions"][0]

    @patch("Tensile.TensileMergeLibrary.getSolutionNameMin")
    @patch("Tensile.TensileMergeLibrary.getKernelNameMin")
    @patch("Tensile.TensileMergeLibrary.assignParameterWithDefault")
    def test_rename_keeps_gsu_for_custom_kernel(
        self,
        mock_assign: Any,
        mock_kernel_name: Any,
        mock_solution_name: Any,
    ) -> None:
        """``GlobalSplitU`` is retained when ``CustomKernelName`` is set."""
        mock_solution_name.return_value = "sol_min"
        mock_kernel_name.return_value = "kernel_min"

        data: dict[str, Any] = {
            "ProblemType": {"OperationType": "GEMM"},
            "DefaultSolution": {"GlobalSplitU": 1},
            "Solutions": [{"GlobalSplitU": 1, "CustomKernelName": "custom"}],
        }

        reNameSolutions(data)

        assert data["Solutions"][0]["GlobalSplitU"] == 1


@pytest.mark.unit
class TestAvoidRegressions:
    """End-to-end tests for ``avoidRegressions`` file merge."""

    def test_merge_matching_basenames_and_copy_extra(
        self, dict_logic: dict[str, Any]
    ) -> None:
        """Matching YAML basenames merge; unmatched incremental files are copied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.path.join(tmpdir, "Other")
            inc_dir = os.path.join(tmpdir, "inc")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(orig_dir)
            os.makedirs(inc_dir)

            orig_logic = deepcopy(dict_logic)
            inc_logic = deepcopy(dict_logic)
            inc_logic["ExactLogic"][0][1][1] = 0.95

            logic_name = "logic.yaml"
            LibraryIO.writeYAML(
                os.path.join(orig_dir, logic_name),
                orig_logic,
                explicit_start=False,
                explicit_end=False,
                sort_keys=False,
            )
            LibraryIO.writeYAML(
                os.path.join(inc_dir, logic_name),
                inc_logic,
                explicit_start=False,
                explicit_end=False,
                sort_keys=False,
            )
            extra_name = "extra.yaml"
            LibraryIO.writeYAML(
                os.path.join(inc_dir, extra_name),
                deepcopy(dict_logic),
                explicit_start=False,
                explicit_end=False,
                sort_keys=False,
            )

            with patch("Tensile.TensileMergeLibrary.compareProblemType"), patch(
                "Tensile.TensileMergeLibrary.compareDestFolderToYaml"
            ), patch("Tensile.TensileMergeLibrary.reNameSolutions"):
                avoidRegressions(orig_dir, inc_dir, out_dir, forceMerge=True)

            merged = LibraryIO.readYAML(os.path.join(out_dir, logic_name))
            assert isinstance(merged, dict)
            assert merged["ExactLogic"][0][1][1] == 0.95
            assert os.path.isfile(os.path.join(out_dir, extra_name))


@pytest.mark.unit
class TestMainFunction:
    """Test main function argument parsing."""

    @patch("Tensile.TensileMergeLibrary.avoidRegressions")
    @patch("sys.argv", ["script", "/orig", "/inc", "/out", "-v", "2"])
    def test_main_with_arguments(self, mock_avoid: Any) -> None:
        """Main forwards positional directories to ``avoidRegressions``."""
        from Tensile.TensileMergeLibrary import main

        main()

        mock_avoid.assert_called_once()
        args = mock_avoid.call_args[0]
        assert args[0] == "/orig"
        assert args[1] == "/inc"
        assert args[2] == "/out"

    @patch("Tensile.TensileMergeLibrary.avoidRegressions")
    @patch("sys.argv", ["script", "/orig", "/inc", "/out", "--force_merge", "true"])
    def test_main_with_force_merge_true(self, mock_avoid: Any) -> None:
        """``--force_merge true`` sets force merge flag."""
        from Tensile.TensileMergeLibrary import main

        main()

        args = mock_avoid.call_args[0]
        assert args[3] is True

    @patch("Tensile.TensileMergeLibrary.avoidRegressions")
    @patch("sys.argv", ["script", "/orig", "/inc", "/out", "--force_merge", "false"])
    def test_main_with_force_merge_false(self, mock_avoid: Any) -> None:
        """``--force_merge false`` clears force merge flag."""
        from Tensile.TensileMergeLibrary import main

        main()

        args = mock_avoid.call_args[0]
        assert args[3] is False

    @patch("Tensile.TensileMergeLibrary.avoidRegressions")
    @patch("sys.argv", ["script", "/orig", "/inc", "/out", "--no_eff"])
    def test_main_with_no_eff_flag(self, mock_avoid: Any) -> None:
        """``--no_eff`` enables zero-efficiency merge mode."""
        from Tensile.TensileMergeLibrary import main

        main()

        kwargs = mock_avoid.call_args[0]
        assert kwargs[4] is True
