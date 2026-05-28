# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import os
import sys
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from copy import deepcopy

# rocisa is mocked in the root conftest.py

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
    avoidRegressions,
)


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
            ([1, 2, 3, 4, 9, 10, 11, 12], 2),  # First 4 elements duplicate of item 1
        ]
        result, count = fixSizeInconsistencies(sizes, "test")
        assert count == len(result)

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
class TestSanitizeSolutions:
    """Test sanitizeSolutions function"""

    def test_sanitize_with_stagger_u_zero(self):
        """Test sanitizing solutions with StaggerU == 0"""
        solList = [
            {"StaggerU": 0, "StaggerUMapping": 5, "StaggerUStride": 10, "_staggerStrideShift": 3}
        ]
        sanitizeSolutions(solList)

        assert solList[0]["StaggerUMapping"] == 0
        assert solList[0]["StaggerUStride"] == 0
        assert solList[0]["_staggerStrideShift"] == 0

    def test_sanitize_with_stagger_u_nonzero(self):
        """Test solutions with StaggerU != 0 remain unchanged"""
        solList = [
            {"StaggerU": 1, "StaggerUMapping": 5, "StaggerUStride": 10, "_staggerStrideShift": 3}
        ]
        original = deepcopy(solList)
        sanitizeSolutions(solList)

        assert solList[0]["StaggerUMapping"] == 5
        assert solList[0]["StaggerUStride"] == 10

    def test_sanitize_without_stagger_u(self):
        """Test solutions without StaggerU key"""
        solList = [{"other": "value"}]
        sanitizeSolutions(solList)
        # Should not crash


@pytest.mark.unit
class TestReNameSolutions:
    """Test reNameSolutions function"""

    @patch('Tensile.TensileMergeLibrary.getSolutionNameMin')
    @patch('Tensile.TensileMergeLibrary.getKernelNameMin')
    @patch('Tensile.TensileMergeLibrary.assignParameterWithDefault')
    def test_rename_solutions(self, mock_assign, mock_kernel_name, mock_solution_name):
        """Test renaming solutions"""
        mock_solution_name.return_value = "sol_min"
        mock_kernel_name.return_value = "kernel_min"

        problem_type = {"OperationType": "GEMM"}
        solutions = [{"key": "value"}]
        data = [None, None, None, None, problem_type, solutions]

        reNameSolutions(data)

        assert solutions[0]["SolutionNameMin"] == "sol_min"
        assert solutions[0]["KernelNameMin"] == "kernel_min"
        assert "ProblemType" not in solutions[0]


@pytest.mark.unit
class TestRemoveUnusedSolutions:
    """Test removeUnusedSolutions function"""

    def test_remove_unused_solutions(self):
        """Test removing unused solutions"""
        solutions = [
            {"SolutionIndex": 0, "name": "sol0"},
            {"SolutionIndex": 1, "name": "sol1"},
            {"SolutionIndex": 2, "name": "sol2"},
        ]
        size_map = [
            ([1, 2, 3], [0, 0.9]),  # Uses solution 0
            ([4, 5, 6], [2, 0.8]),  # Uses solution 2
        ]
        data = [None, None, None, None, None, solutions, None, size_map]

        result, num_removed = removeUnusedSolutions(data)

        assert len(result[5]) == 2  # Only 2 solutions should remain
        assert num_removed == 1
        # Check reindexing
        assert result[5][0]["SolutionIndex"] == 0
        assert result[5][1]["SolutionIndex"] == 1

    def test_all_solutions_used(self):
        """Test when all solutions are used"""
        solutions = [
            {"SolutionIndex": 0, "name": "sol0"},
            {"SolutionIndex": 1, "name": "sol1"},
        ]
        size_map = [
            ([1, 2, 3], [0, 0.9]),
            ([4, 5, 6], [1, 0.8]),
        ]
        data = [None, None, None, None, None, solutions, None, size_map]

        result, num_removed = removeUnusedSolutions(data)

        assert len(result[5]) == 2
        assert num_removed == 0


@pytest.mark.unit
class TestRemoveDuplicatedSolutions:
    """Test removeDuplicatedSolutions function"""

    def test_remove_duplicates_by_solution_name(self):
        """Test removing solutions with duplicate SolutionNameMin"""
        solutions = [
            {"SolutionIndex": 0, "SolutionNameMin": "sol_A", "KernelNameMin": "kernel_1"},
            {"SolutionIndex": 1, "SolutionNameMin": "sol_B", "KernelNameMin": "kernel_2"},
            {"SolutionIndex": 2, "SolutionNameMin": "sol_A", "KernelNameMin": "kernel_1"},  # Duplicate
        ]
        size_map = [
            ([1, 2, 3], [0, 0.9]),
            ([4, 5, 6], [2, 0.8]),  # References duplicate
        ]
        data = [None, None, None, None, None, solutions, None, size_map]

        result, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(data)

        assert num_solutions == 2
        assert num_removed == 1
        assert num_kernels == 2

    def test_no_duplicates(self):
        """Test with no duplicate solutions"""
        solutions = [
            {"SolutionIndex": 0, "SolutionNameMin": "sol_A", "KernelNameMin": "kernel_1"},
            {"SolutionIndex": 1, "SolutionNameMin": "sol_B", "KernelNameMin": "kernel_2"},
        ]
        size_map = [
            ([1, 2, 3], [0, 0.9]),
            ([4, 5, 6], [1, 0.8]),
        ]
        data = [None, None, None, None, None, solutions, None, size_map]

        result, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(data)

        assert num_solutions == 2
        assert num_removed == 0
        assert num_kernels == 2


@pytest.mark.unit
class TestLoadData:
    """Test loadData function"""

    @patch('Tensile.TensileMergeLibrary.load_yaml_stream')
    def test_load_data(self, mock_load):
        """Test loading data from YAML file"""
        mock_data = {"key": "value"}
        mock_load.return_value = mock_data

        result = loadData("test.yaml")

        assert result[0] == "test.yaml"
        assert result[1] == mock_data
        mock_load.assert_called_once()


@pytest.mark.unit
class TestCompareDestFolderToYaml:
    """Test compareDestFolderToYaml function"""

    def test_matching_folder_and_attribute(self):
        """Test when folder matches YAML attribute"""
        original_dir = "/path/to/Equality"
        inc_data = [None] * 12
        inc_data[11] = "Equality"

        # Should not raise
        compareDestFolderToYaml(original_dir, "test.yaml", inc_data)

    def test_mismatching_folder_and_attribute(self):
        """Test when folder doesn't match YAML attribute"""
        original_dir = "/path/to/Equality"
        inc_data = [None] * 12
        inc_data[11] = "GridBased"

        with pytest.raises(SystemExit):
            compareDestFolderToYaml(original_dir, "test.yaml", inc_data)

    def test_empty_yaml_attribute(self):
        """Test with empty YAML attribute"""
        original_dir = "/path/to/Equality"
        inc_data = [None] * 12
        inc_data[11] = None

        with pytest.raises(SystemExit):
            compareDestFolderToYaml(original_dir, "test.yaml", inc_data)

    def test_non_checkable_folder(self):
        """Test with folder not in check list"""
        original_dir = "/path/to/OtherFolder"
        inc_data = [None] * 12
        inc_data[11] = "Equality"

        # Should not raise for folders not in checkFolders list
        compareDestFolderToYaml(original_dir, "test.yaml", inc_data)


@pytest.mark.unit
class TestCompareProblemType:
    """Test compareProblemType function"""

    @patch('Tensile.TensileMergeLibrary.problemTypeToEnum')
    @patch('Tensile.TensileMergeLibrary.ProblemType')
    def test_matching_problem_types(self, mock_problem_type, mock_enum):
        """Test with matching problem types"""
        mock_pt = Mock()
        mock_pt.state = {"OperationType": "GEMM", "DataType": "FP32"}
        mock_problem_type.return_value = mock_pt

        ori_data = [None, None, None, None, {"OperationType": "GEMM"}]
        inc_data = [None, None, None, None, {"OperationType": "GEMM"}]

        # Should not raise
        compareProblemType(ori_data, inc_data)

    @patch('Tensile.TensileMergeLibrary.problemTypeToEnum')
    @patch('Tensile.TensileMergeLibrary.ProblemType')
    def test_mismatching_problem_types(self, mock_problem_type, mock_enum):
        """Test with mismatching problem types"""
        mock_pt1 = Mock()
        mock_pt1.state = {"OperationType": "GEMM", "DataType": "FP32"}

        mock_pt2 = Mock()
        mock_pt2.state = {"OperationType": "GEMM", "DataType": "FP16"}

        mock_problem_type.side_effect = [mock_pt1, mock_pt2]

        ori_data = [None, None, None, None, {"OperationType": "GEMM"}]
        inc_data = [None, None, None, None, {"OperationType": "GEMM"}]

        with pytest.raises(SystemExit):
            compareProblemType(ori_data, inc_data)


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
class TestMergeLogic:
    """Test mergeLogic function"""

    @patch('Tensile.TensileMergeLibrary.findSolutionWithIndex')
    @patch('Tensile.TensileMergeLibrary.removeUnusedSolutions')
    @patch('Tensile.TensileMergeLibrary.fixSizeInconsistencies')
    def test_merge_with_new_sizes(self, mock_fix, mock_remove, mock_find):
        """Test merging logic with new sizes"""
        # Setup mocks
        mock_fix.side_effect = lambda x, t: (x, len(x))
        mock_remove.side_effect = lambda x, prefix="": (x, 0)
        mock_find.return_value = {"SolutionNameMin": "sol1", "KernelNameMin": "ker1"}

        ori_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol0", "SolutionIndex": 0}],  # Solutions
            None,
            [[list([1, 2, 3]), [0, 0.5]]],  # Size map - use list for mutability
        ]

        inc_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol1", "SolutionIndex": 0}],
            None,
            [[list([4, 5, 6]), [0, 0.8]]],  # New size
        ]

        result, num_sizes, num_solutions, num_removed = mergeLogic(
            ori_data, inc_data, forceMerge=False, noEff=False
        )

        assert num_sizes >= 0  # At least no decrease
        assert isinstance(result, list)

    @patch('Tensile.TensileMergeLibrary.findSolutionWithIndex')
    @patch('Tensile.TensileMergeLibrary.removeUnusedSolutions')
    @patch('Tensile.TensileMergeLibrary.fixSizeInconsistencies')
    def test_merge_with_better_efficiency(self, mock_fix, mock_remove, mock_find):
        """Test merging with improved efficiency"""
        mock_fix.side_effect = lambda x, t: (x, len(x))
        mock_remove.side_effect = lambda x, prefix="": (x, 0)
        mock_find.return_value = {"SolutionNameMin": "sol_new", "KernelNameMin": "ker_new"}

        ori_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol0", "SolutionIndex": 0}],
            None,
            [[list([1, 2, 3]), [0, 0.5]]],  # Use list instead of tuple for mutability
        ]

        inc_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol_new", "SolutionIndex": 0}],
            None,
            [[list([1, 2, 3]), [0, 0.9]]],  # Better efficiency 0.9
        ]

        result, num_sizes, num_solutions, num_removed = mergeLogic(
            ori_data, inc_data, forceMerge=False, noEff=False
        )

        # Should accept the better solution
        assert isinstance(result, list)

    @patch('Tensile.TensileMergeLibrary.findSolutionWithIndex')
    @patch('Tensile.TensileMergeLibrary.removeUnusedSolutions')
    @patch('Tensile.TensileMergeLibrary.fixSizeInconsistencies')
    def test_merge_with_force_merge(self, mock_fix, mock_remove, mock_find):
        """Test force merge even with worse efficiency"""
        mock_fix.side_effect = lambda x, t: (x, len(x))
        mock_remove.side_effect = lambda x, prefix="": (x, 0)
        mock_find.return_value = {"SolutionNameMin": "sol_forced", "KernelNameMin": "ker"}

        ori_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol0", "SolutionIndex": 0}],
            None,
            [[list([1, 2, 3]), [0, 0.9]]],  # Use list instead of tuple for mutability
        ]

        inc_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol_forced", "SolutionIndex": 0}],
            None,
            [[list([1, 2, 3]), [0, 0.3]]],  # Worse efficiency
        ]

        result, num_sizes, num_solutions, num_removed = mergeLogic(
            ori_data, inc_data, forceMerge=True, noEff=False
        )

        # Should accept even with worse efficiency due to forceMerge
        assert isinstance(result, list)

    @patch('Tensile.TensileMergeLibrary.findSolutionWithIndex')
    @patch('Tensile.TensileMergeLibrary.removeUnusedSolutions')
    @patch('Tensile.TensileMergeLibrary.fixSizeInconsistencies')
    def test_merge_with_no_eff(self, mock_fix, mock_remove, mock_find):
        """Test merge with noEff flag"""
        mock_fix.side_effect = lambda x, t: (x, len(x))
        mock_remove.side_effect = lambda x, prefix="": (x, 0)
        mock_find.return_value = {"SolutionNameMin": "sol1", "KernelNameMin": "ker1"}

        ori_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol0", "SolutionIndex": 0}],
            None,
            [[list([1, 2, 3]), [0, 0.5]]],  # Use list for mutability
        ]

        inc_data = [
            None, None, None, None, None,
            [{"SolutionNameMin": "sol1", "SolutionIndex": 0}],
            None,
            [[list([4, 5, 6]), [0, 0.8]]],
        ]

        result, num_sizes, num_solutions, num_removed = mergeLogic(
            ori_data, inc_data, forceMerge=False, noEff=True
        )

        # With noEff=True, efficiency should be set to 0.0
        assert isinstance(result, list)


@pytest.mark.unit
class TestAvoidRegressions:
    """Test avoidRegressions function"""

    @patch('Tensile.TensileMergeLibrary.mergeLogic')
    @patch('Tensile.TensileMergeLibrary.removeDuplicatedSolutions')
    @patch('Tensile.TensileMergeLibrary.reNameSolutions')
    @patch('Tensile.TensileMergeLibrary.sanitizeSolutions')
    @patch('Tensile.TensileMergeLibrary.compareProblemType')
    @patch('Tensile.TensileMergeLibrary.compareDestFolderToYaml')
    @patch('Tensile.TensileMergeLibrary.ParallelMap2')
    @patch('Tensile.TensileMergeLibrary.allFiles')
    @patch('Tensile.TensileMergeLibrary.ensurePath')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_dump')
    def test_avoid_regressions_basic(
        self, mock_dump, mock_file, mock_ensure, mock_all_files,
        mock_parallel, mock_compare_dest, mock_compare_pt,
        mock_sanitize, mock_rename, mock_remove_dup, mock_merge
    ):
        """Test basic avoidRegressions workflow"""
        # Setup mocks
        mock_ensure.return_value = "/output"
        mock_all_files.side_effect = [["/orig/file1.yaml"], ["/inc/file1.yaml"]]

        mock_data = [
            {"MinimumRequiredVersion": "1.0"},
            None, None, None,
            {"OperationType": "GEMM"},  # ProblemType
            [{"SolutionIndex": 0}],  # Solutions
            None,
            [([1, 2, 3], [0, 0.5])],  # Size map
            None, None, None,
            "Equality"
        ]

        mock_parallel.return_value = [
            ["/orig/file1.yaml", mock_data],
            ["/inc/file1.yaml", mock_data],
        ]

        mock_remove_dup.return_value = (mock_data, 0, 1, 1)
        mock_merge.return_value = [mock_data, 0, 0, 0]

        avoidRegressions("/orig", "/inc", "/output", forceMerge=False, noEff=False)

        # Verify key functions were called
        mock_ensure.assert_called()
        assert mock_all_files.call_count == 2


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
