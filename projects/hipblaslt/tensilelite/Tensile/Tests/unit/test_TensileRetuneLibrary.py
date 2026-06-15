# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for TensileRetuneLibrary.py
"""

import pytest
import os
import yaml
import tempfile
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestWorkingPathFunctions:
    """Test working path management functions"""

    def test_ensurepath_creates_directory(self):
        """ensurePath should create directory if it doesn't exist"""
        from Tensile.TensileRetuneLibrary import ensurePath

        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "newdir", "nested")
            result = ensurePath(new_path)

            # Verify directory was created
            assert os.path.exists(new_path)
            assert os.path.isdir(new_path)
            assert result == new_path

    def test_ensurepath_handles_existing_directory(self):
        """ensurePath should handle existing directory without error"""
        from Tensile.TensileRetuneLibrary import ensurePath

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call twice to test idempotency
            result1 = ensurePath(tmpdir)
            result2 = ensurePath(tmpdir)

            assert result1 == tmpdir
            assert result2 == tmpdir
            assert os.path.exists(tmpdir)

    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/base/path"})
    def test_push_working_path(self):
        """pushWorkingPath should append folder and create directory"""
        from Tensile.TensileRetuneLibrary import pushWorkingPath, globalParameters

        with tempfile.TemporaryDirectory() as tmpdir:
            globalParameters["WorkingPath"] = tmpdir

            result = pushWorkingPath("subfolder")

            # Verify path was updated and directory created
            expected_path = os.path.join(tmpdir, "subfolder")
            assert globalParameters["WorkingPath"] == expected_path
            assert os.path.exists(expected_path)
            assert result == expected_path

    @patch('Tensile.TensileRetuneLibrary.workingDirectoryStack', [])
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/base/path/subfolder"})
    def test_pop_working_path_empty_stack(self):
        """popWorkingPath should go up one level when stack is empty"""
        from Tensile.TensileRetuneLibrary import popWorkingPath, globalParameters

        popWorkingPath()

        assert globalParameters["WorkingPath"] == "/base/path"

    @patch('Tensile.TensileRetuneLibrary.workingDirectoryStack', ["/saved/path"])
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/current/path"})
    def test_pop_working_path_with_stack(self):
        """popWorkingPath should restore from stack when available"""
        from Tensile.TensileRetuneLibrary import popWorkingPath, globalParameters, workingDirectoryStack

        popWorkingPath()

        assert globalParameters["WorkingPath"] == "/saved/path"
        assert len(workingDirectoryStack) == 0

    @patch('Tensile.TensileRetuneLibrary.workingDirectoryStack', [])
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/initial/path"})
    def test_set_working_path(self):
        """setWorkingPath should save current path and set new one"""
        from Tensile.TensileRetuneLibrary import setWorkingPath, globalParameters, workingDirectoryStack

        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = os.path.join(tmpdir, "old")
            new_path = os.path.join(tmpdir, "new")
            os.makedirs(old_path)

            globalParameters["WorkingPath"] = old_path

            setWorkingPath(new_path)

            # Verify new path was set and created
            assert globalParameters["WorkingPath"] == new_path
            assert os.path.exists(new_path)
            # Verify old path was saved to stack
            assert old_path in workingDirectoryStack


@pytest.mark.unit
class TestParseCurrentLibrary:
    """Test parseCurrentLibrary function"""

    @patch('Tensile.TensileRetuneLibrary.globalParameters', {})
    @patch('Tensile.TensileRetuneLibrary.ProblemSizes')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.parseLibraryLogicData')
    def test_parses_library_without_size_file(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should parse library and create sizes from exactLogic"""
        from Tensile.TensileRetuneLibrary import parseCurrentLibrary, globalParameters

        mock_problem_type = Mock()
        mock_solution1 = MagicMock()
        mock_solution1.__eq__ = Mock(return_value=False)
        mock_exact_logic = [([64, 64, 1, 64], {}), ([128, 128, 1, 128], {})]

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution1], mock_exact_logic, None, None)
        mock_problem_sizes.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            lib_file = os.path.join(tmpdir, "lib.yaml")
            lib_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "PerformanceMetric"]

            with open(lib_file, 'w') as f:
                yaml.dump(lib_data, f)

            result = parseCurrentLibrary(lib_file, None)

            # Verify return structure
            assert len(result) == 3
            assert result[1] is not None  # solutions
            assert len(result[1]) == 1  # One solution after dedup

            # Verify performance metric was set
            assert globalParameters["PerformanceMetric"] == "PerformanceMetric"

            # Verify ProblemSizes was called with exactLogic converted
            mock_problem_sizes.assert_called_once()
            call_args = mock_problem_sizes.call_args
            sizes_arg = call_args[0][1]
            assert len(sizes_arg) == 2
            assert sizes_arg[0] == {"Exact": [64, 64, 1, 64]}
            assert sizes_arg[1] == {"Exact": [128, 128, 1, 128]}

    @patch('Tensile.TensileRetuneLibrary.ProblemSizes')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.parseLibraryLogicData')
    def test_parses_library_with_size_file(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should use sizes from file when provided"""
        from Tensile.TensileRetuneLibrary import parseCurrentLibrary

        mock_problem_type = Mock()
        mock_solution = MagicMock()
        mock_solution.__eq__ = Mock(return_value=False)

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution], [], None, None)
        mock_problem_sizes.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            lib_file = os.path.join(tmpdir, "lib.yaml")
            size_file = os.path.join(tmpdir, "sizes.yaml")

            lib_data = [1, 2, 3]
            size_data = [{"Exact": [256, 256, 1, 256]}, {"Exact": [512, 512, 1, 512]}]

            with open(lib_file, 'w') as f:
                yaml.dump(lib_data, f)
            with open(size_file, 'w') as f:
                yaml.dump(size_data, f)

            result = parseCurrentLibrary(lib_file, size_file)

            # Verify size file was used
            mock_problem_sizes.assert_called_once()
            call_args = mock_problem_sizes.call_args
            sizes_arg = call_args[0][1]
            assert sizes_arg == size_data

    @patch('Tensile.TensileRetuneLibrary.ProblemSizes')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.parseLibraryLogicData')
    def test_removes_duplicate_solutions(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should remove duplicate solutions and reindex"""
        from Tensile.TensileRetuneLibrary import parseCurrentLibrary

        mock_problem_type = Mock()

        # Create dict solutions (not mocks) with controlled equality
        # Use custom class to control __eq__ behavior
        class TestSolution(dict):
            def __init__(self, data, eq_id):
                super().__init__(data)
                self.eq_id = eq_id

            def __eq__(self, other):
                if isinstance(other, TestSolution):
                    return self.eq_id == other.eq_id
                return False

        # sol1 and sol2 are duplicates (eq_id=1), sol3 is unique (eq_id=2)
        sol1 = TestSolution({"Name": "Sol1"}, eq_id=1)
        sol2 = TestSolution({"Name": "Sol2"}, eq_id=1)
        sol3 = TestSolution({"Name": "Sol3"}, eq_id=2)

        mock_parse.return_value = (None, None, mock_problem_type, [sol1, sol2, sol3], [], None, None)
        mock_problem_sizes.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            lib_file = os.path.join(tmpdir, "lib.yaml")
            with open(lib_file, 'w') as f:
                yaml.dump([1, 2, 3], f)

            result = parseCurrentLibrary(lib_file, None)

            # Should have 2 unique solutions (sol1 and sol3; sol2 was duplicate of sol1)
            assert len(result[1]) == 2

            # Verify solutions were reindexed
            assert result[1][0]["SolutionIndex"] == 0
            assert result[1][1]["SolutionIndex"] == 1


@pytest.mark.unit
class TestRunBenchmarking:
    """Test runBenchmarking function"""

    @patch('Tensile.TensileRetuneLibrary.shutil.copy')
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.runClient')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeSolutions')
    @patch('Tensile.TensileRetuneLibrary.BenchmarkProblems.writeBenchmarkFiles')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.getClientExecutablePath')
    @patch('Tensile.TensileRetuneLibrary.popWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.pushWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/work"})
    def test_runs_benchmarking_creates_correct_structure(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should create directory structure and run benchmarks"""
        from Tensile.TensileRetuneLibrary import runBenchmarking

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ensure.return_value = tmpdir
            mock_solutions = [{"ISA": (9, 0, 6), "Name": "Sol1"}]
            mock_problem_sizes = Mock()
            mock_run_client.return_value = 0

            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, False, "g++", "gcc", "as", "bundler")

            # Verify benchmark files were written
            mock_write_bench.assert_called_once()

            # Verify solutions were written
            mock_write_sol.assert_called_once()

            # Verify client was run
            mock_run_client.assert_called_once()

            # Verify working path was managed (push twice, pop twice)
            assert mock_push.call_count == 2
            assert mock_pop.call_count == 2

    @patch('Tensile.TensileRetuneLibrary.ClientWriter.runClient')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeSolutions')
    @patch('Tensile.TensileRetuneLibrary.BenchmarkProblems.writeBenchmarkFiles')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.getClientExecutablePath')
    @patch('Tensile.TensileRetuneLibrary.shutil.copy')
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.popWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.pushWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/work"})
    def test_sets_update_file_when_update_true(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should set LibraryUpdateFile when update=True"""
        from Tensile.TensileRetuneLibrary import runBenchmarking, globalParameters

        mock_solutions = [{"ISA": (9, 0, 6)}]
        mock_problem_sizes = Mock()
        mock_run_client.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ensure.return_value = tmpdir
            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, True, "g++", "gcc", "as", "bundler")

            # Verify LibraryUpdateFile was set
            assert "LibraryUpdateFile" in globalParameters
            assert globalParameters["LibraryUpdateFile"] is not None
            assert "update.yaml" in globalParameters["LibraryUpdateFile"]

    @patch('Tensile.TensileRetuneLibrary.shutil.copy')
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.runClient')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeSolutions')
    @patch('Tensile.TensileRetuneLibrary.BenchmarkProblems.writeBenchmarkFiles')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.getClientExecutablePath')
    @patch('Tensile.TensileRetuneLibrary.popWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.pushWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/work"})
    def test_converts_isa_to_list(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should convert ISA tuples to lists"""
        from Tensile.TensileRetuneLibrary import runBenchmarking

        mock_solutions = [
            {"ISA": (9, 0, 6), "Name": "Sol1"},
            {"ISA": (10, 1, 0), "Name": "Sol2"}
        ]
        mock_problem_sizes = Mock()
        mock_run_client.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ensure.return_value = tmpdir
            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, False, "g++", "gcc", "as", "bundler")

            # Verify ISAs were converted to lists
            # LibraryIO.writeSolutions(libraryFile, problemSizes, "", solutions)
            # So solutions is the 4th arg (index 3)
            write_sol_call_args = mock_write_sol.call_args[0]
            solutions_arg = write_sol_call_args[3]

            assert all(isinstance(sol["ISA"], list) for sol in solutions_arg)
            assert solutions_arg[0]["ISA"] == [9, 0, 6]
            assert solutions_arg[1]["ISA"] == [10, 1, 0]


@pytest.mark.unit
class TestTensileRetuneLibrary:
    """Test TensileRetuneLibrary main function"""

    @patch('Tensile.TensileRetuneLibrary.LibraryLogic.main')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.read')
    @patch('Tensile.TensileRetuneLibrary.runBenchmarking')
    @patch('Tensile.TensileRetuneLibrary.parseCurrentLibrary')
    @patch('Tensile.TensileRetuneLibrary.validateToolchain')
    @patch('Tensile.TensileRetuneLibrary.assignGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.print1')
    def test_retune_library_remake_mode(
        self, mock_print, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read,
        mock_write, mock_logic_main
    ):
        """TensileRetuneLibrary should run benchmarking and rebuild logic in remake mode"""
        from Tensile.TensileRetuneLibrary import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            # Create dummy logic file
            with open(logic_file, 'w') as f:
                yaml.dump({"dummy": "data"}, f)

            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = ["v1", "schedule", "arch", "devices", "pt", "sol", "idx", "logic"]
            mock_parse.return_value = (raw_yaml, [], Mock())

            args = [logic_file, output_path, "--update-method", "remake"]
            TensileRetuneLibrary(args)

            # Verify key steps were executed
            mock_restore.assert_called_once()
            mock_assign.assert_called_once()
            mock_parse.assert_called_once()
            mock_bench.assert_called_once()
            mock_logic_main.assert_called_once()

            # Verify output directory was created
            assert os.path.exists(output_path)

    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.read')
    @patch('Tensile.TensileRetuneLibrary.runBenchmarking')
    @patch('Tensile.TensileRetuneLibrary.parseCurrentLibrary')
    @patch('Tensile.TensileRetuneLibrary.validateToolchain')
    @patch('Tensile.TensileRetuneLibrary.assignGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.print1')
    def test_retune_library_update_mode_writes_update_logic(
        self, mock_print, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read, mock_write
    ):
        """TensileRetuneLibrary should update logic from file in update mode"""
        from Tensile.TensileRetuneLibrary import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            with open(logic_file, 'w') as f:
                yaml.dump({"dummy": "data"}, f)

            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = [1, 2, 3, 4, 5, 6, 7, "old_logic"]
            update_logic = "new_logic"
            mock_parse.return_value = (raw_yaml, [], Mock())
            mock_read.return_value = update_logic

            args = [logic_file, output_path, "--update-method", "update"]
            TensileRetuneLibrary(args)

            # Verify YAML was written
            mock_write.assert_called_once()

            # Verify update logic was written to the correct position
            write_args = mock_write.call_args[0]
            written_data = write_args[1]
            assert written_data[7] == update_logic


@pytest.mark.unit
class TestMain:
    """Test main entry point"""

    @patch('Tensile.TensileRetuneLibrary.TensileRetuneLibrary')
    @patch('sys.argv', ['prog', 'logic.yaml', 'output/'])
    def test_main_passes_arguments_correctly(self, mock_func):
        """main should pass command line arguments to TensileRetuneLibrary"""
        from Tensile.TensileRetuneLibrary import main

        main()

        # Verify TensileRetuneLibrary was called with correct arguments
        mock_func.assert_called_once_with(['logic.yaml', 'output/'])
