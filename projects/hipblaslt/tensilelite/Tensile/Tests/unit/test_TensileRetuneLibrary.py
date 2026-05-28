# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for TensileRetuneLibrary.py
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call


@pytest.mark.unit
class TestWorkingPathFunctions:
    """Test working path management functions"""

    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/base"})
    def test_ensurepath_creates_directory(self):
        """ensurePath should create directory if it doesn't exist"""
        from Tensile.TensileRetuneLibrary import ensurePath

        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "newdir")
            result = ensurePath(new_path)

            assert os.path.exists(new_path)
            assert result == new_path

    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/base"})
    def test_ensurepath_handles_existing_directory(self):
        """ensurePath should handle existing directory without error"""
        from Tensile.TensileRetuneLibrary import ensurePath

        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensurePath(tmpdir)
            assert result == tmpdir

    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/base/path"})
    def test_push_working_path(self, mock_ensure):
        """pushWorkingPath should append folder to working path"""
        from Tensile.TensileRetuneLibrary import pushWorkingPath, globalParameters

        mock_ensure.return_value = "/base/path/newfolder"

        result = pushWorkingPath("newfolder")

        assert globalParameters["WorkingPath"] == "/base/path/newfolder"
        mock_ensure.assert_called_once_with("/base/path/newfolder")

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

    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.workingDirectoryStack', [])
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/old/path"})
    def test_set_working_path(self, mock_ensure):
        """setWorkingPath should save current path and set new one"""
        from Tensile.TensileRetuneLibrary import setWorkingPath, globalParameters, workingDirectoryStack

        mock_ensure.return_value = "/new/path"

        setWorkingPath("/new/path")

        assert globalParameters["WorkingPath"] == "/new/path"
        assert "/old/path" in workingDirectoryStack
        mock_ensure.assert_called_once_with("/new/path")


@pytest.mark.unit
class TestParseCurrentLibrary:
    """Test parseCurrentLibrary function"""

    @patch('Tensile.TensileRetuneLibrary.globalParameters', {})
    @patch('Tensile.TensileRetuneLibrary.ProblemSizes')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.read')
    def test_parses_library_without_size_file(self, mock_read, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should parse library and create sizes from exactLogic"""
        from Tensile.TensileRetuneLibrary import parseCurrentLibrary

        mock_problem_type = Mock()
        mock_solution1 = MagicMock()
        mock_solution1.__eq__ = Mock(return_value=False)
        mock_exact_logic = [([64, 64, 1, 64], {}), ([128, 128, 1, 128], {})]

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution1], mock_exact_logic, None, None)
        mock_read.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "PerformanceMetric"]
        mock_problem_sizes.return_value = Mock()

        result = parseCurrentLibrary("/path/to/lib.yaml", None)

        assert len(result) == 3
        assert result[0] == mock_read.return_value
        assert len(result[1]) == 1  # One solution after dedup
        mock_problem_sizes.assert_called_once()

    @patch('Tensile.TensileRetuneLibrary.ProblemSizes')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.read')
    def test_parses_library_with_size_file(self, mock_read, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should use sizes from file when provided"""
        from Tensile.TensileRetuneLibrary import parseCurrentLibrary

        mock_problem_type = Mock()
        mock_solution = MagicMock()
        mock_solution.__eq__ = Mock(return_value=False)

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution], [], None, None)

        lib_yaml = [1, 2, 3]
        size_yaml = [{"Exact": [256, 256, 1, 256]}]
        mock_read.side_effect = [lib_yaml, size_yaml]
        mock_problem_sizes.return_value = Mock()

        result = parseCurrentLibrary("/path/to/lib.yaml", "/path/to/sizes.yaml")

        assert mock_read.call_count == 2
        mock_problem_sizes.assert_called_once()

    @patch('Tensile.TensileRetuneLibrary.ProblemSizes')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.read')
    def test_removes_duplicate_solutions(self, mock_read, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should remove duplicate solutions"""
        from Tensile.TensileRetuneLibrary import parseCurrentLibrary

        mock_problem_type = Mock()

        # Create mock solutions with controlled equality
        mock_sol1 = MagicMock()
        mock_sol2 = MagicMock()
        mock_sol3 = MagicMock()

        # sol1 and sol2 are duplicates
        mock_sol1.__eq__ = lambda self, other: other is mock_sol2
        mock_sol2.__eq__ = lambda self, other: other is mock_sol1
        mock_sol3.__eq__ = lambda self, other: False

        mock_parse.return_value = (None, None, mock_problem_type, [mock_sol1, mock_sol2, mock_sol3], [], None, None)
        mock_read.return_value = [1, 2, 3]
        mock_problem_sizes.return_value = Mock()

        result = parseCurrentLibrary("/path/to/lib.yaml", None)

        # Should have 2 unique solutions
        assert len(result[1]) == 2


@pytest.mark.unit
class TestRunBenchmarking:
    """Test runBenchmarking function"""

    @patch('Tensile.TensileRetuneLibrary.ClientWriter.runClient')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeSolutions')
    @patch('Tensile.TensileRetuneLibrary.BenchmarkProblems.writeBenchmarkFiles')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.getClientExecutablePath')
    @patch('Tensile.TensileRetuneLibrary.shutil.copy')
    @patch('Tensile.TensileRetuneLibrary.popWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.pushWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/work", "LibraryUpdateFile": None})
    def test_runs_benchmarking_without_update(
        self, mock_ensure, mock_push, mock_pop, mock_copy, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client
    ):
        """runBenchmarking should run benchmarks without update file"""
        from Tensile.TensileRetuneLibrary import runBenchmarking

        mock_solutions = [{"ISA": (9, 0, 6), "Name": "Sol1"}]
        mock_problem_sizes = Mock()
        mock_ensure.return_value = "/ensured/path"
        mock_run_client.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, False, "g++", "gcc", "as", "bundler")

            mock_write_bench.assert_called_once()
            mock_write_sol.assert_called_once()
            mock_run_client.assert_called_once()
            assert mock_copy.call_count == 2  # Copy csv and yaml

    @patch('Tensile.TensileRetuneLibrary.ClientWriter.runClient')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeSolutions')
    @patch('Tensile.TensileRetuneLibrary.BenchmarkProblems.writeBenchmarkFiles')
    @patch('Tensile.TensileRetuneLibrary.ClientWriter.getClientExecutablePath')
    @patch('Tensile.TensileRetuneLibrary.shutil.copy')
    @patch('Tensile.TensileRetuneLibrary.popWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.pushWorkingPath')
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.globalParameters', {"WorkingPath": "/work"})
    def test_runs_benchmarking_with_update(
        self, mock_ensure, mock_push, mock_pop, mock_copy, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client
    ):
        """runBenchmarking should set update file when update=True"""
        from Tensile.TensileRetuneLibrary import runBenchmarking, globalParameters

        mock_solutions = [{"ISA": (9, 0, 6)}]
        mock_problem_sizes = Mock()
        mock_ensure.return_value = "/ensured/path"
        mock_run_client.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, True, "g++", "gcc", "as", "bundler")

            assert "LibraryUpdateFile" in globalParameters


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
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.print1')
    def test_retune_library_remake_mode(
        self, mock_print, mock_ensure, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read,
        mock_write, mock_logic_main
    ):
        """TensileRetuneLibrary should run in remake mode"""
        from Tensile.TensileRetuneLibrary import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            # Create dummy logic file
            with open(logic_file, 'w') as f:
                f.write("dummy")

            mock_ensure.return_value = output_path
            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = ["v1", "schedule", "arch", "devices", "pt", "sol", "idx", "logic"]
            mock_parse.return_value = (raw_yaml, [], Mock())

            args = [logic_file, output_path, "--update-method", "remake"]
            TensileRetuneLibrary(args)

            mock_restore.assert_called_once()
            mock_assign.assert_called_once()
            mock_parse.assert_called_once()
            mock_bench.assert_called_once()
            mock_logic_main.assert_called_once()

    @patch('Tensile.TensileRetuneLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileRetuneLibrary.LibraryIO.read')
    @patch('Tensile.TensileRetuneLibrary.runBenchmarking')
    @patch('Tensile.TensileRetuneLibrary.parseCurrentLibrary')
    @patch('Tensile.TensileRetuneLibrary.validateToolchain')
    @patch('Tensile.TensileRetuneLibrary.assignGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileRetuneLibrary.ensurePath')
    @patch('Tensile.TensileRetuneLibrary.print1')
    def test_retune_library_update_mode(
        self, mock_print, mock_ensure, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read, mock_write
    ):
        """TensileRetuneLibrary should run in update mode"""
        from Tensile.TensileRetuneLibrary import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            with open(logic_file, 'w') as f:
                f.write("dummy")

            mock_ensure.return_value = output_path
            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = [1, 2, 3, 4, 5, 6, 7, "old_logic"]
            update_logic = "new_logic"
            mock_parse.return_value = (raw_yaml, [], Mock())
            mock_read.return_value = update_logic

            args = [logic_file, output_path, "--update-method", "update"]
            TensileRetuneLibrary(args)

            mock_write.assert_called_once()
            # Verify update logic was written
            write_args = mock_write.call_args[0]
            assert write_args[1][7] == update_logic


@pytest.mark.unit
class TestMain:
    """Test main entry point"""

    @patch('Tensile.TensileRetuneLibrary.TensileRetuneLibrary')
    @patch('sys.argv', ['prog', 'logic.yaml', 'output/'])
    def test_main_calls_tensile_retune_library(self, mock_func):
        """main should call TensileRetuneLibrary with sys.argv[1:]"""
        from Tensile.TensileRetuneLibrary import main

        main()

        mock_func.assert_called_once_with(['logic.yaml', 'output/'])
