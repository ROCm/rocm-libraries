# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for retune_library.py
"""

import pytest
import os
import yaml
import tempfile
from unittest.mock import DEFAULT, Mock, patch, MagicMock


@pytest.mark.unit
class TestWorkingPathFunctions:
    """Test working path management functions"""

    def test_ensurepath_creates_directory(self):
        """ensurePath should create directory if it doesn't exist"""
        from tensilelite.retune_library import ensurePath

        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "newdir", "nested")
            result = ensurePath(new_path)

            # Verify directory was created
            assert os.path.exists(new_path)
            assert os.path.isdir(new_path)
            assert result == new_path

    def test_ensurepath_handles_existing_directory(self):
        """ensurePath should handle existing directory without error"""
        from tensilelite.retune_library import ensurePath

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call twice to test idempotency
            result1 = ensurePath(tmpdir)
            result2 = ensurePath(tmpdir)

            assert result1 == tmpdir
            assert result2 == tmpdir
            assert os.path.exists(tmpdir)

    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/base/path"})
    def test_push_working_path(self):
        """pushWorkingPath should append folder and create directory"""
        from tensilelite.retune_library import pushWorkingPath, globalParameters

        with tempfile.TemporaryDirectory() as tmpdir:
            globalParameters["WorkingPath"] = tmpdir

            result = pushWorkingPath("subfolder")

            # Verify path was updated and directory created
            expected_path = os.path.join(tmpdir, "subfolder")
            assert globalParameters["WorkingPath"] == expected_path
            assert os.path.exists(expected_path)
            assert result == expected_path

    @patch('tensilelite.retune_library.workingDirectoryStack', [])
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/base/path/subfolder"})
    def test_pop_working_path_empty_stack(self):
        """popWorkingPath should go up one level when stack is empty"""
        from tensilelite.retune_library import popWorkingPath, globalParameters

        popWorkingPath()

        assert globalParameters["WorkingPath"] == "/base/path"

    @patch('tensilelite.retune_library.workingDirectoryStack', ["/saved/path"])
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/current/path"})
    def test_pop_working_path_with_stack(self):
        """popWorkingPath should restore from stack when available"""
        from tensilelite.retune_library import popWorkingPath, globalParameters, workingDirectoryStack

        popWorkingPath()

        assert globalParameters["WorkingPath"] == "/saved/path"
        assert len(workingDirectoryStack) == 0

    @patch('tensilelite.retune_library.workingDirectoryStack', [])
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/initial/path"})
    def test_set_working_path(self):
        """setWorkingPath should save current path and set new one"""
        from tensilelite.retune_library import setWorkingPath, globalParameters, workingDirectoryStack

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


def _problem_type(**fields):
    """The fields of a ProblemType that runBenchmarking reads."""
    return {
        "UseBias": 0,
        "UseScaleAlphaVec": 0,
        "UseGateResidual": False,
        "BiasDataTypeList": [],
        "GateResidualDataTypeList": [],
        "ActivationType": "none",
        **fields,
    }


@pytest.mark.unit
class TestLibraryArchitectureName:
    """Test libraryArchitectureName function"""

    @pytest.mark.parametrize("logic, expected", [
        (["v", "sched", "gfx942", "devices"], "gfx942"),
        (["v", "sched", {"Architecture": "gfx942", "CUCount": 228}, "devices"], "gfx942"),
        (["v", "sched", "gfx942:xnack-", "devices"], "gfx942"),
        ({"ArchitectureName": "gfx950"}, "gfx950"),
        (["v", "sched"], None),
        (["v", "sched", None], None),
        ({"ScheduleName": "sched"}, None),
    ])
    def test_reads_the_base_architecture_name(self, logic, expected):
        """libraryArchitectureName should name the architecture of dict and list logic, if any"""
        from tensilelite.retune_library import libraryArchitectureName

        assert libraryArchitectureName(logic) == expected


@pytest.mark.unit
class TestParseCurrentLibrary:
    """Test parseCurrentLibrary function"""

    @patch('tensilelite.retune_library.globalParameters', {})
    @patch('tensilelite.retune_library.ProblemSizes')
    @patch('tensilelite.retune_library.LibraryIO.parseLibraryLogicData')
    def test_parses_library_without_size_file(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should parse library and create sizes from exactLogic"""
        from tensilelite.retune_library import parseCurrentLibrary, globalParameters

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

            result = parseCurrentLibrary(lib_file, None, Mock(), {"isa": "info"}, False)

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

    @patch('tensilelite.retune_library.globalParameters', {})
    @patch('tensilelite.retune_library.ProblemSizes')
    @patch('tensilelite.retune_library.LibraryIO.parseLibraryLogicData')
    def test_parses_dict_library_sets_perf_metric(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should read PerfMetric from dict-format logic."""
        from tensilelite.retune_library import parseCurrentLibrary, globalParameters

        mock_problem_type = Mock()
        mock_solution1 = MagicMock()
        mock_solution1.__eq__ = Mock(return_value=False)
        mock_exact_logic = [([64, 64, 1, 64], {})]

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution1], mock_exact_logic, None, None)
        mock_problem_sizes.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            lib_file = os.path.join(tmpdir, "lib.yaml")
            lib_data = {"PerfMetric": "DeviceEfficiency"}

            with open(lib_file, 'w') as f:
                yaml.dump(lib_data, f)

            result = parseCurrentLibrary(lib_file, None, Mock(), {"isa": "info"}, False)

            assert len(result) == 3
            assert globalParameters["PerformanceMetric"] == "DeviceEfficiency"

    @patch('tensilelite.retune_library.ProblemSizes')
    @patch('tensilelite.retune_library.LibraryIO.parseLibraryLogicData')
    def test_parses_library_with_size_file(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should use sizes from file when provided"""
        from tensilelite.retune_library import parseCurrentLibrary

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

            result = parseCurrentLibrary(lib_file, size_file, Mock(), {"isa": "info"}, False)

            # Verify size file was used
            mock_problem_sizes.assert_called_once()
            call_args = mock_problem_sizes.call_args
            sizes_arg = call_args[0][1]
            assert sizes_arg == size_data

    @patch('tensilelite.retune_library.ProblemSizes')
    @patch('tensilelite.retune_library.LibraryIO.parseLibraryLogicData')
    def test_removes_duplicate_solutions(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should remove duplicate solutions and reindex"""
        from tensilelite.retune_library import parseCurrentLibrary

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

            result = parseCurrentLibrary(lib_file, None, Mock(), {"isa": "info"}, False)

            # Should have 2 unique solutions (sol1 and sol3; sol2 was duplicate of sol1)
            assert len(result[1]) == 2

            # Verify solutions were reindexed
            assert result[1][0]["SolutionIndex"] == 0
            assert result[1][1]["SolutionIndex"] == 1

    @patch('tensilelite.retune_library.globalParameters', {})
    @patch('tensilelite.retune_library.ProblemSizes')
    @patch('tensilelite.retune_library.LibraryIO.parseLibraryLogicData')
    def test_parser_does_not_modify_the_returned_logic(self, mock_parse, mock_problem_sizes):
        """parseCurrentLibrary should hand the parser a copy, since the raw logic is written back in update mode"""
        from tensilelite.retune_library import parseCurrentLibrary

        def modifying_parse(data, *args):
            data.append("modified by the parser")
            return (None, None, Mock(), [], [], None, None)

        mock_parse.side_effect = modifying_parse

        with tempfile.TemporaryDirectory() as tmpdir:
            lib_file = os.path.join(tmpdir, "lib.yaml")
            lib_data = [1, 2, 3]
            with open(lib_file, 'w') as f:
                yaml.dump(lib_data, f)

            result = parseCurrentLibrary(lib_file, None, Mock(), {"isa": "info"}, False)

            assert result[0] == lib_data


@pytest.mark.unit
class TestRunBenchmarking:
    """Test runBenchmarking function"""

    @patch('tensilelite.retune_library.shutil.copy')
    @patch('tensilelite.retune_library.ensurePath')
    @patch('tensilelite.retune_library.ClientWriter.runClient')
    @patch('tensilelite.retune_library.LibraryIO.writeSolutions')
    @patch('tensilelite.retune_library.BenchmarkProblems.writeBenchmarkFiles')
    @patch('tensilelite.retune_library.ClientWriter.getClientExecutablePath')
    @patch('tensilelite.retune_library.popWorkingPath')
    @patch('tensilelite.retune_library.pushWorkingPath')
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/work"})
    def test_runs_benchmarking_creates_correct_structure(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should create directory structure and run benchmarks"""
        from tensilelite.Common.Constants import BENCHMARK_DATA_DIR
        from tensilelite.retune_library import runBenchmarking

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ensure.return_value = tmpdir
            mock_solutions = [{"ISA": (9, 0, 6), "Name": "Sol1"}]
            mock_problem_sizes = Mock()
            mock_run_client.return_value = 0

            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, False, "g++", "gcc", "as", "bundler",
                            {}, 0, "gfx942", Mock(), _problem_type())

            # Verify benchmark files were written
            mock_write_bench.assert_called_once()

            # Verify solutions were written
            mock_write_sol.assert_called_once()

            # Verify client was run
            mock_run_client.assert_called_once()

            # Verify working path was managed (push twice, pop twice)
            assert mock_push.call_count == 2
            assert mock_pop.call_count == 2

            # Verify the results were handed to the library logic step, which reads BENCHMARK_DATA_DIR
            for name in ("benchmark.csv", "benchmark.yaml"):
                mock_copy.assert_any_call(os.path.join("/work", "Data", name),
                                          os.path.join("/work", BENCHMARK_DATA_DIR, name))

    @patch('tensilelite.retune_library.ClientWriter.runClient')
    @patch('tensilelite.retune_library.LibraryIO.writeSolutions')
    @patch('tensilelite.retune_library.BenchmarkProblems.writeBenchmarkFiles')
    @patch('tensilelite.retune_library.ClientWriter.getClientExecutablePath')
    @patch('tensilelite.retune_library.shutil.copy')
    @patch('tensilelite.retune_library.ensurePath')
    @patch('tensilelite.retune_library.popWorkingPath')
    @patch('tensilelite.retune_library.pushWorkingPath')
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/work"})
    def test_sets_update_file_when_update_true(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should set LibraryUpdateFile when update=True"""
        from tensilelite.retune_library import runBenchmarking, globalParameters

        mock_solutions = [{"ISA": (9, 0, 6)}]
        mock_problem_sizes = Mock()
        mock_run_client.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ensure.return_value = tmpdir
            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, True, "g++", "gcc", "as", "bundler",
                            {}, 0, "gfx942", Mock(), _problem_type())

            # Verify LibraryUpdateFile was set
            assert "LibraryUpdateFile" in globalParameters
            assert globalParameters["LibraryUpdateFile"] is not None
            assert "update.yaml" in globalParameters["LibraryUpdateFile"]

    @patch('tensilelite.retune_library.shutil.copy')
    @patch('tensilelite.retune_library.ensurePath')
    @patch('tensilelite.retune_library.ClientWriter.runClient')
    @patch('tensilelite.retune_library.LibraryIO.writeSolutions')
    @patch('tensilelite.retune_library.BenchmarkProblems.writeBenchmarkFiles')
    @patch('tensilelite.retune_library.ClientWriter.getClientExecutablePath')
    @patch('tensilelite.retune_library.popWorkingPath')
    @patch('tensilelite.retune_library.pushWorkingPath')
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/work"})
    def test_converts_isa_to_list(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should convert ISA tuples to lists"""
        from tensilelite.retune_library import runBenchmarking

        mock_solutions = [
            {"ISA": (9, 0, 6), "Name": "Sol1"},
            {"ISA": (10, 1, 0), "Name": "Sol2"}
        ]
        mock_problem_sizes = Mock()
        mock_run_client.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ensure.return_value = tmpdir
            runBenchmarking(mock_solutions, mock_problem_sizes, tmpdir, False, "g++", "gcc", "as", "bundler",
                            {}, 0, "gfx942", Mock(), _problem_type())

            # Verify ISAs were converted to lists
            # LibraryIO.writeSolutions(libraryFile, problemSizes, biasTypeArgs, activationArgs, solutions, ...)
            # So solutions is the 5th arg (index 4)
            write_sol_call_args = mock_write_sol.call_args[0]
            solutions_arg = write_sol_call_args[4]

            assert all(isinstance(sol["ISA"], list) for sol in solutions_arg)
            assert solutions_arg[0]["ISA"] == [9, 0, 6]
            assert solutions_arg[1]["ISA"] == [10, 1, 0]

    @patch('tensilelite.retune_library.shutil.copy')
    @patch('tensilelite.retune_library.ensurePath')
    @patch('tensilelite.retune_library.ClientWriter.runClient')
    @patch('tensilelite.retune_library.LibraryIO.writeSolutions')
    @patch('tensilelite.retune_library.BenchmarkProblems.writeBenchmarkFiles')
    @patch('tensilelite.retune_library.ClientWriter.getClientExecutablePath')
    @patch('tensilelite.retune_library.popWorkingPath')
    @patch('tensilelite.retune_library.pushWorkingPath')
    @patch('tensilelite.retune_library.globalParameters', {"WorkingPath": "/work"})
    def test_benchmarks_bias_gate_and_activation_variants(
        self, mock_push, mock_pop, mock_get_client,
        mock_write_bench, mock_write_sol, mock_run_client, mock_ensure, mock_copy
    ):
        """runBenchmarking should benchmark and write the bias, gate and activation variants of the problem"""
        from tensilelite.Common.DataType import DataType
        from tensilelite.retune_library import runBenchmarking

        mock_run_client.return_value = 0
        problem_type = _problem_type(
            UseBias=1, BiasDataTypeList=[DataType(0), DataType(1)],
            UseGateResidual=True, GateResidualDataTypeList=[DataType(1), DataType(0)],
            ActivationType="hipblaslt_all")

        runBenchmarking([{"ISA": (9, 0, 6)}], Mock(), "/out", False, "g++", "gcc", "as", "bundler",
                        {}, 0, "gfx942", Mock(), problem_type)

        # BenchmarkProblems.writeBenchmarkFiles(dir, solutions, sizes, biasTypeArgs, factorDimArgs,
        #                                       activationArgs, ..., debugConfig, gateTypeArgs, ...)
        bench_args = mock_write_bench.call_args[0]
        bias_args, factor_dim_args, activation_args, gate_args = (
            bench_args[3], bench_args[4], bench_args[5], bench_args[13])
        assert bias_args.biasTypes == [DataType(0)]
        assert factor_dim_args.factorDims == [0]
        assert [s.activationEnum for s in activation_args.settingList] == ["relu"]
        assert gate_args.gateTypes == [DataType(1)]

        # The solutions file is written for the same variants
        write_sol_args, write_sol_kwargs = mock_write_sol.call_args
        assert write_sol_args[2] is bias_args
        assert write_sol_args[3] is activation_args
        assert write_sol_kwargs["gateTypeArgs"] is gate_args


@pytest.mark.unit
class TestTensileRetuneLibrary:
    """Test TensileRetuneLibrary main function"""

    @pytest.fixture(autouse=True)
    def _no_compiler_and_global_state(self):
        """Stand-ins for what needs a compiler, and a copy of globalParameters for the tool to assign to"""
        with patch.multiple('tensilelite.retune_library',
                            makeIsaInfoMap=DEFAULT, applyArchCapOverrides=DEFAULT,
                            makeAssemblyToolchain=DEFAULT, makeSourceToolchain=DEFAULT), \
                patch.dict('tensilelite.retune_library.globalParameters'):
            yield

    @patch('tensilelite.retune_library.LibraryLogic.main')
    @patch('tensilelite.retune_library.LibraryIO.writeYAML')
    @patch('tensilelite.retune_library.LibraryIO.read')
    @patch('tensilelite.retune_library.runBenchmarking')
    @patch('tensilelite.retune_library.parseCurrentLibrary')
    @patch('tensilelite.retune_library.validateToolchain')
    @patch('tensilelite.retune_library.assignGlobalParameters')
    @patch('tensilelite.retune_library.restoreDefaultGlobalParameters')
    @patch('tensilelite.retune_library.argUpdatedGlobalParameters')
    @patch('tensilelite.retune_library.print1')
    def test_retune_library_remake_mode(
        self, mock_print, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read,
        mock_write, mock_logic_main
    ):
        """TensileRetuneLibrary should run benchmarking and rebuild logic in remake mode"""
        from tensilelite.retune_library import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            # Create dummy logic file
            with open(logic_file, 'w') as f:
                yaml.dump({"dummy": "data"}, f)

            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = ["v1", "schedule", "gfx942", "devices", "pt", "sol", "idx", "logic"]
            mock_read.return_value = raw_yaml
            mock_parse.return_value = (raw_yaml, [{"ProblemType": Mock()}], Mock())

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

            # Verify the library is built as msgpack, the only format the standard (YAML-disabled) client loads
            assert mock_assign.call_args[0][0]["LibraryFormat"] == "msgpack"

    @patch('tensilelite.retune_library.LibraryIO.writeYAML')
    @patch('tensilelite.retune_library.LibraryIO.read')
    @patch('tensilelite.retune_library.runBenchmarking')
    @patch('tensilelite.retune_library.parseCurrentLibrary')
    @patch('tensilelite.retune_library.validateToolchain')
    @patch('tensilelite.retune_library.assignGlobalParameters')
    @patch('tensilelite.retune_library.restoreDefaultGlobalParameters')
    @patch('tensilelite.retune_library.argUpdatedGlobalParameters')
    @patch('tensilelite.retune_library.print1')
    def test_retune_library_update_mode_writes_update_logic(
        self, mock_print, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read, mock_write
    ):
        """TensileRetuneLibrary should update logic from file in update mode"""
        from tensilelite.retune_library import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            with open(logic_file, 'w') as f:
                yaml.dump({"dummy": "data"}, f)

            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = [1, 2, "gfx942", 4, 5, 6, 7, "old_logic"]
            update_logic = "new_logic"
            mock_parse.return_value = (raw_yaml, [{"ProblemType": Mock()}], Mock())
            # The logic file is read for its architecture first, then the update file
            mock_read.side_effect = [raw_yaml, update_logic]

            args = [logic_file, output_path, "--update-method", "update"]
            TensileRetuneLibrary(args)

            # Verify YAML was written
            mock_write.assert_called_once()

            # Verify update logic was written to the correct position
            write_args = mock_write.call_args[0]
            written_data = write_args[1]
            assert written_data[7] == update_logic

    @pytest.mark.parametrize("raw_yaml", [
        {
            "ScheduleName": "sched",
            "ArchitectureName": "gfx950",
            "DeviceNames": ["Device 0049"],
            "ExactLogic": [],
        },
        ["v1", "sched", "gfx950", ["Device 0049"], "pt", "sol", "idx", "logic"],
        ["v1", "sched", {"Architecture": "gfx950", "CUCount": 256}, ["Device 0049"], "pt", "sol", "idx", "logic"],
    ], ids=["dict", "list", "list-with-cu-count"])
    @patch('tensilelite.retune_library.LibraryLogic.main')
    @patch('tensilelite.retune_library.LibraryIO.writeYAML')
    @patch('tensilelite.retune_library.LibraryIO.read')
    @patch('tensilelite.retune_library.runBenchmarking')
    @patch('tensilelite.retune_library.parseCurrentLibrary')
    @patch('tensilelite.retune_library.validateToolchain')
    @patch('tensilelite.retune_library.assignGlobalParameters')
    @patch('tensilelite.retune_library.restoreDefaultGlobalParameters')
    @patch('tensilelite.retune_library.argUpdatedGlobalParameters')
    @patch('tensilelite.retune_library.print1')
    def test_retune_library_remake_mode_with_logic_header(
        self, mock_print, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read,
        mock_write, mock_logic_main, raw_yaml
    ):
        """TensileRetuneLibrary should pass the header fields of dict and list logic to LibraryLogic.main in remake mode."""
        from tensilelite.retune_library import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            with open(logic_file, 'w') as f:
                yaml.dump({"dummy": "data"}, f)

            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            mock_read.return_value = raw_yaml
            mock_parse.return_value = (raw_yaml, [{"ProblemType": Mock()}], Mock())

            args = [logic_file, output_path, "--update-method", "remake"]
            TensileRetuneLibrary(args)

            mock_logic_main.assert_called_once()
            header_arg = mock_logic_main.call_args[0][0]
            assert header_arg["ScheduleName"] == "sched"
            assert header_arg["ArchitectureName"] == "gfx950"
            assert header_arg["DeviceNames"] == ["Device 0049"]

    @patch('tensilelite.retune_library.LibraryIO.writeYAML')
    @patch('tensilelite.retune_library.LibraryIO.read')
    @patch('tensilelite.retune_library.runBenchmarking')
    @patch('tensilelite.retune_library.parseCurrentLibrary')
    @patch('tensilelite.retune_library.validateToolchain')
    @patch('tensilelite.retune_library.assignGlobalParameters')
    @patch('tensilelite.retune_library.restoreDefaultGlobalParameters')
    @patch('tensilelite.retune_library.argUpdatedGlobalParameters')
    @patch('tensilelite.retune_library.print1')
    def test_retune_library_update_mode_writes_update_logic_dict(
        self, mock_print, mock_arg_updated, mock_restore,
        mock_assign, mock_validate, mock_parse, mock_bench, mock_read, mock_write
    ):
        """TensileRetuneLibrary should write update logic into ExactLogic for dict-format logic."""
        from tensilelite.retune_library import TensileRetuneLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic.yaml")
            output_path = os.path.join(tmpdir, "output")

            with open(logic_file, 'w') as f:
                yaml.dump({"dummy": "data"}, f)

            mock_validate.return_value = ("g++", "gcc", "as", "bundler")
            mock_arg_updated.return_value = {}

            raw_yaml = {
                "ScheduleName": "sched",
                "ArchitectureName": "gfx950",
                "DeviceNames": ["Device 0049"],
                "ExactLogic": "old_logic",
            }
            update_logic = "new_logic"
            mock_parse.return_value = (raw_yaml, [{"ProblemType": Mock()}], Mock())
            # The logic file is read for its architecture first, then the update file
            mock_read.side_effect = [raw_yaml, update_logic]

            args = [logic_file, output_path, "--update-method", "update"]
            TensileRetuneLibrary(args)

            mock_write.assert_called_once()
            write_args = mock_write.call_args[0]
            written_data = write_args[1]
            assert written_data["ExactLogic"] == update_logic

    @patch('tensilelite.retune_library.restoreDefaultGlobalParameters')
    @patch('tensilelite.retune_library.validateToolchain')
    @patch('tensilelite.retune_library.LibraryIO.read')
    def test_retune_library_exits_without_architecture(self, mock_read, mock_validate, mock_restore, capsys):
        """TensileRetuneLibrary should stop, and say why, when the logic file names no architecture"""
        from tensilelite.retune_library import TensileRetuneLibrary

        mock_validate.return_value = ("g++", "gcc", "as", "bundler")
        mock_read.return_value = ["v1", "schedule"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(SystemExit):
                TensileRetuneLibrary([os.path.join(tmpdir, "logic.yaml"), os.path.join(tmpdir, "output")])

        assert "names no architecture" in capsys.readouterr().out

    @patch('tensilelite.retune_library.runBenchmarking')
    @patch('tensilelite.retune_library.parseCurrentLibrary')
    @patch('tensilelite.retune_library.validateToolchain')
    @patch('tensilelite.retune_library.assignGlobalParameters')
    @patch('tensilelite.retune_library.restoreDefaultGlobalParameters')
    @patch('tensilelite.retune_library.LibraryIO.read')
    def test_retune_library_exits_without_solutions(
        self, mock_read, mock_restore, mock_assign, mock_validate, mock_parse, mock_bench, capsys
    ):
        """TensileRetuneLibrary should stop, and say why, when the logic file holds no solutions"""
        from tensilelite.retune_library import TensileRetuneLibrary

        raw_yaml = ["v1", "schedule", "gfx942", "devices", "pt", "sol", "idx", "logic"]
        mock_validate.return_value = ("g++", "gcc", "as", "bundler")
        mock_read.return_value = raw_yaml
        mock_parse.return_value = (raw_yaml, [], Mock())

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(SystemExit):
                TensileRetuneLibrary([os.path.join(tmpdir, "logic.yaml"), os.path.join(tmpdir, "output")])

        assert "No solutions found" in capsys.readouterr().out
        mock_bench.assert_not_called()


@pytest.mark.unit
class TestMain:
    """Test main entry point"""

    @patch('tensilelite.retune_library.TensileRetuneLibrary')
    @patch('sys.argv', ['prog', 'logic.yaml', 'output/'])
    def test_main_passes_arguments_correctly(self, mock_func):
        """main should pass command line arguments to TensileRetuneLibrary"""
        from tensilelite.retune_library import main

        main()

        # Verify TensileRetuneLibrary was called with correct arguments
        mock_func.assert_called_once_with(['logic.yaml', 'output/'])
