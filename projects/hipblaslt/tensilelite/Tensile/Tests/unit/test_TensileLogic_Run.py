# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile/TensileLogic/Run.py
"""

import pytest
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from Tensile.TensileLogic.Run import _runChecks, _setup, _progress_loop, Check


@pytest.mark.unit
class TestRunChecks:
    """Test _runChecks function"""

    def create_mock_yaml_data(self, num_solutions=1):
        """Helper to create mock YAML data structure"""
        problem_type = {
            "OperationType": "GEMM",
            "DataType": "f32"
        }
        solutions = []
        for i in range(num_solutions):
            solutions.append({
                "SolutionIndex": i,
                "KernelLanguage": "Assembly",
            })
        return [None, None, None, None, problem_type, solutions]

    def test_runs_checks_on_files(self):
        """_runChecks should process files and run validators"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id, \
             patch('Tensile.TensileLogic.Run._validateMatrixInstruction') as mock_matrix, \
             patch('Tensile.TensileLogic.Run._validateWorkGroup') as mock_workgroup, \
             patch('Tensile.TensileLogic.Run._validateWorkGroupMappingXCC') as mock_mapping, \
             patch('Tensile.TensileLogic.Run.handleCustomKernel') as mock_handle_custom, \
             patch('Tensile.TensileLogic.Run.hasCustomKernel') as mock_has_custom:

            # Setup mocks
            mock_chip_id.return_value = True
            mock_matrix.return_value = True
            mock_workgroup.return_value = True
            mock_mapping.return_value = True
            mock_has_custom.return_value = False
            mock_handle_custom.return_value = ({"SolutionIndex": 0}, False)
            mock_read_yaml.return_value = self.create_mock_yaml_data(num_solutions=2)

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test_logic.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # Verify results
                assert total == 2
                assert keep == 2
                assert known_bug_skips == 0
                assert chip_id_failures == 0

                # Verify validators were called
                assert mock_chip_id.call_count == 1
                assert mock_matrix.call_count == 2
                assert mock_workgroup.call_count == 2
                assert mock_mapping.call_count == 2

    def test_skips_experimental_files(self):
        """_runChecks should skip files in Experimental directory"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id:

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                exp_dir = logic_path / "Experimental"
                exp_dir.mkdir()
                test_file = exp_dir / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # Should skip experimental file
                assert total == 0
                assert keep == 0
                mock_chip_id.assert_not_called()
                mock_read_yaml.assert_not_called()

    def test_handles_chip_id_failure(self):
        """_runChecks should track chip ID failures"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id:

            mock_chip_id.return_value = False

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # File should fail chip ID validation
                assert chip_id_failures == 1
                assert total == 0  # No solutions processed
                mock_read_yaml.assert_not_called()

    def test_rejects_invalid_solutions(self):
        """_runChecks should reject solutions that fail validation"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id, \
             patch('Tensile.TensileLogic.Run._validateMatrixInstruction') as mock_matrix, \
             patch('Tensile.TensileLogic.Run._validateWorkGroup') as mock_workgroup, \
             patch('Tensile.TensileLogic.Run._validateWorkGroupMappingXCC') as mock_mapping, \
             patch('Tensile.TensileLogic.Run.handleCustomKernel') as mock_handle_custom, \
             patch('Tensile.TensileLogic.Run.hasCustomKernel') as mock_has_custom:

            # Setup: first solution passes, second fails
            mock_chip_id.return_value = True
            mock_has_custom.return_value = False
            mock_handle_custom.return_value = ({"SolutionIndex": 0}, False)
            mock_matrix.side_effect = [True, False]
            mock_workgroup.return_value = True
            mock_mapping.return_value = True
            mock_read_yaml.return_value = self.create_mock_yaml_data(num_solutions=2)

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                assert total == 2
                assert keep == 1  # Only first solution passed
                assert known_bug_skips == 0

    def test_handles_known_bugs(self):
        """_runChecks should skip validation for known bugs"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id, \
             patch('Tensile.TensileLogic.Run._validateMatrixInstruction') as mock_matrix, \
             patch('Tensile.TensileLogic.Run.handleCustomKernel') as mock_handle_custom, \
             patch('Tensile.TensileLogic.Run.hasCustomKernel') as mock_has_custom, \
             patch('Tensile.TensileLogic.Run.is_known_bug') as mock_is_known_bug:

            # Setup
            mock_chip_id.return_value = True
            mock_has_custom.return_value = False
            mock_handle_custom.return_value = ({"SolutionIndex": 0}, False)
            mock_is_known_bug.return_value = True  # Mark as known bug
            mock_read_yaml.return_value = self.create_mock_yaml_data(num_solutions=1)

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                # KnownBugKey is Tuple[str, int] - (path_string, solution_index)
                known_bugs = frozenset([("test.yaml", 0)])
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # Known bug should be kept without running validators
                assert total == 1
                assert keep == 1
                assert known_bug_skips == 1
                mock_matrix.assert_not_called()

    def test_check_only_custom_kernels(self):
        """_runChecks should only process custom kernels when flag is set"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id, \
             patch('Tensile.TensileLogic.Run._validateMatrixInstruction') as mock_matrix, \
             patch('Tensile.TensileLogic.Run._validateWorkGroup') as mock_workgroup, \
             patch('Tensile.TensileLogic.Run._validateWorkGroupMappingXCC') as mock_mapping, \
             patch('Tensile.TensileLogic.Run.handleCustomKernel') as mock_handle_custom, \
             patch('Tensile.TensileLogic.Run.hasCustomKernel') as mock_has_custom:

            mock_chip_id.return_value = True
            mock_has_custom.return_value = True
            # First solution is custom, second is not
            mock_handle_custom.side_effect = [
                ({"SolutionIndex": 0}, True),
                ({"SolutionIndex": 1}, False)
            ]
            mock_read_yaml.return_value = self.create_mock_yaml_data(num_solutions=2)
            mock_matrix.return_value = True
            mock_workgroup.return_value = True
            mock_mapping.return_value = True

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=True, All=False)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # Only custom kernel should be processed
                assert total == 1
                assert keep == 1

    def test_multiple_validator_failures(self):
        """_runChecks should reject when any validator fails"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id, \
             patch('Tensile.TensileLogic.Run._validateMatrixInstruction') as mock_matrix, \
             patch('Tensile.TensileLogic.Run._validateWorkGroup') as mock_workgroup, \
             patch('Tensile.TensileLogic.Run._validateWorkGroupMappingXCC') as mock_mapping, \
             patch('Tensile.TensileLogic.Run.handleCustomKernel') as mock_handle_custom, \
             patch('Tensile.TensileLogic.Run.hasCustomKernel') as mock_has_custom:

            mock_chip_id.return_value = True
            mock_has_custom.return_value = False
            mock_handle_custom.return_value = ({"SolutionIndex": 0}, False)
            mock_matrix.return_value = True
            mock_workgroup.return_value = False  # This validator fails
            mock_mapping.return_value = True
            mock_read_yaml.return_value = self.create_mock_yaml_data(num_solutions=1)

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # Solution should be rejected
                assert total == 1
                assert keep == 0

    def test_solution_without_solution_index(self):
        """_runChecks should use list index when SolutionIndex is missing"""
        with patch('Tensile.TensileLogic.Run.readYAML') as mock_read_yaml, \
             patch('Tensile.TensileLogic.Run._validateChipId') as mock_chip_id, \
             patch('Tensile.TensileLogic.Run._validateMatrixInstruction') as mock_matrix, \
             patch('Tensile.TensileLogic.Run._validateWorkGroup') as mock_workgroup, \
             patch('Tensile.TensileLogic.Run._validateWorkGroupMappingXCC') as mock_mapping, \
             patch('Tensile.TensileLogic.Run.handleCustomKernel') as mock_handle_custom, \
             patch('Tensile.TensileLogic.Run.hasCustomKernel') as mock_has_custom:

            mock_chip_id.return_value = True
            mock_has_custom.return_value = False
            mock_handle_custom.return_value = ({}, False)  # No SolutionIndex
            mock_matrix.return_value = True
            mock_workgroup.return_value = True
            mock_mapping.return_value = True

            # Create solution without SolutionIndex
            problem_type = {"OperationType": "GEMM"}
            solutions = [{"KernelLanguage": "Assembly"}]  # No SolutionIndex
            mock_read_yaml.return_value = [None, None, None, None, problem_type, solutions]

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_path = Path(tmpdir)
                test_file = logic_path / "test.yaml"
                test_file.write_text("dummy")

                isa_info_map = {}
                check = Check(OnlyCustomKernels=False, All=True)
                known_bugs = frozenset()
                files = [test_file]

                keep, total, known_bug_skips, chip_id_failures = _runChecks(
                    logic_path, isa_info_map, check, known_bugs, files
                )

                # Should process using list_idx as solution index
                assert total == 1
                assert keep == 1


@pytest.mark.unit
class TestSetup:
    """Test _setup function"""

    def test_setup_basic(self):
        """_setup should initialize all components"""
        with patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.makeIsaInfoMap') as mock_make_isa_map, \
             patch('Tensile.TensileLogic.Run.assignGlobalParameters') as mock_assign_gp, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity, \
             patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args:

            # Mock arguments
            mock_args = Mock()
            mock_args.Verbose = 1
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = True
            mock_args.CheckOnlyCustomKernels = False
            mock_args.KnownBugs = None

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_file = Path(tmpdir) / "logic.yaml"
                logic_file.write_text("dummy")
                mock_args.LogicPath = str(logic_file)

                mock_parse_args.return_value = mock_args
                mock_validate_toolchain.return_value = "/usr/bin/g++"
                mock_make_isa_map.return_value = {}

                jobs, isaInfoMap, logicPath, files, check, args = _setup()

                # Verify results
                assert jobs == 4
                assert logicPath == logic_file
                assert len(files) == 1
                assert files[0] == logic_file
                assert check.All is True
                assert check.OnlyCustomKernels is False

                # Verify functions were called
                mock_parse_args.assert_called_once()
                mock_validate_toolchain.assert_called_once_with("/usr/bin/g++")
                mock_make_isa_map.assert_called_once()
                mock_assign_gp.assert_called_once()

    def test_setup_directory_glob(self):
        """_setup should glob for yaml files in directory"""
        with patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.makeIsaInfoMap') as mock_make_isa_map, \
             patch('Tensile.TensileLogic.Run.assignGlobalParameters') as mock_assign_gp, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity, \
             patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args:

            mock_args = Mock()
            mock_args.Verbose = 1
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = True
            mock_args.CheckOnlyCustomKernels = False

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_dir = Path(tmpdir)
                (logic_dir / "logic1.yaml").write_text("dummy1")
                (logic_dir / "logic2.yaml").write_text("dummy2")
                (logic_dir / "readme.txt").write_text("not yaml")

                mock_args.LogicPath = str(logic_dir)
                mock_parse_args.return_value = mock_args
                mock_validate_toolchain.return_value = "/usr/bin/g++"
                mock_make_isa_map.return_value = {}

                jobs, isaInfoMap, logicPath, files, check, args = _setup()

                # Should find 2 yaml files
                assert len(files) == 2
                yaml_names = {f.name for f in files}
                assert "logic1.yaml" in yaml_names
                assert "logic2.yaml" in yaml_names

    def test_setup_exits_with_no_checks(self):
        """_setup should exit if no checks specified"""
        with patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args, \
             patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity:

            mock_args = Mock()
            mock_args.Verbose = 1
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = False
            mock_args.CheckOnlyCustomKernels = False
            mock_args.LogicPath = "/tmp"

            mock_parse_args.return_value = mock_args
            mock_validate_toolchain.return_value = "/usr/bin/g++"

            with pytest.raises(SystemExit) as exc_info:
                _setup()

            assert exc_info.value.code == 0

    def test_setup_exits_with_no_files(self):
        """_setup should exit if no files found"""
        with patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity:

            mock_args = Mock()
            mock_args.Verbose = 1
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = True
            mock_args.CheckOnlyCustomKernels = False

            with tempfile.TemporaryDirectory() as tmpdir:
                empty_dir = Path(tmpdir)
                mock_args.LogicPath = str(empty_dir)

                mock_parse_args.return_value = mock_args
                mock_validate_toolchain.return_value = "/usr/bin/g++"

                with pytest.raises(SystemExit) as exc_info:
                    _setup()

                assert exc_info.value.code == 1

    def test_setup_verbose_mode(self):
        """_setup should handle verbose mode correctly"""
        with patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.makeIsaInfoMap') as mock_make_isa_map, \
             patch('Tensile.TensileLogic.Run.assignGlobalParameters') as mock_assign_gp, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity, \
             patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args:

            mock_args = Mock()
            mock_args.Verbose = 2  # High verbosity
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = True
            mock_args.CheckOnlyCustomKernels = False

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_file = Path(tmpdir) / "logic.yaml"
                logic_file.write_text("dummy")
                mock_args.LogicPath = str(logic_file)

                mock_parse_args.return_value = mock_args
                mock_validate_toolchain.return_value = "/usr/bin/g++"
                mock_make_isa_map.return_value = {}

                jobs, isaInfoMap, logicPath, files, check, args = _setup()

                # In verbose mode (>= 2), setVerbosity is called only once at the start
                assert mock_set_verbosity.call_count == 1
                mock_set_verbosity.assert_called_with(2)

                # Verify PrintSolutionRejectionReason is set in verbose mode
                assert mock_assign_gp.called
                gp_config = mock_assign_gp.call_args[0][0]
                assert "PrintSolutionRejectionReason" in gp_config
                assert gp_config["PrintSolutionRejectionReason"] is True

    def test_setup_non_verbose_mode(self):
        """_setup should not set PrintSolutionRejectionReason in non-verbose mode"""
        with patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.makeIsaInfoMap') as mock_make_isa_map, \
             patch('Tensile.TensileLogic.Run.assignGlobalParameters') as mock_assign_gp, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity, \
             patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args:

            mock_args = Mock()
            mock_args.Verbose = 0  # Not verbose
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = True
            mock_args.CheckOnlyCustomKernels = False

            with tempfile.TemporaryDirectory() as tmpdir:
                logic_file = Path(tmpdir) / "logic.yaml"
                logic_file.write_text("dummy")
                mock_args.LogicPath = str(logic_file)

                mock_parse_args.return_value = mock_args
                mock_validate_toolchain.return_value = "/usr/bin/g++"
                mock_make_isa_map.return_value = {}

                jobs, isaInfoMap, logicPath, files, check, args = _setup()

                # In non-verbose mode (< 2), setVerbosity is called 3 times:
                # 1. Initial setVerbosity(0)
                # 2. setVerbosity(0) before makeIsaInfoMap
                # 3. setVerbosity(0) after assignGlobalParameters
                assert mock_set_verbosity.call_count == 3

                # In non-verbose mode, gp_config should be empty
                gp_config = mock_assign_gp.call_args[0][0]
                assert gp_config == {}

    def test_setup_with_single_file(self):
        """_setup should handle single file path correctly"""
        with patch('Tensile.TensileLogic.Run.validateToolchain') as mock_validate_toolchain, \
             patch('Tensile.TensileLogic.Run.makeIsaInfoMap') as mock_make_isa_map, \
             patch('Tensile.TensileLogic.Run.assignGlobalParameters') as mock_assign_gp, \
             patch('Tensile.TensileLogic.Run.setVerbosity') as mock_set_verbosity, \
             patch('Tensile.TensileLogic.Run.parseArguments') as mock_parse_args:

            mock_args = Mock()
            mock_args.Verbose = 1
            mock_args.Jobs = 4
            mock_args.CxxCompiler = "/usr/bin/g++"
            mock_args.CheckAll = True
            mock_args.CheckOnlyCustomKernels = False

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a single YAML file
                logic_file = Path(tmpdir) / "single_logic.yaml"
                logic_file.write_text("dummy")
                mock_args.LogicPath = str(logic_file)

                mock_parse_args.return_value = mock_args
                mock_validate_toolchain.return_value = "/usr/bin/g++"
                mock_make_isa_map.return_value = {}

                jobs, isaInfoMap, logicPath, files, check, args = _setup()

                # Should return single file in list
                assert len(files) == 1
                assert files[0] == logic_file
                assert logicPath == logic_file


@pytest.mark.unit
class TestProgressLoop:
    """Test _progress_loop function"""

    def test_progress_loop_stops_on_event(self):
        """_progress_loop should stop when event is set"""
        stop_event = threading.Event()

        thread = threading.Thread(
            target=_progress_loop,
            args=(stop_event, 0.1),
            daemon=True
        )
        thread.start()

        # Stop immediately
        stop_event.set()
        thread.join(timeout=1.0)

        # Thread should have stopped
        assert not thread.is_alive()


@pytest.mark.unit
class TestCheckNamedTuple:
    """Test Check NamedTuple"""

    def test_check_creation(self):
        """Check should be created with named fields"""
        check = Check(OnlyCustomKernels=True, All=False)

        assert check.OnlyCustomKernels is True
        assert check.All is False

    def test_check_default_values(self):
        """Check fields should be accessible"""
        check = Check(OnlyCustomKernels=False, All=True)

        assert hasattr(check, 'OnlyCustomKernels')
        assert hasattr(check, 'All')


@pytest.mark.unit
class TestMain:
    """Test main function"""

    def test_main_basic_execution(self):
        """main should execute full workflow"""
        from Tensile.TensileLogic.Run import main

        with patch('Tensile.TensileLogic.Run.ParallelMap2') as mock_parallel_map, \
             patch('Tensile.TensileLogic.Run.load_known_bugs') as mock_load_bugs, \
             patch('Tensile.TensileLogic.Run._setup') as mock_setup, \
             patch('Tensile.TensileLogic.Run.reset_reported_failures') as mock_reset, \
             patch('warnings.filterwarnings'):

            # Mock setup
            mock_args = Mock()
            mock_args.Verbose = 0
            mock_args.KnownBugs = None

            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "logic.yaml"
                test_file.write_text("dummy")

                mock_setup.return_value = (
                    4,  # jobs
                    {},  # isaInfoMap
                    Path(tmpdir),  # logicPath
                    [test_file],  # files
                    Check(OnlyCustomKernels=False, All=True),  # check
                    mock_args
                )

                mock_load_bugs.return_value = frozenset()
                # ParallelMap2 returns list of (keep, total, known_bug_skips, chip_id_failures)
                mock_parallel_map.return_value = [(5, 5, 0, 0)]

                # Should not raise - exits with None
                try:
                    main()
                except SystemExit as e:
                    # Exit code None or 0 means success
                    assert e.code in (None, 0)

                mock_reset.assert_called_once()
                mock_setup.assert_called_once()
                mock_load_bugs.assert_called_once()
                mock_parallel_map.assert_called_once()

    def test_main_with_rejects(self):
        """main should exit with code 1 when solutions are rejected"""
        from Tensile.TensileLogic.Run import main

        with patch('Tensile.TensileLogic.Run.ParallelMap2') as mock_parallel_map, \
             patch('Tensile.TensileLogic.Run.load_known_bugs') as mock_load_bugs, \
             patch('Tensile.TensileLogic.Run._setup') as mock_setup, \
             patch('Tensile.TensileLogic.Run.reset_reported_failures') as mock_reset, \
             patch('warnings.filterwarnings'):

            mock_args = Mock()
            mock_args.Verbose = 0
            mock_args.KnownBugs = None

            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "logic.yaml"
                test_file.write_text("dummy")

                mock_setup.return_value = (
                    4, {}, Path(tmpdir), [test_file],
                    Check(OnlyCustomKernels=False, All=True),
                    mock_args
                )

                mock_load_bugs.return_value = frozenset()
                # 3 kept out of 5 total = 2 rejects
                mock_parallel_map.return_value = [(3, 5, 0, 0)]

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    def test_main_with_chip_id_failures(self):
        """main should exit with code 1 when chip ID failures occur"""
        from Tensile.TensileLogic.Run import main

        with patch('Tensile.TensileLogic.Run.ParallelMap2') as mock_parallel_map, \
             patch('Tensile.TensileLogic.Run.load_known_bugs') as mock_load_bugs, \
             patch('Tensile.TensileLogic.Run._setup') as mock_setup, \
             patch('Tensile.TensileLogic.Run.reset_reported_failures') as mock_reset, \
             patch('warnings.filterwarnings'):

            mock_args = Mock()
            mock_args.Verbose = 0
            mock_args.KnownBugs = None

            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "logic.yaml"
                test_file.write_text("dummy")

                mock_setup.return_value = (
                    4, {}, Path(tmpdir), [test_file],
                    Check(OnlyCustomKernels=False, All=True),
                    mock_args
                )

                mock_load_bugs.return_value = frozenset()
                # keep=5, total=5, known_bug_skips=0, chip_id_failures=1
                mock_parallel_map.return_value = [(5, 5, 0, 1)]

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    def test_main_handles_known_bugs_error(self):
        """main should exit with code 1 on known bugs loading error"""
        from Tensile.TensileLogic.Run import main

        with patch('Tensile.TensileLogic.Run.load_known_bugs') as mock_load_bugs, \
             patch('Tensile.TensileLogic.Run._setup') as mock_setup, \
             patch('Tensile.TensileLogic.Run.reset_reported_failures') as mock_reset, \
             patch('warnings.filterwarnings'):

            mock_args = Mock()
            mock_args.Verbose = 0
            mock_args.KnownBugs = "invalid.yaml"

            mock_setup.return_value = (
                4, {}, Path("/tmp"), [],
                Check(OnlyCustomKernels=False, All=True),
                mock_args
            )

            mock_load_bugs.side_effect = ValueError("Invalid YAML")

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_main_aggregates_multiple_batches(self):
        """main should aggregate results from multiple batches"""
        from Tensile.TensileLogic.Run import main

        with patch('Tensile.TensileLogic.Run.ParallelMap2') as mock_parallel_map, \
             patch('Tensile.TensileLogic.Run.load_known_bugs') as mock_load_bugs, \
             patch('Tensile.TensileLogic.Run._setup') as mock_setup, \
             patch('Tensile.TensileLogic.Run.reset_reported_failures') as mock_reset, \
             patch('warnings.filterwarnings'):

            mock_args = Mock()
            mock_args.Verbose = 0
            mock_args.KnownBugs = None

            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "logic.yaml"
                test_file.write_text("dummy")

                mock_setup.return_value = (
                    4, {}, Path(tmpdir), [test_file],
                    Check(OnlyCustomKernels=False, All=True),
                    mock_args
                )

                mock_load_bugs.return_value = frozenset()
                # Multiple batch results
                mock_parallel_map.return_value = [
                    (5, 5, 1, 0),  # Batch 1
                    (3, 5, 0, 1),  # Batch 2
                    (4, 4, 0, 0),  # Batch 3
                ]

                # Total: 12 keep, 14 total, 1 known_bug_skip, 1 chip_id_failure
                # Rejects: 2, should exit with code 1
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    def test_main_verbose_mode_no_progress(self):
        """main should not show progress in verbose mode"""
        from Tensile.TensileLogic.Run import main

        with patch('Tensile.TensileLogic.Run.ParallelMap2') as mock_parallel_map, \
             patch('Tensile.TensileLogic.Run.load_known_bugs') as mock_load_bugs, \
             patch('Tensile.TensileLogic.Run._setup') as mock_setup, \
             patch('Tensile.TensileLogic.Run.reset_reported_failures') as mock_reset, \
             patch('warnings.filterwarnings'), \
             patch('threading.Thread') as mock_thread:

            mock_args = Mock()
            mock_args.Verbose = 2  # Verbose mode
            mock_args.KnownBugs = None

            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "logic.yaml"
                test_file.write_text("dummy")

                mock_setup.return_value = (
                    4, {}, Path(tmpdir), [test_file],
                    Check(OnlyCustomKernels=False, All=True),
                    mock_args
                )

                mock_load_bugs.return_value = frozenset()
                mock_parallel_map.return_value = [(5, 5, 0, 0)]

                try:
                    main()
                except SystemExit:
                    pass

                # Progress thread should not be created in verbose mode
                mock_thread.assert_not_called()
