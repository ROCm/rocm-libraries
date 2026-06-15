# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile/TensileLogic/Run.py
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from Tensile.TensileLogic.Run import _setup, Check


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
            mock_args.Verbose = 2  # Use Verbose=2 to avoid threading behavior
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
            mock_args.Verbose = 2  # Use Verbose=2 to avoid threading behavior
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
            mock_args.Verbose = 2  # Use Verbose=2 to avoid threading behavior
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
            mock_args.Verbose = 2  # Use Verbose=2 to avoid threading behavior
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
            mock_args.Verbose = 2  # Use Verbose=2 to avoid threading behavior
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
