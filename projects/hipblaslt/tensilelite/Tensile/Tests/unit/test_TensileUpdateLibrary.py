# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for TensileUpdateLibrary.py
"""

import pytest
import os
import yaml
import tempfile
from unittest.mock import Mock, patch
from enum import Enum


# Mock DataType enum for testing
class MockDataType(Enum):
    Float = 0
    Double = 1
    Half = 4
    BFloat16 = 7
    Int8 = 8


class MockActivationType(Enum):
    None_ = 0
    All = 1


class MockF32XdlMathOp(Enum):
    XFLOAT32 = 0
    FLOAT32 = 1


@pytest.mark.unit
class TestUpdateLogic:
    """Test UpdateLogic function"""

    def create_mock_problem_type(self):
        """Helper to create mock problem type with state"""
        mock_problem_type = Mock()
        mock_problem_type.state = {
            "DataType": MockDataType.Float,
            "MacDataTypeA": MockDataType.Float,
            "MacDataTypeB": MockDataType.Float,
            "DataTypeA": MockDataType.Float,
            "DataTypeB": MockDataType.Float,
            "DataTypeE": MockDataType.Float,
            "DataTypeAmaxD": MockDataType.Float,
            "DestDataType": MockDataType.Float,
            "ComputeDataType": MockDataType.Float,
            "BiasDataTypeList": [MockDataType.Float],
            "ActivationComputeDataType": MockDataType.Float,
            "ActivationType": MockActivationType.None_,
            "F32XdlMathOp": MockF32XdlMathOp.FLOAT32,
        }
        return mock_problem_type

    def create_mock_solution(self, isa=(9, 0, 6)):
        """Helper to create mock solution"""
        mock_solution = Mock()
        mock_solution_problem_type = self.create_mock_problem_type()

        mock_solution.getAttributes.return_value = {
            "ProblemType": mock_solution_problem_type,
            "ISA": isa,
            "KernelLanguage": "Assembly",
            "SolutionIndex": 0,
        }
        return mock_solution

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    def test_updates_logic_file_basic(self, mock_parse):
        """UpdateLogic should convert enums to values and write updated YAML"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        mock_problem_type = self.create_mock_problem_type()
        mock_solution = self.create_mock_solution()

        # Mock parseLibraryLogicData return
        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution], None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input YAML file with real data
            input_file = os.path.join(tmpdir, "logic.yaml")
            input_data = [
                {"MinimumRequiredVersion": "4.0.0"},
                {"schedule": "data"},
                {"arch": "data"},
                {"devices": "data"},
                {"problem_type": "old"},
                [{"solution": "old"}],
            ]

            with open(input_file, 'w') as f:
                yaml.dump(input_data, f)

            # Run UpdateLogic
            UpdateLogic(input_file, tmpdir, "")

            # Verify output file was written
            assert os.path.exists(input_file)

            # Read and verify the updated YAML
            with open(input_file, 'r') as f:
                updated_data = yaml.safe_load(f)

            # Verify version was updated
            assert "MinimumRequiredVersion" in updated_data[0]

            # Verify problem type was converted to state with enum values
            assert updated_data[4]["DataType"] == MockDataType.Float.value
            assert updated_data[4]["ActivationType"] == MockActivationType.None_.value

            # Verify solution was converted
            assert len(updated_data[5]) == 1
            assert updated_data[5][0]["ISA"] == [9, 0, 6]  # Tuple converted to list

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    def test_updates_with_output_path(self, mock_parse):
        """UpdateLogic should write to output path when provided"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        mock_problem_type = self.create_mock_problem_type()
        mock_parse.return_value = (None, None, mock_problem_type, [], None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)

            input_file = os.path.join(input_dir, "logic.yaml")
            input_data = [
                {"MinimumRequiredVersion": "4.0.0"},
                {}, {}, {}, {}, []
            ]

            with open(input_file, 'w') as f:
                yaml.dump(input_data, f)

            # Run UpdateLogic with output path
            UpdateLogic(input_file, input_dir, output_dir)

            # Verify output file was created in output directory
            expected_output = os.path.join(output_dir, "logic.yaml")
            assert os.path.exists(expected_output)

            # Verify original file is unchanged
            with open(input_file, 'r') as f:
                original_data = yaml.safe_load(f)
            assert original_data == input_data

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    def test_handles_data_type_metadata(self, mock_parse):
        """UpdateLogic should convert DataTypeMetadata when present"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        mock_problem_type = self.create_mock_problem_type()
        mock_problem_type.state["DataTypeMetadata"] = MockDataType.Int8

        mock_parse.return_value = (None, None, mock_problem_type, [], None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "logic.yaml")
            input_data = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

            with open(input_file, 'w') as f:
                yaml.dump(input_data, f)

            UpdateLogic(input_file, tmpdir, "")

            # Verify DataTypeMetadata was converted
            with open(input_file, 'r') as f:
                updated_data = yaml.safe_load(f)

            assert updated_data[4]["DataTypeMetadata"] == MockDataType.Int8.value

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    def test_converts_isa_tuple_to_list(self, mock_parse):
        """UpdateLogic should convert ISA tuple to list in solution"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        mock_problem_type = self.create_mock_problem_type()
        mock_solution = self.create_mock_solution(isa=(9, 4, 2))

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution], None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "logic.yaml")
            input_data = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

            with open(input_file, 'w') as f:
                yaml.dump(input_data, f)

            UpdateLogic(input_file, tmpdir, "")

            with open(input_file, 'r') as f:
                updated_data = yaml.safe_load(f)

            # Verify ISA was converted to list
            assert isinstance(updated_data[5][0]["ISA"], list)
            assert updated_data[5][0]["ISA"] == [9, 4, 2]

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    def test_converts_bias_data_type_list(self, mock_parse):
        """UpdateLogic should convert BiasDataTypeList enums to values"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        mock_problem_type = self.create_mock_problem_type()
        mock_problem_type.state["BiasDataTypeList"] = [MockDataType.Float, MockDataType.Half]

        mock_parse.return_value = (None, None, mock_problem_type, [], None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "logic.yaml")
            input_data = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

            with open(input_file, 'w') as f:
                yaml.dump(input_data, f)

            UpdateLogic(input_file, tmpdir, "")

            with open(input_file, 'r') as f:
                updated_data = yaml.safe_load(f)

            # Verify BiasDataTypeList was converted
            assert updated_data[4]["BiasDataTypeList"] == [MockDataType.Float.value, MockDataType.Half.value]

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    def test_handles_multiple_solutions(self, mock_parse):
        """UpdateLogic should handle multiple solutions correctly"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        mock_problem_type = self.create_mock_problem_type()
        mock_solution1 = self.create_mock_solution(isa=(9, 0, 6))
        mock_solution2 = self.create_mock_solution(isa=(10, 1, 0))
        mock_solution2.getAttributes.return_value["SolutionIndex"] = 1

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution1, mock_solution2], None, None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "logic.yaml")
            input_data = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

            with open(input_file, 'w') as f:
                yaml.dump(input_data, f)

            UpdateLogic(input_file, tmpdir, "")

            with open(input_file, 'r') as f:
                updated_data = yaml.safe_load(f)

            # Verify both solutions were processed
            assert len(updated_data[5]) == 2
            assert updated_data[5][0]["ISA"] == [9, 0, 6]
            assert updated_data[5][1]["ISA"] == [10, 1, 0]


@pytest.mark.unit
class TestTensileUpdateLibrary:
    """Test TensileUpdateLibrary main function"""

    @patch('Tensile.TensileUpdateLibrary.ParallelMap')
    @patch('Tensile.TensileUpdateLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.assignGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.print1')
    def test_finds_and_processes_logic_files(
        self,
        mock_print1,
        mock_restore,
        mock_assign,
        mock_arg_updated,
        mock_parallel
    ):
        """TensileUpdateLibrary should find logic files and process them"""
        from Tensile.TensileUpdateLibrary import TensileUpdateLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test logic files using architecture names from architectureMap
            # Implementation searches for values (arcturus, aldebaran), not keys (gfx908, gfx90a)
            logic_file1 = os.path.join(tmpdir, "logic_arcturus.yaml")
            logic_file2 = os.path.join(tmpdir, "logic_aldebaran.yaml")
            other_file = os.path.join(tmpdir, "other.txt")

            for f in [logic_file1, logic_file2, other_file]:
                with open(f, 'w') as file:
                    file.write("test")

            mock_arg_updated.return_value = {}

            args = ["--logic_path", tmpdir]
            TensileUpdateLibrary(args)

            # Verify ParallelMap was called
            assert mock_parallel.called

            # Get the files that were passed to ParallelMap
            call_args = mock_parallel.call_args
            files_iter = list(call_args[0][1])

            # Verify correct files were found (architecture names, not gfx IDs)
            file_paths = [f[0] for f in files_iter]
            assert len(file_paths) == 2  # Exactly the two architecture logic files
            assert any("arcturus" in f for f in file_paths)
            assert any("aldebaran" in f for f in file_paths)
            assert not any("other.txt" in f for f in file_paths)

    @patch('Tensile.TensileUpdateLibrary.ParallelMap')
    @patch('Tensile.TensileUpdateLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.assignGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.print1')
    def test_handles_output_path(
        self,
        mock_print1,
        mock_restore,
        mock_assign,
        mock_arg_updated,
        mock_parallel
    ):
        """TensileUpdateLibrary should create output directory when specified"""
        from Tensile.TensileUpdateLibrary import TensileUpdateLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            logic_file = os.path.join(tmpdir, "logic_gfx908.yaml")

            with open(logic_file, 'w') as f:
                f.write("test")

            mock_arg_updated.return_value = {}

            args = ["--logic_path", tmpdir, "--output_path", output_dir]
            TensileUpdateLibrary(args)

            # Verify output directory was created
            assert os.path.exists(output_dir)
            assert os.path.isdir(output_dir)

    @patch('Tensile.TensileUpdateLibrary.globalParameters', {})
    @patch('Tensile.TensileUpdateLibrary.ParallelMap')
    @patch('Tensile.TensileUpdateLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.assignGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.print1')
    def test_applies_global_parameter_overrides(
        self,
        mock_print1,
        mock_restore,
        mock_assign,
        mock_arg_updated,
        mock_parallel
    ):
        """TensileUpdateLibrary should apply argument-based parameter overrides"""
        from Tensile.TensileUpdateLibrary import TensileUpdateLibrary, globalParameters

        with tempfile.TemporaryDirectory() as tmpdir:
            logic_file = os.path.join(tmpdir, "logic_gfx908.yaml")
            with open(logic_file, 'w') as f:
                f.write("test")

            # Return some overrides
            mock_arg_updated.return_value = {"TestParam": "TestValue"}

            args = ["--logic_path", tmpdir]
            TensileUpdateLibrary(args)

            # Verify globalParameters was updated
            assert "TestParam" in globalParameters
            assert globalParameters["TestParam"] == "TestValue"


@pytest.mark.unit
class TestMain:
    """Test main entry point"""

    @patch('Tensile.TensileUpdateLibrary.TensileUpdateLibrary')
    @patch('sys.argv', ['prog', '--logic_path', '/some/path'])
    def test_main_passes_arguments_correctly(self, mock_func):
        """main should pass command line arguments to TensileUpdateLibrary"""
        from Tensile.TensileUpdateLibrary import main

        main()

        # Verify TensileUpdateLibrary was called with correct arguments
        mock_func.assert_called_once_with(['--logic_path', '/some/path'])
