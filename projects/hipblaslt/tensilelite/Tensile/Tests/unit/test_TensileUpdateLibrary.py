# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for TensileUpdateLibrary.py
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
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

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.readYAML')
    def test_updates_logic_file_basic(self, mock_read, mock_parse, mock_ensure, mock_write):
        """UpdateLogic should read, update, and write library logic file"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        # Create mock problem type
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

        # Create mock solution
        mock_solution = Mock()
        mock_solution_problem_type = Mock()
        mock_solution_problem_type.state = {
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

        mock_solution.getAttributes.return_value = {
            "ProblemType": mock_solution_problem_type,
            "ISA": (9, 0, 6),
            "KernelLanguage": "Assembly"
        }

        # Mock parseLibraryLogicData return
        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution], None, None, None)

        # Mock readYAML return
        mock_read.return_value = [
            {"MinimumRequiredVersion": "4.0.0"},
            {},
            {},
            {},
            {},
            []
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "logic.yaml")
            logicPath = tmpdir
            outputPath = ""

            UpdateLogic(filename, logicPath, outputPath)

            # Verify functions were called
            mock_read.assert_called_once_with(filename)
            mock_parse.assert_called_once()
            mock_ensure.assert_called_once()
            mock_write.assert_called_once()

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.readYAML')
    def test_updates_with_output_path(self, mock_read, mock_parse, mock_ensure, mock_write):
        """UpdateLogic should use output path when provided"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        # Create mock problem type
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

        mock_parse.return_value = (None, None, mock_problem_type, [], None, None, None)
        mock_read.return_value = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "input", "logic.yaml")
            logicPath = os.path.join(tmpdir, "input")
            outputPath = os.path.join(tmpdir, "output")

            UpdateLogic(filename, logicPath, outputPath)

            # Verify write was called with replaced path
            write_call_args = mock_write.call_args
            written_filename = write_call_args[0][0]
            assert outputPath in written_filename

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.readYAML')
    def test_handles_data_type_metadata(self, mock_read, mock_parse, mock_ensure, mock_write):
        """UpdateLogic should handle DataTypeMetadata when present"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

        # Create mock problem type with metadata
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
            "DataTypeMetadata": MockDataType.Int8,
        }

        mock_parse.return_value = (None, None, mock_problem_type, [], None, None, None)
        mock_read.return_value = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "logic.yaml")
            logicPath = tmpdir
            outputPath = ""

            UpdateLogic(filename, logicPath, outputPath)

            # Verify writeYAML was called with updated data
            mock_write.assert_called_once()
            written_data = mock_write.call_args[0][1]
            assert written_data[4]["DataTypeMetadata"] == MockDataType.Int8.value

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.readYAML')
    def test_converts_isa_tuple_to_list(self, mock_read, mock_parse, mock_ensure, mock_write):
        """UpdateLogic should convert ISA tuple to list in solution state"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

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

        # Create solution with ISA tuple
        mock_solution = Mock()
        mock_solution_problem_type = Mock()
        mock_solution_problem_type.state = {
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

        mock_solution.getAttributes.return_value = {
            "ProblemType": mock_solution_problem_type,
            "ISA": (9, 0, 6),  # Tuple
        }

        mock_parse.return_value = (None, None, mock_problem_type, [mock_solution], None, None, None)
        mock_read.return_value = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "logic.yaml")
            UpdateLogic(filename, tmpdir, "")

            # Verify ISA was converted to list
            written_data = mock_write.call_args[0][1]
            assert isinstance(written_data[5][0]["ISA"], list)
            assert written_data[5][0]["ISA"] == [9, 0, 6]

    @patch('Tensile.TensileUpdateLibrary.LibraryIO.writeYAML')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.parseLibraryLogicData')
    @patch('Tensile.TensileUpdateLibrary.LibraryIO.readYAML')
    def test_converts_bias_data_type_list(self, mock_read, mock_parse, mock_ensure, mock_write):
        """UpdateLogic should convert BiasDataTypeList enums to values"""
        from Tensile.TensileUpdateLibrary import UpdateLogic

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
            "BiasDataTypeList": [MockDataType.Float, MockDataType.Half],
            "ActivationComputeDataType": MockDataType.Float,
            "ActivationType": MockActivationType.None_,
            "F32XdlMathOp": MockF32XdlMathOp.FLOAT32,
        }

        mock_parse.return_value = (None, None, mock_problem_type, [], None, None, None)
        mock_read.return_value = [{"MinimumRequiredVersion": "4.0.0"}, {}, {}, {}, {}, []]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "logic.yaml")
            UpdateLogic(filename, tmpdir, "")

            # Verify BiasDataTypeList was converted
            written_data = mock_write.call_args[0][1]
            assert written_data[4]["BiasDataTypeList"] == [MockDataType.Float.value, MockDataType.Half.value]


@pytest.mark.unit
class TestTensileUpdateLibrary:
    """Test TensileUpdateLibrary main function"""

    @patch('Tensile.TensileUpdateLibrary.ParallelMap')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.assignGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.print1')
    @patch('os.walk')
    def test_basic_execution(
        self,
        mock_walk,
        mock_print1,
        mock_restore,
        mock_assign,
        mock_arg_updated,
        mock_ensure,
        mock_parallel
    ):
        """TensileUpdateLibrary should execute basic workflow"""
        from Tensile.TensileUpdateLibrary import TensileUpdateLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock os.walk to return some yaml files
            mock_walk.return_value = [
                (tmpdir, [], ["logic_gfx908.yaml", "logic_gfx90a.yaml", "other.txt"])
            ]

            mock_arg_updated.return_value = {}
            mock_ensure.return_value = tmpdir

            args = ["--logic_path", tmpdir]
            TensileUpdateLibrary(args)

            # Verify key functions were called
            mock_restore.assert_called_once()
            mock_assign.assert_called_once()
            mock_parallel.assert_called_once()

    @patch('Tensile.TensileUpdateLibrary.ParallelMap')
    @patch('Tensile.TensileUpdateLibrary.ensurePath')
    @patch('Tensile.TensileUpdateLibrary.argUpdatedGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.assignGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.restoreDefaultGlobalParameters')
    @patch('Tensile.TensileUpdateLibrary.print1')
    @patch('os.walk')
    def test_handles_output_path(
        self,
        mock_walk,
        mock_print1,
        mock_restore,
        mock_assign,
        mock_arg_updated,
        mock_ensure,
        mock_parallel
    ):
        """TensileUpdateLibrary should use output_path when provided"""
        from Tensile.TensileUpdateLibrary import TensileUpdateLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")

            mock_walk.return_value = [
                (tmpdir, [], ["logic_gfx908.yaml"])
            ]

            mock_arg_updated.return_value = {}
            mock_ensure.return_value = output_dir

            args = ["--logic_path", tmpdir, "--output_path", output_dir]
            TensileUpdateLibrary(args)

            # Verify ensurePath was called
            mock_ensure.assert_called()


@pytest.mark.unit
class TestMain:
    """Test main entry point"""

    @patch('Tensile.TensileUpdateLibrary.TensileUpdateLibrary')
    @patch('sys.argv', ['prog', '--logic_path', '/some/path'])
    def test_main_calls_tensile_update_library(self, mock_func):
        """main should call TensileUpdateLibrary with sys.argv[1:]"""
        from Tensile.TensileUpdateLibrary import main

        main()

        mock_func.assert_called_once_with(['--logic_path', '/some/path'])

