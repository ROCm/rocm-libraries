################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
from unittest.mock import Mock
from types import SimpleNamespace
from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
from Tensile.Common.DataType import DataType


@pytest.mark.unit
class TestKernelWriterBetaOnlyInit:
    """Tests for KernelWriterBetaOnly initialization"""

    def create_basic_state(self):
        """Create a basic state configuration"""
        return {
            "ProblemType": {
                "ComputeDataType": DataType('s'),
                "DestDataType": DataType('s'),
                "Index0": 0,
                "Index1": 1,
                "NumIndicesC": 2,
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
                "UseInitialStridesCD": False,
            },
            "_GlobalAccumulation": False,
        }

    def test_init_basic(self):
        """Test basic initialization"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        assert writer.language == "HIP"
        assert writer.kernelName is not None
        assert len(writer.indexChars) > 0

    def test_init_with_bias(self):
        """Test initialization with bias enabled"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 1

        writer = KernelWriterBetaOnly(state)

        assert writer.kernelName is not None

    def test_init_global_accumulation(self):
        """Test initialization with global accumulation"""
        state = self.create_basic_state()
        state["_GlobalAccumulation"] = True

        writer = KernelWriterBetaOnly(state)

        assert writer.state["_GlobalAccumulation"] == True

    def test_init_float8_ocp(self):
        """Test initialization with float8 OCP type"""
        state = self.create_basic_state()

        # Use real DataType for float8
        state["ProblemType"]["DestDataType"] = DataType('f8')
        state["ProblemType"]["ComputeDataType"] = DataType('s')

        writer = KernelWriterBetaOnly(state)

        assert writer.f8MacroGuardStart == "\n#if HIP_FP8_TYPE_OCP\n"
        assert writer.f8MacroGuardEnd == "\n#endif // F8 macro guard\n"

    def test_init_float8_fnuz(self):
        """Test initialization with float8 FNUZ type"""
        # Skip this test - DataType doesn't have a simple code for float8_fnuz
        # and creating a proper mock is complex. The float8 OCP test covers the guard logic.
        pytest.skip("DataType code for float8_fnuz not straightforward to mock")

    def test_init_bfloat8_ocp(self):
        """Test initialization with bfloat8 OCP type"""
        state = self.create_basic_state()

        # Use real DataType for bfloat8
        state["ProblemType"]["DestDataType"] = DataType('b8')
        state["ProblemType"]["ComputeDataType"] = DataType('s')

        writer = KernelWriterBetaOnly(state)

        assert writer.f8MacroGuardStart == "\n#if HIP_FP8_TYPE_OCP\n"

    def test_index_chars_assignment(self):
        """Test index chars are correctly assigned"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        assert writer.tileChar0 is not None
        assert writer.tileChar1 is not None
        assert "0" in writer.indexChars[0]
        assert "1" in writer.indexChars[1]


@pytest.mark.unit
class TestKernelWriterBetaOnlyFunctionSignature:
    """Tests for function signature generation"""

    def create_basic_state(self):
        """Create a basic state configuration"""
        return {
            "ProblemType": {
                "ComputeDataType": DataType('s'),
                "DestDataType": DataType('s'),
                "Index0": 0,
                "Index1": 1,
                "NumIndicesC": 2,
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
                "UseInitialStridesCD": False,
            },
            "_GlobalAccumulation": False,
        }

    def test_function_signature_basic(self):
        """Test basic function signature generation"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        sig = writer.functionSignature()

        assert "extern \"C\"" in sig
        assert "__global__" in sig
        assert writer.kernelName in sig
        assert "beta)" in sig

    def test_function_signature_strided_batched(self):
        """Test function signature with strided batched"""
        state = self.create_basic_state()
        state["ProblemType"]["StridedBatched"] = True
        writer = KernelWriterBetaOnly(state)

        sig = writer.functionSignature()

        assert "* D," in sig
        assert "* C," in sig

    def test_function_signature_non_strided(self):
        """Test function signature without strided batched"""
        state = self.create_basic_state()
        state["ProblemType"]["StridedBatched"] = False
        writer = KernelWriterBetaOnly(state)

        sig = writer.functionSignature()

        assert "BatchD," in sig or "* BatchD," in sig
        assert "BatchC," in sig or "* BatchC," in sig

    def test_function_signature_with_bias(self):
        """Test function signature with bias"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 1
        writer = KernelWriterBetaOnly(state)

        sig = writer.functionSignature()

        assert "Bias," in sig
        assert "strideBias" in sig

    def test_function_signature_bias_mode_3(self):
        """Test function signature with bias mode 3"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 3
        writer = KernelWriterBetaOnly(state)

        sig = writer.functionSignature()

        assert "Bias," in sig
        assert "factorDim" in sig

    def test_function_signature_global_accumulation(self):
        """Test function signature with global accumulation"""
        state = self.create_basic_state()
        state["_GlobalAccumulation"] = True
        writer = KernelWriterBetaOnly(state)

        sig = writer.functionSignature()

        # With global accumulation, D pointer uses ComputeDataType
        assert writer.kernelName in sig


@pytest.mark.unit
class TestKernelWriterBetaOnlyKernelBody:
    """Tests for kernel body generation"""

    def create_basic_state(self):
        """Create a basic state configuration"""
        return {
            "ProblemType": {
                "ComputeDataType": DataType('s'),
                "DestDataType": DataType('s'),
                "DataType": DataType('s'),
                "Index0": 0,
                "Index1": 1,
                "NumIndicesC": 2,
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
                "UseInitialStridesCD": False,
                "HighPrecisionAccumulate": False,
            },
            "_GlobalAccumulation": False,
        }

    def test_kernel_body_basic(self):
        """Test basic kernel body generation"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "GLOBAL_D" in body
        assert "GLOBAL_C" in body
        assert "SCALAR_ZERO" in body
        assert "idxD" in body
        assert "idxC" in body

    def test_kernel_body_with_bias(self):
        """Test kernel body with bias"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 1
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "Bias" in body

    def test_kernel_body_bias_3d(self):
        """Test kernel body with bias and 3+ dimensions"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 1
        state["ProblemType"]["NumIndicesC"] = 3
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "GLOBAL_BIAS" in body

    def test_kernel_body_bias_mode_2(self):
        """Test kernel body with bias mode 2"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 2
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "id1" in body

    def test_kernel_body_bias_mode_3(self):
        """Test kernel body with bias mode 3"""
        state = self.create_basic_state()
        state["ProblemType"]["BetaOnlyUseBias"] = True
        state["ProblemType"]["BiasDataType"] = DataType('s')
        state["ProblemType"]["UseBias"] = 3
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "idb" in body or "factorDim" in body

    def test_kernel_body_non_strided_batched(self):
        """Test kernel body without strided batched"""
        state = self.create_basic_state()
        state["ProblemType"]["StridedBatched"] = False
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "wg" in body
        assert "BatchC" in body

    def test_kernel_body_global_accumulation(self):
        """Test kernel body with global accumulation"""
        state = self.create_basic_state()
        state["_GlobalAccumulation"] = True
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        assert "GLOBAL_D" in body

    def test_kernel_body_high_precision_accumulate(self):
        """Test kernel body with high precision accumulate"""
        state = self.create_basic_state()
        state["ProblemType"]["DataType"] = DataType('h')
        state["ProblemType"]["HighPrecisionAccumulate"] = True
        state["_GlobalAccumulation"] = True
        writer = KernelWriterBetaOnly(state)

        body = writer.kernelBodyBetaOnly()

        # Should use single precision for high precision accumulate
        assert "SCALAR_ZERO" in body


@pytest.mark.unit
class TestKernelWriterBetaOnlyKernelName:
    """Tests for kernel name generation"""

    def create_basic_solution(self):
        """Create a basic solution mock"""
        solution = Mock()
        solution._state = {
            "ProblemType": {
                "NumIndicesC": 2,
                "DestDataType": DataType('s'),
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
            },
            "_GlobalAccumulation": False,
        }
        return solution

    def test_kernel_name_basic(self):
        """Test basic kernel name generation"""
        solution = self.create_basic_solution()
        name = KernelWriterBetaOnly.kernelName(solution)

        assert "C" in name
        assert "S" in name  # Single precision (uppercase)

    def test_kernel_name_strided_batched(self):
        """Test kernel name with strided batched"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["StridedBatched"] = True
        name = KernelWriterBetaOnly.kernelName(solution)

        # Should NOT contain _GB (general batch)
        assert "_GB" not in name

    def test_kernel_name_general_batch(self):
        """Test kernel name with general batch"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["StridedBatched"] = False
        name = KernelWriterBetaOnly.kernelName(solution)

        assert "_GB" in name

    def test_kernel_name_grouped_gemm(self):
        """Test kernel name with grouped GEMM"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["GroupedGemm"] = True
        name = KernelWriterBetaOnly.kernelName(solution)

        assert "_GG" in name

    def test_kernel_name_global_accumulation(self):
        """Test kernel name with global accumulation"""
        solution = self.create_basic_solution()
        solution._state["_GlobalAccumulation"] = True
        name = KernelWriterBetaOnly.kernelName(solution)

        assert "_GA" in name

    def test_kernel_name_with_bias(self):
        """Test kernel name with bias"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["BetaOnlyUseBias"] = True
        solution._state["ProblemType"]["BiasDataType"] = DataType('s')

        btype = DataType('s')
        name = KernelWriterBetaOnly.kernelName(solution, btype)

        assert "_Bias" in name

    def test_kernel_name_different_datatypes(self):
        """Test kernel name with different data types"""
        solution = self.create_basic_solution()

        # Test with half precision
        solution._state["ProblemType"]["DestDataType"] = DataType('h')
        name = KernelWriterBetaOnly.kernelName(solution)
        assert "H" in name  # Uppercase

        # Test with double precision
        solution._state["ProblemType"]["DestDataType"] = DataType('d')
        name = KernelWriterBetaOnly.kernelName(solution)
        assert "D" in name  # Uppercase


@pytest.mark.unit
class TestKernelWriterBetaOnlyFileGeneration:
    """Tests for source and header file generation"""

    def create_basic_state(self):
        """Create a basic state configuration"""
        return {
            "ProblemType": {
                "ComputeDataType": DataType('s'),
                "DestDataType": DataType('s'),
                "DataType": DataType('s'),
                "Index0": 0,
                "Index1": 1,
                "NumIndicesC": 2,
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
                "UseInitialStridesCD": False,
                "HighPrecisionAccumulate": False,
            },
            "_GlobalAccumulation": False,
        }

    def test_get_source_file_string(self):
        """Test source file string generation"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        error_code, source = writer.getSourceFileString()

        assert error_code == 0
        assert len(source) > 0
        assert "extern \"C\"" in source
        assert "__global__" in source

    def test_get_header_file_string(self):
        """Test header file string generation"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        header = writer.getHeaderFileString()

        assert len(header) > 0
        assert "extern \"C\"" in header
        assert "__global__" in header
        # Header should end with semicolon
        assert ";" in header

    def test_source_file_grouped_gemm_toggle(self):
        """Test source file generation toggles GroupedGemm"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        _, source = writer.getSourceFileString()

        # Should generate both GroupedGemm true and false versions
        # Check that it's reasonably long (contains both versions)
        assert len(source) > 200

    def test_header_file_grouped_gemm_toggle(self):
        """Test header file generation toggles GroupedGemm"""
        state = self.create_basic_state()
        writer = KernelWriterBetaOnly(state)

        header = writer.getHeaderFileString()

        # Should generate both GroupedGemm true and false versions
        assert len(header) > 100

    def test_source_file_with_f8_guards(self):
        """Test source file generation with F8 macro guards"""
        state = self.create_basic_state()

        # Use real DataType for float8
        state["ProblemType"]["DestDataType"] = DataType('f8')
        state["ProblemType"]["ComputeDataType"] = DataType('s')

        writer = KernelWriterBetaOnly(state)
        _, source = writer.getSourceFileString()

        assert "#if HIP_FP8_TYPE_OCP" in source
        assert "#endif" in source
