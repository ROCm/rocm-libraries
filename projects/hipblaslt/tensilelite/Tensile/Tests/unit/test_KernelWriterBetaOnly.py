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
from typing import Dict, Any
from types import SimpleNamespace
from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
from Tensile.Common.DataType import DataType


@pytest.fixture
def basic_state() -> Dict[str, Any]:
    """Create a basic state configuration for KernelWriterBetaOnly tests"""
    return {
        "ProblemType": {
            "ComputeDataType": DataType('s'),
            "DestDataType": DataType('s'),
            "DataType": DataType('s'),  # Needed for some tests
            "Index0": 0,
            "Index1": 1,
            "NumIndicesC": 2,
            "StridedBatched": True,
            "GroupedGemm": False,
            "BetaOnlyUseBias": False,
            "UseInitialStridesCD": False,
            "HighPrecisionAccumulate": False,  # Needed for some tests
        },
        "_GlobalAccumulation": False,
    }


@pytest.mark.unit
class TestKernelWriterBetaOnlyInit:
    """Tests for KernelWriterBetaOnly initialization"""

    def test_init_basic(self, basic_state):
        """Test basic initialization"""
        writer = KernelWriterBetaOnly(basic_state)

        assert writer.language == "HIP"
        assert writer.kernelName is not None
        assert len(writer.indexChars) > 0

    def test_init_with_bias(self, basic_state):
        """Test initialization with bias enabled"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 1

        writer = KernelWriterBetaOnly(basic_state)

        # Verify bias is reflected in kernel name: should contain "_BiasS" for single precision
        assert "_BiasS" in writer.kernelName, \
            f"Kernel name should contain '_BiasS' when bias is enabled with DataType('s'), got: {writer.kernelName}"

    def test_init_global_accumulation(self, basic_state):
        """Test initialization with global accumulation"""
        basic_state["_GlobalAccumulation"] = True

        writer = KernelWriterBetaOnly(basic_state)

        # Verify GlobalAccumulation is reflected in kernel name: should contain "_GA"
        assert "_GA" in writer.kernelName, \
            f"Kernel name should contain '_GA' when GlobalAccumulation is enabled, got: {writer.kernelName}"

    @pytest.mark.parametrize("datatype_code,expected_guard", [
        ('f8', "\n#if HIP_FP8_TYPE_OCP\n"),
        ('b8', "\n#if HIP_FP8_TYPE_OCP\n"),
        ('f8n', "\n#if HIP_FP8_TYPE_FNUZ\n"),
        ('b8n', "\n#if HIP_FP8_TYPE_FNUZ\n"),
    ])
    def test_init_float8_types(self, basic_state, datatype_code, expected_guard):
        """Test initialization with float8/bfloat8 OCP and FNUZ types"""

        basic_state["ProblemType"]["DestDataType"] = DataType(datatype_code)
        basic_state["ProblemType"]["ComputeDataType"] = DataType('s')

        writer = KernelWriterBetaOnly(basic_state)

        assert writer.f8MacroGuardStart == expected_guard

    def test_index_chars_assignment(self, basic_state):
        """Test index chars are correctly assigned"""
        writer = KernelWriterBetaOnly(basic_state)

        assert writer.tileChar0 is not None
        assert writer.tileChar1 is not None
        assert "0" in writer.indexChars[0]
        assert "1" in writer.indexChars[1]


@pytest.mark.unit
class TestKernelWriterBetaOnlyFunctionSignature:
    """Tests for function signature generation"""


    def test_function_signature_basic(self, basic_state):
        """Test basic function signature generation"""
        writer = KernelWriterBetaOnly(basic_state)

        sig = writer.functionSignature()

        assert "extern \"C\"" in sig
        assert "__global__" in sig
        assert writer.kernelName in sig
        assert "beta)" in sig

    def test_function_signature_strided_batched(self, basic_state):
        """Test function signature with strided batched"""
        basic_state["ProblemType"]["StridedBatched"] = True
        writer = KernelWriterBetaOnly(basic_state)

        sig = writer.functionSignature()

        assert "* D," in sig
        assert "* C," in sig

    def test_function_signature_non_strided(self, basic_state):
        """Test function signature without strided batched"""
        basic_state["ProblemType"]["StridedBatched"] = False
        writer = KernelWriterBetaOnly(basic_state)

        sig = writer.functionSignature()

        assert "BatchD," in sig or "* BatchD," in sig
        assert "BatchC," in sig or "* BatchC," in sig

    def test_function_signature_with_bias(self, basic_state):
        """Test function signature with bias"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 1
        writer = KernelWriterBetaOnly(basic_state)

        sig = writer.functionSignature()

        assert "Bias," in sig
        assert "strideBias" in sig

    def test_function_signature_bias_mode_3(self, basic_state):
        """Test function signature with bias mode 3"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 3
        writer = KernelWriterBetaOnly(basic_state)

        sig = writer.functionSignature()

        assert "Bias," in sig
        assert "factorDim" in sig

    def test_function_signature_global_accumulation(self, basic_state):
        """Test function signature with global accumulation"""
        # The key difference: with GlobalAccumulation, D pointer uses ComputeDataType
        # Without GlobalAccumulation, D pointer uses DestDataType

        # Test WITH GlobalAccumulation - D pointer should use ComputeDataType
        basic_state["_GlobalAccumulation"] = True
        basic_state["ProblemType"]["ComputeDataType"] = DataType('s')  # float
        basic_state["ProblemType"]["DestDataType"] = DataType('h')     # half
        writer_with_ga = KernelWriterBetaOnly(basic_state)
        sig_with_ga = writer_with_ga.functionSignature()

        # Test WITHOUT GlobalAccumulation - D pointer should use DestDataType
        basic_state_no_ga = basic_state.copy()
        basic_state_no_ga["ProblemType"] = basic_state["ProblemType"].copy()
        basic_state_no_ga["_GlobalAccumulation"] = False
        writer_no_ga = KernelWriterBetaOnly(basic_state_no_ga)
        sig_no_ga = writer_no_ga.functionSignature()

        # With GlobalAccumulation: D pointer should be float (ComputeDataType)
        assert "float * D," in sig_with_ga, \
            "With GlobalAccumulation, D pointer should use ComputeDataType (float)"

        # Without GlobalAccumulation: D pointer should be tensile_half (DestDataType)
        # Note: with StridedBatched=True, it's "D" not "BatchD"
        assert "tensile_half * D," in sig_no_ga, \
            "Without GlobalAccumulation, D pointer should use DestDataType (tensile_half)"

        # Verify they're different
        assert sig_with_ga != sig_no_ga, \
            "GlobalAccumulation should change the D pointer type in function signature"


@pytest.mark.unit
class TestKernelWriterBetaOnlyKernelBody:
    """Tests for kernel body generation"""


    def test_kernel_body_basic(self, basic_state):
        """Test basic kernel body generation"""
        writer = KernelWriterBetaOnly(basic_state)

        body = writer.kernelBodyBetaOnly()

        assert "GLOBAL_D" in body
        assert "GLOBAL_C" in body
        assert "SCALAR_ZERO" in body
        assert "idxD" in body
        assert "idxC" in body

    def test_kernel_body_with_bias(self, basic_state):
        """Test kernel body with bias"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 1
        writer = KernelWriterBetaOnly(basic_state)

        body = writer.kernelBodyBetaOnly()

        assert "Bias" in body

    def test_kernel_body_bias_3d(self, basic_state):
        """Test kernel body with bias and 3+ dimensions"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 1
        basic_state["ProblemType"]["NumIndicesC"] = 3
        writer = KernelWriterBetaOnly(basic_state)

        body = writer.kernelBodyBetaOnly()

        assert "GLOBAL_BIAS" in body

    def test_kernel_body_bias_mode_2(self, basic_state):
        """Test kernel body with bias mode 2"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 2
        writer = KernelWriterBetaOnly(basic_state)

        body = writer.kernelBodyBetaOnly()

        # Verify bias mode 2 uses id1 as the bias index: Bias[id1]
        assert "Bias[id1]" in body, \
            f"Bias mode 2 should access bias with id1 index (Bias[id1]), body:\n{body[:500]}"

    def test_kernel_body_bias_mode_3(self, basic_state):
        """Test kernel body with bias mode 3"""
        basic_state["ProblemType"]["BetaOnlyUseBias"] = True
        basic_state["ProblemType"]["BiasDataType"] = DataType('s')
        basic_state["ProblemType"]["UseBias"] = 3
        writer = KernelWriterBetaOnly(basic_state)

        body = writer.kernelBodyBetaOnly()

        assert "idb" in body or "factorDim" in body

    def test_kernel_body_non_strided_batched(self, basic_state):
        """Test kernel body without strided batched"""
        basic_state["ProblemType"]["StridedBatched"] = False
        writer = KernelWriterBetaOnly(basic_state)

        body = writer.kernelBodyBetaOnly()

        assert "wg" in body
        assert "BatchC" in body

    def test_kernel_body_global_accumulation(self, basic_state):
        """Test kernel body with global accumulation"""
        # GlobalAccumulation affects behavior when StridedBatched=False
        # Set up different data types to see the difference
        basic_state["ProblemType"]["StridedBatched"] = False
        basic_state["ProblemType"]["ComputeDataType"] = DataType('s')  # float
        basic_state["ProblemType"]["DestDataType"] = DataType('h')     # half

        # Test WITH GlobalAccumulation
        basic_state["_GlobalAccumulation"] = True
        writer_with_ga = KernelWriterBetaOnly(basic_state)
        body_with_ga = writer_with_ga.kernelBodyBetaOnly()

        # Test WITHOUT GlobalAccumulation
        basic_state_no_ga = basic_state.copy()
        basic_state_no_ga["ProblemType"] = basic_state["ProblemType"].copy()
        basic_state_no_ga["_GlobalAccumulation"] = False
        writer_no_ga = KernelWriterBetaOnly(basic_state_no_ga)
        body_no_ga = writer_no_ga.kernelBodyBetaOnly()

        # Without GlobalAccumulation, there's a line: "tensile_half * D = BatchD[wg];"
        # With GlobalAccumulation, this line is not present
        assert "D = BatchD[wg];" in body_no_ga, \
            "Without GlobalAccumulation, should have 'D = BatchD[wg];' assignment"
        assert "D = BatchD[wg];" not in body_with_ga, \
            "With GlobalAccumulation, should NOT have 'D = BatchD[wg];' assignment"

        # Verify the bodies are actually different
        assert body_with_ga != body_no_ga, \
            "GlobalAccumulation should produce different kernel bodies"

    def test_kernel_body_high_precision_accumulate(self, basic_state):
        """Test that HighPrecisionAccumulate changes SCALAR_ZERO type for half precision"""
        # The key difference: HighPrecisionAccumulate affects the SCALAR_ZERO type
        # when GlobalAccumulation=True, DataType=half, and ComputeDataType=half
        #
        # With HighPrecisionAccumulate=True: SCALAR_ZERO becomes float
        # With HighPrecisionAccumulate=False: SCALAR_ZERO stays as ComputeDataType (half)

        # Case 1: WITH HighPrecisionAccumulate (should promote half to float)
        state_with_hpa = {
            "ProblemType": {
                "ComputeDataType": DataType('h'),  # half - this is key!
                "DestDataType": DataType('h'),
                "DataType": DataType('h'),
                "Index0": 0,
                "Index1": 1,
                "NumIndicesC": 2,
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
                "UseInitialStridesCD": False,
                "HighPrecisionAccumulate": True,  # Should promote to float
            },
            "_GlobalAccumulation": True,
        }
        writer_with_hpa = KernelWriterBetaOnly(state_with_hpa)
        body_with_hpa = writer_with_hpa.kernelBodyBetaOnly()

        # Case 2: WITHOUT HighPrecisionAccumulate (should keep half)
        state_without_hpa = {
            "ProblemType": {
                "ComputeDataType": DataType('h'),  # half
                "DestDataType": DataType('h'),
                "DataType": DataType('h'),
                "Index0": 0,
                "Index1": 1,
                "NumIndicesC": 2,
                "StridedBatched": True,
                "GroupedGemm": False,
                "BetaOnlyUseBias": False,
                "UseInitialStridesCD": False,
                "HighPrecisionAccumulate": False,  # Should keep half
            },
            "_GlobalAccumulation": True,
        }
        writer_without_hpa = KernelWriterBetaOnly(state_without_hpa)
        body_without_hpa = writer_without_hpa.kernelBodyBetaOnly()

        # Verify the actual difference: WITH HPA uses float, WITHOUT HPA uses tensile_half
        assert "#define SCALAR_ZERO ((float)(0))" in body_with_hpa, \
            "HighPrecisionAccumulate=True should promote SCALAR_ZERO to float"

        assert "#define SCALAR_ZERO ((tensile_half)(0))" in body_without_hpa, \
            "HighPrecisionAccumulate=False should keep SCALAR_ZERO as tensile_half"

        # Ensure they're actually different
        assert body_with_hpa != body_without_hpa, \
            "HighPrecisionAccumulate should produce different output"


@pytest.mark.unit
class TestKernelWriterBetaOnlyKernelName:
    """Tests for kernel name generation"""

    def create_basic_solution(self):
        """Create a basic solution object with minimal state for kernelName tests"""
        return SimpleNamespace(
            _state={
                "ProblemType": {
                    "NumIndicesC": 2,
                    "DestDataType": DataType('s'),
                    "StridedBatched": True,
                    "GroupedGemm": False,
                    "BetaOnlyUseBias": False,
                },
                "_GlobalAccumulation": False,
            }
        )

    def test_kernel_name_basic(self, basic_state):
        """Test basic kernel name generation"""
        solution = self.create_basic_solution()
        name = KernelWriterBetaOnly.kernelName(solution)

        # Format: C + indices + _ + datatype
        # NumIndicesC=2 -> ij, DestDataType=s -> S
        assert name == "Cij_S", f"Expected 'Cij_S', got '{name}'"

    def test_kernel_name_strided_batched(self, basic_state):
        """Test kernel name with strided batched"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["StridedBatched"] = True
        name = KernelWriterBetaOnly.kernelName(solution)

        # StridedBatched=True adds no suffix
        assert name == "Cij_S", f"Expected 'Cij_S', got '{name}'"

    def test_kernel_name_general_batch(self, basic_state):
        """Test kernel name with general batch"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["StridedBatched"] = False
        name = KernelWriterBetaOnly.kernelName(solution)

        # StridedBatched=False adds _GB suffix
        assert name == "Cij_S_GB", f"Expected 'Cij_S_GB', got '{name}'"

    def test_kernel_name_grouped_gemm(self, basic_state):
        """Test kernel name with grouped GEMM"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["GroupedGemm"] = True
        name = KernelWriterBetaOnly.kernelName(solution)

        # GroupedGemm=True adds _GG suffix (takes precedence over StridedBatched)
        assert name == "Cij_S_GG", f"Expected 'Cij_S_GG', got '{name}'"

    def test_kernel_name_global_accumulation(self, basic_state):
        """Test kernel name with global accumulation"""
        solution = self.create_basic_solution()
        solution._state["_GlobalAccumulation"] = True
        name = KernelWriterBetaOnly.kernelName(solution)

        # GlobalAccumulation=True adds _GA suffix at the end
        assert name == "Cij_S_GA", f"Expected 'Cij_S_GA', got '{name}'"

    def test_kernel_name_with_bias(self, basic_state):
        """Test kernel name with bias"""
        solution = self.create_basic_solution()
        solution._state["ProblemType"]["BetaOnlyUseBias"] = True
        solution._state["ProblemType"]["BiasDataType"] = DataType('s')

        btype = DataType('s')
        name = KernelWriterBetaOnly.kernelName(solution, btype)

        # Bias adds _BiasX where X is the bias datatype character
        assert name == "Cij_S_BiasS", f"Expected 'Cij_S_BiasS', got '{name}'"

    def test_kernel_name_different_datatypes(self, basic_state):
        """Test kernel name with different data types"""
        solution = self.create_basic_solution()

        # Test with half precision
        solution._state["ProblemType"]["DestDataType"] = DataType('h')
        name = KernelWriterBetaOnly.kernelName(solution)
        assert name == "Cij_H", f"Expected 'Cij_H' for half precision, got '{name}'"

        # Test with double precision
        solution._state["ProblemType"]["DestDataType"] = DataType('d')
        name = KernelWriterBetaOnly.kernelName(solution)
        assert name == "Cij_D", f"Expected 'Cij_D' for double precision, got '{name}'"


@pytest.mark.unit
class TestKernelWriterBetaOnlyFileGeneration:
    """Tests for source and header file generation"""


    def test_get_source_file_string(self, basic_state):
        """Test source file string generation"""
        writer = KernelWriterBetaOnly(basic_state)

        error_code, source = writer.getSourceFileString()

        assert error_code == 0
        assert len(source) > 0
        assert "extern \"C\"" in source
        assert "__global__" in source

    def test_get_header_file_string(self, basic_state):
        """Test header file string generation"""
        writer = KernelWriterBetaOnly(basic_state)

        header = writer.getHeaderFileString()

        assert len(header) > 0
        assert "extern \"C\"" in header
        assert "__global__" in header
        # Header should end with semicolon
        assert ";" in header

    def test_source_file_grouped_gemm_toggle(self, basic_state):
        """Test source file generation toggles GroupedGemm"""
        writer = KernelWriterBetaOnly(basic_state)

        _, source = writer.getSourceFileString()

        # Should generate both GroupedGemm true and false versions
        # GroupedGemm=True adds "_GG" suffix, GroupedGemm=False has no suffix for StridedBatched
        # Expected kernels: "Cij_S_GG" and "Cij_S" (2 indices, single precision)

        # Check for _GG variant (GroupedGemm=True)
        assert "void Cij_S_GG(" in source, "Should contain GroupedGemm variant Cij_S_GG"

        # Check for non-GG variant (GroupedGemm=False with StridedBatched=True)
        assert "void Cij_S(" in source, "Should contain non-GroupedGemm variant Cij_S"

        # Verify both function definitions exist
        assert source.count("__global__ void") == 2, "Should define exactly 2 kernels (GG and non-GG)"

        # Verify the kernels are distinct (not identical)
        assert source.index("Cij_S_GG") != source.index("Cij_S("), "GG and non-GG variants should be at different positions"

    def test_header_file_grouped_gemm_toggle(self, basic_state):
        """Test header file generation toggles GroupedGemm"""
        writer = KernelWriterBetaOnly(basic_state)

        header = writer.getHeaderFileString()

        # Should generate both GroupedGemm true and false versions
        # Expected declarations: "Cij_S_GG" and "Cij_S"

        # Check for _GG variant (GroupedGemm=True)
        assert "void Cij_S_GG(" in header, "Header should contain GroupedGemm variant Cij_S_GG"

        # Check for non-GG variant (GroupedGemm=False with StridedBatched=True)
        assert "void Cij_S(" in header, "Header should contain non-GroupedGemm variant Cij_S"

        # Both variants should have extern declarations ending with semicolon
        assert header.count("extern \"C\"") == 2, "Should declare exactly 2 kernels (GG and non-GG)"
        assert header.count(";") == 2, "Both declarations should end with semicolon"

    @pytest.mark.parametrize("datatype_code,expected_macro", [
        ('f8', "#if HIP_FP8_TYPE_OCP"),
        ('b8', "#if HIP_FP8_TYPE_OCP"),
        ('f8n', "#if HIP_FP8_TYPE_FNUZ"),
        ('b8n', "#if HIP_FP8_TYPE_FNUZ"),
    ])
    def test_source_file_with_f8_guards(self, basic_state, datatype_code, expected_macro):
        """Test source file generation with F8 macro guards for float8/bfloat8 OCP and FNUZ types"""

        basic_state["ProblemType"]["DestDataType"] = DataType(datatype_code)
        basic_state["ProblemType"]["ComputeDataType"] = DataType('s')

        writer = KernelWriterBetaOnly(basic_state)
        _, source = writer.getSourceFileString()

        assert expected_macro in source
        assert "#endif" in source
