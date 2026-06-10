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
from unittest.mock import Mock, MagicMock, patch
from rocisa.code import Module
from rocisa.container import vgpr, sgpr
from rocisa.instruction import SNop, Instruction, VAddF32

import Tensile.Activation as Activation


@pytest.mark.unit
class TestFindUseFunction:
    """Tests for FindUse and FindUseIter functions"""

    def test_find_use_iter_empty_module(self):
        """Test FindUseIter with empty module returns False"""
        FindUseIter = Activation.FindUseIter

        module = Module("test")
        target_inst = Mock(spec=Instruction)
        var_target = vgpr(0)

        isEnd, isUse = FindUseIter(module, target_inst, var_target)

        # Empty module should return False for both
        assert isEnd == False
        assert isUse == False

    def test_find_use_iter_finds_usage_in_srcs(self):
        """Test FindUseIter detects variable usage in instruction sources"""
        FindUseIter = Activation.FindUseIter

        module = Module("test")
        var_target = vgpr(0)
        var_other = vgpr(1)

        # Create instruction that uses var_target in srcs
        inst = VAddF32(dst=var_other, src0=var_target, src1=vgpr(2))
        module.add(inst)

        target_inst = Mock(spec=Instruction)

        isEnd, isUse = FindUseIter(module, target_inst, var_target)

        # Should find the variable used in sources
        assert isEnd == True
        assert isUse == True

    def test_find_use_iter_does_not_find_dst_when_srcs_exist(self):
        """Test FindUseIter only checks dst when instruction has no srcs"""
        FindUseIter = Activation.FindUseIter

        module = Module("test")
        var_target = vgpr(0)

        # Create instruction that assigns to var_target but has srcs
        # FindUseIter only checks dst when srcs is falsy, so this won't be detected
        inst = VAddF32(dst=var_target, src0=vgpr(1), src1=vgpr(2))
        module.add(inst)

        target_inst = Mock(spec=Instruction)

        isEnd, isUse = FindUseIter(module, target_inst, var_target)

        # Should NOT find the assignment because instruction has srcs
        assert isEnd == False
        assert isUse == False

    def test_find_use_wrapper(self):
        """Test FindUse wrapper returns only isUse value"""
        FindUse = Activation.FindUse

        module = Module("test")
        var_target = vgpr(0)

        # Create instruction that uses var_target
        inst = VAddF32(dst=vgpr(1), src0=var_target, src1=vgpr(2))
        module.add(inst)

        target_inst = Mock(spec=Instruction)

        isUse = FindUse(module, target_inst, var_target)

        # Should return True for usage
        assert isUse == True


@pytest.mark.unit
class TestFindAssignAndUseFunction:
    """Tests for FindAssignAndUse and FindAssignAndUseIter functions"""

    def test_find_assign_and_use_iter_empty_module(self):
        """Test FindAssignAndUseIter with empty module returns False"""
        FindAssignAndUseIter = Activation.FindAssignAndUseIter

        module = Module("test")
        end_inst = Mock(spec=Instruction)
        assign_var = vgpr(0)
        use_var = vgpr(1)

        isEnd, isUse = FindAssignAndUseIter(module, end_inst, assign_var, use_var)

        # Empty module should return False for both
        assert isEnd == False
        assert isUse == False

    def test_find_assign_and_use_iter_finds_assignment(self):
        """Test FindAssignAndUseIter detects assignment to assignVar"""
        FindAssignAndUseIter = Activation.FindAssignAndUseIter

        module = Module("test")
        assign_var = vgpr(0)
        use_var = vgpr(1)

        # Create instruction that assigns to assign_var
        inst = VAddF32(dst=assign_var, src0=vgpr(2), src1=vgpr(3))
        module.add(inst)

        end_inst = Mock(spec=Instruction)

        isEnd, isUse = FindAssignAndUseIter(module, end_inst, assign_var, use_var)

        # Should detect assignment
        assert isEnd == True
        assert isUse == True

    def test_find_assign_and_use_iter_finds_use(self):
        """Test FindAssignAndUseIter detects usage of useVar"""
        FindAssignAndUseIter = Activation.FindAssignAndUseIter

        module = Module("test")
        assign_var = vgpr(0)
        use_var = vgpr(1)

        # Create instruction that uses use_var in sources
        inst = VAddF32(dst=vgpr(2), src0=use_var, src1=vgpr(3))
        module.add(inst)

        end_inst = Mock(spec=Instruction)

        isEnd, isUse = FindAssignAndUseIter(module, end_inst, assign_var, use_var)

        # Should detect usage
        assert isEnd == True
        assert isUse == True

    def test_find_assign_and_use_iter_stops_at_end_inst(self):
        """Test FindAssignAndUseIter stops when encountering endInst"""
        FindAssignAndUseIter = Activation.FindAssignAndUseIter

        module = Module("test")
        assign_var = vgpr(0)
        use_var = vgpr(1)

        # Create end instruction
        end_inst = VAddF32(dst=vgpr(2), src0=vgpr(3), src1=vgpr(4))
        module.add(end_inst)

        # Add another instruction after (should not be checked)
        inst_after = VAddF32(dst=assign_var, src0=vgpr(5), src1=vgpr(6))
        module.add(inst_after)

        isEnd, isUse = FindAssignAndUseIter(module, end_inst, assign_var, use_var)

        # Should stop at end_inst without finding assignment
        assert isEnd == True
        assert isUse == False


@pytest.mark.unit
class TestConvertCoeffToHex:
    """Tests for ConvertCoeffToHex function"""

    def test_convert_coeff_to_hex_basic(self):
        """Test ConvertCoeffToHex handles module without error"""
        from Tensile.Common.DataType import DataType

        ConvertCoeffToHex = Activation.ConvertCoeffToHex

        module = Module("test")
        dt = DataType("s")

        # ConvertCoeffToHex takes (module, cDataType, isPack)
        result = ConvertCoeffToHex(module, dt, False)

        # Should return a module (may be empty or with converted coefficients)
        assert result is not None
        assert isinstance(result, Module)


@pytest.mark.unit
class TestHexToStr:
    """Tests for HexToStr function"""

    def test_hex_to_str_basic(self):
        """Test HexToStr converts hex to string representation"""
        from Tensile.Common.DataType import DataType

        HexToStr = Activation.HexToStr

        dt = DataType("s")
        result = HexToStr(dt, False, 0x3f800000)  # 1.0 in float

        # Should return the correct hex string
        assert result == "0x3f800000"

    def test_hex_to_str_zero(self):
        """Test HexToStr with zero value"""
        from Tensile.Common.DataType import DataType

        HexToStr = Activation.HexToStr

        dt = DataType("s")
        result = HexToStr(dt, False, 0x0)

        assert result == "0x0"

    def test_hex_to_str_half_packed(self):
        """Test HexToStr with packed half precision"""
        from Tensile.Common.DataType import DataType

        HexToStr = Activation.HexToStr

        dt = DataType("h")
        # When isPack=True and datatype is half, it should duplicate the value
        # 0x3c00 (1.0 in half) -> should become 0x3c003c00
        result = HexToStr(dt, True, 0x3c00)

        assert result == "0x3c003c00"

    def test_hex_to_str_half_not_packed(self):
        """Test HexToStr with non-packed half precision"""
        from Tensile.Common.DataType import DataType

        HexToStr = Activation.HexToStr

        dt = DataType("h")
        result = HexToStr(dt, False, 0x3c00)

        assert result == "0x3c00"


@pytest.mark.unit
class TestFuseInstructionIntegration:
    """Integration tests for instruction fusion"""

    def test_combine_instructions_between_modules_with_empty(self):
        """Test CombineInstructionsBetweenModules handles empty module"""
        CombineInstructionsBetweenModules = Activation.CombineInstructionsBetweenModules

        module = Module("test")
        moduleAndIndex = {}

        # Should handle empty module without error
        CombineInstructionsBetweenModules(module, moduleAndIndex, False)

        # Module should remain valid (empty or unchanged)
        assert module is not None
        instructions = module.items()
        assert len(instructions) == 0, "Empty module should remain empty"

    def test_combine_instructions_between_modules_with_nested(self):
        """Test CombineInstructionsBetweenModules handles nested structure without error"""
        CombineInstructionsBetweenModules = Activation.CombineInstructionsBetweenModules

        outer = Module("outer")
        inner = Module("inner")
        outer.appendModule(inner)

        moduleAndIndex = {}

        # Should handle nested modules without error
        CombineInstructionsBetweenModules(outer, moduleAndIndex, False)

        # Function should complete without raising an exception
        assert outer is not None
        assert isinstance(outer, Module)


@pytest.mark.unit
class TestActivationMagicNumbersUsage:
    """Tests for using activation magic numbers"""

    def test_float_union_with_gelu_k0(self):
        """Test floatUnion with GELU K0 magic number"""
        floatUnion = Activation.floatUnion
        ActivationMagicNumbers = Activation.ActivationMagicNumbers

        f = floatUnion()
        f.u = ActivationMagicNumbers['FloatGeluK0']

        # Should be a valid float
        assert isinstance(f.f, float)
        assert f.f != 0.0

    def test_float_union_with_gelu_k1(self):
        """Test floatUnion with GELU K1 magic number"""
        floatUnion = Activation.floatUnion
        ActivationMagicNumbers = Activation.ActivationMagicNumbers

        f = floatUnion()
        f.u = ActivationMagicNumbers['FloatGeluK1']

        # Should be a valid float
        assert isinstance(f.f, float)

    def test_all_magic_numbers_are_valid(self):
        """Test that all magic numbers are valid integers"""
        ActivationMagicNumbers = Activation.ActivationMagicNumbers

        for key, value in ActivationMagicNumbers.items():
            assert isinstance(value, int)
            assert value > 0


@pytest.mark.unit
class TestActivationLookupVeri:
    """Tests for ActivationType.lookupVeri dictionary"""

    def test_lookup_veri_contains_exp(self):
        """Test that lookupVeri contains 'exp'"""
        ActivationType = Activation.ActivationType

        assert 'exp' in ActivationType.lookupVeri

    def test_lookup_veri_exp_structure(self):
        """Test lookupVeri['exp'] structure"""
        ActivationType = Activation.ActivationType

        exp = ActivationType.lookupVeri['exp']
        assert hasattr(exp, 'name')
        assert exp.name == 'exp'
        assert hasattr(exp, 'isGradient')


@pytest.mark.unit
class TestActivationTypeInstantiation:
    """Tests for ActivationType instantiation with lookupVeri"""

    def test_activation_type_exp(self):
        """Test ActivationType with 'exp' from lookupVeri"""
        ActivationType = Activation.ActivationType

        act = ActivationType('exp')
        assert act.value == 'exp'

    def test_activation_type_exp_case_insensitive(self):
        """Test ActivationType 'exp' is case insensitive"""
        ActivationType = Activation.ActivationType

        act = ActivationType('EXP')
        assert act.value == 'exp'
