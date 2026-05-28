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
from rocisa.instruction import SNop, Instruction


@pytest.fixture(scope="module")
def Activation():
    """Lazy import Activation module"""
    import Tensile.Activation as act
    return act


@pytest.mark.unit
class TestFindUseFunction:
    """Tests for FindUse and FindUseIter functions"""

    def test_find_use_iter_empty_module(self, Activation):
        """Test FindUseIter with empty module returns False"""
        from rocisa.container import vgpr
        FindUseIter = Activation.FindUseIter

        module = Module("test")
        target_inst = Mock(spec=Instruction)
        var_target = vgpr(0)

        isEnd, isUse = FindUseIter(module, target_inst, var_target)

        # Empty module should return False for both
        assert isEnd == False
        assert isUse == False

    def test_find_use_iter_finds_usage_in_srcs(self, Activation):
        """Test FindUseIter detects variable usage in instruction sources"""
        from rocisa.container import vgpr
        from rocisa.instruction import VAddF32
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

    def test_find_use_iter_does_not_find_dst_when_srcs_exist(self, Activation):
        """Test FindUseIter only checks dst when instruction has no srcs"""
        from rocisa.container import vgpr
        from rocisa.instruction import VAddF32
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

    def test_find_use_wrapper(self, Activation):
        """Test FindUse wrapper returns only isUse value"""
        from rocisa.container import vgpr
        from rocisa.instruction import VAddF32
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

    def test_find_assign_and_use_iter_empty_module(self, Activation):
        """Test FindAssignAndUseIter with empty module returns False"""
        from rocisa.container import vgpr
        FindAssignAndUseIter = Activation.FindAssignAndUseIter

        module = Module("test")
        end_inst = Mock(spec=Instruction)
        assign_var = vgpr(0)
        use_var = vgpr(1)

        isEnd, isUse = FindAssignAndUseIter(module, end_inst, assign_var, use_var)

        # Empty module should return False for both
        assert isEnd == False
        assert isUse == False

    def test_find_assign_and_use_iter_finds_assignment(self, Activation):
        """Test FindAssignAndUseIter detects assignment to assignVar"""
        from rocisa.container import vgpr
        from rocisa.instruction import VAddF32
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

    def test_find_assign_and_use_iter_finds_use(self, Activation):
        """Test FindAssignAndUseIter detects usage of useVar"""
        from rocisa.container import vgpr
        from rocisa.instruction import VAddF32
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

    def test_find_assign_and_use_iter_stops_at_end_inst(self, Activation):
        """Test FindAssignAndUseIter stops when encountering endInst"""
        from rocisa.container import vgpr
        from rocisa.instruction import VAddF32
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
class TestReplaceAndRemoveInstFunctions:
    """Tests for replaceInst and removeOldInst helper functions"""

    # These functions are internal helpers - no need to test existence


@pytest.mark.unit
class TestConvertCoeffToHex:
    """Tests for ConvertCoeffToHex function"""

    def test_convert_coeff_to_hex_basic(self, Activation):
        """Test ConvertCoeffToHex with basic module"""
        from Tensile.Common.DataType import DataType

        ConvertCoeffToHex = Activation.ConvertCoeffToHex

        module = Module("test")
        dt = DataType("s")

        # ConvertCoeffToHex may have different signature - check first
        import inspect
        sig = inspect.signature(ConvertCoeffToHex)

        if len(sig.parameters) == 3:
            result = ConvertCoeffToHex(module, dt, False)
        else:
            result = ConvertCoeffToHex(module, dt)

        # Should return a module
        assert result is not None


@pytest.mark.unit
class TestHexToStr:
    """Tests for HexToStr function"""

    def test_hex_to_str_basic(self, Activation):
        """Test HexToStr with basic input"""
        from Tensile.Common.DataType import DataType

        HexToStr = Activation.HexToStr

        dt = DataType("s")
        result = HexToStr(dt, False, 0x3f800000)  # 1.0 in float

        # Should return a string
        assert isinstance(result, str)


@pytest.mark.unit
class TestHolderToGpr:
    """Tests for HolderToGpr function"""

    def test_holder_to_gpr_basic(self, Activation):
        """Test HolderToGpr with basic module"""
        HolderToGpr = Activation.HolderToGpr

        module = Module("test")
        result = HolderToGpr(module, 0, "v")

        # Should return a module
        assert result is not None


@pytest.mark.unit
class TestCreateVgprIdxList:
    """Tests for createVgprIdxList function"""

    def test_create_vgpr_idx_list_basic(self, Activation):
        """Test createVgprIdxList with basic module"""
        createVgprIdxList = Activation.createVgprIdxList

        module = Module("test")
        result = createVgprIdxList(module, [0, 1], "")

        # Should return a list
        assert isinstance(result, list)


@pytest.mark.unit
class TestFuseInstructionIntegration:
    """Integration tests for instruction fusion"""

    def test_fuse_instruction_basic(self, Activation):
        """Test FuseInstruction with basic instruction"""
        FuseInstruction = Activation.FuseInstruction

        # Create mock instruction
        inst = Mock(spec=Instruction)
        inst.dst = Mock()
        inst.srcs = []

        moduleAndIndex = {}

        # Should not crash
        result = FuseInstruction(inst, moduleAndIndex, fuseDebug=False)
        assert result is not None

    def test_combine_instructions_between_modules_with_empty(self, Activation):
        """Test CombineInstructionsBetweenModules with empty module"""
        CombineInstructionsBetweenModules = Activation.CombineInstructionsBetweenModules

        module = Module("test")
        moduleAndIndex = {}

        # Should handle empty module
        CombineInstructionsBetweenModules(module, moduleAndIndex, False)

        # Module should still exist
        assert module is not None

    def test_combine_instructions_between_modules_with_nested(self, Activation):
        """Test CombineInstructionsBetweenModules with nested modules"""
        CombineInstructionsBetweenModules = Activation.CombineInstructionsBetweenModules

        outer = Module("outer")
        inner = Module("inner")
        outer.appendModule(inner)

        moduleAndIndex = {}

        # Should handle nested modules
        CombineInstructionsBetweenModules(outer, moduleAndIndex, False)

        # Module should still exist
        assert outer is not None


@pytest.mark.unit
class TestActivationMagicNumbersUsage:
    """Tests for using activation magic numbers"""

    def test_float_union_with_gelu_k0(self, Activation):
        """Test floatUnion with GELU K0 magic number"""
        floatUnion = Activation.floatUnion
        ActivationMagicNumbers = Activation.ActivationMagicNumbers

        f = floatUnion()
        f.u = ActivationMagicNumbers['FloatGeluK0']

        # Should be a valid float
        assert isinstance(f.f, float)
        assert f.f != 0.0

    def test_float_union_with_gelu_k1(self, Activation):
        """Test floatUnion with GELU K1 magic number"""
        floatUnion = Activation.floatUnion
        ActivationMagicNumbers = Activation.ActivationMagicNumbers

        f = floatUnion()
        f.u = ActivationMagicNumbers['FloatGeluK1']

        # Should be a valid float
        assert isinstance(f.f, float)

    def test_all_magic_numbers_are_valid(self, Activation):
        """Test that all magic numbers are valid integers"""
        ActivationMagicNumbers = Activation.ActivationMagicNumbers

        for key, value in ActivationMagicNumbers.items():
            assert isinstance(value, int)
            assert value > 0


@pytest.mark.unit
class TestActivationLookupVeri:
    """Tests for ActivationType.lookupVeri dictionary"""

    def test_lookup_veri_contains_exp(self, Activation):
        """Test that lookupVeri contains 'exp'"""
        ActivationType = Activation.ActivationType

        assert 'exp' in ActivationType.lookupVeri

    def test_lookup_veri_exp_structure(self, Activation):
        """Test lookupVeri['exp'] structure"""
        ActivationType = Activation.ActivationType

        exp = ActivationType.lookupVeri['exp']
        assert hasattr(exp, 'name')
        assert exp.name == 'exp'
        assert hasattr(exp, 'isGradient')


@pytest.mark.unit
class TestActivationTypeInstantiation:
    """Tests for ActivationType instantiation with lookupVeri"""

    def test_activation_type_exp(self, Activation):
        """Test ActivationType with 'exp' from lookupVeri"""
        ActivationType = Activation.ActivationType

        act = ActivationType('exp')
        assert act.value == 'exp'

    def test_activation_type_exp_case_insensitive(self, Activation):
        """Test ActivationType 'exp' is case insensitive"""
        ActivationType = Activation.ActivationType

        act = ActivationType('EXP')
        assert act.value == 'exp'
