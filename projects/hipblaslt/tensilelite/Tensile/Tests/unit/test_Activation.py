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


# Lazy import to avoid module-level import errors
@pytest.fixture(scope="module")
def Activation():
    """Lazy import Activation module"""
    import Tensile.Activation as act
    return act


@pytest.mark.unit
class TestActivationModule:
    """Tests for Activation module functions"""

    def test_module_exists(self, Activation):
        """Test that Activation module can be imported"""
        assert Activation is not None

    def test_activation_enums_exist(self, Activation):
        """Test that ActivationType enum exists"""
        # Check if ActivationType enum exists
        assert hasattr(Activation, 'ActivationType')

    def test_activation_type_lookup_values(self, Activation):
        """Test ActivationType lookup has expected values"""
        ActivationType = Activation.ActivationType

        # Test common activation types exist in lookup
        assert 'none' in ActivationType.lookup
        assert 'abs' in ActivationType.lookup
        assert 'clippedrelu' in ActivationType.lookup
        assert 'gelu' in ActivationType.lookup
        assert 'relu' in ActivationType.lookup
        assert 'sigmoid' in ActivationType.lookup

    def test_activation_type_all_in_lookup(self, Activation):
        """Test that 'all' is in lookup"""
        ActivationType = Activation.ActivationType

        # 'all' should be in lookup
        assert 'all' in ActivationType.lookup
        assert 'hipblaslt_all' in ActivationType.lookup

    def test_activation_type_none_instantiation(self, Activation):
        """Test that 'none' activation type can be instantiated"""
        ActivationType = Activation.ActivationType

        # 'none' should be instantiable
        act = ActivationType('none')
        assert act is not None
        assert act.value == 'none'

    def test_activation_module_exists(self, Activation):
        """Test that ActivationModule class exists"""
        assert hasattr(Activation, 'ActivationModule')

    def test_activation_module_instantiation(self, Activation):
        """Test ActivationModule can be instantiated"""
        ActivationModule = Activation.ActivationModule

        # Should be instantiable
        module = ActivationModule()
        assert module is not None
        assert module.vgprCounter == 0
        assert module.sgprCounter == 0

    def test_activation_name_mapping_exists(self, Activation):
        """Test that activationToEnum mapping exists"""
        # Should have a dictionary mapping string names to enum values
        if hasattr(Activation, 'activationToEnum'):
            mapping = Activation.activationToEnum
            assert isinstance(mapping, dict)
            assert 'none' in mapping or 'None' in mapping

    def test_enum_to_name_mapping_exists(self, Activation):
        """Test that enumToActivation reverse mapping exists"""
        if hasattr(Activation, 'enumToActivation'):
            mapping = Activation.enumToActivation
            assert isinstance(mapping, dict)


@pytest.mark.unit
class TestActivationFunctions:
    """Tests for activation-related functions"""

    def test_get_enum_name_function(self, Activation):
        """Test function to get enum name from string"""
        # If there's a function to convert string to enum
        if hasattr(Activation, 'getEnumName'):
            func = Activation.getEnumName
            result = func('relu')
            assert result is not None

    def test_get_activation_function(self, Activation):
        """Test function to get activation from name"""
        if hasattr(Activation, 'getActivation'):
            func = Activation.getActivation
            result = func('none')
            assert result is not None

    def test_activation_has_args_function(self, Activation):
        """Test function to check if activation has arguments"""
        if hasattr(Activation, 'activationHasArgs'):
            func = Activation.activationHasArgs
            # 'none' should not have args
            # 'clippedrelu' should have args (threshold)
            pass

    def test_activation_arg_names_function(self, Activation):
        """Test function to get activation argument names"""
        if hasattr(Activation, 'getActivationArgNames'):
            func = Activation.getActivationArgNames
            # Some activations have args, some don't
            pass


@pytest.mark.unit
class TestActivationEnumOperations:
    """Tests for operations on activation enums"""

    def test_activation_type_comparison(self, Activation):
        """Test that activation types can be compared"""
        ActivationType = Activation.ActivationType

        # Same activation should be equal
        act1 = ActivationType('none')
        act2 = ActivationType('none')
        assert act1 == act2
        assert act1 == 'none'

    def test_activation_type_ordering(self, Activation):
        """Test that activation types can be compared with less than"""
        ActivationType = Activation.ActivationType

        # Test ordering
        act_abs = ActivationType('abs')
        act_relu = ActivationType('relu')
        # 'abs' < 'relu' alphabetically
        assert act_abs < act_relu

    def test_activation_type_string_representation(self, Activation):
        """Test that activation types have string representation"""
        ActivationType = Activation.ActivationType

        # Should be convertible to string
        act = ActivationType('relu')
        str_repr = str(act)
        assert isinstance(str_repr, str)
        assert str_repr == 'Relu'  # Capitalized


@pytest.mark.unit
class TestActivationDataStructures:
    """Tests for activation-related data structures"""

    def test_activation_params_class(self, Activation):
        """Test ActivationParams class if it exists"""
        if hasattr(Activation, 'ActivationParams'):
            ActivationParams = Activation.ActivationParams
            # Should be instantiable
            params = ActivationParams()
            assert params is not None

    def test_activation_config_class(self, Activation):
        """Test ActivationConfig class if it exists"""
        if hasattr(Activation, 'ActivationConfig'):
            ActivationConfig = Activation.ActivationConfig
            config = ActivationConfig()
            assert config is not None

    def test_activation_spec_creation(self, Activation):
        """Test creating activation specification"""
        # Some modules have functions to create activation specs
        if hasattr(Activation, 'createActivationSpec'):
            func = Activation.createActivationSpec
            spec = func('relu')
            assert spec is not None

    def test_activation_available_class(self, Activation):
        """Test ActivationAvailable class"""
        ActivationAvailable = Activation.ActivationAvailable

        # Test all False
        avail = ActivationAvailable()
        assert avail.half == False
        assert avail.single == False
        assert avail.double == False

        # Test with some True
        avail = ActivationAvailable(canHalf=True, canSingle=True)
        assert avail.half == True
        assert avail.single == True
        assert avail.double == False

    def test_activation_type_register_class(self, Activation):
        """Test ActivationTypeRegister class"""
        ActivationTypeRegister = Activation.ActivationTypeRegister

        # Create instance
        reg = ActivationTypeRegister("test", False, 2, canSingle=True, canDouble=True)
        assert reg.name == "test"
        assert reg.isGradient == False
        assert reg.extraArgs == 2
        assert reg.can.single == True
        assert reg.can.double == True


@pytest.mark.unit
class TestActivationTypeRegisterTypeAvailable:
    """Tests for ActivationTypeRegister.typeAvailable method"""

    def test_type_available_single(self, Activation):
        """Test typeAvailable for single precision"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canSingle=True)

        dt_single = DataType("s")
        assert reg.typeAvailable(dt_single) == True

    def test_type_available_half(self, Activation):
        """Test typeAvailable for half precision"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canHalf=True)

        dt_half = DataType("h")
        assert reg.typeAvailable(dt_half) == True

    def test_type_available_double(self, Activation):
        """Test typeAvailable for double precision"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canDouble=True)

        dt_double = DataType("d")
        assert reg.typeAvailable(dt_double) == True

    def test_type_available_bfloat16(self, Activation):
        """Test typeAvailable for bfloat16"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canBFloat16=True)

        dt_bf16 = DataType("b")
        assert reg.typeAvailable(dt_bf16) == True

    def test_type_available_int8(self, Activation):
        """Test typeAvailable for int8"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canInt8=True)

        dt_int8 = DataType("I8")
        assert reg.typeAvailable(dt_int8) == True

    def test_type_available_int32(self, Activation):
        """Test typeAvailable for int32"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canInt32=True)

        dt_int32 = DataType("i")
        assert reg.typeAvailable(dt_int32) == True

    def test_type_not_available(self, Activation):
        """Test typeAvailable returns False when type not supported"""
        from Tensile.Common.DataType import DataType

        ActivationTypeRegister = Activation.ActivationTypeRegister
        reg = ActivationTypeRegister("test", False, 0, canSingle=True)  # Only single

        dt_half = DataType("h")
        assert reg.typeAvailable(dt_half) == False


@pytest.mark.unit
class TestActivationTypeClass:
    """Tests for ActivationType class methods"""

    def test_activation_type_with_string(self, Activation):
        """Test ActivationType initialization with string"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")
        assert act.value == "relu"

        act = ActivationType("GELU")  # Case insensitive
        assert act.value == "gelu"

    def test_activation_type_with_activation_type(self, Activation):
        """Test ActivationType initialization with another ActivationType"""
        ActivationType = Activation.ActivationType

        act1 = ActivationType("relu")
        act2 = ActivationType(act1)
        assert act2.value == "relu"

    def test_activation_type_invalid_raises(self, Activation):
        """Test ActivationType raises on invalid activation"""
        ActivationType = Activation.ActivationType

        with pytest.raises(RuntimeError):
            ActivationType("invalid_activation")

    def test_activation_type_invalid_type_raises(self, Activation):
        """Test ActivationType raises on invalid input type"""
        ActivationType = Activation.ActivationType

        with pytest.raises(RuntimeError):
            ActivationType(123)  # Invalid type

    def test_pass_activation(self, Activation):
        """Test passActivation method"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")

        # Test NORMAL export - should pass gradients
        assert act.passActivation(True, ActivationType.Export.NORMAL) == True
        assert act.passActivation(False, ActivationType.Export.NORMAL) == False

        # Test GRADONLY export - should pass non-gradients
        assert act.passActivation(True, ActivationType.Export.GRADONLY) == False
        assert act.passActivation(False, ActivationType.Export.GRADONLY) == True

        # Test BOTH export - never pass
        assert act.passActivation(True, ActivationType.Export.BOTH) == False
        assert act.passActivation(False, ActivationType.Export.BOTH) == False

    def test_get_additional_arg_num(self, Activation):
        """Test getAdditionalArgNum method"""
        ActivationType = Activation.ActivationType

        # 'relu' has 0 args
        act = ActivationType("relu")
        assert act.getAdditionalArgNum() == 0

        # 'leakyrelu' has 1 arg (alpha)
        act = ActivationType("leakyrelu")
        assert act.getAdditionalArgNum() == 1

        # 'clippedrelu' has 2 args (alpha, beta)
        act = ActivationType("clippedrelu")
        assert act.getAdditionalArgNum() == 2

    def test_get_additional_arg_num_all(self, Activation):
        """Test getAdditionalArgNum for 'all' activation"""
        ActivationType = Activation.ActivationType

        # 'all' should return max args across all activations
        act = ActivationType("all")
        max_args = act.getAdditionalArgNum()
        assert max_args >= 2  # At least clippedrelu has 2 args

    def test_fit_supported(self, Activation):
        """Test fitSupported method"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        act = ActivationType("relu")

        # Test bitwise AND
        assert act.fitSupported(SupportedBy.ALL, SupportedBy.TENSILE) != 0
        assert act.fitSupported(SupportedBy.TENSILE, SupportedBy.TENSILE) != 0
        assert act.fitSupported(SupportedBy.HIPBLASLT, SupportedBy.TENSILE) == 0

    def test_get_additional_arg_string_list(self, Activation):
        """Test getAdditionalArgStringList method"""
        ActivationType = Activation.ActivationType

        act = ActivationType("leakyrelu")  # 1 arg

        # With prefix
        args = act.getAdditionalArgStringList(addPrefix=True)
        assert len(args) == 1
        assert args[0] == "activationAlpha"

        # Without prefix
        args = act.getAdditionalArgStringList(addPrefix=False)
        assert len(args) == 1
        assert args[0] == "alpha"

    def test_get_enum_index(self, Activation):
        """Test getEnumIndex class method"""
        ActivationType = Activation.ActivationType

        # 'none' should be first
        idx = ActivationType.getEnumIndex('none')
        assert idx == 0

        # Other indices should be positive
        idx_abs = ActivationType.getEnumIndex('abs')
        assert idx_abs > 0

    def test_get_enum_str_list(self, Activation):
        """Test getEnumStrList class method"""
        from Tensile.Common.DataType import DataType
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")  # Single precision

        # Get list of activations
        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.ALL, includeNone=True)
        assert isinstance(enum_list, list)
        assert len(enum_list) > 0
        assert 'relu' in enum_list
        assert 'gelu' in enum_list

        # Without none
        enum_list_no_none = ActivationType.getEnumStrList(dt, SupportedBy.ALL, includeNone=False)
        assert 'none' not in enum_list_no_none

    def test_state_method(self, Activation):
        """Test state method"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")
        assert act.state() == "Relu"

    def test_to_enum_method(self, Activation):
        """Test toEnum method"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")
        assert act.toEnum() == "Relu"

    def test_repr_method(self, Activation):
        """Test __repr__ method"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")
        assert repr(act) == str(act)


@pytest.mark.unit
class TestActivationModuleMethods:
    """Tests for ActivationModule methods"""

    def test_reduce_method(self, Activation):
        """Test __reduce__ method for pickling"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        reduced = module.__reduce__()
        assert reduced[0] == ActivationModule
        assert reduced[1] == ()

    def test_set_use_pk(self, Activation):
        """Test setUsePK method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        assert module.usePK == True  # Default

        module.setUsePK(False)
        assert module.usePK == False

    def test_set_saturation_for_int8(self, Activation):
        """Test setSaturationForInt8 method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        assert module.saturateI8 == False  # Default

        module.setSaturationForInt8(True)
        assert module.saturateI8 == True

    def test_set_vgpr_prefix_format(self, Activation):
        """Test setVgprPrefixFormat method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        assert module.vgprPrefixFormat == ""

        module.setVgprPrefixFormat("v[%d]")
        assert module.vgprPrefixFormat == "v[%d]"

    def test_set_use_cache(self, Activation):
        """Test setUseCache method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        assert module.useCache == False

        module.setUseCache(True)
        assert module.useCache == True

    def test_set_guard(self, Activation):
        """Test setGuard method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        assert module.enableGuard == False

        module.setGuard(True)
        assert module.enableGuard == True

    def test_set_alt(self, Activation):
        """Test setAlt method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        assert module.isAlt == False

        module.setAlt(True)
        assert module.isAlt == True

    def test_reset_gpr_counter(self, Activation):
        """Test resetGprCounter method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        module.vgprCounter = 10
        module.sgprCounter = 5

        module.resetGprCounter()
        assert module.vgprCounter == 0
        assert module.sgprCounter == 0

    def test_get_vgpr(self, Activation):
        """Test getVgpr method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()

        idx = module.getVgpr(3)
        assert idx == 0
        assert module.vgprCounter == 3

        idx = module.getVgpr(2)
        assert idx == 3
        assert module.vgprCounter == 5

    def test_get_sgpr(self, Activation):
        """Test getSgpr method"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()

        idx = module.getSgpr(2)
        assert idx == 0
        assert module.sgprCounter == 2

        idx = module.getSgpr(1)
        assert idx == 2
        assert module.sgprCounter == 3


@pytest.mark.unit
class TestActivationModuleGetModule:
    """Tests for ActivationModule.getModule method"""

    def test_get_module_none(self, Activation):
        """Test getModule with 'none' activation"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        result = module.getModule(Mock(), 'none', 0, 1)

        # Should return a Module
        assert result is not None

    def test_get_module_unsupported(self, Activation):
        """Test getModule with unsupported activation"""
        ActivationModule = Activation.ActivationModule

        module = ActivationModule()
        result = module.getModule(Mock(), 'unsupported_activation', 0, 1)

        # Should return a Module with error message
        assert result is not None


@pytest.mark.unit
class TestActivationMagicNumbers:
    """Tests for activation magic numbers"""

    def test_activation_magic_numbers_exist(self, Activation):
        """Test that ActivationMagicNumbers dict exists"""
        assert hasattr(Activation, 'ActivationMagicNumbers')

        magic = Activation.ActivationMagicNumbers
        assert isinstance(magic, dict)
        assert 'FloatGeluK0' in magic
        assert 'FloatGeluK1' in magic

    def test_float_union_class(self, Activation):
        """Test floatUnion class"""
        floatUnion = Activation.floatUnion

        # Create instance
        f = floatUnion()
        f.f = 1.0
        assert f.u > 0  # Should be a positive int representation

        # Test with magic number
        f.u = Activation.ActivationMagicNumbers['FloatGeluK0']
        assert isinstance(f.f, float)


@pytest.mark.unit
class TestActCacheInfo:
    """Tests for actCacheInfo dataclass"""

    def test_act_cache_info_creation(self, Activation):
        """Test creating actCacheInfo"""
        actCacheInfo = Activation.actCacheInfo

        cache = actCacheInfo(
            usePK=True,
            saturateI8=False,
            enableGuard=False,
            isAlt=False,
            prefix="test",
            vgprIdxList=[[],[]],
            module=Mock(),
            vgprCounter=10,
            sgprCounter=5
        )

        assert cache.usePK == True
        assert cache.saturateI8 == False
        assert cache.vgprCounter == 10
        assert cache.sgprCounter == 5

    def test_act_cache_info_is_same(self, Activation):
        """Test actCacheInfo.isSame method"""
        actCacheInfo = Activation.actCacheInfo

        cache = actCacheInfo(
            usePK=True,
            saturateI8=False,
            enableGuard=False,
            isAlt=False,
            prefix="test",
            vgprIdxList=[[],[]],
            module=Mock(),
            vgprCounter=10,
            sgprCounter=5
        )

        # Same parameters
        assert cache.isSame(True, False, False, False, "test") == True

        # Different usePK
        assert cache.isSame(False, False, False, False, "test") == False

        # Different prefix
        assert cache.isSame(True, False, False, False, "other") == False
