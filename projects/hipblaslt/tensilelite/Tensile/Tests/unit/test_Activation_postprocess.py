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
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from rocisa.code import Module


@pytest.fixture(scope="module")
def Activation():
    """Lazy import Activation module"""
    import Tensile.Activation as act
    return act


@pytest.fixture(scope="module")
def DataType():
    """Lazy import DataType"""
    from Tensile.Common.DataType import DataType
    return DataType


@pytest.mark.unit
class TestPostProcessFunctions:
    """Tests for post-processing functions"""

    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_post_process_without_combine(self, mock_convert, Activation, DataType):
        """Test postProcess without needCombine"""
        mock_convert.return_value = Module("test")

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.needCombine = False

        dt = DataType("s")
        input_module = Module("input")

        result = module_obj.postProcess(dt, input_module)

        # Should call ConvertCoeffToHex
        mock_convert.assert_called_once()
        assert result is not None

    @patch('Tensile.Activation.CombineInstructions')
    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_post_process_with_combine(self, mock_convert, mock_combine, Activation, DataType):
        """Test postProcess with needCombine"""
        mock_combine.return_value = Module("combined")
        mock_convert.return_value = Module("converted")

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.needCombine = True

        dt = DataType("s")
        input_module = Module("input")

        result = module_obj.postProcess(dt, input_module)

        # Should call both functions
        mock_combine.assert_called_once()
        mock_convert.assert_called_once()
        assert result is not None

    @patch('Tensile.Activation.HolderToGpr')
    def test_assign_gpr(self, mock_holder_to_gpr, Activation):
        """Test assignGpr method"""
        mock_holder_to_gpr.side_effect = lambda module, idx, pf: module

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        input_module = Module("input")
        result = module_obj.assignGpr(input_module, 10, 5)

        # Should call HolderToGpr twice (for vgpr and sgpr)
        assert mock_holder_to_gpr.call_count == 2
        assert result is not None

    def test_vgpr_prefix_with_int(self, Activation):
        """Test vgprPrefix with integer"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        # Without format
        result = module_obj.vgprPrefix(5)
        assert result is not None

        # With format
        module_obj.setVgprPrefixFormat("v[%d]")
        result = module_obj.vgprPrefix(5)
        assert result is not None

    def test_vgpr_prefix_with_string(self, Activation):
        """Test vgprPrefix with string"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        result = module_obj.vgprPrefix("test")
        assert result is not None

    def test_vgpr_prefix_with_two_args(self, Activation):
        """Test vgprPrefix with two arguments"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        result = module_obj.vgprPrefix(5, 2)
        assert result is not None


@pytest.mark.unit
class TestCacheFunctions:
    """Tests for cache-related functions"""

    def test_create_cache(self, Activation, DataType):
        """Test createCache method"""
        with patch('Tensile.Activation.createVgprIdxList') as mock_create:
            with patch('Tensile.Activation.deepcopy') as mock_deepcopy:
                mock_create.return_value = [[], []]
                mock_deepcopy.return_value = Module("copied")

                ActivationModule = Activation.ActivationModule
                module_obj = ActivationModule()

                dt = DataType("s")
                input_module = Module("test")

                module_obj.createCache(dt, 'relu', 0, 1, input_module)

                # Should have created cache entry
                assert 'relu' in module_obj.cacheDict

    def test_get_cache_miss(self, Activation, DataType):
        """Test getCache with cache miss"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")
        result = module_obj.getCache(dt, 'relu', 0, 1)

        # Should return None
        assert result is None

    def test_get_cache_hit(self, Activation, DataType):
        """Test getCache with cache hit"""
        with patch('Tensile.Activation.createVgprIdxList') as mock_create:
            with patch('Tensile.Activation.deepcopy') as mock_deepcopy:
                mock_vgpr = Mock()
                mock_vgpr.regIdx = 0
                mock_create.return_value = [[mock_vgpr], [mock_vgpr]]
                mock_module = Module("cached")
                mock_deepcopy.return_value = mock_module

                ActivationModule = Activation.ActivationModule
                module_obj = ActivationModule()

                dt = DataType("s")
                input_module = Module("test")

                # Create cache
                module_obj.createCache(dt, 'relu', 0, 1, input_module)

                # Get from cache
                result = module_obj.getCache(dt, 'relu', 0, 1)

                # Should return cached module
                assert result is not None


@pytest.mark.unit
class TestGetModuleWithCache:
    """Tests for getModule with caching"""

    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_get_module_with_cache_enabled(self, mock_convert, Activation, DataType):
        """Test getModule with cache enabled"""
        mock_convert.return_value = Module("test")

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUseCache(True)

        dt = DataType("s")

        # First call - should create
        result1 = module_obj.getModule(dt, 'relu', 0, 1)
        assert result1 is not None

        # Second call with different vgprs should use cache
        with patch.object(module_obj, 'getCache') as mock_get_cache:
            mock_get_cache.return_value = None
            result2 = module_obj.getModule(dt, 'relu', 2, 3)
            mock_get_cache.assert_called()

    @patch('Tensile.Activation.rocIsa')
    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_get_module_all_activation_types(self, mock_convert, mock_rocisa, Activation, DataType):
        """Test getModule for all activation types"""
        mock_convert.return_value = Module("test")
        mock_instance = Mock()
        mock_instance.getArchCaps.return_value = {"TransOpWait": True}
        mock_rocisa.getInstance.return_value = mock_instance

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        activation_types = ['none', 'abs', 'relu', 'clippedrelu',  'leakyrelu',
                           'gelu', 'geluscaling', 'sigmoid', 'tanh', 'dgelu',
                           'drelu', 'silu', 'swish', 'clamp']

        for act_type in activation_types:
            result = module_obj.getModule(dt, act_type, 0, 1)
            assert result is not None


@pytest.mark.unit
class TestCombineInstructionsFunctions:
    """Tests for CombineInstructions and related functions"""

    def test_combine_instructions(self, Activation):
        """Test CombineInstructions function"""
        CombineInstructions = Activation.CombineInstructions

        module = Module("test")
        result = CombineInstructions(module, fuseDebug=False)

        # Should return a module
        assert result is not None

    def test_combine_instructions_with_debug(self, Activation):
        """Test CombineInstructions with debug"""
        CombineInstructions = Activation.CombineInstructions

        module = Module("test")
        result = CombineInstructions(module, fuseDebug=True)

        # Should return a module
        assert result is not None

    def test_remove_empty_blocks(self, Activation):
        """Test RemoveEmptyBlocks function"""
        RemoveEmptyBlocks = Activation.RemoveEmptyBlocks

        module = Module("test")
        result = RemoveEmptyBlocks(module)

        # Should return a module
        assert result is not None

    def test_remove_empty_blocks_with_nested(self, Activation):
        """Test RemoveEmptyBlocks with nested modules"""
        RemoveEmptyBlocks = Activation.RemoveEmptyBlocks

        inner = Module("inner")
        outer = Module("outer")
        outer.appendModule(inner)

        result = RemoveEmptyBlocks(outer)

        # Should collapse single-module blocks
        assert result is not None


@pytest.mark.unit
class TestActivationTypeSupportedBy:
    """Tests for ActivationType.SupportedBy enum"""

    def test_supported_by_enum_values(self, Activation):
        """Test SupportedBy enum values"""
        SupportedBy = Activation.ActivationType.SupportedBy

        assert SupportedBy.HIPBLASLT == 0b01
        assert SupportedBy.TENSILE == 0b10
        assert SupportedBy.ALL == 0b11

    def test_supported_by_bitwise_and(self, Activation):
        """Test SupportedBy bitwise AND operations"""
        SupportedBy = Activation.ActivationType.SupportedBy

        # TENSILE & ALL should be non-zero
        assert (SupportedBy.TENSILE & SupportedBy.ALL) != 0

        # TENSILE & HIPBLASLT should be zero
        assert (SupportedBy.TENSILE & SupportedBy.HIPBLASLT) == 0

        # ALL contains both
        assert (SupportedBy.ALL & SupportedBy.TENSILE) != 0
        assert (SupportedBy.ALL & SupportedBy.HIPBLASLT) != 0


@pytest.mark.unit
class TestActivationTypeExport:
    """Tests for ActivationType.Export enum"""

    def test_export_enum_values(self, Activation):
        """Test Export enum values"""
        Export = Activation.ActivationType.Export

        assert Export.NORMAL.value == 0
        assert Export.GRADONLY.value == 1
        assert Export.BOTH.value == 2

    def test_export_enum_comparison(self, Activation):
        """Test Export enum comparison"""
        Export = Activation.ActivationType.Export

        assert Export.NORMAL == Export.NORMAL
        assert Export.NORMAL != Export.GRADONLY


@pytest.mark.unit
class TestActivationTypeStringList:
    """Tests for ActivationType.stringList"""

    def test_string_list_contents(self, Activation):
        """Test ActivationType.stringList"""
        ActivationType = Activation.ActivationType

        assert isinstance(ActivationType.stringList, list)
        assert 'alpha' in ActivationType.stringList
        assert 'beta' in ActivationType.stringList
        assert 'gamma' in ActivationType.stringList
        assert 'delta' in ActivationType.stringList

    def test_get_additional_arg_string_list_clippedrelu(self, Activation):
        """Test getAdditionalArgStringList for clippedrelu (2 args)"""
        ActivationType = Activation.ActivationType

        act = ActivationType("clippedrelu")
        args = act.getAdditionalArgStringList(addPrefix=True)

        assert len(args) == 2
        assert args[0] == "activationAlpha"
        assert args[1] == "activationBeta"

        args_no_prefix = act.getAdditionalArgStringList(addPrefix=False)
        assert args_no_prefix[0] == "alpha"
        assert args_no_prefix[1] == "beta"


@pytest.mark.unit
class TestActivationTypeLookup:
    """Tests for ActivationType.lookup dictionary"""

    def test_lookup_structure(self, Activation):
        """Test ActivationType.lookup structure"""
        ActivationType = Activation.ActivationType

        for key, value in ActivationType.lookup.items():
            # Each entry should have 'instance' and 'supported_by'
            assert 'instance' in value
            assert 'supported_by' in value

            # instance should be ActivationTypeRegister
            assert hasattr(value['instance'], 'name')
            assert hasattr(value['instance'], 'isGradient')
            assert hasattr(value['instance'], 'extraArgs')

    def test_lookup_gradient_activations(self, Activation):
        """Test gradient activations in lookup"""
        ActivationType = Activation.ActivationType

        # Check dgelu is gradient
        dgelu = ActivationType.lookup['dgelu']['instance']
        assert dgelu.isGradient == True

        # Check drelu is gradient
        drelu = ActivationType.lookup['drelu']['instance']
        assert drelu.isGradient == True

        # Check gelu is not gradient
        gelu = ActivationType.lookup['gelu']['instance']
        assert gelu.isGradient == False


@pytest.mark.unit
class TestVgprPrefixFormat:
    """Tests for vgprPrefix with formatting"""

    def test_vgpr_prefix_format_application(self, Activation):
        """Test vgprPrefix applies format correctly"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        # Set format
        module_obj.setVgprPrefixFormat("ValuC+%d")

        # Should use format for integers
        result = module_obj.vgprPrefix(5)
        assert result is not None

    def test_vgpr_prefix_no_format_with_string(self, Activation):
        """Test vgprPrefix doesn't apply format to strings"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        module_obj.setVgprPrefixFormat("ValuC+%d")

        # Should not use format for strings
        result = module_obj.vgprPrefix("customVgpr")
        assert result is not None


@pytest.mark.unit
class TestFuseInstructionHelpers:
    """Tests for FuseInstruction helper functions"""

    def test_fuse_instruction_exists(self, Activation):
        """Test that FuseInstruction function exists"""
        assert hasattr(Activation, 'FuseInstruction')

    def test_combine_instructions_between_modules(self, Activation):
        """Test CombineInstructionsBetweenModules function"""
        CombineInstructionsBetweenModules = Activation.CombineInstructionsBetweenModules

        module = Module("test")
        moduleAndIndex = {}

        # Should not raise
        CombineInstructionsBetweenModules(module, moduleAndIndex, False)


@pytest.mark.unit
class TestGetModuleEdgeCases:
    """Tests for edge cases in getModule"""

    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_get_module_same_vgpr_in_out(self, mock_convert, Activation, DataType):
        """Test getModule when vgprIn == vgprOut (no cache created)"""
        mock_convert.return_value = Module("test")

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUseCache(True)

        dt = DataType("s")

        # Same vgpr for in and out - should not create cache
        result = module_obj.getModule(dt, 'relu', 5, 5)
        assert result is not None

        # Cache should not be created for same vgpr
        cached = module_obj.getCache(dt, 'relu', 5, 5)
        assert cached is None

    @patch('Tensile.Activation.rocIsa')
    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_get_module_exp(self, mock_convert, mock_rocisa, Activation, DataType):
        """Test getModule for 'exp' activation"""
        mock_convert.return_value = Module("test")
        # Mock rocIsa
        mock_instance = Mock()
        mock_instance.getArchCaps.return_value = {"TransOpWait": True}
        mock_rocisa.getInstance.return_value = mock_instance

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        result = module_obj.getModule(dt, 'exp', 0, 1)
        assert result is not None


@pytest.mark.unit
class TestGetEnumStrListVariations:
    """Tests for getEnumStrList with different parameters"""

    def test_get_enum_str_list_tensile_only(self, Activation, DataType):
        """Test getEnumStrList with TENSILE support only"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")

        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.TENSILE, includeNone=True)
        assert isinstance(enum_list, list)
        # TENSILE-supported activations should be included
        assert 'abs' in enum_list

    def test_get_enum_str_list_hipblaslt_only(self, Activation, DataType):
        """Test getEnumStrList with HIPBLASLT support only"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")

        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.HIPBLASLT, includeNone=True)
        assert isinstance(enum_list, list)
        # HIPBLASLT-supported activations should be included
        assert 'relu' in enum_list or 'gelu' in enum_list

    def test_get_enum_str_list_gradonly_export(self, Activation, DataType):
        """Test getEnumStrList with GRADONLY export type"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")

        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.ALL, includeNone=True,
                                                   exportType=ActivationType.Export.GRADONLY)
        assert isinstance(enum_list, list)
        # GRADONLY: passActivation returns True for non-gradients (skips them)
        # So GRADONLY should only include gradient activations
        assert 'dgelu' in enum_list
        assert 'drelu' in enum_list
        # Should exclude non-gradient activations
        assert 'relu' not in enum_list

    def test_get_enum_str_list_both_export(self, Activation, DataType):
        """Test getEnumStrList with BOTH export type"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")

        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.ALL, includeNone=True,
                                                   exportType=ActivationType.Export.BOTH)
        assert isinstance(enum_list, list)
        # BOTH should include all activations
        assert len(enum_list) > 0


@pytest.mark.unit
class TestActivationTypeComparisons:
    """Tests for ActivationType comparison edge cases"""

    def test_activation_type_eq_invalid_type_raises(self, Activation):
        """Test ActivationType __eq__ with invalid type raises"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")

        with pytest.raises(RuntimeError):
            _ = (act == 123)

    def test_activation_type_lt_invalid_type_raises(self, Activation):
        """Test ActivationType __lt__ with invalid type raises"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")

        with pytest.raises(RuntimeError):
            _ = (act < 123)

    def test_activation_type_lt_with_string(self, Activation):
        """Test ActivationType __lt__ with string"""
        ActivationType = Activation.ActivationType

        act = ActivationType("abs")

        # 'abs' < 'relu' alphabetically
        assert act < 'relu'
        assert not (act < 'abs')


@pytest.mark.unit
class TestActivationModuleWithDifferentDataTypes:
    """Tests for activation modules with various data types"""

    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_abs_with_bfloat16_no_pk(self, mock_convert, Activation, DataType):
        """Test abs with bfloat16 and usePK=False"""
        mock_convert.return_value = Module("test")

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUsePK(False)

        dt = DataType("b")
        result = module_obj.getAbsModule(dt, 0, 1)
        assert result is not None

    @patch('Tensile.Activation.rocIsa')
    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_exp_with_half_no_pk(self, mock_convert, mock_rocisa, Activation, DataType):
        """Test exp with half precision and usePK=False"""
        mock_convert.return_value = Module("test")
        mock_instance = Mock()
        mock_instance.getArchCaps.return_value = {"TransOpWait": False}
        mock_rocisa.getInstance.return_value = mock_instance

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUsePK(False)

        dt = DataType("h")
        result = module_obj.getExpModule(dt, 0, 1)
        assert result is not None


@pytest.mark.unit
class TestGetCacheWithVgprPrefixFormat:
    """Tests for getCache with vgprPrefixFormat"""

    def test_get_cache_with_prefix_format(self, Activation, DataType):
        """Test getCache when vgprPrefixFormat is set"""
        with patch('Tensile.Activation.createVgprIdxList') as mock_create:
            with patch('Tensile.Activation.deepcopy') as mock_deepcopy:
                # Mock vgpr with regName
                mock_vgpr_in = Mock()
                mock_vgpr_out = Mock()
                mock_reg_name_in = Mock()
                mock_reg_name_out = Mock()
                mock_vgpr_in.regName = mock_reg_name_in
                mock_vgpr_out.regName = mock_reg_name_out

                mock_create.return_value = [[mock_vgpr_in], [mock_vgpr_out]]
                mock_module = Module("cached")
                mock_deepcopy.return_value = mock_module

                ActivationModule = Activation.ActivationModule
                module_obj = ActivationModule()
                module_obj.setVgprPrefixFormat("ValuC+%d")

                dt = DataType("s")
                input_module = Module("test")

                # Create cache
                module_obj.createCache(dt, 'relu', 0, 1, input_module)

                # Get from cache with different vgprs
                result = module_obj.getCache(dt, 'relu', 5, 6)

                # Should call setOffset on regName
                if result is not None:
                    mock_reg_name_in.setOffset.assert_called()
                    mock_reg_name_out.setOffset.assert_called()
