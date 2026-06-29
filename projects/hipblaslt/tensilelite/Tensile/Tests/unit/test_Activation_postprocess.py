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
from rocisa.container import vgpr

import Tensile.Activation as Activation
from Tensile.Common.DataType import DataType


@pytest.mark.unit
class TestPostProcessFunctions:
    """Tests for post-processing functions"""

    @patch('Tensile.Activation.CombineInstructions')
    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_post_process_without_combine(self, mock_convert, mock_combine):
        """Test postProcess skips CombineInstructions when needCombine is False"""
        converted_module = Module("converted")
        mock_convert.return_value = converted_module

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.needCombine = False

        dt = DataType("s")
        input_module = Module("input")

        result = module_obj.postProcess(dt, input_module)

        # Should call ConvertCoeffToHex but NOT CombineInstructions
        # usePK defaults to True in ActivationModule.__init__
        mock_convert.assert_called_once_with(input_module, dt, True)
        mock_combine.assert_not_called()
        assert result == converted_module

    @patch('Tensile.Activation.CombineInstructions')
    @patch('Tensile.Activation.ConvertCoeffToHex')
    def test_post_process_with_combine(self, mock_convert, mock_combine):
        """Test postProcess calls CombineInstructions then ConvertCoeffToHex"""
        converted_module = Module("converted")
        # CombineInstructions modifies in place, no return value
        mock_combine.return_value = None
        mock_convert.return_value = converted_module

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.needCombine = True

        dt = DataType("s")
        input_module = Module("input")

        result = module_obj.postProcess(dt, input_module)

        # Should call CombineInstructions first, then ConvertCoeffToHex with same module
        # usePK defaults to True in ActivationModule.__init__
        mock_combine.assert_called_once_with(input_module)
        mock_convert.assert_called_once_with(input_module, dt, True)
        assert result == converted_module

    @patch('Tensile.Activation.HolderToGpr')
    def test_assign_gpr(self, mock_holder_to_gpr):
        """Test assignGpr calls HolderToGpr for vgpr and sgpr"""
        final_module = Module("final")
        mock_holder_to_gpr.return_value = final_module

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        input_module = Module("input")
        result = module_obj.assignGpr(input_module, 10, 5)

        # Should call HolderToGpr twice with correct indices and prefixes
        assert mock_holder_to_gpr.call_count == 2
        # First call for vgpr
        assert mock_holder_to_gpr.call_args_list[0][0][0] == input_module  # module
        assert mock_holder_to_gpr.call_args_list[0][0][1] == 10  # vgpr index
        assert mock_holder_to_gpr.call_args_list[0][0][2] == "v"  # vgpr prefix
        # Second call for sgpr (gets result from first call)
        assert mock_holder_to_gpr.call_args_list[1][0][0] == final_module  # module from first call
        assert mock_holder_to_gpr.call_args_list[1][0][1] == 5   # sgpr index
        assert mock_holder_to_gpr.call_args_list[1][0][2] == "s"  # sgpr prefix
        assert result == final_module

    def test_vgpr_prefix_returns_formatted_string(self):
        """Test vgprPrefix returns correctly formatted register names"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        # With format string
        module_obj.setVgprPrefixFormat("ValuC+%d")
        result = module_obj.vgprPrefix(5)
        # Should apply format to integer
        assert result == vgpr("ValuC+5")

        # With string input, wraps in vgpr()
        result_str = module_obj.vgprPrefix("myVgpr")
        assert result_str == vgpr("myVgpr")

        # With two args (range)
        result_range = module_obj.vgprPrefix(5, 2)
        # Should handle range notation with format applied
        assert result_range == vgpr("ValuC+5", 2)


@pytest.mark.unit
class TestCacheFunctions:
    """Tests for cache-related functions"""

    def test_create_cache_populates_cache_dict(self):
        """Test createCache populates cacheDict with correct structure"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")  # single precision
        input_module = Module("test")

        # Cache should be empty initially
        assert module_obj.cacheDict == {}

        # Create cache entry
        module_obj.createCache(dt, 'relu', 0, 1, input_module)

        # Verify cache structure
        assert 'relu' in module_obj.cacheDict, "Cache should have 'relu' activation type"
        # Cache uses uppercase datatype characters
        assert 'S' in module_obj.cacheDict['relu'], "Cache should have 'S' data type for relu"
        assert isinstance(module_obj.cacheDict['relu']['S'], list), "Cache entries should be a list"
        assert len(module_obj.cacheDict['relu']['S']) == 1, "Should have exactly one cache entry"

        # Verify the cached info has the expected structure
        cache_info = module_obj.cacheDict['relu']['S'][0]
        assert hasattr(cache_info, 'module'), "Cache info should have module attribute"
        assert hasattr(cache_info, 'vgprIdxList'), "Cache info should have vgprIdxList"

    def test_create_cache_different_datatypes(self):
        """Test createCache stores different datatypes separately"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        input_module = Module("test")

        # Create cache entries for different data types
        module_obj.createCache(DataType("s"), 'relu', 0, 1, input_module)  # float
        module_obj.createCache(DataType("d"), 'relu', 0, 1, input_module)  # double

        # Both should be in cache under different type chars (uppercase)
        assert 'S' in module_obj.cacheDict['relu'], "Should cache single precision"
        assert 'D' in module_obj.cacheDict['relu'], "Should cache double precision"
        assert len(module_obj.cacheDict['relu']) == 2, "Should have 2 different data types cached"

    def test_get_cache_miss_empty_cache(self):
        """Test getCache returns None when cache is empty"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")
        result = module_obj.getCache(dt, 'relu', 0, 1)

        assert result is None, "Should return None for cache miss with empty cache"

    def test_get_cache_miss_wrong_activation(self):
        """Test getCache returns None for wrong activation type"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")
        input_module = Module("test")

        # Cache 'relu'
        module_obj.createCache(dt, 'relu', 0, 1, input_module)

        # Try to get 'gelu' (not cached)
        result = module_obj.getCache(dt, 'gelu', 0, 1)

        assert result is None, "Should return None when activation type not in cache"

    def test_get_cache_hit_returns_module(self):
        """Test getCache returns a Module on cache hit"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")
        input_module = Module("test")

        # Create cache
        module_obj.createCache(dt, 'relu', 0, 1, input_module)

        # Get from cache
        result = module_obj.getCache(dt, 'relu', 0, 1)

        # Should return a Module instance
        assert result is not None, "Cache hit should return a module"
        assert isinstance(result, Module), "Cache hit should return a Module instance"


@pytest.mark.unit
class TestGetModuleWithCache:
    """Tests for getModule with caching"""

    def test_get_module_with_cache_enabled(self):
        """Test getModule creates cache and reuses it with register remapping"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUseCache(True)

        dt = DataType("s")

        # Cache should be empty initially
        assert module_obj.cacheDict == {}

        # First call - should create cache entry
        result1 = module_obj.getModule(dt, 'relu', 0, 1)
        assert isinstance(result1, Module), "Should return a Module"

        # Verify cache was populated (cache uses uppercase datatype chars)
        assert 'relu' in module_obj.cacheDict, "Cache should now contain 'relu'"
        assert 'S' in module_obj.cacheDict['relu'], "Cache should have 'S' datatype entry"
        cache_entries_count = len(module_obj.cacheDict['relu']['S'])
        assert cache_entries_count == 1, "Should have exactly one cache entry"

        # Second call with different vgprs - should hit cache and remap registers
        result2 = module_obj.getModule(dt, 'relu', 5, 6)
        assert isinstance(result2, Module), "Cache hit with remapping should return a Module"

        # Verify cache wasn't duplicated - still same number of entries
        assert len(module_obj.cacheDict['relu']['S']) == cache_entries_count, \
            "Cache hit should not create duplicate entries"

        # Third call with same vgprs as first - should also hit cache
        result3 = module_obj.getModule(dt, 'relu', 0, 1)
        assert isinstance(result3, Module), "Cache hit with original vgprs should return a Module"
        assert len(module_obj.cacheDict['relu']['S']) == cache_entries_count, \
            "Repeated call should not create duplicate entries"

    @patch('Tensile.Activation.rocIsa')
    def test_get_module_all_activation_types(self, mock_rocisa):
        """Test getModule routes to correct implementation for each activation type"""
        mock_instance = Mock()
        mock_instance.getArchCaps.return_value = {"TransOpWait": True}
        mock_rocisa.getInstance.return_value = mock_instance

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        # Map activation types to their expected characteristics
        # min_instructions = 0 means no-op, = 1 means at least one real instruction
        activation_tests = {
            'none': {'min_instructions': 0, 'max_instructions': 0},  # no-op
            'abs': {'min_instructions': 1, 'max_instructions': 10},  # v_and to clear sign bit
            'relu': {'min_instructions': 1, 'max_instructions': 20},  # max(0, x)
            'clippedrelu': {'min_instructions': 1, 'max_instructions': 30},  # min(max(0, x), alpha)
            'leakyrelu': {'min_instructions': 1, 'max_instructions': 30},  # x < 0 ? alpha*x : x
            'gelu': {'min_instructions': 1, 'max_instructions': 50},  # complex polynomial
            'geluscaling': {'min_instructions': 1, 'max_instructions': 50},  # gelu variant
            'sigmoid': {'min_instructions': 1, 'max_instructions': 40},  # 1/(1+exp(-x))
            'tanh': {'min_instructions': 1, 'max_instructions': 40},  # tanh(x)
            'dgelu': {'min_instructions': 1, 'max_instructions': 50},  # gelu gradient
            'drelu': {'min_instructions': 1, 'max_instructions': 20},  # relu gradient
            'silu': {'min_instructions': 1, 'max_instructions': 40},  # x * sigmoid(x)
            'swish': {'min_instructions': 1, 'max_instructions': 40},  # same as silu
            'clamp': {'min_instructions': 1, 'max_instructions': 30},  # min(max(x, alpha), beta)
        }

        activation_modules = {}
        for act_type, expected in activation_tests.items():
            result = module_obj.getModule(dt, act_type, 0, 1)

            # Verify it's a Module instance
            assert isinstance(result, Module), f"{act_type} should return a Module instance"

            # Verify the module can be queried for instructions
            instructions = result.items()
            assert isinstance(instructions, list), f"{act_type} should have a list of instructions"

            # Verify instruction count is within expected range
            inst_count = len(instructions)
            assert inst_count >= expected['min_instructions'], \
                f"{act_type} should have at least {expected['min_instructions']} instructions, got {inst_count}"
            assert inst_count <= expected['max_instructions'], \
                f"{act_type} should have at most {expected['max_instructions']} instructions, got {inst_count}"

            # Store the module for comparison
            activation_modules[act_type] = result

        # Verify different activations produce different instruction sequences
        # (except for silu/swish which are the same)
        relu_insts = str(activation_modules['relu'].items())
        gelu_insts = str(activation_modules['gelu'].items())
        sigmoid_insts = str(activation_modules['sigmoid'].items())

        # relu, gelu, and sigmoid should all be different
        assert relu_insts != gelu_insts, "relu and gelu should produce different instructions"
        assert relu_insts != sigmoid_insts, "relu and sigmoid should produce different instructions"
        assert gelu_insts != sigmoid_insts, "gelu and sigmoid should produce different instructions"


@pytest.mark.unit
class TestActivationTypeSupportedBy:
    """Tests for ActivationType.SupportedBy enum"""

    def test_supported_by_bitwise_and(self):
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

    def test_export_enum_comparison(self):
        """Test Export enum values are distinct"""
        Export = Activation.ActivationType.Export

        # Verify values are distinct (not testing specific numbers)
        # Removed tautology: assert Export.NORMAL == Export.NORMAL
        assert Export.NORMAL != Export.GRADONLY
        assert Export.NORMAL != Export.BOTH
        assert Export.GRADONLY != Export.BOTH


@pytest.mark.unit
class TestActivationTypeStringList:
    """Tests for ActivationType.stringList"""

    def test_string_list_contents(self):
        """Test ActivationType.stringList"""
        ActivationType = Activation.ActivationType

        assert isinstance(ActivationType.stringList, list)
        assert 'alpha' in ActivationType.stringList
        assert 'beta' in ActivationType.stringList
        assert 'gamma' in ActivationType.stringList
        assert 'delta' in ActivationType.stringList

    def test_get_additional_arg_string_list_clippedrelu(self):
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

    def test_lookup_structure(self):
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

    def test_lookup_gradient_activations(self):
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

    def test_vgpr_prefix_format_application(self):
        """Test vgprPrefix applies format correctly"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        # Set format
        module_obj.setVgprPrefixFormat("ValuC+%d")

        # Should use format for integers
        result = module_obj.vgprPrefix(5)
        assert result == vgpr("ValuC+5"), "Format should be applied to integer argument"

    def test_vgpr_prefix_no_format_with_string(self):
        """Test vgprPrefix doesn't apply format to strings"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        module_obj.setVgprPrefixFormat("ValuC+%d")

        # Should wrap string in vgpr() without applying format
        result = module_obj.vgprPrefix("customVgpr")
        assert result == vgpr("customVgpr"), "String should be wrapped in vgpr() without format"


@pytest.mark.unit
class TestFuseInstructionHelpers:
    """Tests for FuseInstruction helper functions"""

    def test_combine_instructions_between_modules(self):
        """Test CombineInstructionsBetweenModules executes without error on empty module"""
        CombineInstructionsBetweenModules = Activation.CombineInstructionsBetweenModules

        module = Module("test")
        moduleAndIndex = {}

        # Should execute without raising on empty module
        CombineInstructionsBetweenModules(module, moduleAndIndex, False)

        # Module should still be valid and have no instructions (was empty)
        assert isinstance(module, Module), "Should still be a Module instance"
        assert len(module.items()) == 0, "Empty module should have no instructions"


@pytest.mark.unit
class TestGetModuleEdgeCases:
    """Tests for edge cases in getModule"""

    def test_get_module_same_vgpr_in_out(self):
        """Test getModule when vgprIn == vgprOut (no cache created)"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUseCache(True)

        dt = DataType("s")

        # Same vgpr for in and out - should not create cache
        result = module_obj.getModule(dt, 'relu', 5, 5)
        assert isinstance(result, Module), "Should return a Module even when vgprIn == vgprOut"

        # Cache should not be created for same vgpr (in-place operation)
        cached = module_obj.getCache(dt, 'relu', 5, 5)
        assert cached is None, "Cache should not be created when vgprIn == vgprOut"

    @patch('Tensile.Activation.rocIsa')
    def test_get_module_exp(self, mock_rocisa):
        """Test getModule for 'exp' activation returns valid Module"""
        # Mock rocIsa
        mock_instance = Mock()
        mock_instance.getArchCaps.return_value = {"TransOpWait": True}
        mock_rocisa.getInstance.return_value = mock_instance

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        result = module_obj.getModule(dt, 'exp', 0, 1)

        # Should return a Module instance
        assert isinstance(result, Module), "exp activation should return a Module instance"

        # Verify the module has instructions (exp is not a no-op)
        instructions = result.items()
        assert isinstance(instructions, list), "Module should have a list of instructions"
        assert len(instructions) > 0, "exp activation should generate instructions"


@pytest.mark.unit
class TestGetEnumStrListVariations:
    """Tests for getEnumStrList with different parameters"""

    def test_get_enum_str_list_tensile_only(self):
        """Test getEnumStrList with TENSILE support only"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")

        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.TENSILE, includeNone=True)
        assert isinstance(enum_list, list)
        # TENSILE-supported activations should be included
        assert 'abs' in enum_list

    def test_get_enum_str_list_hipblaslt_only(self):
        """Test getEnumStrList with HIPBLASLT support only"""
        ActivationType = Activation.ActivationType
        SupportedBy = ActivationType.SupportedBy

        dt = DataType("s")

        enum_list = ActivationType.getEnumStrList(dt, SupportedBy.HIPBLASLT, includeNone=True)
        assert isinstance(enum_list, list)
        # HIPBLASLT-supported activations should be included
        assert 'relu' in enum_list or 'gelu' in enum_list

    def test_get_enum_str_list_gradonly_export(self):
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

    def test_get_enum_str_list_both_export(self):
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

    def test_activation_type_eq_invalid_type_raises(self):
        """Test ActivationType __eq__ with invalid type raises"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")

        with pytest.raises(RuntimeError):
            _ = (act == 123)

    def test_activation_type_lt_invalid_type_raises(self):
        """Test ActivationType __lt__ with invalid type raises"""
        ActivationType = Activation.ActivationType

        act = ActivationType("relu")

        with pytest.raises(RuntimeError):
            _ = (act < 123)

    def test_activation_type_lt_with_string(self):
        """Test ActivationType __lt__ with string"""
        ActivationType = Activation.ActivationType

        act = ActivationType("abs")

        # 'abs' < 'relu' alphabetically
        assert act < 'relu'
        assert not (act < 'abs')


@pytest.mark.unit
class TestActivationModuleWithDifferentDataTypes:
    """Tests for activation modules with various data types"""

    def test_abs_with_bfloat16_no_pk(self):
        """Test abs with bfloat16 and usePK=False returns valid Module"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUsePK(False)

        dt = DataType("b")
        result = module_obj.getAbsModule(dt, 0, 1)

        # Should return a Module instance
        assert isinstance(result, Module), "getAbsModule should return a Module instance"

        # Verify module has instructions
        instructions = result.items()
        assert isinstance(instructions, list), "Module should have a list of instructions"
        # abs generates actual instructions (v_and to clear sign bit)
        assert len(instructions) > 0, "abs activation should generate instructions"

    @patch('Tensile.Activation.rocIsa')
    def test_exp_with_half_no_pk(self, mock_rocisa):
        """Test exp with half precision and usePK=False returns valid Module"""
        mock_instance = Mock()
        mock_instance.getArchCaps.return_value = {"TransOpWait": False}
        mock_rocisa.getInstance.return_value = mock_instance

        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()
        module_obj.setUsePK(False)

        dt = DataType("h")
        result = module_obj.getExpModule(dt, 0, 1)

        # Should return a Module instance
        assert isinstance(result, Module), "getExpModule should return a Module instance"

        # Verify module has instructions
        instructions = result.items()
        assert isinstance(instructions, list), "Module should have a list of instructions"
        # exp generates actual instructions (exponential computation)
        assert len(instructions) > 0, "exp activation should generate instructions"


@pytest.mark.unit
class TestGetCacheWithVgprPrefixFormat:
    """Tests for getCache with register remapping"""

    def test_get_cache_with_same_indices_returns_module(self):
        """Test getCache returns cached module when using same vgpr indices"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        # Generate a real abs activation module (simple, predictable)
        original_module = module_obj.getAbsModule(dt, 0, 1)

        # Create cache entry for abs with vgpr indices 0 (in), 1 (out)
        module_obj.createCache(dt, 'abs', 0, 1, original_module)

        # Verify cache was created
        assert 'abs' in module_obj.cacheDict
        assert 'S' in module_obj.cacheDict['abs']

        # Now get from cache with same vgpr indices
        cached_result = module_obj.getCache(dt, 'abs', 0, 1)

        # Should return a valid Module (cache hit)
        assert cached_result is not None, "Should get cache hit"
        assert isinstance(cached_result, Module), "Should return a Module"

        # Verify the cached module has instructions
        instructions = cached_result.items()
        assert len(instructions) > 0, "Cached module should have instructions"

        # The module should be a copy, not the same object
        assert cached_result is not original_module, "Should return a copy, not the original"

    def test_get_cache_miss_returns_none(self):
        """Test getCache returns None when no cache entry exists"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        # Try to get from empty cache
        result = module_obj.getCache(dt, 'relu', 0, 1)

        # Should return None (cache miss)
        assert result is None, "Cache miss should return None"

    def test_get_cache_with_different_activation_type_misses(self):
        """Test getCache returns None when requesting different activation type"""
        ActivationModule = Activation.ActivationModule
        module_obj = ActivationModule()

        dt = DataType("s")

        # Generate and cache an abs activation module
        original_module = module_obj.getAbsModule(dt, 0, 1)
        module_obj.createCache(dt, 'abs', 0, 1, original_module)

        # Try to get 'relu' from cache (only 'abs' is cached)
        result = module_obj.getCache(dt, 'relu', 0, 1)

        # Should return None (wrong activation type)
        assert result is None, "Should miss cache for different activation type"

        # Verify abs is still cached
        abs_result = module_obj.getCache(dt, 'abs', 0, 1)
        assert abs_result is not None, "abs should still be in cache"
