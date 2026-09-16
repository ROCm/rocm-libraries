"""Tests for the StinkyTofu pass plugin mechanism."""

import pytest
import stinkytofu


class TestExtensionPointEnum:
    def test_enum_exists(self):
        assert hasattr(stinkytofu, "PipelineExtensionPoint")

    def test_values_exist(self):
        EP = stinkytofu.PipelineExtensionPoint
        assert hasattr(EP, "BeforeRegionPasses")
        assert hasattr(EP, "InnerRegionBegin")
        assert hasattr(EP, "InnerRegionEnd")
        assert hasattr(EP, "AfterRegionPasses")

    def test_values_are_distinct(self):
        EP = stinkytofu.PipelineExtensionPoint
        eps = {
            EP.BeforeRegionPasses,
            EP.InnerRegionBegin,
            EP.InnerRegionEnd,
            EP.AfterRegionPasses,
        }
        assert len(eps) == 4


class TestPluginDataOnStinkyAsmModule:
    """Test plugin data API on StinkyAsmModule via stinkytofu bindings."""

    def test_has_plugin_data_methods(self):
        assert hasattr(stinkytofu.StinkyAsmModule, "setPluginDataI64")
        assert hasattr(stinkytofu.StinkyAsmModule, "getPluginDataI64")
        assert hasattr(stinkytofu.StinkyAsmModule, "setPluginDataStr")
        assert hasattr(stinkytofu.StinkyAsmModule, "getPluginDataStr")

    def test_has_register_pass_method(self):
        assert hasattr(stinkytofu.StinkyAsmModule, "registerPassAtExtensionPoint")
