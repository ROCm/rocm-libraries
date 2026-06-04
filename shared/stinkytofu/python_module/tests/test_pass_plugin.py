"""Tests for the StinkyTofu pass plugin mechanism."""

import pytest
import stinkytofu


class TestExtensionPointConstants:
    def test_constants_exist(self):
        assert hasattr(stinkytofu, "EP_BeforeRegionPasses")
        assert hasattr(stinkytofu, "EP_InnerRegionBegin")
        assert hasattr(stinkytofu, "EP_InnerRegionEnd")
        assert hasattr(stinkytofu, "EP_AfterRegionPasses")

    def test_constants_are_distinct(self):
        eps = {
            stinkytofu.EP_BeforeRegionPasses,
            stinkytofu.EP_InnerRegionBegin,
            stinkytofu.EP_InnerRegionEnd,
            stinkytofu.EP_AfterRegionPasses,
        }
        assert len(eps) == 4

    def test_constants_are_ints(self):
        assert isinstance(stinkytofu.EP_BeforeRegionPasses, int)
        assert isinstance(stinkytofu.EP_InnerRegionEnd, int)


class TestPluginDataOnStinkyAsmModule:
    """Test plugin data API on StinkyAsmModule via stinkytofu bindings."""

    def test_has_plugin_data_methods(self):
        assert hasattr(stinkytofu.StinkyAsmModule, "setPluginDataI64")
        assert hasattr(stinkytofu.StinkyAsmModule, "getPluginDataI64")
        assert hasattr(stinkytofu.StinkyAsmModule, "setPluginDataStr")
        assert hasattr(stinkytofu.StinkyAsmModule, "getPluginDataStr")

    def test_has_register_pass_method(self):
        assert hasattr(stinkytofu.StinkyAsmModule, "registerPassAtExtensionPoint")
