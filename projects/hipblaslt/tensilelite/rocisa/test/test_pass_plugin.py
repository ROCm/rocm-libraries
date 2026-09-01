################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Tests for the StinkyTofu pass plugin mechanism exposed through rocisa."""

import os

import pytest
import rocisa
from rocisa.code import Module, SignatureBase

_ISA = (12, 5, 0)

# Both HelloWorldPass integration classes below need a plugin that only exists when
# rocisa/stinkytofu were configured for it, so by default they skip when it is
# missing. A skip is worthless as CI coverage, so CI pins the configuration it built
# via ROCISA_PLUGIN_TEST_MODE and those skips become hard failures:
#
#   shared  stinkytofu built shared with STINKYTOFU_BUILD_EXAMPLES=ON and rocisa with
#           ROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN=ON: both the dlopen and the
#           compiled-in path must be available and must pass.
#   static  same, but stinkytofu built with BUILD_SHARED_LIBS=OFF: the compiled-in
#           path must be available and must pass, and the dlopen path must be
#           *absent*. Asserting the absence is what stops a build that silently
#           came out shared from passing as the static configuration — with
#           STINKYTOFU_BUILD_EXAMPLES=ON a shared build always produces a
#           dlopen'able plugin, so a non-empty path means it was not static.
#
# Unset (the default, e.g. a developer's local build) keeps the lenient skips.
_PLUGIN_TEST_MODE = os.environ.get("ROCISA_PLUGIN_TEST_MODE", "")

if _PLUGIN_TEST_MODE not in ("", "shared", "static"):
    raise ValueError(
        f"ROCISA_PLUGIN_TEST_MODE must be unset, 'shared' or 'static', "
        f"got {_PLUGIN_TEST_MODE!r}"
    )


def _missing_prerequisite(reason):
    """Skip on a missing plugin prerequisite, or fail if CI pinned the configuration."""
    if _PLUGIN_TEST_MODE:
        pytest.fail(f"{reason} (ROCISA_PLUGIN_TEST_MODE={_PLUGIN_TEST_MODE})")
    pytest.skip(reason)


_ISA_SUPPORTED = rocisa.isSupportedByStinkyTofu(_ISA)

# A CI configuration that pinned ROCISA_PLUGIN_TEST_MODE must not go green by skipping
# the whole module; raising here surfaces it as a collection error instead.
if _PLUGIN_TEST_MODE and not _ISA_SUPPORTED:
    raise RuntimeError(
        f"ROCISA_PLUGIN_TEST_MODE={_PLUGIN_TEST_MODE} requires a StinkyTofu backend for "
        f"gfx{''.join(str(v) for v in _ISA)}, but none is registered"
    )

pytestmark = pytest.mark.skipif(
    not _ISA_SUPPORTED,
    reason=f"gfx{''.join(str(v) for v in _ISA)} not registered in StinkyTofu BackendRegistry",
)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    import shutil

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(_ISA, assembler, False)


def _make_stinky_module(name="test_kernel"):
    mod = Module(name)
    sig = SignatureBase(
        kernelName=name,
        kernArgsVersion=0,
        codeObjectVersion="default",
        groupSegmentSize=0,
        sgprWorkGroup=[0, 1, 2],
        vgprWorkItem=0,
        flatWorkGroupSize=32,
    )
    opts = {"OptLevel": 0, "wavefrontSize": 32}
    return rocisa.toStinkyTofuModule(mod, _ISA, name, sig, opts)


class TestExtensionPointEnum:
    def test_enum_exists(self):
        assert hasattr(rocisa, "PipelineExtensionPoint")

    def test_values_exist(self):
        EP = rocisa.PipelineExtensionPoint
        assert hasattr(EP, "BeforeRegionPasses")
        assert hasattr(EP, "InnerRegionBegin")
        assert hasattr(EP, "InnerRegionEnd")
        assert hasattr(EP, "AfterRegionPasses")

    def test_values_are_distinct(self):
        EP = rocisa.PipelineExtensionPoint
        eps = {
            EP.BeforeRegionPasses,
            EP.InnerRegionBegin,
            EP.InnerRegionEnd,
            EP.AfterRegionPasses,
        }
        assert len(eps) == 4


class TestPluginData:
    def test_set_and_get_i64(self):
        st = _make_stinky_module()
        st.setPluginDataI64("enableTDMA", 1)
        assert st.getPluginDataI64("enableTDMA") == 1

    def test_get_i64_default(self):
        st = _make_stinky_module()
        assert st.getPluginDataI64("nonexistent", -1) == -1

    def test_set_and_get_str(self):
        st = _make_stinky_module()
        st.setPluginDataStr("kernelName", "my_kernel")
        assert st.getPluginDataStr("kernelName") == "my_kernel"

    def test_get_str_default(self):
        st = _make_stinky_module()
        assert st.getPluginDataStr("nonexistent", "fallback") == "fallback"

    def test_overwrite_i64(self):
        st = _make_stinky_module()
        st.setPluginDataI64("key", 42)
        assert st.getPluginDataI64("key") == 42
        st.setPluginDataI64("key", 99)
        assert st.getPluginDataI64("key") == 99

    def test_multiple_keys(self):
        st = _make_stinky_module()
        st.setPluginDataI64("a", 1)
        st.setPluginDataI64("b", 2)
        st.setPluginDataStr("c", "three")
        assert st.getPluginDataI64("a") == 1
        assert st.getPluginDataI64("b") == 2
        assert st.getPluginDataStr("c") == "three"


class TestRegisterPassAtExtensionPoint:
    def test_register_unknown_pass_does_not_crash_until_run(self):
        st = _make_stinky_module()
        st.registerPassAtExtensionPoint(rocisa.PipelineExtensionPoint.InnerRegionEnd, "UnknownPass")

    def test_pipeline_runs_with_no_plugins(self):
        st = _make_stinky_module()
        st.runOptimizationPipeline()

    def test_plugin_data_survives_pipeline_run(self):
        st = _make_stinky_module()
        st.setPluginDataI64("testKey", 123)
        st.runOptimizationPipeline()
        assert st.getPluginDataI64("testKey") == 123


class _HelloWorldPassBehavior:
    """HelloWorldPass end-to-end checks, shared by both registration paths.

    The behavior of the pass is identical however it was registered, so the
    assertions live here once and each subclass only supplies the fixture that
    makes the pass available. Not named Test* so pytest collects it only through
    the subclasses below."""

    def test_hello_world_pass_executes(self):
        st = _make_stinky_module()
        st.setPluginDataStr("greeting", "Hello from rocisa test!")
        st.registerPassAtExtensionPoint(
            rocisa.PipelineExtensionPoint.AfterRegionPasses, "HelloWorldPass"
        )
        st.runOptimizationPipeline()
        assert st.getPluginDataI64("pass_executed") == 1
        assert st.getPluginDataStr("greeting_result") == "executed: Hello from rocisa test!"

    def test_hello_world_pass_default_greeting(self):
        st = _make_stinky_module()
        st.registerPassAtExtensionPoint(
            rocisa.PipelineExtensionPoint.AfterRegionPasses, "HelloWorldPass"
        )
        st.runOptimizationPipeline()
        assert st.getPluginDataI64("pass_executed") == 1
        assert st.getPluginDataStr("greeting_result") == "executed: Hello from plugin!"


class TestHelloWorldPassIntegrationDynamic(_HelloWorldPassBehavior):
    """End-to-end test: dynamically load HelloWorldPass plugin (dlopen), set
    plugin data, run pipeline, verify the pass executed.

    Requires stinkytofu built shared with STINKYTOFU_BUILD_EXAMPLES=ON, so the
    plugin exists as a dlopen'able module. See TestHelloWorldPassIntegrationStatic
    for the compiled-in counterpart, which works regardless of how stinkytofu
    itself was built."""

    @pytest.fixture(autouse=True)
    def _load_plugin(self):
        # Ask stinkytofu where its own example plugin is; it resolves the path
        # relative to the loaded libstinkytofu. Empty means it was built with
        # STINKYTOFU_BUILD_EXAMPLES=OFF, or stinkytofu is static (no dlopen'able
        # module exists in that case — see PassBuilder::examplePluginPath()).
        path = rocisa.stinkytofuExamplePluginPath()
        if _PLUGIN_TEST_MODE == "static":
            assert path == "", (
                "static configuration produced a dlopen'able example plugin at "
                f"{path!r}; stinkytofu was built shared, not static"
            )
            pytest.skip("static configuration: no dlopen'able plugin exists by design")
        if not path:
            _missing_prerequisite(
                "StinkyTofu example plugin not built as a dlopen'able module"
            )
        rocisa.loadPlugin(path)


class TestHelloWorldPassIntegrationStatic(_HelloWorldPassBehavior):
    """Same end-to-end test as the dynamic version above, but via the
    LLVM-style static-plugin path: HelloWorldPass.cpp is compiled directly
    into _rocisa and registered with an explicit call instead of dlopen.

    Requires rocisa built with ROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN=ON. Unlike
    the dynamic path, this works whether stinkytofu itself is static or
    shared."""

    @pytest.fixture(autouse=True)
    def _register_plugin(self):
        if not hasattr(rocisa, "registerExamplePlugin"):
            _missing_prerequisite(
                "rocisa built without ROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN"
            )
        rocisa.registerExamplePlugin()
