// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// End-to-end integration test for the per-device lazy AOT load path:
//   ROCKE_CLIENT_AOT_BUNDLE_DIR (env) -> aotBundleDir() -> aotKpackPath() ->
//   RockeClientDispatcher::catalogForDevice() -> AotCatalog::loadForDevice() ->
//   kpack_open -> kpack_get_kernel -> hipModuleLoadData -> hipModuleGetFunction
//   -> hipModuleUnload -> empty catalog (engine stays inert; scaffolding only).
//
// The lazy load fires on the first selection attempt per device, which
// graph.is_supported_ext(handle) triggers via:
//   hipdnnBackendFinalize -> engine heuristic -> selectInstance ->
//   catalogForDevice(deviceId, arch) [first call, so loadForDevice runs].
//
// Log capture: main.cpp registers initializeChainedTestLogRecordingShared()
//   (logChainedRecordingCallback with force=true), routing plugin logs to
//   stderr regardless of HIPDNN_LOG_LEVEL. testing::internal::CaptureStderr()
//   captures that output so tests can assert the stable markers emitted by
//   loadForDevice() without GPU result validation.
//
// TODO(AICK-1484): this whole test is temporary. It proves the load wiring by
//   asserting on log markers; replace it with a real E2E test (graph submit +
//   kernel launch + result validation) once plan-based execution lands. The
//   AotSkeletonMarkers.hpp constants are removed with it.
//
// Bundle layout (test-only; installed + relocatable, NOT under arch_content):
//   <exeDir>/hip_kernel_provider/tests/aot_test_bundles/valid/<arch>/rocke_client_<arch>.{kpack,json}
//   <exeDir>/hip_kernel_provider/tests/aot_test_bundles/corrupt/<arch>/rocke_client_<arch>.{kpack,json}
// Generated at build time and installed by integration_tests/CMakeLists.txt;
// captured by the hipkernelprovider [test] artifact via bin/hip_kernel_provider/tests/**.

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <string>

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend.hpp>

#include "dispatcher/AotSkeletonMarkers.hpp"

namespace
{

// ---- RAII helpers ----------------------------------------------------------

class HipdnnHandle
{
public:
    explicit HipdnnHandle(hipdnnStatus_t* statusOut)
    {
        *statusOut = hipdnnCreate(&_handle);
    }

    ~HipdnnHandle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    hipdnnHandle_t get() const
    {
        return _handle;
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

class HipStream
{
public:
    HipStream()
    {
        EXPECT_EQ(hipStreamCreate(&_stream), hipSuccess);
    }

    ~HipStream()
    {
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    hipStream_t get() const
    {
        return _stream;
    }

private:
    hipStream_t _stream = nullptr;
};

// ---- Helpers ---------------------------------------------------------------

void setRockeClientPluginPath()
{
    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
    const std::string pluginPathStr = pluginPath.string();
    const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS)
        << "Failed to set plugin path: " << pluginPathStr;
}

// Return the bare GFX arch string for device 0 (e.g. "gfx942").
std::string runningDeviceArch()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return {};
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return {};
    }
    std::string arch{props.gcnArchName};
    const auto colon = arch.find(':');
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

// Root of the valid/corrupt test bundles, resolved RELATIVE TO THE TEST
// EXECUTABLE so it works both in the build tree and after a relocatable install:
//   <exe_dir>/hip_kernel_provider/tests/aot_test_bundles
// CMake stages and installs the bundles there (see integration_tests/
// CMakeLists.txt); the hipkernelprovider [test] artifact captures them via
// bin/hip_kernel_provider/tests/**. Each test sets ROCKE_CLIENT_AOT_BUNDLE_DIR
// to <root>/<valid|corrupt>/<arch> so aotBundleDir() returns that dir directly.
// Test-only; never under arch_content or the runtime lib artifact.
std::filesystem::path bundleRootDir()
{
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / "hip_kernel_provider"
           / "tests" / "aot_test_bundles";
}

// Minimal fp16 SDPA forward graph that triggers loadForDevice() on the first
// call to graph.is_supported_ext(). Shape mirrors buildMinimalSdpaForwardGraph
// in TestRockeClientApplicability.cpp.
hipdnn_frontend::graph::Graph buildSdpaForwardGraph()
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    const std::vector<int64_t> qkvDims = {2, 8, 32, 64};
    const std::vector<int64_t> qkvStrides = hipdnn_data_sdk::utilities::generateStrides(qkvDims);

    Graph graph;
    graph.set_name("RockeClientAotLoadTest_SdpaFwd")
        .set_io_data_type(DataType::HALF)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto qAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("Q", qkvDims, qkvStrides));
    auto kAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("K", qkvDims, qkvStrides));
    auto vAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("V", qkvDims, qkvStrides));

    const SdpaAttributes sdpaAttrs;
    auto [oAttr, statsAttr] = graph.sdpa(qAttr, kAttr, vAttr, sdpaAttrs);
    oAttr->set_output(true);
    return graph;
}

} // namespace

// ============================================================================
// POSITIVE: valid probe bundle -> load succeeds -> LOAD OK marker in stderr
// ============================================================================

TEST(TestRockeClientAotLoad, ValidBundleDrivesAotLoad)
{
    // The lazy load only fires when a HIP device is present: loadForDevice
    // returns empty immediately when no device is available.
    SKIP_IF_NO_DEVICES();

    const std::string arch = runningDeviceArch();
    if(arch.empty())
    {
        GTEST_SKIP() << "Could not determine device arch";
    }

    const auto validDir = bundleRootDir() / "valid" / arch;
    if(!std::filesystem::exists(validDir / ("rocke_client_" + arch + ".kpack")))
    {
        GTEST_SKIP() << "No valid probe bundle for arch '" << arch << "' at " << validDir.string()
                     << "; build with ROCKE_CLIENT_ENABLE_TESTS=ON and the rocke pyenv "
                        "(ROCKE_PYENV_PYTHON + ROCKE_KPACK_PYTHON_DIR) configured so "
                        "the CMake add_custom_command generates it";
    }

    // Point aotBundleDir() at the valid test bundle dir (NOT arch_content).
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envGuard(
        "ROCKE_CLIENT_AOT_BUNDLE_DIR", validDir.string());

    // Backend + plugin log level must be INFO so loadForDevice emits its markers.
    hipdnnSeverity_t originalLevel{HIPDNN_SEV_OFF};
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&originalLevel), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Load plugin DSO: plugin's log level is set to INFO at load time.
    setRockeClientPluginPath();

    hipdnnStatus_t createStatus = HIPDNN_STATUS_INTERNAL_ERROR;
    const HipdnnHandle handle{&createStatus};
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);

    const HipStream stream;
    ASSERT_EQ(hipdnnSetStream(handle.get(), stream.get()), HIPDNN_STATUS_SUCCESS);

    // Capture stderr around the first selection attempt. is_supported_ext drives:
    //   hipdnnBackendFinalize -> selectInstance -> catalogForDevice (first call)
    //   -> loadForDevice -> kpack_open -> kpack_get_kernel -> hipModuleLoadData
    //   -> hipModuleGetFunction -> hipModuleUnload -> emits kAotSkeletonLoadOk.
    testing::internal::CaptureStderr();

    auto graph = buildSdpaForwardGraph();
    const auto result = graph.is_supported_ext(handle.get());

    const std::string captured = testing::internal::GetCapturedStderr();

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(originalLevel), HIPDNN_STATUS_SUCCESS);

    // The engine always declines (empty catalog, skeleton only); that's expected.
    // The decisive assertion is the SUCCESS marker in the captured plugin logs.
    ASSERT_NE(captured.find(rocke_client::dispatcher::AOT_SKELETON_LOAD_OK), std::string::npos)
        << "loadForDevice() did not emit '" << rocke_client::dispatcher::AOT_SKELETON_LOAD_OK
        << "'. Either the probe bundle was not reached (check ROCKE_CLIENT_AOT_BUNDLE_DIR) "
           "or the load path was silently skipped. Plugin log capture:\n"
        << captured;

    static_cast<void>(result); // engine declines — expected; not asserted
}

// ============================================================================
// NEGATIVE: corrupt kpack -> load fails loudly -> LOAD FAILED marker in stderr
// ============================================================================

TEST(TestRockeClientAotLoad, CorruptBundleFailsLoudly)
{
    // GPU required: without a device, loadForDevice skips the bundle entirely.
    SKIP_IF_NO_DEVICES();

    const std::string arch = runningDeviceArch();
    if(arch.empty())
    {
        GTEST_SKIP() << "Could not determine device arch";
    }

    const auto corruptDir = bundleRootDir() / "corrupt" / arch;
    if(!std::filesystem::exists(corruptDir / ("rocke_client_" + arch + ".kpack")))
    {
        GTEST_SKIP() << "No corrupt probe bundle for arch '" << arch << "' at "
                     << corruptDir.string();
    }

    // Point aotBundleDir() at the corrupt test bundle dir.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envGuard(
        "ROCKE_CLIENT_AOT_BUNDLE_DIR", corruptDir.string());

    hipdnnSeverity_t originalLevel{HIPDNN_SEV_OFF};
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&originalLevel), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    setRockeClientPluginPath();

    hipdnnStatus_t createStatus = HIPDNN_STATUS_INTERNAL_ERROR;
    const HipdnnHandle handle{&createStatus};
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS);

    const HipStream stream;
    ASSERT_EQ(hipdnnSetStream(handle.get(), stream.get()), HIPDNN_STATUS_SUCCESS);

    // Capture stderr: loadForDevice reaches kpack_open, which rejects the
    // corrupt file (invalid KPAK magic), logs kAotSkeletonLoadFailed, and
    // returns an empty catalog. No exception escapes (noexcept path).
    testing::internal::CaptureStderr();

    auto graph = buildSdpaForwardGraph();
    const auto result = graph.is_supported_ext(handle.get());

    const std::string captured = testing::internal::GetCapturedStderr();

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(originalLevel), HIPDNN_STATUS_SUCCESS);

    // The FAIL marker proves the load path was executed and failed loudly —
    // not silently skipped. loadForDevice() does NOT throw (noexcept path):
    // the error is an ERROR-level log, not an exception.
    ASSERT_NE(captured.find(rocke_client::dispatcher::AOT_SKELETON_LOAD_FAILED), std::string::npos)
        << "loadForDevice() did not emit '" << rocke_client::dispatcher::AOT_SKELETON_LOAD_FAILED
        << "'. Either the corrupt kpack was not reached (check ROCKE_CLIENT_AOT_BUNDLE_DIR) "
           "or the load path was silently skipped.\nPlugin log capture:\n"
        << captured;

    static_cast<void>(result);
}
