// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TODO(kpack-fastfollow): Temporary — proves the AOT discover->unpack->
// hipModuleLoad->getFunction wiring via captured plugin logs because
// loadDefault()'s throw is swallowed by EnginePluginResourceManager
// (EnginePluginResourceManager.cpp:230-238) and execution is not yet wired.
// Replace with a real graph submit + kernel launch + result assertion once
// selection/execution lands (at which point log matching becomes redundant).
//
// How log capture works:
//   main.cpp calls initializeTestLogRecordingShared(), which registers the
//   SHARED logRecordingCallback with the data-SDK logger. The backend routes
//   plugin logs through its own data-SDK logger (same shared library copy),
//   so HIPDNN_PLUGIN_LOG_INFO/ERROR calls in rocke-client reach the recorder.
//   SharedLogRecorder::withOverrideLevel(INFO) starts a fresh capture window
//   and restores the log level on destruction.
//
// POSITIVE test (ValidBundleDrivesAotLoad):
//   Requires: GPU device + a probe bundle beside the plugin for the running arch.
//   Drives: hipdnnCreate -> RockeClientEngine ctor -> loadDefault() ->
//           kpack_open -> kpack_get_kernel -> hipModuleLoadData ->
//           hipModuleGetFunction -> hipModuleUnload.
//   Asserts: recorder.hasLogContaining(kAotSkeletonLoadOk).
//   This proves the full load path ran, not that it was silently skipped.
//
// NEGATIVE test (CorruptBundleFailsLoudly):
//   Requires: GPU device (so loadDefault() attempts the load, not skips).
//   Drives: same flow with a corrupt .kpack; loadDefault() throws.
//   The throw IS swallowed by EnginePluginResourceManager; hipdnnCreate returns
//   HIPDNN_STATUS_SUCCESS. The observable signal is the ERROR log emitted
//   BEFORE the throw: recorder.hasLogContaining(kAotSkeletonLoadFailed).
//   This proves the load path was exercised and fails loudly in logs even
//   though the public API reflects resilience (SUCCESS).

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

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

// ---- Helpers ---------------------------------------------------------------

// Set the rocke-client plugin as the sole loaded engine plugin.
// Uses ABSOLUTE mode so each test starts with a clean plugin list.
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

// Return the plugin DSO's directory (parent of PLUGIN_PATH).
std::filesystem::path pluginDir()
{
    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
    return pluginPath.parent_path();
}

// Return the bare GFX arch string for device 0 (e.g. "gfx942"), or "" on failure.
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

// Path where loadDefault() expects the probe bundle for the given arch:
//   <plugin_dir>/arch_content/rocke/<arch>/rocke_client_<arch>.kpack
std::filesystem::path probeBundleKpackPath(const std::string& arch)
{
    return pluginDir() / "arch_content" / "rocke" / arch / ("rocke_client_" + arch + ".kpack");
}

std::filesystem::path probeBundleManifestPath(const std::string& arch)
{
    return pluginDir() / "arch_content" / "rocke" / arch / ("rocke_client_" + arch + ".json");
}

// Write a minimal valid-looking manifest for the probe toc_key/symbol so that
// loadDefault() gets past JSON parsing before failing on the corrupt kpack.
void writeProbeManifest(const std::filesystem::path& path, const std::string& arch)
{
    std::ofstream out{path};
    out << R"({
  "schema": "rocke.aot.bundle/v1",
  "arch": ")"
        << arch << R"(",
  "kpack": "rocke_client_)"
        << arch << R"(.kpack",
  "entries": [
    { "toc_key": "rocke/test/skeleton/rocke_test_probe", "symbol": "rocke_test_probe" }
  ]
}
)";
}

} // namespace

// ---- POSITIVE test ---------------------------------------------------------

TEST(TestRockeClientAotLoad, ValidBundleDrivesAotLoad)
{
    // Requires a HIP device: without one, loadDefault() returns early without
    // touching the bundle and never emits the success marker.
    SKIP_IF_NO_DEVICES();

    const std::string arch = runningDeviceArch();
    if(arch.empty())
    {
        GTEST_SKIP() << "Could not determine device arch";
    }

    const auto kpackPath = probeBundleKpackPath(arch);
    if(!std::filesystem::exists(kpackPath))
    {
        GTEST_SKIP() << "No probe bundle for arch '" << arch << "' at " << kpackPath.string()
                     << "; build with ROCKE_CLIENT_ENABLE_TESTS=ON and the rocke pyenv "
                        "configured (ROCKE_PYENV_PYTHON + ROCKE_KPACK_PYTHON_DIR) so the "
                        "CMake POST_BUILD step generates it beside the plugin";
    }

    // Ensure the backend and plugin both emit INFO-level logs.
    hipdnnSeverity_t originalLevel{HIPDNN_SEV_OFF};
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&originalLevel), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Load plugin DSO now (with INFO level active) so the plugin's own log level
    // is set to INFO when loadPluginFromFile calls plugin->setLogLevel.
    setRockeClientPluginPath();

    // Capture stderr around hipdnnCreate: loadDefault() (invoked via
    // RockeClientEngine's ctor) emits its marker through the plugin logger to
    // stderr. Capturing stderr is the simplest robust observable and mirrors
    // hipDNN's TestBackendLogger/TestGraphLogger pattern.
    // TODO(kpack-fastfollow): replace with a real graph submit + kernel launch +
    // result assertion once selection/execution lands.
    testing::internal::CaptureStderr();
    hipdnnStatus_t createStatus = HIPDNN_STATUS_INTERNAL_ERROR;
    {
        const HipdnnHandle handle{&createStatus};
    }
    const std::string logs = testing::internal::GetCapturedStderr();

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(originalLevel), HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS)
        << "hipdnnCreate failed unexpectedly; loadDefault() may have thrown.\n"
        << logs;

    // Decisive assertion: loadDefault() emitted the success marker, proving the
    // full kpack_open -> kpack_get_kernel -> hipModuleLoadData ->
    // hipModuleGetFunction path ran for this arch without throwing.
    ASSERT_NE(logs.find(rocke_client::dispatcher::AOT_SKELETON_LOAD_OK), std::string::npos)
        << "loadDefault() did not emit '" << rocke_client::dispatcher::AOT_SKELETON_LOAD_OK
        << "'. The load path may have been skipped or failed before hipModuleGetFunction.\n"
        << logs;
}

// ---- NEGATIVE test ---------------------------------------------------------

TEST(TestRockeClientAotLoad, CorruptBundleFailsLoudly)
{
    // Requires a HIP device: without one, loadDefault() detects no device and
    // returns an empty catalog without ever trying to open the bundle.
    SKIP_IF_NO_DEVICES();

    const std::string arch = runningDeviceArch();
    if(arch.empty())
    {
        GTEST_SKIP() << "Could not determine device arch";
    }

    const auto kpackPath = probeBundleKpackPath(arch);
    const auto mfstPath = probeBundleManifestPath(arch);
    const auto backupKpack = kpackPath.parent_path() / (kpackPath.filename().string() + ".bak");

    // Back up the valid bundle (if present) and write a corrupt kpack.
    // A valid manifest is kept so loadDefault() gets past JSON parsing and
    // actually reaches kpack_open (which will then fail and emit the ERROR marker).
    const bool hadOriginal = std::filesystem::exists(kpackPath);
    std::filesystem::create_directories(kpackPath.parent_path());
    if(hadOriginal)
    {
        std::filesystem::copy_file(
            kpackPath, backupKpack, std::filesystem::copy_options::overwrite_existing);
    }
    {
        std::ofstream{kpackPath, std::ios::binary} << "CORRUPT_KPACK_NOT_A_VALID_ARCHIVE";
    }
    writeProbeManifest(mfstPath, arch);

    hipdnnSeverity_t originalLevel{HIPDNN_SEV_OFF};
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&originalLevel), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Load plugin DSO with INFO level so the plugin's log level is INFO.
    setRockeClientPluginPath();

    // Capture stderr around hipdnnCreate; loadDefault() emits the ERROR marker
    // through the plugin logger to stderr before throwing. Mirrors hipDNN's
    // TestBackendLogger/TestGraphLogger stderr-capture pattern.
    testing::internal::CaptureStderr();
    hipdnnStatus_t createStatus = HIPDNN_STATUS_INTERNAL_ERROR;
    {
        const HipdnnHandle handle{&createStatus};
    }
    const std::string logs = testing::internal::GetCapturedStderr();

    // Restore the valid bundle BEFORE asserting so a failing assertion does not
    // leave the environment broken.
    if(hadOriginal)
    {
        std::filesystem::copy_file(
            backupKpack, kpackPath, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::remove(backupKpack);
    }
    else
    {
        std::filesystem::remove(kpackPath);
    }

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(originalLevel), HIPDNN_STATUS_SUCCESS);

    // The exception IS swallowed at EnginePluginResourceManager ctor
    // (catch(std::exception&) + continue), so hipdnnCreate returns SUCCESS even
    // though loadDefault() threw.
    ASSERT_EQ(createStatus, HIPDNN_STATUS_SUCCESS)
        << "Unexpected: hipdnnCreate returned non-SUCCESS with a corrupt bundle; the "
           "swallow finding may have changed.\n"
        << logs;

    // Observable signal: loadDefault() emits the ERROR marker BEFORE throwing,
    // proving the load path was exercised (not skipped) and fails loudly in logs
    // even though the failure is absorbed by the backend.
    ASSERT_NE(logs.find(rocke_client::dispatcher::AOT_SKELETON_LOAD_FAILED), std::string::npos)
        << "loadDefault() did not emit '" << rocke_client::dispatcher::AOT_SKELETON_LOAD_FAILED
        << "'. The corrupt kpack may not have been reached (load path skipped).\n"
        << logs;
}
