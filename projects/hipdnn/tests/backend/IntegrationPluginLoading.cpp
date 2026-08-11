// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include "TestUtil.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include <HipdnnBackendAttributeName.h>
#include <HipdnnBackendAttributeType.h>
#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <test_plugins/TestPluginEngineIdMap.hpp>

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_tests::plugin_constants;

class IntegrationPluginLoading : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _engineConfig = nullptr;
    hipdnnBackendDescriptor_t _engine = nullptr;
    hipdnnBackendDescriptor_t _graph = nullptr;
    hipdnnBackendDescriptor_t _heuristicDescriptor = nullptr;
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;

    void SetUp() override {}

    // Bind a real stream to the handle. Required for tests that finalize a
    // heuristic descriptor with a non-empty applicable-engine list, since
    // EngineHeuristicDescriptor::finalize() resolves the device through
    // hipStreamGetDevice(handle->getStream(), ...). Caller must invoke
    // SKIP_IF_NO_DEVICES() before this so the test skips on no-GPU runners.
    void bindStream()
    {
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_engineConfig != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engineConfig), HIPDNN_STATUS_SUCCESS);
            _engineConfig = nullptr;
        }
        if(_engine != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_engine), HIPDNN_STATUS_SUCCESS);
            _engine = nullptr;
        }
        if(_graph != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_graph), HIPDNN_STATUS_SUCCESS);
            _graph = nullptr;
        }
        if(_heuristicDescriptor != nullptr)
        {
            EXPECT_EQ(hipdnnBackendDestroyDescriptor(_heuristicDescriptor), HIPDNN_STATUS_SUCCESS);
            _heuristicDescriptor = nullptr;
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
            _stream = nullptr;
        }
    }
};

namespace
{
// Installs a user log callback and raises the backend's global log level, then
// puts both back on scope exit. The restoration has to be unconditional: a
// failed assertion inside a test leaves the rest of the test body unrun, and a
// callback or log level left installed contaminates every later test in the
// binary.
class ScopedBackendLogCapture
{
public:
    ScopedBackendLogCapture(hipdnnUserLogCallback_t callback,
                            hipdnnSeverity_t level,
                            void* userData)
        : _callback(callback)
        , _userData(userData)
    {
        EXPECT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&_originalLevel), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            hipdnnSetUserLogCallback_ext(_callback, level, HIPDNN_LOG_CALLBACK_SYNC, _userData),
            HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(level), HIPDNN_STATUS_SUCCESS);
    }

    ~ScopedBackendLogCapture()
    {
        EXPECT_EQ(hipdnnSetUserLogCallback_ext(
                      _callback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, _userData),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(_originalLevel), HIPDNN_STATUS_SUCCESS);
    }

    ScopedBackendLogCapture(const ScopedBackendLogCapture&) = delete;
    ScopedBackendLogCapture& operator=(const ScopedBackendLogCapture&) = delete;
    ScopedBackendLogCapture(ScopedBackendLogCapture&&) = delete;
    ScopedBackendLogCapture& operator=(ScopedBackendLogCapture&&) = delete;

private:
    hipdnnUserLogCallback_t _callback;
    void* _userData;
    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};
};

void createHeuristicDescriptor(hipdnnBackendDescriptor_t* heuristicDescriptor,
                               hipdnnBackendDescriptor_t* graph,
                               bool finalize = false)
{
    EXPECT_EQ(
        hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, heuristicDescriptor),
        HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(*heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(graph)),
              HIPDNN_STATUS_SUCCESS);

    auto backendModes = HIPDNN_HEUR_MODE_FALLBACK;

    EXPECT_EQ(hipdnnBackendSetAttribute(*heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_MODE,
                                        HIPDNN_TYPE_HEUR_MODE,
                                        1,
                                        &backendModes),
              HIPDNN_STATUS_SUCCESS);

    if(finalize)
    {
        EXPECT_EQ(hipdnnBackendFinalize(*heuristicDescriptor), HIPDNN_STATUS_SUCCESS);
    }
}

// One engine as reported by hipdnnGetEngineInfo_ext.
struct ReportedEngine
{
    int64_t engineId{0};
    std::string engineName;
};

// Enumerate every loaded engine through the public two-call hipdnnGetEngineInfo_ext pattern,
// the same path tools/ListEngines.cpp drives.
std::vector<ReportedEngine> queryReportedEngines(hipdnnHandle_t handle)
{
    std::vector<ReportedEngine> engines;

    auto engineCount = size_t{0};
    EXPECT_EQ(hipdnnGetEngineCount_ext(handle, &engineCount), HIPDNN_STATUS_SUCCESS);
    engines.reserve(engineCount);

    for(size_t index = 0; index < engineCount; ++index)
    {
        auto engineId = int64_t{0};
        auto engineNameLen = size_t{0};
        auto pluginNameLen = size_t{0};
        auto versionLen = size_t{0};
        auto typeLen = size_t{0};
        EXPECT_EQ(hipdnnGetEngineInfo_ext(handle,
                                          index,
                                          &engineId,
                                          nullptr,
                                          &engineNameLen,
                                          nullptr,
                                          &pluginNameLen,
                                          nullptr,
                                          &versionLen,
                                          nullptr,
                                          &typeLen),
                  HIPDNN_STATUS_SUCCESS);

        std::vector<char> engineName(engineNameLen);
        std::vector<char> pluginName(pluginNameLen);
        std::vector<char> version(versionLen);
        std::vector<char> type(typeLen);
        EXPECT_EQ(hipdnnGetEngineInfo_ext(handle,
                                          index,
                                          nullptr,
                                          engineName.data(),
                                          &engineNameLen,
                                          pluginName.data(),
                                          &pluginNameLen,
                                          version.data(),
                                          &versionLen,
                                          type.data(),
                                          &typeLen),
                  HIPDNN_STATUS_SUCCESS);

        engines.push_back(ReportedEngine{engineId, std::string(engineName.data())});
    }

    return engines;
}

// Render every reported engine as "name (0xID)" so a failing expectation shows the whole listing.
std::string describeReportedEngines(const std::vector<ReportedEngine>& engines)
{
    std::string description;
    for(const auto& engine : engines)
    {
        description += "  " + engine.engineName + " ("
                       + hipdnn_data_sdk::utilities::formatEngineIdHex(engine.engineId) + ")\n";
    }
    return description;
}

// Load a single engine plugin by absolute path, replacing any previously configured paths.
void setSingleEnginePluginPath(const std::string& pluginPath)
{
    const std::array<const char*, 1> paths = {pluginPath.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
}
} // namespace

TEST_F(IntegrationPluginLoading, EmptyPluginPath)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory pluginDir("empty_plugins");
    auto pluginPath = pluginDir.path().string();
    const std::array<const char*, 1> paths = {pluginPath.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 0);
}

TEST_F(IntegrationPluginLoading, IncorrectEngineID)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    test_util::createTestEngine(&_engine, &_graph, _handle, -193489);

    ASSERT_EQ(hipdnnBackendFinalize(_engine), HIPDNN_STATUS_BAD_PARAM);

    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer;
    hipdnnGetLastErrorString(buffer.data(), buffer.size());

    ASSERT_EQ(
        std::string{buffer.data()},
        "EngineDescriptor::finalize() failed: Engine id is not in a valid range of engine IDs");
}

TEST_F(IntegrationPluginLoading, DuplicateEngineIds)
{
    const std::array<const char*, 2> paths
        = {hipdnn_tests::plugin_constants::testDuplicateIdAPluginPath().c_str(),
           hipdnn_tests::plugin_constants::testDuplicateIdBPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer;
    hipdnnGetLastErrorString(buffer.data(), buffer.size());

    const std::string expectedError
        = fmt::format("Engine ID {} already exists",
                      hipdnn_tests::plugin_constants::engineId<DuplicateIdBPlugin>());

    EXPECT_NE(std::string{buffer.data()}.find(expectedError), std::string::npos);

    EXPECT_EQ(test_util::getLoadedPlugins(_handle).size(), 1);
}

TEST_F(IntegrationPluginLoading, IncompleteAPI)
{
    using namespace hipdnn_data_sdk::utilities;
    using namespace hipdnn_tests::plugin_constants;

    const std::array<const char*, 1> paths = {testIncompleteApiPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer;
    hipdnnGetLastErrorString(buffer.data(), buffer.size());

    EXPECT_NE(std::string{buffer.data()}.find("Failed to get symbol"), std::string::npos);
    EXPECT_EQ(test_util::getLoadedPlugins(_handle).size(), 0);
}

TEST_F(IntegrationPluginLoading, SinglePluginNoApplicableEngines)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 0);
}

TEST_F(IntegrationPluginLoading, MultiplePluginsNoApplicableEngines)
{
    const std::array<const char*, 2> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str(),
           hipdnn_tests::plugin_constants::testNoApplicableEnginesBPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 0);
}

TEST_F(IntegrationPluginLoading, MultiplePluginsOneApplicableEngine)
{
    SKIP_IF_NO_DEVICES();

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE),
        HIPDNN_STATUS_SUCCESS);

    const std::array<const char*, 1> heuristicPaths
        = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath().c_str()};
    ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                  heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter policyEnv(
        "HIPDNN_HEUR_POLICY_ORDER", hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName());

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    bindStream();
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 1);
}

TEST_F(IntegrationPluginLoading, MultiplePluginsMultipleApplicableEngines)
{
    SKIP_IF_NO_DEVICES();

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE),
        HIPDNN_STATUS_SUCCESS);

    const std::array<const char*, 1> heuristicPaths
        = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath().c_str()};
    ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                  heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter policyEnv(
        "HIPDNN_HEUR_POLICY_ORDER", hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName());

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    bindStream();
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &_engineConfig),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(_engineConfig, nullptr);

    test_util::createTestGraph(&_graph, _handle);
    hipdnnBackendFinalize(_graph);

    createHeuristicDescriptor(&_heuristicDescriptor, &_graph, true);

    auto availableEngineCount = int64_t{-1};
    EXPECT_EQ(hipdnnBackendGetAttribute(_heuristicDescriptor,
                                        HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        0,
                                        &availableEngineCount,
                                        nullptr),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(availableEngineCount, 2);
}

TEST_F(IntegrationPluginLoading, PluginWithIncompatibleApiVersion)
{

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testIncompatibleVersionPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> buffer;
    hipdnnGetLastErrorString(buffer.data(), buffer.size());

    EXPECT_NE(std::string{buffer.data()}.find("does not match expected engine API major version"),
              std::string::npos);
    EXPECT_EQ(test_util::getLoadedPlugins(_handle).size(), 0);
}

// End-to-end regression coverage for ALMIOPEN-1782: a plugin that exports
// hipdnnEnginePluginGetEngineName must have that name surfaced verbatim by
// hipdnnGetEngineInfo_ext, which is the same query tools/ListEngines.cpp prints from.
TEST_F(IntegrationPluginLoading, PluginSuppliedEngineNameIsReportedByGetEngineInfo)
{
    const std::string pluginPath = hipdnn_tests::plugin_constants::testDefaultGoodPluginPath();
    ASSERT_NO_FATAL_FAILURE(setSingleEnginePluginPath(pluginPath));

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    const auto engines = queryReportedEngines(_handle);
    ASSERT_FALSE(engines.empty());

    const auto expectedId = hipdnn_tests::plugin_constants::engineId<GoodDefaultPlugin>();
    const auto engine
        = std::find_if(engines.begin(), engines.end(), [expectedId](const auto& candidate) {
              return candidate.engineId == expectedId;
          });

    ASSERT_NE(engine, engines.end())
        << "Engine " << hipdnn_data_sdk::utilities::formatEngineIdHex(expectedId)
        << " was not reported. Reported engines:\n"
        << describeReportedEngines(engines);

    EXPECT_EQ(engine->engineName, hipdnn_tests::plugin_constants::K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME)
        << "Reported engines:\n"
        << describeReportedEngines(engines);
}

// A plugin that exports neither hipdnnEnginePluginGetEngineName nor an EngineDetails.name, and
// whose id is absent from the static registry, falls through to the zero-padded uppercase
// hexadecimal rendering of its engine id.
TEST_F(IntegrationPluginLoading, PluginWithoutEngineNameEntryPointFallsBackToHexId)
{
    ASSERT_NO_FATAL_FAILURE(
        setSingleEnginePluginPath(hipdnn_tests::plugin_constants::testGoodPluginPath()));

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    const auto engines = queryReportedEngines(_handle);
    ASSERT_FALSE(engines.empty());

    const auto expectedId = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    const auto engine
        = std::find_if(engines.begin(), engines.end(), [expectedId](const auto& candidate) {
              return candidate.engineId == expectedId;
          });

    ASSERT_NE(engine, engines.end())
        << "Engine " << hipdnn_data_sdk::utilities::formatEngineIdHex(expectedId)
        << " was not reported. Reported engines:\n"
        << describeReportedEngines(engines);

    EXPECT_EQ(engine->engineName, "0xFFFFFFFFFFFFFFFE") << "Reported engines:\n"
                                                        << describeReportedEngines(engines);
}

// test_good_default_plugin carries a hardcoded engine id that its name deliberately does not hash
// back to, so name resolution reports the disagreement while keeping the plugin-reported id.
TEST_F(IntegrationPluginLoading, PluginSuppliedEngineNameNotMatchingIdLogsWarning)
{
    // The recorder is constructed first so that it saves, and on destruction restores, the log
    // level in force before this test touched it. The scope guard nests inside it and is destroyed
    // first, so the two restorations unwind in the order they were applied.
    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    const ScopedBackendLogCapture logCapture(
        hipdnn_test_sdk::utilities::IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
        HIPDNN_SEV_WARN,
        this);

    const std::string pluginPath = hipdnn_tests::plugin_constants::testDefaultGoodPluginPath();
    ASSERT_NO_FATAL_FAILURE(setSingleEnginePluginPath(pluginPath));

    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    const auto engines = queryReportedEngines(_handle);
    EXPECT_FALSE(engines.empty());

    const std::string expectedFragment
        = std::string("reports engine name '")
          + hipdnn_tests::plugin_constants::K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME + "' for engine ID "
          + hipdnn_data_sdk::utilities::formatEngineIdHex(
              hipdnn_tests::plugin_constants::engineId<GoodDefaultPlugin>());

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, expectedFragment))
        << "Expected a name/id disagreement warning. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}
