// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include "TestUtil.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include <HipdnnBackendAttributeName.h>
#include <HipdnnBackendAttributeType.h>
#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <test_plugins/TestPluginEngineIdMap.hpp>

#include <filesystem>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

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
