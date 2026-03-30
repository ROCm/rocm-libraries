// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/PluginPaths.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

namespace
{
// GMock matcher: verifies that a const char* const* array contains the expected strings
MATCHER_P2(pathArrayMatches, expectedPaths, count, "")
{
    if(arg == nullptr)
    {
        return count == 0;
    }
    for(size_t i = 0; i < count; ++i)
    {
        if(std::strcmp(arg[i], expectedPaths[i].c_str()) != 0)
        {
            *result_listener << "path[" << i << "] is \"" << arg[i] << "\", expected \""
                             << expectedPaths[i] << "\"";
            return false;
        }
    }
    return true;
}
} // namespace

class TestSetEnginePluginPaths : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, getLastErrorString(_, _))
            .WillByDefault([](char* errorString, size_t size) {
                const std::string fakeError = "Fake backend error";
                hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
                    errorString, fakeError.c_str(), size - 1);
            });
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestSetEnginePluginPaths, SetEnginePluginPathsAbsoluteSuccess)
{
    const std::vector<std::filesystem::path> paths
        = {"/path/to/plugin_a", "/path/to/plugin_b", "/path/to/plugins"};

    // Expected strings after std::filesystem::path conversion
    std::vector<std::string> expectedStrings;
    expectedStrings.reserve(paths.size());
    for(const auto& p : paths)
    {
        expectedStrings.push_back(p.string());
    }

    EXPECT_CALL(*_mockBackend,
                setEnginePluginPathsExt(paths.size(),
                                        pathArrayMatches(expectedStrings, paths.size()),
                                        HIPDNN_PLUGIN_LOADING_ABSOLUTE))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto error = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ABSOLUTE);
    EXPECT_TRUE(error.is_good());
}

TEST_F(TestSetEnginePluginPaths, SetEnginePluginPathsAdditiveSuccess)
{
    const std::vector<std::filesystem::path> paths
        = {"/path/to/plugin_a", "/path/to/plugin_b", "/path/to/plugins"};

    // Expected strings after std::filesystem::path conversion
    std::vector<std::string> expectedStrings;
    expectedStrings.reserve(paths.size());
    for(const auto& p : paths)
    {
        expectedStrings.push_back(p.string());
    }

    EXPECT_CALL(*_mockBackend,
                setEnginePluginPathsExt(paths.size(),
                                        pathArrayMatches(expectedStrings, paths.size()),
                                        HIPDNN_PLUGIN_LOADING_ADDITIVE))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto error = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ADDITIVE);
    EXPECT_TRUE(error.is_good());
}

TEST_F(TestSetEnginePluginPaths, SetEnginePluginPathsAdditiveBackendFailure)
{
    const std::array<const char*, 2> paths = {"/some/path1", "/some/path2"};

    const std::vector<std::string> expectedStrings = {"/some/path1", "/some/path2"};

    EXPECT_CALL(*_mockBackend,
                setEnginePluginPathsExt(paths.size(),
                                        pathArrayMatches(expectedStrings, paths.size()),
                                        HIPDNN_PLUGIN_LOADING_ADDITIVE))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    auto error = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ADDITIVE);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestSetEnginePluginPaths, SetEnginePluginPathsAdditiveEmptyPaths)
{
    const std::vector<std::filesystem::path> paths = {};

    EXPECT_CALL(*_mockBackend, setEnginePluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto error = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ADDITIVE);
    EXPECT_TRUE(error.is_good());
}

TEST_F(TestSetEnginePluginPaths, SetEnginePluginPathsInvalidMode)
{
    const std::vector<std::filesystem::path> paths = {"/path/to/plugin"};

    // Cast an invalid integer to PluginLoadingMode to simulate an unrecognized value
    auto invalidMode = static_cast<PluginLoadingMode>(99);

    auto error = setEnginePluginPaths(paths, invalidMode);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), ErrorCode::INVALID_VALUE);
}

TEST_F(TestSetEnginePluginPaths, GetLoadedEnginePluginPathsSuccess)
{
    auto handle = reinterpret_cast<hipdnnHandle_t>(1);
    const std::vector<std::string> expectedPaths
        = {"/opt/plugins/engine_a.so", "/opt/plugins/engine_b.so"};
    const size_t numPlugins = expectedPaths.size();

    // Determine the max string length (including null terminator)
    size_t maxLen = 0;
    for(const auto& p : expectedPaths)
    {
        maxLen = std::max(maxLen, p.size() + 1);
    }

    // Query call: pluginPaths is nullptr, set numPluginPaths and maxStringLen
    EXPECT_CALL(*_mockBackend, getLoadedEnginePluginPathsExt(handle, _, nullptr, _))
        .WillOnce(DoAll(
            SetArgPointee<1>(numPlugins), SetArgPointee<3>(maxLen), Return(HIPDNN_STATUS_SUCCESS)));

    // Retrieve call: pluginPaths is non-null, populate buffers with path strings
    EXPECT_CALL(*_mockBackend, getLoadedEnginePluginPathsExt(handle, _, Ne(nullptr), _))
        .WillOnce(Invoke([&](hipdnnHandle_t /*handle*/,
                             size_t* /*numPluginPaths*/,
                             char** pluginPaths,
                             size_t* /*maxStringLen*/) {
            for(size_t i = 0; i < numPlugins; ++i)
            {
                hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
                    pluginPaths[i], expectedPaths[i].c_str(), maxLen);
            }
            return HIPDNN_STATUS_SUCCESS;
        }));

    std::vector<std::filesystem::path> resultPaths;
    auto error = getLoadedEnginePluginPaths(handle, resultPaths);
    EXPECT_TRUE(error.is_good());
    ASSERT_EQ(resultPaths.size(), numPlugins);
    for(size_t i = 0; i < numPlugins; ++i)
    {
        EXPECT_EQ(resultPaths[i].string(), expectedPaths[i]);
    }
}

TEST_F(TestSetEnginePluginPaths, GetLoadedEnginePluginPathsBackendFailure)
{
    auto handle = reinterpret_cast<hipdnnHandle_t>(1);

    // Query call fails with HIPDNN_STATUS_INTERNAL_ERROR
    EXPECT_CALL(*_mockBackend, getLoadedEnginePluginPathsExt(handle, _, nullptr, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<std::filesystem::path> resultPaths;
    auto error = getLoadedEnginePluginPaths(handle, resultPaths);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestSetEnginePluginPaths, GetLoadedEnginePluginPathsEmptyPlugins)
{
    auto handle = reinterpret_cast<hipdnnHandle_t>(1);

    // Query call returns zero plugins
    EXPECT_CALL(*_mockBackend, getLoadedEnginePluginPathsExt(handle, _, nullptr, _))
        .WillOnce(DoAll(SetArgPointee<1>(size_t{0}),
                        SetArgPointee<3>(size_t{0}),
                        Return(HIPDNN_STATUS_SUCCESS)));

    std::vector<std::filesystem::path> resultPaths;
    auto error = getLoadedEnginePluginPaths(handle, resultPaths);
    EXPECT_TRUE(error.is_good());
    EXPECT_TRUE(resultPaths.empty());
}
