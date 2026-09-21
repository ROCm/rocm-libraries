// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <string>
#include <vector>

#include <hipdnn_backend.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

namespace hipdnn_tests
{

/// Base test fixture for hipDNN integration tests.
///
/// Provides:
/// - Device availability check via SKIP_IF_NO_DEVICES()
/// - HIP runtime initialization
/// - Plugin path loading (defaults to testGoodPluginPath())
/// - hipdnnHandle_t creation and cleanup
///
/// Tests needing custom plugin paths should override getPluginPaths().
class IntegrationTestFixture : public ::testing::Test
{
protected:
    // NOTE: derived SetUp() overrides must check IsSkipped() after calling the base — GTEST_SKIP() does not unwind into the derived frame.
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        auto pluginPaths = getPluginPaths();
        std::vector<const char*> pathPtrs;
        pathPtrs.reserve(pluginPaths.size());
        for(const auto& p : pluginPaths)
        {
            pathPtrs.push_back(p.c_str());
        }
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      pathPtrs.size(), pathPtrs.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
        }
    }

    /// Override to provide custom plugin paths. Default loads testGoodPluginPath().
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    virtual std::vector<std::string> getPluginPaths() const
    {
        return {hipdnn_tests::plugin_constants::testGoodPluginPath()};
    }

    hipdnnHandle_t _handle = nullptr;
};

} // namespace hipdnn_tests
