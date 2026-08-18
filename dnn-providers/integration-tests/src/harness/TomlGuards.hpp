// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests
{

inline std::string currentTestName()
{
    auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    if(info == nullptr)
    {
        return {};
    }
    return std::string(info->test_suite_name()) + "." + info->name();
}

inline bool applyTomlToleranceOverride(const std::string& testName, float& atol, float& rtol)
{
    if(testName.empty())
    {
        return false;
    }
    auto ovr = TestConfig::get().findToleranceOverride(testName);
    if(!ovr)
    {
        return false;
    }
    atol = ovr->atol;
    rtol = ovr->rtol;
    HIPDNN_PLUGIN_LOG_INFO("Tolerance override applied for " << testName << ": atol=" << atol
                                                             << " rtol=" << rtol);
    return true;
}

inline std::optional<std::string> checkTomlSkip(const std::string& testName)
{
    if(testName.empty())
    {
        return std::nullopt;
    }
    return TestConfig::get().findSkipForTest(testName);
}

} // namespace hipdnn_integration_tests
