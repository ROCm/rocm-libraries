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

// The relative-RMS threshold this engine's TOML selects for one output tensor of one
// test, or nullopt when no [[validator_overrides]] entry names it — which means
// allclose, the default and the only thing any other code path can produce.
//
// Shared by both verification harnesses: the selection is a per-engine numerical
// property, and two copies of it would be two chances to disagree about what the
// config said. `tensorLabel` must be the label form the TOML globs are written
// against — bundle::tensorLabel(uid, name), never a raw tensor name.
inline std::optional<float> findTomlRmsThreshold(const std::string& testName,
                                                 const std::string& tensorLabel)
{
    if(testName.empty())
    {
        return std::nullopt;
    }
    const auto selected = TestConfig::get().findValidatorOverride(testName, tensorLabel);
    if(!selected || selected->kind != ValidatorOverrideKind::RMS)
    {
        return std::nullopt;
    }
    HIPDNN_PLUGIN_LOG_INFO("Validator override applied for "
                           << testName << " tensor " << tensorLabel
                           << ": rms, threshold=" << selected->rmsThreshold);
    return selected->rmsThreshold;
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
