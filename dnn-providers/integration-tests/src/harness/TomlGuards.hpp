// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "harness/TestConfig.hpp"
#include "harness/bundle/OutputComparison.hpp"

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

// Applies this engine's [[tolerance_overrides]] entry for `testName`, if one matches.
//
// Silent by design. A tolerance is not necessarily the check that grades a tensor — a
// [[validator_overrides]] entry outranks it — and this function cannot know: it is given
// a test, not a tensor. Logging belongs to gradingForTensor(), which knows both.
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
    return true;
}

// The validator this engine's TOML selects for one output tensor of one test, or
// nullopt when no [[validator_overrides]] entry matches it — which means allclose, the
// default and the only thing any other code path can produce.
//
// Shared by both verification harnesses: the selection is a per-engine numerical
// property, and two copies of it would be two chances to disagree about what the
// config said. `tensorLabel` must be the label form the TOML globs are written
// against — bundle::tensorLabel(uid, name), never a raw tensor name.
inline std::optional<ValidatorOverride> findTomlValidatorOverride(const std::string& testName,
                                                                  const std::string& tensorLabel)
{
    if(testName.empty())
    {
        return std::nullopt;
    }
    return TestConfig::get().findValidatorOverride(testName, tensorLabel);
}

// How one output tensor is graded, and the one place that says so out loud.
//
// The validator override is read first because it outranks atol/rtol: announcing a
// tolerance before knowing whether it survives is how a reader is told the wrong check
// ran. `atol`/`rtol` are the fallback the caller resolved. On the default allclose path
// they are logged only when a TOML entry changed them, so a quiet run means the defaults
// graded the tensor; a selected validator instead announces itself and its tolerances
// unconditionally, so a logged `atol=` there says nothing about whether an override matched.
//
// Both verification harnesses resolve every output tensor through this, so neither can
// report a check the other would not have run.
inline bundle::ComparisonTolerance gradingForTensor(const std::string& testName,
                                                    const std::string& tensorLabel,
                                                    float atol,
                                                    float rtol)
{
    if(const auto selected = findTomlValidatorOverride(testName, tensorLabel))
    {
        switch(selected->kind)
        {
        case ValidatorOverrideKind::RMS:
            HIPDNN_PLUGIN_LOG_INFO("Validator override applied for "
                                   << testName << " tensor " << tensorLabel
                                   << ": rms, threshold=" << selected->rmsThreshold);
            return bundle::ComparisonTolerance::rms(selected->rmsThreshold);

        case ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES:
            // This kind grades finite elements by atol/rtol, so a [[tolerance_overrides]]
            // entry still applies to it exactly as it does to allclose.
            applyTomlToleranceOverride(testName, atol, rtol);
            HIPDNN_PLUGIN_LOG_INFO("Validator override applied for "
                                   << testName << " tensor " << tensorLabel
                                   << ": allclose_matching_infinities, atol=" << atol
                                   << " rtol=" << rtol);
            return bundle::ComparisonTolerance::allCloseMatchingInfinities(atol, rtol);

        case ValidatorOverrideKind::ALLCLOSE:
            break; // an explicit allclose entry takes the default path below

        default:
            // A kind with no case here, i.e. a new ValidatorOverrideKind nobody wired.
            // Falling through to allclose would grade the tensor with a validator its
            // config did not ask for, silently.
            throw std::invalid_argument("gradingForTensor: unhandled ValidatorOverrideKind");
        }
    }

    if(applyTomlToleranceOverride(testName, atol, rtol))
    {
        HIPDNN_PLUGIN_LOG_INFO("Tolerance override applied for "
                               << testName << " tensor " << tensorLabel
                               << ": allclose, atol=" << atol << " rtol=" << rtol);
    }
    return bundle::ComparisonTolerance::allClose(atol, rtol);
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
