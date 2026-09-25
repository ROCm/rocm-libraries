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

// The [[tolerance_overrides]] entry this engine's TOML selects for `testName`, or
// nullopt when none matches.
inline std::optional<ToleranceOverride> findTomlToleranceOverride(const std::string& testName)
{
    if(testName.empty())
    {
        return std::nullopt;
    }
    return TestConfig::get().findToleranceOverride(testName);
}

// Applies this engine's [[tolerance_overrides]] entry for `testName`, if one matches.
//
// Silent by design. A tolerance is not necessarily the check that grades a tensor — a
// [[validator_overrides]] entry outranks it — and this function cannot know: it is given
// a test, not a tensor. Logging belongs to gradingForTensor(), which knows both.
inline bool applyTomlToleranceOverride(const std::string& testName, float& atol, float& rtol)
{
    const auto ovr = findTomlToleranceOverride(testName);
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
// `validatorOverride` and `toleranceOverride` are the [[validator_overrides]] entry the
// config selects for this tensor and the [[tolerance_overrides]] entry it selects for this
// test, or nullopt where none matches; this overload reads no TestConfig state.
//
// The validator override is consulted first because it outranks atol/rtol: announcing a
// tolerance before knowing whether it survives is how a reader is told the wrong check
// ran. `atol`/`rtol` are the fallback the caller resolved. On the default allclose path
// they are logged only when a tolerance override changed them, so a quiet run means the
// defaults graded the tensor; a selected validator instead announces itself and its
// tolerances unconditionally, so a logged `atol=` there says nothing about whether a
// tolerance override matched.
inline bundle::ComparisonTolerance
    gradingForTensor(const std::string& testName,
                     const std::string& tensorLabel,
                     const std::optional<ValidatorOverride>& validatorOverride,
                     const std::optional<ToleranceOverride>& toleranceOverride,
                     float atol,
                     float rtol)
{
    if(validatorOverride)
    {
        switch(validatorOverride->kind)
        {
        case ValidatorOverrideKind::RMS:
            HIPDNN_PLUGIN_LOG_INFO("Validator override applied for "
                                   << testName << " tensor " << tensorLabel
                                   << ": rms, threshold=" << validatorOverride->rmsThreshold);
            return bundle::ComparisonTolerance::rms(validatorOverride->rmsThreshold);

        case ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES:
            // This kind grades finite elements by atol/rtol, so a [[tolerance_overrides]]
            // entry still applies to it exactly as it does to allclose.
            if(toleranceOverride)
            {
                atol = toleranceOverride->atol;
                rtol = toleranceOverride->rtol;
            }
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

    if(toleranceOverride)
    {
        atol = toleranceOverride->atol;
        rtol = toleranceOverride->rtol;
        HIPDNN_PLUGIN_LOG_INFO("Tolerance override applied for "
                               << testName << " tensor " << tensorLabel
                               << ": allclose, atol=" << atol << " rtol=" << rtol);
    }
    return bundle::ComparisonTolerance::allClose(atol, rtol);
}

// gradingForTensor with the overrides looked up in this engine's TOML config.
//
// Both verification harnesses resolve every output tensor through this, so neither can
// report a check the other would not have run.
inline bundle::ComparisonTolerance gradingForTensor(const std::string& testName,
                                                    const std::string& tensorLabel,
                                                    float atol,
                                                    float rtol)
{
    return gradingForTensor(testName,
                            tensorLabel,
                            findTomlValidatorOverride(testName, tensorLabel),
                            findTomlToleranceOverride(testName),
                            atol,
                            rtol);
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
