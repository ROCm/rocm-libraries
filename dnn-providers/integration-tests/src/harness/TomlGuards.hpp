// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "harness/CurrentTestName.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/OutputComparison.hpp"

namespace hipdnn_integration_tests
{

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
// `tensorLabel` must be the label form the TOML globs are written against —
// bundle::tensorLabel(uid, name), never a raw tensor name.
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
// This overload reads no TestConfig state: the caller supplies both overrides.
//
// The validator override is consulted first because it outranks atol/rtol: announcing a
// tolerance before knowing whether it survives is how a reader is told the wrong check
// ran. `atol`/`rtol` are the fallback the caller resolved. On the default allclose path
// they are logged only when an override changed them, so a quiet run means the
// defaults graded the tensor; a selected validator instead announces itself and its
// tolerances unconditionally, so a logged `atol=` there says nothing about whether an
// override matched.
//
// For the two allclose kinds the tolerance is resolved in three layers, each replacing the
// one before it: the caller's `atol`/`rtol`, then the test's [[tolerance_overrides]] entry
// (both values, every tensor of the test), then the validator entry's own `atol` and `rtol`
// (each only if the entry spells it, this tensor only). A validator entry without either
// leaves the tolerance path exactly as it was.
inline bundle::ComparisonTolerance
    gradingForTensor(const std::string& testName,
                     const std::string& tensorLabel,
                     const std::optional<ValidatorOverride>& validatorOverride,
                     const std::optional<ToleranceOverride>& toleranceOverride,
                     float atol,
                     float rtol)
{
    bool matchingInfinities = false;
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
            // This kind grades finite elements by atol/rtol, so the tolerance layers
            // below apply to it exactly as they do to allclose.
            matchingInfinities = true;
            break;

        case ValidatorOverrideKind::ALLCLOSE:
            break; // an explicit allclose entry takes the default path below

        default:
            // Falling through to allclose would grade the tensor with a validator its
            // config did not ask for, silently.
            throw std::invalid_argument("gradingForTensor: unhandled ValidatorOverrideKind");
        }
    }

    if(toleranceOverride)
    {
        atol = toleranceOverride->atol;
        rtol = toleranceOverride->rtol;
    }
    const bool validatorTolerance
        = validatorOverride && (validatorOverride->atol || validatorOverride->rtol);
    if(validatorTolerance)
    {
        atol = validatorOverride->atol.value_or(atol);
        rtol = validatorOverride->rtol.value_or(rtol);
    }

    if(matchingInfinities)
    {
        HIPDNN_PLUGIN_LOG_INFO("Validator override applied for "
                               << testName << " tensor " << tensorLabel
                               << ": allclose_matching_infinities, atol=" << atol
                               << " rtol=" << rtol);
        return bundle::ComparisonTolerance::allCloseMatchingInfinities(atol, rtol);
    }

    if(toleranceOverride || validatorTolerance)
    {
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
