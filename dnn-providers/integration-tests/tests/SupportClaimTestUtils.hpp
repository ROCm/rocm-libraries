// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared fixtures for the support-claim tests.
//
// The observation builders are the ones that matter: they construct their
// locator with the same factories BundleRegistration uses, so a test can never
// disagree with production about which file a claim lands in. That is the one
// bug that would let every unit test here pass while a real run writes to the
// wrong path.

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/SupportObservationLog.hpp"

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

namespace hipdnn_integration_tests::bundle::test_utils
{

inline hipdnn_test_sdk::utilities::ScopedDirectory makeScopedTestDir(const std::string& prefix)
{
    auto path
        = std::filesystem::temp_directory_path()
          / (prefix + "_"
             + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
    std::filesystem::remove_all(path);
    return {path};
}

inline std::string readFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

/// One observed cell of a single-graph bundle, as the harness would record it.
inline ObservedSupportCell singleGraphObservation(const std::filesystem::path& bundleJsonPath,
                                                  const std::string& engineName,
                                                  const std::string& arch,
                                                  const std::string& platform,
                                                  ObservedSupport support)
{
    // enforcementLevel is left off the end deliberately: a trailing `{}` would
    // value-initialize it to APPLICABILITY and quietly override the struct's
    // FULL default, which is what these tests mean.
    return {singleGraphClaimLocator(bundleJsonPath), engineName, arch, platform, support};
}

/// Resolved-query shorthand: the overwhelmingly common case in these tests, and
/// the only one that existed before UNKNOWN was representable.
inline ObservedSupportCell singleGraphObservation(const std::filesystem::path& bundleJsonPath,
                                                  const std::string& engineName,
                                                  const std::string& arch,
                                                  const std::string& platform,
                                                  bool engineIsSupported)
{
    return singleGraphObservation(bundleJsonPath,
                                  engineName,
                                  arch,
                                  platform,
                                  engineIsSupported ? ObservedSupport::SUPPORTED
                                                    : ObservedSupport::DECLINED);
}

/// One observed cell of a single case of a template sweep.
inline ObservedSupportCell sweepCaseObservation(const std::filesystem::path& sweepJsonPath,
                                                const std::string& caseId,
                                                const std::string& engineName,
                                                const std::string& arch,
                                                const std::string& platform,
                                                ObservedSupport support)
{
    return {sweepCaseClaimLocator(sweepJsonPath, caseId), engineName, arch, platform, support};
}

inline ObservedSupportCell sweepCaseObservation(const std::filesystem::path& sweepJsonPath,
                                                const std::string& caseId,
                                                const std::string& engineName,
                                                const std::string& arch,
                                                const std::string& platform,
                                                bool engineIsSupported)
{
    return sweepCaseObservation(sweepJsonPath,
                                caseId,
                                engineName,
                                arch,
                                platform,
                                engineIsSupported ? ObservedSupport::SUPPORTED
                                                  : ObservedSupport::DECLINED);
}

} // namespace hipdnn_integration_tests::bundle::test_utils
