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

#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/SupportObservationLog.hpp"

namespace hipdnn_integration_tests::bundle::test_utils
{

inline std::string readFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

/// One observation for a single-graph bundle, as the harness would record it.
inline ObservedGraphSupport singleGraphObservation(const std::filesystem::path& bundleJsonPath,
                                                   const std::string& engineName,
                                                   const std::string& arch,
                                                   const std::string& platform,
                                                   bool engineIsSupported)
{
    return {singleGraphClaimLocator(bundleJsonPath), engineName, arch, platform, engineIsSupported};
}

/// One observation for a single case of a template sweep.
inline ObservedGraphSupport sweepCaseObservation(const std::filesystem::path& sweepJsonPath,
                                                 const std::string& caseId,
                                                 const std::string& engineName,
                                                 const std::string& arch,
                                                 const std::string& platform,
                                                 bool engineIsSupported)
{
    return {sweepCaseClaimLocator(sweepJsonPath, caseId),
            engineName,
            arch,
            platform,
            engineIsSupported};
}

} // namespace hipdnn_integration_tests::bundle::test_utils
