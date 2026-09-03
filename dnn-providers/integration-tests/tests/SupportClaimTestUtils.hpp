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
#include <random>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include <gtest/gtest.h>

#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/SupportObservationLog.hpp"

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

namespace hipdnn_integration_tests::bundle::test_utils
{

// A temp directory of this run's own, cleaned up when the returned handle dies.
//
// The name carries the test's line number for the case where a crash skips the
// cleanup, but uniqueness comes from the random draw and the pid: the name used
// to be the line number alone, cleared with remove_all before use, which under
// two concurrent runs deletes the other run's directory out from under it.
// ScopedDirectory refuses to adopt a directory it did not create, so a
// construction that returns is itself proof this run owns the name -- there is
// no window between finding the name free and taking it.
inline hipdnn_test_sdk::utilities::ScopedDirectory makeScopedTestDir(const std::string& prefix)
{
    static std::random_device s_entropy;
    const auto line = ::testing::UnitTest::GetInstance()->current_test_info()->line();
    const auto pid = static_cast<long>(::getpid());

    // Sixteen draws all colliding does not happen while the draws differ, so
    // exhausting this loop means the generator is stuck, not that /tmp is busy.
    // The pid keeps that from being a cross-process collision even then.
    for(int attempt = 0; attempt < 16; ++attempt)
    {
        const auto path = std::filesystem::temp_directory_path()
                          / (prefix + "_" + std::to_string(line) + "_" + std::to_string(pid) + "_"
                             + std::to_string(s_entropy()));
        try
        {
            return hipdnn_test_sdk::utilities::ScopedDirectory{path};
        }
        catch(const std::filesystem::filesystem_error&)
        {
            // EACCES, ENOENT on the temp root, ENOSPC. No redraw fixes any of
            // these, and retrying would bury the real cause behind this
            // function's exhaustion message.
            throw;
        }
        catch(const std::runtime_error&)
        {
            // ScopedDirectory's "Directory already exists": name taken, redraw.
            continue;
        }
    }

    throw std::runtime_error("makeScopedTestDir: no unique temp directory for " + prefix);
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
                                                  bool engineIsSupported)
{
    return {singleGraphClaimLocator(bundleJsonPath), engineName, arch, platform, engineIsSupported};
}

/// One observed cell of a single case of a template sweep.
inline ObservedSupportCell sweepCaseObservation(const std::filesystem::path& sweepJsonPath,
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
