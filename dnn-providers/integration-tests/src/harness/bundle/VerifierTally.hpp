// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iostream>
#include <mutex>
#include <string>

#include <gtest/gtest.h>

#include "harness/bundle/VerificationOutcome.hpp"

namespace hipdnn_integration_tests::bundle
{

// Process-wide count of which oracle graded each verification body, printed as
// "Verified by" in the TEST COVERAGE SUMMARY once RUN_ALL_TESTS() returns.
// GlobalVerificationReporter::recordVerifier() is its only production writer.
//
// It counts one gtest iteration. With --gtest_repeat=N the gtest counts printed
// beside it are those of the last iteration, so main() installs
// VerifierTallyIterationReset to clear the tally as each iteration starts.
class VerifierTally
{
public:
    struct Counts
    {
        size_t golden = 0;
        size_t gpuReference = 0;
        size_t cpuReference = 0;
        size_t none = 0;

        size_t total() const
        {
            return golden + gpuReference + cpuReference + none;
        }
    };

    static VerifierTally& get()
    {
        static VerifierTally s_instance;
        return s_instance;
    }

    VerifierTally(const VerifierTally&) = delete;
    VerifierTally& operator=(const VerifierTally&) = delete;
    VerifierTally(VerifierTally&&) = delete;
    VerifierTally& operator=(VerifierTally&&) = delete;

    // Counts `verifier` and writes the "[ VERIFIER ] <verifier>: <bundle>" line
    // that sits between a test's [ RUN ] and result lines.
    void record(const std::string& bundlePath, Verifier verifier, std::ostream& os = std::cout)
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            switch(verifier)
            {
            case Verifier::GOLDEN:
                ++_counts.golden;
                break;
            case Verifier::GPU_REFERENCE:
                ++_counts.gpuReference;
                break;
            case Verifier::CPU_REFERENCE:
                ++_counts.cpuReference;
                break;
            case Verifier::NONE:
            default:
                ++_counts.none;
                break;
            }
        }
        os << "[ VERIFIER ] " << toString(verifier) << ": " << bundlePath << '\n';
    }

    Counts counts() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _counts;
    }

    void reset()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _counts = {};
    }

private:
    VerifierTally() = default;

    mutable std::mutex _mutex;
    Counts _counts;
};

// Clears VerifierTally as each gtest iteration starts, so the "Verified by" line
// covers the same iteration as the Passed/Skipped/Failed counts printed with it.
class VerifierTallyIterationReset : public ::testing::EmptyTestEventListener
{
public:
    void OnTestIterationStart(const ::testing::UnitTest& /*unitTest*/, int /*iteration*/) override
    {
        VerifierTally::get().reset();
    }
};

} // namespace hipdnn_integration_tests::bundle
