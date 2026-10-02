// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iostream>
#include <string>

#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"
#include "harness/bundle/UnverifiableBundleReport.hpp"
#include "harness/bundle/VerificationOutcome.hpp"

namespace hipdnn_integration_tests::bundle
{

/// How many test bodies each oracle graded this run, for the coverage summary.
struct VerifierTally
{
    size_t golden = 0;
    size_t gpuReference = 0;
    size_t cpuReference = 0;
    size_t none = 0;

    void add(Verifier verifier)
    {
        switch(verifier)
        {
        case Verifier::GOLDEN:
            ++golden;
            return;
        case Verifier::GPU_REFERENCE:
            ++gpuReference;
            return;
        case Verifier::CPU_REFERENCE:
            ++cpuReference;
            return;
        case Verifier::NONE:
        default:
            ++none;
            return;
        }
    }

    size_t total() const
    {
        return golden + gpuReference + cpuReference + none;
    }
};

inline VerifierTally& verifierTally()
{
    static VerifierTally s_tally;
    return s_tally;
}

/// Where a test body's findings go once they are decided.
///
/// All three destinations behind it are process-wide singletons. Reached through
/// this seam, a test asserts on what the harness published instead of clearing
/// global state in SetUp and hoping no other suite wrote to it in between.
///
/// Deciding stays in the harness — this only publishes.
class IVerificationReporter
{
public:
    IVerificationReporter() = default;
    virtual ~IVerificationReporter() = default;

    IVerificationReporter(const IVerificationReporter&) = delete;
    IVerificationReporter& operator=(const IVerificationReporter&) = delete;
    IVerificationReporter(IVerificationReporter&&) = delete;
    IVerificationReporter& operator=(IVerificationReporter&&) = delete;

    /// Applies one graph's coverage update to the run counters. `missedQuery` is not
    /// published here: it is a harness bug rather than a coverage fact, and the
    /// caller raises it as a GTest failure instead.
    virtual void recordCoverage(const CoverageUpdate& update) = 0;

    /// One claim-bearing graph survived --gtest_filter. Separate from recordCoverage
    /// because it is published from SetUp(), before there is an observation to build a
    /// CoverageUpdate from -- which is the point: a graph SetUp() goes on to skip is
    /// counted here and nowhere else.
    virtual void recordSelectedWithClaims() = 0;

    /// One claim-bearing graph entered its test body. Same reason it is not part of
    /// recordCoverage: the fact is already true on the body's first line, and building
    /// it into the observation's update would lose it whenever reading the sidecar --
    /// or opening the graph, which happens first -- throws. A body that ran and threw
    /// must not be attributed to SetUp() skipping it.
    virtual void recordReachedBody() = 0;
    virtual void recordVerdict(const SupportResult& record) = 0;
    virtual void recordUnverifiable(const std::string& bundlePath, const std::string& reason) = 0;
    virtual void recordReferenceError(const std::string& bundlePath, const std::string& reason) = 0;

    /// The oracle that graded this test body's outputs, NONE when nothing was
    /// compared. Called once per verification body, including one that throws
    /// before it reaches an outcome. Support-claim authoring runs do not call it.
    virtual void recordVerifier(const std::string& bundlePath, Verifier verifier) = 0;
};

/// The production sinks: the run's coverage counters, verdict table, and
/// unverifiable-bundle report.
class GlobalVerificationReporter : public IVerificationReporter
{
public:
    void recordCoverage(const CoverageUpdate& update) override
    {
        if(update.queried)
        {
            supportClaimCoverage().graphsQueried++;
        }
        if(update.noApplicableClaim)
        {
            supportClaimCoverage().graphsWithNoApplicableClaim++;
        }
        if(update.notOpened)
        {
            supportClaimCoverage().graphsNotOpened++;
        }
    }

    void recordSelectedWithClaims() override
    {
        supportClaimCoverage().graphsSelectedWithClaims++;
    }

    void recordReachedBody() override
    {
        supportClaimCoverage().graphsReachedBody++;
    }

    void recordVerdict(const SupportResult& record) override
    {
        SupportClaimVerdicts::get().record(record);
    }

    void recordUnverifiable(const std::string& bundlePath, const std::string& reason) override
    {
        UnverifiableBundleReport::get().record(
            bundlePath, reason, UnverifiableSeverity::UNVERIFIABLE);
    }

    void recordReferenceError(const std::string& bundlePath, const std::string& reason) override
    {
        UnverifiableBundleReport::get().record(bundlePath, reason, UnverifiableSeverity::REF_ERROR);
    }

    void recordVerifier(const std::string& bundlePath, Verifier verifier) override
    {
        verifierTally().add(verifier);
        std::cout << "[ VERIFIER ] " << toString(verifier) << ": " << bundlePath << '\n';
    }
};

} // namespace hipdnn_integration_tests::bundle
