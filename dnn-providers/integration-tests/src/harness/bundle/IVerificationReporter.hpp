// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"
#include "harness/bundle/UnverifiableBundleReport.hpp"

namespace hipdnn_integration_tests::bundle
{

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
};

} // namespace hipdnn_integration_tests::bundle
