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

    /// Applies one graph's coverage update to the run counters. `missedQuery` is
    /// not published here: it is a harness bug and becomes a GTest failure instead.
    virtual void recordCoverage(const CoverageUpdate& update) = 0;
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
