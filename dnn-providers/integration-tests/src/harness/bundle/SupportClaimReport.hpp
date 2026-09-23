// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <vector>

#include "harness/bundle/SupportVerdict.hpp"

namespace hipdnn_integration_tests::bundle
{

// Single-threaded by construction: registration finishes before the first test body,
// and GTest runs bodies sequentially. Deliberately not locked — if this ever goes
// parallel, give each worker its own copy and sum them, don't add a mutex.
struct SupportClaimCoverage
{
    size_t graphsFound = 0; // seeded by registration
    size_t graphsWithClaims = 0; // seeded by registration
    // Bumped once per graph whose sidecar was read, from SupportObservation::sidecar
    // — never from the verdict count. A sidecar naming only engines this build does
    // not load leaves no verdicts and must still count.
    size_t graphsQueried = 0;
    // Of those queried, how many carried a sidecar that promised nothing about the
    // arch/platform (or sweep case) this run is on. Not a failure — but it is the
    // difference between "this cell is claimed and holds" and "nobody ever said",
    // which the verdict counts alone cannot show.
    size_t graphsWithNoApplicableClaim = 0;
    // Claim-bearing graphs whose graph never opened, so the query was impossible
    // rather than skipped. Counted apart from graphsQueried because they are the
    // one shortfall the summary must not attribute to --gtest_filter: the test ran,
    // and it is already failing on the graph itself.
    size_t graphsNotOpened = 0;
};

// Process-wide because the harness reaches this from inside a test body built by a
// registration-time factory lambda, so there is no seam to inject it through.
SupportClaimCoverage& supportClaimCoverage();

// What one graph's observation does to the coverage counters, and whether it is a
// harness bug. Separated from the counters themselves so the rules are testable
// without the process-wide singleton below.
struct CoverageUpdate
{
    bool queried = false; ///< bump graphsQueried
    bool noApplicableClaim = false; ///< bump graphsWithNoApplicableClaim
    bool notOpened = false; ///< bump graphsNotOpened
    /// A sidecar exists and enforcement is on, but the query never happened. The
    /// run-level guard only fires when *no* graph anywhere was queried, so a partial
    /// gap needs its own signal; this one fails the individual test.
    bool missedQuery = false;
};

// `enforcementExpected` is the harness's shouldEnforceClaims(): a sidecar exists,
// enforcement is on, and an engine was named to decide against.
CoverageUpdate coverageFor(const SupportObservation& observation, bool enforcementExpected);

class SupportClaimVerdicts
{
public:
    static SupportClaimVerdicts& get()
    {
        static SupportClaimVerdicts s_instance;
        return s_instance;
    }

    SupportClaimVerdicts(const SupportClaimVerdicts&) = delete;
    SupportClaimVerdicts& operator=(const SupportClaimVerdicts&) = delete;
    SupportClaimVerdicts(SupportClaimVerdicts&&) = delete;
    SupportClaimVerdicts& operator=(SupportClaimVerdicts&&) = delete;

    void record(const SupportResult& result)
    {
        _records.push_back(result);
    }

    const std::vector<SupportResult>& all() const
    {
        return _records;
    }

    size_t count(SupportVerdict verdict) const
    {
        return static_cast<size_t>(
            std::count_if(_records.begin(), _records.end(), [verdict](const SupportResult& r) {
                return r.verdict == verdict;
            }));
    }

    bool hasFailures() const
    {
        return std::any_of(_records.begin(), _records.end(), [](const SupportResult& r) {
            return isFailure(r.verdict);
        });
    }

    size_t total() const
    {
        return _records.size();
    }

    void clear()
    {
        _records.clear();
    }

private:
    SupportClaimVerdicts() = default;

    std::vector<SupportResult> _records;
};

// Enforcement that passed having queried nothing is a lie, not a pass (RFC 0015 §7.2).
bool verifiedNothing(const SupportClaimCoverage& coverage);

void printSupportClaimSummary(const SupportClaimCoverage& coverage,
                              const SupportClaimVerdicts& verdicts,
                              std::ostream& os);

} // namespace hipdnn_integration_tests::bundle
