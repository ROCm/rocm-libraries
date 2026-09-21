// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string_view>
#include <vector>

#include "harness/bundle/HarnessPolicy.hpp"
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
    // Of the claim-bearing graphs, how many a test body actually reached. Seeded at
    // run time, unlike graphsWithClaims above, and that distinction is the whole
    // reason it exists: registration is blind to --gtest_filter (GTest only applies
    // the filter inside RUN_ALL_TESTS()), so graphsWithClaims counts every bundle on
    // disk whether or not this run was ever going to touch it. Anything that divides
    // a run-time numerator by that registration-time denominator reads a legitimately
    // narrowed suite as a run that verified nothing. Bumped from the sidecar's
    // presence on disk and *not* from shouldObserveClaims(), which would go false in
    // exactly the case the guard exists to catch (engine plugin failed to load).
    size_t graphsSelectedWithClaims = 0;
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
    bool selectedWithClaims = false; ///< bump graphsSelectedWithClaims
    bool noApplicableClaim = false; ///< bump graphsWithNoApplicableClaim
    bool notOpened = false; ///< bump graphsNotOpened
    /// A sidecar exists and claim checking is on, but the query never happened. The
    /// run-level guard only fires when *no* graph anywhere was queried, so a partial
    /// gap needs its own signal. Under enforcement this fails the individual test;
    /// under report mode the caller demotes it to a warning.
    bool missedQuery = false;
};

// `observationExpected` is shouldObserveClaims() -- the observe predicate, not the
// enforce one, because report mode has to arrive at the same counters enforcement
// would or it cannot predict it.
//
// `carriesSidecar` is the weaker fact that a sidecar is sitting next to this bundle,
// implying nothing about the engine or the mode. That is why it, and not
// observationExpected, is what verifiedNothing() counts against.
CoverageUpdate coverageFor(const SupportObservation& observation,
                           bool observationExpected,
                           bool carriesSidecar);

/// The complaint owed for a coverage gap, or nullopt when there is none. `fatal` is
/// the caller's enforce predicate; the wording is the same either way, so a CI log
/// reader greps one string whichever mode produced it.
std::optional<HarnessComplaint>
    missedQueryComplaint(const CoverageUpdate& update, std::string_view bundlePath, bool fatal);

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
// Scoped to the claim-bearing graphs this run actually selected: a suite filtered
// onto bundles that carry no claims enforced nothing because there was nothing to
// enforce, which is not the same thing as enforcement failing to look.
bool verifiedNothing(const SupportClaimCoverage& coverage);

// `claims` only labels the header. Report and enforce produce byte-identical bodies,
// CLAIM FAILURES block included, and differ solely in the exit code -- so a scraped
// log showing failures next to a green lane is unreadable without the label.
void printSupportClaimSummary(const SupportClaimCoverage& coverage,
                              const SupportClaimVerdicts& verdicts,
                              ClaimMode claims,
                              std::ostream& os);

} // namespace hipdnn_integration_tests::bundle
