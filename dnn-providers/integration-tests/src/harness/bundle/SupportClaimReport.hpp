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
    // Of the claim-bearing graphs, how many survived --gtest_filter. Bumped in SetUp(),
    // the earliest hook GTest runs after the filter has already chosen this test, so
    // the gap below it is the filter's doing and the gap above it is not. Registration
    // cannot supply this: GTest only applies the filter inside RUN_ALL_TESTS(), so
    // graphsWithClaims counts every bundle on disk whether or not this run was ever
    // going to touch one.
    size_t graphsSelectedWithClaims = 0;
    // Of the selected graphs, how many reached a test body. The difference against
    // graphsSelectedWithClaims is exactly the bundles SetUp() skipped -- arch guard,
    // TOML skip-list, no device -- which is why the two are separate counters rather
    // than one bumped somewhere in between. Both are bumped from shouldObserveClaims(),
    // the same predicate registration seeds graphsWithClaims from, so the four nest and
    // countersAreConsistent() below can say so. A plugin that failed to load needs no
    // special handling here: main() exits non-zero on it before a body runs.
    size_t graphsReachedBody = 0;
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
    /// A sidecar exists and this run was meant to read it, but the query never
    /// happened. The run-level guard only fires when *no* graph anywhere was queried,
    /// so a partial gap needs its own signal.
    bool missedQuery = false;
};

// Everything here derives from `observation`, so nothing here survives the read
// throwing. graphsReachedBody deliberately does not: it is true before the read and
// is published straight to the reporter, ahead of it.
//
// `observationExpected` is the caller's shouldObserveClaims(): this graph's claims
// were this run's business, so a sidecar that went unread is a gap rather than a
// bundle the run was never going to look at.
CoverageUpdate coverageFor(const SupportObservation& observation, bool observationExpected);

/// The complaint owed for a coverage gap, or nullopt when there is none. Takes no
/// severity, and none is owed to the claim mode: a missed query is the harness failing
/// to do the one thing it was asked to, not a claim that came out badly, so it goes red
/// whether or not this run enforces.
std::optional<HarnessComplaint> missedQueryComplaint(const CoverageUpdate& update,
                                                     std::string_view bundlePath);

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
// Scoped to the claim-bearing graphs whose bodies actually ran: a suite filtered onto
// bundles that carry no claims enforced nothing because there was nothing to enforce,
// and a bundle SetUp() skipped never got as far as a query it could have made. Neither
// is enforcement failing to look, which is the only thing this guard is about.
bool verifiedNothing(const SupportClaimCoverage& coverage);

// Whether the counters still nest. Every number above is a subset of the one before
// it, so the ladder can only ever descend:
//
//     found >= withClaims >= selected >= ran >= queried + notOpened
//
// (queried and notOpened are disjoint halves of the bodies that ran: the sidecar was
// read, or the graph never opened so it could not be. A shortfall against their sum
// is the harness losing a query it owed, which the summary reports separately -- the
// invariant here is only that the sum cannot exceed the bodies it is drawn from.)
//
// Holds for a single pass only. found and withClaims are counted once at registration,
// the rest once per test execution, so --gtest_repeat breaks the ladder with nothing
// wrong.
//
// Nothing asserts this today; the relations live in the field comments above and in
// the arithmetic printSupportClaimSummary() does on them. That is the fragile part:
// a new skip path added to SetUp() without a counter to match, or two bumps keyed on
// different predicates, silently rewrites every attribution line downstream and no
// test notices. Checked rather than described, because the failure mode is a
// confident wrong answer rather than a crash.
//
// Deliberately not a hard assert: a summary is a diagnostic, and aborting the run
// that produced it helps nobody. The caller prints and suppresses the attributions
// instead, since those are the lines the broken numbers would corrupt.
bool countersAreConsistent(const SupportClaimCoverage& coverage);

// `claims` only labels the header; the body is the same either way. Taken as an
// argument rather than read from TestConfig so the summary can be rendered without a
// singleton, and so the label cannot disagree with the mode the caller ran under.
void printSupportClaimSummary(const SupportClaimCoverage& coverage,
                              const SupportClaimVerdicts& verdicts,
                              ClaimMode claims,
                              std::ostream& os);

} // namespace hipdnn_integration_tests::bundle
