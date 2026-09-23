// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaimReport.hpp"

#include <algorithm>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn_integration_tests::bundle
{

SupportClaimCoverage& supportClaimCoverage()
{
    static SupportClaimCoverage s_coverage;
    return s_coverage;
}

CoverageUpdate coverageFor(const SupportObservation& observation, bool observationExpected)
{
    const bool read = observation.sidecar == SidecarState::CHECKED;

    CoverageUpdate update;
    update.queried = read;
    // Read in full, but silent about this arch/platform/case. Counted so "we checked
    // and it holds" reads differently from "we checked and nobody had said anything"
    // — the verdict tallies look the same for both, and only one of them means the
    // cell is covered.
    update.noApplicableClaim = read && !observation.hasApplicableClaim();
    // The graph never opened, so the query was impossible rather than skipped.
    update.notOpened = observationExpected && observation.sidecar == SidecarState::NOT_QUERIED;
    // NONE with observation expected means a sidecar is sitting there and nothing
    // looked at it, which is a harness bug. NOT_QUERIED is the honest case, already
    // reported where it happened.
    update.missedQuery = observationExpected && observation.sidecar == SidecarState::NONE;
    return update;
}

std::optional<HarnessComplaint> missedQueryComplaint(const CoverageUpdate& update,
                                                     std::string_view bundlePath)
{
    if(!update.missedQuery)
    {
        return std::nullopt;
    }

    return HarnessComplaint{std::string("support claims exist for ") + std::string(bundlePath)
                            + " but were never queried; enforcement would have passed "
                              "without checking them"};
}

bool verifiedNothing(const SupportClaimCoverage& coverage)
{
    // graphsReachedBody, not graphsWithClaims: the latter is seeded at registration
    // and cannot see --gtest_filter, so against it every suite that selects only
    // unclaimed bundles -- hipblaslt's ffm-quick tier, ASM SDPA's gpu-reference
    // target -- looks identical to a run whose engine never loaded.
    //
    // Nor graphsSelectedWithClaims, which is bumped before SetUp()'s skip exits: a
    // lane whose claim-bearing bundles are all arch-skipped would then go from green
    // to fatal for skipping exactly what it is configured to skip. What this guard is
    // about is enforcement reaching a body and failing to look, so it counts bodies.
    return coverage.graphsReachedBody > 0 && coverage.graphsQueried == 0;
}

bool countersAreConsistent(const SupportClaimCoverage& coverage)
{
    return coverage.graphsFound >= coverage.graphsWithClaims
           && coverage.graphsWithClaims >= coverage.graphsSelectedWithClaims
           && coverage.graphsSelectedWithClaims >= coverage.graphsReachedBody
           && coverage.graphsReachedBody >= coverage.graphsQueried + coverage.graphsNotOpened;
}

namespace
{

// Both modes print the same tallies, so the header is the only thing telling a reader
// whether the numbers below cost the run anything.
std::string_view modeLabel(ClaimMode claims)
{
    switch(claims)
    {
    case ClaimMode::ENFORCE:
        return " (ENFORCING)";
    case ClaimMode::REPORT_ONLY:
        return " (REPORT ONLY -- NOT ENFORCED)";
    default:
        // A header that misdescribes the run is worse than no header: every tally
        // below it is then read in the wrong mode. Throwing keeps a value that is not
        // a ClaimMode from being labelled as one.
        throw std::logic_error("printSupportClaimSummary: unhandled ClaimMode");
    }
}

} // namespace

void printSupportClaimSummary(const SupportClaimCoverage& coverage,
                              const SupportClaimVerdicts& verdicts,
                              ClaimMode claims,
                              std::ostream& os)
{
    const std::vector<SupportResult>& records = verdicts.all();

    if(records.empty() && coverage.graphsWithClaims == 0)
    {
        return;
    }

    const auto tally = [&records](SupportVerdict verdict) {
        return static_cast<size_t>(
            std::count_if(records.begin(), records.end(), [verdict](const SupportResult& r) {
                return r.verdict == verdict;
            }));
    };

    const size_t confirmed = tally(SupportVerdict::CLAIM_CONFIRMED);
    const size_t accepted = tally(SupportVerdict::CLAIM_ACCEPTED);
    const size_t failedInUse = tally(SupportVerdict::CLAIM_FAILED_IN_USE);
    const size_t broke = tally(SupportVerdict::CLAIM_BROKEN);
    const size_t err = tally(SupportVerdict::QUERY_ERRORED);
    const size_t unc = tally(SupportVerdict::UNCLAIMED_SUPPORT);

    os << "\n==== SUPPORT CLAIM SUMMARY" << modeLabel(claims) << " ====\n"
       << "  graphs: " << coverage.graphsFound << " found, " << coverage.graphsWithClaims
       << " with claims, " << coverage.graphsSelectedWithClaims << " selected, "
       << coverage.graphsReachedBody << " ran, " << coverage.graphsQueried << " queried ("
       << records.size() << " verdicts)\n"
       << "  confirmed: " << confirmed << "  accepted: " << accepted
       << "  failed-in-use: " << failedInUse << "  broken: " << broke << "  errored: " << err
       << "  unclaimed: " << unc << "\n"
       << "  (accepted = engine advertises support; confirmed = the run reached the "
          "depth this bundle's enforcement_level declares)\n";

    // Printed between the counters and the lines drawn from them: the numbers above
    // are the evidence, and everything below is the arithmetic that just became
    // meaningless. A reader who sees only one of the two is misled either way.
    const bool consistent = countersAreConsistent(coverage);
    if(!consistent)
    {
        os << "\n  WARNING: the counters above do not nest (each should be a subset of "
              "the one\n"
              "  before it), so the harness is miscounting and the attribution lines "
              "that would\n"
              "  follow are suppressed. The verdicts below are unaffected -- they come "
              "from the\n"
              "  claim records, not from these counters.\n";
    }

    // A graph that never opened ran and failed; it is not a graph the filter left
    // out. Subtracted before the remainder is attributed, so the filter line counts
    // only bundles that genuinely never ran.
    //
    // Printed even when the ladder is broken: this is a counter read straight out,
    // not a difference between two of them, so a miscount elsewhere cannot turn it
    // into a wrong claim about which graphs these were.
    if(coverage.graphsNotOpened > 0)
    {
        os << "  " << coverage.graphsNotOpened
           << " claim-bearing graph(s) could not be opened, so their claims could not "
              "be checked;\n"
              "  those tests are already failing on the graph itself.\n";
    }

    // Each remaining shortfall is the difference between two adjacent counters, so it
    // has exactly one cause and one remedy. Nothing here is a guess: the counters are
    // bumped at the three points a claim-bearing graph can stop -- discovery, SetUp(),
    // the test body -- and subtracting neighbours names which one it stopped at.
    //
    // All three are gated on `consistent`, because that reasoning is exactly what a
    // broken ladder invalidates: subtract counters that are not nested and the result
    // is a number of graphs that does not correspond to any set of graphs, printed
    // beside a confident sentence about what happened to them.

    // A body ran and neither queried the sidecar nor failed to open the graph. No
    // configuration produces this; it is the harness losing a query it owed, which
    // missedQueryComplaint() has already reported per-bundle.
    const size_t accountedFor = coverage.graphsQueried + coverage.graphsNotOpened;
    if(consistent && coverage.graphsReachedBody > accountedFor)
    {
        os << "  " << (coverage.graphsReachedBody - accountedFor)
           << " claim-bearing graph(s) ran without ever being queried;\n"
              "  this is a harness defect, not a configuration choice.\n";
    }

    // Selected, then stopped in SetUp(). The remedy is a skip-list edit or different
    // hardware -- never widening the filter, which already let these through.
    if(consistent && coverage.graphsSelectedWithClaims > coverage.graphsReachedBody)
    {
        os << "  " << (coverage.graphsSelectedWithClaims - coverage.graphsReachedBody)
           << " claim-bearing graph(s) were selected but skipped before running "
              "(arch guard, skip-list, or no device);\n"
              "  their claims are unenforced by this run.\n";
    }

    // Discovery counts every claim-bearing bundle on disk; only selected ones reach
    // SetUp(). The gap between the two is the filter's doing and is named as such
    // rather than left as a bare mismatch a reader has to interpret.
    if(consistent && coverage.graphsWithClaims > coverage.graphsSelectedWithClaims)
    {
        os << "  " << (coverage.graphsWithClaims - coverage.graphsSelectedWithClaims)
           << " claim-bearing graph(s) were discovered but not selected to run "
              "(--gtest_filter);\n"
              "  their claims are unenforced by this run.\n";
    }

    // Otherwise invisible: a sidecar read in full that promised nothing about this
    // arch/platform/case leaves no verdict, so the tallies above look identical to a
    // graph that was never claimed at all. On a bring-up ASIC that is usually the
    // whole tree, and it is the difference between "enforced and green" and
    // "enforced nothing here".
    if(coverage.graphsWithNoApplicableClaim > 0)
    {
        os << "  " << coverage.graphsWithNoApplicableClaim
           << " queried graph(s) carry a sidecar that claims nothing for this "
              "arch/platform;\n"
              "  nothing was promised for them, so nothing was enforced.\n";
    }

    const auto totalFailures = static_cast<size_t>(
        std::count_if(records.begin(), records.end(), [](const SupportResult& r) {
            return isFailure(r.verdict);
        }));
    if(totalFailures > 0)
    {
        os << "\n---- CLAIM FAILURES (" << totalFailures << ") ----\n";
        for(const auto& r : records)
        {
            if(!isFailure(r.verdict))
            {
                continue;
            }
            os << "  " << toString(r.verdict) << "  " << r.bundlePath << "\n"
               << "    engine=" << r.engineName << "  arch=" << r.arch
               << "  platform=" << r.platform << "\n"
               << "    " << r.detail << "\n";
            if(!r.queryMessage.empty())
            {
                os << "    query: " << r.queryMessage << "\n";
            }
        }
    }

    // Not a claim failure — the claim held and the run is already red for another
    // reason — but it is the one signal that says "do not publish this cell as
    // working support", so it gets named rather than counted.
    if(failedInUse > 0)
    {
        os << "\n---- FAILED IN USE (" << failedInUse << ") ----\n";
        for(const auto& r : records)
        {
            if(r.verdict != SupportVerdict::CLAIM_FAILED_IN_USE)
            {
                continue;
            }
            os << "  " << r.bundlePath << "\n"
               << "    engine=" << r.engineName << "  arch=" << r.arch
               << "  platform=" << r.platform << "\n"
               << "    " << r.detail << "\n";
        }
        os << "\nThe engine accepted these graphs but the test did not pass.\n";
    }

    if(unc > 0)
    {
        os << "\n---- UNCLAIMED SUPPORT (" << unc << ") ----\n";
        for(const auto& r : records)
        {
            if(r.verdict != SupportVerdict::UNCLAIMED_SUPPORT)
            {
                continue;
            }
            os << "  " << r.bundlePath << "\n"
               << "    engine=" << r.engineName << "  arch=" << r.arch
               << "  platform=" << r.platform << "\n"
               << "    " << r.detail << "\n";
        }
        os << "\nThese are supported but not recorded in a sidecar.\n";
    }
}

} // namespace hipdnn_integration_tests::bundle
