// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaimReport.hpp"

#include <algorithm>
#include <ostream>
#include <vector>

namespace hipdnn_integration_tests::bundle
{

SupportClaimCoverage& supportClaimCoverage()
{
    static SupportClaimCoverage s_coverage;
    return s_coverage;
}

bool verifiedNothing(const SupportClaimCoverage& coverage)
{
    return coverage.graphsWithClaims > 0 && coverage.graphsQueried == 0;
}

void printSupportClaimSummary(const SupportClaimCoverage& coverage,
                              const SupportClaimVerdicts& verdicts,
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

    os << "\n==== SUPPORT CLAIM SUMMARY ====\n"
       << "  graphs: " << coverage.graphsFound << " found, " << coverage.graphsWithClaims
       << " with claims, " << coverage.graphsQueried << " queried (" << records.size()
       << " verdicts)\n"
       << "  confirmed: " << confirmed << "  accepted: " << accepted
       << "  failed-in-use: " << failedInUse << "  broken: " << broke << "  errored: " << err
       << "  unclaimed: " << unc << "\n"
       << "  (accepted = engine advertises support; only confirmed was executed and "
          "verified)\n";

    // Discovery counts every claim-bearing bundle; only selected tests run. A
    // selected one cannot go unqueried — the harness fails the test if its sidecar
    // was never read — so the whole remainder is attributable to the filter and is
    // named as such rather than left as a bare mismatch a reader has to interpret.
    if(coverage.graphsWithClaims > coverage.graphsQueried)
    {
        os << "  " << (coverage.graphsWithClaims - coverage.graphsQueried)
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
        os << "\n---- ACCEPTED BUT UNCONFIRMED (" << failedInUse << ") ----\n";
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
               << "  platform=" << r.platform << "\n";
        }
        os << "\nThese are supported but not recorded in a sidecar.\n";
    }
}

} // namespace hipdnn_integration_tests::bundle
