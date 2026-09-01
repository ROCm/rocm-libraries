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

    const size_t sat = tally(SupportVerdict::SATISFIED);
    const size_t broke = tally(SupportVerdict::CLAIM_BROKEN);
    const size_t err = tally(SupportVerdict::QUERY_ERRORED);
    const size_t notLoaded = tally(SupportVerdict::ENGINE_NOT_LOADED);
    const size_t unc = tally(SupportVerdict::UNCLAIMED_SUPPORT);
    const size_t notEnf = tally(SupportVerdict::NOT_ENFORCED);

    os << "\n==== SUPPORT CLAIM SUMMARY ====\n"
       << "  graphs: " << coverage.graphsFound << " found, " << coverage.graphsWithClaims
       << " with claims, " << coverage.graphsQueried << " queried (" << records.size()
       << " verdicts)\n"
       << "  satisfied: " << sat << "  broken: " << broke << "  errored: " << err
       << "  not-loaded: " << notLoaded << "  unclaimed: " << unc << "  not-enforced: " << notEnf
       << "\n";

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
