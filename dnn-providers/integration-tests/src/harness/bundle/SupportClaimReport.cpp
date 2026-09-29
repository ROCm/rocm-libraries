// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/SupportClaimReport.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <hipdnn_frontend/Error.hpp>
#include <nlohmann/json.hpp>

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

std::string reportBundlePath(std::string_view recordedPath,
                             std::string_view caseId,
                             const std::filesystem::path& bundleRoot)
{
    std::string path(recordedPath);

    const std::string caseSuffix = "#" + std::string(caseId);
    if(!caseId.empty() && path.size() > caseSuffix.size()
       && path.compare(path.size() - caseSuffix.size(), caseSuffix.size(), caseSuffix) == 0)
    {
        path.erase(path.size() - caseSuffix.size());
    }

    if(bundleRoot.empty())
    {
        return path;
    }

    // "dir/" normalizes to a path with an empty last component, whose filename() is
    // empty; drop it so the prefix below is the folder's name.
    std::filesystem::path root = bundleRoot.lexically_normal();
    if(!root.has_filename())
    {
        root = root.parent_path();
    }

    const auto relative = std::filesystem::path(path).lexically_normal().lexically_relative(root);
    if(relative.empty() || *relative.begin() == ".." || *relative.begin() == ".")
    {
        return path;
    }
    return (root.filename() / relative).generic_string();
}

namespace
{

struct ModeNames
{
    const char* header;
    const char* key;
};

// Both modes report the same numbers, so this is the only thing telling a reader
// whether they cost the run anything.
ModeNames modeNames(ClaimMode claims)
{
    switch(claims)
    {
    case ClaimMode::ENFORCE:
        return {"ENFORCING", "enforcing"};
    case ClaimMode::WARN:
        return {"WARNING ONLY -- NOT ENFORCED", "warning_only"};
    default:
        // A summary that misdescribes the run is worse than none: every number in it
        // is then read in the wrong mode. Throwing keeps a value that is not a
        // ClaimMode from being labelled as one.
        throw std::logic_error("buildSupportClaimSummary: unhandled ClaimMode");
    }
}

// Where the verdict points. Engine, arch and platform are written only when they
// differ from the "run" block: one lane tests one engine on one machine, so repeating
// them on every entry is noise, and an entry that does differ is worth noticing.
nlohmann::json locate(const SupportResult& r, const SupportClaimRunContext& run)
{
    nlohmann::json entry;
    entry["bundle"] = reportBundlePath(r.bundlePath, r.caseId, run.bundleRoot);
    if(!r.caseId.empty())
    {
        entry["case"] = r.caseId;
    }
    if(r.engineName != run.engine)
    {
        entry["engine"] = r.engineName;
    }
    if(r.arch != run.arch)
    {
        entry["arch"] = r.arch;
    }
    if(r.platform != run.platform)
    {
        entry["platform"] = r.platform;
    }
    return entry;
}

void addDepths(nlohmann::json& entry, const SupportResult& r)
{
    if(r.reachedDepth.has_value())
    {
        entry["reached"] = toString(*r.reachedDepth);
    }
    if(r.requiredDepth.has_value())
    {
        entry["required"] = toString(*r.requiredDepth);
    }
}

// By bundle, then by everything else. Comparing the entries alone would not do it:
// JSON objects compare key by key in key order, so an entry carrying an "arch" would
// sort ahead of every entry that does not, whatever its bundle.
void sortEntries(nlohmann::json& list)
{
    std::sort(list.begin(), list.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
        return std::tie(a.at("bundle"), a) < std::tie(b.at("bundle"), b);
    });
}

} // namespace

std::optional<nlohmann::json> buildSupportClaimSummary(const SupportClaimCoverage& coverage,
                                                       const SupportClaimVerdicts& verdicts,
                                                       ClaimMode claims,
                                                       const SupportClaimRunContext& run)
{
    const std::vector<SupportResult>& records = verdicts.all();

    if(records.empty() && coverage.graphsWithClaims == 0)
    {
        return std::nullopt;
    }

    nlohmann::json summary;
    summary["schema_version"] = 1;
    summary["mode"] = modeNames(claims).key;
    summary["run"] = {{"engine", run.engine}, {"arch", run.arch}, {"platform", run.platform}};

    // Graphs, not verdicts: a graph checked against several engines is still one
    // graph queried.
    summary["graphs"] = {{"found", coverage.graphsFound},
                         {"with_claims", coverage.graphsWithClaims},
                         {"selected", coverage.graphsSelectedWithClaims},
                         {"ran", coverage.graphsReachedBody},
                         {"queried", coverage.graphsQueried}};

    summary["verdicts"] = {{"confirmed", verdicts.count(SupportVerdict::CLAIM_CONFIRMED)},
                           {"accepted", verdicts.count(SupportVerdict::CLAIM_ACCEPTED)},
                           {"failed_in_use", verdicts.count(SupportVerdict::CLAIM_FAILED_IN_USE)},
                           {"broken", verdicts.count(SupportVerdict::CLAIM_BROKEN)},
                           {"errored", verdicts.count(SupportVerdict::QUERY_ERRORED)},
                           {"unclaimed", verdicts.count(SupportVerdict::UNCLAIMED_SUPPORT)}};

    const bool consistent = countersAreConsistent(coverage);
    summary["counters_consistent"] = consistent;

    // Claim-bearing graphs whose claims this run did not check, by the reason why.
    nlohmann::json unenforced = nlohmann::json::object();

    // A graph that never opened ran and failed; it is not a graph the filter left out.
    // Reported even when the ladder is broken: it is a counter read straight out, not
    // a difference between two of them, so a miscount elsewhere cannot make it wrong.
    unenforced["not_opened"] = coverage.graphsNotOpened;

    // Otherwise invisible: a sidecar read in full that promised nothing about this
    // arch/platform/case leaves no verdict, so the tallies look identical to a graph
    // that was never claimed at all. On a bring-up ASIC that is usually the whole
    // tree, and it is the difference between "enforced and green" and "enforced
    // nothing here".
    unenforced["no_applicable_claim"] = coverage.graphsWithNoApplicableClaim;

    nlohmann::json harnessDefects = nlohmann::json::object();

    // Each value below is the difference between two adjacent counters, so it has
    // exactly one cause and one remedy: the counters are bumped at the three points a
    // claim-bearing graph can stop -- discovery, SetUp(), the test body -- and
    // subtracting neighbours names which one it stopped at.
    //
    // Left out when the ladder is broken, because that reasoning is exactly what a
    // broken ladder invalidates: subtract counters that are not nested and the result
    // is a number of graphs that does not correspond to any set of graphs.
    if(consistent)
    {
        // Discovery counts every claim-bearing bundle on disk; only the ones
        // --gtest_filter selected reach SetUp().
        unenforced["not_selected"] = coverage.graphsWithClaims - coverage.graphsSelectedWithClaims;

        // Selected, then stopped in SetUp() -- arch guard, skip-list, or no device. The
        // remedy is a skip-list edit or different hardware, never widening the filter,
        // which already let these through.
        unenforced["skipped_before_run"]
            = coverage.graphsSelectedWithClaims - coverage.graphsReachedBody;

        // A body ran and neither queried the sidecar nor failed to open the graph. No
        // configuration produces this; it is the harness losing a query it owed, which
        // missedQueryComplaint() has already reported per bundle.
        harnessDefects["missed_query"]
            = coverage.graphsReachedBody - (coverage.graphsQueried + coverage.graphsNotOpened);
    }

    summary["unenforced"] = std::move(unenforced);
    summary["harness_defects"] = std::move(harnessDefects);

    nlohmann::json claimFailures = nlohmann::json::array();
    nlohmann::json failedInUse = nlohmann::json::array();

    // Unclaimed support is grouped: a sweep the engine takes whole would otherwise
    // repeat one bundle once per case, and what a reader acts on is the bundle.
    std::map<nlohmann::json, std::vector<std::string>> unclaimed;

    for(const auto& r : records)
    {
        if(isFailure(r.verdict))
        {
            nlohmann::json entry = locate(r, run);
            entry["verdict"] = toString(r.verdict);
            entry["reason"] = r.detail;
            if(r.queryStatus != hipdnn_frontend::ErrorCode::OK)
            {
                entry["status"] = hipdnn_frontend::to_string(r.queryStatus);
            }
            if(!r.queryMessage.empty())
            {
                entry["query_message"] = r.queryMessage;
            }
            claimFailures.push_back(std::move(entry));
        }
        else if(r.verdict == SupportVerdict::CLAIM_FAILED_IN_USE)
        {
            // Not a claim failure -- the claim held and the run is already red for
            // another reason -- but it is the one signal that says "do not publish
            // this cell as working support", so it gets named rather than counted.
            nlohmann::json entry = locate(r, run);
            addDepths(entry, r);
            entry["reason"] = r.detail;
            failedInUse.push_back(std::move(entry));
        }
        else if(r.verdict == SupportVerdict::UNCLAIMED_SUPPORT)
        {
            nlohmann::json key = locate(r, run);
            key.erase("case");
            addDepths(key, r);
            auto& cases = unclaimed[key];
            if(!r.caseId.empty())
            {
                cases.push_back(r.caseId);
            }
        }
    }

    nlohmann::json unclaimedSupport = nlohmann::json::array();
    for(auto& [key, cases] : unclaimed)
    {
        nlohmann::json entry = key;
        if(!cases.empty())
        {
            std::sort(cases.begin(), cases.end());
            entry["cases"] = cases;
        }
        unclaimedSupport.push_back(std::move(entry));
    }

    sortEntries(claimFailures);
    sortEntries(failedInUse);
    sortEntries(unclaimedSupport);

    summary["claim_failures"] = std::move(claimFailures);
    summary["failed_in_use"] = std::move(failedInUse);
    summary["unclaimed_support"] = std::move(unclaimedSupport);
    return summary;
}

void printSupportClaimSummary(const SupportClaimCoverage& coverage,
                              const SupportClaimVerdicts& verdicts,
                              ClaimMode claims,
                              const SupportClaimRunContext& run,
                              std::ostream& os)
{
    const auto summary = buildSupportClaimSummary(coverage, verdicts, claims, run);
    if(!summary.has_value())
    {
        return;
    }

    // Wrapped under one named key so the block still says what it is once someone
    // has cut it out of a CI log.
    const nlohmann::json document = {{"support_claim_summary", *summary}};

    // A query message is the backend's own text and nothing promises it is UTF-8;
    // replacing a bad byte beats throwing out of the last thing the run prints.
    os << "\n==== SUPPORT CLAIM SUMMARY (" << modeNames(claims).header << ") ====\n"
       << document.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
}

} // namespace hipdnn_integration_tests::bundle
