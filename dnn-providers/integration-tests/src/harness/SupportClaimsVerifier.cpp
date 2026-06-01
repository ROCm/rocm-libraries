// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/SupportClaimsVerifier.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include "harness/SupportMatrixCollector.hpp"

namespace hipdnn_integration_tests
{

namespace
{

enum class TestOutcome
{
    UNKNOWN,
    PASSED,
    FAILED,
    SKIPPED,
};

TestOutcome testOutcomeFor(const std::string& testName)
{
    auto* unitTest = ::testing::UnitTest::GetInstance();
    if(unitTest == nullptr)
    {
        return TestOutcome::UNKNOWN;
    }
    const auto dotPos = testName.find('.');
    if(dotPos == std::string::npos)
    {
        return TestOutcome::UNKNOWN;
    }
    const std::string suiteName = testName.substr(0, dotPos);
    const std::string caseName = testName.substr(dotPos + 1);
    for(int s = 0; s < unitTest->total_test_suite_count(); ++s)
    {
        const auto* suite = unitTest->GetTestSuite(s);
        if(suite == nullptr || suite->name() != suiteName)
        {
            continue;
        }
        for(int t = 0; t < suite->total_test_count(); ++t)
        {
            const auto* info = suite->GetTestInfo(t);
            if(info == nullptr || info->name() != caseName)
            {
                continue;
            }
            const auto* result = info->result();
            if(result == nullptr)
            {
                return TestOutcome::UNKNOWN;
            }
            if(result->Skipped())
            {
                return TestOutcome::SKIPPED;
            }
            if(result->Failed())
            {
                return TestOutcome::FAILED;
            }
            if(result->Passed())
            {
                return TestOutcome::PASSED;
            }
            return TestOutcome::UNKNOWN;
        }
    }
    return TestOutcome::UNKNOWN;
}

// Format a record's full dtype signature for display in the
// named-field schema:
//   {io=fp16, compute=fp32}                 (symmetric, no intermediate)
//   {io=fp16, output=fp32, compute=fp32}    (asymmetric)
//   {io=fp16, compute=fp32, intermediate=fp32}
// Matches the dtype_combos inline-table entry an engineer would edit
// in the sidecar.
std::string formatDtypeForDisplay(const GraphSupportRecord& record)
{
    std::string s = "{io=" + record.ioDtype;
    if(!record.outputDtype.empty() && record.outputDtype != record.ioDtype)
    {
        s += ", output=" + record.outputDtype;
    }
    s += ", compute=" + record.computeDtype;
    if(!record.intermediateDtype.empty())
    {
        s += ", intermediate=" + record.intermediateDtype;
    }
    s += "}";
    return s;
}

SupportClaimFinding buildRuleA(const GraphSupportRecord& record, const SupportMatcher& matcher)
{
    std::ostringstream body;
    body << "  CLAIM BROKEN (Rule A):\n"
         << "    " << record.testName << "\n"
         << "      observed: op_chain=\"" << record.opChain << "\"\n"
         << "                " << "dtype_combo" << "=\"" << formatDtypeForDisplay(record)
         << "\" layout=\"" << record.layout << "\"\n"
         << "      claim source: " << matcher.sourceLocation << "\n"
         << "      engine returned no support for this graph\n"
         << "      Action: narrow op_chains/io_dtypes/layouts to exclude this tuple, "
            "add a [[test_skips]]\n"
         << "              if it's broken-but-supported, or fix the engine.\n"
         << "      See docs/support-claims-failures.md" << support_claim_anchors::RULE_A << "\n";
    return SupportClaimFinding{
        SupportClaimFinding::Rule::A_CLAIM_BROKEN,
        SupportClaimFinding::Severity::FAIL,
        body.str(),
    };
}

SupportClaimFinding buildRuleB(const std::string& testName, const ::testing::TestResult& result)
{
    std::ostringstream body;
    body << "  ISSUE BEFORE THE TEST RUNS (Rule B):\n"
         << "    " << testName << "\n"
         << "      status: " << (result.Failed() ? "FAILED" : "INCOMPLETE")
         << ", no SupportMatrixCollector record — verifyGraph() did not record\n"
         << "      Action: fix the underlying SetUp()/test failure first. The verifier "
            "cannot evaluate\n"
         << "              support claims for tests that don't reach the record point.\n"
         << "      See docs/support-claims-failures.md" << support_claim_anchors::RULE_B << "\n";
    return SupportClaimFinding{
        SupportClaimFinding::Rule::B_ISSUE_BEFORE_TEST,
        SupportClaimFinding::Severity::FAIL,
        body.str(),
    };
}

SupportClaimFinding
    buildRuleC(const SupportMatcher& matcher, const SupportBlock& block, bool fullCiMode)
{
    const auto severity
        = fullCiMode ? SupportClaimFinding::Severity::FAIL : SupportClaimFinding::Severity::NOTE;
    std::ostringstream body;
    body << "  ZERO-COVERAGE MATCHER (Rule C, " << (fullCiMode ? "FAIL" : "note — partial run")
         << "):\n"
         << "    " << matcher.sourceLocation << "\n"
         << "      arch=" << block.arch
         << " platform=" << (block.platform.has_value() ? *block.platform : "any") << "\n"
         << "      op_chains[0] = \"" << (matcher.opChains.empty() ? "" : matcher.opChains.front())
         << "\"  (and " << (matcher.opChains.empty() ? 0 : matcher.opChains.size() - 1)
         << " more) -- 0 observed tests in this matcher's cross-product\n"
         << "      Action: regenerate with --write-support-claims, or hand-edit to remove "
            "the stale matcher.\n"
         << "      See docs/support-claims-failures.md" << support_claim_anchors::RULE_C << "\n";
    return SupportClaimFinding{
        SupportClaimFinding::Rule::C_ZERO_COVERAGE_MATCHER,
        severity,
        body.str(),
    };
}

SupportClaimFinding buildRuleD(const GraphSupportRecord& record, const std::string& engineName)
{
    std::ostringstream body;
    body << "  ENGINE OVER-CLAIM (Rule D, note on existing test failure):\n"
         << "    " << record.testName << "\n"
         << "      test FAILED; engine '" << engineName
         << "' returned support; no matcher covers this graph.\n"
         << "      observed: op_chain=\"" << record.opChain << "\" " << "dtype_combo" << "=\""
         << formatDtypeForDisplay(record) << "\" layout=\"" << record.layout << "\"\n"
         << "      Action: tighten the engine's get_ranked_engine_ids logic, or add a "
            "[[test_skips]]\n"
         << "              entry with a reason if the engine should claim then skip.\n"
         << "      See docs/support-claims-failures.md" << support_claim_anchors::RULE_D << "\n";
    return SupportClaimFinding{
        SupportClaimFinding::Rule::D_ENGINE_OVER_CLAIM,
        SupportClaimFinding::Severity::NOTE,
        body.str(),
    };
}

SupportClaimFinding buildRuleE(const GraphSupportRecord& record,
                               const std::string& engineName,
                               const std::string& archToken)
{
    std::ostringstream body;
    body << "  UNCLAIMED GAIN (Rule E, warning):\n"
         << "    " << record.testName << "\n"
         << "      observed: (\"" << record.opChain << "\",\"" << formatDtypeForDisplay(record)
         << "\",\"" << record.layout << "\"); engine '" << engineName
         << "' returned support; no matcher covers it.\n"
         << "      Action: if this support is intentional, add the tuple to a [[supported]] "
            "block for arch=\""
         << archToken << "\" — easiest path is to regenerate with --write-support-claims.\n"
         << "      See docs/support-claims-failures.md" << support_claim_anchors::RULE_E << "\n";
    return SupportClaimFinding{
        SupportClaimFinding::Rule::E_UNCLAIMED_GAIN,
        SupportClaimFinding::Severity::WARNING,
        body.str(),
    };
}

size_t countSeverity(const std::vector<SupportClaimFinding>& findings,
                     SupportClaimFinding::Severity severity)
{
    return static_cast<size_t>(
        std::count_if(findings.begin(), findings.end(), [&](const SupportClaimFinding& finding) {
            return finding.severity == severity;
        }));
}

void emitFindings(const std::vector<SupportClaimFinding>& findings,
                  const std::string& archToken,
                  const std::string& platform,
                  const std::string& engineName,
                  const std::string& artifactPath)
{
    const auto fails = countSeverity(findings, SupportClaimFinding::Severity::FAIL);
    const auto notes = countSeverity(findings, SupportClaimFinding::Severity::NOTE);
    const auto warns = countSeverity(findings, SupportClaimFinding::Severity::WARNING);

    std::ostringstream header;
    header << "[SUPPORT CLAIMS] arch=" << archToken << " platform=" << platform
           << " engine=" << engineName << ": " << fails << " failure" << (fails == 1 ? "" : "s")
           << ", " << notes << " note" << (notes == 1 ? "" : "s") << ", " << warns << " warning"
           << (warns == 1 ? "" : "s") << ".";

    std::cerr << "\n" << header.str() << "\n\n";

    std::ofstream artifact;
    if(!artifactPath.empty())
    {
        artifact.open(artifactPath);
        if(artifact.is_open())
        {
            artifact << header.str() << "\n\n";
        }
    }

    for(const auto& finding : findings)
    {
        std::cerr << finding.body << "\n";
        if(artifact.is_open())
        {
            artifact << finding.body << "\n";
        }
    }

    if(artifact.is_open())
    {
        artifact.close();
        std::cerr << "[SUPPORT CLAIMS] long-form findings written to: " << artifactPath << "\n";
    }
}

} // namespace

SupportClaimsVerifier::SupportClaimsVerifier(const SupportClaims& claims,
                                             std::string engineName,
                                             std::string archToken,
                                             std::string platform,
                                             bool fullCiMode,
                                             std::string artifactPath)
    : _claims(claims)
    , _engineName(std::move(engineName))
    , _archToken(std::move(archToken))
    , _platform(std::move(platform))
    , _fullCiMode(fullCiMode)
    , _artifactPath(std::move(artifactPath))
{
}

bool SupportClaimsVerifier::runAndReport()
{
    const auto records = SupportMatrixCollector::get().getRecords();
    const auto harnessTests = SupportMatrixCollector::get().getHarnessTests();

    // Pre-index records by test name for O(1) "did this test record?"
    // lookups in Rule B. A test that records >1 time (e.g. multiple
    // verifyGraph calls in one TEST) keeps all entries — every
    // observation is still subject to Rule A.
    std::map<std::string, std::vector<const GraphSupportRecord*>> byTest;
    for(const auto& record : records)
    {
        byTest[record.testName].push_back(&record);
    }

    const auto* block = _claims.blockFor(_archToken, _platform);

    // Rule A: claimed observation with empty engine support.
    // Rule D: failed test + engine returned support + no matcher.
    // Rule E: passing test + engine returned support + no matcher.
    for(const auto& record : records)
    {
        if(record.opChain.empty())
        {
            // Legacy call site that bypassed describeGraphStructured.
            // Can't evaluate against matchers — skip silently so old
            // tests don't block bring-up.
            continue;
        }

        const bool engineSupports
            = record.supportingEngines.find(_engineName) != record.supportingEngines.end();
        // Find the matching matcher (not just whether one exists) so the
        // Rule A failure can point at the exact [[supported.matchers]]
        // entry the engineer needs to edit.
        const SupportMatcher* matchingMatcher = nullptr;
        if(block != nullptr)
        {
            for(const auto& matcher : block->matchers)
            {
                if(matcher.contains(record.opChain,
                                    record.ioDtype,
                                    record.outputDtype,
                                    record.computeDtype,
                                    record.intermediateDtype,
                                    record.layout))
                {
                    matchingMatcher = &matcher;
                    break;
                }
            }
        }
        const bool isClaimed = matchingMatcher != nullptr;

        if(isClaimed && !engineSupports)
        {
            _findings.push_back(buildRuleA(record, *matchingMatcher));
            continue;
        }

        if(!isClaimed && engineSupports)
        {
            const auto outcome = testOutcomeFor(record.testName);
            if(outcome == TestOutcome::FAILED)
            {
                _findings.push_back(buildRuleD(record, _engineName));
            }
            else if(outcome == TestOutcome::PASSED)
            {
                _findings.push_back(buildRuleE(record, _engineName, _archToken));
            }
            // SKIPPED: don't surface — the test wasn't actually run, so
            // the support signal is unreliable.
        }
    }

    // Rule B: harness-registered, non-skipped test with zero records.
    // Iterate the gtest test info to learn each test's final status —
    // the harness registry only tells us SetUp() was entered.
    if(auto* unitTest = ::testing::UnitTest::GetInstance(); unitTest != nullptr)
    {
        for(int s = 0; s < unitTest->total_test_suite_count(); ++s)
        {
            const auto* suite = unitTest->GetTestSuite(s);
            if(suite == nullptr)
            {
                continue;
            }
            for(int t = 0; t < suite->total_test_count(); ++t)
            {
                const auto* info = suite->GetTestInfo(t);
                if(info == nullptr || !info->should_run())
                {
                    continue;
                }
                const auto* testResult = info->result();
                if(testResult == nullptr || testResult->Skipped())
                {
                    continue;
                }
                const std::string testName
                    = std::string(info->test_suite_name()) + "." + info->name();
                if(harnessTests.find(testName) == harnessTests.end())
                {
                    // Not a verifyGraph-based test — utility tests
                    // (no recorded observation) are deliberately
                    // outside Rule B's scope.
                    continue;
                }
                if(byTest.find(testName) != byTest.end())
                {
                    continue;
                }
                _findings.push_back(buildRuleB(testName, *testResult));
            }
        }
    }

    // Rule C: every matcher in the active block should cover ≥1
    // observation. In full CI runs missing coverage is a hard error;
    // partial runs downgrade to a note because the missing tuple may
    // simply have been filtered out.
    if(block != nullptr)
    {
        // Observed 6-tuples: (opChain, io, output, compute, intermediate,
        // layout). output is normalized to io for symmetric records so
        // forEachTuple's visitor compares against one canonical form.
        std::set<
            std::
                tuple<std::string, std::string, std::string, std::string, std::string, std::string>>
            observedTuples;
        for(const auto& record : records)
        {
            if(record.opChain.empty())
            {
                continue;
            }
            const std::string outDtype
                = record.outputDtype.empty() ? record.ioDtype : record.outputDtype;
            observedTuples.emplace(record.opChain,
                                   record.ioDtype,
                                   outDtype,
                                   record.computeDtype,
                                   record.intermediateDtype,
                                   record.layout);
        }

        for(const auto& matcher : block->matchers)
        {
            bool covered = false;
            matcher.forEachTuple([&](const std::string& op,
                                     const std::string& io,
                                     const std::string& out,
                                     const std::string& compute,
                                     const std::string& intermediate,
                                     const std::string& layout) {
                if(observedTuples.find({op, io, out, compute, intermediate, layout})
                   != observedTuples.end())
                {
                    covered = true;
                    return false;
                }
                return true;
            });
            if(!covered)
            {
                _findings.push_back(buildRuleC(matcher, *block, _fullCiMode));
            }
        }
    }

    emitFindings(_findings, _archToken, _platform, _engineName, _artifactPath);
    return countSeverity(_findings, SupportClaimFinding::Severity::FAIL) == 0;
}

bool detectFullCiMode()
{
    // Use the project's cross-platform getEnv (MSVC mode errors on
    // std::getenv via -Werror -Wdeprecated-declarations). getEnv returns
    // an empty string for unset env vars, which collapses the
    // unset-vs-empty distinction we don't care about here.
    const auto envFilter = hipdnn_data_sdk::utilities::getEnv("GTEST_FILTER");
    const auto envTotalShards = hipdnn_data_sdk::utilities::getEnv("GTEST_TOTAL_SHARDS");
    const auto envShardIndex = hipdnn_data_sdk::utilities::getEnv("GTEST_SHARD_INDEX");
    // gtest also exposes the active filter via the flag once
    // InitGoogleTest has parsed argv. The default "*" means "all tests"
    // and is treated as a non-filter.
    const std::string flagFilter = ::testing::GTEST_FLAG(filter);
    const bool filterIsTrivial = flagFilter.empty() || flagFilter == "*";
    const bool envFilterUnset = envFilter.empty();
    const bool envShardUnset
        = (envTotalShards.empty() || envTotalShards == "1") && envShardIndex.empty();
    return filterIsTrivial && envFilterUnset && envShardUnset;
}

} // namespace hipdnn_integration_tests
