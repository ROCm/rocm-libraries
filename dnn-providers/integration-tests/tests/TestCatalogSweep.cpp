// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The loop that decides what a sweep runs, filters and reports, driven through
// ICatalogGraphRunner with a fake: no device, no graph. What these pin is the accounting
// -- no entry an engine returns goes unreported, and the graph fails when it should.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness/catalog-sweep/CatalogSweep.hpp"

using namespace hipdnn_integration_tests::catalog_sweep;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr const char* K_GRAPH = "quick_Pointwise.Small";

CatalogEntry entry(std::int64_t tile)
{
    return {{{"tile", tile}}};
}

EntryOutcome passed()
{
    return {};
}

EntryOutcome stoppedAt(EntryStage stage, std::string message = "boom")
{
    EntryOutcome outcome;
    outcome.failedStage = stage;
    outcome.message = std::move(message);
    return outcome;
}

EntryOutcome mismatched()
{
    EntryOutcome outcome;
    outcome.outputMismatch = true;
    outcome.maxAbsErr = 3.0;
    return outcome;
}

class FakeRunner final : public ICatalogGraphRunner
{
public:
    ReferenceResult reference;
    EngineList engines;
    std::map<std::int64_t, CatalogFetch> catalogs;
    std::map<std::pair<std::int64_t, std::size_t>, EntryOutcome> outcomes;
    std::set<std::pair<std::int64_t, std::size_t>> throwing;

    int referenceCalls = 0;
    int listCalls = 0;
    std::vector<std::pair<std::int64_t, std::size_t>> entriesRun;

    void addEngine(std::int64_t id, const std::string& name, std::vector<CatalogEntry> entries)
    {
        engines.engines.push_back({id, name});
        CatalogFetch fetch;
        fetch.entries = std::move(entries);
        catalogs[id] = std::move(fetch);
    }

    void addEngineWithoutCatalog(std::int64_t id, const std::string& name, const std::string& why)
    {
        engines.engines.push_back({id, name});
        CatalogFetch fetch;
        fetch.ok = false;
        fetch.error = why;
        catalogs[id] = std::move(fetch);
    }

    ReferenceResult prepareReference() override
    {
        ++referenceCalls;
        return reference;
    }

    EngineList listEngines() override
    {
        ++listCalls;
        return engines;
    }

    CatalogFetch fetchCatalog(const CatalogEngine& engine) override
    {
        return catalogs.at(engine.id);
    }

    EntryOutcome runEntry(const CatalogEngine& engine, std::size_t entryIndex) override
    {
        entriesRun.emplace_back(engine.id, entryIndex);
        if(throwing.count({engine.id, entryIndex}) != 0)
        {
            throw std::runtime_error("device lost");
        }
        const auto it = outcomes.find({engine.id, entryIndex});
        return it == outcomes.end() ? passed() : it->second;
    }
};

struct Swept
{
    GraphSweepResult result;
    std::vector<nlohmann::json> records;
    std::string log;
};

Swept sweep(FakeRunner& runner,
            const CatalogFilter* filter = nullptr,
            std::optional<std::string> pinnedEngine = std::nullopt,
            bool failOnUnsupported = false)
{
    Swept swept;
    GraphSweepRequest request;
    request.graph = K_GRAPH;
    request.filter = filter;
    request.pinnedEngine = std::move(pinnedEngine);
    request.failOnUnsupported = failOnUnsupported;
    request.context = {"gfx1151", "1.0"};

    std::ostringstream log;
    swept.result = sweepGraph(
        runner,
        request,
        [&](const nlohmann::json& record) { swept.records.push_back(record); },
        log);
    swept.log = log.str();
    return swept;
}

std::vector<std::string> statuses(const Swept& swept)
{
    std::vector<std::string> result;
    for(const auto& record : swept.records)
    {
        if(!record.contains("kind"))
        {
            result.push_back(record.at("status").get<std::string>());
        }
    }
    return result;
}

} // namespace

TEST(TestCatalogSweepLoop, EveryReturnedEntryIsReportedOnceInCatalogOrder)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8), entry(16), entry(32)});
    runner.addEngine(2, "ENGINE_B", {entry(4)});
    runner.outcomes[{1, 1}] = mismatched();
    runner.outcomes[{2, 0}] = stoppedAt(EntryStage::BUILD_PLANS);

    const auto swept = sweep(runner);

    EXPECT_EQ(statuses(swept),
              (std::vector<std::string>{"PASS", "FAIL_NUMERIC", "PASS", "FAIL_BUILD"}));
    EXPECT_EQ(swept.records[0].at("engine_name"), "ENGINE_A");
    EXPECT_EQ(swept.records[0].at("entry_index"), 0);
    EXPECT_EQ(swept.records[1].at("entry_index"), 1);
    EXPECT_EQ(swept.records[1].at("knobs")[0].at("value"), 16);
    EXPECT_EQ(swept.records[3].at("engine_name"), "ENGINE_B");

    EXPECT_EQ(swept.result.tally.returned, 4U);
    EXPECT_EQ(swept.result.tally.passed, 2U);
    EXPECT_EQ(swept.result.tally.failed(), 2U);
    EXPECT_TRUE(swept.result.tally.reconciles());
    EXPECT_TRUE(swept.result.failed());
}

TEST(TestCatalogSweepLoop, ASummaryRecordClosesTheGraphAndMatchesThePrintedLine)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8), entry(16)});

    const auto swept = sweep(runner);

    ASSERT_EQ(swept.records.size(), 3U);
    const auto& summary = swept.records.back();
    EXPECT_EQ(summary.at("kind"), "reconcile");
    EXPECT_EQ(summary.at("entries"), 2);
    EXPECT_EQ(summary.at("passed"), 2);
    const auto line = summary.at("line").get<std::string>();
    EXPECT_NE(swept.log.find(line), std::string::npos) << "the line recorded is the line printed";
    EXPECT_NE(line.find("entries=2 = passed=2 + failed=0 + filtered=0"), std::string::npos) << line;
    EXPECT_FALSE(swept.result.failed());
}

TEST(TestCatalogSweepLoop, AnEntryThatThrowsIsReportedAndTheRestStillRun)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8), entry(16), entry(32)});
    runner.throwing.insert({1, 1});

    const auto swept = sweep(runner);

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"PASS", "FAIL_EXECUTE", "PASS"}));
    EXPECT_NE(swept.records[1].at("message").get<std::string>().find("device lost"),
              std::string::npos);
    EXPECT_TRUE(swept.result.tally.reconciles());
}

TEST(TestCatalogSweepLoop, AnEngineWithoutACatalogIsRecordedOnceAndFailsTheGraph)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});
    runner.addEngineWithoutCatalog(2, "ENGINE_B", "engine has knobs but no catalog");

    const auto swept = sweep(runner);

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"PASS", "NO_CATALOG"}));
    EXPECT_EQ(swept.records[1].at("engine_name"), "ENGINE_B");
    EXPECT_NE(swept.records[1].at("message").get<std::string>().find("no catalog"),
              std::string::npos);
    EXPECT_EQ(swept.result.tally.noCatalog, 1U);
    EXPECT_TRUE(swept.result.tally.reconciles()) << "the engine counts as one returned slot";
    EXPECT_TRUE(swept.result.failed());
}

TEST(TestCatalogSweepLoop, AnEmptyCatalogIsAMissingCatalogNotASilentPass)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {});

    const auto swept = sweep(runner);

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"NO_CATALOG"}));
    EXPECT_TRUE(swept.result.failed());
    EXPECT_TRUE(runner.entriesRun.empty());
}

TEST(TestCatalogSweepLoop, AFilteredEngineIsEnumeratedAndReportedButNeverRun)
{
    FakeRunner runner;
    runner.addEngine(1, "ROCKE_X", {entry(8), entry(16)});
    runner.addEngine(2, "MIOPEN_ENGINE", {entry(4), entry(5), entry(6)});
    const auto filter = CatalogFilter::parse("engine=ROCKE*");

    const auto swept = sweep(runner, &filter);

    EXPECT_EQ(statuses(swept),
              (std::vector<std::string>{"PASS", "PASS", "FILTERED", "FILTERED", "FILTERED"}));
    EXPECT_EQ(swept.records[2].at("knobs")[0].at("value"), 4) << "a filtered entry keeps its knobs";
    for(const auto& ran : runner.entriesRun)
    {
        EXPECT_EQ(ran.first, 1) << "an entry of the filtered engine was executed";
    }
    EXPECT_EQ(swept.result.tally.filtered, 3U);
    EXPECT_EQ(swept.result.tally.returned, 5U);
    EXPECT_TRUE(swept.result.tally.reconciles());
    EXPECT_FALSE(swept.result.failed());
}

TEST(TestCatalogSweepLoop, AGraphTheFilterExcludesNeedsNeitherReferenceNorExecution)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8), entry(16)});
    const auto filter = CatalogFilter::parse("graph~Matmul");

    const auto swept = sweep(runner, &filter);

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"FILTERED", "FILTERED"}));
    EXPECT_EQ(runner.referenceCalls, 0);
    EXPECT_TRUE(runner.entriesRun.empty());
    EXPECT_FALSE(swept.result.failed());
}

// The filter must not turn an engine the user excluded into a failure of the graph.
TEST(TestCatalogSweepLoop, AFilteredEngineWithoutACatalogIsFilteredNotMissing)
{
    FakeRunner runner;
    runner.addEngine(1, "ROCKE_X", {entry(8)});
    runner.addEngineWithoutCatalog(2, "ASM_SDPA", "not supported");
    const auto filter = CatalogFilter::parse("engine=ROCKE*");

    const auto swept = sweep(runner, &filter);

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"PASS", "FILTERED"}));
    EXPECT_EQ(swept.result.tally.noCatalog, 0U);
    EXPECT_FALSE(swept.result.failed());
}

TEST(TestCatalogSweepLoop, APinnedEngineIsTheOnlyOneSwept)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});
    runner.addEngine(2, "ENGINE_B", {entry(4), entry(5)});

    const auto swept = sweep(runner, nullptr, std::string("ENGINE_B"));

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"PASS", "PASS"}));
    EXPECT_EQ(swept.records[0].at("engine_name"), "ENGINE_B");
    EXPECT_EQ(swept.result.tally.returned, 2U) << "the engine pinned away is not part of the run";
}

TEST(TestCatalogSweepLoop, AGraphThePinnedEngineDoesNotListIsSkippedNotFailed)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});

    const auto swept = sweep(runner, nullptr, std::string("ENGINE_B"));

    EXPECT_EQ(swept.result.outcome, GraphOutcome::SKIPPED);
    EXPECT_FALSE(swept.result.failed());
    EXPECT_TRUE(runner.entriesRun.empty());
    ASSERT_EQ(swept.records.size(), 1U) << "the skip is still on the record";
    EXPECT_EQ(swept.records[0].at("graph_outcome"), "SKIPPED");
}

TEST(TestCatalogSweepLoop, AGraphNoReferenceCanRunIsSkippedOnceItsEnginesAreKnown)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});
    runner.reference = {ReferenceResult::Status::UNAVAILABLE, "no reference can run this graph"};

    const auto swept = sweep(runner);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::SKIPPED);
    EXPECT_FALSE(swept.result.failed());
    EXPECT_EQ(runner.listCalls, 1);
    EXPECT_TRUE(runner.entriesRun.empty());
    EXPECT_EQ(swept.records.back().at("message"), "no reference can run this graph");
}

// A reference problem is only a problem for a graph some engine serves. A graph nothing
// serves must be skipped, and must not even pay for a reference.
TEST(TestCatalogSweepLoop, AGraphNoEngineSupportsNeverRunsOrFailsOnTheReference)
{
    FakeRunner runner;
    runner.engines.ok = false;
    runner.engines.notSupported = true;
    runner.reference = {ReferenceResult::Status::FAILED, "CpuFpReferenceSdpa::backward: lse rank"};

    const auto swept = sweep(runner);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::SKIPPED);
    EXPECT_FALSE(swept.result.failed());
    EXPECT_EQ(runner.referenceCalls, 0);
}

TEST(TestCatalogSweepLoop, AGraphThePinnedEngineDoesNotListNeverRunsOrFailsOnTheReference)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});
    runner.reference = {ReferenceResult::Status::FAILED, "reference threw"};

    const auto swept = sweep(runner, nullptr, std::string("ENGINE_B"));

    EXPECT_EQ(swept.result.outcome, GraphOutcome::SKIPPED);
    EXPECT_EQ(runner.referenceCalls, 0);
}

TEST(TestCatalogSweepLoop, AGraphEveryEngineOfWhichTheFilterExcludesNeedsNoReference)
{
    FakeRunner runner;
    runner.addEngine(1, "MIOPEN_ENGINE", {entry(8)});
    runner.reference = {ReferenceResult::Status::FAILED, "reference threw"};
    const auto filter = CatalogFilter::parse("engine=ROCKE*");

    const auto swept = sweep(runner, &filter);

    EXPECT_EQ(statuses(swept), (std::vector<std::string>{"FILTERED"}));
    EXPECT_EQ(runner.referenceCalls, 0);
    EXPECT_FALSE(swept.result.failed());
}

TEST(TestCatalogSweepLoop, AReferenceThatBrokeFailsTheGraph)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});
    runner.reference = {ReferenceResult::Status::FAILED, "reference threw"};

    const auto swept = sweep(runner);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::FAILED);
    EXPECT_TRUE(swept.result.failed());
    EXPECT_EQ(runner.referenceCalls, 1) << "a graph an engine serves does need its reference";
    EXPECT_TRUE(runner.entriesRun.empty());
}

TEST(TestCatalogSweepLoop, TheReferenceRunsOncePerGraphNotOncePerEntry)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8), entry(16)});
    runner.addEngine(2, "ENGINE_B", {entry(4), entry(5)});

    sweep(runner);

    EXPECT_EQ(runner.referenceCalls, 1);
    EXPECT_EQ(runner.listCalls, 1);
}

TEST(TestCatalogSweepLoop, AnEngineListingThatFailedFailsTheGraph)
{
    FakeRunner runner;
    runner.engines.ok = false;
    runner.engines.error = "graph not built";

    const auto swept = sweep(runner);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::FAILED);
    EXPECT_NE(swept.result.message.find("graph not built"), std::string::npos);
}

TEST(TestCatalogSweepLoop, AGraphNoEngineSupportsIsSkippedByDefault)
{
    FakeRunner runner;
    runner.engines.ok = false;
    runner.engines.notSupported = true;
    runner.engines.error = "No engine configurations available for the graph.";

    const auto swept = sweep(runner);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::SKIPPED);
    EXPECT_FALSE(swept.result.failed());
    EXPECT_EQ(swept.result.message, "no engine supports this graph");
    EXPECT_EQ(swept.result.tally.returned, 0U);
    ASSERT_EQ(swept.records.size(), 1U) << "the skip is still on the record";
    EXPECT_EQ(swept.records[0].at("kind"), "reconcile");
    EXPECT_EQ(swept.records[0].at("graph_outcome"), "SKIPPED");
    EXPECT_EQ(swept.records[0].at("entries"), 0);
    EXPECT_TRUE(runner.entriesRun.empty());
}

TEST(TestCatalogSweepLoop, AGraphNoEngineSupportsFailsWithFailOnUnsupported)
{
    FakeRunner runner;
    runner.engines.ok = false;
    runner.engines.notSupported = true;
    runner.engines.error = "No engine configurations available for the graph.";

    const auto swept = sweep(runner, nullptr, std::nullopt, /*failOnUnsupported=*/true);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::FAILED);
    EXPECT_TRUE(swept.result.failed());
    EXPECT_EQ(swept.records.back().at("graph_outcome"), "FAILED");
}

TEST(TestCatalogSweepLoop, APinnedEngineThatDoesNotListTheGraphFailsWithFailOnUnsupported)
{
    FakeRunner runner;
    runner.addEngine(1, "ENGINE_A", {entry(8)});

    const auto swept = sweep(runner, nullptr, std::string("ENGINE_B"), /*failOnUnsupported=*/true);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::FAILED);
    EXPECT_TRUE(runner.entriesRun.empty());
}

// Only the "not supported" answer is a skip: any other listing failure is a defect and
// stays a failure whatever the flag says.
TEST(TestCatalogSweepLoop, AnyOtherListingFailureStaysAFailureEvenWithoutFailOnUnsupported)
{
    FakeRunner runner;
    runner.engines.ok = false;
    runner.engines.notSupported = false;
    runner.engines.error = "backend error";

    const auto swept = sweep(runner, nullptr, std::nullopt, /*failOnUnsupported=*/false);

    EXPECT_EQ(swept.result.outcome, GraphOutcome::FAILED);
    EXPECT_TRUE(swept.result.failed());
}

// NOLINTEND(readability-identifier-naming)
