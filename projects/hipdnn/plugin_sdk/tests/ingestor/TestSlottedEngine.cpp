// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestSlottedEngine.cpp
 * @brief GenericEngine over the synthetic three-slot engine (prep / dkdv / dq, RFC 0021): the
 *        slot-program plan's launch order and workspace contract, workspace sizing and limit,
 *        `slot.field` knobs, product enumeration, and predictions.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using namespace hipdnn_plugin_sdk::ingestor::testing::slotted;
using hipdnn_plugin_sdk::HipdnnPluginException;

using Settings = std::vector<std::pair<std::string, int64_t>>;
using Kernels = std::vector<std::vector<SlotKernel>>;

constexpr size_t SHARED_ALIGNED = 1024;

std::string idOf(uint8_t seed)
{
    return toString(testId(seed));
}

std::string composite(uint8_t prep, uint8_t dkdv, uint8_t dq)
{
    return "prep=" + idOf(prep) + "|dkdv=" + idOf(dkdv) + "|dq=" + idOf(dq);
}

/// The config of @p settings, kept alive with its buffer.
class Config
{
public:
    explicit Config(const Settings& settings = {})
        : _wrapper(makeIntKnobsEngineConfig(_builder, settings))
    {
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& get() const
    {
        return _wrapper;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper _wrapper;
};

StubContext
    build(SlottedEngineFixture& fixture, const TestGraph& graph, const Settings& settings = {})
{
    const Config config(settings);
    return fixture.planFor(graph, config.get());
}

void executeWith(const StubContext& context, void* workspace)
{
    const StubHandle handle;
    context.plan().execute(handle, nullptr, 0, workspace);
}

/// The status and text of the exception @p action throws.
template <typename Action>
std::pair<hipdnnPluginStatus_t, std::string> failureOf(Action&& action)
{
    try
    {
        action();
    }
    catch(const HipdnnPluginException& error)
    {
        return {error.getStatus(), error.what()};
    }
    return {HIPDNN_PLUGIN_STATUS_SUCCESS, "did not throw"};
}

struct Page
{
    uint64_t total = 0;
    std::vector<std::string> ids;
    std::vector<std::vector<std::string>> knobIds;
    std::string problemFeatures;
};

Page enumerate(SlottedEngineFixture& fixture,
               const TestGraph& graph,
               const Settings& settings,
               uint64_t offset,
               uint64_t limit)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    StubHandle handle;
    const Config config(settings);
    hipdnnPluginConstData_t details{};
    fixture.engine().enumerateCandidates(handle, graph, config.get(), offset, limit, details);
    const auto* page = GetEngineDetails(details.ptr)->candidate_page();
    Page result;
    result.total = page->total_count();
    result.problemFeatures = page->problem_features()->str();
    if(page->candidates() == nullptr)
    {
        return result;
    }
    for(const auto* candidate : *page->candidates())
    {
        result.ids.push_back(candidate->id()->str());
        auto& knobs = result.knobIds.emplace_back();
        for(const auto* setting : *candidate->knob_settings())
        {
            knobs.push_back(setting->knob_id()->str());
        }
    }
    return result;
}

TEST(TestSlottedEngine, RunsThePicksInSlotOrderWithTheUnshiftedWorkspaceBase)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    const auto context = build(fixture, graph);

    std::vector<char> workspace(SHARED_ALIGNED + 16384);
    executeWith(context, workspace.data());

    const auto& launches = fixture.log().launches;
    ASSERT_EQ(launches.size(), 3U);
    EXPECT_EQ(fixture.log().slotOrder(), (std::vector<std::string>{"prep", "dkdv", "dq"}));
    // Declared order without a ranker: each slot's first kernel.
    EXPECT_EQ(launches[0].kernel, testId(0x11));
    EXPECT_EQ(launches[1].kernel, testId(0x21));
    EXPECT_EQ(launches[2].kernel, testId(0x31));
    for(const auto& launch : launches)
    {
        EXPECT_EQ(launch.workspace, workspace.data()) << launch.slot << " got a shifted base";
        // The offsets come from the graph match's tokens, read when the plan was built.
        EXPECT_EQ(launch.scratchOffset, static_cast<int64_t>(SHARED_ALIGNED));
        EXPECT_EQ(launch.deltaOffset, DELTA_OFFSET);
    }
}

TEST(TestSlottedEngine, ThePlansWorkspaceIsTheAlignedSharedRegionPlusItsLargestScratch)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;

    // Picks prep_a (0), dkdv_a (4096), dq_a (2048): scratch is reused, not summed.
    EXPECT_EQ(build(fixture, graph).plan().getWorkspaceSize(handle), SHARED_ALIGNED + 4096);

    // Pinning dkdv_c (8192) moves the largest scratch.
    EXPECT_EQ(build(fixture, graph, {{"dkdv.block_m", 256}}).plan().getWorkspaceSize(handle),
              SHARED_ALIGNED + 8192);
}

TEST(TestSlottedEngine, TheEnginesMaximumIsTheAlignedSharedRegionPlusTheLargestSurvivingScratch)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;

    const Config none;
    EXPECT_EQ(fixture.engine().getMaxWorkspaceSize(handle, graph, none.get()),
              SHARED_ALIGNED + 16384);

    // dkdv_b (16384) leaves with the pin; dq_b (12288) is now the largest.
    const Config pinned({{"dkdv.block_m", 64}});
    EXPECT_EQ(fixture.engine().getMaxWorkspaceSize(handle, graph, pinned.get()),
              SHARED_ALIGNED + 12288);
}

TEST(TestSlottedEngine, ANullWorkspaceIsRejectedAgainstTheTotalNotTheScratchAlone)
{
    // No kernel needs scratch, so only the shared region makes a workspace necessary.
    const Kernels noScratch{
        {{0x11, "prep_a", 64, 0}}, {{0x21, "dkdv_a", 64, 0}}, {{0x31, "dq_a", 64, 0}}};
    SlottedEngineFixture fixture(noScratch);
    const TestGraph graph(makeGraphId(1));
    const auto context = build(fixture, graph);

    const auto [status, text] = failureOf([&] { executeWith(context, nullptr); });

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_TRUE(fixture.log().launches.empty()) << "no slot may run before the check";
}

TEST(TestSlottedEngine, AnEngineWithNoSharedRegionAndNoScratchRunsWithoutAWorkspace)
{
    const Kernels noScratch{
        {{0x11, "prep_a", 64, 0}}, {{0x21, "dkdv_a", 64, 0}}, {{0x31, "dq_a", 64, 0}}};
    SlottedEngineFixture fixture(noScratch);
    graphMatchConfig().sharedBytes = 0;
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;
    const auto context = build(fixture, graph);

    EXPECT_EQ(context.plan().getWorkspaceSize(handle), 0U);
    EXPECT_NO_THROW(executeWith(context, nullptr));
    EXPECT_EQ(fixture.log().launches.size(), 3U);
}

TEST(TestSlottedEngine, TheWorkspaceLimitPrunesAKernelWhoseTotalExceedsIt)
{
    // dq's first kernel needs 12288 of scratch; with the 1024 shared region it is over 9000.
    const Kernels kernels{{{0x11, "prep_a", 64, 0}},
                          {{0x21, "dkdv_a", 64, 4096}},
                          {{0x31, "dq_big", 64, 12288}, {0x32, "dq_small", 128, 100}}};
    SlottedEngineFixture fixture(kernels);
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;
    const Settings limit{{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 9000}};

    const auto context = build(fixture, graph, limit);
    std::vector<char> workspace(9000);
    executeWith(context, workspace.data());

    ASSERT_EQ(fixture.log().launches.size(), 3U);
    EXPECT_EQ(fixture.log().launches[2].kernel, testId(0x32)) << "dq_big is over the limit";
    EXPECT_EQ(context.plan().getWorkspaceSize(handle), SHARED_ALIGNED + 4096);

    const Config config(limit);
    EXPECT_EQ(fixture.engine().getMaxWorkspaceSize(handle, graph, config.get()),
              SHARED_ALIGNED + 4096);
}

TEST(TestSlottedEngine, ALimitThatLeavesASlotWithNoKernelIsRejectedNamingTheSlot)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));

    // Every dkdv kernel needs at least 4096 of scratch.
    const auto [status, text] = failureOf([&] {
        build(fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 5000}});
    });

    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_NE(text.find("slot 'dkdv'"), std::string::npos) << text;
}

TEST(TestSlottedEngine, TheEngineDeclinesWhenTheSharedRegionAloneExceedsTheLimit)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;
    // 1000 bytes shared, 1024 aligned: the region exceeds a 1000-byte limit.
    const Settings limit{{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 1000}};

    EXPECT_EQ(failureOf([&] { build(fixture, graph, limit); }).first,
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    const Config config(limit);
    EXPECT_EQ(
        failureOf([&] { fixture.engine().getMaxWorkspaceSize(handle, graph, config.get()); }).first,
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

/// One kernel per slot; the shared region is 1000 bytes, which is 1024 aligned.
Kernels singleKernels(int64_t dkdvScratch)
{
    return {
        {{0x11, "prep_a", 64, 0}}, {{0x21, "dkdv_a", 64, dkdvScratch}}, {{0x31, "dq_a", 64, 0}}};
}

TEST(TestSlottedEngine, ALimitEqualToTheKernelTotalAcceptsItAndOneByteBelowPrunesIt)
{
    SlottedEngineFixture fixture(singleKernels(4096));
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;
    constexpr int64_t TOTAL = SHARED_ALIGNED + 4096;

    const auto context
        = build(fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, TOTAL}});
    EXPECT_EQ(context.plan().getWorkspaceSize(handle), static_cast<size_t>(TOTAL));

    const auto failure = failureOf([&] {
        build(fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, TOTAL - 1}});
    });
    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_NE(failure.second.find("no kernel in slot 'dkdv' within the workspace limit"),
              std::string::npos)
        << failure.second;
}

TEST(TestSlottedEngine, ALimitEqualToTheAlignedSharedRegionAcceptsItAndOneByteBelowDeclines)
{
    SlottedEngineFixture fixture(singleKernels(0));
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;

    const auto context = build(
        fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, SHARED_ALIGNED}});
    EXPECT_EQ(context.plan().getWorkspaceSize(handle), SHARED_ALIGNED);

    // The limit is judged against the aligned region: 1000 raw bytes still decline at 1000 and
    // at 1023, naming the shared region rather than a slot.
    for(const int64_t limit : {int64_t{1000}, static_cast<int64_t>(SHARED_ALIGNED) - 1})
    {
        const auto failure = failureOf([&] {
            build(fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, limit}});
        });
        EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE) << limit;
        EXPECT_NE(failure.second.find("declines the graph: its shared workspace region of 1000 "
                                      "bytes (1024 aligned) exceeds the workspace limit of "
                                      + std::to_string(limit) + " bytes"),
                  std::string::npos)
            << failure.second;
    }
}

TEST(TestSlottedEngine, AKernelTotalIsJudgedAgainstTheAlignedSharedRegionNotTheRawOne)
{
    // 1000 raw shared bytes plus 4000 of scratch fit a 5000-byte limit, but the plan reserves
    // 1024 aligned bytes and so needs 5024.
    SlottedEngineFixture fixture(singleKernels(4000));
    const TestGraph graph(makeGraphId(1));
    const StubHandle handle;

    const auto failure = failureOf([&] {
        build(fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 5000}});
    });
    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_NE(failure.second.find("no kernel in slot 'dkdv'"), std::string::npos) << failure.second;

    const auto context
        = build(fixture, graph, {{hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 5024}});
    EXPECT_EQ(context.plan().getWorkspaceSize(handle), SHARED_ALIGNED + 4000);
}

TEST(TestSlottedEngine, AdvertisesOneKnobPerSlotFieldWithValuesOverThatSlotsCatalog)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    StubHandle handle;
    hipdnnPluginConstData_t details{};
    fixture.engine().getDetails(handle, graph, details);

    std::set<std::string> ids;
    std::vector<int64_t> dkdvValues;
    for(const auto* knob : *GetEngineDetails(details.ptr)->knobs())
    {
        ids.insert(knob->knob_id()->str());
        if(knob->knob_id()->str() == "dkdv.block_m")
        {
            for(const auto value : *knob->constraint_as_IntConstraint()->valid_values())
            {
                dkdvValues.push_back(value);
            }
            EXPECT_EQ(knob->default_value_as_IntValue()->value(), 64);
        }
    }

    EXPECT_EQ(ids,
              (std::set<std::string>{hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME,
                                     "prep.block_m",
                                     "dkdv.block_m",
                                     "dq.block_m"}));
    std::sort(dkdvValues.begin(), dkdvValues.end());
    EXPECT_EQ(dkdvValues, (std::vector<int64_t>{64, 128, 256}));
}

TEST(TestSlottedEngine, ASlotKnobFiltersThatSlotOnly)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    const auto context = build(fixture, graph, {{"dkdv.block_m", 256}, {"dq.block_m", 128}});

    std::vector<char> workspace(SHARED_ALIGNED + 16384);
    executeWith(context, workspace.data());

    ASSERT_EQ(fixture.log().launches.size(), 3U);
    EXPECT_EQ(fixture.log().launches[0].kernel, testId(0x11)) << "prep was not pinned";
    EXPECT_EQ(fixture.log().launches[1].kernel, testId(0x23));
    EXPECT_EQ(fixture.log().launches[2].kernel, testId(0x32));
}

TEST(TestSlottedEngine, ASettingThatEmptiesASlotIsRejectedAtSetTimeNamingTheKnob)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    StubHandle handle;
    const Config config({{"dkdv.block_m", 999}});
    StubContext context;

    const auto failure = failureOf(
        [&] { fixture.engine().initializeExecutionContext(handle, graph, config.get(), context); });

    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_NE(failure.second.find("dkdv.block_m"), std::string::npos) << failure.second;
}

TEST(TestSlottedEngine, ABareFieldNameIsNotAKnob)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));

    const auto failure = failureOf([&] { enumerate(fixture, graph, {{BLOCK_M, 64}}, 0, 10); });

    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_EQ(failure.second, "Unknown or duplicate enumeration knob 'block_m'");
}

TEST(TestSlottedEngine, AKernelThatFailsToBuildCostsOnlyItsSlotsPick)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    fixture.log().failingPrepare = {testId(0x21)};

    const auto context = build(fixture, graph);
    std::vector<char> workspace(SHARED_ALIGNED + 16384);
    executeWith(context, workspace.data());

    ASSERT_EQ(fixture.log().launches.size(), 3U);
    EXPECT_EQ(fixture.log().launches[0].kernel, testId(0x11));
    EXPECT_EQ(fixture.log().launches[1].kernel, testId(0x22)) << "dkdv falls to its next kernel";
    EXPECT_EQ(fixture.log().launches[2].kernel, testId(0x31));
}

TEST(TestSlottedEngine, AKernelWhoseBuildRejectsTheProblemEndsThePlanBuildInsteadOfFallingBack)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    // INVALID_VALUE is the caller's problem, not the kernel's: the next kernel is not tried.
    fixture.log().failingPrepareWithStatus[testId(0x21)] = HIPDNN_PLUGIN_STATUS_INVALID_VALUE;

    const auto failure = failureOf([&] { build(fixture, graph); });

    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

TEST(TestSlottedEngine, AKernelWhoseBuildFailsWithAnotherStatusCostsOnlyItsSlotsPick)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    fixture.log().failingPrepareWithStatus[testId(0x21)] = HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;

    const auto context = build(fixture, graph);
    std::vector<char> workspace(SHARED_ALIGNED + 16384);
    executeWith(context, workspace.data());

    ASSERT_EQ(fixture.log().launches.size(), 3U);
    EXPECT_EQ(fixture.log().launches[1].kernel, testId(0x22)) << "dkdv falls to its next kernel";
}

TEST(TestSlottedEngine, ASlotWithNoBuildableKernelFailsThePlanNamingTheSlot)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    fixture.log().failingPrepare = {testId(0x31), testId(0x32)};

    const auto failure = failureOf([&] { build(fixture, graph); });

    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_NE(failure.second.find("could not build a plan for any of its 2 applicable kernel(s) "
                                  "in slot 'dq'"),
              std::string::npos)
        << failure.second;
}

TEST(TestSlottedEngine, DeclinesAGraphTheGraphMatchRefuses)
{
    SlottedEngineFixture fixture;
    graphMatchConfig().declines = true;
    StubHandle handle;
    const TestGraph graph(makeGraphId(1));

    EXPECT_FALSE(fixture.engine().isApplicable(handle, graph));
    const auto failure = failureOf([&] { build(fixture, graph); });
    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_NE(failure.second.find("graph_match declined the graph"), std::string::npos);
}

TEST(TestSlottedEngine, EnumerationListsTheProductInSlotOrderWithCompositeIds)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));

    const auto page = enumerate(fixture, graph, {}, 0, 100);

    EXPECT_EQ(page.total, 12U);
    ASSERT_EQ(page.ids.size(), 12U);
    EXPECT_EQ(page.ids.front(), composite(0x11, 0x21, 0x31));
    EXPECT_EQ(page.ids[1], composite(0x11, 0x21, 0x32)) << "the last slot varies fastest";
    EXPECT_EQ(page.ids[2], composite(0x11, 0x22, 0x31));
    EXPECT_EQ(page.ids.back(), composite(0x12, 0x23, 0x32));
    EXPECT_EQ(std::set<std::string>(page.ids.begin(), page.ids.end()).size(), 12U);
    // A candidate's knob settings cover every slot.
    for(const auto& knobs : page.knobIds)
    {
        EXPECT_EQ(std::set<std::string>(knobs.begin(), knobs.end()),
                  (std::set<std::string>{"prep.block_m", "dkdv.block_m", "dq.block_m"}));
    }
}

TEST(TestSlottedEngine, EnumerationPagesAreStableAndConcatenateToTheWholeProduct)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    const auto whole = enumerate(fixture, graph, {}, 0, 100).ids;

    std::vector<std::string> paged;
    for(uint64_t offset = 0; offset < 12; offset += 5)
    {
        const auto page = enumerate(fixture, graph, {}, offset, 5);
        EXPECT_EQ(page.total, 12U);
        paged.insert(paged.end(), page.ids.begin(), page.ids.end());
    }

    EXPECT_EQ(paged, whole);
    EXPECT_TRUE(enumerate(fixture, graph, {}, 12, 5).ids.empty());
    EXPECT_EQ(failureOf([&] { enumerate(fixture, graph, {}, 13, 5); }).first,
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

TEST(TestSlottedEngine, EnumerationOfALargeProductPagesWithoutMaterializingIt)
{
    // 60 x 60 x 60 = 216000 combinations; a page asks for ten near the end.
    Kernels kernels(3);
    static const std::vector<std::string> s_names = [] {
        std::vector<std::string> names;
        names.reserve(180);
        for(int i = 0; i < 180; ++i)
        {
            names.push_back("kernel_" + std::to_string(i));
        }
        return names;
    }();
    for(size_t slot = 0; slot < 3; ++slot)
    {
        for(size_t i = 0; i < 60; ++i)
        {
            const auto index = slot * 60 + i;
            kernels[slot].push_back({static_cast<uint8_t>(index + 1),
                                     s_names[index].c_str(),
                                     static_cast<int64_t>(i),
                                     0});
        }
    }
    SlottedEngineFixture fixture(kernels);
    const TestGraph graph(makeGraphId(1));

    const auto page = enumerate(fixture, graph, {}, 215990, 10);

    EXPECT_EQ(page.total, 216000U);
    ASSERT_EQ(page.ids.size(), 10U);
    EXPECT_EQ(page.ids.back(), composite(60, 120, 180));
    EXPECT_EQ(page.ids.front(), composite(60, 120, 171));
}

TEST(TestSlottedEngine, AKnobPinNarrowsOnlyItsSlotsFactorOfTheProduct)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));

    const auto page = enumerate(fixture, graph, {{"dkdv.block_m", 128}}, 0, 100);

    EXPECT_EQ(page.total, 4U);
    for(const auto& id : page.ids)
    {
        EXPECT_NE(id.find("dkdv=" + idOf(0x22)), std::string::npos) << id;
    }
}

TEST(TestSlottedEngine, ACandidatesKnobSettingsReplayToExactlyThatCandidate)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    StubHandle handle;
    hipdnnPluginConstData_t details{};
    const Config none;
    fixture.engine().enumerateCandidates(handle, graph, none.get(), 7, 1, details);
    const auto* candidate = hipdnn_flatbuffers_sdk::data_objects::GetEngineDetails(details.ptr)
                                ->candidate_page()
                                ->candidates()
                                ->Get(0);
    Settings pins;
    for(const auto* setting : *candidate->knob_settings())
    {
        pins.emplace_back(setting->knob_id()->str(), setting->value_as_IntValue()->value());
    }
    const auto expected = candidate->id()->str();

    const auto replayed = enumerate(fixture, graph, pins, 0, 100);

    EXPECT_EQ(replayed.total, 1U);
    EXPECT_EQ(replayed.ids, (std::vector<std::string>{expected}));
}

TEST(TestSlottedEngine, ADuplicateKnobTupleWithinASlotIsAnErrorNamingTheSlot)
{
    // Two dkdv kernels share block_m = 64, the slot's only knob, and differ in scratch.
    const Kernels kernels{{{0x11, "prep_a", 64, 0}},
                          {{0x21, "dkdv_a", 64, 4096}, {0x22, "dkdv_b", 64, 8192}},
                          {{0x31, "dq_a", 64, 0}}};
    SlottedEngineFixture fixture(kernels);
    const TestGraph graph(makeGraphId(1));

    const auto failure = failureOf([&] { enumerate(fixture, graph, {}, 0, 10); });

    EXPECT_EQ(failure.first, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_NE(failure.second.find("slot 'dkdv'"), std::string::npos) << failure.second;
}

TEST(TestSlottedEngine, TheWorkspaceTokensStayOutOfTheProblemFeatures)
{
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));

    const auto features
        = nlohmann::json::parse(enumerate(fixture, graph, {}, 0, 1).problemFeatures);

    bool sawOrdinaryToken = false;
    for(const auto& [key, value] : features.items())
    {
        EXPECT_EQ(key.find("ws."), std::string::npos) << key;
        sawOrdinaryToken = sawOrdinaryToken || key.find(SEQLEN_TOKEN) != std::string::npos;
    }
    EXPECT_TRUE(sawOrdinaryToken) << "a non-ws graph-match token must remain a feature";
}

TEST(TestSlottedEngine, AConfigurationPredictionIsUnavailableWithoutAValidProgramRecord)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    SlottedEngineFixture fixture;
    const TestGraph graph(makeGraphId(1));
    StubHandle handle;
    const Config config;

    const auto prediction = fixture.engine().getPrediction(
        handle, graph, config.get(), HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);

    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.engine_config, nullptr);
    EXPECT_NE(prediction.reason.find("No valid program record exists"), std::string::npos)
        << prediction.reason;
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
