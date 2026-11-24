// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/Container.hpp"
#include "core/GraphSchema.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/SdpaGraphMutators.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <atomic>
#include <set>
#include <thread>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#ifdef HIPDNN_ENGINE_FLYDSL
#include "tests/engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"
#endif
#endif

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::core;

/// The SDK detects getEngineName by callability, so a member it cannot call reads as
/// absent: the entry point reports NOT_APPLICABLE and every engine renders as a
/// hexadecimal ID. Nothing at runtime tells that apart from a provider that never named
/// its engines, so the compiler is the only place to enforce the distinction.
static_assert(hipdnn_plugin_sdk::HasGetEngineName<Container>::value,
              "Container::getEngineName must match the signature the plugin SDK calls");

/// Engines this build configuration exposes: every compiled-in native engine, plus every
/// shipped descriptor set. Spelled out here rather than read from the provider inventory,
/// so an engine that disappears from that inventory fails these tests.
///
/// Containment, not equality: the catalog is descriptor-driven, so a build that stages
/// additional descriptor sets legitimately exposes more engines. These names are a floor.
///
/// The ingestor names are literals where the others are shared constants, and must stay
/// that way. The others are declared in C++ and read here from the same constant as
/// production, so a rename moves both sides together and goes undetected. These two are
/// declared in descriptor JSON, so restating them here is an independent check.
static std::vector<std::string> expectedEngineNames()
{
    std::vector<std::string> names;
#ifdef HIPDNN_ENGINE_ASM_SDPA
    names.emplace_back(hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_NAME);
#endif
#ifdef HIPDNN_ENGINE_HIP_MLOPS
    names.emplace_back(hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_NAME);
#endif
#ifdef HIPDNN_ENGINE_HIP_FLASH2
    names.emplace_back(hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_NAME);
#endif
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    names.emplace_back("hipkernel:ConvFwd");
    names.emplace_back("hipkernel:Pointwise");
#endif
    return names;
}

/// Upper bound for the fixed-size buffers below. Raise it when a configuration exposes
/// more engines than this. FlyDSL alone adds one engine per kernel family.
constexpr uint32_t MAX_EXPECTED_ENGINES = 32;

TEST(TestContainer, ConstructsSuccessfully)
{
    const Container container;
}

TEST(TestContainer, CopyEngineIdsAdvertisesEveryExpectedEngine)
{
    std::array<int64_t, MAX_EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    auto totalEngines
        = Container::copyEngineIds(engineIds.data(), MAX_EXPECTED_ENGINES, numEngines);

    ASSERT_LE(totalEngines, MAX_EXPECTED_ENGINES) << "raise MAX_EXPECTED_ENGINES";
    EXPECT_EQ(numEngines, totalEngines);

    const auto expectedNames = expectedEngineNames();
    EXPECT_GE(totalEngines, static_cast<uint32_t>(expectedNames.size()));

    const auto last = engineIds.begin() + static_cast<std::ptrdiff_t>(numEngines);
    for(const auto& name : expectedNames)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(name);
        EXPECT_NE(std::find(engineIds.begin(), last, engineId), last)
            << "engine '" << name << "' is not advertised";
    }
}

TEST(TestContainer, CopyEngineIdsReportsTheSameCountWithAndWithoutABuffer)
{
    uint32_t countedEngines = 0;
    const auto totalFromCount = Container::copyEngineIds(nullptr, 0, countedEngines);

    std::array<int64_t, MAX_EXPECTED_ENGINES> engineIds = {};
    uint32_t copiedEngines = 0;
    const auto totalFromCopy
        = Container::copyEngineIds(engineIds.data(), MAX_EXPECTED_ENGINES, copiedEngines);

    EXPECT_EQ(countedEngines, totalFromCount);
    EXPECT_EQ(totalFromCopy, totalFromCount);
    EXPECT_EQ(copiedEngines, std::min(totalFromCount, MAX_EXPECTED_ENGINES));
}

TEST(TestContainer, CopyEngineIdsWithBufferContainsHipMlopsEngineId)
{
#ifndef HIPDNN_ENGINE_HIP_MLOPS
    GTEST_SKIP();
#else
    std::array<int64_t, MAX_EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    auto totalEngines
        = Container::copyEngineIds(engineIds.data(), MAX_EXPECTED_ENGINES, numEngines);

    EXPECT_EQ(numEngines, totalEngines);

    bool containsHipMlopsEngine = false;
    for(const int64_t engine : engineIds)
    {
        containsHipMlopsEngine |= (engine == hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_ID);
    }
    EXPECT_EQ(containsHipMlopsEngine, true);
#endif
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
TEST(TestContainer, ExposesAnEngineForEveryDiscoveredDescriptorSet)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    // Named rather than just counted: neither a count nor an emptiness check can tell
    // a missing engine (e.g. a pack table dropped from a static-archive link) from a
    // renamed one.
    //
    // Containment, not equality: the catalog is descriptor-driven, so a build that stages
    // additional descriptor sets into the descriptor tree — an integration fixture, say —
    // legitimately discovers more than the two shipped sets. Requiring an exact set would
    // make every such build fail here, which tests the staging rather than the engines.
    const auto& sets = discoverDescriptorSets();

    std::vector<std::string> names;
    names.reserve(sets.size());
    for(const auto& set : sets)
    {
        names.push_back(set.engine.name);
    }
    std::sort(names.begin(), names.end());
    // Literals for the reason expectedEngineNames() records: a constant shared with
    // the production table would make a rename propagate here and this never fail.
    for(const auto* expected : {"hipkernel:ConvFwd", "hipkernel:Pointwise"})
    {
        EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end())
            << "shipped descriptor set '" << expected << "' was not discovered";
    }

    Container container;
    const auto allEngineIds = container.getEngineManager().getAllEngineIds();

    for(const auto& set : sets)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(
            hipdnn_plugin_sdk::ingestor::engineIdentityName(set.engine));
        EXPECT_NE(std::find(allEngineIds.begin(), allEngineIds.end(), engineId), allEngineIds.end())
            << "no engine for descriptor set '" << set.engine.name << "'";
    }
}

TEST(TestContainer, AContainerRebuiltFromTheRetainedSetsExposesEveryIngestorEngine)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());

    const auto expectEveryIngestorEngine = [&sets](Container& container, const char* which) {
        const auto engineIds = container.getEngineManager().getAllEngineIds();
        for(const auto& set : sets)
        {
            const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(
                hipdnn_plugin_sdk::ingestor::engineIdentityName(set.engine));
            EXPECT_NE(std::find(engineIds.begin(), engineIds.end(), engineId), engineIds.end())
                << which << " Container has no engine for descriptor set '" << set.engine.name
                << "'";
        }
    };

    // `earlier` takes whatever state managers are still in their slots, discovery's or a
    // rediscovery's, so these two are on the rebuild path whatever ran before.
    {
        const Container earlier;
    }
    Container afterDestroyed;
    expectEveryIngestorEngine(afterDestroyed, "the after-destroyed");
    Container whileAlive;
    expectEveryIngestorEngine(whileAlive, "the while-alive");
}

TEST(TestContainer, EachDiscoveredSetsSharedStateIsItsOwnAndIsHandedOutUnchanged)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());

    // The schema ids are what a state exposes of the set it was built from (one per kernel
    // slot of a slotted engine), so a slot serving another set's state shows up only
    // between sets whose schemas differ.
    const auto schemaIdsOfSet = [](const hipdnn_plugin_sdk::ingestor::DescriptorSet& set) {
        std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId> ids;
        if(set.slots.empty())
        {
            ids.push_back(set.schema.id);
        }
        for(const auto& slot : set.slots)
        {
            ids.push_back(slot.schema.id);
        }
        return ids;
    };
    std::set<std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId>> schemaIds;
    for(const auto& set : sets)
    {
        schemaIds.insert(schemaIdsOfSet(set));
    }
    ASSERT_GE(schemaIds.size(), 2U)
        << "every discovered set shares one metadata schema, so a mispaired slot is invisible";

    for(const auto& set : sets)
    {
        const auto state = discoveredSharedState(set.engine.id);
        ASSERT_NE(state.get(), nullptr)
            << "no shared state for descriptor set '" << set.engine.name << "'";
        std::vector<hipdnn_plugin_sdk::ingestor::DescriptorId> stateIds;
        stateIds.reserve(state->slots().size());
        for(const auto& slot : state->slots())
        {
            stateIds.push_back(slot.schema.id);
        }
        EXPECT_EQ(stateIds, schemaIdsOfSet(set))
            << "the id of '" << set.engine.name << "' serves the state of another set";
        // Shared, not rebuilt or handed out once: a second lookup is the same object.
        EXPECT_EQ(discoveredSharedState(set.engine.id).get(), state.get())
            << "'" << set.engine.name << "' built a second state";
    }

    EXPECT_EQ(discoveredSharedState(hipdnn_plugin_sdk::ingestor::DescriptorId{}).get(), nullptr)
        << "an id no discovered set has must have no state";
}

TEST(TestContainer, ConcurrentLookupsOfOneDiscoveredSharedStateAllSeeTheSameState)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());
    const auto& engineId = sets.front().engine.id;
    const auto expected = discoveredSharedState(engineId);
    ASSERT_NE(expected.get(), nullptr);

    constexpr size_t THREAD_COUNT = 8;
    std::atomic<bool> start{false};
    std::atomic<size_t> matching{0};
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(size_t threadIndex = 0; threadIndex < THREAD_COUNT; ++threadIndex)
    {
        threads.emplace_back([&start, &matching, &engineId, &expected] {
            while(!start.load())
            {
                std::this_thread::yield();
            }
            if(discoveredSharedState(engineId) == expected)
            {
                ++matching;
            }
        });
    }
    start.store(true);
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(matching.load(), THREAD_COUNT);
}

TEST(TestContainer, EveryContainerSharesTheDiscoveredStateAndReleasesItOnDestruction)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());

    // use_count, read through a held copy, counts the engines serving each state: one
    // more per live Container proves each Container shares it rather than building its
    // own, and the count falling back proves a destroyed Container lets go.
    std::vector<
        std::shared_ptr<const hipdnn_plugin_sdk::ingestor::KernelIngestorSharedState<Handle>>>
        states;
    std::vector<long> baseline;
    for(const auto& set : sets)
    {
        states.push_back(discoveredSharedState(set.engine.id));
        ASSERT_NE(states.back().get(), nullptr);
        baseline.push_back(states.back().use_count());
    }

    {
        const Container first;
        const Container second;
        for(size_t index = 0; index < sets.size(); ++index)
        {
            EXPECT_EQ(states[index].use_count(), baseline[index] + 2)
                << "two Containers do not both serve the shared state of '"
                << sets[index].engine.name << "'";
        }
    }

    for(size_t index = 0; index < sets.size(); ++index)
    {
        EXPECT_EQ(states[index].use_count(), baseline[index])
            << "a destroyed Container still holds the shared state of '" << sets[index].engine.name
            << "'";
    }
}

TEST(TestContainer, RediscoveryReplacesTheSharedStateWithoutInvalidatingALiveContainer)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());
    const auto& engineId = sets.front().engine.id;
    const auto engineIdValue = hipdnn_data_sdk::utilities::engineNameToId(
        hipdnn_plugin_sdk::ingestor::engineIdentityName(sets.front().engine));

    const auto before = discoveredSharedState(engineId);
    ASSERT_NE(before.get(), nullptr);

    Container alive;
    ASSERT_NO_THROW(rediscoverSharedStatesForTesting());

    const auto after = discoveredSharedState(engineId);
    ASSERT_NE(after.get(), nullptr);
    EXPECT_NE(after.get(), before.get()) << "the reset hook did not rebuild the shared state";
    EXPECT_EQ(after->slots().front().schema.id, before->slots().front().schema.id);

    // The live Container's engine still holds what it was built over; the next Container
    // serves the rebuilt state.
    const auto engineIds = alive.getEngineManager().getAllEngineIds();
    EXPECT_NE(std::find(engineIds.begin(), engineIds.end(), engineIdValue), engineIds.end());
    const long afterBaseline = after.use_count();
    {
        const Container next;
        EXPECT_EQ(after.use_count(), afterBaseline + 1);
    }
}

TEST(TestContainer, GetEngineNameReturnsTheDeclaredNameForEveryIngestorEngine)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());

    for(const auto& set : sets)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(
            hipdnn_plugin_sdk::ingestor::engineIdentityName(set.engine));

        const char* name = nullptr;
        EXPECT_EQ(Container::getEngineName(engineId, &name), HIPDNN_PLUGIN_STATUS_SUCCESS)
            << "no name for descriptor set '" << set.engine.name << "'";
        ASSERT_NE(name, nullptr);
        // The advertised name: a set taking over a built-in is reported under its name.
        EXPECT_EQ(std::string(name), hipdnn_plugin_sdk::ingestor::engineIdentityName(set.engine));

        // What hipDNN checks before it admits the engine: a name that does not hash back
        // to the ID it was reported for drops the engine at load time.
        EXPECT_EQ(hipdnn_data_sdk::utilities::engineNameToId(name), engineId);
    }
}
#endif

TEST(TestContainer, GetEngineNameNamesEveryAdvertisedEngine)
{
    // Read from the same table copyEngineIds reports from, so no build configuration
    // leaves an advertised engine unasserted.
    std::array<int64_t, MAX_EXPECTED_ENGINES> engineIds = {};
    uint32_t numEngines = 0;
    Container::copyEngineIds(engineIds.data(), MAX_EXPECTED_ENGINES, numEngines);
    ASSERT_GT(numEngines, 0U);

    for(uint32_t i = 0; i < numEngines; ++i)
    {
        const char* name = nullptr;
        EXPECT_EQ(Container::getEngineName(engineIds[i], &name), HIPDNN_PLUGIN_STATUS_SUCCESS)
            << "no name for advertised engine id " << engineIds[i];
        ASSERT_NE(name, nullptr);
        EXPECT_FALSE(std::string(name).empty());
        EXPECT_EQ(hipdnn_data_sdk::utilities::engineNameToId(name), engineIds[i]);
    }
}

TEST(TestContainer, GetEngineNameDeclinesAnUnknownEngineId)
{
    // Hashed from a name no descriptor set or registered engine carries, so the ID is
    // unknown by construction rather than by picking a literal no engine happens to use.
    const auto unknownEngineId
        = hipdnn_data_sdk::utilities::engineNameToId("hipkernel:NoSuchEngine");

    const char* name = nullptr;
    EXPECT_EQ(Container::getEngineName(unknownEngineId, &name),
              HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);
}

TEST(TestContainer, GetEngineNameRejectsANullNameArgument)
{
    EXPECT_EQ(Container::getEngineName(hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_ID, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// The provider's schema guard's opt-in list (GraphSchema.hpp): the HIP MLOps and HIP Flash2
// engines claim graph schema 1.4.0 (each of them gates ragged tensors, declared alignments and
// offset multipliers), and the ASM SDPA engine and the six FlyDSL SDPA forward engines claim 1.6.0,
// the SDK's newest (their forward matchers also gate cu_seq_len), and the two FlyDSL SDPA
// backward engines claim 1.4.0 (their UEDs' sdk_version; their matchers reviewed against every field
// up to it, and a backward graph never carries cu_seq_len). So does the RDNA WMMA prefill engine,
// whose matcher gates paged K/V, ragged Q / O and their offset multipliers, and every declared
// alignment. The gfx950 rocKE conv dgrad engine
// claims 1.3.0: its graph matcher gates alignment (multiples of 16 bytes) and declines ragged
// tensors. Every other engine stays at or below the 1.2.0 baseline, so ragged, aligned and
// multiplier graphs never reach it. Opting an engine in is an edit here, reviewed with that
// engine's matchers.
TEST(TestContainer, PinsEveryEnginesDeclaredGraphSchema)
{
    const std::map<std::string, std::string> optedIn{
        {"hipkernel:Gfx950ConvDgradAot", "1.3.0"},
        {"hipkernel:Gfx1151WmmaAttention", "1.4.0"},
        {std::string(hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_NAME), "1.6.0"},
        {std::string(hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_NAME), "1.4.0"},
        {std::string(hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_NAME), "1.4.0"},
        {"hipkernel:flydsl_sdpa_rdna", "1.6.0"},
        {"hipkernel:flydsl_sdpa_rdna_decode", "1.6.0"},
        {"hipkernel:flydsl_sdpa_aiw", "1.6.0"},
        {"hipkernel:flydsl_sdpa_cdna", "1.6.0"},
        {"hipkernel:flydsl_sdpa_cdna_dualwave", "1.6.0"},
        {"hipkernel:flydsl_sdpa_cdna_decode", "1.6.0"},
        {"hipkernel:flydsl_sdpa_bwd_merged", "1.4.0"},
        {"hipkernel:flydsl_sdpa_bwd_split", "1.4.0"}};
    Container container;
    auto& manager = container.getEngineManager();
    const auto baseline = providerBaselineGraphSchema();
    for(const auto engineId : manager.getAllEngineIds())
    {
        const char* name = nullptr;
        ASSERT_EQ(Container::getEngineName(engineId, &name), HIPDNN_PLUGIN_STATUS_SUCCESS);
        const auto declared = manager.maxGraphSchema(engineId);
        std::cout << "[ schema   ] " << name << " " << declared.str() << "\n";
        const auto opted = optedIn.find(name);
        if(opted != optedIn.end())
        {
            EXPECT_EQ(declared.str(), opted->second) << name;
        }
        else
        {
            EXPECT_FALSE(baseline < declared)
                << name << " claims graph schema " << declared.str()
                << " past the provider baseline without being on the opt-in list";
        }
    }
}

TEST(TestContainer, GetEngineManagerReturnsValidReference)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    (void)engineManager;
}

TEST(TestContainer, GetApplicableEngineIdsSdpaGraph)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    Handle handle;
    auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    if(deviceString != "gfx90a" && deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }
    Container container;
    auto& engineManager = container.getEngineManager();

    const std::vector<int64_t> dims{4, 8, 256, 128};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto graph = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     dims,
                                                                     strides,
                                                                     DataType::BFLOAT16,
                                                                     DataType::FLOAT);
    auto graphBuffer = graph.Release();

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuffer.data(), graphBuffer.size());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);
    // Unused when neither the ASM nor the FlyDSL engines are built.
    [[maybe_unused]] const auto applies = [&applicableEngines](auto engineId) {
        return std::find(applicableEngines.begin(), applicableEngines.end(), engineId)
               != applicableEngines.end();
    };

    // The engines that serve this graph on gfx90a, gfx942 and gfx950, and no others.
    // NOLINTNEXTLINE(misc-const-correctness)
    size_t expected = 0;
#ifdef HIPDNN_ENGINE_ASM_SDPA
    // The assembly kernels are built for gfx942 and gfx950 only.
    if(deviceString != "gfx90a")
    {
        EXPECT_TRUE(applies(asm_sdpa_engine::AsmSdpaEngine::staticId()));
        ++expected;
    }
#endif
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)
    // FlyDSL's CDNA SDPA engines, one per kernel family: each family that ships objects for
    // this device has a bf16 head-dim-128 non-causal one, so it must serve this graph, and a
    // family that ships none must not (the dual-wave kernel ships for gfx950 only).
    for(const auto& [op, engine] : std::array<std::pair<std::string_view, std::string_view>, 3>{{
            {"sdpa_cdna", "hipkernel:flydsl_sdpa_cdna"},
            {"sdpa_cdna_dualwave", "hipkernel:flydsl_sdpa_cdna_dualwave"},
            {"sdpa_cdna_decode", "hipkernel:flydsl_sdpa_cdna_decode"},
        }})
    {
        const bool ships = kernel_ingestor_engine::testing::flydslShipsForThisDevice(op);
        EXPECT_EQ(applies(hipdnn_data_sdk::utilities::engineNameToId(std::string(engine))), ships)
            << engine;
        if(ships)
        {
            ++expected;
        }
    }
#endif
    EXPECT_EQ(applicableEngines.size(), expected);
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
TEST(TestContainer, GetApplicableEngineIdsPointwiseAddGraph)
{
    // Applicability is device-resolved: with no device, matchers decline.
    SKIP_IF_NO_DEVICES();

    using namespace hip_kernel_provider::kernel_ingestor_engine;
    using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

    Handle handle;
    Container container;
    auto& engineManager = container.getEngineManager();

    const auto graph = buildPointwiseGraph();
    const auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graph.GetBufferPointer(), graph.GetSize());

    auto applicableEngines = engineManager.getApplicableEngineIds(handle, graphWrapper);

    EXPECT_NE(std::find(applicableEngines.begin(),
                        applicableEngines.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              applicableEngines.end());
}
#endif

TEST(TestContainer, GetAllEngineIds)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    auto allEngines = engineManager.getAllEngineIds();

    const auto expectedNames = expectedEngineNames();
    ASSERT_GE(allEngines.size(), expectedNames.size());

    for(const auto& name : expectedNames)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(name);
        EXPECT_NE(std::find(allEngines.begin(), allEngines.end(), engineId), allEngines.end())
            << "engine '" << name << "' is missing from the engine manager";
    }
}

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)
// The provider at engine plugin API 1.6.0 with its schema guard (core/GraphSchema.hpp): a packed
// (THD) SDPA graph, schema 1.4.0 through its ragged offset multipliers, reaches the FlyDSL SDPA
// engines, which opted in, and no other engine claims it.
TEST(TestContainer, RoutesAPackedSdpaGraphOnlyToTheFlydslEngines)
{
    SKIP_IF_NO_DEVICES();
    using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
    SdpaGraphSpec spec;
    setPacked(spec, {128, 37}, {100, 128});
    const auto buffer = buildFlydslSdpaRdnaGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        buffer.GetBufferPointer(), buffer.GetSize());
    ASSERT_EQ(graphSchemaFloor(graph).str(), "1.4.0");
    Handle handle;
    Container container;
    const auto applicable = container.getEngineManager().getApplicableEngineIds(handle, graph);
    bool rdna = false;
    for(const auto engineId : applicable)
    {
        const char* name = nullptr;
        ASSERT_EQ(Container::getEngineName(engineId, &name), HIPDNN_PLUGIN_STATUS_SUCCESS);
        EXPECT_EQ(std::string_view(name).rfind("hipkernel:flydsl_sdpa", 0), 0U)
            << name << " claims a graph past its schema";
        rdna = rdna || std::string_view(name) == FLYDSL_SDPA_RDNA.engineName;
    }
    // The RDNA prefill objects serve packed sequences on every gfx11 device this build ships.
    if(packedArchDeviceProperties().gcnArchName.rfind("gfx11", 0) == 0
       && findLoadedSet(FLYDSL_SDPA_RDNA.engineName) != nullptr)
    {
        EXPECT_TRUE(rdna) << "the RDNA prefill engine declined a packed graph it serves";
    }
}
#endif

#ifdef HIPDNN_ENGINE_ASM_SDPA
// The schema guard passes the ASM engine the ragged graphs it serves (AITER group mode):
// ragged Q/K/V/O raise the graph's floor to 1.4.0, the schema the engine declares.
TEST(TestContainer, RoutesARaggedSdpaGraphToTheAsmEngine)
{
    SKIP_IF_NO_DEVICES();
    Handle handle;
    const auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP() << "the ASM group-mode kernels ship for gfx942 and gfx950 only";
    }
    const std::vector<int64_t> dims{4, 8, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, {3, 1, 2, 0});
    auto ragged = hipdnn_test_sdk::utilities::sdpa::withRaggedTensors(
        hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
            dims,
            strides,
            dims,
            strides,
            dims,
            strides,
            dims,
            strides,
            hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT),
        {"q", "k", "v", "o"});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper raggedGraph(
        ragged.GetBufferPointer(), ragged.GetSize());
    ASSERT_EQ(graphSchemaFloor(raggedGraph).str(), "1.4.0");

    Container container;
    auto& manager = container.getEngineManager();
    const auto ids = manager.getApplicableEngineIds(handle, raggedGraph);
    EXPECT_NE(std::find(ids.begin(), ids.end(), asm_sdpa_engine::AsmSdpaEngine::staticId()),
              ids.end());
}

// With the default static ordering the ASM engine ranks above every other engine that
// serves SDPA forward (EngineOrdering.hpp), so it runs every graph it accepts. These are
// the forward classes it serves through causal masks, stats and group-mode lengths, several
// of which no other engine serves; each must rank ASM first. An output the kernels cannot
// address must leave ASM out of the list, so the graph falls to another engine or to none.
TEST(TestContainer, RanksTheAsmEngineFirstForTheSdpaForwardGraphsItServes)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    namespace sdpa = hipdnn_test_sdk::utilities::sdpa;

    Handle handle;
    const auto deviceString = hip_kernel_provider_common::getDeviceString(handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP() << "the ASM forward kernels ship for gfx942 and gfx950 only";
    }

    const std::vector<int64_t> dims{4, 8, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, {3, 1, 2, 0});
    // A bf16 head-dim-128 [B, S, H, D] graph.
    const auto bshdGraph = [&](bool withStats,
                               bool causalMask,
                               bool paddingMask = false,
                               hipdnn_test_sdk::utilities::CuSeqLenSides cuSeqLenSides
                               = hipdnn_test_sdk::utilities::CuSeqLenSides::NONE) {
        return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(dims,
                                                                   strides,
                                                                   dims,
                                                                   strides,
                                                                   dims,
                                                                   strides,
                                                                   dims,
                                                                   strides,
                                                                   DataType::BFLOAT16,
                                                                   DataType::FLOAT,
                                                                   /*withAttnMask=*/false,
                                                                   /*withScale=*/false,
                                                                   withStats,
                                                                   /*alibiMask=*/false,
                                                                   paddingMask,
                                                                   causalMask,
                                                                   /*overrideShapeEnabled=*/false,
                                                                   DataType::UNSET,
                                                                   cuSeqLenSides);
    };
    const auto mutate
        = [](flatbuffers::FlatBufferBuilder&& builder, const std::function<void(GraphT&)>& change) {
              auto graph = UnPackGraph(builder.GetBufferPointer());
              change(*graph);
              flatbuffers::FlatBufferBuilder mutated;
              mutated.Finish(CreateGraph(mutated, graph.get()));
              return mutated;
          };
    const auto tensorNamed = [](GraphT& graph, const std::string& name) -> TensorAttributesT& {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->name == name)
            {
                return *tensor;
            }
        }
        throw std::runtime_error("No tensor named " + name);
    };
    const auto group = [&](flatbuffers::FlatBufferBuilder&& builder) {
        return sdpa::withRaggedTensors(std::move(builder), {"q", "k", "v", "o"});
    };

    struct Case
    {
        std::string name;
        flatbuffers::FlatBufferBuilder graph;
        bool asmServes;
    };
    std::vector<Case> cases;
    // Top-left causal with K and V on the Q/O offset table, so both sides share their lengths.
    cases.push_back({"group top-left causal, one offset table",
                     mutate(group(bshdGraph(false, true)),
                            [&](GraphT& graph) {
                                const auto qoOffsetUid
                                    = tensorNamed(graph, "q").ragged_offset_tensor_uid;
                                tensorNamed(graph, "k").ragged_offset_tensor_uid = qoOffsetUid;
                                tensorNamed(graph, "v").ragged_offset_tensor_uid = qoOffsetUid;
                            }),
                     true});
    // AITER's [H, total_q] LSE, ragged on the Q/O offset table.
    const auto withGroupStats = [&](GraphT& graph) {
        auto& stats = tensorNamed(graph, "stats");
        stats.strides = {dims[2], dims[0] * dims[2], 1, 1};
        stats.ragged_offset_tensor_uid = tensorNamed(graph, "q").ragged_offset_tensor_uid;
        stats.ragged_offset_multiplier = 1;
    };
    cases.push_back({"group stats", mutate(group(bshdGraph(true, false)), withGroupStats), true});
    cases.push_back(
        {"group cu_seq_len with padding_mask",
         group(bshdGraph(false, false, true, hipdnn_test_sdk::utilities::CuSeqLenSides::BOTH)),
         true});
    // The batch-mode causal rows ship for gfx942 only.
    if(deviceString == "gfx942")
    {
        cases.push_back({"batch top-left causal at S_q == S_kv", bshdGraph(false, true), true});
        cases.push_back({"batch causal with stats", bshdGraph(true, true), true});
    }
    // Outputs whose slices overlap: batch mode with the O batch stride equal to its head
    // stride, and group mode with an O head stride of 0.
    cases.push_back({"batch O batch stride equal to its head stride",
                     mutate(bshdGraph(false, false),
                            [&](GraphT& graph) {
                                auto& o = tensorNamed(graph, "o");
                                o.strides[0] = o.strides[1];
                            }),
                     false});
    cases.push_back({"group O head stride 0",
                     mutate(group(bshdGraph(false, false)),
                            [&](GraphT& graph) { tensorNamed(graph, "o").strides[1] = 0; }),
                     false});

    Container container;
    auto& manager = container.getEngineManager();
    const auto asmId = asm_sdpa_engine::AsmSdpaEngine::staticId();
    for(auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            testCase.graph.GetBufferPointer(), testCase.graph.GetSize());
        auto ids = manager.getApplicableEngineIds(handle, graph);
        hipdnn_data_sdk::utilities::sortEngineIds(ids);
        if(testCase.asmServes)
        {
            ASSERT_FALSE(ids.empty());
            EXPECT_EQ(ids.front(), asmId);
        }
        else
        {
            EXPECT_EQ(std::find(ids.begin(), ids.end(), asmId), ids.end());
        }
    }
}
#endif
