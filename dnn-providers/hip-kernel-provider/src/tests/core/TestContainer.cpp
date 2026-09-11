// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <string>

#include <gtest/gtest.h>

#include "core/Container.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"
#endif

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::core;

/// The SDK detects getEngineName by callability, so a member it cannot call reads as
/// absent: the entry point reports NOT_APPLICABLE and every engine renders as a
/// hexadecimal ID. Nothing at runtime tells that apart from a provider that never named
/// its engines, so the compiler is the only place to enforce the distinction.
static_assert(hipdnn_plugin_sdk::HasGetEngineName<Container>::value,
              "Container::getEngineName must match the signature the plugin SDK calls");

/// Engines the provider exposes: one per compiled-in native engine, plus one per
/// discovered descriptor set, read from the inventory rather than hardcoded so a
/// newly shipped pack is never silently uncounted.
static uint32_t expectedEngines()
{
    uint32_t expected = 0;
#ifdef HIPDNN_ENGINE_ASM_SDPA
    ++expected;
#endif
#ifdef HIPDNN_ENGINE_HIP_MLOPS
    ++expected;
#endif
#ifdef HIPDNN_ENGINE_HIP_FLASH2
    ++expected;
#endif
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    expected += static_cast<uint32_t>(
        hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets().size());
#endif
    return expected;
}

/// Upper bound for the fixed-size buffers below; only needs to be at least
/// expectedEngines().
constexpr uint32_t MAX_EXPECTED_ENGINES = 8;

TEST(TestContainer, ConstructsSuccessfully)
{
    const Container container;
}

TEST(TestContainer, CopyEngineIdsReturnsExpectedEngineCount)
{
    uint32_t numEngines = 0;
    auto totalEngines = Container::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(totalEngines, expectedEngines());
    EXPECT_EQ(numEngines, expectedEngines());
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

    EXPECT_EQ(totalEngines, expectedEngines());
    EXPECT_EQ(numEngines, expectedEngines());

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
    for(const auto* expected : {"hipkernel:ConvFwd", "hipkernel:Pointwise"})
    {
        EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end())
            << "shipped descriptor set '" << expected << "' was not discovered";
    }

    Container container;
    const auto allEngineIds = container.getEngineManager().getAllEngineIds();

    for(const auto& set : sets)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(set.engine.name);
        EXPECT_NE(std::find(allEngineIds.begin(), allEngineIds.end(), engineId), allEngineIds.end())
            << "no engine for descriptor set '" << set.engine.name << "'";
    }
}

TEST(TestContainer, GetEngineNameReturnsTheDeclaredNameForEveryIngestorEngine)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine;

    const auto& sets = discoverDescriptorSets();
    ASSERT_FALSE(sets.empty());

    for(const auto& set : sets)
    {
        const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(set.engine.name);

        const char* name = nullptr;
        EXPECT_EQ(Container::getEngineName(engineId, &name), HIPDNN_PLUGIN_STATUS_SUCCESS)
            << "no name for descriptor set '" << set.engine.name << "'";
        ASSERT_NE(name, nullptr);
        EXPECT_EQ(std::string(name), set.engine.name);

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
    if(deviceString != "gfx942" && deviceString != "gfx950")
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

#ifdef HIPDNN_ENGINE_ASM_SDPA
    ASSERT_EQ(applicableEngines.size(), 1);
    EXPECT_EQ(applicableEngines.front(), asm_sdpa_engine::AsmSdpaEngine::staticId());
#else
    EXPECT_TRUE(applicableEngines.empty());
#endif
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

    ASSERT_EQ(allEngines.size(), expectedEngines());
}
