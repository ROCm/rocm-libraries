// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestKernelIngestorEngine.cpp
 * @brief Tests registerNativeIngestorSymbols() and makePointwiseAddEngine(); GenericEngine
 *        itself is covered by the SDK's suite. Reached through Container and
 *        EngineManager since makePointwiseAddEngine() takes no injectable seams.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::core::Container;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::MockEngineConfig;

GraphWrapper wrap(const flatbuffers::FlatBufferBuilder& builder)
{
    return GraphWrapper(builder.GetBufferPointer(), builder.GetSize());
}

// SymbolScope stand-ins: content-indifferent, and a pack's real functions are internal
// to their native file and unreachable from here.
bool acceptAnyGraph(const hipdnn_plugin_sdk::ingestor::MatchContext& /*context*/,
                    hipdnn_plugin_sdk::ingestor::BoundTokens& /*bound*/)
{
    return true;
}

double scoreNothing(const hipdnn_plugin_sdk::ingestor::KernelDefinition& /*kernel*/,
                    const hipdnn_plugin_sdk::ingestor::MatchContext& /*context*/)
{
    return 0.0;
}

/// Keyed the way getMaxWorkspaceSize()/initializeExecutionContext() look it up.
void stubAsThisEnginesConfig(MockEngineConfig& config)
{
    EXPECT_CALL(config, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(config, engineId())
        .WillRepeatedly(::testing::Return(
            hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)));
}

// registerNativeIngestorSymbols(): idempotent across repeated calls

TEST(TestKernelIngestorEngine, RegisterNativeIngestorSymbolsIsIdempotentAcrossRepeatedCalls)
{
    // once_flag-guarded; Container's constructor calls this on every Container built.
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
}

TEST(TestKernelIngestorEngine, AFailedPackUnregistersItsOwnSymbolsAndLeavesOthersAlone)
{
    using hipdnn_plugin_sdk::ingestor::GraphMatcherRegistry;
    using hipdnn_plugin_sdk::ingestor::ScoreRegistry;
    using hipdnn_plugin_sdk::ingestor::SymbolScope;

    // A neighbour pack's symbol, committed before the failing pack runs.
    const std::string neighbourSymbol = "test.neighbour.graph_match";
    SymbolScope<Handle> neighbour;
    neighbour.add(neighbourSymbol, &acceptAnyGraph);
    neighbour.commit();

    // Occupies the symbol the failing pack tries second, forcing a partial registration.
    const std::string contendedSymbol = "test.contended.score";
    SymbolScope<Handle> squatter;
    squatter.add(contendedSymbol, &scoreNothing);
    squatter.commit();

    const std::string firstSymbol = "test.failing.graph_match";
    {
        SymbolScope<Handle> failing;
        failing.add(firstSymbol, &acceptAnyGraph);
        EXPECT_THROW(failing.add(contendedSymbol, &scoreNothing), std::runtime_error);
    }

    EXPECT_THROW(GraphMatcherRegistry::resolve(firstSymbol), std::runtime_error);
    // ...while the neighbour's survives: one pack failing must not affect others.
    EXPECT_NO_THROW(GraphMatcherRegistry::resolve(neighbourSymbol));
    EXPECT_NO_THROW(ScoreRegistry::resolve(contendedSymbol));

    GraphMatcherRegistry::unregisterSymbol(neighbourSymbol);
    ScoreRegistry::unregisterSymbol(contendedSymbol);
}

TEST(TestKernelIngestorEngine, ACommittedScopeKeepsItsSymbols)
{
    using hipdnn_plugin_sdk::ingestor::GraphMatcherRegistry;
    using hipdnn_plugin_sdk::ingestor::SymbolScope;

    const std::string symbol = "test.committed.graph_match";
    {
        SymbolScope<Handle> scope;
        scope.add(symbol, &acceptAnyGraph);
        scope.commit();
    }

    EXPECT_NO_THROW(GraphMatcherRegistry::resolve(symbol));

    GraphMatcherRegistry::unregisterSymbol(symbol);
}

// makePointwiseAddEngine(): a working GenericEngine, reached through Container

TEST(TestKernelIngestorEngine, MakePointwiseAddEngineIsReachableWithTheDescriptorEngineId)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    const auto allEngineIds = engineManager.getAllEngineIds();
    EXPECT_NE(std::find(allEngineIds.begin(),
                        allEngineIds.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              allEngineIds.end());
}

TEST(TestKernelIngestorEngine, IsApplicableAcceptsAGraphThisPacksMatchersAccept)
{
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    const auto graph = buildPointwiseGraph();
    const auto applicable = engineManager.getApplicableEngineIds(handle, wrap(graph));

    EXPECT_NE(std::find(applicable.begin(),
                        applicable.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              applicable.end());
}

TEST(TestKernelIngestorEngine, GetEngineDetailsReportsTheBlockSizeKnob)
{
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    const auto graph = buildPointwiseGraph();
    hipdnnPluginConstData_t details{};
    engineManager.getEngineDetails(
        handle,
        wrap(graph),
        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName),
        details);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    ASSERT_EQ(wrapper.knobCount(), 1U);
    EXPECT_EQ(wrapper.getKnobByName("block_size").knobId(), "block_size");
}

TEST(TestKernelIngestorEngine, GetMaxWorkspaceSizeReportsTheLargerBlocksRequirement)
{
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    const Handle handle;

    const auto graph = buildPointwiseGraph();
    MockEngineConfig engineConfig;
    stubAsThisEnginesConfig(engineConfig);

    // Two surviving FLOAT kernels report 0 and 1024 bytes; answer is a max across both.
    const auto workspaceSize = engineManager.getMaxWorkspaceSize(handle, wrap(graph), engineConfig);
    EXPECT_EQ(workspaceSize, 1024U);
}

TEST(TestKernelIngestorEngine, InitializeExecutionContextBuildsAPlanForTheTopRankedKernel)
{
    // Compiles the selected kernel through hiprtc, so unlike the tests above needs a device.
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    const Handle handle;

    const auto graph = buildPointwiseGraph();
    MockEngineConfig engineConfig;
    stubAsThisEnginesConfig(engineConfig);

    Context context;
    ASSERT_NO_THROW(
        engineManager.initializeExecutionContext(handle, wrap(graph), engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());

    // pointwiseAddScore ranks on block size; both FLOAT kernels admitted, 256 wins.
    const auto& plan
        = dynamic_cast<const hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>&>(context.plan());
    EXPECT_EQ(plan.kernel().getIntMetadata(std::string(BLOCK_SIZE_FIELD)), 256);
}

// ---------------------------------------------------------------------------
// Three packs under one engine, end to end
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorEngine, ServesAllItsPacksOperationsUnderOneEngineId)
{
    // Matchers decline outright with no device resolved, so an accept is only
    // meaningful where there is one.
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    // The whole point of the multi-pack topology: one engine id answers for every
    // operation, reached through a shared graph matcher and three different kernels.
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName);

    for(const auto operation : {hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                                hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL,
                                hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB})
    {
        const auto graph = buildPointwiseGraph(operation);
        const auto applicable = engineManager.getApplicableEngineIds(handle, wrap(graph));

        EXPECT_NE(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
            << "engine did not claim operation " << static_cast<int>(operation);
    }
}

TEST(TestKernelIngestorEngine, DeclinesAGraphNoPackOfItsClaims)
{
    // Matchers decline outright with no device resolved, so a decline here would be
    // vacuous rather than proof the operation matchers refused DIV.
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    // DIV passes the shared shape matcher, but every pack's operation matcher declines
    // it -- distinguishing "no pack claims this op" from "the matcher rejected the
    // shape" outright. Not SUB: that's now one of the three claimed operations.
    const auto graph
        = buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::DIV);
    const auto applicable = engineManager.getApplicableEngineIds(handle, wrap(graph));

    EXPECT_EQ(std::find(applicable.begin(),
                        applicable.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              applicable.end());
}

// ---------------------------------------------------------------------------
// descriptorSearchDirectory(): which of the three sources answers
// ---------------------------------------------------------------------------

/// The env override is what every test and every run-from-build-dir depends on, so a real
/// directory there has to beat both the module-relative path and the configure-time one.
TEST(TestKernelIngestorEngine, PrefersHipdnnDescriptorDirOverEverything)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory existing(
        std::filesystem::temp_directory_path() / "hip_kernel_provider_descriptor_env");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter override(
        "HIPDNN_DESCRIPTOR_DIR", existing.path().string());

    EXPECT_EQ(descriptorSearchDirectory(), existing.path());
}

/// A stale override must be ignored, not obeyed: the install-tree CTestTestfile.cmake
/// bakes in this build's absolute staging path via ENVIRONMENT, which won't exist on
/// another machine -- without this guard the installed suite would silently load zero
/// descriptor sets.
TEST(TestKernelIngestorEngine, IgnoresAHipdnnDescriptorDirThatDoesNotExist)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter stale(
        "HIPDNN_DESCRIPTOR_DIR", "/nowhere/in/particular");

    const auto resolved = descriptorSearchDirectory();

    EXPECT_NE(resolved, std::filesystem::path("/nowhere/in/particular"));
    EXPECT_TRUE(resolved.generic_string().find(HIPDNN_DESCRIPTOR_SUBDIR) != std::string::npos)
        << "resolved to " << resolved;
}

/// With the env unset, resolution falls through to step 2 or 3, both ending in the same
/// fixed suffix. Pinning that suffix catches a silent fallthrough to an empty path,
/// which would otherwise be indistinguishable from a real resolved directory.
TEST(TestKernelIngestorEngine, FallsBackToAModuleRelativeOrInstalledPath)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter unset("HIPDNN_DESCRIPTOR_DIR",
                                                                            "");

    const auto resolved = descriptorSearchDirectory();

    ASSERT_FALSE(resolved.empty());
    EXPECT_TRUE(resolved.generic_string().find(HIPDNN_DESCRIPTOR_SUBDIR) != std::string::npos)
        << "resolved to " << resolved << ", which does not end in " << HIPDNN_DESCRIPTOR_SUBDIR;
}

/// Asserts the mechanism step 2 rests on: an address resolves to the module containing
/// it. Keyed on address rather than symbol name because every provider exports the same
/// plugin entry points, so a name lookup could answer for the wrong module.
TEST(TestKernelIngestorEngine, ResolvesAModuleDirectoryFromAnAddressWithinIt)
{
    const auto directory = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
        reinterpret_cast<const void*>(&descriptorSearchDirectory));

    std::error_code exists;
    EXPECT_TRUE(std::filesystem::is_directory(directory, exists)) << directory;
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
