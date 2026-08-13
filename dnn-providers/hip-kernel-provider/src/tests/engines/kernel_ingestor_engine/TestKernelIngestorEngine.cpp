// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
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
 * @brief Tests registerNativeIngestorSymbols() and makePointwiseAddEngine(), which
 *        replace the deleted PointwiseAddEngine forwarding wrapper. GenericEngine
 *        itself is the SDK's type and covered by the SDK's suite; this file only
 *        verifies the factory wires it up correctly, reached through Container and
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

/// Wraps a built graph buffer as the frontend hands one to an engine.
GraphWrapper wrap(const flatbuffers::FlatBufferBuilder& builder)
{
    return GraphWrapper(builder.GetBufferPointer(), builder.GetSize());
}

// Stand-ins for the SymbolScope tests below. Those assert the scope's all-or-nothing
// behaviour, which is indifferent to what a symbol points at, so they use trivial
// functions rather than a pack's, which are internal to their native file and
// correctly unreachable from here.
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

/// Stubs a config naming this pack's engine, keyed the way getMaxWorkspaceSize() and
/// initializeExecutionContext() both look it up: EngineManager::getEngine(engineId()).
void stubAsThisEnginesConfig(MockEngineConfig& config)
{
    EXPECT_CALL(config, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(config, engineId())
        .WillRepeatedly(::testing::Return(
            hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)));
}

// ---------------------------------------------------------------------------
// registerNativeIngestorSymbols(): idempotent across repeated calls
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorEngine, RegisterNativeIngestorSymbolsIsIdempotentAcrossRepeatedCalls)
{
    // once_flag-guarded; Container's constructor calls this on every Container built
    // (may be many, via SharedContainerManager's weak_ptr), so repeats must be a no-op.
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
}

TEST(TestKernelIngestorEngine, AFailedPackUnregistersItsOwnSymbolsAndLeavesOthersAlone)
{
    using hipdnn_plugin_sdk::ingestor::GraphMatcherRegistry;
    using hipdnn_plugin_sdk::ingestor::ScoreRegistry;
    using hipdnn_plugin_sdk::ingestor::SymbolScope;

    // A neighbour pack's symbol, registered and committed before the failing pack runs.
    const std::string neighbourSymbol = "test.neighbour.graph_match";
    SymbolScope<Handle> neighbour;
    neighbour.add(neighbourSymbol, &acceptAnyGraph);
    neighbour.commit();

    // Occupies the symbol the failing pack will try second, so its registration fails
    // partway, the shape rollback exists for.
    const std::string contendedSymbol = "test.contended.score";
    SymbolScope<Handle> squatter;
    squatter.add(contendedSymbol, &scoreNothing);
    squatter.commit();

    const std::string firstSymbol = "test.failing.graph_match";
    {
        SymbolScope<Handle> failing;
        failing.add(firstSymbol, &acceptAnyGraph);
        EXPECT_THROW(failing.add(contendedSymbol, &scoreNothing), std::runtime_error);
        // Never committed: the destructor rolls this pack back.
    }

    // The failing pack's own symbol is gone...
    EXPECT_THROW(GraphMatcherRegistry::resolve(firstSymbol), std::runtime_error);
    // ...while the neighbour's survives, which is what "one pack failing must not
    // unregister everyone else's symbols" means.
    EXPECT_NO_THROW(GraphMatcherRegistry::resolve(neighbourSymbol));
    // The contended symbol still belongs to its original owner, not the failed pack.
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

// ---------------------------------------------------------------------------
// makePointwiseAddEngine(): a working GenericEngine, reached through Container
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorEngine, MakePointwiseAddEngineIsReachableWithTheDescriptorEngineId)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    // hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName) registers the engine's name on first call;
    // getAllEngineIds() proves the factory installed it, not just compiled.
    const auto allEngineIds = engineManager.getAllEngineIds();
    EXPECT_NE(std::find(allEngineIds.begin(),
                        allEngineIds.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              allEngineIds.end());
}

TEST(TestKernelIngestorEngine, IsApplicableAcceptsAGraphThisPacksMatchersAccept)
{
    // Matchers decline outright with no device resolved, so an accept is only
    // meaningful where there is one.
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
    // The reported knob set comes from the matched catalog, empty without a device.
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
    // Sizes a workspace from the kernels the graph matched: none, without a device.
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    const Handle handle;

    const auto graph = buildPointwiseGraph();
    MockEngineConfig engineConfig;
    stubAsThisEnginesConfig(engineConfig);

    // The catalog's two surviving FLOAT kernels report 0 and 1024 bytes; the answer is
    // a max across admitted kernels, proving the query reaches the real catalog.
    const auto workspaceSize = engineManager.getMaxWorkspaceSize(handle, wrap(graph), engineConfig);
    EXPECT_EQ(workspaceSize, 1024U);
}

TEST(TestKernelIngestorEngine, InitializeExecutionContextBuildsAPlanForTheTopRankedKernel)
{
    // Builds a plan by compiling the selected kernel through hiprtc, so unlike the
    // tests above this needs a device.
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

    // pointwiseAddScore ranks on block size and both FLOAT kernels are admitted, so 256
    // is the defined winner; hasValidPlan() alone would not catch a wrong selection.
    const auto& plan
        = dynamic_cast<const hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>&>(context.plan());
    EXPECT_EQ(plan.kernel().getIntMetadata(std::string(BLOCK_SIZE_FIELD)), 256);
}

// ---------------------------------------------------------------------------
// Unhappy path: a graph none of this pack's kernels serve
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorEngine, DeclinesAGraphThisPacksMatchersRefuse)
{
    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    // A multiplication is refused at the graph-scoped matcher (also covered directly by
    // TestPointwiseAddGraphMatcherRefusal), exercised here through the full engine.
    const auto graph
        = buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL);
    const auto applicable = engineManager.getApplicableEngineIds(handle, wrap(graph));

    EXPECT_EQ(std::find(applicable.begin(),
                        applicable.end(),
                        hipdnn_data_sdk::utilities::engineNameToId(POINTWISE_ADD.engineName)),
              applicable.end());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
