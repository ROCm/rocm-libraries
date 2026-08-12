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

#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddPack.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseAddTestGraphs.hpp"

/**
 * @file TestKernelIngestorEngine.cpp
 * @brief What now exists in place of the deleted PointwiseAddEngine forwarding wrapper:
 *        registerNativeIngestorSymbols() and makePointwiseAddEngine().
 *
 * There is no engine class left to test directly -- GenericEngine<Handle, Settings,
 * Context> already satisfies IEngine end to end, and it is the SDK's own type, covered
 * by the SDK's own suite. What is this provider's to test is that the factory wires it
 * up correctly: the resulting engine answers every IEngine method for a graph this
 * pack's matchers accept, and declines one they do not. Reached through Container and
 * EngineManager rather than constructed piecemeal, since makePointwiseAddEngine() itself
 * takes no seams a test could inject through -- the device resolver and dispatch
 * handler it wires up are process-lifetime statics (see KernelIngestorEngine.cpp).
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

/// Stubs a config naming this pack's engine and setting no knobs -- the shape
/// getMaxWorkspaceSize() and initializeExecutionContext() both key their engine lookup
/// on, per EngineManager::getEngine(engineConfig.engineId()).
void stubAsThisEnginesConfig(MockEngineConfig& config)
{
    EXPECT_CALL(config, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(config, engineId()).WillRepeatedly(::testing::Return(pointwiseAddEngineId()));
}

// ---------------------------------------------------------------------------
// registerNativeIngestorSymbols(): idempotent across repeated calls
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorEngine, RegisterNativeIngestorSymbolsIsIdempotentAcrossRepeatedCalls)
{
    // once_flag-guarded (see its own doc): Container's constructor calls this on every
    // Container built over the process's life (SharedContainerManager's weak_ptr means
    // that can be many times), so a second call must be a silent no-op rather than a
    // throw on the duplicate registration NativeRegistry::registerSymbol() would
    // otherwise report.
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
    EXPECT_NO_THROW(registerNativeIngestorSymbols());
}

// ---------------------------------------------------------------------------
// makePointwiseAddEngine(): a working GenericEngine, reached through Container
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorEngine, MakePointwiseAddEngineIsReachableWithTheDescriptorEngineId)
{
    Container container;
    auto& engineManager = container.getEngineManager();

    // The engine table keys this entry on pointwiseAddEngineId(), which registers the
    // engine's name on first call; getAllEngineIds() proves the factory above actually
    // installed an engine under that id rather than merely compiling.
    const auto allEngineIds = engineManager.getAllEngineIds();
    EXPECT_NE(std::find(allEngineIds.begin(), allEngineIds.end(), pointwiseAddEngineId()),
              allEngineIds.end());
}

TEST(TestKernelIngestorEngine, IsApplicableAcceptsAGraphThisPacksMatchersAccept)
{
    // Applicability resolves the call's device and matches on its properties, and the
    // matchers decline outright when no device can be named -- so an accept is only
    // meaningful where there is a device to accept for.
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    const auto graph = buildPointwiseGraph();
    const auto applicable = engineManager.getApplicableEngineIds(handle, wrap(graph));

    EXPECT_NE(std::find(applicable.begin(), applicable.end(), pointwiseAddEngineId()),
              applicable.end());
}

TEST(TestKernelIngestorEngine, GetEngineDetailsReportsTheBlockSizeKnob)
{
    // The reported knob's value set comes from the catalog this graph matched, which is
    // empty without a device (see the applicability test above).
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    const auto graph = buildPointwiseGraph();
    hipdnnPluginConstData_t details{};
    engineManager.getEngineDetails(handle, wrap(graph), pointwiseAddEngineId(), details);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    ASSERT_EQ(wrapper.knobCount(), 1U);
    EXPECT_EQ(wrapper.getKnobByName("block_size").knobId(), "block_size");
}

TEST(TestKernelIngestorEngine, GetMaxWorkspaceSizeReportsTheLargerBlocksRequirement)
{
    // Sizes a workspace from the kernels the graph matched -- none, without a device.
    SKIP_IF_NO_DEVICES();

    Container container;
    auto& engineManager = container.getEngineManager();
    const Handle handle;

    const auto graph = buildPointwiseGraph();
    MockEngineConfig engineConfig;
    stubAsThisEnginesConfig(engineConfig);

    // The catalog's two surviving FLOAT kernels report 0 and 1024 bytes; the engine's
    // answer is a maximum across whichever kernels the graph admits, not a fixed value,
    // so this also proves the workspace query reaches the real catalog rather than a
    // stub.
    const auto workspaceSize = engineManager.getMaxWorkspaceSize(handle, wrap(graph), engineConfig);
    EXPECT_EQ(workspaceSize, 1024U);
}

TEST(TestKernelIngestorEngine, InitializeExecutionContextBuildsAPlanForTheTopRankedKernel)
{
    // Builds a plan, which compiles the selected kernel through hiprtc -- so unlike the
    // applicability and knob-reporting tests above, this one needs a device. Every
    // device-touching test in this engine's suite guards the same way.
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

    // Which kernel, not merely that there is one: pointwiseAddScore ranks on block size
    // and the graph admits both FLOAT kernels, so the 256 entry is the defined winner.
    // Asserting only hasValidPlan() would pass just as well if selection handed back the
    // 64 kernel, which is the property this test is named for.
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

    // A multiplication, not this pack's add: refused at the graph-scoped matcher, the
    // same refusal TestPointwiseAddGraphMatcherRefusal covers directly, exercised here
    // through the full engine rather than the bare matcher function.
    const auto graph
        = buildPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL);
    const auto applicable = engineManager.getApplicableEngineIds(handle, wrap(graph));

    EXPECT_EQ(std::find(applicable.begin(), applicable.end(), pointwiseAddEngineId()),
              applicable.end());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
