// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledLaunch.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "TestDescriptorRoot.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanCapture.hpp"
#include "engines/kernel_ingestor_engine/IngestorPlanRestore.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdPackedCase.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanCodec.hpp"

/**
 * @file TestConvFwdDeclarativeLaunch.cpp
 * @brief Tests the declarative launch handler on `hipkernel:ConvFwdDeclarative`, a twin of
 *        `hipkernel:ConvFwd` whose UDD restates ConvNative.cpp's launch as formulas.
 *
 * The twin compiles the same kernel source with the same defines and reuses the native
 * kernel match and score, so the two engines differ only in how they bind the graph and
 * how they launch. Each case holds the declarative launch to the native one: the same
 * kernels admitted, the same grid, block and arguments, and the same output.
 *
 * The device cases read a copy of the packed set in a scratch directory. Other suites
 * corrupt or delete the staged archives.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* SCRATCH_LABEL = "convfwddeclarativelaunch";
constexpr std::string_view NATIVE_ENGINE = "hipkernel:ConvFwd";
constexpr std::string_view DECLARATIVE_ENGINE = "hipkernel:ConvFwdDeclarative";

/// One parity graph and its name for the failure trace.
struct ParityGraph
{
    const char* name;
    std::function<flatbuffers::FlatBufferBuilder()> build;
};

/// Graphs the native pack serves. The larger shape has n, c and k above 1, h != width,
/// and an output count (2*4*7*6 = 336) that no block size divides, so a formula that
/// drops a factor, swaps an extent or floors the grid gives another launch.
flatbuffers::FlatBufferBuilder buildLargerConvGraph(data_objects::DataType dataType)
{
    return packs::buildConvFwdGraph(dataType,
                                    data_objects::ConvMode::CROSS_CORRELATION,
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    {2, 3, 9, 7},
                                    std::vector<int64_t>{4, 3, 3, 2});
}

const std::vector<ParityGraph>& parityGraphs()
{
    static const std::vector<ParityGraph> s_graphs{
        {"FloatSingleChannel", [] { return packs::buildConvFwdGraph(); }},
        {"HalfWideInput", [] { return packs::buildConvFwdPackedRoundTripGraph(); }},
        {"FloatBatchedMultiChannel",
         [] { return buildLargerConvGraph(data_objects::DataType::FLOAT); }},
        {"HalfBatchedMultiChannel",
         [] { return buildLargerConvGraph(data_objects::DataType::HALF); }},
    };
    return s_graphs;
}

/// The launch ConvNative.cpp performs for @p inputs, in the form a declarative launch
/// resolves to: three operand pointers, then the seven extents as int32.
ResolvedLaunch nativeLaunchOf(const ConvFwdLaunchInputs& inputs)
{
    const auto geometry = convFwdLaunchGeometry(inputs);
    ResolvedLaunch launch;
    launch.grid = {geometry.gridX, 1, 1};
    launch.block = {geometry.block, 1, 1};

    for(const int64_t uid : {inputs.binding.x, inputs.binding.w, inputs.binding.y})
    {
        ResolvedLaunchArgument pointer;
        pointer.kind = LaunchArgumentKind::POINTER;
        pointer.tensorUid = uid;
        launch.arguments.push_back(pointer);
    }
    const auto& extents = inputs.extents;
    for(const int32_t value :
        {extents.n, extents.c, extents.h, extents.width, extents.k, extents.r, extents.s})
    {
        ResolvedLaunchArgument scalar;
        scalar.kind = LaunchArgumentKind::SCALAR;
        scalar.scalarType = LaunchScalarType::I32;
        std::memcpy(scalar.bytes.data(), &value, sizeof(value));
        launch.arguments.push_back(scalar);
    }
    return launch;
}

/// The key two twin kernels share: they differ in id and name, not in what they build.
std::string variantKey(const KernelDefinition& kernel)
{
    return kernel.getStringMetadata(std::string(packs::DTYPE_FIELD)) + "/"
           + std::to_string(kernel.getIntMetadata(std::string(packs::BLOCK_SIZE_FIELD)));
}

std::map<std::string, const KernelDefinition*> byVariant(const Catalog& catalog)
{
    std::map<std::string, const KernelDefinition*> kernels;
    for(const auto* kernel : catalog.entries)
    {
        kernels.emplace(variantKey(*kernel), kernel);
    }
    return kernels;
}

/// Expects @p call to throw HipdnnPluginException with @p status.
template <typename Call>
void expectRefusal(Call call, hipdnnPluginStatus_t status, const std::string& what)
{
    try
    {
        call();
        ADD_FAILURE() << what << " did not throw";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), status) << what << ": " << error.getMessage();
    }
}

const IKernelDispatchHandler<Handle>& declarativeHandler()
{
    registerNativeIngestorSymbols();
    return *DispatchRegistry<Handle>::resolve(std::string(DECLARATIVE_LAUNCH_SYMBOL));
}

/// The twin's own pattern, binding @p context as its engine does.
std::optional<BoundTokens> bindTwin(const DescriptorSet& twin, const MatchContext& context)
{
    const auto* pattern
        = std::get_if<std::shared_ptr<const CompiledGraphPattern>>(&twin.engine.graphMatch);
    if(pattern == nullptr || *pattern == nullptr)
    {
        return std::nullopt;
    }
    return matchGraphPattern(**pattern, context);
}

// ---------------------------------------------------------------------------
// Without a device
// ---------------------------------------------------------------------------

/// The twin's formulas, evaluated over its own pattern's binding, give the grid, block and
/// argument list ConvNative.cpp computes for the same graph and block size.
TEST(TestConvFwdDeclarativeLaunch, FormulasReproduceTheNativeGeometryAndArguments)
{
    const auto& twin = packs::loadedSet(DECLARATIVE_ENGINE);
    ASSERT_EQ(twin.dispatches.size(), 1U);
    const auto& launch = twin.dispatches.front().launch;
    ASSERT_NE(launch, nullptr) << "the twin's UDD is not declarative";

    struct Case
    {
        flatbuffers::FlatBufferBuilder (*build)();
        ConvFwdExtents extents;
    };
    const std::vector<Case> cases = {
        {[] { return packs::buildConvFwdGraph(); }, ConvFwdExtents{1, 1, 3, 3, 1, 2, 2}},
        {[] { return buildLargerConvGraph(data_objects::DataType::FLOAT); },
         ConvFwdExtents{2, 3, 9, 7, 4, 3, 2}},
    };
    for(const auto& graphCase : cases)
    {
        const packs::GraphFixture fixture(graphCase.build());
        const auto bound = bindTwin(twin, fixture.context());
        ASSERT_TRUE(bound.has_value());
        for(const int64_t blockSize : {64, 256})
        {
            SCOPED_TRACE("block size " + std::to_string(blockSize));
            const auto kernel = packs::makeKernel(blockSize, "FLOAT", "ConvFwd");
            const ConvFwdLaunchInputs inputs{
                ConvFwdBinding{packs::CONV_X_UID, packs::CONV_W_UID, packs::CONV_Y_UID},
                graphCase.extents,
                blockSize};

            EXPECT_TRUE(launch->resolve(fixture.context(), *bound, kernel)
                        == nativeLaunchOf(inputs));
            EXPECT_EQ(launch->workspaceBytes(fixture.context(), *bound, kernel), 0U);
        }
    }
}

/// A kernel that reaches the handler without a compiled launch belongs to a pack whose
/// UDD is not declarative. The handler cannot launch it, and says so on every hook that
/// receives the kernel.
TEST(TestConvFwdDeclarativeLaunch, HandlerRefusesAKernelWithoutALaunch)
{
    const packs::GraphFixture fixture(packs::buildConvFwdGraph());
    const auto bound = packs::matchPattern(packs::convPattern(), fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto kernel = packs::makeKernel(64, "FLOAT", "ConvFwd");
    kernel.source.kind = KernelSourceKind::KPACK;
    ASSERT_EQ(kernel.launch, nullptr);

    const auto& handler = declarativeHandler();
    expectRefusal(
        [&] { static_cast<void>(handler.workspaceBytes(fixture.context(), *bound, kernel)); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "workspaceBytes");
    expectRefusal([&] { static_cast<void>(handler.prepare(fixture.context(), *bound, kernel)); },
                  HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                  "prepare");
}

/// The handler launches prebuilt code only. An embedded_source kernel with a valid launch
/// is refused at prepare(), before anything is compiled.
TEST(TestConvFwdDeclarativeLaunch, HandlerRefusesAnEmbeddedSourceKernel)
{
    const auto& twin = packs::loadedSet(DECLARATIVE_ENGINE);
    ASSERT_EQ(twin.dispatches.size(), 1U);
    const packs::GraphFixture fixture(packs::buildConvFwdGraph());
    const auto bound = bindTwin(twin, fixture.context());
    ASSERT_TRUE(bound.has_value());

    auto kernel = packs::makeKernel(64, "FLOAT", "ConvFwd");
    kernel.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    kernel.launch = twin.dispatches.front().launch;
    ASSERT_NE(kernel.launch, nullptr);

    expectRefusal(
        [&] { static_cast<void>(declarativeHandler().prepare(fixture.context(), *bound, kernel)); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "prepare");
}

// ---------------------------------------------------------------------------
// On a device
// ---------------------------------------------------------------------------

/// Both engines, validated from one copy of the packed set for device 0.
struct LoadedTwins
{
    std::filesystem::path copy;
    DeviceProperties deviceProperties;
    std::vector<ValidatedSet<Handle>> sets;
    const ValidatedSet<Handle>* native = nullptr;
    const ValidatedSet<Handle>* declarative = nullptr;
};

// Copies the packed set of device 0 into `scratch` and validates it. `twins.copy` stays
// empty when nothing was packed for the device.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void loadTwins(const ScopedDirectory& scratch, LoadedTwins& twins)
{
    ASSERT_NO_FATAL_FAILURE(
        packs::copyPackedArchForDevice(hip_kernel_provider::testing::unitKpackRoot(),
                                       scratch.path(),
                                       twins.deviceProperties,
                                       twins.copy));
    if(twins.copy.empty())
    {
        return;
    }

    registerNativeIngestorSymbols();
    twins.sets = loadValidatedDescriptorSets<Handle>(twins.copy);
    for(const auto& validated : twins.sets)
    {
        if(validated.set.engine.name == NATIVE_ENGINE)
        {
            twins.native = &validated;
        }
        else if(validated.set.engine.name == DECLARATIVE_ENGINE)
        {
            twins.declarative = &validated;
        }
    }
    ASSERT_NE(twins.native, nullptr) << NATIVE_ENGINE << " was not validated";
    ASSERT_NE(twins.declarative, nullptr) << DECLARATIVE_ENGINE << " was not validated";
}

/// A plan over @p kernel, dispatched as its engine's state manager dispatches it.
std::unique_ptr<GenericPlan<Handle>> planFor(const ValidatedSet<Handle>& engine,
                                             const KernelDefinition& kernel,
                                             const MatchContext& context,
                                             const BoundTokens& bound)
{
    return std::make_unique<GenericPlan<Handle>>(
        engine.stateManager->getDispatchDetails(kernel), context, bound);
}

/// The launch @p plan's prepared dispatch holds, read back through its saved form.
ResolvedLaunch preparedLaunchOf(const GenericPlan<Handle>& plan)
{
    const auto saved = plan.handler().saveLaunchInputs(plan.prepared());
    if(!saved.has_value())
    {
        ADD_FAILURE() << "the declarative handler saved nothing";
        return {};
    }
    EXPECT_EQ(saved->dispatchSymbol, DECLARATIVE_LAUNCH_SYMBOL_V1);
    return restoreResolvedLaunch(saved->values);
}

// Holds the twin to the native engine on `graph`: the same kernels admitted, and for each
// one the same workspace, the same launch and the same output.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void expectParity(const LoadedTwins& twins, const ParityGraph& graph)
{
    SCOPED_TRACE(graph.name);
    const packs::GraphFixture fixture(graph.build(), twins.deviceProperties);
    const auto context = fixture.context();

    const auto nativeCatalog = twins.native->stateManager->unsortedCatalog(context);
    const auto declarativeCatalog = twins.declarative->stateManager->unsortedCatalog(context);
    ASSERT_FALSE(nativeCatalog.entries.empty()) << "the native engine does not serve this graph";
    const auto nativeKernels = byVariant(nativeCatalog);
    const auto declarativeKernels = byVariant(declarativeCatalog);
    ASSERT_EQ(declarativeKernels.size(), nativeKernels.size());

    auto tensors = hipdnn_test_sdk::utilities::GraphTensorBundle(context.graph.getTensorMap());
    tensors.getTensor(packs::CONV_X_UID).fillTensorWithRandomValues(-2.0F, 2.0F, 7);
    tensors.getTensor(packs::CONV_W_UID).fillTensorWithRandomValues(-2.0F, 2.0F, 11);
    tensors.outputTensorIds = {packs::CONV_Y_UID};

    const Handle handle;
    for(const auto& [variant, nativeKernel] : nativeKernels)
    {
        SCOPED_TRACE(variant);
        const auto twin = declarativeKernels.find(variant);
        ASSERT_NE(twin, declarativeKernels.end()) << "the declarative engine does not admit it";
        const KernelDefinition& declarativeKernel = *twin->second;
        ASSERT_NE(declarativeKernel.launch, nullptr);

        const auto nativePlan = planFor(*twins.native, *nativeKernel, context, nativeCatalog.bound);
        const auto declarativePlan
            = planFor(*twins.declarative, declarativeKernel, context, declarativeCatalog.bound);

        // The native launch, as its own handler saved it.
        const auto nativeSaved = nativePlan->handler().saveLaunchInputs(nativePlan->prepared());
        ASSERT_TRUE(nativeSaved.has_value());
        const auto nativeInputs = readConvFwdLaunchInputs(*nativeSaved);
        EXPECT_TRUE(preparedLaunchOf(*declarativePlan) == nativeLaunchOf(nativeInputs));
        EXPECT_EQ(declarativePlan->getWorkspaceSize(handle), nativePlan->getWorkspaceSize(handle));

        std::vector<uint8_t> nativeOutput;
        std::vector<uint8_t> declarativeOutput;
        ASSERT_NO_FATAL_FAILURE(packs::runPlan(*nativePlan, handle, tensors, nativeOutput));
        ASSERT_NO_FATAL_FAILURE(
            packs::runPlan(*declarativePlan, handle, tensors, declarativeOutput));
        EXPECT_EQ(declarativeOutput, nativeOutput);
    }
}

TEST(TestConvFwdDeclarativeLaunch, MatchesTheNativeDispatchOnEveryServedGraph)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    LoadedTwins twins;
    ASSERT_NO_FATAL_FAILURE(loadTwins(scratch, twins));
    if(twins.copy.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << twins.deviceProperties.gcnArchName
                     << ")";
    }

    for(const auto& graph : parityGraphs())
    {
        ASSERT_NO_FATAL_FAILURE(expectParity(twins, graph));
    }
}

/// A declarative plan saves its resolved launch under the versioned alias, restores
/// through the same handler, and computes the exact result the original plan did.
TEST(TestConvFwdDeclarativeLaunch, ASavedPlanRestoresAndComputesTheSameOutput)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    LoadedTwins twins;
    ASSERT_NO_FATAL_FAILURE(loadTwins(scratch, twins));
    if(twins.copy.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << twins.deviceProperties.gcnArchName
                     << ")";
    }

    const packs::GraphFixture fixture(packs::buildConvFwdPackedRoundTripGraph(),
                                      twins.deviceProperties);
    const auto context = fixture.context();
    const auto catalog = twins.declarative->stateManager->unsortedCatalog(context);
    ASSERT_EQ(catalog.entries.size(), 1U) << "one HALF kernel serves the round-trip graph";
    const auto plan = planFor(*twins.declarative, *catalog.entries.front(), context, catalog.bound);

    auto tensors = hipdnn_test_sdk::utilities::GraphTensorBundle(context.graph.getTensorMap());
    packs::fillConvFwdPackedInputs(tensors);
    tensors.outputTensorIds = {packs::CONV_Y_UID};

    const Handle handle;
    std::vector<uint8_t> original;
    ASSERT_NO_FATAL_FAILURE(packs::runPlan(*plan, handle, tensors, original));
    packs::expectConvFwdPackedOutput(original);

    const auto payload = captureIngestorPlan(*plan, handle);
    EXPECT_EQ(payload.dispatchSymbol, DECLARATIVE_LAUNCH_SYMBOL_V1);
    EXPECT_EQ(payload.launchValues, saveResolvedLaunch(preparedLaunchOf(*plan)));
    const auto encoded = serialization::encodeIngestorPlan(payload);

    const auto restored = restoreIngestorPlan(encoded.data(), encoded.size(), handle);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->getWorkspaceSize(handle), 0U);

    std::vector<uint8_t> output;
    ASSERT_NO_FATAL_FAILURE(packs::runPlan(*restored, handle, tensors, output));
    EXPECT_EQ(output, original);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
