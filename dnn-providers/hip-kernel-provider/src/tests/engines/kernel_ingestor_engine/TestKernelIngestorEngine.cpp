// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
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
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestKernelIngestorEngine.cpp
 * @brief Tests registerNativeIngestorSymbols(), makePointwiseAddEngine() and the computed
 *        execution plan serialization note;
 *        GenericEngine itself is covered by the SDK's suite. The ingestor engine tests go
 *        through Container and EngineManager, since makePointwiseAddEngine() takes no
 *        injectable seams.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hip_kernel_provider::core::Container;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_plugin_sdk::ingestor::DescriptorSet;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::MockEngineConfig;
using hipdnn_test_sdk::utilities::ScopedDirectory;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* SCRATCH_LABEL = "ingestorengine";

GraphWrapper wrap(const flatbuffers::FlatBufferBuilder& builder)
{
    return GraphWrapper(builder.GetBufferPointer(), builder.GetSize());
}

// SymbolScope stand-ins: content-indifferent, and a pack's real functions are internal
// to their native file and unreachable from here.
std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    acceptAnyGraph(const hipdnn_plugin_sdk::ingestor::MatchContext& /*context*/)
{
    return hipdnn_plugin_sdk::ingestor::BoundTokens{};
}

double scoreNothing(const hipdnn_plugin_sdk::ingestor::MatchContext& /*context*/,
                    const hipdnn_plugin_sdk::ingestor::BoundTokens& /*bound*/,
                    const hipdnn_plugin_sdk::ingestor::KernelDefinition& /*kernel*/)
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
    using hipdnn_plugin_sdk::ingestor::GraphMatchRegistry;
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

    EXPECT_THROW(GraphMatchRegistry::resolve(firstSymbol), std::runtime_error);
    // ...while the neighbour's survives: one pack failing must not affect others.
    EXPECT_NO_THROW(GraphMatchRegistry::resolve(neighbourSymbol));
    EXPECT_NO_THROW(ScoreRegistry::resolve(contendedSymbol));

    GraphMatchRegistry::unregisterSymbol(neighbourSymbol);
    ScoreRegistry::unregisterSymbol(contendedSymbol);
}

TEST(TestKernelIngestorEngine, ACommittedScopeKeepsItsSymbols)
{
    using hipdnn_plugin_sdk::ingestor::GraphMatchRegistry;
    using hipdnn_plugin_sdk::ingestor::SymbolScope;

    const std::string symbol = "test.committed.graph_match";
    {
        SymbolScope<Handle> scope;
        scope.add(symbol, &acceptAnyGraph);
        scope.commit();
    }

    EXPECT_NO_THROW(GraphMatchRegistry::resolve(symbol));

    GraphMatchRegistry::unregisterSymbol(symbol);
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
    // block_size plus the benchmarking knob every descriptor-backed engine advertises
    // out-of-band; looked up by name, since that knob is prepended.
    ASSERT_EQ(wrapper.knobCount(), 2U);
    EXPECT_EQ(wrapper.getKnobByName("block_size").knobId(), "block_size");
    EXPECT_EQ(wrapper.getKnobByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME).knobId(),
              hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);
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
    const ScopedDirectory existing = claimScratchDirectory(SCRATCH_LABEL);
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
    EXPECT_TRUE(resolved.generic_string().find(HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR)
                != std::string::npos)
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
    EXPECT_TRUE(resolved.generic_string().find(HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR)
                != std::string::npos)
        << "resolved to " << resolved << ", which does not end in "
        << HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR;
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

// ---------------------------------------------------------------------------
// descriptorSearchDirectories(): appending the runtime drop-in root
// ---------------------------------------------------------------------------

/// With no runtime dir set, the provider's own tree is the only root -- no phantom
/// second entry. HIPDNN_DESCRIPTOR_RUNTIME_DIR and HIPDNN_DESCRIPTOR_PATH may already be
/// set from an outer shell, so both are cleared here the same way
/// FallsBackToAModuleRelativeOrInstalledPath clears HIPDNN_DESCRIPTOR_DIR: an explicit
/// empty value, which the loader treats as absent. Every test below that counts roots
/// clears HIPDNN_DESCRIPTOR_PATH for the same reason: each of its entries is another root.
TEST(TestKernelIngestorEngine, ReturnsOnlyTheProviderTreeWhenRuntimeDirIsUnset)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter unset(
        "HIPDNN_DESCRIPTOR_RUNTIME_DIR", "");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter noSearchPath(
        "HIPDNN_DESCRIPTOR_PATH", "");

    const auto roots = descriptorSearchDirectories();

    ASSERT_EQ(roots.size(), 1U);
    EXPECT_EQ(roots.front(), descriptorSearchDirectory());
}

/// A real runtime dir is additive, not a replacement: the provider's own tree still
/// leads, the runtime dir lands second. Order matters to the loader's incumbent-wins
/// duplicate rule, so it's asserted position by position rather than as a set.
TEST(TestKernelIngestorEngine, AppendsHipdnnDescriptorRuntimeDirAfterTheProviderTree)
{
    const ScopedDirectory runtimeDir = claimScratchDirectory(SCRATCH_LABEL);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter override(
        "HIPDNN_DESCRIPTOR_RUNTIME_DIR", runtimeDir.path().string());
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter noSearchPath(
        "HIPDNN_DESCRIPTOR_PATH", "");

    const auto roots = descriptorSearchDirectories();

    ASSERT_EQ(roots.size(), 2U);
    EXPECT_EQ(roots[0], descriptorSearchDirectory());
    EXPECT_EQ(roots[1], runtimeDir.path());
}

/// A stale runtime dir must not add a root at all: the loader treats a missing root as
/// "nothing to add", so a typo would otherwise silently vanish instead of failing loud.
TEST(TestKernelIngestorEngine, IgnoresAHipdnnDescriptorRuntimeDirThatDoesNotExist)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter stale(
        "HIPDNN_DESCRIPTOR_RUNTIME_DIR", "/nowhere/in/particular");
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter noSearchPath(
        "HIPDNN_DESCRIPTOR_PATH", "");

    const auto roots = descriptorSearchDirectories();

    ASSERT_EQ(roots.size(), 1U);
    EXPECT_EQ(roots.front(), descriptorSearchDirectory());
}

// ---------------------------------------------------------------------------
// The execution plan serialization note
// ---------------------------------------------------------------------------

constexpr int32_t SERIALIZATION_NOTE
    = static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION);
constexpr int32_t RUNTIME_COMPILATION_NOTE
    = static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION);

constexpr const char* GFX1151_WMMA_ATTENTION_DISPATCH_SYMBOL
    = "hipkernel.gfx1151_wmma_attention.dispatch";

// A set with one pack for each entry of `packs`, and one kernel for each source kind in it.
// Each pack has its own dispatch, which names `dispatchSymbol`.
DescriptorSet setWithKernels(const std::vector<std::vector<KernelSourceKind>>& packs,
                             std::string_view dispatchSymbol)
{
    DescriptorSet set{};
    for(const auto& kinds : packs)
    {
        hipdnn_plugin_sdk::ingestor::DispatchDescriptor dispatch{};
        dispatch.id.back() = static_cast<uint8_t>(set.dispatches.size() + 1);
        dispatch.dispatchSymbol = std::string(dispatchSymbol);

        hipdnn_plugin_sdk::ingestor::KernelDescriptorPack pack{};
        pack.dispatchId = dispatch.id;
        for(const auto kind : kinds)
        {
            pack.kernels.emplace_back().source.kind = kind;
        }
        set.dispatches.push_back(std::move(dispatch));
        set.packs.push_back(std::move(pack));
    }
    return set;
}

TEST(TestKernelIngestorEngine, SupportsSerializationOnlyWhenEveryKernelPassesTheGate)
{
    EXPECT_TRUE(supportsExecutionPlanSerialization(setWithKernels(
        {{KernelSourceKind::KPACK}, {KernelSourceKind::KPACK, KernelSourceKind::KPACK}},
        CONV_FWD.dispatch)));

    // Saving accepts only kpack kernels: embedded_source kernels have no recorded argument
    // signature, and their code bytes are not exposed.
    EXPECT_FALSE(supportsExecutionPlanSerialization(setWithKernels(
        {{KernelSourceKind::KPACK}, {KernelSourceKind::KPACK, KernelSourceKind::EMBEDDED_SOURCE}},
        CONV_FWD.dispatch)));

    EXPECT_FALSE(supportsExecutionPlanSerialization(DescriptorSet{}));
    EXPECT_FALSE(supportsExecutionPlanSerialization(setWithKernels({{}, {}}, CONV_FWD.dispatch)));
}

TEST(TestKernelIngestorEngine, NoNoteWhenAHandlerCannotSave)
{
    using hipdnn_plugin_sdk::ingestor::DispatchRegistry;
    using hipdnn_plugin_sdk::ingestor::SymbolScope;

    const std::string symbol = "hipkernel.engine_test.no_save.dispatch";
    const packs::StubDispatchHandler handler;
    // Not committed: the symbol is unregistered again when the scope ends.
    SymbolScope<Handle> scope;
    scope.add(symbol, &handler);

    EXPECT_FALSE(
        supportsExecutionPlanSerialization(setWithKernels({{KernelSourceKind::KPACK}}, symbol)));

    DescriptorSet oneHandlerCannotSave
        = setWithKernels({{KernelSourceKind::KPACK}, {KernelSourceKind::KPACK}}, CONV_FWD.dispatch);
    oneHandlerCannotSave.dispatches.back().dispatchSymbol = symbol;
    EXPECT_FALSE(supportsExecutionPlanSerialization(oneHandlerCannotSave));

    registerNativeIngestorSymbols();
    ASSERT_NE(DispatchRegistry<Handle>::tryResolve(GFX1151_WMMA_ATTENTION_DISPATCH_SYMBOL),
              nullptr);
    EXPECT_FALSE(supportsExecutionPlanSerialization(
        setWithKernels({{KernelSourceKind::KPACK}}, GFX1151_WMMA_ATTENTION_DISPATCH_SYMBOL)));
}

TEST(TestKernelIngestorEngine, NoNoteWithoutAResolvableDispatch)
{
    using hipdnn_plugin_sdk::ingestor::DispatchRegistry;

    DescriptorSet missingDispatch
        = setWithKernels({{KernelSourceKind::KPACK}, {KernelSourceKind::KPACK}}, CONV_FWD.dispatch);
    missingDispatch.dispatches.pop_back();
    EXPECT_FALSE(supportsExecutionPlanSerialization(missingDispatch));

    const std::string unregistered = "hipkernel.engine_test.unregistered.dispatch";
    ASSERT_EQ(DispatchRegistry<Handle>::tryResolve(unregistered), nullptr);
    EXPECT_FALSE(supportsExecutionPlanSerialization(
        setWithKernels({{KernelSourceKind::KPACK}}, unregistered)));
}

TEST(TestKernelIngestorEngine, WithComputedBehaviorNotesAddsTheNoteOnce)
{
    DescriptorSet saveable
        = setWithKernels({{KernelSourceKind::KPACK}, {KernelSourceKind::KPACK}}, CONV_FWD.dispatch);
    saveable.engine.behaviorNotes = {RUNTIME_COMPILATION_NOTE};

    const std::vector<int32_t> expected{RUNTIME_COMPILATION_NOTE, SERIALIZATION_NOTE};
    const DescriptorSet once = withComputedBehaviorNotes(saveable);
    EXPECT_EQ(once.engine.behaviorNotes, expected);
    EXPECT_EQ(withComputedBehaviorNotes(once).engine.behaviorNotes, expected);

    DescriptorSet mixed = setWithKernels(
        {{KernelSourceKind::KPACK}, {KernelSourceKind::EMBEDDED_SOURCE}}, CONV_FWD.dispatch);
    mixed.engine.behaviorNotes = {RUNTIME_COMPILATION_NOTE};
    EXPECT_EQ(withComputedBehaviorNotes(mixed).engine.behaviorNotes,
              std::vector<int32_t>{RUNTIME_COMPILATION_NOTE});
}

TEST(TestKernelIngestorEngine, ReportsTheSerializationNoteOnlyOnSaveableEngines)
{
    // getDetails reads the knob values from the device.
    SKIP_IF_NO_DEVICES();

    ASSERT_TRUE(supportsExecutionPlanSerialization(loadedSet(CONV_FWD.engineName)));
    ASSERT_FALSE(supportsExecutionPlanSerialization(loadedSet(POINTWISE_ADD.engineName)));

    Handle handle;

    const auto notesOf = [&handle](Container& container,
                                   std::string_view engineName,
                                   const flatbuffers::FlatBufferBuilder& graph) {
        hipdnnPluginConstData_t details{};
        container.getEngineManager().getEngineDetails(
            handle,
            wrap(graph),
            hipdnn_data_sdk::utilities::engineNameToId(std::string(engineName)),
            details);
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper(details.ptr,
                                                                                  details.size)
            .behaviorNotes();
    };
    const auto expectTheNoteOnlyOnConvFwd = [&notesOf](Container& container) {
        const auto convFwdNotes = notesOf(container, CONV_FWD.engineName, buildConvFwdGraph());
        EXPECT_EQ(std::count(convFwdNotes.begin(), convFwdNotes.end(), SERIALIZATION_NOTE), 1);

        const auto pointwiseNotes
            = notesOf(container, POINTWISE_ADD.engineName, buildPointwiseGraph());
        EXPECT_EQ(std::count(pointwiseNotes.begin(), pointwiseNotes.end(), SERIALIZATION_NOTE), 0);
    };

    // The first Container takes the state managers that discovery built; a later one builds
    // its own. Both construction paths must report the note.
    rediscoverStateManagersForTesting();
    {
        SCOPED_TRACE("first Container");
        Container container;
        ASSERT_EQ(takeDiscoveredStateManager(loadedSet(CONV_FWD.engineName).engine.id).get(),
                  nullptr);
        expectTheNoteOnlyOnConvFwd(container);
    }
    {
        SCOPED_TRACE("later Container");
        Container container;
        expectTheNoteOnlyOnConvFwd(container);
    }
}

// The SDK defaults return nullopt and nullptr without throwing, so only a throw is
// evidence of an override. A handler's own save down-casts the prepared dispatch, which
// rejects a foreign one with std::bad_cast; its own restore runs the shared restore checks,
// which refuse an empty saved plan with HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE. Both reject
// before any GPU work. The registry has no enumeration, so the handlers come from the
// discovered sets and from one dispatch symbol for each pack.
TEST(TestKernelIngestorEngine, EveryHandlerReportsSupportsSavingExactlyWhenItSavesAndRestores)
{
    using hipdnn_plugin_sdk::HipdnnPluginException;
    using hipdnn_plugin_sdk::ingestor::DispatchRegistry;
    using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
    using hipdnn_plugin_sdk::ingestor::PreparedDispatch;
    using hipdnn_plugin_sdk::ingestor::SavedKernelCode;
    using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

    constexpr std::array<std::string_view, 5> PACK_DISPATCH_SYMBOLS{
        "hipkernel.pointwise.dispatch",
        "hipkernel.conv_fwd.dispatch",
        "hipkernel.gfx950_attention_dense.dispatch",
        "hipkernel.gfx1151_wmma_attention.dispatch",
        "hipkernel.gfx1151_wmma_attention_decode.dispatch"};
    EXPECT_EQ(PACK_DISPATCH_SYMBOLS.size(), ingestorPacks().size())
        << "add the dispatch symbol of each new pack to this list";

    // Each handler once, with the first symbol that resolved to it.
    std::map<const IKernelDispatchHandler<Handle>*, std::string> handlers;
    for(const auto& set : discoverDescriptorSets())
    {
        for(const auto& pack : set.packs)
        {
            const auto dispatch
                = std::find_if(set.dispatches.begin(),
                               set.dispatches.end(),
                               [&pack](const auto& entry) { return entry.id == pack.dispatchId; });
            ASSERT_NE(dispatch, set.dispatches.end()) << set.engine.name << ": " << pack.name;
            const auto* handler = DispatchRegistry<Handle>::resolve(dispatch->dispatchSymbol);
            ASSERT_NE(handler, nullptr) << dispatch->dispatchSymbol;
            handlers.emplace(handler, dispatch->dispatchSymbol);
        }
    }
    registerNativeIngestorSymbols();
    for(const auto symbol : PACK_DISPATCH_SYMBOLS)
    {
        const auto* handler = DispatchRegistry<Handle>::tryResolve(std::string(symbol));
        ASSERT_NE(handler, nullptr) << symbol;
        handlers.emplace(handler, std::string(symbol));
    }

    size_t saving = 0;
    size_t notSaving = 0;
    for(const auto& [handler, symbol] : handlers)
    {
        bool saves = false;
        try
        {
            const PreparedDispatch foreign{};
            static_cast<void>(handler->saveLaunchInputs(foreign));
        }
        catch(const std::bad_cast&)
        {
            saves = true;
        }

        bool restores = false;
        try
        {
            static_cast<void>(handler->restoreLaunch(SavedLaunchInputs{}, SavedKernelCode{}, 0));
        }
        catch(const HipdnnPluginException& refusal)
        {
            EXPECT_EQ(refusal.getStatus(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE)
                << symbol << ": " << refusal.getMessage();
            restores = true;
        }

        EXPECT_EQ(saves, restores) << symbol << ": saves " << saves << ", restores " << restores;
        EXPECT_EQ(handler->supportsSaving(), saves && restores)
            << symbol << ": supportsSaving() disagrees with the save and restore overrides";
        if(saves && restores)
        {
            ++saving;
        }
        else
        {
            ++notSaving;
        }
    }
    EXPECT_GT(saving, 0U) << "no handler that saves was probed";
    EXPECT_GT(notSaving, 0U) << "no handler that cannot save was probed";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
