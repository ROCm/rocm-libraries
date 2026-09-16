// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestBatchnormInferencePacks.cpp
 * @brief The hipkernel:BatchnormInference descriptor set as installed, and the dispatch
 *        handler the matcher tests never reach.
 *
 * The matcher suite hand-builds KernelDefinitions, so none of it reads the JSON under
 * descriptors/batchnorm_inference. Without this file a wrong entry_point, a dropped
 * kernel or a knob naming no KMD field passes the whole fast suite and only surfaces
 * when the GPU suite tries to compile it.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;

// ---------------------------------------------------------------------------
// Shipped descriptor set
// ---------------------------------------------------------------------------

TEST(TestBatchnormInferencePack, ShipsEveryIoDtypeCrossedWithEveryBlockSize)
{
    const auto& set = loadedSet("hipkernel:BatchnormInference");

    ASSERT_EQ(set.packs.size(), 1U);
    const auto& kernels = set.packs.front().kernels;
    ASSERT_EQ(kernels.size(), 9U);

    const auto describes = [&kernels](int64_t blockSize, std::string_view ioDtype) {
        return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD))) == blockSize
                   && std::get<std::string>(kernel.metadata.at("io_dtype")) == ioDtype
                   && std::get<std::string>(kernel.metadata.at("param_dtype")) == "FLOAT";
        });
    };

    // The full cross-product, enumerated rather than sampled: block size is a measured
    // axis, so a missing cell is a candidate the sweep can never select and would not
    // otherwise be noticed.
    for(const auto* ioDtype : {"FLOAT", "HALF", "BFLOAT16"})
    {
        EXPECT_TRUE(describes(64, ioDtype)) << ioDtype;
        EXPECT_TRUE(describes(256, ioDtype)) << ioDtype;
        EXPECT_TRUE(describes(1024, ioDtype)) << ioDtype;
    }
}

/// Every corpus graph pairs a float or 16-bit io dtype with float parameters, so no
/// candidate ships for a 16-bit parameter type even though the kernel's header accepts
/// one and the pairing compiles. This is the test that stops a candidate for an
/// unselectable -- and unvalidated -- parameter dtype being added and silently ranked.
TEST(TestBatchnormInferencePack, ShipsNoCandidateForAnUnprovedDtype)
{
    const auto& set = loadedSet("hipkernel:BatchnormInference");

    for(const auto& kernel : set.packs.front().kernels)
    {
        EXPECT_EQ(std::get<std::string>(kernel.metadata.at("param_dtype")), "FLOAT") << kernel.name;

        const auto ioDtype = std::get<std::string>(kernel.metadata.at("io_dtype"));
        EXPECT_TRUE(ioDtype == "FLOAT" || ioDtype == "HALF" || ioDtype == "BFLOAT16")
            << kernel.name << " io_dtype=" << ioDtype;
    }
}

TEST(TestBatchnormInferencePack, PinsTheEmbeddedKernelSource)
{
    const auto& set = loadedSet("hipkernel:BatchnormInference");

    ASSERT_FALSE(set.packs.front().kernels.empty());
    for(const auto& kernel : set.packs.front().kernels)
    {
        EXPECT_EQ(kernel.source.kind,
                  hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
        // The embedding helper keys its map on the file name, and the entry point is the
        // symbol hipRTC resolves. A mismatch in either is a plan-build failure at the
        // first dispatch and is invisible until then.
        EXPECT_EQ(kernel.source.sourceFile, "BatchnormInference.cpp");
        EXPECT_EQ(kernel.source.entryPoint, "BatchnormInference");
    }
}

TEST(TestBatchnormInferencePack, ExposesBlockSizeAsTheOneKnob)
{
    const auto& set = loadedSet("hipkernel:BatchnormInference");

    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestBatchnormInferencePack, HasAGraphMatchAndOneKernelMatcher)
{
    const auto& set = loadedSet("hipkernel:BatchnormInference");

    // Single pack, so there is nothing for a graph-scoped criterion to discriminate:
    // graph_match both admits the node type and validates the operands.
    EXPECT_FALSE(set.engine.graphMatchNativeSymbol.empty());
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              0);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

TEST(TestBatchnormInferencePack, DeclaresRuntimeCompilation)
{
    const auto& notes = loadedSet("hipkernel:BatchnormInference").engine.behaviorNotes;

    EXPECT_NE(std::find(notes.begin(),
                        notes.end(),
                        static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION)),
              notes.end());
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Bindings a real plan build would hand the handler, from running the graph match.
BoundTokens batchnormBindingsFor(const MatchContext& context)
{
    auto bound = matchesGraph(BATCHNORM_INFERENCE, context);
    if(!bound.has_value())
    {
        throw std::logic_error("test graph does not match the batchnorm_inference pack");
    }
    return std::move(*bound);
}

/// If IngestorPacks drops this pack's row, or registerBatchnormInferenceSymbols stops
/// registering the dispatch symbol, this resolves to nullptr and every plan build
/// null-derefs at dispatch time.
TEST(TestBatchnormInferenceDispatch, DispatchSymbolResolves)
{
    registerNativeIngestorSymbols();
    EXPECT_NE(hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
                  std::string(BATCHNORM_INFERENCE.dispatch)),
              nullptr);
}

/// Zero, and it must stay zero: one launch, no virtual tensors, no scratch. The plan
/// builder never checks the value, so this is the only place pinning it.
TEST(TestBatchnormInferenceDispatch, WorkspaceBytesIsAlwaysZero)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());
    const auto& handler = dispatchHandler(BATCHNORM_INFERENCE);

    EXPECT_EQ(handler.workspaceBytes(fixture.context(),
                                     batchnormBindingsFor(fixture.context()),
                                     makeBatchnormKernel(64)),
              0U);
}

/// Empty BoundTokens is what a mismatched catalog entry would hand prepare(). Without
/// this the handler reads uninitialized bindings and launches whatever uid 0 resolves
/// to, instead of failing cleanly at plan-build time.
TEST(TestBatchnormInferenceDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());
    const auto& handler = dispatchHandler(BATCHNORM_INFERENCE);

    EXPECT_THROW(handler.prepare(fixture.context(), BoundTokens{}, makeBatchnormKernel(64)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// dtypeTagFor() throws on a tag BatchnormInferenceTypes.h does not define, reached only
/// inside prepare() after binding succeeds. Distinct from the kernel matcher's refusal,
/// which returns false earlier and would keep passing if this throw were deleted --
/// leaving an unbound HIPDNN_BN_IO_DTYPE to paste into the header's #error instead.
TEST(TestBatchnormInferenceDispatch, PrepareRejectsAKernelDeclaringAnUndefinedDtypeTag)
{
    const GraphFixture fixture(buildBatchnormInferenceGraph());
    const auto& handler = dispatchHandler(BATCHNORM_INFERENCE);

    EXPECT_THROW(handler.prepare(fixture.context(),
                                 batchnormBindingsFor(fixture.context()),
                                 makeBatchnormKernel(64, "FLOAT64", "FLOAT")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
