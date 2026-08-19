// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestConvFwdPack.cpp
 * @brief The conv-forward pack's matcher shapes -- what it accepts and refuses -- plus
 *        the claim the engine split by graph node type exists to make: a conv graph and
 *        a pointwise graph each reach only their own matcher. Modelled on
 *        TestPointwiseAddMatchers.cpp; this pack has no operation-matcher section since
 *        it is the engine's only pack.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

bool matches(const MatchContext& context)
{
    BoundTokens bound;
    return matchesGraph(CONV_FWD, context, bound);
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: the supported case
// ---------------------------------------------------------------------------

TEST(TestConvFwdGraphMatcher, AcceptsUnitStrideNoPaddingCrossCorrelation)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestConvFwdGraphMatcher, AcceptsAHalfPrecisionConv)
{
    const GraphFixture fixture(buildConvFwdGraph(data_objects::DataType::HALF));

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestConvFwdBinding, BindsAllThreeOperandUids)
{
    const GraphFixture fixture(buildConvFwdGraph());

    BoundTokens bound;
    ASSERT_TRUE(matchesGraph(CONV_FWD, fixture.context(), bound));

    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.inputAToken), CONV_X_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.inputBToken), CONV_W_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CONV_FWD.outputToken), CONV_Y_UID);
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: refusals
// ---------------------------------------------------------------------------

/// One graph the matcher must refuse, plus a readable name for a failing run. The builder
/// is a plain function pointer, not std::function, since FlatBufferBuilder is move-only.
struct GraphMatcherRefusalCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

class TestConvFwdGraphMatcherRefusal : public ::testing::TestWithParam<GraphMatcherRefusalCase>
{
};

TEST_P(TestConvFwdGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestConvFwdGraphMatcherRefusal,
    ::testing::ValuesIn(std::vector<GraphMatcherRefusalCase>{
        {"StrideTwo",
         // The in-kernel p = h - r + 1 formula is only correct for unit stride.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{2, 2});
         }},
        {"DilationTwo",
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{2, 2});
         }},
        {"Padded",
         // The kernel's flat index arithmetic never adds padding.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{1, 1});
         }},
        {"ConvolutionMode",
         // Only CROSS_CORRELATION is supported; true CONVOLUTION flips the kernel,
         // which this reference implementation does not.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CONVOLUTION);
         }},
        {"CrossOperandDtypeMismatch",
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::nullopt,
                                      /*yDims=*/std::nullopt,
                                      /*wDataType=*/data_objects::DataType::HALF);
         }},
        {"APointwiseGraph",
         // This engine's matcher only ever admits a ConvolutionFwdAttributes node.
         []() { return buildPointwiseGraph(); }},
        {"FilterChannelsDisagreeWithInput",
         // w's channel count (4) disagrees with x's (1) -- also the group-count
         // refusal, since this pack has no notion of groups.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::vector<int64_t>{1, 4, 2, 2});
         }},
        {"OutputDimsInconsistentWithInputAndFilter",
         // y's shape disagrees with n/k/p/q, which is entirely what the kernel
         // actually computes it from -- a smaller y is an out-of-bounds write.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::nullopt,
                                      /*yDims=*/std::vector<int64_t>{1, 3, 9, 9});
         }},
        {"PostPaddingOnly",
         // Padding on one side is still padding; the flat index arithmetic never
         // adds any.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{1, 1});
         }},
        {"NonPackedStrides",
         // Valid strides, but not packed row-major; the kernel takes no strides of
         // its own and assumes contiguous NCHW.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3, 3},
                                      /*wDims=*/std::nullopt,
                                      /*yDims=*/std::nullopt,
                                      /*wDataType=*/std::nullopt,
                                      /*xStridesOverride=*/std::vector<int64_t>{9, 9, 1, 3});
         }},
        {"Rank3Tensors",
         // Rank 4 is required; a rank-3 x is refused before any cross-operand
         // comparison runs, so w/y here only need to be constructible.
         []() {
             return buildConvFwdGraph(data_objects::DataType::FLOAT,
                                      data_objects::ConvMode::CROSS_CORRELATION,
                                      /*stride=*/std::vector<int64_t>{1, 1},
                                      /*dilation=*/std::vector<int64_t>{1, 1},
                                      /*prePadding=*/std::vector<int64_t>{0, 0},
                                      /*postPadding=*/std::vector<int64_t>{0, 0},
                                      /*xDims=*/std::vector<int64_t>{1, 1, 3},
                                      /*wDims=*/std::vector<int64_t>{1, 1, 2, 2},
                                      /*yDims=*/std::vector<int64_t>{1, 1, 2, 2});
         }},
        {"UnsupportedDtype",
         // Only FLOAT and HALF are supported; the reference kernel has no other
         // instantiation.
         []() { return buildConvFwdGraph(data_objects::DataType::INT32); }},
    }),
    [](const ::testing::TestParamInfo<GraphMatcherRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// The engine split: each graph type reaches only its own matcher
// ---------------------------------------------------------------------------

/// The claim the split by graph node type exists to make: a conv graph never satisfies
/// the pointwise matcher, and a pointwise graph never satisfies the conv matcher.
TEST(TestConvFwdGraphMatcher, DoesNotOverlapWithThePointwiseEngine)
{
    const GraphFixture convFixture(buildConvFwdGraph());
    const GraphFixture pointwiseFixture(buildPointwiseGraph());

    BoundTokens bound;
    EXPECT_TRUE(matchesGraph(CONV_FWD, convFixture.context(), bound));
    EXPECT_FALSE(matchesGraph(CONV_FWD, pointwiseFixture.context(), bound));

    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, pointwiseFixture.context(), bound));
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, convFixture.context(), bound));
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestConvFwdKernelMatcher, AcceptsAKernelWhoseDtypeMatchesTheGraph)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_TRUE(matchesKernel(CONV_FWD, fixture.context(), makeKernel(64, "FLOAT", "ConvFwd")));
}

TEST(TestConvFwdKernelMatcher, RefusesAKernelBakedForAnotherDtype)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_FALSE(matchesKernel(CONV_FWD, fixture.context(), makeKernel(64, "HALF", "ConvFwd")));
}

TEST(TestConvFwdKernelMatcher, AcceptsAHalfKernelForAHalfGraph)
{
    const GraphFixture fixture(buildConvFwdGraph(data_objects::DataType::HALF));

    EXPECT_TRUE(matchesKernel(CONV_FWD, fixture.context(), makeKernel(64, "HALF", "ConvFwd")));
}

// ---------------------------------------------------------------------------
// Score
// ---------------------------------------------------------------------------

TEST(TestConvFwdScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_GT(scoreKernel(CONV_FWD, makeKernel(256, "FLOAT", "ConvFwd"), fixture.context()),
              scoreKernel(CONV_FWD, makeKernel(64, "FLOAT", "ConvFwd"), fixture.context()));
}

// ---------------------------------------------------------------------------
// Shipped descriptor set
// ---------------------------------------------------------------------------
//
// Every test above hand-builds KernelDefinitions via makeKernel() -- none of it loads
// conv_fwd/*.json. Without this section, a broken shipped descriptor (wrong entry_point,
// a missing kernel, a knob naming no KMD field) passes every unit test and only shows up
// in the slow GPU suite.

TEST(TestConvFwdPack, ShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(set.packs.size(), 1U);
    const auto& kernels = set.packs.front().kernels;
    ASSERT_EQ(kernels.size(), 3U);

    const auto describes = [&kernels](int64_t blockSize, const std::string& dtype) {
        return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD))) == blockSize
                   && std::get<std::string>(kernel.metadata.at(std::string(DTYPE_FIELD))) == dtype
                   && kernel.source.entryPoint == "ConvFwd";
        });
    };

    EXPECT_TRUE(describes(64, "FLOAT"));
    EXPECT_TRUE(describes(256, "FLOAT"));
    EXPECT_TRUE(describes(64, "HALF"));
}

TEST(TestConvFwdPack, ExposesBlockSizeAsTheOneKnob)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestConvFwdPack, HasOneGraphMatcherAndOneKernelMatcher)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::GRAPH;
                            }),
              1);
    EXPECT_EQ(std::count_if(set.matchers.begin(),
                            set.matchers.end(),
                            [](const auto& matcher) {
                                return matcher.scope
                                       == hipdnn_plugin_sdk::ingestor::MatchScope::KERNEL;
                            }),
              1);
}

// ---------------------------------------------------------------------------
// Dispatch: this pack's IKernelDispatchHandler, unreached above
// ---------------------------------------------------------------------------
//
// Every test above resolves the graph/kernel matchers and the scorer directly;
// dispatchHandler(CONV_FWD) -- prepare()/workspaceBytes()/launch() in ConvNative.cpp --
// is never touched here, so short of the slow GPU integration test, nothing catches a
// broken registration or a broken prepare() unhappy path.

/// Bindings a real plan build would hand the handler, from running the graph matcher.
BoundTokens convBindingsFor(const MatchContext& context)
{
    BoundTokens bound;
    if(!matchesGraph(CONV_FWD, context, bound))
    {
        throw std::logic_error("test graph does not match the conv pack");
    }
    return bound;
}

/// If IngestorPacks drops the ConvFwd row, or registerConvFwdSymbols stops registering
/// DISPATCH_SYMBOL, this resolves to nullptr and every plan build null-derefs at
/// dispatch time -- nothing else in the fast suite asks the registry for this symbol.
TEST(TestConvFwdDispatch, DispatchSymbolResolves)
{
    registerNativeIngestorSymbols();
    EXPECT_NE(hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
                  std::string(CONV_FWD.dispatch)),
              nullptr);
}

/// This reference kernel needs no scratch -- every output element is accumulated once
/// and written directly. Its only caller (execution-plan build) never checks the value
/// it returns, so this is the only place pinning workspaceBytes() to 0.
TEST(TestConvFwdDispatch, WorkspaceBytesIsAlwaysZero)
{
    const GraphFixture fixture(buildConvFwdGraph());
    const auto& handler = dispatchHandler(CONV_FWD);

    EXPECT_EQ(handler.workspaceBytes(fixture.context(),
                                     convBindingsFor(fixture.context()),
                                     makeKernel(64, "FLOAT", "ConvFwd")),
              0U);
}

/// convFwdBinding() must throw before touching HIP: empty BoundTokens is what a
/// mismatched matcher's catalog entry would hand prepare(). Without this guard, a plan
/// build reads uninitialized bindings and launches wrong tensors or crashes in HIP
/// instead of failing cleanly at plan-build time.
TEST(TestConvFwdDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildConvFwdGraph());
    const auto& handler = dispatchHandler(CONV_FWD);

    EXPECT_THROW(
        handler.prepare(fixture.context(), BoundTokens{}, makeKernel(64, "FLOAT", "ConvFwd")),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

/// elementTypeFor() throws on a dtype its `if` branches don't name, reached only inside
/// prepare() after binding/lookup succeed. Distinct from the kernel matcher's
/// RefusesAKernelBakedForAnotherDtype above, which returns false before dispatch and
/// would keep passing even if this throw were deleted.
TEST(TestConvFwdDispatch, PrepareRejectsAKernelDeclaringAnUnsupportedDtype)
{
    const GraphFixture fixture(buildConvFwdGraph());
    const auto& handler = dispatchHandler(CONV_FWD);

    EXPECT_THROW(handler.prepare(fixture.context(),
                                 convBindingsFor(fixture.context()),
                                 makeKernel(64, "BFLOAT16", "ConvFwd")),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// Shipped descriptor set: source kind/file, behavior notes, operation metadata
// ---------------------------------------------------------------------------

/// Pins kernel_source.kind and source_file, the conv analogue of TestPointwisePacks.cpp's
/// EveryKernelNamesItsPacksEmbeddedSource. Without it, a bad source kind or a misspelled
/// source_file passes this whole fast suite -- prepare() only discovers it when the slow
/// GPU integration test tries to compile it.
TEST(TestConvFwdPack, PinsTheEmbeddedConvSource)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(set.packs.size(), 1U);
    ASSERT_FALSE(set.packs.front().kernels.empty());
    for(const auto& kernel : set.packs.front().kernels)
    {
        EXPECT_EQ(kernel.source.kind,
                  hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
        EXPECT_EQ(kernel.source.sourceFile, "ConvFwd.cpp");
        EXPECT_EQ(kernel.source.entryPoint, "ConvFwd");
    }
}

/// behavior_notes: ["runtime_compilation"] is parsed by the loader onto the
/// EngineDetails flatbuffer, but nothing else asserts it survives. Losing it (a dropped
/// descriptor line, or a loader regression) would tell a caller-facing framework this
/// engine never JIT-compiles, which is false for both engines here.
TEST(TestConvFwdPack, BothShippedEnginesDeclareRuntimeCompilation)
{
    for(const std::string_view engineName : {"hipkernel:Pointwise", "hipkernel:ConvFwd"})
    {
        const auto& notes = loadedSet(engineName).engine.behaviorNotes;
        EXPECT_NE(std::find(notes.begin(),
                            notes.end(),
                            static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION)),
                  notes.end())
            << engineName;
    }
}

/// Nothing in production reads a kernel's "operation" metadata -- matching runs on
/// block_size/dtype alone. All three Pointwise packs could ship "operation": "ADD" and
/// every other test would keep passing; this is the one place tying each pack's kernels
/// back to the operation its own source file actually implements.
TEST(TestConvFwdPack, PointwisePacksClaimTheOperationTheyActuallyImplement)
{
    const auto& set = loadedSet("hipkernel:Pointwise");

    for(const auto& expected : {std::pair{"PointwiseAdd.cpp", "ADD"},
                                std::pair{"PointwiseMul.cpp", "MUL"},
                                std::pair{"PointwiseSub.cpp", "SUB"}})
    {
        // Not a structured binding: capturing one in the lambda below is C++20, and this
        // project is C++17.
        const auto* const sourceFile = expected.first;
        const auto* const operation = expected.second;

        const auto pack = std::find_if(set.packs.begin(), set.packs.end(), [&](const auto& p) {
            return !p.kernels.empty() && p.kernels.front().source.sourceFile == sourceFile;
        });
        ASSERT_NE(pack, set.packs.end()) << sourceFile;

        for(const auto& kernel : pack->kernels)
        {
            EXPECT_EQ(std::get<std::string>(kernel.metadata.at(std::string(OPERATION_FIELD))),
                      operation)
                << kernel.name;
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
