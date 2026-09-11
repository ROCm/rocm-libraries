// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestConvFwdPack.cpp
 * @brief The conv-forward pack's matcher shapes -- what it accepts and refuses -- plus
 *        the claim the engine split by graph node type exists to make: a conv graph and
 *        a pointwise graph each reach only their own matcher. Modelled on
 *        TestPointwiseAddMatchers.cpp; this pack has no operation-matcher section since
 *        it is the engine's only pack.
 *
 *        Also the kpack dispatch case, which lives here because this is the pack whose
 *        descriptors the unit binary's own root stages with a real archive.
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
    return matchesGraph(CONV_FWD, context).has_value();
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

    const auto bound = matchesGraph(CONV_FWD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, CONV_FWD.inputAToken),
              CONV_X_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, CONV_FWD.inputBToken),
              CONV_W_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, CONV_FWD.outputToken),
              CONV_Y_UID);
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

    EXPECT_TRUE(matchesGraph(CONV_FWD, convFixture.context()).has_value());
    EXPECT_FALSE(matchesGraph(CONV_FWD, pointwiseFixture.context()).has_value());

    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, pointwiseFixture.context()).has_value());
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, convFixture.context()).has_value());
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

    EXPECT_GT(scoreKernel(CONV_FWD, fixture.context(), makeKernel(256, "FLOAT", "ConvFwd")),
              scoreKernel(CONV_FWD, fixture.context(), makeKernel(64, "FLOAT", "ConvFwd")));
}

// ---------------------------------------------------------------------------
// Shipped descriptor set
// ---------------------------------------------------------------------------
//
// Every test above hand-builds KernelDefinitions via makeKernel() -- none of it loads
// conv_fwd/*.json. Without this section, a broken shipped descriptor (wrong symbol, a
// missing kernel, a knob naming no KMD field) passes every unit test and only shows up in
// the slow GPU suite.

TEST(TestConvFwdPack, ShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(distinctPackIdCount(set), 1U);
    for(const auto& pack : set.packs)
    {
        const auto& kernels = pack.kernels;
        ASSERT_EQ(kernels.size(), 3U) << pack.name;

        const auto describes = [&kernels](int64_t blockSize, const std::string& dtype) {
            return std::any_of(kernels.begin(), kernels.end(), [&](const auto& kernel) {
                return std::get<int64_t>(kernel.metadata.at(std::string(BLOCK_SIZE_FIELD)))
                           == blockSize
                       && std::get<std::string>(kernel.metadata.at(std::string(DTYPE_FIELD)))
                              == dtype
                       && kernel.source.symbol == "ConvFwd";
            });
        };

        EXPECT_TRUE(describes(64, "FLOAT")) << pack.name;
        EXPECT_TRUE(describes(256, "FLOAT")) << pack.name;
        EXPECT_TRUE(describes(64, "HALF")) << pack.name;
    }
}

TEST(TestConvFwdPack, ExposesBlockSizeAsTheOneKnob)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(set.engine.knobs.size(), 1U);
    EXPECT_EQ(set.engine.knobs.front(), std::string(BLOCK_SIZE_FIELD));
}

TEST(TestConvFwdPack, HasAGraphMatchAndOneKernelMatcher)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    // A single-pack engine has nothing to discriminate between, so it carries no
    // graph-scoped criterion at all: the engine's graph_match both admits the node type
    // and fully validates the shape.
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

// ---------------------------------------------------------------------------
// Dispatch: this pack's IKernelDispatchHandler, unreached above
// ---------------------------------------------------------------------------
//
// Every test above resolves the graph/kernel matchers and the scorer directly;
// dispatchHandler(CONV_FWD) -- prepare()/workspaceBytes()/launch() in ConvNative.cpp --
// is never touched here, so short of the slow GPU integration test, nothing catches a
// broken registration or a broken prepare() unhappy path.

/// Bindings a real plan build would hand the handler, from running the graph match.
BoundTokens convBindingsFor(const MatchContext& context)
{
    auto bound = matchesGraph(CONV_FWD, context);
    if(!bound.has_value())
    {
        throw std::logic_error("test graph does not match the conv pack");
    }
    return std::move(*bound);
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

/// Pins kernel_source.kind and the archive coordinates the packer wrote, the conv
/// analogue of TestPointwisePacks.cpp's EveryKernelNamesItsPacksEmbeddedSource. Without
/// it, a bad source kind, a dropped library reference or a misspelled symbol passes this
/// whole fast suite -- prepare() only discovers it when the slow GPU integration test
/// tries to load the archive.
TEST(TestConvFwdPack, PinsThePackedConvSource)
{
    const auto& set = loadedSet("hipkernel:ConvFwd");

    ASSERT_EQ(distinctPackIdCount(set), 1U);
    for(const auto& pack : set.packs)
    {
        ASSERT_FALSE(pack.kernels.empty()) << pack.name;
        for(const auto& kernel : pack.kernels)
        {
            EXPECT_EQ(kernel.source.kind, hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK);
            EXPECT_FALSE(kernel.source.library.empty()) << kernel.name;
            EXPECT_EQ(kernel.source.symbol, "ConvFwd");
        }
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

    for(const auto& expected : {std::pair{"kernels/PointwiseAdd.cpp", "ADD"},
                                std::pair{"kernels/PointwiseMul.cpp", "MUL"},
                                std::pair{"kernels/PointwiseSub.cpp", "SUB"}})
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

// ---------------------------------------------------------------------------
// One module across two dispatches
// ---------------------------------------------------------------------------

/// The standalone descriptor the [GPU] case takes its archive and toc_key from. Read out
/// of the built file rather than written here: a copy would silently decouple this test
/// from the artifact it exists to read.
constexpr const char* PACKED_UKD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

/// The entry point that descriptor names.
constexpr const char* PACKED_SYMBOL = "ConvFwd";

/// A KernelDefinition whose code comes from a kpack archive at
/// `originDirectory / library`. Metadata carries exactly what the conv handler reads, so
/// the only thing that differs from the embedded-source path is the source.
///
/// `treeRoot` is the containment boundary the loader would have stamped. Passed
/// separately from originDirectory because they differ for a nested descriptor, which is
/// exactly the case whose archive lives at the arch root above it.
hipdnn_plugin_sdk::ingestor::KernelDefinition
    makeKpackConvKernel(const std::filesystem::path& originDirectory,
                        const std::filesystem::path& treeRoot,
                        const std::string& library,
                        const std::string& tocKey,
                        int64_t blockSize)
{
    auto kernel = makeKernel(blockSize, "HALF", PACKED_SYMBOL);
    kernel.name = "conv_fwd_f16_kpack";
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
    kernel.source.library = library;
    kernel.source.tocKey = tocKey;
    kernel.source.symbol = PACKED_SYMBOL;
    kernel.originDirectory = originDirectory;
    kernel.treeRoot = treeRoot;
    return kernel;
}

/// Two kernels differing only by block size, both naming one (archive, toc_key, arch):
/// the cache must grow by exactly one, because a second load of a blob already resident
/// is the cost this cache exists to avoid and nothing else observes it.
///
/// Measured as a delta rather than against 1, so a module some earlier case left resident
/// does not decide the outcome. That holds only while this is the one case in the binary
/// that loads a conv module: discovery reads descriptor JSON and never loads code, and
/// the other two conv dispatch cases refuse before reaching the loader. A case added
/// later that genuinely loads THIS (archive, toc_key, arch) would make the assertion
/// depend on suite order -- reset the cache in that case rather than widening this one.
TEST(TestConvFwdDispatch, LoadsTheModuleOnceAcrossTwoDispatches)
{
    SKIP_IF_NO_DEVICES();

    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(
        hip_kernel_provider::testing::findPackedArchDirectory(properties, arch, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << arch
                     << "): " << hip_kernel_provider::testing::unitKpackRoot() / arch
                     << " does not exist. Environmental -- the build packs per arch and this "
                        "device is outside GPU_TARGETS.";
    }

    hip_kernel_provider::testing::PackedKernelSource source;
    ASSERT_NO_FATAL_FAILURE(hip_kernel_provider::testing::readPackedKernelSource(
        packed, PACKED_UKD_DESCRIPTOR, source));

    // Built from the stripped arch rather than currentDeviceProperties(), which keeps the
    // feature flags hipGetDeviceProperties reports ("gfx1152:xnack-"). The packer names
    // shards with the bare arch, so the archive lookup has to be asked in that spelling.
    hipdnn_plugin_sdk::ingestor::DeviceProperties deviceProperties;
    deviceProperties.gcnArchName = arch;
    deviceProperties.warpSize = properties.warpSize;

    const GraphFixture fixture(buildConvFwdGraph(data_objects::DataType::HALF), deviceProperties);
    const auto bound = matchesGraph(CONV_FWD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // originDirectory is the descriptor's own (nested) folder; the arch root is the tree,
    // and the archive sits under it -- the real shipped shape.
    const auto first
        = makeKpackConvKernel(source.originDirectory, packed, source.library, source.tocKey, 64);
    const auto second
        = makeKpackConvKernel(source.originDirectory, packed, source.library, source.tocKey, 256);

    const auto& handler = dispatchHandler(CONV_FWD);
    const size_t before = convFwdKpackModuleCache().size();

    const auto preparedFirst = handler.prepare(fixture.context(), *bound, first);
    const auto preparedSecond = handler.prepare(fixture.context(), *bound, second);
    ASSERT_NE(preparedFirst, nullptr);
    ASSERT_NE(preparedSecond, nullptr);

    EXPECT_EQ(convFwdKpackModuleCache().size(), before + 1);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
