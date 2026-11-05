// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestCatalog.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx1250ConvFwdAotMatchers.cpp
 * @brief hipkernel:Gfx1250ConvFwdAot's native hooks on hand-built graphs: which graphs the
 *        engine serves (packed channels-last 2-D forward cross-correlation, fp16/bf16,
 *        symmetric padding, groups inferred), what it binds for dispatch and ranking, which
 *        catalog kernels then serve a graph, the fallback order, and the dispatch handler
 *        up to code loading. Host-only.
 *
 * The kernels are real catalog instances, their KMD fields taken from the golden vectors
 * (Gfx1250ConvFwdAotTestVectors.hpp); which problems each admits is the dispatcher's
 * answer (grouped_convolution._admits), and TestGfx1250ConvFwdAotGeometry checks every
 * instance against it.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::testing::gfx1250_conv_fwd_aot;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols GFX1250_CONV_FWD_AOT{GFX1250_CONV_FWD_AOT_ENGINE_NAME,
                                           "hipkernel.gfx1250_conv_fwd_aot.graph_match",
                                           "",
                                           "hipkernel.gfx1250_conv_fwd_aot.kernel_match",
                                           "hipkernel.gfx1250_conv_fwd_aot.score",
                                           "hipkernel.gfx1250_conv_fwd_aot.dispatch",
                                           "gfx1250_conv_fwd_aot.x.uid",
                                           "gfx1250_conv_fwd_aot.w.uid",
                                           "gfx1250_conv_fwd_aot.y.uid"};

using catalog::ConvShape;
using catalog::gfx1250Device;
using catalog::graphOf;
using catalog::W_UID;
using catalog::X_UID;
using catalog::Y_UID;

/// The dtype of every golden instance, in list order.
const std::vector<std::string>& goldenDtypes()
{
    static const std::vector<std::string> s_dtypes = [] {
        std::vector<std::string> dtypes;
        dtypes.reserve(gfx1250_conv_fwd_aot_vectors::KERNELS.size());
        for(const auto& kernel : gfx1250_conv_fwd_aot_vectors::KERNELS)
        {
            dtypes.push_back(catalog::instanceGeometry(kernel).str("dtype"));
        }
        return dtypes;
    }();
    return s_dtypes;
}

/// Golden instance @p kernel as a catalog kernel: its KMD fields, every field the catalog
/// gives a kernel, and the priority its list position gives it (falling within a dtype).
KernelDefinition kernelOf(const gfx1250_conv_fwd_aot_vectors::KernelVector& kernel,
                          const std::string& uuid)
{
    KernelDefinition out;
    out.kernelId = hipdnn_flatbuffers_sdk::utilities::parseUuid(uuid);
    out.name = kernel.instanceId;
    out.source.kind = KernelSourceKind::KPACK;
    const auto geometry = catalog::instanceGeometry(kernel);
    for(const auto* name : gfx1250_conv_fwd_aot_vectors::INT_FIELDS)
    {
        out.metadata[name] = geometry.integer(name);
    }
    for(const auto* name : gfx1250_conv_fwd_aot_vectors::STRING_FIELDS)
    {
        out.metadata[name] = geometry.str(name);
    }
    const auto dtype = geometry.str("dtype");
    int64_t index = 0;
    int64_t count = 0;
    for(size_t i = 0; i < gfx1250_conv_fwd_aot_vectors::KERNELS.size(); ++i)
    {
        if(goldenDtypes()[i] == dtype)
        {
            if(std::string_view(gfx1250_conv_fwd_aot_vectors::KERNELS[i].instanceId)
               == kernel.instanceId)
            {
                index = count;
            }
            ++count;
        }
    }
    out.priority = count - 1 - index;
    return out;
}

/// Runs the pack's KernelPrepareFn over @p kernels as the engine's shared state does at
/// load, so each carries its prepared payload.
void prepare(std::vector<KernelDefinition>& kernels)
{
    registerNativeIngestorSymbols();
    const auto prepareFn
        = KernelPrepareRegistry::tryResolve(std::string(GFX1250_CONV_FWD_AOT.kernelMatcher));
    ASSERT_NE(prepareFn, nullptr);
    for(auto& kernel : kernels)
    {
        kernel.prepared = prepareFn(kernel);
        kernel.preparedBy = prepareFn;
    }
}

/// The first golden instance @p accept takes.
const gfx1250_conv_fwd_aot_vectors::KernelVector&
    instanceWhere(const std::function<bool(const gfx1250_conv_fwd::InstanceGeometry&)>& accept)
{
    for(const auto& kernel : gfx1250_conv_fwd_aot_vectors::KERNELS)
    {
        if(accept(catalog::instanceGeometry(kernel)))
        {
            return kernel;
        }
    }
    throw std::logic_error("no golden instance qualifies");
}

/// An fp16 implicit-GEMM instance (128x128 tile, mem pipeline): runtime geometry, any
/// grouping.
KernelDefinition implicitGemm(const std::string& uuid = "6b7f3c52-0000-4000-8000-000000000001")
{
    return kernelOf(instanceWhere([](const auto& g) {
                        return g.str("dtype") == "fp16" && g.algorithm() == "implicit_gemm"
                               && g.integer("tile_m") == 128 && g.integer("tile_n") == 128
                               && g.str("pipeline") == "mem" && g.integer("filter_w") == 0;
                    }),
                    uuid);
}

/// An fp16 direct WMMA instance compiled for 1x1 stride-2 filters.
KernelDefinition direct1x1()
{
    return kernelOf(instanceWhere([](const auto& g) {
                        return g.str("dtype") == "fp16" && g.algorithm() == "direct_fwd_wmma"
                               && g.integer("filter_h") == 1 && g.integer("stride_h") == 2;
                    }),
                    "6b7f3c52-0000-4000-8000-000000000002");
}

constexpr std::string_view PROBLEM_TOKEN = "_gfx1250_conv_fwd_aot.problem";

/// The problem graph_match bound: uids x/w/y, N C K H W Y X G, stride, pad, dilation, dtype.
std::vector<int64_t> boundProblem(const BoundTokens& bound)
{
    const auto it = bound.find(std::string(PROBLEM_TOKEN));
    if(it == bound.end() || !std::holds_alternative<std::vector<int64_t>>(it->second))
    {
        ADD_FAILURE() << "the problem token is not bound";
        return {};
    }
    return std::get<std::vector<int64_t>>(it->second);
}

int64_t boundInt(const BoundTokens& bound, const std::string& token)
{
    const auto value = tryGetBoundInt(bound, token);
    EXPECT_TRUE(value.has_value()) << "token '" << token << "' is not bound";
    return value.value_or(-1);
}

bool serves(const ConvShape& shape)
{
    const auto graph = graphOf(shape);
    return matchesGraph(GFX1250_CONV_FWD_AOT, graph.context()).has_value();
}

TEST(TestGfx1250ConvFwdAotMatchers, ServesAPackedChannelsLastConvAndBindsItsProblem)
{
    ConvShape shape;
    shape.stride = {2, 1};
    shape.prePadding = {1, 2};
    shape.postPadding = {1, 2};
    shape.x = 5;
    const auto graph = graphOf(shape);
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(boundInt(*bound, "gfx1250_conv_fwd_aot.x.uid"), X_UID);
    EXPECT_EQ(boundInt(*bound, "gfx1250_conv_fwd_aot.w.uid"), W_UID);
    EXPECT_EQ(boundInt(*bound, "gfx1250_conv_fwd_aot.y.uid"), Y_UID);
    // uids, then N C K H W Y X G, stride h/w, pad h/w, dilation h/w, dtype (HALF).
    const std::vector<int64_t> expected = {X_UID,
                                           W_UID,
                                           Y_UID,
                                           2,
                                           64,
                                           64,
                                           14,
                                           14,
                                           3,
                                           5,
                                           1,
                                           2,
                                           1,
                                           1,
                                           2,
                                           1,
                                           1,
                                           static_cast<int64_t>(gfx1250_conv_fwd::DataType::FP16)};
    EXPECT_EQ(boundProblem(*bound), expected);
    // The problem is private (`_`): the graph features describe it, so nothing else is
    // bound for a ranker.
    EXPECT_EQ(bound->size(), 4U);
}

// An extent-one axis is only ever indexed at zero, so its stride is free: a batch-1 x whose
// N stride is 0, as a framework may leave it, is still packed channels-last.
TEST(TestGfx1250ConvFwdAotMatchers, ServesAnyStrideOnAnExtentOneAxis)
{
    ConvShape single;
    single.n = 1;
    single.xStrides = std::vector<int64_t>{0, 1, int64_t{14} * 64, 64};
    const auto graph = graphOf(single);
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value()) << "x's extent-one batch axis with stride 0";
    EXPECT_EQ(boundProblem(*bound).at(3), 1);
}

TEST(TestGfx1250ConvFwdAotMatchers, InfersGroupsFromTheFiltersChannels)
{
    ConvShape shape;
    shape.groups = 4;
    const auto graph = graphOf(shape);
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundProblem(*bound).at(10), 4);
}

TEST(TestGfx1250ConvFwdAotMatchers, ServesBf16)
{
    ConvShape shape;
    shape.dataType = data_objects::DataType::BFLOAT16;
    EXPECT_TRUE(serves(shape));
}

TEST(TestGfx1250ConvFwdAotMatchers, DeclinesAnotherDevice)
{
    auto gfx950 = gfx1250Device();
    gfx950.gcnArchName = "gfx950";
    gfx950.warpSize = 64;
    EXPECT_FALSE(matchesGraph(GFX1250_CONV_FWD_AOT, graphOf(ConvShape{}, gfx950).context()))
        << "gfx950";
    auto wave64 = gfx1250Device();
    wave64.warpSize = 64;
    EXPECT_FALSE(matchesGraph(GFX1250_CONV_FWD_AOT, graphOf(ConvShape{}, wave64).context()))
        << "a wave64 device";
    auto variant = gfx1250Device();
    variant.gcnArchName = "gfx1250:sramecc+:xnack-";
    EXPECT_TRUE(matchesGraph(GFX1250_CONV_FWD_AOT, graphOf(ConvShape{}, variant).context()))
        << "gfx1250 with target features";
}

TEST(TestGfx1250ConvFwdAotMatchers, DeclinesGraphsOutsideItsContract)
{
    ConvShape halfCompute;
    halfCompute.computeType = data_objects::DataType::HALF;
    EXPECT_FALSE(serves(halfCompute)) << "a compute type other than float";

    ConvShape fp32;
    fp32.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(serves(fp32)) << "fp32";

    ConvShape mixed;
    mixed.wDataType = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(serves(mixed)) << "mixed operand dtypes";

    ConvShape convolution;
    convolution.mode = data_objects::ConvMode::CONVOLUTION;
    EXPECT_FALSE(serves(convolution)) << "flipped-filter convolution";

    ConvShape channelsFirst;
    channelsFirst.channelsFirst = true;
    EXPECT_FALSE(serves(channelsFirst)) << "NCHW strides";

    ConvShape asymmetric;
    asymmetric.postPadding = {0, 1};
    EXPECT_FALSE(serves(asymmetric)) << "pre and post padding differ";

    ConvShape wrongOutput;
    wrongOutput.yDims = std::array<int64_t, 4>{2, 64, 13, 14};
    EXPECT_FALSE(serves(wrongOutput)) << "y is not the convolution's output";

    ConvShape yMixed;
    yMixed.yDataType = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(serves(yMixed)) << "y dtype differs from x and w";

    ConvShape negativePad;
    negativePad.prePadding = {-1, 1};
    negativePad.postPadding = {-1, 1};
    EXPECT_FALSE(serves(negativePad)) << "negative padding";

    ConvShape cNotPerFilter; // w carries 64 / 3 = 21 channels, which do not divide C = 64
    cNotPerFilter.groups = 3;
    cNotPerFilter.k = 63;
    EXPECT_FALSE(serves(cNotPerFilter)) << "C not divisible by the filter's channels";

    ConvShape emptyOutput; // a 3x3 over 2 unpadded rows has no output row
    emptyOutput.h = 2;
    emptyOutput.prePadding = {0, 0};
    emptyOutput.postPadding = {0, 0};
    emptyOutput.yDims = std::array<int64_t, 4>{2, 64, 0, 12};
    EXPECT_FALSE(serves(emptyOutput)) << "empty output";

    ConvShape zeroStride;
    zeroStride.stride = {0, 1};
    zeroStride.yDims = std::array<int64_t, 4>{2, 64, 14, 14};
    EXPECT_FALSE(serves(zeroStride)) << "stride 0";

    ConvShape zeroDilation;
    zeroDilation.dilation = {1, 0};
    zeroDilation.yDims = std::array<int64_t, 4>{2, 64, 14, 16}; // the output a 0 would give
    EXPECT_FALSE(serves(zeroDilation)) << "dilation 0";

    ConvShape kNotPerGroup;
    kNotPerGroup.groups = 4;
    kNotPerGroup.k = 66;
    EXPECT_FALSE(serves(kNotPerGroup)) << "K not divisible by the groups";

    ConvShape paddedRow; // channels-last, each 14-pixel row of x padded to 15 pixels
    paddedRow.xStrides = std::vector<int64_t>{int64_t{14} * 15 * 64, 1, int64_t{15} * 64, 64};
    EXPECT_FALSE(serves(paddedRow)) << "x rows padded past packed";

    ConvShape virtualY;
    virtualY.virtualY = true;
    EXPECT_FALSE(serves(virtualY)) << "virtual y";

    ConvShape fused;
    fused.nodeCount = 2;
    EXPECT_FALSE(serves(fused)) << "two nodes";

    ConvShape overrideShape;
    overrideShape.overrideShape = true;
    EXPECT_FALSE(serves(overrideShape)) << "shapes overridable at execution";

    ConvShape aliased; // y names x's uid; x's dims are the output's, so only the uid differs
    aliased.yUid = X_UID;
    EXPECT_FALSE(serves(aliased)) << "y aliases x";
}

// The kernels take every tensor's byte size and byte offsets as i32 arguments: a tensor of
// more than 2^31 - 1 bytes (2^30 - 1 fp16/bf16 elements) is declined, x, w and y alike.
TEST(TestGfx1250ConvFwdAotMatchers, DeclinesATensorTheKernelsI32SizesCannotHold)
{
    ConvShape underLimit; // x and y: 1 * 64 * 4095 * 4096 = 2^30 - 2^18 elements
    underLimit.n = 1;
    underLimit.h = 4095;
    underLimit.w = 4096;
    EXPECT_TRUE(serves(underLimit)) << "every tensor within the i32 byte limit";

    ConvShape xOver = underLimit; // x: 1 * 64 * 4096 * 4096 = 2^30 elements; y has 8
    xOver.h = 4096;
    xOver.k = 8;
    EXPECT_FALSE(serves(xOver)) << "x of 2^31 bytes";

    ConvShape wOver; // w: 32768 * 1 * 1 * 32768 = 2^30 elements; x and y small
    wOver.n = 1;
    wOver.c = 32768;
    wOver.k = 32768;
    wOver.h = 1;
    wOver.w = 1;
    wOver.y = 1;
    wOver.x = 1;
    wOver.prePadding = {0, 0};
    wOver.postPadding = {0, 0};
    EXPECT_FALSE(serves(wOver)) << "w of 2^31 bytes";

    ConvShape yOver; // y: 1 * 4096 * 4096 * 64 = 2^30 elements; x has 8 channels
    yOver.n = 1;
    yOver.c = 8;
    yOver.k = 64;
    yOver.h = 4096;
    yOver.w = 4096;
    EXPECT_FALSE(serves(yOver)) << "y of 2^31 bytes";

    ConvShape strideOver; // a stride the kernels' i32 arguments cannot hold
    strideOver.stride = {int64_t{1} << 31, 1};
    EXPECT_FALSE(serves(strideOver)) << "stride beyond i32";
}

TEST(TestGfx1250ConvFwdAotMatchers, ServesAGraphWithTheKernelsTheDispatcherAdmits)
{
    // 3x3: the implicit GEMM runs it, the 1x1 direct kernel does not.
    {
        const auto graph = graphOf(ConvShape{});
        const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_TRUE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), implicitGemm(), *bound));
        EXPECT_FALSE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), direct1x1(), *bound));
    }
    // A 1x1 stride-2 convolution: both run it.
    {
        ConvShape shape;
        shape.n = 1;
        shape.c = 128;
        shape.k = 128;
        shape.h = 28;
        shape.w = 28;
        shape.y = 1;
        shape.x = 1;
        shape.stride = {2, 2};
        shape.prePadding = {0, 0};
        shape.postPadding = {0, 0};
        const auto graph = graphOf(shape);
        const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_TRUE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), implicitGemm(), *bound));
        EXPECT_TRUE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), direct1x1(), *bound));
    }
}

TEST(TestGfx1250ConvFwdAotMatchers, AKernelOfAnotherDtypeDoesNotServe)
{
    ConvShape shape;
    shape.dataType = data_objects::DataType::BFLOAT16;
    const auto graph = graphOf(shape);
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_FALSE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), implicitGemm(), *bound));
}

TEST(TestGfx1250ConvFwdAotMatchers, AnInstanceItCannotReadDoesNotServe)
{
    const auto graph = graphOf(ConvShape{});
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());
    auto missingField = implicitGemm("6b7f3c52-0000-4000-8000-000000000003");
    missingField.metadata.erase("tile_m");
    EXPECT_FALSE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), missingField, *bound));
    auto unknownFamily = implicitGemm("6b7f3c52-0000-4000-8000-000000000008");
    unknownFamily.metadata["algorithm"] = std::string("winograd");
    EXPECT_FALSE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), unknownFamily, *bound));
    auto notKpack = implicitGemm("6b7f3c52-0000-4000-8000-000000000009");
    notKpack.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
    EXPECT_FALSE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), notKpack, *bound));
}

TEST(TestGfx1250ConvFwdAotMatchers, ABakedFieldIsCheckedAgainstTheWidthFoldedProblem)
{
    // A 4x4 stride-4 stem folds its width into its channels (X 4 -> 1), and the kernel runs
    // the folded problem: a baked filter_w is checked against that, as rocKE does.
    ConvShape stem;
    stem.n = 1;
    stem.c = 8;
    stem.h = 32;
    stem.w = 32;
    stem.y = 4;
    stem.x = 4;
    stem.stride = {4, 4};
    stem.prePadding = {0, 0};
    stem.postPadding = {0, 0};
    const auto graph = graphOf(stem);
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());
    const auto baking = [](const char* uuid, const std::string& field, int64_t value) {
        auto kernel = implicitGemm(uuid);
        kernel.metadata[field] = value;
        return kernel;
    };
    const auto admits = [&](const KernelDefinition& kernel) {
        return matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), kernel, *bound);
    };
    EXPECT_TRUE(admits(implicitGemm()));
    EXPECT_TRUE(admits(baking("6b7f3c52-0000-4000-8000-000000000004", "filter_w", 1)));
    EXPECT_FALSE(admits(baking("6b7f3c52-0000-4000-8000-000000000005", "filter_w", 4)));
    // The fold moves the width into the channels: 8 per group become 32.
    EXPECT_TRUE(admits(baking("6b7f3c52-0000-4000-8000-000000000006", "cpg", 32)));
    EXPECT_FALSE(admits(baking("6b7f3c52-0000-4000-8000-000000000007", "cpg", 8)));
}

TEST(TestGfx1250ConvFwdAotMatchers, FallbackOrderIsTheCatalogsOrder)
{
    const auto graph = graphOf(ConvShape{});
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());
    // The catalog gives the instance list's earlier entry the higher priority, so the native
    // score and the SDK's declared order (used wherever no score decides) agree on it. Both
    // kernels carry the priority their list position gives them (kernelOf).
    const auto direct = direct1x1();
    const auto implicit = implicitGemm();
    EXPECT_GT(scoreKernel(GFX1250_CONV_FWD_AOT, graph.context(), direct, *bound),
              scoreKernel(GFX1250_CONV_FWD_AOT, graph.context(), implicit, *bound));
    const auto declared = hipdnn_plugin_sdk::ingestor::detail::declaredOrder({&implicit, &direct});
    ASSERT_EQ(declared.size(), 2U);
    EXPECT_EQ(declared.front(), &direct);
}

/// Golden problem @p p as the graph spells it.
ConvShape shapeOf(const gfx1250_conv_fwd::Problem& p)
{
    ConvShape shape;
    shape.n = p.n;
    shape.c = p.c;
    shape.k = p.k;
    shape.h = p.h;
    shape.w = p.w;
    shape.y = p.y;
    shape.x = p.x;
    shape.groups = p.groups;
    shape.stride = {p.strideH, p.strideW};
    shape.dilation = {p.dilationH, p.dilationW};
    shape.prePadding = {p.padH, p.padW};
    shape.postPadding = shape.prePadding;
    shape.dataType = p.dtype == gfx1250_conv_fwd::DataType::BF16 ? data_objects::DataType::BFLOAT16
                                                                 : data_objects::DataType::HALF;
    shape.yDims = std::array<int64_t, 4>{p.n, p.k, p.ho(), p.wo()};
    return shape;
}

/// A distinct kernel id per golden instance.
std::string catalogUuid(size_t index)
{
    std::ostringstream out;
    out << "6b7f3c52-0000-4000-8001-" << std::hex << std::setw(12) << std::setfill('0') << index;
    return out.str();
}

// rocKE builds the spec of every WMMA family at its own wave size, 32, whatever wave_size
// the identity spells, and so does the port: an instance of each family that serves a
// golden problem still serves it when its identity says 64.
TEST(TestGfx1250ConvFwdAotMatchers, AWmmaKernelKeepsItsSpecsWaveSize)
{
    const auto& golden = gfx1250_conv_fwd_aot_vectors::KERNELS;
    for(const std::string family : {"direct_fwd_wmma",
                                    "direct_fwd_gstream_wmma",
                                    "direct_fwd_smallc_wmma",
                                    "direct_fwd_smallk_wmma",
                                    "direct_fwd_tapouter_wmma"})
    {
        bool checked = false;
        for(size_t i = 0; i < golden.size() && !checked; ++i)
        {
            const auto geometry = catalog::instanceGeometry(golden[i]);
            if(geometry.algorithm() != family)
            {
                continue;
            }
            const auto port = gfx1250_conv_fwd::CatalogKernel::parse(geometry);
            for(size_t index = 0; index < gfx1250_conv_fwd_aot_vectors::PROBLEMS.size() && !checked;
                ++index)
            {
                const auto problem = catalog::problemAt(index);
                if(gfx1250_conv_fwd::toString(problem.dtype) != geometry.str("dtype")
                   || port.refuse(problem).has_value())
                {
                    continue;
                }
                const auto graph = graphOf(shapeOf(problem));
                const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
                if(!bound)
                {
                    continue;
                }
                auto wave64 = kernelOf(golden[i], catalogUuid(i));
                wave64.metadata["wave_size"] = int64_t{64};
                EXPECT_TRUE(matchesKernel(GFX1250_CONV_FWD_AOT, graph.context(), wave64, *bound))
                    << golden[i].instanceId << " with wave_size 64 on golden problem " << index;
                checked = true;
            }
        }
        EXPECT_TRUE(checked) << "no golden " << family << " instance serves a golden problem";
    }
}

// Every golden problem, built as a graph, reaches the port as the problem it states: the
// kernels the hooks serve are exactly those the port admits on that problem (the port is
// checked against rocKE's dispatcher by TestGfx1250ConvFwdAotGeometry). The problem set has
// H != W, Y != X and per-axis stride, padding and dilation, so a hook that reads an axis
// of the bound problem for the other one serves another set.
TEST(TestGfx1250ConvFwdAotMatchers, ServesExactlyWhatThePortAdmitsOverTheGoldenProblems)
{
    const auto& golden = gfx1250_conv_fwd_aot_vectors::KERNELS;
    std::vector<KernelDefinition> kernels;
    std::vector<gfx1250_conv_fwd::CatalogKernel> ports;
    for(size_t i = 0; i < golden.size(); ++i)
    {
        kernels.push_back(kernelOf(golden[i], catalogUuid(i)));
        ports.push_back(
            gfx1250_conv_fwd::CatalogKernel::parse(catalog::instanceGeometry(golden[i])));
    }
    prepare(kernels);
    const auto matcher = kernelMatcher(GFX1250_CONV_FWD_AOT);
    size_t graphs = 0;
    size_t served = 0;
    for(size_t index = 0; index < gfx1250_conv_fwd_aot_vectors::PROBLEMS.size(); ++index)
    {
        const auto problem = catalog::problemAt(index);
        const auto graph = graphOf(shapeOf(problem));
        const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
        if(!bound)
        {
            // Declined graphs (a tensor past the kernels' i32 sizes) are covered above.
            continue;
        }
        ++graphs;
        const auto* dtype = gfx1250_conv_fwd::toString(problem.dtype);
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const bool expected
                = goldenDtypes()[i] == dtype && !ports[i].refuse(problem).has_value();
            const bool actual = matcher(graph.context(), *bound, kernels[i]);
            EXPECT_EQ(actual, expected) << golden[i].instanceId << " on golden problem " << index;
            served += actual ? 1 : 0;
        }
    }
    EXPECT_GT(graphs, 4000U);
    EXPECT_GT(served, 10000U);
}

TEST(TestGfx1250ConvFwdAotDispatch, NoWorkspaceAndPrepareReachesCodeLoading)
{
    std::vector<KernelDefinition> kernels{implicitGemm(), direct1x1()};
    prepare(kernels);
    const auto& handler = dispatchHandler(GFX1250_CONV_FWD_AOT);
    // A 3x3: the implicit GEMM runs it, the 1x1 direct kernel does not.
    const auto graph = graphOf(ConvShape{});
    const auto bound = matchesGraph(GFX1250_CONV_FWD_AOT, graph.context());
    ASSERT_TRUE(bound.has_value());

    // No plan of this engine sizes a workspace.
    EXPECT_EQ(handler.workspaceBytes(graph.context(), *bound, kernels[0]), 0U);
    try
    {
        // The kernel names no archive, so loading is the first step that can fail.
        handler.prepare(graph.context(), *bound, kernels[0]);
        ADD_FAILURE() << "preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        for(const auto* earlier :
            {"cannot run this convolution", "an empty launch plan", "different launch geometry"})
        {
            EXPECT_EQ(message.find(earlier), std::string::npos) << message;
        }
    }

    // A kernel that refuses the graph is refused before any code loads, in both.
    EXPECT_THROW(handler.workspaceBytes(graph.context(), *bound, kernels[1]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(handler.prepare(graph.context(), *bound, kernels[1]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
