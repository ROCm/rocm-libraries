// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace aot = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd_aot;

constexpr PackSymbols GFX950_CONV_FWD_BIAS_AOT{"hipkernel:Gfx950ConvFwdBiasAot",
                                               "hipkernel.gfx950_conv_fwd_bias_aot.graph_match",
                                               "",
                                               "hipkernel.gfx950_conv_fwd_bias_aot.kernel_match",
                                               "hipkernel.gfx950_conv_fwd_bias_aot.score",
                                               "hipkernel.gfx950_conv_fwd_bias_aot.dispatch",
                                               "gfx950_conv_fwd_bias.x.uid",
                                               "gfx950_conv_fwd_bias.w.uid",
                                               "gfx950_conv_fwd_bias.y.uid"};

constexpr PackSymbols GFX950_CONV_FWD_AOT{"hipkernel:Gfx950ConvFwdAot",
                                          "hipkernel.gfx950_conv_fwd_aot.graph_match",
                                          "",
                                          "hipkernel.gfx950_conv_fwd_aot.kernel_match",
                                          "hipkernel.gfx950_conv_fwd_aot.score",
                                          "hipkernel.gfx950_conv_fwd_aot.dispatch",
                                          "gfx950_conv_fwd.x.uid",
                                          "gfx950_conv_fwd.w.uid",
                                          "gfx950_conv_fwd.y.uid"};

constexpr const char* BIAS_TOKEN = "gfx950_conv_fwd_bias.b.uid";
constexpr const char* PROBLEM_TOKEN = "_gfx950_conv_fwd_bias.problem";

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t YC_UID = 3;
constexpr int64_t B_UID = 4;
constexpr int64_t Y_UID = 5;
constexpr int64_t Z_UID = 6;

struct Tensor
{
    int64_t uid;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dtype;
    bool isVirtual = false;
    bool passByValue = false;
};

/// conv(x, w) -> yc ; ADD(yc, b) -> y: a 2 x 32 x 14 x 14 channels-last fp16 input, a
/// 32 x (32 / groups) x 3 x 3 filter, "same" padding, and a (1, K, 1, 1) bias. Every
/// field is one thing the bias engine's graph match gates.
struct BiasGraphSpec
{
    int64_t groups = 1;
    int64_t k = 32;
    data_objects::DataType io = data_objects::DataType::HALF;
    data_objects::DataType convOutDtype = data_objects::DataType::FLOAT;
    bool convOutVirtual = true;
    std::vector<int64_t> biasDims{1, 32, 1, 1};
    std::vector<int64_t> biasStrides{32, 1, 32, 32};
    std::optional<data_objects::DataType> biasDtype = std::nullopt;
    bool biasVirtual = false;
    bool biasPassByValue = false;
    data_objects::PointwiseMode operation = data_objects::PointwiseMode::ADD;
    data_objects::DataType convCompute = data_objects::DataType::FLOAT;
    data_objects::DataType addCompute = data_objects::DataType::FLOAT;
    /// ADD(b, yc) instead of ADD(yc, b).
    bool biasFirst = false;
    /// The ADD node ahead of the convolution in the node list.
    bool addFirst = false;
    /// The ADD's operands, overriding (yc, b).
    std::optional<std::pair<int64_t, int64_t>> addInputs = std::nullopt;
    /// A third ADD operand (the bias again).
    bool ternary = false;
    /// A RELU after the ADD: y virtual, z the graph output.
    bool trailingRelu = false;
    /// The convolution alone, writing y; no ADD and no bias.
    bool convOnly = false;
    /// A 3-D convolution: NDHWC x, y and the convolution output with depth 4, a
    /// 3 x 3 x 3 filter and a (1, K, 1, 1, 1) bias (biasDims / biasStrides ignored).
    bool threeD = false;
};

flatbuffers::FlatBufferBuilder buildBiasGraph(const BiasGraphSpec& spec = {})
{
    flatbuffers::FlatBufferBuilder builder;
    const int64_t n = 2;
    const int64_t c = 32;
    const int64_t h = 14;
    const int64_t w = 14;
    const int64_t d = 4;
    const int64_t cpg = c / spec.groups;
    // Channels-last dims and strides of an activation with @p channels channels.
    const auto activation = [&](int64_t channels) {
        return spec.threeD
                   ? std::pair<std::vector<int64_t>, std::vector<int64_t>>{{n, channels, d, h, w},
                                                                           {d * h * w * channels,
                                                                            1,
                                                                            h * w * channels,
                                                                            w * channels,
                                                                            channels}}
                   : std::pair<std::vector<int64_t>, std::vector<int64_t>>{
                         {n, channels, h, w}, {h * w * channels, 1, w * channels, channels}};
    };
    const auto [xDims, xStrides] = activation(c);
    const auto [yDims, yStrides] = activation(spec.k);
    const auto filter = spec.threeD
                            ? std::pair<std::vector<int64_t>,
                                        std::vector<int64_t>>{{spec.k, cpg, 3, 3, 3},
                                                              {27 * cpg, 1, 9 * cpg, 3 * cpg, cpg}}
                            : std::pair<std::vector<int64_t>, std::vector<int64_t>>{
                                  {spec.k, cpg, 3, 3}, {9 * cpg, 1, 3 * cpg, cpg}};
    std::vector<Tensor> tensors{{X_UID, xDims, xStrides, spec.io},
                                {W_UID, filter.first, filter.second, spec.io},
                                {Y_UID, yDims, yStrides, spec.io, spec.trailingRelu}};
    if(!spec.convOnly)
    {
        tensors.push_back({YC_UID, yDims, yStrides, spec.convOutDtype, spec.convOutVirtual});
        tensors.push_back({B_UID,
                           spec.threeD ? std::vector<int64_t>{1, spec.k, 1, 1, 1} : spec.biasDims,
                           spec.threeD ? std::vector<int64_t>{spec.k, 1, spec.k, spec.k, spec.k}
                                       : spec.biasStrides,
                           spec.biasDtype.value_or(spec.io),
                           spec.biasVirtual,
                           spec.biasPassByValue});
    }
    if(spec.trailingRelu)
    {
        tensors.push_back({Z_UID, yDims, yStrides, spec.io});
    }

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensorOffsets;
    for(const auto& tensor : tensors)
    {
        const auto dims = builder.CreateVector(tensor.dims);
        const auto strides = builder.CreateVector(tensor.strides);
        data_objects::TensorAttributesBuilder tensorBuilder(builder);
        tensorBuilder.add_uid(tensor.uid);
        tensorBuilder.add_data_type(tensor.dtype);
        tensorBuilder.add_dims(dims);
        tensorBuilder.add_strides(strides);
        tensorBuilder.add_virtual_(tensor.isVirtual);
        tensorBuilder.add_is_runtime_pass_by_value(tensor.passByValue);
        tensorBuilder.add_alignment(16);
        tensorOffsets.push_back(tensorBuilder.Finish());
    }

    const std::vector<int64_t> pad(spec.threeD ? 3 : 2, 1);
    const std::vector<int64_t> one(spec.threeD ? 3 : 2, 1);
    const auto convAttributes = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        X_UID,
        W_UID,
        spec.convOnly ? Y_UID : YC_UID,
        &pad,
        &pad,
        &one,
        &one,
        data_objects::ConvMode::CROSS_CORRELATION);
    const auto convNode
        = data_objects::CreateNodeDirect(builder,
                                         "convolution",
                                         spec.convCompute,
                                         data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                         convAttributes.Union());
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes{convNode};
    if(!spec.convOnly)
    {
        const auto inputs
            = spec.addInputs.value_or(spec.biasFirst ? std::pair<int64_t, int64_t>{B_UID, YC_UID}
                                                     : std::pair<int64_t, int64_t>{YC_UID, B_UID});
        data_objects::PointwiseAttributesBuilder addBuilder(builder);
        addBuilder.add_operation(spec.operation);
        addBuilder.add_in_0_tensor_uid(inputs.first);
        addBuilder.add_in_1_tensor_uid(inputs.second);
        if(spec.ternary)
        {
            addBuilder.add_in_2_tensor_uid(B_UID);
        }
        addBuilder.add_out_0_tensor_uid(Y_UID);
        const auto addAttributes = addBuilder.Finish();
        const auto addNode
            = data_objects::CreateNodeDirect(builder,
                                             "bias",
                                             spec.addCompute,
                                             data_objects::NodeAttributes::PointwiseAttributes,
                                             addAttributes.Union());
        if(spec.addFirst)
        {
            nodes.insert(nodes.begin(), addNode);
        }
        else
        {
            nodes.push_back(addNode);
        }
    }
    if(spec.trailingRelu)
    {
        data_objects::PointwiseAttributesBuilder reluBuilder(builder);
        reluBuilder.add_operation(data_objects::PointwiseMode::RELU_FWD);
        reluBuilder.add_in_0_tensor_uid(Y_UID);
        reluBuilder.add_out_0_tensor_uid(Z_UID);
        const auto reluAttributes = reluBuilder.Finish();
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "relu",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           reluAttributes.Union()));
    }

    const auto tensorVector = builder.CreateVector(tensorOffsets);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_tensors(tensorVector);
    graphBuilder.add_nodes(nodeVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// An implicit-GEMM instance as make_conv_aot_config.py --bias emits it: the unfused
/// engine's fields plus `bias` (absent when @p bias is nullopt).
KernelDefinition biasKernel(std::optional<int64_t> bias = 1, const std::string& dtype = "fp16")
{
    KernelDefinition kernel;
    kernel.name = "aot_bias_instance";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.metadata = {{"dtype", dtype},
                       {"layout", std::string("NHWC")},
                       {"tile_m", int64_t{64}},
                       {"tile_n", int64_t{64}},
                       {"tile_k", int64_t{64}},
                       {"warp_m", int64_t{2}},
                       {"warp_n", int64_t{2}},
                       {"warp_tile_m", int64_t{32}},
                       {"warp_tile_n", int64_t{32}},
                       {"warp_tile_k", int64_t{16}},
                       {"wave_size", int64_t{64}},
                       {"pipeline", std::string("mem")},
                       {"epilogue", std::string("cshuffle")},
                       {"vector_size_a", int64_t{8}},
                       {"vector_size_b", int64_t{8}},
                       {"vector_size_c", int64_t{8}},
                       {"unroll_k", int64_t{0}},
                       {"async_dma", int64_t{0}},
                       {"async_chunk_a", int64_t{0}},
                       {"async_chunk_b", int64_t{0}},
                       {"grouped", int64_t{1}},
                       {"instance_id", std::string(40, 'b')},
                       // The descriptor loader completes these from their KMD defaults.
                       {"kernel_family", int64_t{0}},
                       {"direct_variant", std::string("none")},
                       {"filter_h", int64_t{0}},
                       {"filter_w", int64_t{0}},
                       {"stride_h", int64_t{0}},
                       {"stride_w", int64_t{0}},
                       {"pad_h", int64_t{0}},
                       {"pad_w", int64_t{0}},
                       {"dilation_h", int64_t{0}},
                       {"dilation_w", int64_t{0}},
                       {"block_w", int64_t{0}},
                       {"block_waves", int64_t{0}},
                       {"block_h", int64_t{0}},
                       {"unroll_rows", int64_t{0}},
                       {"block_q", int64_t{0}},
                       {"block_groups", int64_t{0}},
                       {"double_buffer", int64_t{0}},
                       {"cpg", int64_t{0}},
                       {"kpg", int64_t{0}},
                       {"tile_h", int64_t{0}},
                       {"tile_w", int64_t{0}},
                       {"tile_k_out", int64_t{0}},
                       {"ck", int64_t{0}},
                       {"waves_m", int64_t{0}},
                       {"waves_n", int64_t{0}},
                       {"atom", std::string("none")},
                       {"swizzle_wgm", int64_t{0}},
                       {"iglp", int64_t{-1}},
                       {"waves_per_eu", int64_t{0}},
                       {"group_merge", int64_t{1}},
                       {"split_c", int64_t{1}},
                       {"prefetch_rows", int64_t{0}},
                       {"w_fold", int64_t{0}}};
    if(bias)
    {
        kernel.metadata["bias"] = *bias;
    }
    return kernel;
}

bool matches(const BiasGraphSpec& spec, std::string arch = "gfx950")
{
    const GraphFixture fixture(buildBiasGraph(spec), gfx950Properties(std::move(arch)));
    return matchesGraph(GFX950_CONV_FWD_BIAS_AOT, fixture.context()).has_value();
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, BindsTheConvolutionTheBiasAndTheAddOutput)
{
    const GraphFixture fixture(buildBiasGraph(), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD_BIAS_AOT, fixture.context());
    ASSERT_TRUE(bound);
    EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD_BIAS_AOT.inputAToken), X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD_BIAS_AOT.inputBToken), W_UID);
    // The kernel stores the ADD's output; the convolution's own output is never written.
    EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD_BIAS_AOT.outputToken), Y_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BIAS_TOKEN), B_UID);
    // The problem token: x, w, y, the 14 problem fields, the dtype, then the bias uid.
    const auto problem = bound->find(PROBLEM_TOKEN);
    ASSERT_NE(problem, bound->end());
    const auto* values = std::get_if<std::vector<int64_t>>(&problem->second);
    ASSERT_NE(values, nullptr);
    ASSERT_EQ(values->size(), 19U);
    EXPECT_EQ(values->front(), X_UID);
    EXPECT_EQ((*values)[2], Y_UID);
    EXPECT_EQ((*values)[5], 32); // K
    EXPECT_EQ(values->back(), B_UID);
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, AcceptsEitherNodeAndOperandOrder)
{
    EXPECT_TRUE(matches({}));
    BiasGraphSpec spec;
    spec.biasFirst = true;
    EXPECT_TRUE(matches(spec));
    spec.addFirst = true;
    EXPECT_TRUE(matches(spec));
    spec.biasFirst = false;
    EXPECT_TRUE(matches(spec));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, AcceptsAPerChannelBiasOfRankFourOrThree)
{
    // (1, K, 1, 1), channels-last or plain strides: the channels are contiguous either way.
    EXPECT_TRUE(matches({}));
    BiasGraphSpec plain;
    plain.biasStrides = {32, 1, 1, 1};
    EXPECT_TRUE(matches(plain));
    // (K, 1, 1) broadcasts from the right onto the channel axis too.
    BiasGraphSpec rankThree;
    rankThree.biasDims = {32, 1, 1};
    rankThree.biasStrides = {1, 1, 1};
    EXPECT_TRUE(matches(rankThree));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, DeclinesEveryOtherBiasShape)
{
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> shapes{
        // (K) broadcasts onto W, not onto the channels.
        {{32}, {1}},
        // (1, K) broadcasts onto H.
        {{1, 32}, {32, 1}},
        // Another channel count.
        {{1, 16, 1, 1}, {16, 1, 16, 16}},
        // A bias per spatial position or per image.
        {{1, 32, 14, 14}, {6272, 1, 448, 32}},
        {{2, 32, 1, 1}, {32, 1, 32, 32}},
        {{1, 1, 1, 32}, {32, 32, 32, 1}},
        // A scalar.
        {{1, 1, 1, 1}, {1, 1, 1, 1}},
        // Channels not contiguous.
        {{1, 32, 1, 1}, {64, 2, 64, 64}},
        {{32, 1, 1}, {2, 1, 1}},
        // Dims and strides of different ranks.
        {{1, 32, 1, 1}, {32, 1, 32}},
        // The 3-D convolution's bias rank: this engine is 2-D only.
        {{1, 32, 1, 1, 1}, {32, 1, 32, 32, 32}},
    };
    for(const auto& [dims, strides] : shapes)
    {
        SCOPED_TRACE(::testing::PrintToString(dims));
        BiasGraphSpec spec;
        spec.biasDims = dims;
        spec.biasStrides = strides;
        EXPECT_FALSE(matches(spec));
    }
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, DeclinesA3DConvolution)
{
    // The packaged kernels are the 2-D (NHWC) catalog: a 5-D graph, bias included, is
    // left to another engine, as the unfused engine leaves the 5-D convolution.
    BiasGraphSpec threeD;
    threeD.threeD = true;
    EXPECT_FALSE(matches(threeD));
    BiasGraphSpec convOnly;
    convOnly.threeD = true;
    convOnly.convOnly = true;
    const GraphFixture single(buildBiasGraph(convOnly), gfx950Properties());
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD_AOT, single.context()));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, TheBiasIsADeviceTensorOfTheIoDtype)
{
    for(const auto dtype : {data_objects::DataType::FLOAT, data_objects::DataType::BFLOAT16})
    {
        SCOPED_TRACE(static_cast<int>(dtype));
        BiasGraphSpec spec;
        spec.biasDtype = dtype;
        EXPECT_FALSE(matches(spec));
    }
    BiasGraphSpec bf16;
    bf16.io = data_objects::DataType::BFLOAT16;
    EXPECT_TRUE(matches(bf16));
    bf16.biasDtype = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(bf16));

    BiasGraphSpec isVirtual;
    isVirtual.biasVirtual = true;
    EXPECT_FALSE(matches(isVirtual));
    BiasGraphSpec byValue;
    byValue.biasPassByValue = true;
    EXPECT_FALSE(matches(byValue));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, TheConvolutionOutputOnlyFeedsTheAdd)
{
    // Stored: the kernel never writes it.
    BiasGraphSpec stored;
    stored.convOutVirtual = false;
    EXPECT_FALSE(matches(stored));
    // fp32 or the io dtype; the kernel keeps it in fp32 either way.
    BiasGraphSpec ioDtype;
    ioDtype.convOutDtype = data_objects::DataType::HALF;
    EXPECT_TRUE(matches(ioDtype));
    BiasGraphSpec otherDtype;
    otherDtype.convOutDtype = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matches(otherDtype));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, DeclinesAnyOtherGraphShape)
{
    BiasGraphSpec mul;
    mul.operation = data_objects::PointwiseMode::MUL;
    EXPECT_FALSE(matches(mul));
    BiasGraphSpec sub;
    sub.operation = data_objects::PointwiseMode::SUB;
    EXPECT_FALSE(matches(sub));
    BiasGraphSpec ternary;
    ternary.ternary = true;
    EXPECT_FALSE(matches(ternary));
    // Both operands the convolution's output, or neither.
    BiasGraphSpec square;
    square.addInputs = std::pair<int64_t, int64_t>{YC_UID, YC_UID};
    EXPECT_FALSE(matches(square));
    BiasGraphSpec unrelated;
    unrelated.addInputs = std::pair<int64_t, int64_t>{X_UID, B_UID};
    EXPECT_FALSE(matches(unrelated));
    // An operand that is the convolution's own input.
    BiasGraphSpec input;
    input.addInputs = std::pair<int64_t, int64_t>{YC_UID, X_UID};
    EXPECT_FALSE(matches(input));
    // A third node.
    BiasGraphSpec relu;
    relu.trailingRelu = true;
    EXPECT_FALSE(matches(relu));
    // The convolution alone is the unfused engine's.
    BiasGraphSpec convOnly;
    convOnly.convOnly = true;
    EXPECT_FALSE(matches(convOnly));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, BothNodesComputeInFp32)
{
    BiasGraphSpec add;
    add.addCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(add));
    BiasGraphSpec conv;
    conv.convCompute = data_objects::DataType::HALF;
    EXPECT_FALSE(matches(conv));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, DeclinesEveryDeviceButAWave64Gfx950)
{
    EXPECT_TRUE(matches({}, "gfx950:sramecc+:xnack-"));
    EXPECT_FALSE(matches({}, "gfx942"));
    EXPECT_FALSE(matches({}, "gfx1250"));
    auto properties = gfx950Properties();
    properties.warpSize = 32;
    const GraphFixture wave32(buildBiasGraph(), properties);
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD_BIAS_AOT, wave32.context()));
}

TEST(TestGfx950ConvFwdBiasAotGraphMatcher, ServesGroupedAndDepthwiseConvolutions)
{
    BiasGraphSpec grouped;
    grouped.groups = 4;
    grouped.k = 64;
    grouped.biasDims = {1, 64, 1, 1};
    grouped.biasStrides = {64, 1, 64, 64};
    EXPECT_TRUE(matches(grouped));
    // K/groups must be whole: the same convolution rules as the unfused engine.
    BiasGraphSpec ragged = grouped;
    ragged.k = 30;
    ragged.biasDims = {1, 30, 1, 1};
    ragged.biasStrides = {30, 1, 30, 30};
    EXPECT_FALSE(matches(ragged));
    BiasGraphSpec depthwise;
    depthwise.groups = 32;
    EXPECT_TRUE(matches(depthwise));
}

TEST(TestGfx950ConvFwdBiasAotKernelMatcher, AdmitsOnlyInstancesBuiltWithTheBias)
{
    const GraphFixture fixture(buildBiasGraph(), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD_BIAS_AOT, fixture.context());
    ASSERT_TRUE(bound);
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel(), *bound));
    // Without a bound token the matcher walks the graph itself, with the same verdict.
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel()));
    // An instance without the bias epilogue would leave the bias out.
    EXPECT_FALSE(matchesKernel(
        GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel(std::nullopt), *bound));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel(0), *bound));
    // A malformed bias field.
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel(2), *bound));
    auto stringBias = biasKernel();
    stringBias.metadata["bias"] = std::string("1");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), stringBias, *bound));
    // The io dtype still decides.
    EXPECT_FALSE(
        matchesKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel(1, "bf16"), *bound));
    // The instance's own capability check still runs: 8-wide stores need K / groups % 8.
    BiasGraphSpec narrow;
    narrow.k = 36;
    narrow.biasDims = {1, 36, 1, 1};
    narrow.biasStrides = {36, 1, 36, 36};
    const GraphFixture narrowFixture(buildBiasGraph(narrow), gfx950Properties());
    ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD_BIAS_AOT, narrowFixture.context()));
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_BIAS_AOT, narrowFixture.context(), biasKernel()));
}

TEST(TestGfx950ConvFwdBiasAotKernelMatcher, ScoresAsTheUnfusedEngineDoes)
{
    const GraphFixture fixture(buildBiasGraph(), gfx950Properties());
    EXPECT_GT(scoreKernel(GFX950_CONV_FWD_BIAS_AOT, fixture.context(), biasKernel()), 0.0);
}

TEST(TestGfx950ConvFwdAotGraphMatcher, DeclinesTheConvolutionPlusBiasGraph)
{
    // The two-node graph is the bias engine's: the unfused engine matches one node only.
    const GraphFixture fixture(buildBiasGraph(), gfx950Properties());
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD_AOT, fixture.context()));
    BiasGraphSpec convOnly;
    convOnly.convOnly = true;
    const GraphFixture single(buildBiasGraph(convOnly), gfx950Properties());
    ASSERT_TRUE(matchesGraph(GFX950_CONV_FWD_AOT, single.context()));
    // A bias instance would read a bias pointer the unfused launch never passes.
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD_AOT, single.context(), biasKernel()));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_FWD_AOT, single.context(), biasKernel(std::nullopt)));
}

TEST(TestGfx950ConvFwdBiasAotDispatch, ABiasInstancePassesTheContractAndReachesCodeLoading)
{
    const GraphFixture fixture(buildBiasGraph(), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD_BIAS_AOT, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX950_CONV_FWD_BIAS_AOT);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, biasKernel()), 0U);
    try
    {
        // The descriptor names no archive, so loading is the first step that can fail.
        handler.prepare(fixture.context(), *bound, biasKernel());
        FAIL() << "Preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        EXPECT_EQ(message.find("tensor bindings"), std::string::npos) << message;
    }
    // An instance without the bias is refused before any code loads.
    EXPECT_THROW(handler.prepare(fixture.context(), *bound, biasKernel(std::nullopt)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    // So is a binding that disagrees with the graph's bias.
    auto wrongBias = *bound;
    wrongBias[BIAS_TOKEN] = int64_t{99};
    EXPECT_THROW(handler.prepare(fixture.context(), wrongBias, biasKernel()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950ConvFwdBiasAotSignature, ABiasKernelEndsWithTheBiasPointerAndItsBytes)
{
    for(const auto& kernargs : {aot::ConvKernargs{aot::conv::ImplicitGemmKernargs{}},
                                aot::ConvKernargs{aot::conv::DirectKernargs{}}})
    {
        SCOPED_TRACE(kernargs.index());
        const auto& plain = aot::kernelSignature(kernargs, false);
        const auto& biased = aot::kernelSignature(kernargs, true);
        ASSERT_EQ(biased.size(), plain.size() + 2);
        for(size_t i = 0; i < plain.size(); ++i)
        {
            EXPECT_EQ(biased[i].name, plain[i].name);
            EXPECT_EQ(biased[i].kind, plain[i].kind);
            EXPECT_EQ(biased[i].offset, plain[i].offset);
        }
        const auto& pointer = biased[plain.size()];
        const auto& bytes = biased[plain.size() + 1];
        EXPECT_EQ(pointer.name, "Bias");
        EXPECT_EQ(pointer.kind, "global_buffer");
        EXPECT_EQ(pointer.size, sizeof(void*));
        // A pointer starts on its own alignment after the i32 block.
        EXPECT_EQ(pointer.offset % sizeof(void*), 0U);
        EXPECT_GE(pointer.offset, plain.back().offset + plain.back().size);
        EXPECT_EQ(bytes.name, "Bias_bytes");
        EXPECT_EQ(bytes.kind, "by_value");
        EXPECT_EQ(bytes.size, sizeof(int32_t));
        EXPECT_EQ(bytes.offset, pointer.offset + pointer.size);
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
