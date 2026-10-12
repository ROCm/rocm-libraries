// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvDgradGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestVectors.hpp"

/**
 * @file TestGfx950ConvDgradMatchers.cpp
 * @brief hipkernel:Gfx950ConvDgradAot's native symbols, reached through the registry by
 *        the names its descriptors carry: the graph matcher's accept/refuse matrix, the
 *        kernel matcher per family against the rocKE adapter's served matrix, the MFMA
 *        helper lookup, the fallback score, the dispatch handler up to code loading, and
 *        -- when this build packed the production tree -- the shipped catalog run through
 *        the real state manager to show every graph's candidates have distinct knob tuples.
 *        Host-only.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace dgrad = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad;
namespace vectors = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_vectors;
using hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols;

constexpr PackSymbols GFX950_CONV_DGRAD_AOT{"hipkernel:Gfx950ConvDgradAot",
                                            "hipkernel.gfx950_conv_dgrad_aot.graph_match",
                                            "",
                                            "hipkernel.gfx950_conv_dgrad_aot.kernel_match",
                                            "hipkernel.gfx950_conv_dgrad_aot.score",
                                            "hipkernel.gfx950_conv_dgrad_aot.dispatch",
                                            "gfx950_conv_dgrad.dy.uid",
                                            "gfx950_conv_dgrad.w.uid",
                                            "gfx950_conv_dgrad.dx.uid"};
constexpr std::string_view PROBLEM_TOKEN = "_gfx950_conv_dgrad.problem";

struct TensorSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dtype = data_objects::DataType::HALF;
    bool isVirtual = false;
    bool omitDims = false;
    int64_t alignment = 16;
};

/// dims [N, C, H, W] with packed NHWC strides.
TensorSpec nhwc(int64_t n, int64_t c, int64_t h, int64_t w)
{
    return TensorSpec{{n, c, h, w}, {h * w * c, 1, w * c, c}};
}

/// A complete backward-data problem: groups divides c and k; the filter holds c / groups
/// channels; Hi and Wi are dX's.
struct DgradShape
{
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t groups = 1;
    int64_t h = 14;
    int64_t w = 14;
    int64_t y = 3;
    int64_t x = 3;
    int64_t sH = 1;
    int64_t sW = 1;
    int64_t pH = 1;
    int64_t pW = 1;
    int64_t dH = 1;
    int64_t dW = 1;
};

DgradShape shapeOf(const vectors::ProblemValues& v)
{
    return DgradShape{
        v[0], v[1], v[2], v[13], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]};
}

struct GraphSpec
{
    /// dY (uid 1), W (uid 2), dX (uid 3).
    std::array<TensorSpec, 3> tensors;
    std::vector<int64_t> stride{1, 1};
    std::vector<int64_t> dilation{1, 1};
    std::vector<int64_t> prePadding{1, 1};
    std::vector<int64_t> postPadding{1, 1};
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    bool forwardNode = false;
    unsigned int nodeCount = 1;
    int64_t dyUid = 1;
    int64_t wUid = 2;
    int64_t dxUid = 3;
};

/// dY's extent by Python's floor rule, for shapes the tests know are valid.
int64_t outputExtent(int64_t in, int64_t filter, int64_t stride, int64_t pad, int64_t dilation)
{
    return (in + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
}

GraphSpec graphSpec(const DgradShape& s,
                    data_objects::DataType dtype = data_objects::DataType::HALF)
{
    GraphSpec spec;
    const auto ho = outputExtent(s.h, s.y, s.sH, s.pH, s.dH);
    const auto wo = outputExtent(s.w, s.x, s.sW, s.pW, s.dW);
    spec.tensors[0] = nhwc(s.n, s.k, ho, wo);
    spec.tensors[1] = nhwc(s.k, s.c / s.groups, s.y, s.x);
    spec.tensors[2] = nhwc(s.n, s.c, s.h, s.w);
    for(auto& tensor : spec.tensors)
    {
        tensor.dtype = dtype;
    }
    spec.stride = {s.sH, s.sW};
    spec.dilation = {s.dH, s.dW};
    spec.prePadding = spec.postPadding = {s.pH, s.pW};
    return spec;
}

DeviceProperties gfx950Properties(std::string arch = "gfx950")
{
    // By value: matcher tests run on CPU-only and non-gfx950 hosts.
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = 64;
    return properties;
}

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < spec.tensors.size(); ++i)
    {
        const auto& tensor = spec.tensors[i];
        const auto dims = tensor.omitDims ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                          : builder.CreateVector(tensor.dims);
        const auto strides = tensor.omitDims ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                             : builder.CreateVector(tensor.strides);
        data_objects::TensorAttributesBuilder tensorBuilder(builder);
        tensorBuilder.add_uid(static_cast<int64_t>(i + 1));
        tensorBuilder.add_data_type(tensor.dtype);
        tensorBuilder.add_dims(dims);
        tensorBuilder.add_strides(strides);
        tensorBuilder.add_virtual_(tensor.isVirtual);
        tensorBuilder.add_alignment(tensor.alignment);
        tensors.push_back(tensorBuilder.Finish());
    }
    flatbuffers::Offset<void> attributes;
    auto attributesType = data_objects::NodeAttributes::ConvolutionBwdAttributes;
    if(spec.forwardNode)
    {
        attributesType = data_objects::NodeAttributes::ConvolutionFwdAttributes;
        attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                        spec.dxUid,
                                                                        spec.wUid,
                                                                        spec.dyUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
    }
    else
    {
        attributes = data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                        spec.dyUid,
                                                                        spec.wUid,
                                                                        spec.dxUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
    }
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(unsigned int i = 0; i < spec.nodeCount; ++i)
    {
        nodes.push_back(data_objects::CreateNodeDirect(
            builder, "gfx950_conv_dgrad_test", spec.computeType, attributesType, attributes));
    }
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_tensors(tensorVector);
    graphBuilder.add_nodes(nodeVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

GraphFixture fixtureOf(const DgradShape& shape,
                       data_objects::DataType dtype = data_objects::DataType::HALF)
{
    return GraphFixture(buildGraph(graphSpec(shape, dtype)), gfx950Properties());
}

DescriptorId idOf(uint32_t value, uint8_t tag)
{
    DescriptorId id{};
    id[0] = tag;
    id[12] = static_cast<uint8_t>(value >> 24U);
    id[13] = static_cast<uint8_t>(value >> 16U);
    id[14] = static_cast<uint8_t>(value >> 8U);
    id[15] = static_cast<uint8_t>(value);
    return id;
}

/// A kpack kernel carrying @p instance's complete metadata, as the descriptor loader
/// delivers it (every field present, omitted ones completed from the KMD defaults).
KernelDefinition
    kernelOf(const vectors::Instance& instance, const std::string& dtype = "fp16", uint8_t pack = 1)
{
    KernelDefinition kernel;
    kernel.kernelId = idOf(
        static_cast<uint32_t>(std::hash<std::string>{}(std::string(instance.instanceId) + dtype)),
        pack);
    kernel.packId = idOf(0, pack);
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.name = std::string("gfx950_conv_dgrad_aot_") + dtype + "_" + instance.instanceId;
    kernel.metadata = {{"dtype", dtype},
                       {"tile_m", instance.tile_m},
                       {"tile_n", instance.tile_n},
                       {"tile_k", instance.tile_k},
                       {"warp_m", instance.warp_m},
                       {"warp_n", instance.warp_n},
                       {"warp_tile_m", instance.warp_tile_m},
                       {"warp_tile_n", instance.warp_tile_n},
                       {"warp_tile_k", instance.warp_tile_k},
                       {"wave_size", int64_t{64}},
                       {"pipeline", std::string(instance.pipeline)},
                       {"epilogue", std::string(instance.epilogue)},
                       {"vector_size_a", instance.vector_size_a},
                       {"vector_size_b", instance.vector_size_b},
                       {"vector_size_c", instance.vector_size_c},
                       {"lds_k_outer", instance.lds_k_outer},
                       {"max_sub_gemms", instance.max_sub_gemms},
                       {"grouped", instance.grouped},
                       {"instance_id", std::string(instance.instanceId)},
                       {"layout", std::string("NHWC")},
                       {"kernel_family", instance.kernel_family},
                       {"direct_variant", std::string(instance.direct_variant)},
                       {"filter_h", instance.filter_h},
                       {"filter_w", instance.filter_w},
                       {"stride_h", instance.stride_h},
                       {"stride_w", instance.stride_w},
                       {"pad_h", instance.pad_h},
                       {"pad_w", instance.pad_w},
                       {"dilation_h", instance.dilation_h},
                       {"dilation_w", instance.dilation_w},
                       {"cpg", instance.cpg},
                       {"kpg", instance.kpg},
                       {"block_w", instance.block_w},
                       {"block_waves", instance.block_waves},
                       {"block_h", instance.block_h},
                       {"block_q", instance.block_q},
                       {"block_groups", instance.block_groups},
                       {"fold_k32", instance.fold_k32},
                       {"waves_k", instance.waves_k},
                       {"waves_q", instance.waves_q},
                       {"persistent_grid", instance.persistent_grid},
                       {"runtime_k_loop", instance.runtime_k_loop}};
    return kernel;
}

/// Runs the pack's KernelPrepareFn over @p kernels as the engine's shared state does at
/// load: each carries its prepared payload, and each helper is filed for the MFMA
/// pipeline of its own pack.
void prepare(std::vector<KernelDefinition>& kernels)
{
    registerNativeIngestorSymbols();
    const auto prepareFn
        = KernelPrepareRegistry::tryResolve(std::string(GFX950_CONV_DGRAD_AOT.kernelMatcher));
    ASSERT_NE(prepareFn, nullptr);
    for(auto& kernel : kernels)
    {
        kernel.prepared = prepareFn(kernel);
        kernel.preparedBy = prepareFn;
    }
}

/// The fp16 catalog as one pack (or the bf16 one, relabeled), prepared.
std::vector<KernelDefinition> catalogPack(const std::string& dtype = "fp16", uint8_t pack = 1)
{
    std::vector<KernelDefinition> kernels;
    for(const auto& instance : vectors::catalogInstances())
    {
        kernels.push_back(kernelOf(instance, dtype, pack));
    }
    prepare(kernels);
    return kernels;
}

const vectors::Instance& firstInstance(const std::function<bool(const vectors::Instance&)>& pick)
{
    const auto& instances = vectors::catalogInstances();
    const auto it = std::find_if(instances.begin(), instances.end(), pick);
    if(it == instances.end())
    {
        throw std::runtime_error("no catalog instance matches");
    }
    return *it;
}

const vectors::Instance& implicitInstance()
{
    return firstInstance([](const auto& i) {
        return i.kernel_family == 0 && i.vector_size_a == 8 && i.vector_size_b == 8
               && i.vector_size_c == 8 && std::string_view(i.pipeline) == "mem";
    });
}

const vectors::Instance& depthwiseInstance(int64_t filter, int64_t stride)
{
    return firstInstance([=](const auto& i) {
        return std::string_view(i.direct_variant) == "depthwise_dgrad" && i.filter_h == filter
               && i.stride_h == stride;
    });
}

const vectors::Instance& mfmaInstance(int64_t cpg, int64_t kpg, bool coalesced)
{
    return firstInstance([=](const auto& i) {
        return std::string_view(i.direct_variant) == "grouped_dgrad_mfma" && i.cpg == cpg
               && i.kpg == kpg && (i.waves_k > 1 || i.runtime_k_loop != 0) == coalesced;
    });
}

std::vector<const vectors::Instance*> helperInstances()
{
    std::vector<const vectors::Instance*> helpers;
    for(const auto& instance : vectors::catalogInstances())
    {
        if(instance.kernel_family == 2)
        {
            helpers.push_back(&instance);
        }
    }
    return helpers;
}

TEST(TestGfx950ConvDgradGraphMatcher, AcceptsBothStorageTypesAndBindsTheBackwardTensors)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        const auto fixture = fixtureOf(DgradShape{}, dtype);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_DGRAD_AOT.inputAToken), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_DGRAD_AOT.inputBToken), 2);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_DGRAD_AOT.outputToken), 3);
        const auto problem = bound->find(std::string(PROBLEM_TOKEN));
        ASSERT_NE(problem, bound->end());
        const auto* values = std::get_if<std::vector<int64_t>>(&problem->second);
        ASSERT_NE(values, nullptr);
        // dY, W, dX, then N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups, dtype.
        EXPECT_EQ(*values,
                  (std::vector<int64_t>{1,
                                        2,
                                        3,
                                        2,
                                        32,
                                        32,
                                        14,
                                        14,
                                        3,
                                        3,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        1,
                                        static_cast<int64_t>(dtype)}));
        EXPECT_EQ(bound->size(), 4U);
    }
}

TEST(TestGfx950ConvDgradGraphMatcher, TheProblemTokenIsPrivateAndNoFeature)
{
    const auto fixture = fixtureOf(DgradShape{});
    const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    std::vector<std::string> features;
    for(const auto& [token, value] : *bound)
    {
        if(!detail::isPrivateToken(token))
        {
            features.push_back(token);
        }
    }
    std::sort(features.begin(), features.end());
    std::vector<std::string> tensors{std::string(GFX950_CONV_DGRAD_AOT.inputAToken),
                                     std::string(GFX950_CONV_DGRAD_AOT.inputBToken),
                                     std::string(GFX950_CONV_DGRAD_AOT.outputToken)};
    std::sort(tensors.begin(), tensors.end());
    // Only the three tensor bindings reach a ranker or a corpus row.
    EXPECT_EQ(features, tensors);
    EXPECT_TRUE(detail::isPrivateToken(std::string(PROBLEM_TOKEN)));
    EXPECT_TRUE(bound->count(std::string(PROBLEM_TOKEN)));
}

TEST(TestGfx950ConvDgradGraphMatcher, InputExtentsComeFromDx)
{
    // Hi = 13 and Hi = 14 both give Ho = 7 under stride 2: dX decides.
    for(const auto h : {int64_t{13}, int64_t{14}})
    {
        DgradShape shape;
        shape.h = shape.w = h;
        shape.sH = shape.sW = 2;
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        const auto& values = std::get<std::vector<int64_t>>(bound->at(std::string(PROBLEM_TOKEN)));
        EXPECT_EQ(values[6], h);
        EXPECT_EQ(values[7], h);
    }
}

struct RefusalCase
{
    const char* name;
    std::function<void(GraphSpec&)> change;
};

TEST(TestGfx950ConvDgradGraphMatcher, RefusesEverythingOutsideTheGraphContract)
{
    const auto shape = DgradShape{};
    const std::vector<RefusalCase> cases{
        {"forward convolution node", [](auto& s) { s.forwardNode = true; }},
        {"fused graph", [](auto& s) { s.nodeCount = 2; }},
        {"convolution mode", [](auto& s) { s.mode = data_objects::ConvMode::CONVOLUTION; }},
        {"half accumulation", [](auto& s) { s.computeType = data_objects::DataType::HALF; }},
        {"asymmetric padding", [](auto& s) { s.postPadding = {1, 2}; }},
        {"3-D padding",
         [](auto& s) {
             s.prePadding = {1, 1, 1};
             s.postPadding = {1, 1, 1};
         }},
        {"float tensors",
         [](auto& s) {
             for(auto& tensor : s.tensors)
             {
                 tensor.dtype = data_objects::DataType::FLOAT;
             }
         }},
        {"mixed dtypes", [](auto& s) { s.tensors[1].dtype = data_objects::DataType::BFLOAT16; }},
        {"NCHW dY", [](auto& s) { s.tensors[0].strides = {6272, 196, 14, 1}; }},
        {"NCHW W", [](auto& s) { s.tensors[1].strides = {288, 9, 3, 1}; }},
        {"NCHW dX", [](auto& s) { s.tensors[2].strides = {6272, 196, 14, 1}; }},
        {"padded dX row", [](auto& s) { s.tensors[2].strides = {7168, 1, 512, 32}; }},
        {"virtual dX", [](auto& s) { s.tensors[2].isVirtual = true; }},
        {"8-byte aligned dY", [](auto& s) { s.tensors[0].alignment = 8; }},
        {"dX without dims", [](auto& s) { s.tensors[2].omitDims = true; }},
        {"dY without dims", [](auto& s) { s.tensors[0].omitDims = true; }},
        {"aliased dX uid", [](auto& s) { s.dxUid = s.dyUid; }},
        {"dangling W uid", [](auto& s) { s.wUid = 999; }},
        {"dY batch mismatch", [](auto& s) { s.tensors[0] = nhwc(3, 32, 14, 14); }},
        {"dY channels mismatch", [](auto& s) { s.tensors[0] = nhwc(2, 16, 14, 14); }},
        {"dY height mismatch", [](auto& s) { s.tensors[0] = nhwc(2, 32, 13, 14); }},
        {"dY width mismatch", [](auto& s) { s.tensors[0] = nhwc(2, 32, 14, 15); }},
        {"filter channels do not divide dX channels",
         [](auto& s) { s.tensors[1] = nhwc(32, 12, 3, 3); }},
        {"groups do not divide K",
         [](auto& s) {
             // C = 32, W [30, 16, 3, 3]: two groups, K = 30 is divisible; make it 31.
             s.tensors[1] = nhwc(31, 16, 3, 3);
             s.tensors[0] = nhwc(2, 31, 14, 14);
         }},
        {"too many groups",
         [](auto& s) {
             s.tensors[2] = nhwc(1, 65536, 1, 1);
             s.tensors[1] = nhwc(65536, 1, 1, 1);
             s.tensors[0] = nhwc(1, 65536, 1, 1);
             s.prePadding = s.postPadding = {0, 0};
         }},
        {"negative output numerator",
         [](auto& s) {
             // Hi = Wi = 1, Y = X = 2, p = 0, s = 2: Python floors to Ho = 0; C++ would
             // truncate to Ho = 1.
             s.tensors[2] = nhwc(2, 32, 1, 1);
             s.tensors[1] = nhwc(32, 32, 2, 2);
             s.tensors[0] = nhwc(2, 32, 1, 1);
             s.prePadding = s.postPadding = {0, 0};
             s.stride = {2, 2};
         }},
    };
    {
        const GraphFixture accepted(buildGraph(graphSpec(shape)), gfx950Properties());
        ASSERT_TRUE(matchesGraph(GFX950_CONV_DGRAD_AOT, accepted.context()));
    }
    for(const auto& [name, change] : cases)
    {
        SCOPED_TRACE(name);
        auto spec = graphSpec(shape);
        change(spec);
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        EXPECT_FALSE(matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context()));
    }
    for(const auto* arch : {"gfx942", "gfx9500", ""})
    {
        SCOPED_TRACE(arch);
        const GraphFixture fixture(buildGraph(graphSpec(shape)), gfx950Properties(arch));
        EXPECT_FALSE(matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context()));
    }
    auto wave32 = gfx950Properties();
    wave32.warpSize = 32;
    const GraphFixture wave32Fixture(buildGraph(graphSpec(shape)), wave32);
    EXPECT_FALSE(matchesGraph(GFX950_CONV_DGRAD_AOT, wave32Fixture.context()));
    // The forward engine does not read a backward graph either.
    const GraphFixture fixture(buildGraph(graphSpec(shape)), gfx950Properties());
    registerNativeIngestorSymbols();
    EXPECT_FALSE(GraphMatchRegistry::resolve("hipkernel.gfx950_conv_fwd_aot.graph_match")(
        fixture.context()));
}

TEST(TestGfx950ConvDgradGraphMatcher, AcceptsGroupedDepthwiseStridedAndDilatedProblems)
{
    std::vector<DgradShape> shapes(6);
    shapes[1].groups = 4;
    shapes[2].groups = shapes[2].c = shapes[2].k = 32;
    shapes[3].sH = 2;
    shapes[3].sW = 3;
    shapes[4].dH = shapes[4].dW = 2;
    shapes[4].pH = shapes[4].pW = 2;
    shapes[5].y = 1;
    shapes[5].x = 5;
    shapes[5].pH = 0;
    shapes[5].pW = 2;
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        SCOPED_TRACE(i);
        const auto fixture = fixtureOf(shapes[i]);
        EXPECT_TRUE(matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context()));
    }
}

TEST(TestGfx950ConvDgradKernelMatcher, EachFamilyServesItsOwnProblemsInItsOwnDtype)
{
    auto kernels = std::vector<KernelDefinition>{kernelOf(implicitInstance()),
                                                 kernelOf(depthwiseInstance(3, 1)),
                                                 kernelOf(mfmaInstance(16, 16, false))};
    for(const auto* helper : helperInstances())
    {
        kernels.push_back(kernelOf(*helper));
    }
    prepare(kernels);
    const auto& implicit = kernels[0];
    const auto& depthwise = kernels[1];
    const auto& mfma = kernels[2];
    const auto matches = [](const DgradShape& shape,
                            const KernelDefinition& kernel,
                            data_objects::DataType dtype = data_objects::DataType::HALF) {
        const auto fixture = fixtureOf(shape, dtype);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        EXPECT_TRUE(bound);
        return bound && matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernel, *bound);
    };
    DgradShape dense;
    DgradShape depthwiseShape;
    depthwiseShape.c = depthwiseShape.k = depthwiseShape.groups = 96;
    DgradShape grouped;
    grouped.c = grouped.k = 128;
    grouped.groups = 8;
    EXPECT_TRUE(matches(dense, implicit));
    EXPECT_FALSE(matches(dense, depthwise));
    EXPECT_FALSE(matches(dense, mfma));
    EXPECT_FALSE(matches(depthwiseShape, implicit));
    EXPECT_TRUE(matches(depthwiseShape, depthwise));
    EXPECT_FALSE(matches(depthwiseShape, mfma));
    EXPECT_TRUE(matches(grouped, implicit));
    EXPECT_FALSE(matches(grouped, depthwise));
    EXPECT_TRUE(matches(grouped, mfma));
    // The dtype is pinned: an fp16 kernel never serves a bf16 graph.
    EXPECT_FALSE(matches(dense, implicit, data_objects::DataType::BFLOAT16));
    EXPECT_FALSE(matches(depthwiseShape, depthwise, data_objects::DataType::BFLOAT16));
    EXPECT_FALSE(matches(grouped, mfma, data_objects::DataType::BFLOAT16));
    // Without the bound token the matcher matches the graph itself, with the same answer.
    const auto fixture = fixtureOf(grouped);
    EXPECT_TRUE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), mfma));
    // The helpers are never admitted, whatever the graph.
    for(size_t i = 3; i < kernels.size(); ++i)
    {
        const auto cpg = std::get<int64_t>(kernels[i].metadata.at("cpg"));
        const auto kpg = std::get<int64_t>(kernels[i].metadata.at("kpg"));
        for(const auto groups : {int64_t{1}, int64_t{2}})
        {
            DgradShape own;
            own.groups = groups;
            own.c = cpg * groups;
            own.k = kpg * groups;
            EXPECT_FALSE(matches(own, kernels[i])) << kernels[i].name;
        }
        EXPECT_FALSE(matches(dense, kernels[i])) << kernels[i].name;
    }
}

TEST(TestGfx950ConvDgradKernelMatcher, AnMfmaInstanceNeedsItsHelpersInItsOwnPack)
{
    DgradShape dense;
    dense.n = 1;
    dense.c = 48;
    dense.k = 192;
    const auto fixture = fixtureOf(dense);
    const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    const auto& coalesced = mfmaInstance(48, 192, true);
    // Pack 7 holds the instance and no helper.
    std::vector<KernelDefinition> lonely{kernelOf(coalesced, "fp16", 7)};
    prepare(lonely);
    EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), lonely[0], *bound));
    // Pack 8 holds only the transpose; this instance reorganizes too.
    std::vector<KernelDefinition> half{kernelOf(coalesced, "fp16", 8)};
    std::vector<KernelDefinition> fullPack{kernelOf(coalesced, "fp16", 9)};
    for(const auto* helper : helperInstances())
    {
        if(helper->cpg != 48 || helper->kpg != 192)
        {
            continue;
        }
        if(std::string_view(helper->direct_variant) == "transpose")
        {
            half.push_back(kernelOf(*helper, "fp16", 8));
        }
        fullPack.push_back(kernelOf(*helper, "fp16", 9));
        // A bf16 helper in the same pack is not this fp16 instance's.
        half.push_back(kernelOf(*helper, "bf16", 8));
    }
    prepare(half);
    prepare(fullPack);
    EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), half[0], *bound));
    EXPECT_TRUE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), fullPack[0], *bound));
    // A reorganize helper keyed on another fold_k32 does not stand in. c4k320 ships a
    // reorganize helper for each fold: one pack holds the instance's transpose and only the
    // other fold's reorganize, another both of the instance's.
    DgradShape wide;
    wide.n = 1;
    wide.c = 4;
    wide.k = 320;
    const auto wideFixture = fixtureOf(wide);
    const auto wideBound = matchesGraph(GFX950_CONV_DGRAD_AOT, wideFixture.context());
    ASSERT_TRUE(wideBound);
    for(const auto fold : {int64_t{0}, int64_t{1}})
    {
        SCOPED_TRACE(fold);
        const auto& instance = firstInstance([=](const auto& i) {
            return std::string_view(i.direct_variant) == "grouped_dgrad_mfma" && i.cpg == 4
                   && i.kpg == 320 && i.waves_k > 1 && i.fold_k32 == fold;
        });
        const auto wrongPack = static_cast<uint8_t>(10 + 2 * fold);
        const auto rightPack = static_cast<uint8_t>(11 + 2 * fold);
        std::vector<KernelDefinition> wrongFold{kernelOf(instance, "fp16", wrongPack)};
        std::vector<KernelDefinition> rightFold{kernelOf(instance, "fp16", rightPack)};
        for(const auto* helper : helperInstances())
        {
            if(helper->cpg != 4 || helper->kpg != 320)
            {
                continue;
            }
            const bool transpose = std::string_view(helper->direct_variant) == "transpose";
            if(transpose || helper->fold_k32 != fold)
            {
                wrongFold.push_back(kernelOf(*helper, "fp16", wrongPack));
            }
            if(transpose || helper->fold_k32 == fold)
            {
                rightFold.push_back(kernelOf(*helper, "fp16", rightPack));
            }
        }
        ASSERT_EQ(wrongFold.size(), 3U);
        ASSERT_EQ(rightFold.size(), 3U);
        prepare(wrongFold);
        prepare(rightFold);
        EXPECT_FALSE(
            matchesKernel(GFX950_CONV_DGRAD_AOT, wideFixture.context(), wrongFold[0], *wideBound));
        EXPECT_TRUE(
            matchesKernel(GFX950_CONV_DGRAD_AOT, wideFixture.context(), rightFold[0], *wideBound));
    }
}

/// Every served-matrix row a graph can state (dY extents positive), both dtypes.
TEST(TestGfx950ConvDgradKernelMatcher, ServesExactlyWhatTheAdapterServesOverTheCatalog)
{
    const auto fp16 = catalogPack("fp16", 1);
    const auto bf16 = catalogPack("bf16", 2);
    const auto& instances = vectors::catalogInstances();
    ASSERT_EQ(fp16.size(), instances.size());
    size_t rows = 0;
    size_t served = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto shape = shapeOf(row.problem);
        const dgrad::Problem problem{row.problem[0],
                                     row.problem[1],
                                     row.problem[2],
                                     row.problem[3],
                                     row.problem[4],
                                     row.problem[5],
                                     row.problem[6],
                                     row.problem[7],
                                     row.problem[8],
                                     row.problem[9],
                                     row.problem[10],
                                     row.problem[11],
                                     row.problem[12],
                                     row.problem[13]};
        if(!dgrad::deriveGeometry(problem))
        {
            // No graph states this problem; GraphMatcher tests cover its refusal.
            EXPECT_EQ(std::string_view(row.served).find('1'), std::string_view::npos);
            continue;
        }
        ++rows;
        for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
        {
            const auto fixture = fixtureOf(shape, dtype);
            const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
            ASSERT_TRUE(bound);
            const auto& own = dtype == data_objects::DataType::HALF ? fp16 : bf16;
            const auto& other = dtype == data_objects::DataType::HALF ? bf16 : fp16;
            for(size_t i = 0; i < instances.size(); ++i)
            {
                const bool expected = row.served[i] == '1';
                const bool actual
                    = matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
                EXPECT_EQ(actual, expected)
                    << own[i].name << " on N" << shape.n << " C" << shape.c << " K" << shape.k
                    << " G" << shape.groups << " " << shape.h << "x" << shape.w << " f" << shape.y
                    << "x" << shape.x << " s" << shape.sH << "x" << shape.sW << " p" << shape.pH
                    << "x" << shape.pW << " d" << shape.dH << "x" << shape.dW;
                EXPECT_FALSE(
                    matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), other[i], *bound));
                // A served kernel scores, a refused one ranks last.
                const auto score
                    = scoreKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
                EXPECT_EQ(score >= 0.0, expected) << own[i].name << " score " << score;
                served += actual ? 1 : 0;
            }
        }
    }
    EXPECT_GT(rows, 200U);
    EXPECT_GT(served, 2000U);
}

TEST(TestGfx950ConvDgradKernelMatcher, RefusesOutOfContractOrIncompleteMetadata)
{
    DgradShape grouped;
    grouped.c = grouped.k = 128;
    grouped.groups = 8;
    DgradShape depthwiseShape;
    depthwiseShape.c = depthwiseShape.k = depthwiseShape.groups = 96;
    const std::vector<std::pair<DgradShape, const vectors::Instance*>> cases{
        {DgradShape{}, &implicitInstance()},
        {depthwiseShape, &depthwiseInstance(3, 1)},
        {grouped, &mfmaInstance(16, 16, false)}};
    for(const auto& [shape, instance] : cases)
    {
        SCOPED_TRACE(instance->direct_variant);
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        std::vector<KernelDefinition> base{kernelOf(*instance, "fp16", 20)};
        for(const auto* helper : helperInstances())
        {
            base.push_back(kernelOf(*helper, "fp16", 20));
        }
        prepare(base);
        ASSERT_TRUE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), base[0], *bound));
        const bool direct = instance->kernel_family == 1;
        const std::vector<std::pair<const char*, MetadataValue>> outOfContract{
            {"layout", std::string("NCHW")},
            {"wave_size", int64_t{32}},
            {"kernel_family", int64_t{3}},
            {"kernel_family", int64_t{2}},
            {"kernel_family", int64_t{direct ? 0 : 1}},
            {"kernel_family", std::string("0")},
            {"direct_variant", std::string("std")},
            {"direct_variant", std::string(direct ? "none" : "depthwise_dgrad")},
            {"pipeline", std::string(direct ? "mem" : "direct")},
            {"pipeline", std::string("wavelet")},
            {"epilogue", std::string(direct ? "cshuffle" : "none")},
            {"tile_k", int64_t{direct ? 32 : 0}},
            {"vector_size_a", int64_t{direct ? 1 : 3}},
            {"lds_k_outer", int64_t{2}},
            {"max_sub_gemms", int64_t{direct ? 1024 : 0}},
            {"grouped", int64_t{direct ? 1 : 2}},
            {"filter_h", int64_t{direct ? 5 : 3}},
            {"dilation_w", int64_t{2}},
            {"stride_w", int64_t{direct ? 2 : 1}},
            {"block_w", int64_t{direct && instance->block_w != 0 ? 0 : 5}},
            {"fold_k32", int64_t{2}},
            {"persistent_grid", int64_t{direct ? 2 : 1}},
            {"waves_k", int64_t{direct && instance->waves_k != 0 ? 0 : 1}}};
        for(const auto& [field, value] : outOfContract)
        {
            SCOPED_TRACE(field);
            std::vector<KernelDefinition> kernel{base[0]};
            kernel[0].metadata[field] = value;
            kernel[0].prepared = nullptr;
            prepare(kernel);
            EXPECT_FALSE(
                matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernel[0], *bound));
        }
        for(const auto& field : base[0].metadata)
        {
            if(field.first == "instance_id")
            {
                continue; // Tracking only; the launch never reads it.
            }
            SCOPED_TRACE(field.first);
            auto kernel = base[0];
            kernel.metadata.erase(field.first);
            kernel.prepared = nullptr;
            kernel.preparedBy = nullptr;
            EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernel, *bound));
        }
        auto embedded = base[0];
        embedded.source.kind = KernelSourceKind::EMBEDDED_SOURCE;
        embedded.prepared = nullptr;
        embedded.preparedBy = nullptr;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), embedded, *bound));
    }
}

TEST(TestGfx950ConvDgradScore, DepthwiseThenMfmaThenImplicitGemmAndRefusalsLast)
{
    auto kernels = catalogPack("fp16", 30);
    const auto& instances = vectors::catalogInstances();
    const auto scores = [&](const DgradShape& shape) {
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        EXPECT_TRUE(bound);
        std::map<int64_t, std::pair<double, double>> range; // family -> (min, max) served
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const auto score
                = scoreKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernels[i], *bound);
            const bool served
                = matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernels[i], *bound);
            if(!served)
            {
                EXPECT_EQ(score, -1.0) << kernels[i].name;
                continue;
            }
            const auto family
                = std::string_view(instances[i].direct_variant) == "grouped_dgrad_mfma"
                      ? int64_t{2}
                      : instances[i].kernel_family;
            auto [it, inserted] = range.emplace(family, std::make_pair(score, score));
            it->second.first = std::min(it->second.first, score);
            it->second.second = std::max(it->second.second, score);
        }
        return range;
    };
    DgradShape depthwise;
    depthwise.c = depthwise.k = depthwise.groups = 96;
    const auto dw = scores(depthwise);
    ASSERT_TRUE(dw.count(1));
    EXPECT_GE(dw.at(1).first, 3.0);
    DgradShape grouped;
    grouped.c = grouped.k = 128;
    grouped.groups = 8;
    const auto mixed = scores(grouped);
    ASSERT_TRUE(mixed.count(0));
    ASSERT_TRUE(mixed.count(2));
    EXPECT_GT(mixed.at(2).first, mixed.at(0).second);
    EXPECT_LT(mixed.at(2).second, 3.0);
    EXPECT_GE(mixed.at(0).first, 0.0);
    EXPECT_LT(mixed.at(0).second, 1.01);
}

TEST(TestGfx950ConvDgradDispatch, WorkspaceFollowsTheFamilyAndPrepareReachesCodeLoading)
{
    auto kernels = std::vector<KernelDefinition>{kernelOf(implicitInstance(), "fp16", 40),
                                                 kernelOf(depthwiseInstance(3, 2), "fp16", 40),
                                                 kernelOf(mfmaInstance(48, 192, true), "fp16", 40)};
    for(const auto* helper : helperInstances())
    {
        kernels.push_back(kernelOf(*helper, "fp16", 40));
    }
    prepare(kernels);
    const auto& handler = dispatchHandler(GFX950_CONV_DGRAD_AOT);
    const auto reachesLoading =
        [&](const GraphFixture& fixture, const BoundTokens& bound, const KernelDefinition& kernel) {
            try
            {
                // The descriptor names no archive, so loading is the first step that can fail.
                handler.prepare(fixture.context(), bound, kernel);
                ADD_FAILURE() << "preparation loaded a kernel from an empty archive path";
            }
            catch(const std::exception& error)
            {
                const std::string message = error.what();
                EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
                EXPECT_EQ(message.find("weight-transform helper"), std::string::npos) << message;
            }
        };

    // Implicit GEMM, strided: (22 + 8) * 4 bytes per sub-GEMM.
    DgradShape strided;
    strided.h = strided.w = 15;
    strided.sH = strided.sW = 2;
    const auto stridedFixture = fixtureOf(strided);
    const auto stridedBound = matchesGraph(GFX950_CONV_DGRAD_AOT, stridedFixture.context());
    ASSERT_TRUE(stridedBound);
    EXPECT_EQ(handler.workspaceBytes(stridedFixture.context(), *stridedBound, kernels[0]),
              4U * 30U * 4U);
    reachesLoading(stridedFixture, *stridedBound, kernels[0]);
    // Refused before any code loads.
    EXPECT_EQ(handler.workspaceBytes(stridedFixture.context(), *stridedBound, kernels[1]), 0U);
    EXPECT_THROW(handler.prepare(stridedFixture.context(), *stridedBound, kernels[1]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(handler.prepare(stridedFixture.context(), *stridedBound, kernels[3]),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    // Depthwise: no workspace.
    DgradShape depthwise;
    depthwise.c = depthwise.k = depthwise.groups = 64;
    depthwise.h = 29;
    depthwise.w = 27;
    depthwise.sH = depthwise.sW = 2;
    const auto depthwiseFixture = fixtureOf(depthwise);
    const auto depthwiseBound = matchesGraph(GFX950_CONV_DGRAD_AOT, depthwiseFixture.context());
    ASSERT_TRUE(depthwiseBound);
    EXPECT_EQ(handler.workspaceBytes(depthwiseFixture.context(), *depthwiseBound, kernels[1]), 0U);
    reachesLoading(depthwiseFixture, *depthwiseBound, kernels[1]);

    // MFMA with the reorganize step: ws_t, then ws_coa at ws_t rounded up to 256.
    DgradShape dense;
    dense.c = 48;
    dense.k = 192;
    const auto mfmaFixture = fixtureOf(dense);
    const auto mfmaBound = matchesGraph(GFX950_CONV_DGRAD_AOT, mfmaFixture.context());
    ASSERT_TRUE(mfmaBound);
    EXPECT_EQ(handler.workspaceBytes(mfmaFixture.context(), *mfmaBound, kernels[2]), 331776U);
    reachesLoading(mfmaFixture, *mfmaBound, kernels[2]);
    // The same instance on a grouped graph is refused (the reorganize helper's group bug).
    DgradShape grouped = dense;
    grouped.c *= 2;
    grouped.k *= 2;
    grouped.groups = 2;
    const auto groupedFixture = fixtureOf(grouped);
    const auto groupedBound = matchesGraph(GFX950_CONV_DGRAD_AOT, groupedFixture.context());
    ASSERT_TRUE(groupedBound);
    EXPECT_EQ(handler.workspaceBytes(groupedFixture.context(), *groupedBound, kernels[2]), 0U);
    EXPECT_THROW(handler.prepare(groupedFixture.context(), *groupedBound, kernels[2]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    // Bindings other than the graph matcher's are refused.
    auto otherBound = *mfmaBound;
    otherBound[std::string(GFX950_CONV_DGRAD_AOT.outputToken)] = int64_t{7};
    EXPECT_THROW(handler.prepare(mfmaFixture.context(), otherBound, kernels[2]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
/// The gfx950 shard of the production tree this build packed, when it holds the dgrad
/// bundle; an empty path otherwise. The whole shard is loaded, not the bundle's folder: a
/// kernel's archive sits at the shard root and must lie inside the tree it was read from.
std::filesystem::path packedDgradShard()
{
    const auto shard = hip_kernel_provider::testing::descriptorSetRoot(
                           HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR)
                       / "gfx950";
    std::error_code missing;
    return std::filesystem::is_directory(shard / "rocKE" / "gfx950_conv_dgrad_aot", missing)
               ? shard
               : std::filesystem::path();
}

/// The shipped bundle's state manager, built once, or null when it is not staged.
const KernelIngestorStateManager<Handle>* shippedManager(std::vector<std::string>* knobs = nullptr)
{
    static const auto s_loaded = []() {
        std::pair<std::unique_ptr<KernelIngestorStateManager<Handle>>, std::vector<std::string>>
            loaded;
        const auto shard = packedDgradShard();
        if(shard.empty())
        {
            return loaded;
        }
        registerNativeIngestorSymbols();
        auto sets = resolveDescriptorSets(loadDescriptorCatalog(shard));
        const auto set = std::find_if(sets.begin(), sets.end(), [](const auto& candidate) {
            return candidate.engine.name == GFX950_CONV_DGRAD_AOT.engineName;
        });
        if(set == sets.end())
        {
            throw std::runtime_error("the staged bundle holds no hipkernel:Gfx950ConvDgradAot");
        }
        loaded.second = set->engine.knobs;
        const auto graphMatch = set->engine.graphMatchNativeSymbol;
        loaded.first = makeStateManager<Handle>(std::move(*set), graphMatch);
        return loaded;
    }();
    if(knobs != nullptr)
    {
        *knobs = s_loaded.second;
    }
    return s_loaded.first.get();
}

constexpr const char* NOT_STAGED
    = "this configuration packed no gfx950 production tree, so the shipped "
      "hipkernel:Gfx950ConvDgradAot bundle is not staged beside this binary";

TEST(TestGfx950ConvDgradCatalog, EveryGraphsCandidatesHaveDistinctKnobTuples)
{
    std::vector<std::string> knobs;
    const auto* manager = shippedManager(&knobs);
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    ASSERT_EQ(knobs.size(), 26U);

    // The fp16 and bf16 catalogs, each run against every served-matrix problem a graph can
    // state: the candidate set is what the real matchers, the shipped metadata (with its
    // KMD defaults) and the helper lookup admit, and its size is the adapter's count.
    size_t graphs = 0;
    size_t candidates = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto shape = shapeOf(row.problem);
        const dgrad::Problem problem{shape.n,
                                     shape.c,
                                     shape.k,
                                     shape.h,
                                     shape.w,
                                     shape.y,
                                     shape.x,
                                     shape.sH,
                                     shape.sW,
                                     shape.pH,
                                     shape.pW,
                                     shape.dH,
                                     shape.dW,
                                     shape.groups};
        if(!dgrad::deriveGeometry(problem))
        {
            continue;
        }
        const auto expected = static_cast<size_t>(
            std::count(row.served, row.served + std::string_view(row.served).size(), '1'));
        for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
        {
            const auto fixture = fixtureOf(shape, dtype);
            const auto catalog = manager->unsortedCatalog(fixture.context());
            EXPECT_EQ(catalog.entries.size(), expected)
                << "N" << shape.n << " C" << shape.c << " K" << shape.k << " G" << shape.groups
                << " " << shape.h << "x" << shape.w << " f" << shape.y << "x" << shape.x;
            std::set<std::vector<int64_t>> tuples;
            for(const auto& kernel : catalog.entries)
            {
                EXPECT_EQ(std::get<std::string>(kernel.metadata.at("dtype")),
                          dtype == data_objects::DataType::HALF ? "fp16" : "bf16");
                EXPECT_NE(std::get<int64_t>(kernel.metadata.at("kernel_family")), 2) << kernel.name;
                std::vector<int64_t> tuple;
                for(const auto& knob : knobs)
                {
                    const auto ordinal = manager->knobOrdinal(knob, kernel.metadata.at(knob));
                    ASSERT_TRUE(ordinal) << knob;
                    tuple.push_back(*ordinal);
                }
                EXPECT_TRUE(tuples.insert(tuple).second)
                    << kernel.name << " repeats another candidate's knob tuple";
            }
            ++graphs;
            candidates += catalog.entries.size();
        }
    }
    EXPECT_GT(graphs, 400U);
    EXPECT_GT(candidates, 4000U);
}

TEST(TestGfx950ConvDgradCatalog, ShippedKernelsMatchThePacksSignaturesAndLoad)
{
    // One shipped kernel of each family through prepare(): requireSignatureMatch compares
    // the ABI this pack marshals (names, kinds, sizes) with what hkp_pack recorded, before
    // any device is needed; with a gfx950 device the code objects then load, the MFMA
    // instance's transpose and reorganize helpers too.
    const auto* manager = shippedManager();
    if(manager == nullptr)
    {
        GTEST_SKIP() << NOT_STAGED;
    }
    DgradShape strided;
    strided.h = strided.w = 15;
    strided.sH = strided.sW = 2;
    DgradShape depthwise;
    depthwise.c = depthwise.k = depthwise.groups = 96;
    DgradShape reorganizing;
    reorganizing.c = 48;
    reorganizing.k = 192;
    const auto device = currentDeviceProperties();
    const bool loadable = device.gcnArchName.rfind("gfx950", 0) == 0;
    const auto& handler = dispatchHandler(GFX950_CONV_DGRAD_AOT);
    for(const auto& row : std::vector<std::tuple<DgradShape, int64_t, std::string>>{
            {strided, 0, "none"},
            {depthwise, 1, "depthwise_dgrad"},
            {reorganizing, 1, "grouped_dgrad_mfma"}})
    {
        const auto& shape = std::get<0>(row);
        const auto family = std::get<1>(row);
        const auto& variant = std::get<2>(row);
        SCOPED_TRACE(variant);
        const GraphFixture fixture(buildGraph(graphSpec(shape)),
                                   loadable ? device : gfx950Properties());
        const auto catalog = manager->unsortedCatalog(fixture.context());
        const auto kernel = std::find_if(
            catalog.entries.begin(), catalog.entries.end(), [&](const auto& candidate) {
                return std::get<int64_t>(candidate.metadata.at("kernel_family")) == family
                       && std::get<std::string>(candidate.metadata.at("direct_variant")) == variant
                       && (variant != "grouped_dgrad_mfma"
                           || std::get<int64_t>(candidate.metadata.at("waves_k")) > 1);
            });
        ASSERT_NE(kernel, catalog.entries.end());
        try
        {
            const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
            ASSERT_TRUE(bound);
            EXPECT_NE(handler.prepare(fixture.context(), *bound, *kernel), nullptr);
            EXPECT_TRUE(loadable) << "loaded a gfx950 code object without a gfx950 device";
        }
        catch(const std::exception& error)
        {
            const std::string message = error.what();
            EXPECT_FALSE(loadable) << message;
            EXPECT_EQ(message.find("is packaged with arguments"), std::string::npos) << message;
            EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
            EXPECT_EQ(message.find("weight-transform helper"), std::string::npos) << message;
        }
    }
}
#endif

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
