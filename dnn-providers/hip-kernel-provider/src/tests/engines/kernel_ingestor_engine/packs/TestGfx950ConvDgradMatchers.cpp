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
#include "tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestCatalog.hpp"

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
namespace catalog = hip_kernel_provider::kernel_ingestor_engine::dgrad_test_catalog;
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

/// A kpack kernel carrying @p entry's complete metadata, as the descriptor loader delivers
/// it (every field present, omitted ones completed from the KMD defaults), in pack @p pack.
KernelDefinition kernelOf(const catalog::Entry& entry, uint8_t pack = 1)
{
    KernelDefinition kernel;
    const auto& id = entry.text("instance_id");
    kernel.kernelId = idOf(static_cast<uint32_t>(std::hash<std::string>{}(id + entry.dtype)), pack);
    kernel.packId = idOf(0, pack);
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.name
        = "gfx950_conv_dgrad_aot_" + entry.dtype + "_" + entry.text("direct_variant") + "_" + id;
    for(const auto& [name, value] : entry.fields)
    {
        if(const auto* integer = std::get_if<int64_t>(&value))
        {
            kernel.metadata[name] = *integer;
        }
        else
        {
            kernel.metadata[name] = std::get<std::string>(value);
        }
    }
    return kernel;
}

/// Runs the pack's KernelPrepareFn over @p kernels as the engine's shared state does at
/// load: each carries its prepared payload, and each helper is filed for the MFMA
/// pipelines of its own pack.
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

/// The @p dtype catalog as one pack, prepared, in catalog order.
std::vector<KernelDefinition> catalogPack(const std::string& dtype, uint8_t pack)
{
    std::vector<KernelDefinition> kernels;
    for(const auto& entry : catalog::catalog(dtype))
    {
        kernels.push_back(kernelOf(entry, pack));
    }
    prepare(kernels);
    return kernels;
}

const catalog::Entry& variantEntry(
    const std::string& variant,
    const std::function<bool(const catalog::Entry&)>& pick
    = [](const catalog::Entry&) { return true; },
    const std::string& dtype = "fp16")
{
    return catalog::firstEntry(dtype, [&](const catalog::Entry& e) {
        return e.text("direct_variant") == variant && pick(e);
    });
}

const catalog::Entry& implicitEntry()
{
    return variantEntry("none", [](const catalog::Entry& e) {
        return e.integer("vector_size_a") == 8 && e.integer("vector_size_b") == 8
               && e.integer("vector_size_c") == 8 && e.text("pipeline") == "mem"
               && e.integer("dy_halo") == 0 && e.integer("patch") == 0
               && e.integer("tap_outer_k") == 0 && e.integer("filter_h") == 0;
    });
}

const catalog::Entry& depthwiseEntry(int64_t filter, int64_t stride)
{
    return variantEntry("depthwise_dgrad", [=](const catalog::Entry& e) {
        return e.integer("filter_h") == filter && e.integer("stride_h") == stride;
    });
}

const catalog::Entry& fusedEntry(int64_t cpg)
{
    return variantEntry("grouped_dgrad_fused",
                        [=](const catalog::Entry& e) { return e.integer("cpg") == cpg; });
}

std::vector<const catalog::Entry*> helperEntries(const std::string& dtype = "fp16")
{
    std::vector<const catalog::Entry*> helpers;
    for(const auto& entry : catalog::catalog(dtype))
    {
        if(entry.isHelper())
        {
            helpers.push_back(&entry);
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

bool matchesShape(const DgradShape& shape,
                  const KernelDefinition& kernel,
                  data_objects::DataType dtype = data_objects::DataType::HALF)
{
    const auto fixture = fixtureOf(shape, dtype);
    const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
    EXPECT_TRUE(bound);
    return bound && matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernel, *bound);
}

DgradShape depthwiseShape(int64_t channels = 96)
{
    DgradShape shape;
    shape.c = shape.k = shape.groups = channels;
    return shape;
}

/// A grouped problem tall enough for every fused row: an xcd_tiles row needs eight image
/// row bands.
DgradShape groupedShape(int64_t cpg, int64_t groups)
{
    DgradShape shape;
    shape.n = 16;
    shape.h = shape.w = 28;
    shape.c = shape.k = cpg * groups;
    shape.groups = groups;
    return shape;
}

/// The dense c4k320 problem the MFMA rows serve.
DgradShape mfmaShape()
{
    DgradShape shape;
    shape.n = 1;
    shape.c = 4;
    shape.k = 320;
    shape.h = shape.w = 32;
    return shape;
}

TEST(TestGfx950ConvDgradKernelMatcher, EachFamilyServesItsOwnProblemsInItsOwnDtype)
{
    auto kernels = std::vector<KernelDefinition>{kernelOf(implicitEntry()),
                                                 kernelOf(depthwiseEntry(3, 1)),
                                                 kernelOf(fusedEntry(16)),
                                                 kernelOf(variantEntry("grouped_dgrad_mfma"))};
    for(const auto* helper : helperEntries())
    {
        kernels.push_back(kernelOf(*helper));
    }
    prepare(kernels);
    const auto& implicit = kernels[0];
    const auto& depthwise = kernels[1];
    const auto& fused = kernels[2];
    const auto& mfma = kernels[3];
    const DgradShape dense;
    const auto grouped = groupedShape(16, 8);
    EXPECT_TRUE(matchesShape(dense, implicit));
    EXPECT_FALSE(matchesShape(dense, depthwise));
    EXPECT_FALSE(matchesShape(dense, fused));
    EXPECT_FALSE(matchesShape(depthwiseShape(), implicit));
    EXPECT_TRUE(matchesShape(depthwiseShape(), depthwise));
    EXPECT_FALSE(matchesShape(depthwiseShape(), fused));
    EXPECT_TRUE(matchesShape(grouped, implicit));
    EXPECT_FALSE(matchesShape(grouped, depthwise));
    EXPECT_TRUE(matchesShape(grouped, fused));
    EXPECT_TRUE(matchesShape(mfmaShape(), mfma));
    EXPECT_FALSE(matchesShape(grouped, mfma));
    // The dtype is pinned: an fp16 kernel never serves a bf16 graph.
    EXPECT_FALSE(matchesShape(dense, implicit, data_objects::DataType::BFLOAT16));
    EXPECT_FALSE(matchesShape(depthwiseShape(), depthwise, data_objects::DataType::BFLOAT16));
    EXPECT_FALSE(matchesShape(grouped, fused, data_objects::DataType::BFLOAT16));
    // Without the bound token the matcher matches the graph itself, with the same answer.
    const auto fixture = fixtureOf(grouped);
    EXPECT_TRUE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), fused));
    // The helpers are never admitted, whatever the graph.
    for(size_t i = 4; i < kernels.size(); ++i)
    {
        EXPECT_FALSE(matchesShape(mfmaShape(), kernels[i])) << kernels[i].name;
        EXPECT_FALSE(matchesShape(dense, kernels[i])) << kernels[i].name;
    }
}

TEST(TestGfx950ConvDgradKernelMatcher, AnMfmaInstanceNeedsItsHelpersInItsOwnPack)
{
    const auto fixture = fixtureOf(mfmaShape());
    const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    // The c4k320 rows split K over two waves: they read the reorganized weights, so the
    // pipeline runs the transpose and the reorganize of its fold_k32.
    const auto& coalesced = variantEntry(
        "grouped_dgrad_mfma", [](const catalog::Entry& e) { return e.integer("waves_k") > 1; });
    const auto fold = coalesced.integer("fold_k32");
    const auto helpersOf
        = [&](uint8_t pack, bool transpose, bool reorganize, int64_t reorganizeFold) {
              std::vector<KernelDefinition> kernels{kernelOf(coalesced, pack)};
              for(const auto* helper : helperEntries())
              {
                  const auto& variant = helper->text("direct_variant");
                  if((variant == "transpose" && transpose)
                     || (variant == "reorganize" && reorganize
                         && helper->integer("fold_k32") == reorganizeFold))
                  {
                      kernels.push_back(kernelOf(*helper, pack));
                  }
                  // A bf16 helper in the same pack is not this fp16 instance's.
                  for(const auto* other : helperEntries("bf16"))
                  {
                      if(other->text("direct_variant") == variant)
                      {
                          kernels.push_back(kernelOf(*other, pack));
                      }
                  }
              }
              prepare(kernels);
              return kernels;
          };
    const auto lonely = helpersOf(7, false, false, fold);
    EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), lonely[0], *bound));
    const auto half = helpersOf(8, true, false, fold);
    EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), half[0], *bound));
    const auto wrongFold = helpersOf(9, true, true, 1 - fold);
    EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), wrongFold[0], *bound));
    const auto full = helpersOf(10, true, true, fold);
    EXPECT_TRUE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), full[0], *bound));
    // The runtime-channel pipeline takes the runtime-channel transforms.
    const auto& rtc = variantEntry("dgrad_mfma_rtc");
    std::vector<KernelDefinition> rtcPack{kernelOf(rtc, 11)};
    for(const auto* helper : helperEntries())
    {
        rtcPack.push_back(kernelOf(*helper, 11));
    }
    prepare(rtcPack);
    DgradShape rtcShape = mfmaShape();
    rtcShape.c = 12;
    EXPECT_TRUE(matchesShape(rtcShape, rtcPack[0]));
    std::vector<KernelDefinition> plainHelpers{kernelOf(rtc, 12)};
    for(const auto* helper : helperEntries())
    {
        if(helper->text("direct_variant").find("_rtc") == std::string::npos)
        {
            plainHelpers.push_back(kernelOf(*helper, 12));
        }
    }
    prepare(plainHelpers);
    EXPECT_FALSE(matchesShape(rtcShape, plainHelpers[0]));
}

/// Every served-matrix row a graph can state, in its own dtype: the matcher admits exactly
/// the adapter's instances, scores exactly those, and the other dtype's catalog admits none.
TEST(TestGfx950ConvDgradKernelMatcher, ServesExactlyWhatTheAdapterServesOverTheCatalog)
{
    const std::map<std::string, std::vector<KernelDefinition>> packs{
        {"fp16", catalogPack("fp16", 1)}, {"bf16", catalogPack("bf16", 2)}};
    size_t rows = 0;
    size_t served = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto problem = catalog::toProblem(row.problem);
        if(!dgrad::deriveGeometry(problem))
        {
            // No graph states this problem; GraphMatcher tests cover its refusal.
            EXPECT_EQ(std::string_view(row.served).find('1'), std::string_view::npos);
            continue;
        }
        // Graphs past 2^31 elements take seconds to build here; the geometry test covers them.
        if(dgrad::deriveGeometry(problem)->tensorBytes[2] > dgrad::I32_MAX)
        {
            continue;
        }
        ++rows;
        const std::string dtypeName = row.dtype;
        const auto dtype
            = dtypeName == "fp16" ? data_objects::DataType::HALF : data_objects::DataType::BFLOAT16;
        const auto shape = shapeOf(row.problem);
        const auto fixture = fixtureOf(shape, dtype);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        const auto& own = packs.at(dtypeName);
        const auto& other = packs.at(dtypeName == "fp16" ? "bf16" : "fp16");
        for(size_t i = 0; i < own.size(); ++i)
        {
            const bool expected = row.served[i] == '1';
            const bool actual
                = matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
            EXPECT_EQ(actual, expected)
                << own[i].name << " on N" << shape.n << " C" << shape.c << " K" << shape.k << " G"
                << shape.groups << " " << shape.h << "x" << shape.w << " f" << shape.y << "x"
                << shape.x << " s" << shape.sH << "x" << shape.sW << " p" << shape.pH << "x"
                << shape.pW << " d" << shape.dH << "x" << shape.dW;
            // A served kernel scores, a refused one ranks last.
            const auto score
                = scoreKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), own[i], *bound);
            EXPECT_EQ(score >= 0.0, expected) << own[i].name << " score " << score;
            served += actual ? 1 : 0;
        }
        for(const auto& kernel : other)
        {
            EXPECT_FALSE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernel, *bound))
                << kernel.name;
        }
    }
    EXPECT_GT(rows, 300U);
    EXPECT_GT(served, 20000U);
}

TEST(TestGfx950ConvDgradKernelMatcher, RefusesOutOfContractOrIncompleteMetadata)
{
    const std::vector<std::pair<DgradShape, const catalog::Entry*>> cases{
        {DgradShape{}, &implicitEntry()},
        {depthwiseShape(), &depthwiseEntry(3, 1)},
        {groupedShape(16, 8), &fusedEntry(16)},
        {mfmaShape(), &variantEntry("grouped_dgrad_mfma")}};
    for(const auto& [shape, entry] : cases)
    {
        SCOPED_TRACE(entry->text("direct_variant"));
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        std::vector<KernelDefinition> base{kernelOf(*entry, 20)};
        for(const auto* helper : helperEntries())
        {
            base.push_back(kernelOf(*helper, 20));
        }
        prepare(base);
        ASSERT_TRUE(matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), base[0], *bound));
        const bool direct = entry->integer("kernel_family") == 1;
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
            {"halo_max_wo", int64_t{17}},
            {"fold_k32", int64_t{2}},
            {"persistent_grid", int64_t{2}},
            {"stage_rows", int64_t{2}}};
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

TEST(TestGfx950ConvDgradScore, DirectPipelinesThenMfmaThenImplicitGemmAndRefusalsLast)
{
    auto kernels = catalogPack("fp16", 30);
    const auto& entries = catalog::catalog("fp16");
    // Per family (0 implicit GEMM, 1 a single-kernel direct pipeline, 2 an MFMA pipeline),
    // the (min, max) score of the kernels serving the shape.
    const auto scores = [&](const DgradShape& shape) {
        const auto fixture = fixtureOf(shape);
        const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
        EXPECT_TRUE(bound);
        std::map<int64_t, std::pair<double, double>> range;
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const auto score
                = scoreKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernels[i], *bound);
            if(!matchesKernel(GFX950_CONV_DGRAD_AOT, fixture.context(), kernels[i], *bound))
            {
                EXPECT_EQ(score, -1.0) << kernels[i].name;
                continue;
            }
            const auto& variant = entries[i].text("direct_variant");
            const int64_t family = variant == "none"                           ? 0
                                   : variant.find("mfma") != std::string::npos ? 2
                                                                               : 1;
            auto [it, inserted] = range.emplace(family, std::make_pair(score, score));
            it->second.first = std::min(it->second.first, score);
            it->second.second = std::max(it->second.second, score);
        }
        return range;
    };
    const auto depthwise = scores(depthwiseShape());
    ASSERT_TRUE(depthwise.count(1));
    EXPECT_GE(depthwise.at(1).first, 3.0);
    const auto grouped = scores(groupedShape(16, 8));
    ASSERT_TRUE(grouped.count(0));
    ASSERT_TRUE(grouped.count(1));
    EXPECT_GT(grouped.at(1).first, grouped.at(0).second);
    EXPECT_GE(grouped.at(0).first, 0.0);
    EXPECT_LT(grouped.at(0).second, 1.01);
    const auto mfma = scores(mfmaShape());
    ASSERT_TRUE(mfma.count(2));
    EXPECT_GE(mfma.at(2).first, 2.0);
    EXPECT_LT(mfma.at(2).second, 3.0);
    EXPECT_GT(mfma.at(2).first, mfma.at(0).second);
}

TEST(TestGfx950ConvDgradDispatch, WorkspaceFollowsThePlanAndPrepareReachesCodeLoading)
{
    auto kernels = std::vector<KernelDefinition>{
        kernelOf(implicitEntry(), 40),
        kernelOf(depthwiseEntry(3, 2), 40),
        kernelOf(variantEntry("grouped_dgrad_mfma",
                              [](const catalog::Entry& e) { return e.integer("waves_k") > 1; }),
                 40),
        kernelOf(fusedEntry(16), 40)};
    for(const auto* helper : helperEntries())
    {
        kernels.push_back(kernelOf(*helper, 40));
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

    // Implicit GEMM, strided: no caller workspace; the prepared plan owns the record tables
    // ((22 + 8) * 4 bytes per sub-GEMM).
    DgradShape strided;
    strided.h = strided.w = 15;
    strided.sH = strided.sW = 2;
    const auto stridedFixture = fixtureOf(strided);
    const auto stridedBound = matchesGraph(GFX950_CONV_DGRAD_AOT, stridedFixture.context());
    ASSERT_TRUE(stridedBound);
    EXPECT_EQ(handler.workspaceBytes(stridedFixture.context(), *stridedBound, kernels[0]), 0U);
    reachesLoading(stridedFixture, *stridedBound, kernels[0]);
    // Refused before any code loads.
    EXPECT_EQ(handler.workspaceBytes(stridedFixture.context(), *stridedBound, kernels[1]), 0U);
    EXPECT_THROW(handler.prepare(stridedFixture.context(), *stridedBound, kernels[1]),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(handler.prepare(stridedFixture.context(), *stridedBound, kernels[4]),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    // Depthwise and fused: no workspace.
    DgradShape depthwise = depthwiseShape(64);
    depthwise.h = 29;
    depthwise.w = 27;
    depthwise.sH = depthwise.sW = 2;
    const auto depthwiseFixture = fixtureOf(depthwise);
    const auto depthwiseBound = matchesGraph(GFX950_CONV_DGRAD_AOT, depthwiseFixture.context());
    ASSERT_TRUE(depthwiseBound);
    EXPECT_EQ(handler.workspaceBytes(depthwiseFixture.context(), *depthwiseBound, kernels[1]), 0U);
    reachesLoading(depthwiseFixture, *depthwiseBound, kernels[1]);
    const auto fusedFixture = fixtureOf(groupedShape(16, 8));
    const auto fusedBound = matchesGraph(GFX950_CONV_DGRAD_AOT, fusedFixture.context());
    ASSERT_TRUE(fusedBound);
    EXPECT_EQ(handler.workspaceBytes(fusedFixture.context(), *fusedBound, kernels[3]), 0U);
    reachesLoading(fusedFixture, *fusedBound, kernels[3]);

    // MFMA with the reorganize step: ws_t, then ws_coa at ws_t rounded up to 256.
    const auto mfmaFixture = fixtureOf(mfmaShape());
    const auto mfmaBound = matchesGraph(GFX950_CONV_DGRAD_AOT, mfmaFixture.context());
    ASSERT_TRUE(mfmaBound);
    EXPECT_EQ(handler.workspaceBytes(mfmaFixture.context(), *mfmaBound, kernels[2]),
              23040U + 92160U);
    reachesLoading(mfmaFixture, *mfmaBound, kernels[2]);
    // The same instance on a grouped graph is refused (the reorganize helper's group bug).
    auto grouped = mfmaShape();
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
    ASSERT_EQ(knobs.size(), 49U);

    // Each served-matrix problem a graph can state, in its own dtype: the candidate set is
    // what the real matchers, the shipped metadata (with its KMD defaults) and the helper
    // lookup admit, and its size is the adapter's count.
    size_t graphs = 0;
    size_t candidates = 0;
    for(const auto& row : vectors::servedMatrix())
    {
        const auto shape = shapeOf(row.problem);
        const auto geometry = dgrad::deriveGeometry(catalog::toProblem(row.problem));
        if(!geometry || geometry->tensorBytes[2] > dgrad::I32_MAX)
        {
            continue;
        }
        const auto expected = static_cast<size_t>(
            std::count(row.served, row.served + std::string_view(row.served).size(), '1'));
        const std::string dtypeName = row.dtype;
        const auto dtype
            = dtypeName == "fp16" ? data_objects::DataType::HALF : data_objects::DataType::BFLOAT16;
        const auto fixture = fixtureOf(shape, dtype);
        const auto shipped = manager->unsortedCatalog(fixture.context());
        EXPECT_EQ(shipped.entries.size(), expected)
            << "N" << shape.n << " C" << shape.c << " K" << shape.k << " G" << shape.groups << " "
            << shape.h << "x" << shape.w << " f" << shape.y << "x" << shape.x;
        std::set<std::vector<int64_t>> tuples;
        for(const auto& kernel : shipped.entries)
        {
            EXPECT_EQ(std::get<std::string>(kernel->metadata.at("dtype")), dtypeName);
            EXPECT_NE(std::get<int64_t>(kernel->metadata.at("kernel_family")), 2) << kernel->name;
            std::vector<int64_t> tuple;
            for(const auto& knob : knobs)
            {
                const auto ordinal = manager->knobOrdinal(knob, kernel->metadata.at(knob));
                ASSERT_TRUE(ordinal) << knob;
                tuple.push_back(*ordinal);
            }
            EXPECT_TRUE(tuples.insert(tuple).second)
                << kernel->name << " repeats another candidate's knob tuple";
        }
        ++graphs;
        candidates += shipped.entries.size();
    }
    EXPECT_GT(graphs, 300U);
    EXPECT_GT(candidates, 20000U);
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
    DgradShape windowed = depthwiseShape(96);
    const auto device = currentDeviceProperties();
    const bool loadable = device.gcnArchName.rfind("gfx950", 0) == 0;
    const auto& handler = dispatchHandler(GFX950_CONV_DGRAD_AOT);
    for(const auto& row : std::vector<std::tuple<DgradShape, int64_t, std::string>>{
            {strided, 0, "none"},
            {depthwiseShape(96), 1, "depthwise_dgrad"},
            {windowed, 1, "depthwise_dgrad_win"},
            {groupedShape(16, 8), 1, "grouped_dgrad_fused"},
            {groupedShape(4, 64), 1, "grouped_dgrad_4c"},
            {mfmaShape(), 1, "grouped_dgrad_mfma"}})
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
                return std::get<int64_t>(candidate->metadata.at("kernel_family")) == family
                       && std::get<std::string>(candidate->metadata.at("direct_variant"))
                              == variant;
            });
        ASSERT_NE(kernel, catalog.entries.end());
        try
        {
            const auto bound = matchesGraph(GFX950_CONV_DGRAD_AOT, fixture.context());
            ASSERT_TRUE(bound);
            EXPECT_NE(handler.prepare(fixture.context(), *bound, **kernel), nullptr);
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
