// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "packs/PointwiseTestGraphs.hpp"

namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr PackSymbols GFX950_CONV_FWD{"hipkernel:Gfx950ConvFwd",
                                      "hipkernel.gfx950_conv_fwd.graph_match",
                                      "",
                                      "hipkernel.gfx950_conv_fwd.kernel_match",
                                      "hipkernel.gfx950_conv_fwd.score",
                                      "hipkernel.gfx950_conv_fwd.dispatch",
                                      "gfx950_conv_fwd.x.uid",
                                      "gfx950_conv_fwd.w.uid",
                                      "gfx950_conv_fwd.y.uid"};

struct TensorSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dtype = data_objects::DataType::HALF;
    bool isVirtual = false;
    bool passByValue = false;
    bool constant = false;
    bool omitDims = false;
    bool omitStrides = false;
    std::optional<int64_t> raggedOffset{};
    int64_t alignment = 16;
};

struct GraphSpec
{
    std::array<TensorSpec, 3> tensors{{{{2, 32, 14, 14}, {6272, 1, 448, 32}},
                                       {{32, 32, 3, 3}, {288, 1, 96, 32}},
                                       {{2, 32, 14, 14}, {6272, 1, 448, 32}}}};
    std::vector<int64_t> stride{1, 1};
    std::vector<int64_t> dilation{1, 1};
    std::vector<int64_t> prePadding{1, 1};
    std::vector<int64_t> postPadding{1, 1};
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    bool omitStride = false;
    bool omitTensor = false;
    bool overrideShapes = false;
    unsigned int nodeCount = 1;
    int64_t xUid = 1;
    int64_t wUid = 2;
    int64_t yUid = 3;
};

DeviceProperties gfx950Properties(std::string arch = "gfx950")
{
    // Deliberately by value: matcher tests must run on CPU-only and non-gfx950 hosts.
    DeviceProperties properties;
    properties.gcnArchName = std::move(arch);
    properties.warpSize = 64;
    return properties;
}

flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec = {})
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < spec.tensors.size(); ++i)
    {
        if(spec.omitTensor && i == 0)
        {
            continue;
        }
        const auto& tensor = spec.tensors[i];
        const auto dims = tensor.omitDims ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                          : builder.CreateVector(tensor.dims);
        const auto strides = tensor.omitStrides
                                 ? flatbuffers::Offset<flatbuffers::Vector<int64_t>>()
                                 : builder.CreateVector(tensor.strides);
        const auto value = tensor.constant
                               ? builder.CreateStruct(data_objects::Float16Value(1.0F)).Union()
                               : flatbuffers::Offset<void>();
        data_objects::TensorAttributesBuilder tensorBuilder(builder);
        tensorBuilder.add_uid(static_cast<int64_t>(i + 1));
        tensorBuilder.add_data_type(tensor.dtype);
        tensorBuilder.add_dims(dims);
        tensorBuilder.add_strides(strides);
        tensorBuilder.add_virtual_(tensor.isVirtual);
        tensorBuilder.add_is_runtime_pass_by_value(tensor.passByValue);
        tensorBuilder.add_alignment(tensor.alignment);
        if(tensor.constant)
        {
            tensorBuilder.add_value_type(data_objects::TensorValue::Float16Value);
            tensorBuilder.add_value(value);
        }
        if(tensor.raggedOffset)
        {
            tensorBuilder.add_ragged_offset_tensor_uid(*tensor.raggedOffset);
        }
        tensors.push_back(tensorBuilder.Finish());
    }
    const auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        spec.xUid,
        spec.wUid,
        spec.yUid,
        &spec.prePadding,
        &spec.postPadding,
        spec.omitStride ? nullptr : &spec.stride,
        &spec.dilation,
        spec.mode);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(unsigned int i = 0; i < spec.nodeCount; ++i)
    {
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "gfx950_conv_fwd_test",
                                           spec.computeType,
                                           data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                           attributes.Union()));
    }
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_tensors(tensorVector);
    graphBuilder.add_nodes(nodeVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapes);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

KernelDefinition smokeKernel(int64_t tileK = 64, const std::string& dtype = "fp16")
{
    KernelDefinition kernel;
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.metadata = {{"N", int64_t{2}},
                       {"C", int64_t{32}},
                       {"K", int64_t{32}},
                       {"Hi", int64_t{14}},
                       {"Wi", int64_t{14}},
                       {"Y", int64_t{3}},
                       {"X", int64_t{3}},
                       {"sH", int64_t{1}},
                       {"sW", int64_t{1}},
                       {"pH", int64_t{1}},
                       {"pW", int64_t{1}},
                       {"dH", int64_t{1}},
                       {"dW", int64_t{1}},
                       {"dtype", dtype},
                       {"tile_m", int64_t{64}},
                       {"tile_n", int64_t{64}},
                       {"tile_k", tileK},
                       {"warp_m", int64_t{2}},
                       {"warp_n", int64_t{2}},
                       {"warp_tile_m", int64_t{32}},
                       {"warp_tile_n", int64_t{32}},
                       {"warp_tile_k", int64_t{16}},
                       {"wave_size", int64_t{64}},
                       {"pipeline", std::string("mem")},
                       {"epilogue", std::string("cshuffle")},
                       {"layout", std::string("NHWC")},
                       {"groups", int64_t{1}}};
    return kernel;
}

TEST(TestGfx950ConvFwdGraphMatcher, AcceptsBothStorageTypesAndBindsActualTensorUids)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        for(auto& tensor : spec.tensors)
        {
            tensor.dtype = dtype;
        }
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
        ASSERT_TRUE(bound);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD.inputAToken), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD.inputBToken), 2);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX950_CONV_FWD.outputToken), 3);
    }
}

TEST(TestGfx950ConvFwdGraphMatcher, ArchGateUnderstandsFeatureSuffixes)
{
    const GraphFixture accepted(buildGraph(), gfx950Properties("gfx950:sramecc+:xnack-"));
    EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, accepted.context()));
    for(const auto* arch : {"gfx942", "gfx9500", "gfx95", "", "prefix-gfx950"})
    {
        SCOPED_TRACE(arch);
        const GraphFixture refused(buildGraph(), gfx950Properties(arch));
        EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, refused.context()));
    }
    auto properties = gfx950Properties();
    properties.warpSize = 32;
    const GraphFixture refused(buildGraph(), properties);
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, refused.context()));
}

TEST(TestGfx950ConvFwdGraphMatcher, AcceptsUnitExtentAxesWithArbitraryStrides)
{
    GraphSpec spec;
    spec.tensors[0].dims = {1, 1, 14, 14};
    spec.tensors[0].strides = {0, 196, 14, 1};
    spec.tensors[1].dims = {1, 1, 3, 3};
    spec.tensors[1].strides = {9, 9, 3, 1};
    spec.tensors[2].dims = {1, 1, 14, 14};
    spec.tensors[2].strides = {196, 196, 14, 1};
    const GraphFixture fixture(buildGraph(spec), gfx950Properties());
    EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
}

TEST(TestGfx950ConvFwdGraphMatcher, AllowsOneByOneStridedDilatedAndNonSquareProblems)
{
    const std::vector<std::function<void(GraphSpec&)>> cases{
        [](auto& s) {
            s.tensors[1].dims = {32, 32, 1, 1};
            s.tensors[1].strides = {32, 1, 32, 32};
        },
        [](auto& s) { s.stride = {2, 3}; },
        [](auto& s) { s.dilation = {2, 1}; },
        [](auto& s) {
            s.tensors[0].dims = {2, 32, 15, 19};
            s.tensors[0].strides = {9120, 1, 608, 32};
        }};
    for(const auto& change : cases)
    {
        GraphSpec spec;
        change(spec);
        // The frontend will infer the changed output dimensions before preparation.
        spec.tensors[2].omitDims = true;
        spec.tensors[2].omitStrides = true;
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        EXPECT_TRUE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
    }
}

struct RefusalCase
{
    const char* name;
    std::function<void(GraphSpec&)> change;
};

TEST(TestGfx950ConvFwdGraphMatcher, RefusesUnsupportedGraphsBeforeReadingTheirGeometry)
{
    const std::vector<RefusalCase> cases{
        {"fused graph", [](auto& s) { s.nodeCount = 2; }},
        {"empty graph", [](auto& s) { s.nodeCount = 0; }},
        {"runtime shape overrides", [](auto& s) { s.overrideShapes = true; }},
        {"convolution mode", [](auto& s) { s.mode = data_objects::ConvMode::CONVOLUTION; }},
        {"half accumulation", [](auto& s) { s.computeType = data_objects::DataType::HALF; }},
        {"unset accumulation", [](auto& s) { s.computeType = data_objects::DataType::UNSET; }},
        {"missing tensor", [](auto& s) { s.omitTensor = true; }},
        {"dangling tensor", [](auto& s) { s.xUid = 999; }},
        {"aliased output uid", [](auto& s) { s.yUid = s.xUid; }},
        {"groups", [](auto& s) { s.tensors[1].dims[1] = 16; }},
        {"NCHW input", [](auto& s) { s.tensors[0].strides = {6272, 196, 14, 1}; }},
        {"KCYX filter", [](auto& s) { s.tensors[1].strides = {288, 9, 3, 1}; }},
        {"nonpacked input", [](auto& s) { s.tensors[0].strides[0] += 32; }},
        {"negative stride", [](auto& s) { s.tensors[0].strides[2] = -448; }},
        {"rank three input", [](auto& s) { s.tensors[0].dims.pop_back(); }},
        {"rank three filter", [](auto& s) { s.tensors[1].dims.pop_back(); }},
        {"rank five input", [](auto& s) { s.tensors[0].dims.push_back(1); }},
        {"missing input dims", [](auto& s) { s.tensors[0].omitDims = true; }},
        {"missing filter strides", [](auto& s) { s.tensors[1].omitStrides = true; }},
        {"zero extent", [](auto& s) { s.tensors[0].dims[2] = 0; }},
        {"negative extent", [](auto& s) { s.tensors[1].dims[3] = -1; }},
        {"overflow extent",
         [](auto& s) { s.tensors[0].dims[0] = std::numeric_limits<int64_t>::max(); }},
        {"missing stride", [](auto& s) { s.omitStride = true; }},
        {"wrong stride rank", [](auto& s) { s.stride = {1}; }},
        {"zero stride", [](auto& s) { s.stride[0] = 0; }},
        {"negative dilation", [](auto& s) { s.dilation[1] = -1; }},
        {"wrong dilation rank", [](auto& s) { s.dilation = {1, 1, 1}; }},
        {"negative padding", [](auto& s) { s.prePadding[0] = s.postPadding[0] = -1; }},
        {"asymmetric padding", [](auto& s) { s.postPadding[0] = 2; }},
        {"empty padding", [](auto& s) { s.prePadding.clear(); }},
        {"empty output extent", [](auto& s) { s.dilation = {100, 100}; }}};
    for(const auto& test : cases)
    {
        SCOPED_TRACE(test.name);
        GraphSpec spec;
        test.change(spec);
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
    }
    const GraphFixture pointwise(buildPointwiseGraph(), gfx950Properties());
    EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, pointwise.context()));
}

TEST(TestGfx950ConvFwdGraphMatcher, RejectsUnsupportedStorageOnEveryOperand)
{
    const std::vector<std::function<void(TensorSpec&)>> changes{
        [](auto& t) { t.dtype = data_objects::DataType::FLOAT; },
        [](auto& t) { t.dtype = data_objects::DataType::BFLOAT16; },
        [](auto& t) { t.isVirtual = true; },
        [](auto& t) { t.passByValue = true; },
        [](auto& t) { t.constant = true; },
        [](auto& t) { t.raggedOffset = 7; },
        [](auto& t) { t.alignment = 8; }};
    for(size_t operand = 0; operand < 3; ++operand)
    {
        SCOPED_TRACE(operand);
        for(const auto& change : changes)
        {
            GraphSpec spec;
            change(spec.tensors[operand]);
            const GraphFixture fixture(buildGraph(spec), gfx950Properties());
            EXPECT_FALSE(matchesGraph(GFX950_CONV_FWD, fixture.context()));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, BothTileKVariantsMatchOnlyTheirStorageType)
{
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        for(auto& tensor : spec.tensors)
        {
            tensor.dtype = dtype;
        }
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        const std::string name = dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
        const std::string otherName = name == "fp16" ? "bf16" : "fp16";
        for(const auto tileK : {64, 128})
        {
            EXPECT_TRUE(
                matchesKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(tileK, name)));
            EXPECT_FALSE(
                matchesKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(tileK, otherName)));
        }
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, EveryBakedShapeAttributeMustMatch)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    for(const auto* field :
        {"N", "C", "K", "Hi", "Wi", "Y", "X", "sH", "sW", "pH", "pW", "dH", "dW"})
    {
        SCOPED_TRACE(field);
        auto kernel = smokeKernel();
        kernel.metadata[field] = kernel.getIntMetadata(field) + 1;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, MissingOrWronglyTypedMetadataDeclinesCleanly)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    const auto original = smokeKernel();
    for(const auto& field : original.metadata)
    {
        SCOPED_TRACE(field.first);
        auto kernel = original;
        kernel.metadata.erase(field.first);
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
        kernel.metadata[field.first] = std::vector<int64_t>{1};
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DeclinesUnreviewedTuningAndUnpackagedCode)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    for(const auto* field : {"tile_m",
                             "tile_n",
                             "tile_k",
                             "warp_m",
                             "warp_n",
                             "warp_tile_m",
                             "warp_tile_n",
                             "warp_tile_k",
                             "wave_size"})
    {
        SCOPED_TRACE(field);
        auto kernel = smokeKernel();
        kernel.metadata[field] = kernel.getIntMetadata(field) + 1;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
    auto kernel = smokeKernel();
    kernel.metadata["pipeline"] = std::string("wavelet");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["epilogue"] = std::string("relu");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["groups"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    kernel.metadata["layout"] = std::string("NCHW");
    EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    kernel = smokeKernel();
    for(const auto kind : {KernelSourceKind::EMBEDDED_SOURCE,
                           KernelSourceKind::ROCKE_BUILDER,
                           KernelSourceKind::HSACO_FILE})
    {
        kernel.source.kind = kind;
        EXPECT_FALSE(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel));
    }
}

TEST(TestGfx950ConvFwdKernelMatcher, DefaultEpilogueRequiresScalarOutputStores)
{
    // is_valid_spec_for_problem accepts K=31 (auto vec_c=1) and rejects K=32
    // (auto vec_c=8) for default epilogues, for both dtypes and tile_k choices.
    for(const auto dtype : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        const std::string name = dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
        SCOPED_TRACE(name);
        for(const auto k : {int64_t{31}, int64_t{32}})
        {
            SCOPED_TRACE(k);
            GraphSpec spec;
            for(auto& tensor : spec.tensors)
            {
                tensor.dtype = dtype;
            }
            spec.tensors[1].dims[0] = k;
            spec.tensors[2].dims[1] = k;
            spec.tensors[2].strides = {14 * 14 * k, 1, 14 * k, k};
            const GraphFixture fixture(buildGraph(spec), gfx950Properties());
            for(const auto tileK : {64, 128})
            {
                SCOPED_TRACE(tileK);
                auto kernel = smokeKernel(tileK, name);
                kernel.metadata["K"] = k;
                kernel.metadata["epilogue"] = std::string("default");
                EXPECT_EQ(matchesKernel(GFX950_CONV_FWD, fixture.context(), kernel), k == 31);
            }
        }
    }
}

TEST(TestGfx950ConvFwdScore, FallbackPrefersDispatcherDefault)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    EXPECT_GT(scoreKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(64)),
              scoreKernel(GFX950_CONV_FWD, fixture.context(), smokeKernel(128)));
}

TEST(TestGfx950ConvFwdDispatch, RegisteredHandlerNeedsNoWorkspaceAndChecksBindings)
{
    const GraphFixture fixture(buildGraph(), gfx950Properties());
    const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX950_CONV_FWD);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, smokeKernel()), 0U);
    EXPECT_THROW(handler.prepare(fixture.context(), {}, smokeKernel()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    auto wrongBound = *bound;
    wrongBound[std::string(GFX950_CONV_FWD.outputToken)] = int64_t{1};
    EXPECT_THROW(handler.prepare(fixture.context(), wrongBound, smokeKernel()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950ConvFwdDispatch, OutputValidationWaitsForPreparationAndPrecedesCodeLoading)
{
    const std::vector<std::function<void(TensorSpec&)>> cases{
        [](auto& t) {
            t.omitDims = true;
            t.omitStrides = true;
        },
        [](auto& t) { t.strides = {6272, 196, 14, 1}; },
        [](auto& t) { t.dims = {2, 32, 14}; },
        [](auto& t) { t.dims[2] = 0; },
        [](auto& t) { t.dims[0] = 1; },
        [](auto& t) {
            t.dims[1] = 16;
            t.strides = {3136, 1, 224, 16};
        }};
    for(const auto& change : cases)
    {
        GraphSpec spec;
        change(spec.tensors[2]);
        const GraphFixture fixture(buildGraph(spec), gfx950Properties());
        const auto bound = matchesGraph(GFX950_CONV_FWD, fixture.context());
        ASSERT_TRUE(bound);
        const auto& handler = dispatchHandler(GFX950_CONV_FWD);
        try
        {
            // An empty archive path must never be reached in any of these cases.
            handler.prepare(fixture.context(), *bound, smokeKernel());
            FAIL() << "Preparation accepted an invalid output tensor";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            EXPECT_NE(std::string(error.what()).find("packed NHWK output"), std::string::npos);
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
