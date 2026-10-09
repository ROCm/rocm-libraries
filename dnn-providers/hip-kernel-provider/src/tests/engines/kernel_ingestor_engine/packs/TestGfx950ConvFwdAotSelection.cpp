// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotSelection.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950ConvFwdAotSelection.cpp
 * @brief hipkernel:Gfx950ConvFwdAot's runtime selection: of the instances that serve a
 *        graph (the kernel matcher), the ones rocKE's own runtime selection runs (the
 *        candidate filter registered under the kernel matcher's symbol).
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
namespace conv = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* AOT_GRAPH_MATCHER = "hipkernel.gfx950_conv_fwd_aot.graph_match";
constexpr const char* AOT_KERNEL_MATCHER = "hipkernel.gfx950_conv_fwd_aot.kernel_match";

/// One forward identity of rocKE's AOT sweep (gfx950, fp16, grouped), as
/// make_conv_aot_config.py packages it. The fields the kernel matcher and the candidate
/// filter read, in this order: epilogue, vector_size_a/b/c, tile_m/n/k, warp_m/n,
/// warp_tile_m/n/k, pipeline, unroll_k, async_dma, async_chunk_a/b.
struct AotSample
{
    const char* epilogue;
    std::array<int64_t, 11> values;
    const char* pipeline;
    std::array<int64_t, 4> loop;
};

/// One instance of each (epilogue, width pair, K loop) the sweep builds, at a 64x64 tile,
/// then four with the extreme M tiles of their width pairs. test_make_conv_aot_config.py
/// reads this table: keep one row per line.
// clang-format off
const std::vector<AotSample>& aotSample()
{
    static const std::vector<AotSample> s_sample{
        {"cshuffle", {8, 8, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 8, 16, 64, 32, 1, 1, 16, 16, 32}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 4, 256, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 2, 256, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {4, 4, 1, 256, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
    };
    return s_sample;
}

/// A graph, as (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups), and the rows of
/// aotSample() selected for it, in table order. gfx950_conv_fwd_aot_runtime_candidates
/// computes them, and test_make_conv_aot_config.py checks this table against it: keep
/// one row per line.
struct SelectionCase
{
    std::array<int64_t, 14> problem;
    std::vector<size_t> selected;
};

const std::vector<SelectionCase>& selectionCases()
{
    static const std::vector<SelectionCase> s_cases{
        {{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 16, 40}},
        {{2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1}, {12}},
        {{2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {3, 19, 32, 36}},
        {{1, 3, 5, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {15, 35}},
        {{2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {8}},
        {{2, 4, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {5, 41}},
        {{2, 12, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {15, 35}},
        {{1, 9, 48, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 3}, {12}},
        {{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {15, 35}},
        {{2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3}, {0, 16, 40}},
        {{2, 16, 24, 17, 19, 3, 3, 2, 2, 2, 2, 2, 2, 1}, {0, 16, 40}},
        {{32, 8, 8, 192, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {0, 16}},
        {{36, 8, 8, 500, 500, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {41}},
    };
    return s_cases;
}
// clang-format on

KernelDefinition sampleKernel(const AotSample& sample)
{
    const auto& v = sample.values;
    KernelDefinition kernel;
    kernel.name = "aot_instance";
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.metadata = {{"dtype", std::string("fp16")},
                       {"layout", std::string("NHWC")},
                       {"vector_size_a", v[0]},
                       {"vector_size_b", v[1]},
                       {"vector_size_c", v[2]},
                       {"tile_m", v[3]},
                       {"tile_n", v[4]},
                       {"tile_k", v[5]},
                       {"warp_m", v[6]},
                       {"warp_n", v[7]},
                       {"warp_tile_m", v[8]},
                       {"warp_tile_n", v[9]},
                       {"warp_tile_k", v[10]},
                       {"wave_size", int64_t{64}},
                       {"pipeline", std::string(sample.pipeline)},
                       {"epilogue", std::string(sample.epilogue)},
                       {"unroll_k", sample.loop[0]},
                       {"async_dma", sample.loop[1]},
                       {"async_chunk_a", sample.loop[2]},
                       {"async_chunk_b", sample.loop[3]},
                       {"grouped", int64_t{1}},
                       {"instance_id", std::string(40, 'a')},
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
                       {"waves_per_eu", int64_t{0}}};
    return kernel;
}

/// A direct std depthwise instance compiled for a 3x3, stride 1, "same"-padded filter.
KernelDefinition directSampleKernel()
{
    auto kernel
        = sampleKernel({"direct", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "direct", {0, 0, 0, 0}});
    kernel.metadata["grouped"] = int64_t{0};
    kernel.metadata["kernel_family"] = int64_t{1};
    kernel.metadata["direct_variant"] = std::string("std");
    kernel.metadata["filter_h"] = kernel.metadata["filter_w"] = int64_t{3};
    kernel.metadata["stride_h"] = kernel.metadata["stride_w"] = int64_t{1};
    kernel.metadata["pad_h"] = kernel.metadata["pad_w"] = int64_t{1};
    kernel.metadata["dilation_h"] = kernel.metadata["dilation_w"] = int64_t{1};
    kernel.metadata["block_w"] = int64_t{4};
    kernel.metadata["block_waves"] = int64_t{1};
    return kernel;
}

/// A packed NHWC fp16 forward graph for @p p, in SelectionCase's field order.
flatbuffers::FlatBufferBuilder convGraph(const std::array<int64_t, 14>& p)
{
    const auto [n, c, k, hi, wi, y, x, sH, sW, pH, pW, dH, dW, groups] = p;
    const auto cpg = c / groups;
    const auto ho = (hi + 2 * pH - ((y - 1) * dH + 1)) / sH + 1;
    const auto wo = (wi + 2 * pW - ((x - 1) * dW + 1)) / sW + 1;
    const std::array<std::vector<int64_t>, 3> dims{
        std::vector<int64_t>{n, c, hi, wi}, {k, cpg, y, x}, {n, k, ho, wo}};
    const std::array<std::vector<int64_t>, 3> strides{
        std::vector<int64_t>{hi * wi * c, 1, wi * c, c},
        {y * x * cpg, 1, x * cpg, cpg},
        {ho * wo * k, 1, wo * k, k}};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        const auto dimsOffset = builder.CreateVector(dims[i]);
        const auto stridesOffset = builder.CreateVector(strides[i]);
        data_objects::TensorAttributesBuilder tensor(builder);
        tensor.add_uid(static_cast<int64_t>(i + 1));
        tensor.add_data_type(data_objects::DataType::HALF);
        tensor.add_dims(dimsOffset);
        tensor.add_strides(stridesOffset);
        tensor.add_alignment(16);
        tensors.push_back(tensor.Finish());
    }
    const std::vector<int64_t> stride{sH, sW};
    const std::vector<int64_t> dilation{dH, dW};
    const std::vector<int64_t> padding{pH, pW};
    const auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        1,
        2,
        3,
        &padding,
        &padding,
        &stride,
        &dilation,
        data_objects::ConvMode::CROSS_CORRELATION);
    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "gfx950_conv_fwd_aot_selection_test",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       attributes.Union())};
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graph(builder);
    graph.add_tensors(tensorVector);
    graph.add_nodes(nodeVector);
    builder.Finish(graph.Finish());
    return builder;
}

DeviceProperties gfx950()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    return properties;
}

/// The engine's pipeline for one graph, as KernelIngestorStateManager runs it: the graph
/// match, the kernel matcher on every kernel, then the candidate filter registered under
/// the kernel matcher's symbol over what it admitted. Returns the indices of
/// @p kernels that serve the graph, and through @p selected the ones that are selected.
std::vector<size_t> serveAndSelect(const MatchContext& context,
                                   const std::vector<KernelDefinition>& kernels,
                                   std::vector<size_t>& selected)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto bound = GraphMatchRegistry::resolve(AOT_GRAPH_MATCHER)(context);
    EXPECT_TRUE(bound.has_value());
    const auto matcher = KernelMatcherRegistry::resolve(AOT_KERNEL_MATCHER);
    const auto filter = CandidateFilterRegistry::resolve(AOT_KERNEL_MATCHER);

    std::vector<size_t> serving;
    std::vector<const KernelDefinition*> candidates;
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        if(bound && matcher(context, *bound, kernels[i]))
        {
            serving.push_back(i);
            candidates.push_back(&kernels[i]);
        }
    }
    const auto keep = filter(context, bound.value_or(BoundTokens{}), candidates);
    EXPECT_EQ(keep.size(), candidates.size());
    selected.clear();
    for(size_t i = 0; i < keep.size() && i < serving.size(); ++i)
    {
        if(keep[i])
        {
            selected.push_back(serving[i]);
        }
    }
    return serving;
}

TEST(TestGfx950ConvFwdAotSelection, SelectsWhatRockesRuntimeSelectionRunsOverTheSample)
{
    std::vector<KernelDefinition> kernels;
    for(const auto& sample : aotSample())
    {
        kernels.push_back(sampleKernel(sample));
    }
    for(const auto& [problem, expected] : selectionCases())
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture fixture(convGraph(problem), gfx950());
        std::vector<size_t> selected;
        const auto serving = serveAndSelect(fixture.context(), kernels, selected);
        EXPECT_EQ(selected, expected);
        // Selection only narrows what serves.
        EXPECT_LE(selected.size(), serving.size());
        EXPECT_FALSE(selected.empty());
    }
}

TEST(TestGfx950ConvFwdAotSelection, KeepsEveryServingDirectInstance)
{
    // Depthwise 3x3: the direct instance and the scalar implicit-GEMM instances serve.
    std::vector<KernelDefinition> kernels{directSampleKernel()};
    for(const auto& sample : aotSample())
    {
        kernels.push_back(sampleKernel(sample));
    }
    const GraphFixture fixture(convGraph({2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}),
                               gfx950());
    std::vector<size_t> selected;
    serveAndSelect(fixture.context(), kernels, selected);
    // Row r of aotSample() is kernel r + 1 here.
    EXPECT_EQ(selected, (std::vector<size_t>{0, 16, 36}));
}

std::shared_ptr<const void> foreignPayload(const KernelDefinition& /*kernel*/)
{
    return std::make_shared<const int>(0);
}

TEST(TestGfx950ConvFwdAotSelection, PreparedKernelsAndTheBoundProblemDecideAsTheGraphAndMetadataDo)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto graphMatch = GraphMatchRegistry::resolve(AOT_GRAPH_MATCHER);
    const auto matcher = KernelMatcherRegistry::resolve(AOT_KERNEL_MATCHER);
    const auto filter = CandidateFilterRegistry::resolve(AOT_KERNEL_MATCHER);
    const auto score = ScoreRegistry::resolve("hipkernel.gfx950_conv_fwd_aot.score");
    const auto prepare = KernelPrepareRegistry::resolve(AOT_KERNEL_MATCHER);

    // The sample, a direct instance, a bf16 instance, and metadata no launch contract
    // describes.
    std::vector<KernelDefinition> plain{directSampleKernel()};
    for(const auto& sample : aotSample())
    {
        plain.push_back(sampleKernel(sample));
    }
    plain.push_back(sampleKernel(aotSample().front()));
    plain.back().metadata["dtype"] = std::string("bf16");
    plain.push_back(sampleKernel(aotSample().front()));
    plain.back().metadata["layout"] = std::string("NCHW");

    // What the engine's shared state carries, and a definition whose payload some other
    // preparer made, which the pack must not trust.
    auto prepared = plain;
    auto foreign = plain;
    for(size_t i = 0; i < plain.size(); ++i)
    {
        prepared[i].prepared = prepare(prepared[i]);
        prepared[i].preparedBy = prepare;
        ASSERT_NE(prepared[i].prepared, nullptr);
        foreign[i].prepared = foreignPayload(foreign[i]);
        foreign[i].preparedBy = &foreignPayload;
    }

    auto problems = selectionCases();
    problems.push_back({{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {}});
    for(const auto& [problem, expected] : problems)
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture fixture(convGraph(problem), gfx950());
        const auto& context = fixture.context();
        const auto bound = graphMatch(context);
        ASSERT_TRUE(bound.has_value());

        std::vector<const KernelDefinition*> plainServing;
        std::vector<const KernelDefinition*> preparedServing;
        for(size_t i = 0; i < plain.size(); ++i)
        {
            // An empty bound makes the matcher read the graph itself.
            const bool served = matcher(context, BoundTokens{}, plain[i]);
            EXPECT_EQ(matcher(context, *bound, plain[i]), served) << i;
            EXPECT_EQ(matcher(context, *bound, prepared[i]), served) << i;
            EXPECT_EQ(matcher(context, *bound, foreign[i]), served) << i;
            EXPECT_EQ(score(context, *bound, prepared[i]), score(context, BoundTokens{}, plain[i]))
                << i;
            if(served)
            {
                plainServing.push_back(&plain[i]);
                preparedServing.push_back(&prepared[i]);
            }
        }
        EXPECT_FALSE(plainServing.empty());
        EXPECT_EQ(filter(context, *bound, preparedServing),
                  filter(context, BoundTokens{}, plainServing));
    }
}

TEST(TestGfx950ConvFwdAotSelection, PolicyKeepsTheWidestCombinationPerEpilogue)
{
    // Every entry already serves the problem. The default epilogue goes as soon as any
    // instance stores wider than 1.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 4, 4, 4},
                                           {"cshuffle", 8, 8, 2},
                                           {"cshuffle", 8, 8, 4},
                                           {"default", 8, 8, 1},
                                           {"cshuffle", 4, 4, 8}}),
              (std::vector<bool>{false, false, true, false, false}));
    // Widest is the greatest a + b + c, then a, b and c in turn.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 8, 8, 1}, {"cshuffle", 4, 4, 8}}),
              (std::vector<bool>{true, false}));
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 2, 2, 8}, {"cshuffle", 4, 4, 4}}),
              (std::vector<bool>{false, true}));
}

TEST(TestGfx950ConvFwdAotSelection, PolicyKeepsBothEpiloguesWhereNothingStoresWider)
{
    // An odd K / groups: every store is scalar, so each epilogue keeps its widest load.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 8, 8, 1},
                                           {"cshuffle", 4, 4, 1},
                                           {"default", 4, 4, 1},
                                           {"default", 8, 8, 1},
                                           {"default", 8, 8, 1}}),
              (std::vector<bool>{true, false, false, true, true}));
    EXPECT_TRUE(conv::widestVectorSelection({}).empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
