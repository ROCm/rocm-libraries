// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotSelection.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950ConvFwdAotSelection.cpp
 * @brief hipkernel:Gfx950ConvFwdAot's runtime selection: of the instances that serve a
 *        graph (the kernel matcher), the ones that run: the graph's widest vector
 *        widths, or every serving instance when none bakes them (the candidate filter
 *        registered under the kernel matcher's symbol). Then hipkernel:Gfx942ConvFwdAot,
 *        the same native hooks over the gfx942 target: its own tokens and device, the
 *        group-merged depthwise instances, and rocKE's per-kind selection.
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
        {{36, 8, 8, 500, 500, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {41, 42, 43}},
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

/// A 64-wide-wave device of @p arch.
DeviceProperties device(const char* arch)
{
    DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = 64;
    return properties;
}

DeviceProperties gfx950()
{
    return device("gfx950");
}

/// The engine's pipeline for one graph, as KernelIngestorStateManager runs it: the graph
/// match, the kernel matcher on every kernel, then the candidate filter registered under
/// the kernel matcher's symbol over what it admitted. Returns the indices of
/// @p kernels that serve the graph, and through @p selected the ones that are selected.
/// @p graphMatchSymbol and @p kernelMatchSymbol name the engine (hipkernel:Gfx950ConvFwdAot
/// by default).
std::vector<size_t> serveAndSelect(const MatchContext& context,
                                   const std::vector<KernelDefinition>& kernels,
                                   std::vector<size_t>& selected,
                                   const char* graphMatchSymbol = AOT_GRAPH_MATCHER,
                                   const char* kernelMatchSymbol = AOT_KERNEL_MATCHER)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto bound = GraphMatchRegistry::resolve(graphMatchSymbol)(context);
    EXPECT_TRUE(bound.has_value());
    const auto matcher = KernelMatcherRegistry::resolve(kernelMatchSymbol);
    const auto filter = CandidateFilterRegistry::resolve(kernelMatchSymbol);

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

TEST(TestGfx950ConvFwdAotSelection, SelectsWhatTheRuntimeSelectionRunsOverTheSample)
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

TEST(TestGfx950ConvFwdAotSelection, WidestVectorWidthIsTheLargestDividingPowerOfTwo)
{
    EXPECT_EQ(conv::widestVectorWidth(64), 8);
    EXPECT_EQ(conv::widestVectorWidth(12), 4);
    EXPECT_EQ(conv::widestVectorWidth(6), 2);
    EXPECT_EQ(conv::widestVectorWidth(5), 1);
}

TEST(TestGfx950ConvFwdAotSelection, PolicyKeepsTheProblemsWidestCombination)
{
    // Every entry already serves the problem. C/groups 16 and K/groups 12 admit 8/8/4,
    // and only the instances baking it run; the default epilogue stores scalars.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 4, 4, 4},
                                           {"cshuffle", 8, 8, 2},
                                           {"cshuffle", 8, 8, 4},
                                           {"default", 8, 8, 1},
                                           {"cshuffle", 8, 8, 4}},
                                          16,
                                          12),
              (std::vector<bool>{false, false, true, false, true}));
    // An odd K / groups: every store is scalar, and both epilogues bake 8/8/1.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 8, 8, 1},
                                           {"cshuffle", 4, 4, 1},
                                           {"default", 4, 4, 1},
                                           {"default", 8, 8, 1}},
                                          32,
                                          5),
              (std::vector<bool>{true, false, false, true}));
}

TEST(TestGfx950ConvFwdAotSelection, PolicyRunsEveryServingInstanceWithoutTheWidest)
{
    // C/groups 16 and K/groups 6 admit 8/8/2, which no instance bakes: every serving
    // instance runs, of either epilogue.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 8, 8, 1},
                                           {"cshuffle", 2, 2, 2},
                                           {"default", 8, 8, 1},
                                           {"cshuffle", 1, 1, 1}},
                                          16,
                                          6),
              (std::vector<bool>{true, true, true, true}));
    EXPECT_TRUE(conv::widestVectorSelection({}, 16, 6).empty());
}

// ---- conv::widestPerKindSelection: rocKE's _widest_vector_sizes ----

TEST(TestGfx950ConvFwdAotSelection,
     PerKindPolicyDropsTheDefaultEpilogueOnlyWhereTheSameMergeStoresWider)
{
    // The unmerged cshuffle stores 4, so the unmerged default drops out; no merged (16)
    // instance stores wider than 1 (they record 0), so the merged default stays.
    EXPECT_EQ(conv::widestPerKindSelection({{"cshuffle", 8, 8, 4, 1},
                                            {"default", 8, 8, 1, 1},
                                            {"cshuffle", 0, 0, 0, 16},
                                            {"default", 0, 0, 0, 16}}),
              (std::vector<bool>{true, false, true, true}));
    // Every store scalar: both epilogues stay, each keeping its own widest.
    EXPECT_EQ(conv::widestPerKindSelection({{"cshuffle", 8, 8, 1},
                                            {"cshuffle", 4, 4, 1},
                                            {"default", 4, 4, 1},
                                            {"default", 8, 8, 1}}),
              (std::vector<bool>{true, false, false, true}));
    // The widest store is per merge degree: a wider store of degree 16 drops only the
    // default of degree 16.
    EXPECT_EQ(conv::widestPerKindSelection({{"cshuffle", 0, 0, 2, 16},
                                            {"default", 0, 0, 0, 16},
                                            {"default", 8, 8, 1, 1},
                                            {"default", 0, 0, 0, 4}}),
              (std::vector<bool>{true, false, true, true}));
}

TEST(TestGfx950ConvFwdAotSelection, PerKindPolicyKeepsTheWidestOfEachKind)
{
    // Of each (epilogue, merge degree), only the greatest (a + b + c, a, b, c) runs.
    EXPECT_EQ(conv::widestPerKindSelection({{"cshuffle", 8, 8, 4},
                                            {"cshuffle", 8, 8, 8},
                                            {"cshuffle", 4, 4, 8},
                                            {"default", 8, 8, 1},
                                            {"default", 4, 4, 1}}),
              (std::vector<bool>{false, true, false, false, false}));
    // Equal sums: the wider A load wins, then the wider B load.
    EXPECT_EQ(conv::widestPerKindSelection(
                  {{"cshuffle", 4, 4, 8}, {"cshuffle", 8, 4, 4}, {"cshuffle", 4, 8, 4}}),
              (std::vector<bool>{false, true, false}));
    EXPECT_EQ(conv::widestPerKindSelection(
                  {{"cshuffle", 4, 2, 8}, {"cshuffle", 2, 8, 4}, {"cshuffle", 4, 8, 2}}),
              (std::vector<bool>{false, false, true}));
    // Each merge degree is a kind of its own, beside the unmerged one.
    EXPECT_EQ(conv::widestPerKindSelection({{"cshuffle", 8, 8, 8, 1},
                                            {"cshuffle", 4, 4, 4, 1},
                                            {"cshuffle", 0, 0, 0, 16},
                                            {"cshuffle", 0, 0, 0, 32},
                                            {"cshuffle", 0, 0, 0, 4}}),
              (std::vector<bool>{true, false, true, true, true}));
}

TEST(TestGfx950ConvFwdAotSelection, PerKindPolicyKeepsTiesEveryMergedInstanceAndNothingOfNothing)
{
    // Instances of one kind with the same widths (other tiles or K loops) all run.
    EXPECT_EQ(conv::widestPerKindSelection(
                  {{"cshuffle", 8, 8, 8}, {"cshuffle", 4, 4, 4}, {"cshuffle", 8, 8, 8}}),
              (std::vector<bool>{true, false, true}));
    // Merged instances record their widths as 0: every serving one of a kind runs.
    EXPECT_EQ(conv::widestPerKindSelection({{"cshuffle", 0, 0, 0, 16},
                                            {"cshuffle", 0, 0, 0, 16},
                                            {"cshuffle", 0, 0, 0, 16},
                                            {"cshuffle", 1, 1, 1, 1}}),
              (std::vector<bool>{true, true, true, true}));
    EXPECT_TRUE(conv::widestPerKindSelection({}).empty());
}

// ---- hipkernel:Gfx942ConvFwdAot: the AOT instances built for gfx942 ----

constexpr PackSymbols GFX942_CONV_FWD_AOT{"hipkernel:Gfx942ConvFwdAot",
                                          "hipkernel.gfx942_conv_fwd_aot.graph_match",
                                          "",
                                          "hipkernel.gfx942_conv_fwd_aot.kernel_match",
                                          "hipkernel.gfx942_conv_fwd_aot.score",
                                          "hipkernel.gfx942_conv_fwd_aot.dispatch",
                                          "gfx942_conv_fwd.x.uid",
                                          "gfx942_conv_fwd.w.uid",
                                          "gfx942_conv_fwd.y.uid"};
constexpr const char* GFX942_PROBLEM_TOKEN = "_gfx942_conv_fwd.problem";

DeviceProperties gfx942()
{
    return device("gfx942");
}

/// A group-merged depthwise instance as the gfx942 catalog packages it (compv3,
/// cshuffle, grouped, 4x1 warps of 16x16x16 over a 64-row tile): @p groupMerge groups
/// per workgroup in a @p tileN wide tile, its derived widths recorded as 0.
KernelDefinition mergedSampleKernel(int64_t groupMerge, int64_t tileN)
{
    auto kernel = sampleKernel(
        {"cshuffle", {0, 0, 0, 64, tileN, 64, 4, 1, 16, 16, 16}, "compv3", {0, 0, 0, 0}});
    kernel.metadata["group_merge"] = groupMerge;
    return kernel;
}

/// aotSample() (kernels 0 to 43), then three merged instances: 44 merges 16 groups in a
/// 16-wide tile, 45 merges 32 in a 32-wide tile, 46 merges 4 in a 16-wide tile.
std::vector<KernelDefinition> gfx942Kernels()
{
    std::vector<KernelDefinition> kernels;
    for(const auto& sample : aotSample())
    {
        kernels.push_back(sampleKernel(sample));
    }
    kernels.push_back(mergedSampleKernel(16, 16));
    kernels.push_back(mergedSampleKernel(32, 32));
    kernels.push_back(mergedSampleKernel(4, 16));
    return kernels;
}

/// A graph, in SelectionCase's field order, and the kernels of gfx942Kernels()
/// hipkernel:Gfx942ConvFwdAot selects for it, in order: what
/// gfx942_conv_fwd_aot_runtime_candidates in convolution_forward.py selects from the same
/// instances. One row per line.
struct Gfx942SelectionCase
{
    const char* name;
    std::array<int64_t, 14> problem;
    std::vector<size_t> selected;
};

// clang-format off
const std::vector<Gfx942SelectionCase>& gfx942SelectionCases()
{
    static const std::vector<Gfx942SelectionCase> s_cases{
        {"dense", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 16, 40}},
        {"odd C, strided 7x7", {2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1}, {12}},
        {"odd K", {2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {3, 19, 32, 36}},
        {"odd C and K", {1, 3, 5, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {15, 35}},
        {"C 6, K 16", {2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {8}},
        {"C 4, K 12", {2, 4, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {5, 41}},
        {"grouped, odd runs", {2, 12, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {15, 35}},
        {"grouped 5x5", {1, 9, 48, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 3}, {12}},
        {"grouped 1x1", {2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3}, {0, 16, 40}},
        {"dilated", {2, 16, 24, 17, 19, 3, 3, 2, 2, 2, 2, 2, 2, 1}, {0, 16, 40}},
        {"tall 1x1", {32, 8, 8, 192, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {0, 16}},
        {"256-row tiles only", {36, 8, 8, 500, 500, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {41}},
        {"depthwise 32", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {15, 35, 44, 45, 46}},
        {"depthwise 64", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {15, 35, 44, 45, 46}},
        {"depthwise 24", {2, 24, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 24}, {15, 35, 46}},
        {"depthwise 48, strided 5x5", {1, 48, 48, 9, 9, 5, 5, 2, 2, 2, 2, 1, 1, 48}, {15, 35, 44, 46}},
        {"kpg 2", {2, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {14}},
        {"cpg and kpg 2", {2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {10, 42}},
    };
    return s_cases;
}
// clang-format on

const std::vector<int64_t>* problemToken(const BoundTokens& bound, const char* token)
{
    const auto it = bound.find(token);
    return it == bound.end() ? nullptr : std::get_if<std::vector<int64_t>>(&it->second);
}

TEST(TestGfx942ConvFwdAot, GraphMatchBindsItsOwnTokensOnlyOnAGfx942Device)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const std::array<int64_t, 14> depthwise{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64};
    const auto gfx950Match = GraphMatchRegistry::resolve(AOT_GRAPH_MATCHER);

    const GraphFixture onGfx950(convGraph(depthwise), gfx950());
    const auto gfx950Bound = gfx950Match(onGfx950.context());
    ASSERT_TRUE(gfx950Bound);
    const auto* gfx950Problem = problemToken(*gfx950Bound, "_gfx950_conv_fwd.problem");
    ASSERT_NE(gfx950Problem, nullptr);
    // The gfx942 engine does not match a gfx950 device.
    EXPECT_FALSE(matchesGraph(GFX942_CONV_FWD_AOT, onGfx950.context()));

    for(const auto* arch : {"gfx942", "gfx942:sramecc+:xnack-"})
    {
        SCOPED_TRACE(arch);
        const GraphFixture fixture(convGraph(depthwise), device(arch));
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        ASSERT_TRUE(bound);
        // Its own tokens and only those: public tokens are UHD query features.
        EXPECT_EQ(bound->size(), 4U);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_FWD_AOT.inputAToken), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_FWD_AOT.inputBToken), 2);
        EXPECT_EQ(tryGetBoundInt(*bound, GFX942_CONV_FWD_AOT.outputToken), 3);
        EXPECT_EQ(bound->count("gfx950_conv_fwd.x.uid"), 0U);
        EXPECT_EQ(bound->count("_gfx950_conv_fwd.problem"), 0U);
        // The private problem token describes the same problem the gfx950 engine reads.
        const auto* problem = problemToken(*bound, GFX942_PROBLEM_TOKEN);
        ASSERT_NE(problem, nullptr);
        EXPECT_EQ(problem->size(), 18U);
        EXPECT_EQ(*problem, *gfx950Problem);
        // The gfx950 AOT engine does not match a gfx942 device.
        EXPECT_FALSE(gfx950Match(fixture.context()));
    }

    for(const auto* arch : {"gfx950", "gfx9420", "gfx94", "gfx90a", "", "prefix-gfx942"})
    {
        SCOPED_TRACE(arch);
        const GraphFixture refused(convGraph(depthwise), device(arch));
        EXPECT_FALSE(matchesGraph(GFX942_CONV_FWD_AOT, refused.context()));
    }
    auto properties = gfx942();
    properties.warpSize = 32;
    const GraphFixture refused(convGraph(depthwise), properties);
    EXPECT_FALSE(matchesGraph(GFX942_CONV_FWD_AOT, refused.context()));
}

TEST(TestGfx942ConvFwdAot, MergedKernelServesOnlyDepthwiseGraphsItsMergeDegreeDivides)
{
    const auto serves = [](const std::array<int64_t, 14>& problem,
                           const KernelDefinition& kernel,
                           const DeviceProperties& properties) {
        const GraphFixture fixture(convGraph(problem), properties);
        const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
        const bool served = matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel);
        // The bound problem decides as the graph does.
        EXPECT_EQ(bound && matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel, *bound),
                  served);
        return served;
    };
    using Graph = std::array<int64_t, 14>;
    const auto merged = mergedSampleKernel(16, 16);
    // Depthwise, with the group count a multiple of 16.
    for(const auto& problem : {Graph{2, 16, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 16},
                               Graph{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32},
                               Graph{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64},
                               Graph{1, 48, 48, 9, 9, 5, 5, 2, 2, 2, 2, 1, 1, 48}})
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        EXPECT_TRUE(serves(problem, merged, gfx942()));
        // Not by the gfx942 engine on another device.
        EXPECT_FALSE(serves(problem, merged, gfx950()));
    }
    // Group counts 16 does not divide, and graphs that are not depthwise.
    for(const auto& problem : {Graph{2, 24, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 24},
                               Graph{2, 8, 8, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8},
                               Graph{2, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64},
                               Graph{2, 128, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64},
                               Graph{2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64},
                               Graph{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}})
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        EXPECT_FALSE(serves(problem, merged, gfx942()));
    }
    // A 4-group merge serves the 24 groups 16 does not divide.
    EXPECT_TRUE(serves(
        {2, 24, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 24}, mergedSampleKernel(4, 16), gfx942()));
    // A merge degree above the tile width.
    EXPECT_FALSE(serves(
        {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, mergedSampleKernel(32, 16), gfx942()));
}

TEST(TestGfx942ConvFwdAot, GroupMergeMetadataDecodesAsTheCatalogDeclaresIt)
{
    const std::array<int64_t, 14> depthwise{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32};
    const GraphFixture fixture(convGraph(depthwise), gfx942());
    const auto serves = [&](const KernelDefinition& kernel) {
        return matchesKernel(GFX942_CONV_FWD_AOT, fixture.context(), kernel);
    };
    const auto merged = mergedSampleKernel(16, 16);
    ASSERT_TRUE(serves(merged));

    // A merged identity records its derived widths as 0, never a width.
    for(const auto* field : {"vector_size_a", "vector_size_b", "vector_size_c"})
    {
        SCOPED_TRACE(field);
        auto changed = merged;
        changed.metadata[field] = int64_t{1};
        EXPECT_FALSE(serves(changed));
    }
    // A merge degree is positive.
    for(const auto groupMerge : {int64_t{0}, int64_t{-16}})
    {
        SCOPED_TRACE(groupMerge);
        auto changed = merged;
        changed.metadata["group_merge"] = groupMerge;
        EXPECT_FALSE(serves(changed));
    }
    // Absent, the degree is 1: 0 widths are then an unmerged identity's, which refuses.
    auto changed = merged;
    changed.metadata.erase("group_merge");
    EXPECT_FALSE(serves(changed));
    changed.metadata["group_merge"] = int64_t{1};
    EXPECT_FALSE(serves(changed));

    // An unmerged identity decodes the same with group_merge 1 stated or absent, and
    // its positive widths refuse a merge degree.
    auto scalar = sampleKernel(aotSample()[15]);
    ASSERT_TRUE(serves(scalar));
    scalar.metadata["group_merge"] = int64_t{1};
    EXPECT_TRUE(serves(scalar));
    scalar.metadata["group_merge"] = int64_t{16};
    EXPECT_FALSE(serves(scalar));

    // A direct identity has no group merge.
    auto direct = directSampleKernel();
    ASSERT_TRUE(serves(direct));
    direct.metadata["group_merge"] = int64_t{1};
    EXPECT_TRUE(serves(direct));
    direct.metadata["group_merge"] = int64_t{16};
    EXPECT_FALSE(serves(direct));
}

TEST(TestGfx942ConvFwdAot, SelectsWhatTheRuntimeSelectionRunsOverTheSampleAndMergedKernels)
{
    const auto kernels = gfx942Kernels();
    for(const auto& [name, problem, expected] : gfx942SelectionCases())
    {
        SCOPED_TRACE(name);
        const GraphFixture fixture(convGraph(problem), gfx942());
        std::vector<size_t> selected;
        const auto serving = serveAndSelect(fixture.context(),
                                            kernels,
                                            selected,
                                            GFX942_CONV_FWD_AOT.graphMatcher.data(),
                                            GFX942_CONV_FWD_AOT.kernelMatcher.data());
        EXPECT_EQ(selected, expected);
        EXPECT_LE(selected.size(), serving.size());
        EXPECT_FALSE(selected.empty());
        // Every serving merged instance runs.
        for(const auto index : serving)
        {
            if(index >= aotSample().size())
            {
                EXPECT_NE(std::find(selected.begin(), selected.end(), index), selected.end())
                    << index;
            }
        }
    }
}

TEST(TestGfx942ConvFwdAot, KeepsEveryServingDirectAndMergedInstance)
{
    std::vector<KernelDefinition> kernels{directSampleKernel()};
    for(auto& kernel : gfx942Kernels())
    {
        kernels.push_back(std::move(kernel));
    }
    const GraphFixture fixture(convGraph({2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}),
                               gfx942());
    std::vector<size_t> selected;
    serveAndSelect(fixture.context(),
                   kernels,
                   selected,
                   GFX942_CONV_FWD_AOT.graphMatcher.data(),
                   GFX942_CONV_FWD_AOT.kernelMatcher.data());
    // Kernel r of gfx942Kernels() is kernel r + 1 here.
    EXPECT_EQ(selected, (std::vector<size_t>{0, 16, 36, 45, 46, 47}));
}

TEST(TestGfx942ConvFwdAot, PreparedKernelsAndTheBoundProblemDecideAsTheGraphAndMetadataDo)
{
    const auto graphMatch = graphMatcher(GFX942_CONV_FWD_AOT);
    const auto matcher = kernelMatcher(GFX942_CONV_FWD_AOT);
    const auto filter
        = CandidateFilterRegistry::resolve(std::string(GFX942_CONV_FWD_AOT.kernelMatcher));
    const auto score = scorer(GFX942_CONV_FWD_AOT);
    const auto prepare
        = KernelPrepareRegistry::resolve(std::string(GFX942_CONV_FWD_AOT.kernelMatcher));

    // The sample and merged instances, a direct instance, and merged metadata no launch
    // contract describes: a width recorded, no degree, a direct identity with one.
    std::vector<KernelDefinition> plain{directSampleKernel()};
    for(auto& kernel : gfx942Kernels())
    {
        plain.push_back(std::move(kernel));
    }
    plain.push_back(mergedSampleKernel(16, 16));
    plain.back().metadata["vector_size_c"] = int64_t{1};
    plain.push_back(mergedSampleKernel(16, 16));
    plain.back().metadata["group_merge"] = int64_t{0};
    plain.push_back(mergedSampleKernel(16, 16));
    plain.back().metadata.erase("group_merge");
    plain.push_back(directSampleKernel());
    plain.back().metadata["group_merge"] = int64_t{16};
    plain.push_back(mergedSampleKernel(16, 16));
    plain.back().metadata["dtype"] = std::string("bf16");

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

    for(const auto& [name, problem, expected] : gfx942SelectionCases())
    {
        SCOPED_TRACE(name);
        const GraphFixture fixture(convGraph(problem), gfx942());
        const auto& context = fixture.context();
        const auto bound = graphMatch(context);
        ASSERT_TRUE(bound.has_value());

        std::vector<const KernelDefinition*> plainServing;
        std::vector<const KernelDefinition*> preparedServing;
        for(size_t i = 0; i < plain.size(); ++i)
        {
            const bool served = matcher(context, BoundTokens{}, plain[i]);
            EXPECT_EQ(matcher(context, *bound, plain[i]), served) << i;
            EXPECT_EQ(matcher(context, *bound, prepared[i]), served) << i;
            EXPECT_EQ(matcher(context, *bound, foreign[i]), served) << i;
            EXPECT_EQ(score(context, *bound, prepared[i]), score(context, BoundTokens{}, plain[i]))
                << i;
            if(i + 5 >= plain.size())
            {
                // The malformed and bf16 instances serve no fp16 graph.
                EXPECT_FALSE(served) << i;
            }
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

TEST(TestGfx942ConvFwdAot, FallbackScoreSizesTheGridToTheEnginesDevice)
{
    // A 16-group merge on 64 depthwise groups launches 1 x 7 x 4 tiles: the gfx942
    // engine weighs that against an MI300X's 304 CUs, the gfx950 engine against 256.
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const std::array<int64_t, 14> depthwise{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64};
    const conv::Problem problem{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64};
    const auto geometry = conv::deriveGeometry(problem);
    ASSERT_TRUE(geometry);
    const conv::AotInstance instance{64, 16, 4, 1, 0, 0, 0, false, 0, 0, true, 16};
    const auto launch = conv::aotInstanceLaunch(problem, *geometry, instance);
    ASSERT_TRUE(launch);
    const auto merged = mergedSampleKernel(16, 16);

    const GraphFixture onGfx942(convGraph(depthwise), gfx942());
    const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, onGfx942.context());
    ASSERT_TRUE(bound);
    const auto gfx942Score = scoreKernel(GFX942_CONV_FWD_AOT, onGfx942.context(), merged, *bound);
    EXPECT_DOUBLE_EQ(gfx942Score,
                     conv::aotInstanceScore(problem, *geometry, instance, *launch, 304.0));

    const GraphFixture onGfx950(convGraph(depthwise), gfx950());
    const auto gfx950Score = ScoreRegistry::resolve("hipkernel.gfx950_conv_fwd_aot.score")(
        onGfx950.context(), BoundTokens{}, merged);
    EXPECT_DOUBLE_EQ(gfx950Score,
                     conv::aotInstanceScore(problem, *geometry, instance, *launch, 256.0));
    EXPECT_GT(gfx950Score, gfx942Score);

    // A graph the instance does not serve ranks it last.
    const GraphFixture unserved(convGraph({2, 24, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 24}),
                                gfx942());
    EXPECT_EQ(scoreKernel(GFX942_CONV_FWD_AOT, unserved.context(), merged), -1.0);
}

TEST(TestGfx942ConvFwdAot, DispatchChecksItsOwnBindingsAndReachesCodeLoading)
{
    const GraphFixture fixture(convGraph({2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}),
                               gfx942());
    const auto bound = matchesGraph(GFX942_CONV_FWD_AOT, fixture.context());
    ASSERT_TRUE(bound);
    const auto& handler = dispatchHandler(GFX942_CONV_FWD_AOT);
    const auto merged = mergedSampleKernel(16, 16);
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), *bound, merged), 0U);

    // The gfx950 engine's tokens are not this engine's bindings.
    BoundTokens gfx950Bound{{"gfx950_conv_fwd.x.uid", int64_t{1}},
                            {"gfx950_conv_fwd.w.uid", int64_t{2}},
                            {"gfx950_conv_fwd.y.uid", int64_t{3}}};
    for(const auto& wrong : {BoundTokens{}, gfx950Bound})
    {
        try
        {
            handler.prepare(fixture.context(), wrong, merged);
            FAIL() << "Preparation accepted bindings another graph matcher made";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            EXPECT_NE(std::string(error.what()).find("gfx942_conv_fwd dispatch requires"),
                      std::string::npos)
                << error.what();
        }
    }

    try
    {
        // The descriptor names no archive, so loading is the first step that can fail.
        handler.prepare(fixture.context(), *bound, merged);
        FAIL() << "Preparation loaded a kernel from an empty archive path";
    }
    catch(const std::exception& error)
    {
        const std::string message = error.what();
        EXPECT_EQ(message.find("convolution contract"), std::string::npos) << message;
        EXPECT_EQ(message.find("tensor bindings"), std::string::npos) << message;
        EXPECT_EQ(message.find("packed NHWK output"), std::string::npos) << message;
    }

    // A group count 16 does not divide refuses before any code loads.
    const GraphFixture unserved(convGraph({2, 24, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 24}),
                                gfx942());
    const auto unservedBound = matchesGraph(GFX942_CONV_FWD_AOT, unserved.context());
    ASSERT_TRUE(unservedBound);
    try
    {
        handler.prepare(unserved.context(), *unservedBound, merged);
        FAIL() << "Preparation accepted a merge degree that does not divide the groups";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_NE(std::string(error.what()).find("gfx942_conv_fwd graph or packaged kernel"),
                  std::string::npos)
            << error.what();
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
