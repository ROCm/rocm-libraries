// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/UnifiedAttentionTestGraphs.hpp"

/**
 * @file Gfx950AttentionUnified2DTestGraphs.hpp
 * @brief Graphs and candidate records the hipkernel:Gfx950AttentionUnified2D suites share.
 *
 * The symbol and token spellings are written out here rather than shared with the pack:
 * they are the contract between the descriptors, the matcher that writes a token and the
 * dispatch that reads it, and a test sharing the constant would keep passing through a rename.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified2d
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;

inline constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_2d.graph_match";
inline constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_2d.kernel_match";
inline constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_unified_2d.score";
inline constexpr std::string_view DISPATCH_SYMBOL
    = "hipkernel.gfx950_attention_unified_2d.dispatch";

inline constexpr std::string_view Q_TOKEN = "gfx950_attention_unified_2d.q.uid";
inline constexpr std::string_view K_TOKEN = "gfx950_attention_unified_2d.k.uid";
inline constexpr std::string_view V_TOKEN = "gfx950_attention_unified_2d.v.uid";
inline constexpr std::string_view O_TOKEN = "gfx950_attention_unified_2d.o.uid";
inline constexpr std::string_view MASK_TOKEN = "gfx950_attention_unified_2d.mask_type";
inline constexpr std::string_view SLIDING_WINDOW_TOKEN
    = "gfx950_attention_unified_2d.sliding_window";
inline constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_unified_2d.scale_bits";
inline constexpr std::string_view STATS_TOKEN = "gfx950_attention_unified_2d.stats.uid";

using namespace unified_attention;

/// Unified2D's default graph: Sq 128 over Skv 256 (prefill).
using GraphSpec = BasicGraphSpec<128>;
using Graph = BasicGraph<GraphSpec>;

/// Runs the pack's graph match over @p graph.
inline std::optional<BoundTokens> matchGraph(const Graph& graph)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    return matcher(graph.context());
}

inline std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    const Graph graph(spec);
    return matchGraph(graph);
}

/// The fields the engine's KMD declares, as a completed catalog record carries them. The
/// defaults are the shipped baseline kernel for the default graph (bf16, D128, Hq 8 / Hkv 2,
/// top-left causal, block 16, T64, 4 warps x 16 rows, num_seqs 511). The engine does not read
/// num_seqs -- the batch is a kernarg -- but the KMD still declares it.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = 128;
    int64_t numQueryHeads = 8;
    int64_t numKvHeads = 2;
    int64_t blockSize = 16;
    int64_t maskType = TOP_LEFT_CAUSAL;
    int64_t slidingWindow = 0;
    int64_t numSeqs = 511;
    int64_t tileSize = 64;
    int64_t numWarps = 4;
    int64_t blockMPerWarp = 16;
    int64_t useMfma32x32 = 0;
    int64_t useTransposedQk32x32 = 0;
    /// 1 for an LSE-on kernel, which serves only graphs that ask for stats.
    int64_t emitLse = 0;
    /// The descriptor id's final byte. Only the ranking cases vary it.
    unsigned idByte = 0xa1;
};

/// The metadata of a completed record, as the descriptor loader hands it over.
inline hipdnn_plugin_sdk::ingestor::MetadataValues metadataOf(const KernelSpec& spec)
{
    return {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("num_query_heads"), spec.numQueryHeads},
        {std::string("num_kv_heads"), spec.numKvHeads},
        {std::string("block_size"), spec.blockSize},
        {std::string("mask_type"), spec.maskType},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("num_seqs"), spec.numSeqs},
        {std::string("tile_size"), spec.tileSize},
        {std::string("num_warps"), spec.numWarps},
        {std::string("block_m_per_warp"), spec.blockMPerWarp},
        {std::string("use_mfma_32x32"), spec.useMfma32x32},
        {std::string("use_transposed_qk_32x32"), spec.useTransposedQk32x32},
        {std::string("emit_lse"), spec.emitLse},
    };
}

inline KernelDefinition makeKernel(const KernelSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId = idEndingIn(spec.idByte);
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea3");
    kernel.name = "attention_unified_2d.bf16_d128_hq8_kv2_m1_bs16_t64_nw4.gfx950";
    kernel.metadata = metadataOf(spec);
    return kernel;
}

/// Runs graph_match then kernel_match for @p kernel, exactly as a catalog build does.
inline bool matchesKernel(const Graph& graph, const KernelDefinition& kernel)
{
    registerNativeIngestorSymbols();
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return kernelMatcher(graph.context(), *bound, kernel);
}

inline bool matchesKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    const Graph graph(graphSpec);
    return matchesKernel(graph, makeKernel(kernelSpec));
}

inline double scoreOf(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
    const Graph graph(graphSpec);
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value());
    return scorer(graph.context(), bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

/// The 21-argument ABI the rocKE builder declares, under the names it gives them.
inline std::vector<KernelArgument> pythonAbi()
{
    constexpr uint32_t POINTER_BYTES = 8;
    constexpr uint32_t SCALAR_BYTES = 4;
    const auto buffer
        = [](const char* name) { return KernelArgument{"global_buffer", POINTER_BYTES, 0, name}; };
    const auto scalar
        = [](const char* name) { return KernelArgument{"by_value", SCALAR_BYTES, 0, name}; };
    return {buffer("output_ptr"),
            buffer("query_ptr"),
            buffer("key_cache_ptr"),
            buffer("value_cache_ptr"),
            buffer("sink_ptr"),
            buffer("block_tables_ptr"),
            buffer("seq_lens_ptr"),
            buffer("alibi_slopes_ptr"),
            buffer("qq_bias_ptr"),
            buffer("query_start_len_ptr"),
            scalar("scale"),
            scalar("k_scale"),
            scalar("v_scale"),
            scalar("out_scale"),
            scalar("softcap"),
            scalar("num_seqs"),
            scalar("block_table_stride"),
            scalar("qq_bias_stride_0"),
            scalar("num_query_heads"),
            scalar("num_kv_heads"),
            scalar("sliding_window")};
}

/// A KPACK descriptor for this engine naming an archive that is not there, recording
/// @p recorded. That absence is the point: every check prepare() performs BEFORE the loader
/// runs is reached, the signature comparison runs next, and when both pass the loader reports
/// the missing file. No device, no compile, no launch.
inline KernelDefinition makeKpackKernel(const KernelSpec& spec,
                                        const std::vector<KernelArgument>& recorded)
{
    auto kernel = makeKernel(spec);
    kernel.source.kind = KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "attention_unified_2d";
    kernel.source.signature = recorded;
    kernel.originDirectory = "/tmp/hkp-unified-2d-no-such-tree";
    kernel.treeRoot = "/tmp/hkp-unified-2d-no-such-tree";
    return kernel;
}

/// The message prepare() throws for @p bound and @p kernel over @p graph, or empty if it
/// returned a prepared dispatch -- which it cannot, since the archive is absent.
inline std::string
    prepareFailure(const Graph& graph, const BoundTokens& bound, const KernelDefinition& kernel)
{
    registerNativeIngestorSymbols();
    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }
    try
    {
        handler->prepare(graph.context(), bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified2d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
