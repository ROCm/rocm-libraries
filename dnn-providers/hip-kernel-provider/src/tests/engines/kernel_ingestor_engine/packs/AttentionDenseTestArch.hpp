// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// What a gfx942 or gfx950 dense attention test needs to know about its engine. The pack
/// is shared, so a case written against one of these runs unchanged on either arch.
struct AttentionDenseTestArch
{
    /// The TEST_P parameter name.
    std::string_view label;
    std::string_view gcnArchName;
    std::string_view graphMatchSymbol;
    std::string_view dispatchSymbol;
    /// The prefix of every bound token, restated rather than shared with the pack so a
    /// rename on one side fails here.
    std::string_view tokenPrefix;
    /// A D128 bm256/bn64 catalog kernel name.
    std::string_view kernelName;
    /// gfx942 records carry the grid-order fields the KMD defaults complete; gfx950
    /// records carry the tile alone.
    bool completesGridOrder;
    /// gfx942 binds a stats output and ships an emit_lse form; gfx950 does neither.
    bool bindsStats;
    std::string_view kernelMatchSymbol;
    /// The head of every decline line the graph match logs.
    std::string_view engineName;
};

inline constexpr AttentionDenseTestArch GFX942_ATTENTION_DENSE{
    "Gfx942",
    "gfx942",
    "hipkernel.gfx942_attention_dense.graph_match",
    "hipkernel.gfx942_attention_dense.dispatch",
    "gfx942_attention_dense",
    "attention_dense.bf16_d128_c1_swa0_bm256_bn64_lds_row_pad8_pv_priority0.gfx942",
    true,
    true,
    "hipkernel.gfx942_attention_dense.kernel_match",
    "hipkernel:Gfx942AttentionDense"};

inline constexpr AttentionDenseTestArch GFX950_ATTENTION_DENSE{
    "Gfx950",
    "gfx950",
    "hipkernel.gfx950_attention_dense.graph_match",
    "hipkernel.gfx950_attention_dense.dispatch",
    "gfx950_attention_dense",
    "attention_dense.bf16_d128_hq4_kv4_c1_bm256_bn64.gfx950",
    false,
    false,
    "hipkernel.gfx950_attention_dense.kernel_match",
    "hipkernel:Gfx950AttentionDense"};

inline std::string
    attentionDenseArchName(const ::testing::TestParamInfo<const AttentionDenseTestArch*>& info)
{
    return std::string(info.param->label);
}

/// The completed metadata prepare() reads before the signature comparison: head_size, the
/// (block_m, block_n) tile and, on gfx942, the qb_minor grid as the KMD defaults complete
/// it.
inline hipdnn_plugin_sdk::ingestor::MetadataValues completedAttentionDenseMetadata(
    const AttentionDenseTestArch& arch, int64_t headSize, int64_t blockM, int64_t blockN)
{
    hipdnn_plugin_sdk::ingestor::MetadataValues metadata{
        {std::string("head_size"), headSize},
        {std::string("block_m"), blockM},
        {std::string("block_n"), blockN},
    };
    if(arch.completesGridOrder)
    {
        metadata[std::string("persistent")] = int64_t{0};
        metadata[std::string("num_persistent")] = int64_t{304};
        metadata[std::string("persist_decode")] = std::string("auto");
        metadata[std::string("nonpersist_decode")] = std::string("qb_minor");
    }
    return metadata;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
