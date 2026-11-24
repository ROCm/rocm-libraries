// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionDenseCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseLaunchValues.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/**
 * @brief The values a save stores for a plan under @p symbol, one of @p arch's contracts: the
 *        call the dispatch handler's save makes (attention_dense::launchValues) with that
 *        contract.
 *
 * @throws std::invalid_argument when @p symbol names none of @p arch's contracts.
 */
inline hipdnn_plugin_sdk::ingestor::MetadataValues
    launchValuesUnder(std::string_view symbol,
                      const attention_dense::AttentionDenseArch& arch,
                      const AttentionDenseBinding& binding,
                      const AttentionDenseProblem& problem,
                      int64_t blockM,
                      const attention_dense::AttentionDenseGrid& grid = {})
{
    const auto* contract = arch.contractNamed(symbol);
    if(contract == nullptr)
    {
        throw std::invalid_argument(std::string(symbol) + " names no contract of "
                                    + std::string(arch.label));
    }
    return attention_dense::launchValues(binding, problem, blockM, arch, *contract, grid);
}

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
    /// Both arches serve an additive bias (attn_mask) with their use_additive_bias kernels,
    /// a field a completed record carries (a record without it reads as 0). A case on the
    /// bias is instantiated only for the arches that serve one (biasServingArches()).
    bool servesBias;
    /// gfx942 binds a causal sliding window (its catalog ships windowed kernels); gfx950's
    /// graph match declines every window.
    bool servesSlidingWindow;
    std::string_view kernelMatchSymbol;
    /// The head of every decline line the graph match logs.
    std::string_view engineName;
    /// The versioned dispatch names a plan is saved under today (the strided contract and the
    /// strided bias contract), restated rather than shared with the pack: without a bias, and
    /// with one (empty for an arch that serves none).
    std::string_view plainContractSymbol;
    std::string_view biasContractSymbol;
    /// The engine's own descriptor, whose contracts a save writes under (launchValuesUnder).
    const attention_dense::AttentionDenseArch& (*engineArch)();
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
    true,
    true,
    "hipkernel.gfx942_attention_dense.kernel_match",
    "hipkernel:Gfx942AttentionDense",
    "hipkernel.gfx942_attention_dense.dispatch.v3",
    "hipkernel.gfx942_attention_dense.dispatch.v4",
    &gfx942AttentionDenseArch};

inline constexpr AttentionDenseTestArch GFX950_ATTENTION_DENSE{
    "Gfx950",
    "gfx950",
    "hipkernel.gfx950_attention_dense.graph_match",
    "hipkernel.gfx950_attention_dense.dispatch",
    "gfx950_attention_dense",
    "attention_dense.bf16_d128_hq4_kv4_c1_bm256_bn64.gfx950",
    false,
    false,
    true,
    false,
    "hipkernel.gfx950_attention_dense.kernel_match",
    "hipkernel:Gfx950AttentionDense",
    "hipkernel.gfx950_attention_dense.dispatch.v4",
    "hipkernel.gfx950_attention_dense.dispatch.v5",
    &gfx950AttentionDenseArch};

/// The arches a case on the additive bias runs on: those that serve one.
inline std::vector<const AttentionDenseTestArch*> biasServingArches()
{
    std::vector<const AttentionDenseTestArch*> arches;
    for(const auto* arch : {&GFX942_ATTENTION_DENSE, &GFX950_ATTENTION_DENSE})
    {
        if(arch->servesBias)
        {
            arches.push_back(arch);
        }
    }
    return arches;
}

inline std::string
    attentionDenseArchName(const ::testing::TestParamInfo<const AttentionDenseTestArch*>& info)
{
    return std::string(info.param->label);
}

/// The completed metadata prepare() reads before the signature comparison: head_size, the
/// (block_m, block_n) tile and, on gfx942, the qb_minor grid and the baseline body knobs
/// (lds_num_buffers, v_reg_prefetch) as the KMD defaults complete them.
inline hipdnn_plugin_sdk::ingestor::MetadataValues completedAttentionDenseMetadata(
    const AttentionDenseTestArch& arch, int64_t headSize, int64_t blockM, int64_t blockN)
{
    hipdnn_plugin_sdk::ingestor::MetadataValues metadata{
        // A record completes to the grid body: the KMD default of the field the stride rule reads.
        {std::string("ragged"), int64_t{0}},
        {std::string("head_size"), headSize},
        {std::string("block_m"), blockM},
        {std::string("block_n"), blockN},
    };
    if(arch.servesBias)
    {
        // A plain record, which the KMD default completes to 0.
        metadata[std::string("use_additive_bias")] = int64_t{0};
    }
    if(arch.completesGridOrder)
    {
        // The baseline body: one K buffer, no V prefetch. It reads the strides.
        metadata[std::string("lds_num_buffers")] = int64_t{1};
        metadata[std::string("v_reg_prefetch")] = int64_t{0};
        metadata[std::string("persistent")] = int64_t{0};
        metadata[std::string("num_persistent")] = int64_t{304};
        metadata[std::string("persist_decode")] = std::string("auto");
        metadata[std::string("nonpersist_decode")] = std::string("qb_minor");
    }
    return metadata;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
