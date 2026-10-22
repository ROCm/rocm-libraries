// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

/**
 * @file AttentionTiled2DSignature.hpp
 * @brief The kernel signature of rocKE's tiled 2D unified attention
 *        (`kernels/{gfx942,gfx950}/attention_tiled_2d.py`, `build_unified_attention_2d_tiled`),
 *        shared by every engine that launches it.
 *
 * NAMES ARE LOAD-BEARING. requireSignatureMatch compares kind and size always, but names
 * only when BOTH sides carry one. Kind and size alone cannot tell the ten pointers apart, nor
 * the f32 scalars from the i32 ones, so without names an operand permutation passes the check
 * and the kernel reads the wrong buffers with no error and no status code. hkp_pack lowers
 * the recorded names out of the rocKE builder's own parameter list, so keep these spellings
 * identical to the Python (build_unified_attention_2d_tiled's `b.param` calls); a divergence
 * here fails every dispatch rather than silently weakening the check.
 *
 * Offsets mirror the packed kernarg layout. They are not compared -- they exist so the
 * mismatch diagnostic prints the real layout beside the recorded one.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Appends one argument at the next offset its size aligns to.
inline void
    appendKernelArgument(std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& signature,
                         std::string_view kind,
                         uint32_t size,
                         std::string_view name)
{
    uint32_t offset = 0;
    if(!signature.empty())
    {
        offset = signature.back().offset + signature.back().size;
    }
    offset = (offset + size - 1) / size * size;
    signature.push_back(hipdnn_plugin_sdk::ingestor::KernelArgument{
        std::string(kind), size, offset, std::string(name)});
}

/// The arguments every tiled 2D body takes: 10 pointers, 6 scalars, the two strides
/// (`_attn_signature(include_bt_stride=True, include_qq_bias_stride=True)`) and the runtime
/// shape kernargs (`RUNTIME_KERNARG_FIELDS`: head counts and window). A runtime_layout
/// kernel appends its own arguments after these.
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> tiled2DKernelSignature()
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> signature;
    for(const auto* name : {"output_ptr",
                            "query_ptr",
                            "key_cache_ptr",
                            "value_cache_ptr",
                            "sink_ptr",
                            "block_tables_ptr",
                            "seq_lens_ptr",
                            "alibi_slopes_ptr",
                            "qq_bias_ptr",
                            "query_start_len_ptr"})
    {
        appendKernelArgument(signature, "global_buffer", PTR, name);
    }
    for(const auto* name : {"scale", "k_scale", "v_scale", "out_scale", "softcap"})
    {
        appendKernelArgument(signature, "by_value", F32, name);
    }
    for(const auto* name : {"num_seqs",
                            "block_table_stride",
                            "qq_bias_stride_0",
                            "num_query_heads",
                            "num_kv_heads",
                            "sliding_window"})
    {
        appendKernelArgument(signature, "by_value", I32, name);
    }
    return signature;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
