// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx942AttentionUnified3dArguments.hpp
 * @brief The segment kernel's ordered argument list for gfx942 unified 3D strided decode,
 *        as plain data: scalar values, and which buffer each pointer argument takes.
 *
 * Restates the `seg_vals` dict `kernels/common/attention_unified.py::_run_3d_tiled_on_stream`
 * builds for a strided decode request, in the order of `_3d_signature(dtype,
 * strided_kv=True)`: 12 pointers, 4 f32, 3 i32, then `StridedKvCacheLayout.arguments()`
 * (`kernels/common/attention_kv_cache.py`) for K then V. The launch passes these values
 * unchanged, so a wrong value or order here is a wrong launch; nothing faults.
 *
 * Header-only and HIP-free: TestGfx942AttentionUnified3dArguments.cpp checks every value
 * against a table rocKE generated (gen_gfx942_attention_unified_3d_arguments.py).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// fp16 / bf16: the only element types the strided entry admits.
inline constexpr int64_t GFX942_UNIFIED_3D_KV_ELEMENT_BYTES = 2;

/// The segment kernel's argument count with strided K/V: 19 base + 8 strided.
inline constexpr size_t GFX942_UNIFIED_3D_SEGMENT_ARGUMENT_COUNT = 27;

/// One K or V tensor's stride arguments (`KvTensorLayout.arguments(prefix)`), in bytes.
struct Gfx942StridedKvArguments
{
    int64_t strideBatchBytes = 0;
    int64_t strideHeadBytes = 0;
    int32_t strideTokenBytes = 0;
    int32_t spanBytes = 0;
};

/**
 * @brief `KvTensorLayout(dims, strides).arguments(prefix)` for one K or V tensor.
 *
 * @param dims    LOGICAL extents (B, H, capacity, D).
 * @param strides Element strides in the same logical order (any physical layout: BHSD,
 *                BSHD, padded rows).
 *
 *     stride_batch_bytes = 2 * strides[0]
 *     stride_head_bytes  = 2 * strides[1]
 *     stride_token_bytes = 2 * strides[2]
 *     span_bytes         = 2 * ((capacity - 1) * strides[2] + D)   (head_span_bytes)
 *
 * Precondition: the tensor passed KvTensorLayout's admission (stridedKvRefusal), which
 * keeps the token stride and the span inside the buffer-offset range. A token stride or
 * span outside int32 throws rather than truncate.
 */
inline Gfx942StridedKvArguments gfx942StridedKvArguments(const std::array<int64_t, 4>& dims,
                                                         const std::array<int64_t, 4>& strides)
{
    constexpr int64_t E = GFX942_UNIFIED_3D_KV_ELEMENT_BYTES;
    const int64_t tokenBytes = strides[2] * E;
    const int64_t spanBytes = E * ((dims[2] - 1) * strides[2] + dims[3]);
    if(tokenBytes < 0 || spanBytes <= 0 || tokenBytes > std::numeric_limits<int32_t>::max()
       || spanBytes > std::numeric_limits<int32_t>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 unified 3d: strided K/V token stride or head span exceeds int32 bytes");
    }
    Gfx942StridedKvArguments args;
    args.strideBatchBytes = strides[0] * E;
    args.strideHeadBytes = strides[1] * E;
    args.strideTokenBytes = static_cast<int32_t>(tokenBytes);
    args.spanBytes = static_cast<int32_t>(spanBytes);
    return args;
}

/// How one argument is passed.
enum class Gfx942Unified3dArgumentKind : uint8_t
{
    POINTER,
    F32,
    I32,
    I64,
};

/// The buffer a pointer argument takes. NONE is a null pointer.
enum class Gfx942Unified3dBuffer : uint8_t
{
    NONE,
    SEGM_OUTPUT, // workspace region segmOutput
    SEGM_MAX, // workspace region segmMax
    SEGM_EXPSUM, // workspace region segmExpsum
    QUERY, // graph Q
    KEY, // graph K
    VALUE, // graph V
    SEQUSED_K, // workspace region sequsedK
    CU_SEQLENS_Q, // workspace region cuSeqlensQ
};

/// One kernel argument. For scalars, @c bits holds the value's bytes zero-extended to
/// 64 bits (little-endian: the launch passes the word's address and the runtime copies
/// the argument's size from its low bytes). For pointers @c bits is 0 and @c buffer names
/// the buffer.
struct Gfx942Unified3dArgument
{
    const char* name = "";
    Gfx942Unified3dArgumentKind kind = Gfx942Unified3dArgumentKind::POINTER;
    Gfx942Unified3dBuffer buffer = Gfx942Unified3dBuffer::NONE;
    uint64_t bits = 0;

    uint32_t bytes() const
    {
        return kind == Gfx942Unified3dArgumentKind::F32 || kind == Gfx942Unified3dArgumentKind::I32
                   ? 4u
                   : 8u;
    }

    int64_t integer() const
    {
        return kind == Gfx942Unified3dArgumentKind::I32
                   ? static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(bits)))
                   : static_cast<int64_t>(bits);
    }

    float f32() const
    {
        const auto word = static_cast<uint32_t>(bits);
        float value = 0.0F;
        std::memcpy(&value, &word, sizeof(value));
        return value;
    }
};

using Gfx942Unified3dSegmentArguments
    = std::array<Gfx942Unified3dArgument, GFX942_UNIFIED_3D_SEGMENT_ARGUMENT_COUNT>;

namespace detail
{

inline Gfx942Unified3dArgument gfx942Pointer(const char* name, Gfx942Unified3dBuffer buffer)
{
    return {name, Gfx942Unified3dArgumentKind::POINTER, buffer, 0};
}

inline Gfx942Unified3dArgument gfx942F32(const char* name, float value)
{
    uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    return {name, Gfx942Unified3dArgumentKind::F32, Gfx942Unified3dBuffer::NONE, word};
}

inline Gfx942Unified3dArgument gfx942I32(const char* name, int32_t value)
{
    return {name,
            Gfx942Unified3dArgumentKind::I32,
            Gfx942Unified3dBuffer::NONE,
            static_cast<uint32_t>(value)};
}

inline Gfx942Unified3dArgument gfx942I64(const char* name, int64_t value)
{
    return {name,
            Gfx942Unified3dArgumentKind::I64,
            Gfx942Unified3dBuffer::NONE,
            static_cast<uint64_t>(value)};
}

} // namespace detail

/**
 * @brief The segment launch's 27 arguments, in kernel order.
 *
 * Sinks, block tables, ALiBi and QQ bias are null (graph_match refuses them);
 * `block_table_stride` is 0 as the strided entry passes; k_scale = v_scale = 1 and
 * softcap = 0 (no FP8, no softcap). seq_lens_ptr is the engine's seqused_k[B] and
 * query_start_len_ptr its cu_seqlens_q[B+1].
 */
inline Gfx942Unified3dSegmentArguments
    gfx942Unified3dSegmentArguments(int32_t numSeqs,
                                    float scale,
                                    const Gfx942StridedKvArguments& k,
                                    const Gfx942StridedKvArguments& v)
{
    using B = Gfx942Unified3dBuffer;
    using detail::gfx942F32;
    using detail::gfx942I32;
    using detail::gfx942I64;
    using detail::gfx942Pointer;
    return {gfx942Pointer("segm_output_ptr", B::SEGM_OUTPUT),
            gfx942Pointer("segm_max_ptr", B::SEGM_MAX),
            gfx942Pointer("segm_expsum_ptr", B::SEGM_EXPSUM),
            gfx942Pointer("query_ptr", B::QUERY),
            gfx942Pointer("key_cache_ptr", B::KEY),
            gfx942Pointer("value_cache_ptr", B::VALUE),
            gfx942Pointer("sink_ptr", B::NONE),
            gfx942Pointer("block_tables_ptr", B::NONE),
            gfx942Pointer("seq_lens_ptr", B::SEQUSED_K),
            gfx942Pointer("alibi_slopes_ptr", B::NONE),
            gfx942Pointer("qq_bias_ptr", B::NONE),
            gfx942Pointer("query_start_len_ptr", B::CU_SEQLENS_Q),
            gfx942F32("scale", scale),
            gfx942F32("k_scale", 1.0F),
            gfx942F32("v_scale", 1.0F),
            gfx942F32("softcap", 0.0F),
            gfx942I32("num_seqs", numSeqs),
            gfx942I32("block_table_stride", 0),
            gfx942I32("qq_bias_stride_0", 0),
            gfx942I64("k_stride_batch_bytes", k.strideBatchBytes),
            gfx942I64("k_stride_head_bytes", k.strideHeadBytes),
            gfx942I32("k_stride_token_bytes", k.strideTokenBytes),
            gfx942I32("k_span_bytes", k.spanBytes),
            gfx942I64("v_stride_batch_bytes", v.strideBatchBytes),
            gfx942I64("v_stride_head_bytes", v.strideHeadBytes),
            gfx942I32("v_stride_token_bytes", v.strideTokenBytes),
            gfx942I32("v_span_bytes", v.spanBytes)};
}

/// The addresses the pointer arguments take, by Gfx942Unified3dBuffer.
struct Gfx942Unified3dBufferAddresses
{
    const void* segmOutput = nullptr;
    const void* segmMax = nullptr;
    const void* segmExpsum = nullptr;
    const void* query = nullptr;
    const void* key = nullptr;
    const void* value = nullptr;
    const void* sequsedK = nullptr;
    const void* cuSeqlensQ = nullptr;
};

/**
 * @brief One 64-bit word per argument, in order: scalars' @c bits, pointers' addresses
 *        (NONE -> 0). The launch passes each word's address as that argument's
 *        kernelParams entry; the runtime copies the argument's size from the low bytes.
 */
inline std::array<uint64_t, GFX942_UNIFIED_3D_SEGMENT_ARGUMENT_COUNT>
    gfx942Unified3dArgumentWords(const Gfx942Unified3dSegmentArguments& arguments,
                                 const Gfx942Unified3dBufferAddresses& addresses)
{
    static_assert(sizeof(void*) == sizeof(uint64_t), "64-bit pointers");
    using B = Gfx942Unified3dBuffer;
    std::array<uint64_t, GFX942_UNIFIED_3D_SEGMENT_ARGUMENT_COUNT> words{};
    for(size_t i = 0; i < arguments.size(); ++i)
    {
        const auto& argument = arguments[i];
        if(argument.kind != Gfx942Unified3dArgumentKind::POINTER)
        {
            words[i] = argument.bits;
            continue;
        }
        const void* address = nullptr;
        switch(argument.buffer)
        {
        case B::NONE:
        default:
            address = nullptr;
            break;
        case B::SEGM_OUTPUT:
            address = addresses.segmOutput;
            break;
        case B::SEGM_MAX:
            address = addresses.segmMax;
            break;
        case B::SEGM_EXPSUM:
            address = addresses.segmExpsum;
            break;
        case B::QUERY:
            address = addresses.query;
            break;
        case B::KEY:
            address = addresses.key;
            break;
        case B::VALUE:
            address = addresses.value;
            break;
        case B::SEQUSED_K:
            address = addresses.sequsedK;
            break;
        case B::CU_SEQLENS_Q:
            address = addresses.cuSeqlensQ;
            break;
        }
        words[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    }
    return words;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
