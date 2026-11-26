// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @file
/// Output functors of DecodeAttentionPhase. The phase's combine step hands one thread one whole
/// (token, q head) row of the final attention output, in fp32 and already normalised over all KV
/// splits, so a per-head epilogue (o_norm, gate) needs no cross-thread reduction.
///
/// A functor type provides:
/// @code
///   using Args;   // trivially copyable, embedded in the phase Args
///   template <index_t D>
///   CK_TILE_DEVICE static void Apply(const Args&, index_t token, index_t q_head,
///                                    array<float, D>& o);   // stores the row
/// @endcode
/// token is the row of the phase's Q input (and of the output); q_head is the TP rank's local head.

namespace detail {
template <typename T, index_t N>
CK_TILE_DEVICE void attention_store_row(T* dst, const array<float, N>& v)
{
    static_assert(std::is_same_v<T, bf16_t>, "attention output rows are bf16");
    static_assert(N % 8 == 0);
    // 8 bf16 per 16-byte store.
    static_for<0, N / 8, 1>{}([&](auto c) {
        ext_vector_t<uint32_t, 4> pk;
        static_for<0, 4, 1>{}([&](auto k) {
            const uint32_t lo =
                bit_cast<uint16_t>(float_to_bf16<bf16_rounding_mode::standard>(v[c * 8 + 2 * k]));
            const uint32_t hi = bit_cast<uint16_t>(
                float_to_bf16<bf16_rounding_mode::standard>(v[c * 8 + 2 * k + 1]));
            pk[k.value] = lo | (hi << 16);
        });
        *static_cast<ext_vector_t<uint32_t, 4>*>(static_cast<void*>(dst + c * 8)) = pk;
    });
}

CK_TILE_DEVICE float attention_round_bf16(float x)
{
    return type_convert<float>(float_to_bf16<bf16_rounding_mode::standard>(x));
}
} // namespace detail

/// Stores the attention output as bf16: out[token * stride_token + q_head * D + d].
struct AttentionOutputStore
{
    struct Args
    {
        bf16_t* out;
        index_t stride_token;
    };

    template <index_t D>
    CK_TILE_DEVICE static void
    Apply(const Args& a, index_t token, index_t q_head, array<float, D>& o)
    {
        detail::attention_store_row(
            a.out + static_cast<long_index_t>(token) * a.stride_token + q_head * D, o);
    }
};

/// Reference fusion of the target model's attention epilogue (placeholder until the epilogue
/// workstream provides its own functor): o = bf16(attn); o = bf16(RMSNorm(o, eps), no weight);
/// out = bf16(o * bf16(sigmoid(gate))). gate[token * stride_gate + q_head * D + d] is bf16, the
/// gate_proj output column block of the same head.
struct AttentionOutputNormGate
{
    struct Args
    {
        bf16_t* out;
        index_t stride_token;
        const bf16_t* gate;
        index_t stride_gate;
        float eps; // 1e-5
    };

    template <index_t D>
    CK_TILE_DEVICE static void
    Apply(const Args& a, index_t token, index_t q_head, array<float, D>& o)
    {
        float ss = 0.f;
        static_for<0, D, 1>{}([&](auto d) {
            o[d] = detail::attention_round_bf16(o[d]);
            ss += o[d] * o[d];
        });
        const float r   = rsqrtf(ss / D + a.eps);
        const bf16_t* g = a.gate + static_cast<long_index_t>(token) * a.stride_gate + q_head * D;
        static_for<0, D / 8, 1>{}([&](auto c) {
            const auto gv =
                *static_cast<const ext_vector_t<uint32_t, 4>*>(static_cast<const void*>(g + c * 8));
            static_for<0, 8, 1>{}([&](auto k) {
                const uint32_t bits = (gv[k.value / 2] >> (16 * (k.value % 2))) & 0xFFFFu;
                const float gate    = bit_cast<float>(bits << 16);
                const float sg      = detail::attention_round_bf16(1.f / (1.f + __expf(-gate)));
                const index_t d     = c * 8 + k;
                o[d]                = detail::attention_round_bf16(o[d] * r) * sg;
            });
        });
        detail::attention_store_row(
            a.out + static_cast<long_index_t>(token) * a.stride_token + q_head * D, o);
    }
};

} // namespace ck_tile
