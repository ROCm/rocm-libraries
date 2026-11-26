// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>

#include "ck_tile/core/config.hpp"

namespace ck_tile {

/// @file
/// Pieces of the target model's head shared by the head phases and host references:
/// the logit softcap, the 64-bit argmax key and the DFlash2 accept rule. No device-only code,
/// so a host test can use them as they are.
///
/// Argmax key: (order-preserving bits of the logit) << 32 | (0xFFFFFFFF - token id). The largest
/// key holds the largest logit and, among equal logits, the lowest token id, so an argmax over
/// tiles, workgroups or TP ranks is a plain unsigned 64-bit max. Token ids are global (rank
/// vocab offset included), so a cross-rank argmax is the max of the ranks' keys. Key 0 is below
/// every real key and is the "empty" value of a slot. -0 is folded to +0 so the two tie.

struct MkHeadConstants
{
    static constexpr int32_t kVocab       = 202048;
    static constexpr float kNormEps       = 1e-5f;
    static constexpr float kOutMultiplier = 0.17677669529663687f;
    static constexpr float kSoftcap       = 20.f;
};

/// How the softcap is evaluated.
///   hf_bf16: as the HF reference does, every intermediate rounded to bf16:
///            bf16(bf16(tanh(bf16(bf16(bf16(acc) * mult) / T))) * T). Many logits tie in bf16, so
///            the lowest-id tie rule decides often.
///   fp32:    T * tanh(acc * mult / T) in fp32, no rounding.
enum struct head_softcap_numerics
{
    hf_bf16,
    fp32
};

/// fp32 -> bf16 (round to nearest even, NaN kept quiet) -> fp32, without the ck_tile bf16 type.
CK_TILE_HOST_DEVICE float head_round_bf16(float x)
{
    const uint32_t u = __builtin_bit_cast(uint32_t, x);
    if((u & 0x7FFFFFFFu) > 0x7F800000u)
        return __builtin_bit_cast(float, (u | 0x00400000u) & 0xFFFF0000u);
    const uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return __builtin_bit_cast(float, r);
}

template <head_softcap_numerics Numerics>
CK_TILE_HOST_DEVICE float head_softcap(float acc, float mult, float cap)
{
    if constexpr(Numerics == head_softcap_numerics::hf_bf16)
    {
        float v = head_round_bf16(acc);
        v       = head_round_bf16(v * mult);
        v       = head_round_bf16(v / cap);
        v       = head_round_bf16(tanhf(v));
        return head_round_bf16(v * cap);
    }
    else
    {
        return cap * tanhf(acc * (mult / cap));
    }
}

/// Order-preserving map of fp32 bits to uint32 (a < b as floats => key(a) < key(b)).
CK_TILE_HOST_DEVICE uint32_t head_ordered_bits(float v)
{
    const uint32_t u = __builtin_bit_cast(uint32_t, v + 0.f); // -0 + 0 = +0
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

CK_TILE_HOST_DEVICE float head_unordered_bits(uint32_t k)
{
    const uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    return __builtin_bit_cast(float, u);
}

CK_TILE_HOST_DEVICE uint64_t head_argmax_key(float value, int32_t token)
{
    return (static_cast<uint64_t>(head_ordered_bits(value)) << 32) |
           static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(token));
}

CK_TILE_HOST_DEVICE int32_t head_key_token(uint64_t key)
{
    return static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(key));
}

CK_TILE_HOST_DEVICE float head_key_value(uint64_t key)
{
    return head_unordered_bits(static_cast<uint32_t>(key >> 32));
}

/// DFlash2 accept of one verify window of @p window rows. Row i's argmax target[i] is the target
/// model's token for position i + 1; draft[i] (i < window - 1) is the drafter's token for the same
/// position. Accepts the longest prefix with draft[i] == target[i] and emits it plus one bonus
/// token, i.e. out[0..k] = target[0..k], fills out[k + 1..window) with -1 and returns k + 1
/// (between 1 and window).
CK_TILE_HOST_DEVICE int32_t head_dflash_accept(const int32_t* target,
                                               const int32_t* draft,
                                               int32_t window,
                                               int32_t* out)
{
    int32_t k = 0;
    while(k < window - 1 && draft[k] == target[k])
    {
        out[k] = target[k];
        ++k;
    }
    out[k] = target[k];
    for(int32_t j = k + 1; j < window; ++j)
        out[j] = -1;
    return k + 1;
}

} // namespace ck_tile
