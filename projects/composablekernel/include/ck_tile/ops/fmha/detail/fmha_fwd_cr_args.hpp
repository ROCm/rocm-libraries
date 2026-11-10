// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <initializer_list>

namespace ck_tile::detail {

// Host-only view used to validate native kernel arguments with the same rules
// as the uncompressed API input. This type is never a device argument ABI.
struct FmhaFwdCrInput
{
    const void *q_ptr{}, *k_ptr{}, *v_ptr{};
    void *o_ptr{}, *lse_ptr{};
    const void *sink_ptr{}, *seqstart_q_ptr{}, *seqstart_k_ptr{};
    const void *seqlen_q_ptr{}, *seqlen_k_ptr{}, *cu_seqlen_q_ptr{}, *cu_seqlen_k_ptr{};
    int batch{}, nhead_q{}, nhead_k{}, seqlen_q{}, seqlen_k{}, max_seqlen_q{};
    int hdim_q{}, hdim_v{}, mask_type{}, window_size_left{}, window_size_right{};
    int64_t stride_q{}, stride_k{}, stride_v{}, stride_o{};
    int64_t nhead_stride_q{}, nhead_stride_k{}, nhead_stride_v{}, nhead_stride_o{},
        nhead_stride_lse{};
    int64_t batch_stride_q{}, batch_stride_k{}, batch_stride_v{}, batch_stride_o{},
        batch_stride_lse{};
    float scale_s{};
};

// Fixed tuning variants take priority over CK's existing dispatch heuristics.
// Generic CR stays available as a fallback, including CR-only filtered builds.
template <typename Args>
bool fmha_fwd_cr_has_static_tuning(const Args& a)
{
    return a.batch == 32 && a.nhead_q == 8 && a.seqlen_q == a.seqlen_k && a.seqlen_q >= 512 &&
           a.seqlen_q <= 32768 && (a.seqlen_q & (a.seqlen_q - 1)) == 0;
}

// Shared by generated dispatch and direct launches. The compact device ABI
// intentionally represents contiguous BSHD only; validate all layout metadata
// before discarding it. Unsupported layouts/features retain the CK fallback.
template <typename Args>
bool fmha_fwd_cr_compact_args(const Args& a)
{
    constexpr auto limit = std::numeric_limits<int32_t>::max();
    if(!a.q_ptr || !a.k_ptr || !a.v_ptr || !a.o_ptr || a.batch <= 0 || a.nhead_q <= 0 ||
       a.nhead_q != a.nhead_k || a.hdim_q != 128 || a.hdim_v != 128 || a.seqlen_q < 128 ||
       a.seqlen_k <= 0 || !std::isfinite(a.scale_s) || a.scale_s <= 0 || a.sink_ptr ||
       a.seqstart_q_ptr || a.seqstart_k_ptr || a.seqlen_q_ptr || a.seqlen_k_ptr ||
       a.cu_seqlen_q_ptr || a.cu_seqlen_k_ptr)
        return false;
    const int64_t stride = int64_t{a.nhead_q} * 128;
    if(stride > limit)
        return false;
    const int64_t qbatch = stride * a.seqlen_q;
    const int64_t kbatch = stride * a.seqlen_k;
    const int64_t heads  = int64_t{a.batch} * a.nhead_q;
    if(qbatch > limit || kbatch > limit || heads > limit / ((int64_t{a.seqlen_q} + 127) / 128) ||
       a.stride_q != stride || a.stride_k != stride || a.stride_v != stride ||
       a.stride_o != stride || a.nhead_stride_q != 128 || a.nhead_stride_k != 128 ||
       a.nhead_stride_v != 128 || a.nhead_stride_o != 128 || a.batch_stride_q != qbatch ||
       a.batch_stride_o != qbatch || a.batch_stride_k != kbatch || a.batch_stride_v != kbatch)
        return false;
    if((reinterpret_cast<uintptr_t>(a.q_ptr) | reinterpret_cast<uintptr_t>(a.k_ptr) |
        reinterpret_cast<uintptr_t>(a.v_ptr) | reinterpret_cast<uintptr_t>(a.o_ptr)) &
       15)
        return false;
    if(a.lse_ptr &&
       (a.nhead_stride_lse != a.seqlen_q || a.batch_stride_lse != int64_t{a.nhead_q} * a.seqlen_q ||
        reinterpret_cast<uintptr_t>(a.lse_ptr) % alignof(float)))
        return false;
    // GenericAttentionMaskEnum: 0=dense, 1=top-left, 2=bottom-right.
    // Square bottom-right causal is identical to top-left causal.
    return a.mask_type == 0 ||
           ((a.mask_type == 1 || (a.mask_type == 2 && a.seqlen_q == a.seqlen_k)) &&
            a.window_size_left < 0 && a.window_size_right == 0);
}

// General tensor views retain contiguous head elements and vector-aligned rows.
template <typename Args>
bool fmha_fwd_cr_supported_args(const Args& a)
{
    const bool group = a.seqstart_q_ptr != nullptr;
    if(!a.q_ptr || !a.o_ptr || ((group || a.seqlen_k > 0) && (!a.k_ptr || !a.v_ptr)) ||
       a.batch <= 0 || a.nhead_q <= 0 || a.nhead_k <= 0 || a.nhead_q % a.nhead_k ||
       a.seqlen_q < 0 || a.seqlen_k < 0 || a.max_seqlen_q < 1 || !std::isfinite(a.scale_s) ||
       a.scale_s <= 0 || a.sink_ptr || group != (a.seqstart_k_ptr != nullptr) || a.mask_type < 0 ||
       a.mask_type > 3)
        return false;
    // Batch mode represents logical lengths with cumulative pointers only.
    // Per-sequence length pointers are part of the group-mode contract.
    if(!group && (a.seqlen_q_ptr || a.seqlen_k_ptr))
        return false;
    if(!((a.hdim_q == 64 && a.hdim_v == 64) || (a.hdim_q == 128 && a.hdim_v == 128) ||
         (a.hdim_q == 192 && a.hdim_v == 128)))
        return false;
    // Tile coordinates and the swizzled linear grid use signed 32-bit indices.
    // Check the smallest tile so both generated geometries are safe to launch.
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    const int64_t max_q     = group ? a.max_seqlen_q : a.seqlen_q;
    const int64_t heads     = int64_t{a.batch} * a.nhead_q;
    if(max_q < 1 || max_q > limit - 128 || a.seqlen_k > limit - 128 ||
       heads > limit / ((max_q + 63) / 64) || a.nhead_q > limit / a.hdim_q)
        return false;
    for(auto ptr : {a.seqstart_q_ptr,
                    a.seqstart_k_ptr,
                    a.seqlen_q_ptr,
                    a.seqlen_k_ptr,
                    a.cu_seqlen_q_ptr,
                    a.cu_seqlen_k_ptr})
        if(reinterpret_cast<uintptr_t>(ptr) % alignof(int32_t))
            return false;
    for(auto stride : {a.stride_q, a.stride_k, a.stride_v, a.stride_o})
        if(stride < 0 || stride > limit || stride % 8)
            return false;
    for(auto stride : {a.nhead_stride_q, a.nhead_stride_k, a.nhead_stride_v, a.nhead_stride_o})
        if(stride < 0 || stride % 8)
            return false;
    if(!group)
        for(auto stride : {a.batch_stride_q, a.batch_stride_k, a.batch_stride_v, a.batch_stride_o})
            if(stride < 0 || stride % 8)
                return false;
    if((reinterpret_cast<uintptr_t>(a.q_ptr) | reinterpret_cast<uintptr_t>(a.k_ptr) |
        reinterpret_cast<uintptr_t>(a.v_ptr) | reinterpret_cast<uintptr_t>(a.o_ptr)) &
       15)
        return false;
    return !a.lse_ptr || (reinterpret_cast<uintptr_t>(a.lse_ptr) % alignof(float) == 0 &&
                          a.nhead_stride_lse >= 0 && a.batch_stride_lse >= 0);
}

} // namespace ck_tile::detail
