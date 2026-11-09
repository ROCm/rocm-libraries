// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

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
bool fmha_fwd_cr_supported_args(const Args& a)
{
    constexpr auto limit = std::numeric_limits<int32_t>::max();
    if(!a.q_ptr || !a.k_ptr || !a.v_ptr || !a.o_ptr || a.batch <= 0 || a.nhead_q <= 0 ||
       a.nhead_q != a.nhead_k || a.hdim_q != 128 || a.hdim_v != 128 || a.seqlen_q < 128 ||
       a.seqlen_k <= 0 || !std::isfinite(a.scale_s) || a.scale_s <= 0 || a.sink_ptr ||
       a.seqstart_q_ptr || a.seqstart_k_ptr || a.seqlen_q_ptr || a.seqlen_k_ptr ||
       a.cu_seqlen_q_ptr || a.cu_seqlen_k_ptr)
        return false;
    const int64_t stride = int64_t{a.nhead_q} * 128;
    const int64_t qbatch = stride * a.seqlen_q;
    const int64_t kbatch = stride * a.seqlen_k;
    const int64_t heads  = int64_t{a.batch} * a.nhead_q;
    if(qbatch > limit || kbatch > limit || heads * ((int64_t{a.seqlen_q} + 127) / 128) > limit ||
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
