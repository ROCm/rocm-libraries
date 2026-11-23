// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"

#include <string>

// Q and O in bf16 or fp16, K and V in fp8
struct FmhaFwdDecodeBf16Fp8
{
};

struct FmhaFwdDecodeFp16Fp8
{
};

template <typename DataType>
struct FmhaFwdDecodeTypeConfig;

template <>
struct FmhaFwdDecodeTypeConfig<FmhaFwdDecodeBf16Fp8>
{
    using QDataType = ck_tile::bf16_t;
    using KDataType = ck_tile::fp8_t;
    using VDataType = ck_tile::fp8_t;
    using ODataType = ck_tile::bf16_t;
};

template <>
struct FmhaFwdDecodeTypeConfig<FmhaFwdDecodeFp16Fp8>
{
    using QDataType = ck_tile::fp16_t;
    using KDataType = ck_tile::fp8_t;
    using VDataType = ck_tile::fp8_t;
    using ODataType = ck_tile::fp16_t;
};

// Decode attention (one query token per sequence) over a paged KV cache, see
// FmhaFwdDecodePagedKVKernel::MakeKargs for the tensors.
struct fmha_fwd_decode_pagedkv_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;

    const int32_t* block_table_ptr;
    ck_tile::index_t block_table_stride;
    const int32_t* seqlen_k_ptr;
    const float* k_descale_ptr;
    const float* v_descale_ptr;

    // fmha_fwd_decode_pagedkv_workspace_bytes / fmha_fwd_decode_pagedkv_num_counters
    void* workspace_ptr;
    int32_t* counters_ptr;

    ck_tile::index_t batch;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    ck_tile::index_t window_size; // keys attended, the query's own included; 0 = no window
    ck_tile::index_t num_groups;  // clusters per (batch, kv head)
    float scale_s;
};

// runtime selection of an instance; wave_split, num_splits and args.num_groups come from
// ck_tile::fmha_fwd_decode_pagedkv_config
struct fmha_fwd_decode_pagedkv_traits
{
    int hdim_q;
    std::string data_type; // of Q and O: bf16 or fp16; K and V are fp8
    int page_block_size;   // tokens per cache page
    bool wave_split;
    int num_splits; // splits (workgroups) per cluster
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
};

// What the kernel does not check and would turn into unwritten rows or out-of-bounds accesses.
inline bool fmha_fwd_decode_pagedkv_is_valid(const fmha_fwd_decode_pagedkv_traits& t,
                                             const fmha_fwd_decode_pagedkv_args& a)
{
    if(a.batch <= 0 || a.nhead_k <= 0 || a.nhead_q <= 0 || a.nhead_q % a.nhead_k != 0 ||
       a.nhead_q / a.nhead_k > 16 || a.block_table_stride <= 0 || a.num_groups <= 0)
        return false;
    if(!a.q_ptr || !a.k_ptr || !a.v_ptr || !a.o_ptr || !a.block_table_ptr || !a.seqlen_k_ptr ||
       !a.k_descale_ptr || !a.v_descale_ptr || !a.workspace_ptr || !a.counters_ptr)
        return false;
    // several clusters merge what each cluster merged: a cluster of one split merges nothing
    if(a.num_groups > 1 && t.num_splits < 2)
        return false;
    return 1 <= t.num_splits && t.num_splits <= ck_tile::kFmhaFwdDecodeMaxClusterSplits;
}

template <typename FmhaKernel>
auto fmha_fwd_decode_pagedkv_create_kargs_and_grids(fmha_fwd_decode_pagedkv_args args)
{
    auto kargs = FmhaKernel::MakeKargs(args.q_ptr,
                                       args.k_ptr,
                                       args.v_ptr,
                                       args.o_ptr,
                                       args.block_table_ptr,
                                       args.block_table_stride,
                                       args.seqlen_k_ptr,
                                       args.k_descale_ptr,
                                       args.v_descale_ptr,
                                       args.workspace_ptr,
                                       args.counters_ptr,
                                       args.batch,
                                       args.nhead_q,
                                       args.nhead_k,
                                       args.window_size,
                                       args.num_groups,
                                       args.scale_s);
    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_k, args.num_groups);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internal kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kPageBlockSize_,
          bool kWaveSplit_,
          ck_tile::index_t kNumSplits_,
          bool kHasGroups_>
struct fmha_fwd_decode_pagedkv_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t kPageBlockSize = kPageBlockSize_;
    static constexpr bool kWaveSplit                 = kWaveSplit_;
    static constexpr ck_tile::index_t kNumSplits     = kNumSplits_;
    static constexpr bool kHasGroups                 = kHasGroups_;
};

template <typename Traits_, typename Arch_ = void>
float fmha_fwd_decode_pagedkv_(const ck_tile::stream_config&, fmha_fwd_decode_pagedkv_args);

// Returns the kernel time (0 unless timed by the stream config), or -1 if no instance matches.
float fmha_fwd_decode_pagedkv(fmha_fwd_decode_pagedkv_traits,
                              fmha_fwd_decode_pagedkv_args,
                              const ck_tile::stream_config&);
