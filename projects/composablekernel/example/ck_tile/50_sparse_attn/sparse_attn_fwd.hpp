// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/sparse_attn/sparge_hyperparam.hpp"

#include "../01_fmha/mask.hpp"

#include <type_traits>
#include <utility>
#include <variant>

namespace ck_tile {
inline bool is_load_tr_supported() { return is_gfx95_supported(); }
} // namespace ck_tile

struct FmhaSparseFwdFp16
{
};

struct FmhaSparseFwdBf16
{
};

template <typename DataType>
struct FmhaSparseFwdTypeConfig;

template <>
struct FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>
{
    using QDataType           = ck_tile::half_t;
    using KDataType           = ck_tile::half_t;
    using VDataType           = ck_tile::half_t;
    using SaccDataType        = float;
    using SMPLComputeDataType = float;
    using PDataType           = ck_tile::half_t;
    using OaccDataType        = float;
    using ODataType           = ck_tile::half_t;
    // Required by BlockFmhaPipelineProblem but unused.
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
};

template <>
struct FmhaSparseFwdTypeConfig<FmhaSparseFwdBf16>
{
    using QDataType           = ck_tile::bf16_t;
    using KDataType           = ck_tile::bf16_t;
    using VDataType           = ck_tile::bf16_t;
    using SaccDataType        = float;
    using SMPLComputeDataType = float;
    using PDataType           = ck_tile::bf16_t;
    using OaccDataType        = float;
    using ODataType           = ck_tile::bf16_t;
    // Required by BlockFmhaPipelineProblem but unused.
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

enum class sparse_attn_mode
{
    batch = 0,
    group = 1
};

struct fmha_jenga_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    // batch: one-hot block map [B,H,Q_blk,K_blk] (1=active)
    // group: fully packed [H, sum_b(q_blocks[b]*k_blocks[b])]
    const void* block_relation_onehot_ptr;
    void* o_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    // Group-mode cu_seqlens (all nullptr in batch mode). cu_seqlen_*_ptr reserved
    // for ABI parity with 01_fmha; ignored by kernel today (no padded varlen).
    const void* seqstart_q_ptr   = nullptr;
    const void* seqstart_k_ptr   = nullptr;
    const void* seqlen_q_ptr     = nullptr;  // optional override of seqstart_*_ptr deltas
    const void* seqlen_k_ptr     = nullptr;
    const void* cu_seqlen_q_ptr  = nullptr;
    const void* cu_seqlen_k_ptr  = nullptr;

    // Group-mode block offset tables.
    const void* seqstart_q_block_ptr   = nullptr;
    const void* mask_batch_offset_ptr  = nullptr;

    // ALIBI: bias_ptr = [nhead] (rank=1) or [1] (rank=0) slope buffer.
    const void*      bias_ptr           = nullptr;
    ck_tile::index_t stride_bias        = 0;
    ck_tile::index_t nhead_stride_bias  = 0;
    ck_tile::index_t batch_stride_bias  = 0;

    float logits_soft_cap = 0.0f;  // NO_BIAS only
};

struct fmha_vsa_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    // batch: int32 LUT [B,H,Q_blk,K_blk] (delta-encoded K-block indices)
    // group: fully packed [H, sum_b(q_blocks[b]*k_blocks[b])]
    const void* lut_ptr;
    // batch: int32 valid count [B,H,Q_blk]
    // group: fully packed [H, sum_b(q_blocks[b])] indexed via seqstart_q_block_ptr
    const void* valid_block_num_ptr;
    void* o_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    // Group-mode cu_seqlens (all nullptr in batch mode). cu_seqlen_*_ptr reserved
    // for ABI parity with 01_fmha; ignored by kernel today (no padded varlen).
    const void* seqstart_q_ptr   = nullptr;
    const void* seqstart_k_ptr   = nullptr;
    const void* seqlen_q_ptr     = nullptr;  // optional override of seqstart_*_ptr deltas
    const void* seqlen_k_ptr     = nullptr;
    const void* cu_seqlen_q_ptr  = nullptr;
    const void* cu_seqlen_k_ptr  = nullptr;

    // Group-mode block offset tables. seqstart_q_block_ptr also offsets valid_block_num_ptr.
    const void* seqstart_q_block_ptr = nullptr;
    const void* lut_batch_offset_ptr = nullptr;

    const void*      bias_ptr           = nullptr;
    ck_tile::index_t stride_bias        = 0;
    ck_tile::index_t nhead_stride_bias  = 0;
    ck_tile::index_t batch_stride_bias  = 0;

    float logits_soft_cap = 0.0f;
};

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_jenga_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&]() {
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.block_relation_onehot_ptr,
                                         args.o_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_o,
                                         static_cast<const int32_t*>(args.seqstart_q_ptr),
                                         static_cast<const int32_t*>(args.seqstart_k_ptr),
                                         static_cast<const int32_t*>(args.seqlen_q_ptr),
                                         static_cast<const int32_t*>(args.seqlen_k_ptr),
                                         static_cast<const int32_t*>(args.seqstart_q_block_ptr),
                                         static_cast<const int32_t*>(args.mask_batch_offset_ptr),
                                         args.batch,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         args.bias_ptr,
                                         args.stride_bias,
                                         args.nhead_stride_bias,
                                         args.batch_stride_bias,
                                         args.logits_soft_cap);
        }
        else
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.block_relation_onehot_ptr,
                                         args.o_ptr,
                                         args.seqlen_q,
                                         args.seqlen_k,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         args.bias_ptr,
                                         args.stride_bias,
                                         args.nhead_stride_bias,
                                         args.batch_stride_bias,
                                         args.logits_soft_cap);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_vsa_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&]() {
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.lut_ptr,
                                         args.valid_block_num_ptr,
                                         args.o_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_o,
                                         static_cast<const int32_t*>(args.seqstart_q_ptr),
                                         static_cast<const int32_t*>(args.seqstart_k_ptr),
                                         static_cast<const int32_t*>(args.seqlen_q_ptr),
                                         static_cast<const int32_t*>(args.seqlen_k_ptr),
                                         static_cast<const int32_t*>(args.seqstart_q_block_ptr),
                                         static_cast<const int32_t*>(args.lut_batch_offset_ptr),
                                         args.batch,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         args.bias_ptr,
                                         args.stride_bias,
                                         args.nhead_stride_bias,
                                         args.batch_stride_bias,
                                         args.logits_soft_cap);
        }
        else
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.lut_ptr,
                                         args.valid_block_num_ptr,
                                         args.o_ptr,
                                         args.seqlen_q,
                                         args.seqlen_k,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         args.bias_ptr,
                                         args.stride_bias,
                                         args.nhead_stride_bias,
                                         args.batch_stride_bias,
                                         args.logits_soft_cap);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

// Pattern-matching trait — not used to instantiate the kernel directly.
template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kIsGroupMode_ = false,
          ck_tile::BlockAttentionBiasEnum BiasEnum_ = ck_tile::BlockAttentionBiasEnum::NO_BIAS>
struct fmha_jenga_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kUseTrLoad                 = kUseTrLoad_;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr auto BiasEnum                   = BiasEnum_;
};

struct fmha_jenga_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_v_rowmajor;
    mask_enum mask_type;
    bool is_group_mode = false;
    int  bias_type     = 0;      // 0=no_bias, 1=elementwise, 2=alibi
    bool has_logits_soft_cap = false;  // NO_BIAS only
};

float fmha_jenga_fwd(fmha_jenga_fwd_traits, fmha_jenga_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_jenga_fwd_(const ck_tile::stream_config&, fmha_jenga_fwd_args);

float fmha_jenga_fwd(fmha_jenga_fwd_args, const ck_tile::stream_config&);

// Sparge: preprocess → mask prediction → attention.
struct fmha_sparge_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    // Mask prediction hard-codes 1/sqrt(hdim_q); divergent scale_s desyncs LUT ranking.
    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    // Preprocess block size (BLKK = BLKQ); attention uses its own compile-time tile.
    ck_tile::index_t pp_block_size = 0;

    ck_tile::index_t causal_type = 0;
    bool attention_sink          = false;

    const void* bias_ptr             = nullptr;
    ck_tile::index_t stride_bias       = 0;
    ck_tile::index_t nhead_stride_bias = 0;
    ck_tile::index_t batch_stride_bias = 0;

    float logits_soft_cap = 0.0f;

    // Internal — codegen overwrites.
    const void* internal_lut_ptr = nullptr;
    const void* internal_vbn_ptr = nullptr;

    void* workspace_ptr = nullptr;

    float* sparsity_out = nullptr;

    ck_tile::sparge_hyperparam_args hp{};

    // Group-mode cu_seqlens (all nullptr in batch mode). cu_seqlen_*_ptr reserved
    // for ABI parity with 01_fmha; ignored by kernel today (no padded varlen).
    const void* seqstart_q_ptr   = nullptr;
    const void* seqstart_k_ptr   = nullptr;
    const void* seqlen_q_ptr     = nullptr;  // optional override of seqstart_*_ptr deltas
    const void* seqlen_k_ptr     = nullptr;
    const void* cu_seqlen_q_ptr  = nullptr;
    const void* cu_seqlen_k_ptr  = nullptr;

    // Group-mode block offset tables. seqstart_q_block_ptr offsets q_means/q_sim/vbn;
    // seqstart_k_block_ptr offsets k_means/k_sim.
    const void* seqstart_q_block_ptr = nullptr;
    const void* seqstart_k_block_ptr = nullptr;
    const void* lut_batch_offset_ptr = nullptr;

    // Group-mode block totals (zero in batch mode); codegen uses these to size workspace.
    ck_tile::index_t total_q_blocks  = 0;
    ck_tile::index_t total_k_blocks  = 0;
    ck_tile::index_t total_qk_blocks = 0;
};

// Sparge workspace layout: [km | k_means | q_means | k_sim | q_sim | LUT | vbn].
struct sparge_workspace_layout
{
    std::size_t km_off,      km_bytes;
    std::size_t k_means_off, k_means_bytes;
    std::size_t q_means_off, q_means_bytes;
    std::size_t k_sim_off,   k_sim_bytes;
    std::size_t q_sim_off,   q_sim_bytes;
    std::size_t lut_off,     lut_bytes;
    std::size_t vbn_off,     vbn_bytes;
    std::size_t total_bytes;
};

inline sparge_workspace_layout compute_sparge_workspace_layout(const fmha_sparge_fwd_args& a)
{
    const auto B   = static_cast<std::size_t>(a.batch);
    const auto Hk  = static_cast<std::size_t>(a.nhead_k);
    const auto Hq  = static_cast<std::size_t>(a.nhead_q);
    const auto D   = static_cast<std::size_t>(a.hdim_q);
    const auto BS  = a.pp_block_size;
    const auto N_k = static_cast<std::size_t>((a.seqlen_k + BS - 1) / BS);
    const auto N_q = static_cast<std::size_t>((a.seqlen_q + BS - 1) / BS);
    const bool sim = (a.hp.simthreshold > 0.0f);

    sparge_workspace_layout L{};
    std::size_t off = 0;
    L.km_off      = off; L.km_bytes      = a.hp.smooth_k ? B * Hk * D * sizeof(float) : 0; off += L.km_bytes;
    L.k_means_off = off; L.k_means_bytes = B * Hk * N_k * D * sizeof(float);                off += L.k_means_bytes;
    L.q_means_off = off; L.q_means_bytes = B * Hq * N_q * D * sizeof(float);                off += L.q_means_bytes;
    L.k_sim_off   = off; L.k_sim_bytes   = sim ? B * Hk * N_k * sizeof(float) : 0;          off += L.k_sim_bytes;
    L.q_sim_off   = off; L.q_sim_bytes   = sim ? B * Hq * N_q * sizeof(float) : 0;          off += L.q_sim_bytes;
    L.lut_off     = off; L.lut_bytes     = B * Hq * N_q * N_k * sizeof(int32_t);            off += L.lut_bytes;
    L.vbn_off     = off; L.vbn_bytes     = B * Hq * N_q * sizeof(int32_t);                  off += L.vbn_bytes;
    L.total_bytes = off;
    return L;
}

// Group-mode shapes (head-major, packed by per-batch block totals; matches VSA):
//   km       : [B, Hk, D]                              (smooth_k only)
//   k_means  : [Hk, total_k_blocks, D]
//   q_means  : [Hq, total_q_blocks, D]
//   k_sim    : [Hk, total_k_blocks]
//   q_sim    : [Hq, total_q_blocks]
//   LUT      : [Hq, sum_b(q_blocks[b] * k_blocks[b])]  int32
//   vbn      : [Hq, total_q_blocks]                    int32
inline sparge_workspace_layout compute_sparge_workspace_layout_group(
    const fmha_sparge_fwd_args& a,
    int32_t total_q_blocks,
    int32_t total_k_blocks,
    int32_t total_qk_blocks)
{
    const auto B   = static_cast<std::size_t>(a.batch);
    const auto Hk  = static_cast<std::size_t>(a.nhead_k);
    const auto Hq  = static_cast<std::size_t>(a.nhead_q);
    const auto D   = static_cast<std::size_t>(a.hdim_q);
    const auto Tk  = static_cast<std::size_t>(total_k_blocks);
    const auto Tq  = static_cast<std::size_t>(total_q_blocks);
    const auto Tqk = static_cast<std::size_t>(total_qk_blocks);
    const bool sim = (a.hp.simthreshold > 0.0f);

    sparge_workspace_layout L{};
    std::size_t off = 0;
    L.km_off      = off; L.km_bytes      = a.hp.smooth_k ? B * Hk * D * sizeof(float) : 0; off += L.km_bytes;
    L.k_means_off = off; L.k_means_bytes = Hk * Tk * D * sizeof(float);                    off += L.k_means_bytes;
    L.q_means_off = off; L.q_means_bytes = Hq * Tq * D * sizeof(float);                    off += L.q_means_bytes;
    L.k_sim_off   = off; L.k_sim_bytes   = sim ? Hk * Tk * sizeof(float) : 0;              off += L.k_sim_bytes;
    L.q_sim_off   = off; L.q_sim_bytes   = sim ? Hq * Tq * sizeof(float) : 0;              off += L.q_sim_bytes;
    L.lut_off     = off; L.lut_bytes     = Hq * Tqk * sizeof(int32_t);                     off += L.lut_bytes;
    L.vbn_off     = off; L.vbn_bytes     = Hq * Tq * sizeof(int32_t);                      off += L.vbn_bytes;
    L.total_bytes = off;
    return L;
}


template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_sparge_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&]() {
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.o_ptr,
                                         args.internal_lut_ptr,
                                         args.internal_vbn_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_o,
                                         static_cast<const int32_t*>(args.seqstart_q_ptr),
                                         static_cast<const int32_t*>(args.seqstart_k_ptr),
                                         static_cast<const int32_t*>(args.seqlen_q_ptr),
                                         static_cast<const int32_t*>(args.seqlen_k_ptr),
                                         static_cast<const int32_t*>(args.seqstart_q_block_ptr),
                                         static_cast<const int32_t*>(args.lut_batch_offset_ptr),
                                         args.batch,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         args.hp.pvthreshd,
                                         args.hp.pvthreshd_per_head_ptr,
                                         args.bias_ptr,
                                         args.stride_bias,
                                         args.nhead_stride_bias,
                                         args.batch_stride_bias,
                                         args.logits_soft_cap);
        }
        else
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.o_ptr,
                                         args.internal_lut_ptr,
                                         args.internal_vbn_ptr,
                                         args.seqlen_q,
                                         args.seqlen_k,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q,
                                         args.nhead_q / args.nhead_k,
                                         args.scale_s,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         args.hp.pvthreshd,
                                         args.hp.pvthreshd_per_head_ptr,
                                         args.bias_ptr,
                                         args.stride_bias,
                                         args.nhead_stride_bias,
                                         args.batch_stride_bias,
                                         args.logits_soft_cap);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kIsGroupMode_ = false,
          ck_tile::BlockAttentionBiasEnum BiasEnum_ = ck_tile::BlockAttentionBiasEnum::NO_BIAS>
struct fmha_sparge_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr auto BiasEnum                   = BiasEnum_;
};

using fmha_sparge_fwd_traits = fmha_jenga_fwd_traits;

float fmha_sparge_fwd(fmha_sparge_fwd_traits, fmha_sparge_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_sparge_fwd_(const ck_tile::stream_config&, fmha_sparge_fwd_args);

template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kIsGroupMode_ = false,
          ck_tile::BlockAttentionBiasEnum BiasEnum_ = ck_tile::BlockAttentionBiasEnum::NO_BIAS>
using fmha_vsa_fwd_traits_ = fmha_jenga_fwd_traits_<HDim_,
                                                    DataType_,
                                                    kM0_,
                                                    kN0_,
                                                    kK0_,
                                                    kN1_,
                                                    kK1_,
                                                    kK0BlockLength_,
                                                    kIsVLayoutRowMajor_,
                                                    FmhaPipelineEnum_,
                                                    kHasLogitsSoftCap_,
                                                    FmhaMask_,
                                                    kPadS_,
                                                    kPadSK_,
                                                    kPadD_,
                                                    kPadDv_,
                                                    kUseTrLoad_,
                                                    kIsGroupMode_,
                                                    BiasEnum_>;

using fmha_vsa_fwd_traits = fmha_jenga_fwd_traits;

float fmha_vsa_fwd(fmha_vsa_fwd_traits, fmha_vsa_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_vsa_fwd_(const ck_tile::stream_config&, fmha_vsa_fwd_args);

float fmha_vsa_fwd(fmha_vsa_fwd_args, const ck_tile::stream_config&);
