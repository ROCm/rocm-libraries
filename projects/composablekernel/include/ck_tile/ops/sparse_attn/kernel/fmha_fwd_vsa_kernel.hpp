// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdVSAKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kHasDropout       = FmhaPipeline::kHasDropout;
    static constexpr auto QScaleEnum        = FmhaPipeline::Problem::QScaleEnum;
    static constexpr bool kDoFp8StaticQuant =
        (QScaleEnum != ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE);
    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kHasBias     = (BiasEnum != BlockAttentionBiasEnum::NO_BIAS);
    static_assert(!kStoreLSE, "VSA sparse attention does not support LSE output.");
    static_assert(!kHasDropout, "VSA sparse attention does not support dropout.");
    static_assert(!kDoFp8StaticQuant,
                  "VSA sparse attention does not support FP8 static quantization yet.");

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    static constexpr bool kUseAsyncCopy = FmhaPipeline::Policy::AsyncCopy;

    template <ck_tile::index_t I> // distinct template arg avoids duplicated base class problem
    struct FmhaFwdEmptyKargs
    {
    };

    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lut_ptr;
        const void* valid_block_num_ptr;
        void* o_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk; // nhead_q / nhead_k; >1 = MQA/GQA
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;
    };

    struct FmhaFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaFwdBiasKargs
    {
        const void* bias_ptr;
        ck_tile::index_t stride_bias;
        ck_tile::index_t nhead_stride_bias;
        ck_tile::index_t batch_stride_bias;
    };

    struct FmhaFwdLogitsSoftCapKargs
    {
        float logits_soft_cap;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kHasBias, FmhaFwdBiasKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<3>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;
    };

    // Group / varlen: mask_batch_offset_ptr is [B+1] cumulative q_blocks*k_blocks per batch;
    // seqstart_q_block_ptr doubles as the valid_block_num offset table.
    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kHasBias, FmhaFwdBiasKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;
        const int32_t* seqlen_k_ptr;
        const int32_t* seqstart_q_block_ptr;
        const int32_t* mask_batch_offset_ptr;
        ck_tile::index_t batch;
    };

    using Kargs =
        std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* lut_ptr,
              const void* valid_block_num_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              const void* bias_ptr               = nullptr,
              ck_tile::index_t stride_bias       = 0,
              ck_tile::index_t nhead_stride_bias = 0,
              ck_tile::index_t batch_stride_bias = 0,
              float logits_soft_cap              = 0.0f)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lut_ptr,
                     valid_block_num_ptr,
                     o_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o},
                    {},
                    {},
                    {},
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.logits_soft_cap = logits_soft_cap;
        }
        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* lut_ptr,
              const void* valid_block_num_ptr,
              void* o_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_o,
              const int32_t* seqstart_q_ptr,
              const int32_t* seqstart_k_ptr,
              const int32_t* seqlen_q_ptr,
              const int32_t* seqlen_k_ptr,
              const int32_t* seqstart_q_block_ptr,
              const int32_t* mask_batch_offset_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              const void* bias_ptr               = nullptr,
              ck_tile::index_t stride_bias       = 0,
              ck_tile::index_t nhead_stride_bias = 0,
              ck_tile::index_t batch_stride_bias = 0,
              float logits_soft_cap              = 0.0f)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lut_ptr,
                     valid_block_num_ptr,
                     o_ptr,
                     /*seqlen_q*/ 0,
                     /*seqlen_k*/ 0,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o},
                    {},
                    {},
                    {},
                    seqstart_q_ptr,
                    seqstart_k_ptr,
                    seqlen_q_ptr,
                    seqlen_k_ptr,
                    seqstart_q_block_ptr,
                    mask_batch_offset_ptr,
                    batch};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.logits_soft_cap = logits_soft_cap;
        }
        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        return dim3(ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                        ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1),
                    nhead_,
                    batch_size_);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        // Masked M-tile reversal below assumes a single N1 tile spans hdim_v (num_tile_n1 == 1),
        // i.e. hdim_v <= kN1.
        static_assert(FmhaPipeline::kN1 >= FmhaPipeline::kQKHeaddim,
                      "vsa masked M-tile reversal assumes a single N1 tile "
                      "(hdim_v <= kN1)");
        const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
            return ck_tile::make_tuple(quotient, modulus);
        };

        const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

        if constexpr(kHasMask)
        {
            // Reverse M tile so masked (top) rows run last (assumes num_tile_n1 == 1).
            return ck_tile::make_tuple(gridDim.x - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
        }
        else
        {
            return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
        }
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // VSA reads the int LUT directly from DRAM (no LDS staging, unlike jenga onehot).
        __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q = 0;
        long_index_t batch_offset_k = 0;
        long_index_t batch_offset_v = 0;
        long_index_t batch_offset_o = 0;

        index_t seqlen_q_actual = 0;
        index_t seqlen_k_actual = 0;

        if constexpr(kIsGroupMode)
        {
            const long_index_t qstart =
                static_cast<long_index_t>(kargs.seqstart_q_ptr[i_batch]);
            const long_index_t kstart =
                static_cast<long_index_t>(kargs.seqstart_k_ptr[i_batch]);
            batch_offset_q = qstart * kargs.stride_q;
            batch_offset_k = kstart * kargs.stride_k;
            batch_offset_v = kstart * kargs.stride_v;
            batch_offset_o = qstart * kargs.stride_o;

            seqlen_q_actual =
                kargs.seqlen_q_ptr != nullptr
                    ? kargs.seqlen_q_ptr[i_batch]
                    : (kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch]);
            seqlen_k_actual =
                kargs.seqlen_k_ptr != nullptr
                    ? kargs.seqlen_k_ptr[i_batch]
                    : (kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch]);

            if(static_cast<index_t>(i_tile_m * FmhaPipeline::kM0) >= seqlen_q_actual)
                return;
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            seqlen_q_actual = kargs.seqlen_q;
            seqlen_k_actual = kargs.seqlen_k;
        }

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;

        // sparse LUT/vbn. Batch: rectangular. Group: batch-outer/head-mid packed, per-batch [H, X_b]
        // (LUT X_b = q_b*k_b via mask_batch_offset_ptr, vbn X_b = q_b via seqstart_q_block_ptr);
        // new_index = Xstart_b*H + head*X_b + local.
        const int* lut_ptr = [&]() -> const int* {
            const auto* base = reinterpret_cast<const int*>(kargs.lut_ptr);
            if constexpr(kIsGroupMode)
            {
                const index_t k_blocks_b =
                    ck_tile::integer_divide_ceil(seqlen_k_actual, FmhaPipeline::kN0);
                const long_index_t xstart_b = __builtin_amdgcn_readfirstlane(
                    kargs.mask_batch_offset_ptr[i_batch]);
                const long_index_t x_b = __builtin_amdgcn_readfirstlane(
                    kargs.mask_batch_offset_ptr[i_batch + 1]) - xstart_b;
                const long_index_t off =
                    xstart_b * kargs.num_head_q +
                    static_cast<long_index_t>(i_nhead) * x_b +
                    static_cast<long_index_t>(i_tile_m) * k_blocks_b;
                return base + off;
            }
            else
            {
                return base +
                       static_cast<long_index_t>(i_batch * kargs.num_head_q + i_nhead) *
                           ck_tile::integer_divide_ceil(kargs.seqlen_q, FmhaPipeline::kM0) *
                           ck_tile::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0) +
                       i_tile_m *
                           ck_tile::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0);
            }
        }();
        const int valid_block_num_value = [&]() -> int {
            const auto* base = reinterpret_cast<const int*>(kargs.valid_block_num_ptr);
            if constexpr(kIsGroupMode)
            {
                const long_index_t xstart_b = __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_q_block_ptr[i_batch]);
                const long_index_t x_b = __builtin_amdgcn_readfirstlane(
                    kargs.seqstart_q_block_ptr[i_batch + 1]) - xstart_b;
                const long_index_t off =
                    xstart_b * kargs.num_head_q +
                    static_cast<long_index_t>(i_nhead) * x_b +
                    static_cast<long_index_t>(i_tile_m);
                return base[off];
            }
            else
            {
                return base[
                    static_cast<long_index_t>(i_batch * kargs.num_head_q + i_nhead) *
                        ck_tile::integer_divide_ceil(kargs.seqlen_q, FmhaPipeline::kM0) +
                    i_tile_m];
            }
        }();

        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(seqlen_q_actual, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});
            if constexpr(FmhaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(seqlen_k_actual, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK_, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(seqlen_k_actual, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                const auto v_dram_transposed =
                    transform_tensor_view(v_dram_naive,
                                          make_tuple(make_pass_through_transform(kargs.hdim_v),
                                                     make_pass_through_transform(seqlen_k_actual)),
                                          make_tuple(sequence<1>{}, sequence<0>{}),
                                          make_tuple(sequence<0>{}, sequence<1>{}));

                constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
                return pad_tensor_view(
                    v_dram_transposed,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV, kPadSeqLenK_>{});
            }
            else
            {
                // ColMajor V: already laid out as [hdim_v, seqlen_k].
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    v_ptr,
                    make_tuple(kargs.hdim_v, seqlen_k_actual),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                constexpr bool kPadHeadDimV_ = kUseAsyncCopy ? kPadHeadDimV : false;
                return pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                    sequence<kPadHeadDimV_, kPadSeqLenK>{});
            }
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(number<FmhaPipeline::kM0>{},
                                      number<FmhaPipeline::kSubQKHeaddim>{});
                else
                    return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                             {i_n1, 0});

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    seqlen_q_actual,
                    seqlen_k_actual,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{seqlen_q_actual, seqlen_k_actual};
        }();

        AttentionVariant variant;
        const auto variant_params = [&]() {
            if constexpr(kHasLogitsSoftCap)
                return ck_tile::LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2>{
                    mask, kargs.scale_s, kargs.logits_soft_cap};
            else
                return ck_tile::StandardAttentionParams<FmhaMask>{mask, kargs.scale_s};
        }();

        BlockIndices block_indices{i_batch, i_nhead, i_nhead / kargs.nhead_ratio_qk};

        auto bias_dram_window = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto* bp =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias +
                    static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_bias;
                const auto bias_naive = make_naive_tensor_view<address_space_enum::global>(
                    bp, make_tuple(seqlen_q_actual, seqlen_k_actual),
                    make_tuple(kargs.stride_bias, 1), number<1>{}, number<1>{});
                const auto bias_dram = pad_tensor_view(
                    bias_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    sequence<kPadSeqLenQ, kPadSeqLenK>{});
                return make_tile_window(
                    bias_dram,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    {i_m0, 0});
            }
            else
            {
                const BiasDataType* null_bias = static_cast<const BiasDataType*>(nullptr);
                const auto dummy_naive = make_naive_tensor_view<address_space_enum::global>(
                    null_bias, make_tuple(1, 1), make_tuple(1, 1), number<1>{}, number<1>{});
                const auto dummy = pad_tensor_view(
                    dummy_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    sequence<true, true>{});
                return make_tile_window(
                    dummy,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{}),
                    {0, 0});
            }
        }();

        auto position_encoding = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto* slope_arr = reinterpret_cast<const SaccDataType*>(kargs.bias_ptr);
                const long_index_t off =
                    static_cast<long_index_t>(i_batch) * kargs.stride_bias + i_nhead;
                SaccDataType slope = slope_arr ? slope_arr[off] : SaccDataType{0};
                // FAST_EXP2: fold log2e into slope to match s_acc (pre-scaled by scale_s*log2e),
                // else alibi ends up scaled by 1/log2e.
#if CK_TILE_FMHA_FWD_FAST_EXP2
                slope = slope * ck_tile::log2e_v<SaccDataType>;
#endif
                return ck_tile::make_alibi_from_lr_mask<SaccDataType, true>(
                    slope,
                    kHasMask ? kargs.window_size_left : -1,
                    kHasMask ? kargs.window_size_right : 0,
                    seqlen_q_actual,
                    seqlen_k_actual,
                    kHasMask ? kargs.mask_type : GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            }
            else
            {
                return ck_tile::EmptyPositionEncoding<SaccDataType>{};
            }
        }();

        auto o_acc_tile = FmhaPipeline{}(q_dram_window,
                                         k_dram_window,
                                         v_dram_window,
                                         bias_dram_window,
                                         position_encoding,
                                         lut_ptr,
                                         valid_block_num_value,
                                         mask,
                                         kargs.scale_s,
                                         variant,
                                         variant_params,
                                         block_indices,
                                         smem_ptr);

        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(seqlen_q_actual, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
