// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Hand-written FmhaFwdSpargeKernel instantiation for BLOCKSCALE int8 Q/K (V/P/O fp16),
// fixed to PVSkipMode::kPerWave / hdim=128 / bm0=64.

#include "fmha_fwd_trek.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/sparse_attn/kernel/fmha_fwd_sparge_kernel.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/block_fmha_pipeline_qr_ks_vs_async_sparge.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/block_fmha_pipeline_qx_ks_vs_sparge_policy.hpp"

#include <iostream>

namespace {

// tile shape: kM0 kN0 kK0 kN1 kK1 kQKHeaddim, matches codegen sparge bm0=64 hdim=128
using sint8_block_tile = ck_tile::sequence<64, 128, 32, 128, 32, 128>;

using sint8_shape = ck_tile::TileFmhaShape<sint8_block_tile,
                                           ck_tile::sequence<2, 1, 1>,    // Gemm0BlockWarps
                                           ck_tile::sequence<32, 32, 16>, // Gemm0WarpTile
                                           ck_tile::sequence<2, 1, 1>,    // Gemm1BlockWarps
                                           ck_tile::sequence<32, 32, 16>, // Gemm1WarpTile
                                           true>;                         // VLayout row-major

// traits: only BLOCKSCALE differs vs codegen NO_SCALE (spad/dpad/dvpad=true, skpad=false)
using sint8_trait = ck_tile::TileFmhaTraits<true,  // kPadSeqLenQ
                                            false, // kPadSeqLenK
                                            true,  // kPadHeadDimQ
                                            true,  // kPadHeadDimV
                                            false, // kHasLogitsSoftCap
                                            ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                            false, // kStoreLSE
                                            false, // kHasDropout
                                            false, // kHasRandVal
                                            ck_tile::BlockAttentionQuantScaleEnum::BLOCKSCALE,
                                            -1,     // kBlockPerCu (pipeline decides)
                                            false>; // kIsVRowMajorSkip

using sint8_variant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
// locked to NoMask (recipe default -mask=0); codegen dispatcher handles other masks
using sint8_mask = ck_tile::GenericAttentionMask<false>;

using sint8_problem = ck_tile::BlockFmhaPipelineProblem<
    int8_t,                                                            // QDataType
    int8_t,                                                            // KDataType
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::VDataType,    // VDataType (fp16)
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::SaccDataType, // SaccDataType
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::SMPLComputeDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::BiasDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::RandValOutputDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::LSEDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::PDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::OaccDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::ODataType,
    sint8_shape,
    false, // kIsGroupMode
    sint8_variant,
    sint8_mask,
    false, // kUseTrLoad
    sint8_trait>;

using sint8_pipeline = ck_tile::BlockFmhaPipelineQRKSVSAsyncSparge<
    sint8_problem,
    ck_tile::BlockFmhaPipelineQRKSVSAsyncSpargeDefaultPolicy,
    ck_tile::PVSkipMode::kPerWave>;

using sint8_epilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::OaccDataType,
    typename FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>::ODataType,
    /*kPadM=*/true,
    /*kPadN=*/true>>;

using sint8_kernel =
    ck_tile::FmhaFwdSpargeKernel<sint8_pipeline, sint8_epilogue, ck_tile::PVSkipMode::kPerWave>;

} // namespace

// one-shot launcher for the BLOCKSCALE kPerWave variant; caller must match shape (fp16/d128/bm64)
void fmha_sparge_int8_fwd_oneshot_fp16_d128_bm64(const ck_tile::stream_config& s,
                                                 fmha_sparge_fwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", fmha_sparge_int8_fwd_iq_ik_d128_bm64_pvst_blockscale" << std::flush;

    auto [kargs, grids]                    = fmha_fwd_create_kargs_and_grids<sint8_kernel>(a);
    const dim3 blocks                      = sint8_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = sint8_kernel::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu>(sint8_kernel{}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{s.stream_id_});
}
