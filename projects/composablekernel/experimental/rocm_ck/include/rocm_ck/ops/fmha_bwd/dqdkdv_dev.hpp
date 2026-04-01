// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side bridge for FMHA BWD dQ/dK/dV. Maps the validated kernel
// descriptor (FmhaBwdDQDKDVSpec) to the CK Tile template chain.
//
// Unpacks generic rocm_ck::Args via named slot constants, then constructs
// CK Tile's Kargs via field-by-field assignment. MakeKargs()/MakeKargsImpl()
// are host-only, so we initialize directly on the device side.
//
// Uses C++20 struct NTTPs: template <FmhaBwdDQDKDVSpec K>.
//
// Compilation boundary:
//   _spec.hpp — consteval factory + slot constants (both passes)
//   _api.hpp  — host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp (this) — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifndef __HIP_DEVICE_COMPILE__
#error "dqdkdv_dev.hpp requires device compilation." \
       " Host code should include <rocm_ck/ops/fmha_bwd/dqdkdv_api.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/ck_type_map.hpp>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"

namespace rocm_ck {

// =========================================================================
// BiasEnum mapping helper
// =========================================================================

/// Map kpack's FmhaBiasType to CK Tile's BlockAttentionBiasEnum.
consteval ck_tile::BlockAttentionBiasEnum biasTypeToCkEnum(FmhaBiasType bt)
{
    switch(bt)
    {
    case FmhaBiasType::NONE: return ck_tile::BlockAttentionBiasEnum::NO_BIAS;
    case FmhaBiasType::ELEMENTWISE: return ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
    case FmhaBiasType::ALIBI: return ck_tile::BlockAttentionBiasEnum::ALIBI;
    }
    return ck_tile::BlockAttentionBiasEnum::NO_BIAS; // unreachable
}

// =========================================================================
// FmhaBwdDQDKDVTypes — maps kernel descriptor to CK Tile type chain
// =========================================================================

/// Maps a FmhaBwdDQDKDVSpec descriptor to the full CK Tile type chain.
///
/// Template chain (matches dispatcher codegen):
///   FmhaBwdDQDKDVSpec<Pipeline, KGradEpi, VGradEpi, QGradEpi>
///     -> BlockFmhaBwdDQDKDVPipeline<PipelineProblem>
///       -> BlockFmhaBwdPipelineProblem<Q,K,V,Gemm,LSE,Acc,D,Bias,RandVal,
///              O,dO,dQ,dK,dV,dBias, Shape, isGroup, isDet, Mask, Dropout,
///              useTrLoad, Traits>
///         -> TileFmhaBwdTraits<padQ, padV, BiasEnum, hasBiasGrad, blockPerCu>
///         -> TileFmhaBwdShape<BlockTile, G0BW, G0WT, G1BW, G1WT,
///              G2BW, G2WT, G3BW, G3WT, G4BW, G4WT, maxSeqLenQ>
template <FmhaBwdDQDKDVSpec K>
struct FmhaBwdDQDKDVTypes
{
    // --- Data types ---
    using QDataType     = typename CkTypeMap<K.dtype>::type;
    using KDataType     = QDataType;
    using VDataType     = QDataType;
    using GemmDataType  = QDataType;
    using ODataType     = QDataType;
    using OGradDataType = QDataType;

    // BiasDataType and BiasGradDataType always match the input dtype.
    // The CK Tile pipeline checks BiasEnum at compile time to decide whether
    // to use the bias pointer; the type itself is always the concrete dtype.
    using BiasDataType     = QDataType;
    using BiasGradDataType = QDataType;

    using LSEDataType           = float;
    using AccDataType           = float;
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;

    using QGradDataType = QDataType;
    using KGradDataType = QDataType;
    using VGradDataType = QDataType;

    // --- BiasEnum ---
    static constexpr auto kBiasEnum = biasTypeToCkEnum(K.bias_type);

    // --- Traits ---
    using Traits = ck_tile::TileFmhaBwdTraits<K.pad_hdim_q,    // kPadHeadDimQ (0, 1, or 8)
                                              K.pad_hdim_v,    // kPadHeadDimV (0, 1, or 8)
                                              kBiasEnum,       // BiasEnum
                                              K.has_bias_grad, // kHasBiasGrad
                                              K.block_per_cu>; // kBlockPerCu

    // Guard: tile geometry is only valid for d128. Other hdim values need
    // different tile configs (see fmha_bwd.py get_dq_dk_dv_tiles()).
    static_assert(K.hdim_q == 128 && K.hdim_v == 128,
                  "Tile geometry is hardcoded for d128. Other hdim values "
                  "require different tile configs.");

    // --- Tile shape (hardcoded for d128 gfx9 — Config 4 from fmha_bwd.py) ---
    //
    // From the gfx9 tile table for fp16/bf16 d128:
    //   bm0=16, bn0=128, bk0=128, bk1=16, bk2=128, bk3=16, bk4=32
    //   bhdq=128, bhdv=128
    //
    //   Gemm0/Gemm2 block_warps: <1, 4, 1>  warp_tile: <16, 16, 32>
    //   Gemm1/Gemm3 block_warps: <4, 1, 1>  warp_tile: <16, 16, 16>
    //   Gemm4       block_warps: <1, 4, 1>  warp_tile: <16, 16, min(32, 32)>
    //
    //   NumWarps = 4, BlockSize = 256, maxSeqLenQ = 0 (unlimited)
    //
    // BlockTile: sequence<bm0, bn0, bk0, bk1, bk2, bk3, bk4, bhdq, bhdv>
    using BlockTile = ck_tile::sequence<16, 128, 128, 16, 128, 16, 32, 128, 128>;

    // Gemm0 & Gemm2: compute S = Q @ K^T and dP = dO @ V^T
    using Gemm0BlockWarps = ck_tile::sequence<1, 4, 1>;
    using Gemm0WarpTile   = ck_tile::sequence<16, 16, 32>;

    // Gemm1 & Gemm3: compute dV = P^T @ dO and dK = dS^T @ Q
    using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm1WarpTile   = ck_tile::sequence<16, 16, 16>;

    // Gemm4: compute dQ = dS @ K
    using Gemm4BlockWarps = ck_tile::sequence<1, 4, 1>;
    using Gemm4WarpTile   = ck_tile::sequence<16, 16, 32>;

    // TileFmhaBwdShape: 5 GEMMs with their block_warps and warp_tiles
    //   G0=G2 (S/dP), G1=G3 (dV/dK), G4 (dQ)
    using FmhaShape = ck_tile::TileFmhaBwdShape<BlockTile,
                                                Gemm0BlockWarps,
                                                Gemm0WarpTile, // Gemm0
                                                Gemm1BlockWarps,
                                                Gemm1WarpTile, // Gemm1
                                                Gemm0BlockWarps,
                                                Gemm0WarpTile, // Gemm2 (same as G0)
                                                Gemm1BlockWarps,
                                                Gemm1WarpTile, // Gemm3 (same as G1)
                                                Gemm4BlockWarps,
                                                Gemm4WarpTile, // Gemm4
                                                0>;            // kMaxSeqLenQ (0 = unlimited)

    // --- Mask type ---
    // has_mask=true  -> GenericAttentionMask<true, true> (full masking)
    // has_mask=false -> GenericAttentionMask<false>      (no masking)
    using Mask = std::conditional_t<K.has_mask,
                                    ck_tile::GenericAttentionMask<true, true>,
                                    ck_tile::GenericAttentionMask<false>>;

    // --- Dropout type ---
    // BlockDropoutBwd<IsDropout, IsWG32, IsStoreRandval>
    // For d128 the warp tile wm0=16, so IsWG32=false -> wg16 variant.
    // We never store randval in backward (IsStoreRandval=false).
    using Dropout = ck_tile::BlockDropoutBwd<K.has_dropout,
                                             false,  // IsWG32 = false (wm0=16 for d128 config)
                                             false>; // IsStoreRandval = false

    // --- Pipeline problem ---
    static constexpr bool kIsGroupMode = (K.mode == FmhaMode::GROUP);
    static constexpr bool kUseTrLoad   = false; // non-trload for gfx9

    using PipelineProblem = ck_tile::BlockFmhaBwdPipelineProblem<QDataType,
                                                                 KDataType,
                                                                 VDataType,
                                                                 GemmDataType,
                                                                 LSEDataType,
                                                                 AccDataType,
                                                                 DDataType,
                                                                 BiasDataType,
                                                                 RandValOutputDataType,
                                                                 ODataType,
                                                                 OGradDataType,
                                                                 QGradDataType,
                                                                 KGradDataType,
                                                                 VGradDataType,
                                                                 BiasGradDataType,
                                                                 FmhaShape,
                                                                 kIsGroupMode,
                                                                 K.is_deterministic,
                                                                 Mask,
                                                                 Dropout,
                                                                 kUseTrLoad,
                                                                 Traits>;

    // --- Pipeline (auto-selected by BlockFmhaBwdDQDKDVPipeline) ---
    using Pipeline = ck_tile::BlockFmhaBwdDQDKDVPipeline<PipelineProblem>;

    // --- Epilogues ---
    // KGrad epilogue: AccDataType -> KGradDataType, kPadM=false, kPadN=(padQ>0)
    using KGradEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<AccDataType,
                                                                     KGradDataType,
                                                                     false,             // kPadM
                                                                     (K.pad_hdim_q > 0) // kPadN
                                                                     >>;

    // VGrad epilogue: AccDataType -> VGradDataType, kPadM=false, kPadN=(padV>0)
    using VGradEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<AccDataType,
                                                                     VGradDataType,
                                                                     false,             // kPadM
                                                                     (K.pad_hdim_v > 0) // kPadN
                                                                     >>;

    // QGrad epilogue: AccDataType -> QGradDataType, kPadM=false, kPadN=(padQ>0)
    using QGradEpilogue =
        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<AccDataType,
                                                                     QGradDataType,
                                                                     false,             // kPadM
                                                                     (K.pad_hdim_q > 0) // kPadN
                                                                     >>;

    // --- Kernel ---
    using Kernel =
        ck_tile::FmhaBwdDQDKDVSpec<Pipeline, KGradEpilogue, VGradEpilogue, QGradEpilogue>;

    using Kargs = typename Kernel::Kargs;
};

// =========================================================================
// runFmhaBwdDQDKDV — device function
// =========================================================================

/// Device function that invokes the CK Tile FMHA BWD dQ/dK/dV kernel.
///
/// Receives generic Args, unpacks tensor pointers and strides via named
/// slot constants, then constructs CK Tile's Kargs via field-by-field
/// assignment. MakeKargsImpl() is host-only (uses std::variant for dropout),
/// so we initialize directly on the device side.
///
/// Tensor layout in generic Args (set by host):
///
///   tensors[S::Q]:      ptr, strides=[stride_q, nhead_stride_q, batch_stride_q]
///   tensors[S::K]:      ptr, strides=[stride_k, nhead_stride_k, batch_stride_k]
///   tensors[S::V]:      ptr, strides=[stride_v, nhead_stride_v, batch_stride_v]
///   tensors[S::LSE]:    ptr, strides=[nhead_stride_lsed, batch_stride_lsed]
///   tensors[S::DO]:     ptr, strides=[stride_do, nhead_stride_do, batch_stride_do]
///   tensors[S::D]:      ptr, strides=[nhead_stride_lsed, batch_stride_lsed]
///   tensors[S::DQ_ACC]: ptr, strides=[stride_dq_acc, nhead_stride_dq_acc,
///                                     batch_stride_dq_acc, split_stride_dq_acc]
///   tensors[S::DK]:     ptr, strides=[stride_dk, nhead_stride_dk, batch_stride_dk]
///   tensors[S::DV]:     ptr, strides=[stride_dv, nhead_stride_dv, batch_stride_dv]
///
///   Optional:
///   tensors[S::BIAS]:   ptr, strides=[stride_bias, nhead_stride_bias, batch_stride_bias]
///   tensors[S::DBIAS]:  ptr, strides=[stride_dbias, nhead_stride_dbias, batch_stride_dbias]
///   tensors[S::RANDVAL]: ptr, strides=[stride_randval, nhead_stride_randval,
///                                      batch_stride_randval]
///
///   Each tensor carries its own natural dimensions in lengths[]:
///     Q:  lengths[0]=seqlen_q, lengths[1]=hdim_q
///     K:  lengths[0]=seqlen_k, lengths[1]=hdim_q
///     V:  lengths[0]=seqlen_k, lengths[1]=hdim_v
///
///   Problem-level dims are passed as scalars (not overloaded in lengths):
///     scalars[S::NUM_HEAD_Q].i32     = number of Q heads
///     scalars[S::NHEAD_RATIO_QK].i32 = Q heads / K heads (GQA/MQA)
///
///   scalars[S::RAW_SCALE].f32 = raw_scale (1/sqrt(hdim))
///   scalars[S::SCALE].f32     = raw_scale * log2(e)
///
///   (dropout only):
///   scalars[S::P_UNDROP].f32    = 1/(1-dropout_rate)
///   scalars[S::RP_UNDROP].f32   = 1/p_undrop  (== 1 - dropout_rate)
///   scalars[S::DROP_SEED].u64   = dropout RNG seed
///   scalars[S::DROP_OFFSET].u64 = dropout RNG offset
///
/// CK Tile stores row strides as index_t (int32). Large strides that
/// exceed INT32_MAX will be silently truncated — a known CK Tile limitation.
/// The nhead_stride_dq_acc is long_index_t (int64).
///
/// Call this from an extern "C" __global__ wrapper.
template <FmhaBwdDQDKDVSpec K>
__device__ void runFmhaBwdDQDKDV(Args args)
{
    using T     = FmhaBwdDQDKDVTypes<K>;
    namespace S = fmha_bwd_dqdkdv_slots;

    // --- Unpack generic Args using named slot constants ---
    const TensorArg& t_q      = args.tensors[S::Q];
    const TensorArg& t_k      = args.tensors[S::K];
    const TensorArg& t_v      = args.tensors[S::V];
    const TensorArg& t_lse    = args.tensors[S::LSE];
    const TensorArg& t_do     = args.tensors[S::DO];
    const TensorArg& t_d      = args.tensors[S::D];
    const TensorArg& t_dq_acc = args.tensors[S::DQ_ACC];
    const TensorArg& t_dk     = args.tensors[S::DK];
    const TensorArg& t_dv     = args.tensors[S::DV];

    // --- Dimensions from tensor metadata and scalars ---
    // Each tensor carries its own natural dimensions in lengths[].
    // Problem-level dims (num_head_q, nhead_ratio_qk) are scalars.
    const index_t seqlen_q       = t_q.lengths[0]; // Q: [seqlen_q, hdim_q]
    const index_t hdim_q         = t_q.lengths[1];
    const index_t seqlen_k       = t_k.lengths[0]; // K: [seqlen_k, hdim_q]
    const index_t hdim_v         = t_v.lengths[1]; // V: [seqlen_k, hdim_v]
    const index_t num_head_q     = args.scalars[S::NUM_HEAD_Q].i32;
    const index_t nhead_ratio_qk = args.scalars[S::NHEAD_RATIO_QK].i32;

    // --- Scalars ---
    const float raw_scale = args.scalars[S::RAW_SCALE].f32;
    const float scale     = args.scalars[S::SCALE].f32;

    // --- Strides ---
    // Q: strides[0]=stride_q, [1]=nhead_stride_q, [2]=batch_stride_q
    const index_t stride_q       = static_cast<index_t>(t_q.strides[0]);
    const index_t nhead_stride_q = static_cast<index_t>(t_q.strides[1]);

    // K: strides[0]=stride_k, [1]=nhead_stride_k, [2]=batch_stride_k
    const index_t stride_k       = static_cast<index_t>(t_k.strides[0]);
    const index_t nhead_stride_k = static_cast<index_t>(t_k.strides[1]);

    // V: strides[0]=stride_v, [1]=nhead_stride_v, [2]=batch_stride_v
    const index_t stride_v       = static_cast<index_t>(t_v.strides[0]);
    const index_t nhead_stride_v = static_cast<index_t>(t_v.strides[1]);

    // LSE: strides[0]=nhead_stride_lsed, [1]=batch_stride_lsed
    const index_t nhead_stride_lsed = static_cast<index_t>(t_lse.strides[0]);

    // DO: strides[0]=stride_do, [1]=nhead_stride_do, [2]=batch_stride_do
    const index_t stride_do       = static_cast<index_t>(t_do.strides[0]);
    const index_t nhead_stride_do = static_cast<index_t>(t_do.strides[1]);

    // DQ_ACC: strides[0]=stride_dq_acc, [1]=nhead_stride_dq_acc (int64!),
    //         [2]=batch_stride_dq_acc (int64!), [3]=split_stride_dq_acc
    const index_t stride_dq_acc            = static_cast<index_t>(t_dq_acc.strides[0]);
    const long_index_t nhead_stride_dq_acc = t_dq_acc.strides[1];

    // DK: strides[0]=stride_dk, [1]=nhead_stride_dk, [2]=batch_stride_dk
    const index_t stride_dk       = static_cast<index_t>(t_dk.strides[0]);
    const index_t nhead_stride_dk = static_cast<index_t>(t_dk.strides[1]);

    // DV: strides[0]=stride_dv, [1]=nhead_stride_dv, [2]=batch_stride_dv
    const index_t stride_dv       = static_cast<index_t>(t_dv.strides[0]);
    const index_t nhead_stride_dv = static_cast<index_t>(t_dv.strides[1]);

    // --- Construct CK Tile Kargs ---
    //
    // The Kargs struct uses multiple inheritance with conditional base classes.
    // Aggregate initialization must match the exact inheritance order:
    //   1. FmhaBwdCommonKargs
    //   2. Bias kargs (conditional)
    //   3. BiasGrad kargs (conditional)
    //   4. Mask kargs (conditional)
    //   5. Dropout kargs (conditional)
    //   6. Deterministic kargs (conditional)
    //   7. Batch/Group-specific members
    //
    // Because the conditional bases make aggregate init fragile, we use
    // aggregate init for the common + empty-placeholder bases, then
    // assign optional fields via if constexpr — following the same pattern
    // as CK Tile's MakeKargsImpl().

    if constexpr(K.mode == FmhaMode::GROUP)
    {
        // Group mode: variable-length sequences
        // TODO: Group mode requires additional tensor slots for seqstart/seqlen
        //       pointers. Full implementation deferred — batch mode is the
        //       demo target.

        typename T::Kargs kargs{
            // FmhaBwdCommonKargs
            {t_q.ptr,                         // q_ptr
             t_k.ptr,                         // k_ptr
             t_v.ptr,                         // v_ptr
             t_lse.ptr,                       // lse_ptr
             t_do.ptr,                        // do_ptr
             t_d.ptr,                         // d_ptr (input: const void*)
             const_cast<void*>(t_dq_acc.ptr), // dq_acc_ptr (output)
             const_cast<void*>(t_dk.ptr),     // dk_ptr (output)
             const_cast<void*>(t_dv.ptr),     // dv_ptr (output)
             -1,                              // seqlen_q (updated per-batch)
             -1,                              // seqlen_k (updated per-batch)
             hdim_q,                          // hdim_q
             hdim_v,                          // hdim_v
             num_head_q,                      // num_head_q
             nhead_ratio_qk,                  // nhead_ratio_qk
             raw_scale,                       // raw_scale
             scale,                           // scale
             stride_q,                        // stride_q
             stride_k,                        // stride_k
             stride_v,                        // stride_v
             stride_do,                       // stride_do
             stride_dq_acc,                   // stride_dq_acc
             stride_dk,                       // stride_dk
             stride_dv,                       // stride_dv
             nhead_stride_q,                  // nhead_stride_q
             nhead_stride_k,                  // nhead_stride_k
             nhead_stride_v,                  // nhead_stride_v
             nhead_stride_do,                 // nhead_stride_do
             nhead_stride_lsed,               // nhead_stride_lsed
             nhead_stride_dq_acc,             // nhead_stride_dq_acc
             nhead_stride_dk,                 // nhead_stride_dk
             nhead_stride_dv},                // nhead_stride_dv
            {},                               // placeholder for bias
            {},                               // placeholder for dbias
            {},                               // placeholder for mask
            {},                               // placeholder for dropout
            {},                               // placeholder for deterministic
            // FmhaBwdGroupModeKargs members
            nullptr, // seqstart_q_ptr  (TODO: populate from Args)
            nullptr, // seqstart_k_ptr  (TODO: populate from Args)
            nullptr, // seqlen_q_ptr    (TODO: populate from Args)
            nullptr, // seqlen_k_ptr    (TODO: populate from Args)
            nullptr, // cu_seqlen_q_ptr (TODO: populate from Args)
            nullptr  // cu_seqlen_k_ptr (TODO: populate from Args)
        };

        // TODO: Populate optional fields (bias, mask, dropout, deterministic)
        //       and group-mode seqstart/seqlen pointers.

        typename T::Kernel{}(kargs);
    }
    else
    {
        // Batch mode: fixed-length sequences

        // Batch strides
        const index_t batch_stride_q           = static_cast<index_t>(t_q.strides[2]);
        const index_t batch_stride_k           = static_cast<index_t>(t_k.strides[2]);
        const index_t batch_stride_v           = static_cast<index_t>(t_v.strides[2]);
        const index_t batch_stride_do          = static_cast<index_t>(t_do.strides[2]);
        const index_t batch_stride_lsed        = static_cast<index_t>(t_lse.strides[1]);
        const long_index_t batch_stride_dq_acc = t_dq_acc.strides[2];
        const index_t batch_stride_dk          = static_cast<index_t>(t_dk.strides[2]);
        const index_t batch_stride_dv          = static_cast<index_t>(t_dv.strides[2]);

        typename T::Kargs kargs{
            // FmhaBwdCommonKargs
            {t_q.ptr,                         // q_ptr
             t_k.ptr,                         // k_ptr
             t_v.ptr,                         // v_ptr
             t_lse.ptr,                       // lse_ptr
             t_do.ptr,                        // do_ptr
             t_d.ptr,                         // d_ptr (input: const void*)
             const_cast<void*>(t_dq_acc.ptr), // dq_acc_ptr (output)
             const_cast<void*>(t_dk.ptr),     // dk_ptr (output)
             const_cast<void*>(t_dv.ptr),     // dv_ptr (output)
             seqlen_q,                        // seqlen_q
             seqlen_k,                        // seqlen_k
             hdim_q,                          // hdim_q
             hdim_v,                          // hdim_v
             num_head_q,                      // num_head_q
             nhead_ratio_qk,                  // nhead_ratio_qk
             raw_scale,                       // raw_scale
             scale,                           // scale
             stride_q,                        // stride_q
             stride_k,                        // stride_k
             stride_v,                        // stride_v
             stride_do,                       // stride_do
             stride_dq_acc,                   // stride_dq_acc
             stride_dk,                       // stride_dk
             stride_dv,                       // stride_dv
             nhead_stride_q,                  // nhead_stride_q
             nhead_stride_k,                  // nhead_stride_k
             nhead_stride_v,                  // nhead_stride_v
             nhead_stride_do,                 // nhead_stride_do
             nhead_stride_lsed,               // nhead_stride_lsed
             nhead_stride_dq_acc,             // nhead_stride_dq_acc
             nhead_stride_dk,                 // nhead_stride_dk
             nhead_stride_dv},                // nhead_stride_dv
            {}, // placeholder for bias    (EmptyKargs<0> or BiasKargs)
            {}, // placeholder for dbias   (EmptyKargs<1> or BiasGradKargs)
            {}, // placeholder for mask    (EmptyKargs<2> or MaskKargs)
            {}, // placeholder for dropout (EmptyKargs<3> or DropoutKargs)
            {}, // placeholder for determ  (EmptyKargs<4> or DetermKargs)
            // FmhaBwdBatchModeKargs own members
            batch_stride_q,      // batch_stride_q
            batch_stride_k,      // batch_stride_k
            batch_stride_v,      // batch_stride_v
            batch_stride_do,     // batch_stride_do
            batch_stride_lsed,   // batch_stride_lsed
            batch_stride_dq_acc, // batch_stride_dq_acc
            batch_stride_dk,     // batch_stride_dk
            batch_stride_dv      // batch_stride_dv
        };

        // --- Populate optional fields via if constexpr ---

        if constexpr(K.bias_type == FmhaBiasType::ELEMENTWISE)
        {
            const TensorArg& t_bias = args.tensors[S::BIAS];
            kargs.bias_ptr          = t_bias.ptr;
            kargs.stride_bias       = static_cast<index_t>(t_bias.strides[0]);
            kargs.nhead_stride_bias = static_cast<index_t>(t_bias.strides[1]);
            kargs.batch_stride_bias = static_cast<index_t>(t_bias.strides[2]);
        }
        else if constexpr(K.bias_type == FmhaBiasType::ALIBI)
        {
            const TensorArg& t_bias  = args.tensors[S::BIAS];
            kargs.alibi_slope_ptr    = t_bias.ptr;
            kargs.alibi_slope_stride = static_cast<index_t>(t_bias.strides[0]);
        }

        if constexpr(K.has_bias_grad)
        {
            const TensorArg& t_dbias = args.tensors[S::DBIAS];
            kargs.dbias_ptr          = const_cast<void*>(t_dbias.ptr);
            kargs.stride_dbias       = static_cast<index_t>(t_dbias.strides[0]);
            kargs.nhead_stride_dbias = static_cast<index_t>(t_dbias.strides[1]);
            kargs.batch_stride_dbias = static_cast<index_t>(t_dbias.strides[2]);
        }

        if constexpr(K.has_mask)
        {
            // Default: causal mask (upper-triangle masking).
            // window_size_left = -1 means unlimited left context.
            // window_size_right = 0 means no future tokens (causal).
            // For generic sliding window, pass window sizes via scalars.
            kargs.window_size_left  = -1;
            kargs.window_size_right = 0;
            kargs.mask_type         = ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
        }

        if constexpr(K.has_dropout)
        {
            // TODO: Dropout initialization requires p_drop computation
            //       and seed/offset from scalars. For the plain-config
            //       demo, has_dropout is false.
            const float p_undrop      = args.scalars[S::P_UNDROP].f32;
            const float rp_undrop     = args.scalars[S::RP_UNDROP].f32;
            kargs.rp_undrop           = rp_undrop;
            kargs.scale_rp_undrop     = rp_undrop * raw_scale;
            kargs.p_undrop_in_uint8_t = static_cast<uint8_t>(__builtin_floorf(p_undrop * 255.0f));

            kargs.drop_seed.val                 = args.scalars[S::DROP_SEED].u64;
            kargs.drop_offset.val               = args.scalars[S::DROP_OFFSET].u64;
            kargs.is_drop_seed_offset_from_host = true;

            // randval not stored in backward pass
            kargs.rand_val_ptr         = nullptr;
            kargs.stride_randval       = 0;
            kargs.nhead_stride_randval = 0;
            kargs.batch_stride_randval = 0;
        }

        if constexpr(K.is_deterministic)
        {
            const index_t split_stride_dq_acc = static_cast<index_t>(t_dq_acc.strides[3]);
            kargs.split_stride_dq_acc         = split_stride_dq_acc;
        }

        typename T::Kernel{}(kargs);
    }
}

} // namespace rocm_ck
