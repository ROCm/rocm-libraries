// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side bridge for FMHA BWD ConvertDQ. Maps the validated kernel
// descriptor (FmhaBwdConvertDQKernel) to the CK Tile template chain.
//
// ConvertDQ sums split-K partial results from the deterministic dQ/dK/dV
// kernel and type-converts dQ_acc (fp32) to dQ (fp16/bf16) in one pass.
//
// Unpacks generic rocm_ck::Args via named slot constants, then constructs
// CK Tile's Kargs via aggregate initialization. No __builtin_bit_cast,
// no ABI matching required.
//
// Uses C++20 struct NTTPs: template <FmhaBwdConvertDQKernel K>.

#pragma once

#include "rocm_fmha_bwd_convert_dq_api.hpp"

#include <rocm_ck/args.hpp>
#include <rocm_ck/ck_type_map.hpp>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace rocm_ck {

/// Maps a FmhaBwdConvertDQKernel descriptor to the full CK Tile type chain.
///
/// Template chain (matches dispatcher codegen):
///   FmhaBwdConvertQGradKernel<Pipeline>
///     -> BlockFmhaBwdConvertQGrad<PipelineProblem>
///       -> BlockFmhaBwdConvertQGradPipelineProblem<AccDataType, QGradDataType,
///              BlockSize, kM0, kN0, kQKHeaddim, kIsGroupMode,
///              kIsDeterministic, Traits>
///         -> TileFmhaBwdConvertQGradTraits<spad, dpad, block_per_cu>
///
/// kM0: tile rows along seqlen_q, fixed at 64 (M0_1D in codegen).
/// kN0: tile size along seqlen_k from the DqDkDv kernel. Determines the
///      number of split-K partials: nsplits = ceil(seqlen_k / kN0).
///      Defaults to hdim_q which matches the most common tile configs
///      (d128->kN0=128, d64->kN0=64). Override via the kN0 template
///      parameter for tile configs where kN0 != hdim_q.
template <FmhaBwdConvertDQKernel K, int kN0 = K.hdim_q>
struct FmhaBwdConvertDQTypes
{
    using QGradDataType    = typename CkTypeMap<K.dtype>::type;
    using QGradAccDataType = float;

    /// kM0 = 64: tile rows along seqlen_q for 1D kernels (OGradDotO, ConvertDQ).
    /// This matches M0_1D in the codegen. The block size (256) is independent;
    /// multiple threads cooperate on each kM0-row tile of width kQKHeaddim.
    static constexpr int kM0 = 64;

    using Traits =
        ck_tile::TileFmhaBwdConvertQGradTraits<K.pad_seqlen_q, K.pad_hdim_q, K.block_per_cu>;

    using PipelineProblem = ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
        QGradAccDataType, // AccDataType (fp32 input)
        QGradDataType,    // QGradDataType (fp16/bf16 output)
        K.block_size,     // BlockSize (256)
        kM0,              // tile rows along seqlen_q
        kN0,              // tile cols along seqlen_k (split-K)
        K.hdim_q,         // kQKHeaddim
        (K.mode == FmhaMode::GROUP),
        K.is_deterministic,
        Traits>;

    using Pipeline = ck_tile::BlockFmhaBwdConvertQGrad<PipelineProblem>;
    using Kernel   = ck_tile::FmhaBwdConvertQGradKernel<Pipeline>;
    using Kargs    = typename Kernel::Kargs;
};

/// Device function that invokes the CK Tile FMHA BWD ConvertDQ kernel.
///
/// Receives generic Args, unpacks tensor pointers and strides via named
/// slot constants, then constructs CK Tile's Kargs via aggregate
/// initialization. MakeKargs() is host-only, so we initialize directly.
///
/// Tensor layout in generic Args (set by host):
///
///   tensors[S::DQ_ACC]: ptr=dq_acc_ptr (fp32 accumulator, input)
///     lengths[0] = seqlen_q
///     lengths[1] = hdim_q
///     lengths[2] = seqlen_k  (needed for nsplits in deterministic mode)
///     strides[0] = stride_dq_acc          (row stride, index_t)
///     strides[1] = nhead_stride_dq_acc    (long_index_t, may exceed int32)
///     strides[2] = batch_stride_dq_acc    (long_index_t, batch mode only)
///     strides[3] = split_stride_dq_acc    (index_t, deterministic mode only)
///
///   tensors[S::DQ]: ptr=dq_ptr (fp16/bf16 output)
///     lengths[0] = seqlen_q
///     lengths[1] = hdim_q
///     strides[0] = stride_dq              (row stride, index_t)
///     strides[1] = nhead_stride_dq        (index_t)
///     strides[2] = batch_stride_dq        (index_t, batch mode only)
///
///   (group mode only):
///   tensors[S::SEQSTART_Q]: ptr = seqstart_q_ptr  ([batch+1] int32)
///   tensors[S::SEQLEN_Q]:   ptr = seqlen_q_ptr    ([batch] int32, or nullptr)
///   tensors[S::SEQSTART_K]: ptr = seqstart_k_ptr  ([batch+1] int32)
///   tensors[S::SEQLEN_K]:   ptr = seqlen_k_ptr    ([batch] int32, or nullptr)
///
///   No scalar slots are used (ConvertDQ has no scalar parameters).
///
/// Note on const_cast: TensorArg::ptr is const void* for uniform ABI.
/// dq_ptr is an output tensor that the kernel writes to, so we const_cast
/// it to void* here. This is safe because the host allocated it as mutable.
///
/// Call this from an extern "C" __global__ wrapper.
template <FmhaBwdConvertDQKernel K, int kN0 = K.hdim_q>
__device__ void runFmhaBwdConvertDQ(Args args)
{
    using T     = FmhaBwdConvertDQTypes<K, kN0>;
    namespace S = fmha_bwd_convert_dq_slots;

    // --- Unpack generic Args using named slot constants ---
    const TensorArg& t_dq_acc = args.tensors[S::DQ_ACC];
    const TensorArg& t_dq     = args.tensors[S::DQ];

    // Dimensions from tensor metadata
    const index_t seqlen_q = t_dq_acc.lengths[0];
    const index_t hdim_q   = t_dq_acc.lengths[1];
    const index_t seqlen_k = t_dq_acc.lengths[2];

    // --- Strides packed into TensorArg::strides[] ---
    //
    // DQ_ACC (fp32 accumulator):
    //   strides[0] = stride_dq_acc         (row stride, int32)
    //   strides[1] = nhead_stride_dq_acc   (int64, may exceed int32)
    //   strides[2] = batch_stride_dq_acc   (int64, batch mode only)
    //   strides[3] = split_stride_dq_acc   (int32, deterministic mode only)
    //
    // DQ (fp16/bf16 output):
    //   strides[0] = stride_dq             (row stride, int32)
    //   strides[1] = nhead_stride_dq       (int32)
    //   strides[2] = batch_stride_dq       (int32, batch mode only)
    //
    // CK Tile stores row strides as index_t (int32). Large strides that
    // exceed INT32_MAX will be silently truncated — a known CK Tile
    // limitation. nhead_stride_dq_acc is long_index_t (int64) because
    // the accumulator buffer can be very large in deterministic mode
    // (nsplits × seqlen_q × hdim_q × nhead).
    const index_t stride_dq_acc            = static_cast<index_t>(t_dq_acc.strides[0]);
    const long_index_t nhead_stride_dq_acc = t_dq_acc.strides[1];

    const index_t stride_dq       = static_cast<index_t>(t_dq.strides[0]);
    const index_t nhead_stride_dq = static_cast<index_t>(t_dq.strides[1]);

    // --- Construct CK Tile Kargs via aggregate initialization ---
    if constexpr(K.mode == FmhaMode::GROUP)
    {
        // Group mode: variable-length sequences.
        // seqlen_q and seqlen_k are set to -1 (updated per-batch from
        // seqstart/seqlen pointers on the device).
        const TensorArg& t_seqstart_q = args.tensors[S::SEQSTART_Q];
        const TensorArg& t_seqlen_q   = args.tensors[S::SEQLEN_Q];
        const TensorArg& t_seqstart_k = args.tensors[S::SEQSTART_K];
        const TensorArg& t_seqlen_k   = args.tensors[S::SEQLEN_K];

        // split_stride_dq_acc for deterministic mode
        const index_t split_stride_dq_acc = static_cast<index_t>(t_dq_acc.strides[3]);

        const typename T::Kargs kargs{
            // FmhaBwdConvertQGradCommonKargs base
            {t_dq_acc.ptr,                // dq_acc_ptr
             const_cast<void*>(t_dq.ptr), // dq_ptr (output, see const_cast note)
             -1,                          // seqlen_q (updated per-batch)
             -1,                          // seqlen_k (updated per-batch)
             hdim_q,                      // hdim_q
             stride_dq,                   // stride_dq
             stride_dq_acc,               // stride_dq_acc
             nhead_stride_dq,             // nhead_stride_dq
             nhead_stride_dq_acc},        // nhead_stride_dq_acc
            // FmhaBwdConvertQGradDeterministicKargs (always present for
            // deterministic ConvertDQ; conditional_t resolves to this)
            {split_stride_dq_acc}, // split_stride_dq_acc
            // FmhaBwdConvertQGradGroupModeKargs extension
            reinterpret_cast<const int32_t*>( // seqstart_q_ptr
                t_seqstart_q.ptr),
            reinterpret_cast<const int32_t*>( // seqstart_k_ptr
                t_seqstart_k.ptr),
            reinterpret_cast<const int32_t*>( // seqlen_q_ptr
                t_seqlen_q.ptr),
            reinterpret_cast<const int32_t*>( // seqlen_k_ptr
                t_seqlen_k.ptr),
            nullptr, // cu_seqlen_q_ptr (unused)
            nullptr  // cu_seqlen_k_ptr (unused)
        };
        typename T::Kernel{}(kargs);
    }
    else
    {
        // Batch mode: fixed-length sequences
        const long_index_t batch_stride_dq_acc = t_dq_acc.strides[2];
        const index_t batch_stride_dq          = static_cast<index_t>(t_dq.strides[2]);

        // split_stride_dq_acc for deterministic mode
        const index_t split_stride_dq_acc = static_cast<index_t>(t_dq_acc.strides[3]);

        const typename T::Kargs kargs{
            // FmhaBwdConvertQGradCommonKargs base
            {t_dq_acc.ptr,                // dq_acc_ptr
             const_cast<void*>(t_dq.ptr), // dq_ptr (output, see const_cast note)
             seqlen_q,                    // seqlen_q
             seqlen_k,                    // seqlen_k
             hdim_q,                      // hdim_q
             stride_dq,                   // stride_dq
             stride_dq_acc,               // stride_dq_acc
             nhead_stride_dq,             // nhead_stride_dq
             nhead_stride_dq_acc},        // nhead_stride_dq_acc
            // FmhaBwdConvertQGradDeterministicKargs (always present for
            // deterministic ConvertDQ; conditional_t resolves to this)
            {split_stride_dq_acc}, // split_stride_dq_acc
            // FmhaBwdConvertQGradBatchModeKargs extension
            batch_stride_dq,    // batch_stride_dq
            batch_stride_dq_acc // batch_stride_dq_acc
        };
        typename T::Kernel{}(kargs);
    }
}

} // namespace rocm_ck
