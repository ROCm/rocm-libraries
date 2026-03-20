// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Device-side bridge for FMHA BWD OGradDotO. Maps the validated kernel
// descriptor (FmhaBwdOGradDotOKernel) to the CK Tile template chain.
//
// Uses C++20 struct NTTPs: template <FmhaBwdOGradDotOKernel K>.

#pragma once

#include "rocm_fmha_bwd_api.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace rocm_ck {

/// Maps a DataType enum value to the corresponding CK Tile numeric type.
/// Only FP16 and BF16 are needed for FMHA BWD OGradDotO.
template <DataType>
struct CkTypeMap;

template <>
struct CkTypeMap<DataType::FP16>
{
    using type = ck_tile::half_t;
};
template <>
struct CkTypeMap<DataType::BF16>
{
    using type = ck_tile::bf16_t;
};

/// Maps a FmhaBwdOGradDotOKernel descriptor to the full CK Tile type chain.
///
/// Template chain (matches dispatcher codegen):
///   FmhaBwdOGradDotOKernel<Pipeline>
///     -> BlockFmhaBwdOGradDotO<PipelineProblem>
///       -> BlockFmhaBwdOGradDotOPipelineProblem<OType, dOType, DType,
///              bm0, hdim_v, is_group, Traits>
///         -> TileFmhaBwdOGradDotOTraits<spad, dvpad, block_per_cu>
template <FmhaBwdOGradDotOKernel K>
struct FmhaBwdOGradDotOTypes
{
    using ODataType     = typename CkTypeMap<K.dtype>::type; // half_t or bf16_t
    using OGradDataType = ODataType;                         // dO always same type as O
    using DDataType     = float;                             // D is always float

    using Traits =
        ck_tile::TileFmhaBwdOGradDotOTraits<K.pad_seqlen_q, K.pad_hdim_v, K.block_per_cu>;

    using PipelineProblem =
        ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<ODataType,
                                                      OGradDataType,
                                                      DDataType,
                                                      K.block_size,
                                                      K.hdim_v,
                                                      (K.mode == FmhaMode::GROUP),
                                                      Traits>;

    using Pipeline = ck_tile::BlockFmhaBwdOGradDotO<PipelineProblem>;
    using Kernel   = ck_tile::FmhaBwdOGradDotOKernel<Pipeline>;
    using Kargs    = typename Kernel::Kargs;
    using ApiArgs  = std::conditional_t<K.mode == FmhaMode::GROUP,
                                        FmhaBwdOGradDotOGroupArgs,
                                        FmhaBwdOGradDotOBatchArgs>;
};

/// Device function that invokes the CK Tile FMHA BWD OGradDotO kernel.
///
/// The API args struct (FmhaBwdOGradDotOBatchArgs or FmhaBwdOGradDotOGroupArgs)
/// has identical memory layout to CK Tile's internal Kargs (verified by
/// static_assert). We use __builtin_bit_cast to convert safely (no strict
/// aliasing violation, zero runtime cost).
///
/// Call this from an extern "C" __global__ wrapper.
template <FmhaBwdOGradDotOKernel K>
__device__ void runFmhaBwdOGradDotO(const typename FmhaBwdOGradDotOTypes<K>::ApiArgs& args)
{
    using T = FmhaBwdOGradDotOTypes<K>;

    // ABI safety: verify our flat API args struct matches CK Tile's internal Kargs
    static_assert(sizeof(typename T::ApiArgs) == sizeof(typename T::Kargs),
                  "API args size mismatch with CK Tile Kargs");
    static_assert(alignof(typename T::ApiArgs) == alignof(typename T::Kargs),
                  "API args alignment mismatch with CK Tile Kargs");

    const auto kargs = __builtin_bit_cast(typename T::Kargs, args);
    typename T::Kernel{}(kargs);
}

} // namespace rocm_ck
