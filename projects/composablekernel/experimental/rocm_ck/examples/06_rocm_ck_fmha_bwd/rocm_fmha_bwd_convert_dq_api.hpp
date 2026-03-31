// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only helpers for the FMHA BWD ConvertDQ kernel family.
//
// HOST ONLY: this header must NOT be included from device code (.hip files).
// Device code should include rocm_fmha_bwd_convert_dq_dev.hpp.
//
// Compilation boundary:
//   _spec.hpp — consteval factory + slot constants (both passes)
//   _api.hpp (this) — host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "rocm_fmha_bwd_convert_dq_api.hpp is host-only." \
       " Device code should include rocm_fmha_bwd_convert_dq_dev.hpp."
#endif

#include "rocm_fmha_bwd_convert_dq_spec.hpp"

#include <hip/hip_runtime.h>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Grid calculation
// ---------------------------------------------------------------------------

/// Compute the launch grid for ConvertDQ.
/// Matches FmhaBwdConvertQGradKernel::GridSize():
///   dim3(ceil(seqlen_q / kM0), nhead, batch).
/// kM0 = 64 (tile rows along seqlen_q for 1D kernels), NOT block_size.
/// Precondition: tile_m0 > 0, seqlen_q >= 0, batch > 0, nhead > 0.
constexpr dim3 convert_dq_grid_size(int batch, int nhead, int seqlen_q, int tile_m0 = 64)
{
    return dim3(static_cast<unsigned>((seqlen_q + tile_m0 - 1) / tile_m0),
                static_cast<unsigned>(nhead),
                static_cast<unsigned>(batch));
}

} // namespace rocm_ck
