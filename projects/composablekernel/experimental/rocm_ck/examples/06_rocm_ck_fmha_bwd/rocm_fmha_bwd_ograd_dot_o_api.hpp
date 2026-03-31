// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only helpers for the FMHA BWD OGradDotO kernel family.
//
// HOST ONLY: this header must NOT be included from device code (.hip files).
// Device code should include rocm_fmha_bwd_ograd_dot_o_dev.hpp.
//
// Compilation boundary:
//   _spec.hpp — consteval factory + slot constants (both passes)
//   _api.hpp (this) — host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "rocm_fmha_bwd_ograd_dot_o_api.hpp is host-only." \
       " Device code should include rocm_fmha_bwd_ograd_dot_o_dev.hpp."
#endif

#include "rocm_fmha_bwd_ograd_dot_o_spec.hpp"

#include <hip/hip_runtime.h>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Grid calculation
// ---------------------------------------------------------------------------

/// Compute the launch grid for OGradDotO.
/// Matches FmhaBwdOGradDotOKernel::GridSize():
///   dim3(ceil(seqlen_q / kM0), nhead, batch).
/// Precondition: block_size > 0, seqlen_q >= 0, batch > 0, nhead > 0.
constexpr dim3 ograd_dot_o_grid_size(int batch, int nhead, int seqlen_q, int block_size)
{
    return dim3(static_cast<unsigned>((seqlen_q + block_size - 1) / block_size),
                static_cast<unsigned>(nhead),
                static_cast<unsigned>(batch));
}

} // namespace rocm_ck
