// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only helpers for the FMHA BWD dQ/dK/dV kernel family.
//
// HOST ONLY: this header must NOT be included from device code (.hip files).
// Device code should include rocm_fmha_bwd_dqdkdv_dev.hpp.
//
// Compilation boundary:
//   _spec.hpp — consteval factory + slot constants (both passes)
//   _api.hpp (this) — host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "rocm_fmha_bwd_dqdkdv_api.hpp is host-only." \
       " Device code should include rocm_fmha_bwd_dqdkdv_dev.hpp."
#endif

#include "rocm_fmha_bwd_dqdkdv_spec.hpp"

#include <hip/hip_runtime.h>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Grid calculation
// ---------------------------------------------------------------------------

/// Compute the launch grid for dQ/dK/dV.
/// Matches CK Tile's FmhaBwdDQDKDVKernel::GridSize():
///   dim3(ceil(seqlen_k / kN0), nhead, batch).
/// block_n0 comes from FmhaBwdDQDKDVKernel::block_n0 (kN0).
/// Precondition: block_n0 > 0, seqlen_k >= 0, batch > 0, nhead > 0.
constexpr dim3 dqdkdv_grid_size(int batch, int nhead, int seqlen_k, int block_n0)
{
    return dim3(static_cast<unsigned>((seqlen_k + block_n0 - 1) / block_n0),
                static_cast<unsigned>(nhead),
                static_cast<unsigned>(batch));
}

} // namespace rocm_ck
