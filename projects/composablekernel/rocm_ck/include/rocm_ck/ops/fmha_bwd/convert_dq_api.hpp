// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only helpers for the FMHA BWD ConvertDQ kernel family.
//
// HOST ONLY: this header must NOT be included from device code (.hip files).
// Device code should include <rocm_ck/ops/fmha_bwd/convert_dq_dev.hpp>.
//
// Compilation boundary:
//   _spec.hpp — consteval factory + slot constants (both passes)
//   _api.hpp (this) — host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "convert_dq_api.hpp is host-only." \
       " Device code should include <rocm_ck/ops/fmha_bwd/convert_dq_dev.hpp>."
#endif

#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <rocm_ck/args.hpp>
#include <rocm_ck/grid_dim.hpp>

#ifndef NDEBUG
#include <cstdio>
#include <cstdlib>
#endif

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Grid calculation
// ---------------------------------------------------------------------------

/// Compute the launch grid for ConvertDQ.
/// Matches FmhaBwdConvertQGradKernel::GridSize():
///   GridDim(ceil(seqlen_q / kM0), nhead, batch).
/// kM0 = 64 (tile rows along seqlen_q for 1D kernels), NOT block_size.
/// Precondition: tile_m0 > 0, seqlen_q >= 0, batch > 0, nhead > 0.
inline GridDim convert_dq_grid_size(int batch, int nhead, int seqlen_q, int tile_m0 = 64)
{
#ifndef NDEBUG
    if(tile_m0 <= 0)
    {
        std::fprintf(
            stderr, "rocm_ck::convert_dq_grid_size: tile_m0 must be positive, got %d\n", tile_m0);
        std::abort();
    }
    if(seqlen_q < 0)
    {
        std::fprintf(stderr,
                     "rocm_ck::convert_dq_grid_size: seqlen_q must be non-negative, got %d\n",
                     seqlen_q);
        std::abort();
    }
#endif
    return {static_cast<unsigned>((seqlen_q + tile_m0 - 1) / tile_m0),
            static_cast<unsigned>(nhead),
            static_cast<unsigned>(batch)};
}

// ---------------------------------------------------------------------------
// Debug-only runtime Args validation
// ---------------------------------------------------------------------------

/// Validate that all required tensor slots for ConvertDQ are populated.
/// Compiles to nothing in release builds.
inline void validateArgs([[maybe_unused]] const Args& args, [[maybe_unused]] FmhaBwdConvertDQSpec k)
{
#ifndef NDEBUG
    namespace S = fmha_bwd_convert_dq_slots;

    // clang-format off
    static constexpr const char* tensor_names[] = {
        "DQ_ACC", "DQ", "SEQSTART_Q", "SEQLEN_Q", "SEQSTART_K", "SEQLEN_K"
    };
    // clang-format on

    int n = S::requiredTensors(k);
    for(int i = 0; i < n; ++i)
    {
        if(args.tensors[i].ptr == nullptr)
        {
            std::fprintf(stderr,
                         "rocm_ck::validateArgs(ConvertDQ): tensor \"%s\" (slot %d)"
                         " has null pointer\n",
                         tensor_names[i],
                         i);
            std::abort();
        }
    }
#endif
}

} // namespace rocm_ck
