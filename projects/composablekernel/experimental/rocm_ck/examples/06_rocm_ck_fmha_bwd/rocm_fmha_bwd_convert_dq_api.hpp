// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compile-time configuration for the FMHA BWD ConvertDQ kernel family.
//
// ConvertDQ converts the fp32 dQ accumulator (produced by the deterministic
// dQ/dK/dV kernel via split-K) to the original dtype (fp16/bf16). It sums
// the per-split partial results and type-converts in one pass.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI and
// compile-time-validated configuration. Requires HIP for dim3.

#pragma once

#include "rocm_fmha_bwd_common.hpp"

#include <rocm_ck/datatype_utils.hpp>

#include <hip/hip_runtime.h>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Signature / Algorithm / Config / Kernel
// ---------------------------------------------------------------------------

/// Signature: describes WHAT the kernel computes (problem shape only).
struct FmhaBwdConvertDQSignature
{
    DataType dtype; // fp16 or bf16 (dQ_acc is always fp32; dQ output is dtype)
    int hdim_q;     // head dimension: 32, 64, 96, 128, 256
    FmhaMode mode;  // batch or group
};

/// Algorithm: describes HOW the kernel executes (tuning and strategy).
struct FmhaBwdConvertDQAlgorithm
{
    bool is_deterministic = true; // always true in practice (ConvertDQ exists
                                  // only for the deterministic path)
    bool pad_seqlen_q = true;     // kPadSeqLenQ
    bool pad_hdim_q   = true;     // kPadHeadDimQ
    int block_per_cu  = 2;        // occupancy hint
};

/// Config: user-facing Signature + Algorithm pair.
struct FmhaBwdConvertDQConfig
{
    FmhaBwdConvertDQSignature signature;
    FmhaBwdConvertDQAlgorithm algorithm;
};

/// Validated kernel descriptor -- structural type, safe for use as NTTP.
/// All optional/default values are resolved; no std::optional.
struct FmhaBwdConvertDQKernel
{
    DataType dtype;
    int hdim_q;
    FmhaMode mode;
    bool is_deterministic;
    bool pad_seqlen_q;
    bool pad_hdim_q;
    int block_per_cu;
    int block_size; // computed: 256 for d128
};

// ---------------------------------------------------------------------------
// Named slot constants for generic rocm_ck::Args
// ---------------------------------------------------------------------------

namespace fmha_bwd_convert_dq_slots {

/// Tensor slots.
constexpr int DQ_ACC = 0; // [seqlen_q, hdim_q] fp32 accumulator (input)
constexpr int DQ     = 1; // [seqlen_q, hdim_q] fp16/bf16 output

// Group-mode additional tensor slots.
// CK Tile's FmhaBwdConvertQGradGroupModeKargs requires both Q and K
// sequence info for computing nsplits in deterministic mode.
constexpr int SEQSTART_Q = 2; // [batch+1] Q-sequence start offsets
constexpr int SEQLEN_Q   = 3; // [batch]   per-sequence Q-lengths
constexpr int SEQSTART_K = 4; // [batch+1] K-sequence start offsets
constexpr int SEQLEN_K   = 5; // [batch]   per-sequence K-lengths

// Scalar slots.
// Note: CK Tile's ConvertQGrad does NOT apply an attention scale.
// It only sums split-K partials and type-converts. No scalar slots
// are currently needed, but we reserve slot 0 for future use.
// constexpr int RESERVED = 0;

/// Number of tensor slots required for a given kernel configuration.
consteval int requiredTensors(FmhaBwdConvertDQKernel k)
{
    int n = 2; // DQ_ACC + DQ
    if(k.mode == FmhaMode::GROUP)
        n += 4; // SEQSTART_Q + SEQLEN_Q + SEQSTART_K + SEQLEN_K
    return n;
}

/// Number of scalar slots required for a given kernel configuration.
consteval int requiredScalars(FmhaBwdConvertDQKernel /* k */)
{
    return 0; // ConvertQGrad has no scalar parameters
}

} // namespace fmha_bwd_convert_dq_slots

// ---------------------------------------------------------------------------
// make_kernel -- consteval validation
// ---------------------------------------------------------------------------

/// Validate config and produce a structural kernel descriptor.
/// Overload resolution: each kernel family has its own Config type,
/// so make_kernel(FmhaBwdConvertDQConfig) is unambiguous.
consteval FmhaBwdConvertDQKernel make_kernel(FmhaBwdConvertDQConfig cfg)
{
    auto sig  = cfg.signature;
    auto algo = cfg.algorithm;

    if(sig.dtype != DataType::FP16 && sig.dtype != DataType::BF16)
        throw "FmhaBwdConvertDQ only supports FP16 or BF16"
              " (dQ_acc is always fp32; dQ output is dtype)";

    if(sig.hdim_q != 32 && sig.hdim_q != 64 && sig.hdim_q != 96 && sig.hdim_q != 128 &&
       sig.hdim_q != 256)
        throw "hdim_q must be one of {32, 64, 96, 128, 256}";

    // Group mode requires seqlen padding (variable-length sequences)
    if(sig.mode == FmhaMode::GROUP && !algo.pad_seqlen_q)
        throw "group mode requires pad_seqlen_q=true"
              " (variable-length sequences)";

    if(algo.block_per_cu <= 0)
        throw "block_per_cu must be positive";

    // Block size: 256 threads (4 warps x 64 threads/warp).
    // Matches the CK Tile ConvertQGrad kernel configuration for d128.
    constexpr int block_size = 256;

    FmhaBwdConvertDQKernel k{sig.dtype,
                             sig.hdim_q,
                             sig.mode,
                             algo.is_deterministic,
                             algo.pad_seqlen_q,
                             algo.pad_hdim_q,
                             algo.block_per_cu,
                             block_size};

    return k;
}

// --- make_kernel compile-time tests ---
// clang-format off

// Valid configs compile:
static_assert(make_kernel(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                  .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}
}).dtype == DataType::FP16);

static_assert(make_kernel(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::BF16, .hdim_q = 64,
                  .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}
}).hdim_q == 64);

static_assert(make_kernel(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                  .mode = FmhaMode::GROUP},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}
}).mode == FmhaMode::GROUP);

static_assert(make_kernel(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                  .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = false, .pad_hdim_q = false}
}).pad_seqlen_q == false);

static_assert(make_kernel(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                  .mode = FmhaMode::BATCH},
    .algorithm = {}
}).block_size == 256);

static_assert(make_kernel(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                  .mode = FmhaMode::BATCH},
    .algorithm = {}
}).is_deterministic == true);

// Slot counts: batch = 2 tensors, group = 6 tensors
static_assert(fmha_bwd_convert_dq_slots::requiredTensors(
    make_kernel(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {}})) == 2);

static_assert(fmha_bwd_convert_dq_slots::requiredTensors(
    make_kernel(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                      .mode = FmhaMode::GROUP},
        .algorithm = {}})) == 6);

// Invalid configs (uncommenting produces consteval compile errors):
// make_kernel({.signature = {.dtype = DataType::FP32, ...}})
//   -- FP32 not supported
// make_kernel({.signature = {.hdim_q = 100, ...}})
//   -- invalid hdim
// make_kernel({.signature = {.mode = FmhaMode::GROUP},
//              .algorithm = {.pad_seqlen_q = false}})
//   -- group mode requires pad_seqlen_q
// make_kernel({..., .algorithm = {.block_per_cu = 0}})
//   -- block_per_cu must be positive

// clang-format on

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
