// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compile-time configuration for the FMHA BWD ConvertDQ kernel family.
//
// ConvertDQ converts the fp32 dQ accumulator (produced by the deterministic
// dQ/dK/dV kernel via split-K) to the original dtype (fp16/bf16). It sums
// the per-split partial results and type-converts in one pass.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types, consteval make_spec() factory, and named slot
// constants. No runtime code, no HIP dependency.
//
// Compilation boundary:
//   _spec.hpp (this) — consteval factory + slot constants (both passes)
//   _api.hpp         — host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp         — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#include <rocm_ck/ops/fmha_bwd/common.hpp>

#include <rocm_ck/datatype_utils.hpp>

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
struct FmhaBwdConvertDQSpec
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
constexpr int requiredTensors(FmhaBwdConvertDQSpec k)
{
    int n = 2; // DQ_ACC + DQ
    if(k.mode == FmhaMode::GROUP)
        n += 4; // SEQSTART_Q + SEQLEN_Q + SEQSTART_K + SEQLEN_K
    return n;
}

/// Number of scalar slots required for a given kernel configuration.
constexpr int requiredScalars(FmhaBwdConvertDQSpec /* k */)
{
    return 0; // ConvertQGrad has no scalar parameters
}

} // namespace fmha_bwd_convert_dq_slots

// ---------------------------------------------------------------------------
// make_spec -- consteval validation
// ---------------------------------------------------------------------------

/// Validate config and produce a structural kernel descriptor.
/// Overload resolution: each kernel family has its own Config type,
/// so make_spec(FmhaBwdConvertDQConfig) is unambiguous.
consteval FmhaBwdConvertDQSpec make_spec(FmhaBwdConvertDQConfig cfg)
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

    FmhaBwdConvertDQSpec k{sig.dtype,
                           sig.hdim_q,
                           sig.mode,
                           algo.is_deterministic,
                           algo.pad_seqlen_q,
                           algo.pad_hdim_q,
                           algo.block_per_cu,
                           block_size};

    return k;
}

// Compile canary: GROUP mode exercises pad_seqlen_q and mode constraints.
// clang-format off
static_assert(make_spec(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                  .mode = FmhaMode::GROUP},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}
}).mode == FmhaMode::GROUP);
// clang-format on

} // namespace rocm_ck
