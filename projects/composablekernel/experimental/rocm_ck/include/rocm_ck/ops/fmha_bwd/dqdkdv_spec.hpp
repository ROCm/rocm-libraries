// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compile-time configuration and argument slot definitions for the FMHA BWD
// dQ/dK/dV kernel family — the main backward kernel that computes query, key,
// and value gradients via 5 GEMMs.
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

#include <cstdint>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Signature / Algorithm / Config / Kernel
// ---------------------------------------------------------------------------

/// Signature: describes WHAT the kernel computes (problem shape only).
/// dtype + head dimensions + batch/group mode.
struct FmhaBwdDQDKDVSignature
{
    DataType dtype; // fp16 or bf16 (Q/K/V/dO types; LSE/D/dQ_acc are float)
    int hdim_q;     // Q/K head dimension: 32, 64, 96, 128, 256
    int hdim_v;     // V head dimension: 32, 64, 96, 128, 256
    FmhaMode mode;  // batch or group
};

/// Algorithm: describes HOW the kernel executes (feature flags + tuning).
struct FmhaBwdDQDKDVAlgorithm
{
    // Feature flags — which variation of the computation
    FmhaBiasType bias_type = FmhaBiasType::NONE;
    bool has_bias_grad     = false;
    bool has_mask          = false;
    bool has_dropout       = false;
    bool is_deterministic  = false;

    // Tuning — padding and occupancy
    int pad_hdim_q   = 0;  // 0 (no pad), 1 (small pad), or 8 (full vec pad)
    int pad_hdim_v   = 0;  // 0, 1, or 8
    int block_per_cu = -1; // occupancy hint (-1 = auto, resolved in make_spec)
};

/// Config: user-facing Signature + Algorithm pair.
struct FmhaBwdDQDKDVConfig
{
    FmhaBwdDQDKDVSignature signature;
    FmhaBwdDQDKDVAlgorithm algorithm;
};

/// Validated kernel descriptor — structural type, safe for use as NTTP.
/// All optional/default values are resolved; no std::optional.
struct FmhaBwdDQDKDVSpec
{
    // From Signature
    DataType dtype;
    int hdim_q;
    int hdim_v;
    FmhaMode mode;

    // From Algorithm — feature flags
    FmhaBiasType bias_type;
    bool has_bias_grad;
    bool has_mask;
    bool has_dropout;
    bool is_deterministic;

    // From Algorithm — tuning
    int pad_hdim_q;
    int pad_hdim_v;
    int block_per_cu;

    // Computed tile geometry (architecture-dependent)
    int block_size; // num_warps * warp_size (e.g. 4 * 64 = 256)
    int block_n0;   // kN0: K-sequence tile size (for grid calculation)
};

// ---------------------------------------------------------------------------
// Named slot constants for generic rocm_ck::Args
// ---------------------------------------------------------------------------

/// Named tensor and scalar slot indices for the dQ/dK/dV kernel.
/// These map directly to indices in rocm_ck::Args::tensors[] and
/// rocm_ck::Args::scalars[]. The device bridge reads from these slots;
/// host code populates the same slots — named constants prevent
/// off-by-one errors.
namespace fmha_bwd_dqdkdv_slots {

// Tensor slots (indices into Args::tensors[])
constexpr int Q      = 0;
constexpr int K      = 1;
constexpr int V      = 2;
constexpr int LSE    = 3;
constexpr int DO     = 4;
constexpr int D      = 5;
constexpr int DQ_ACC = 6;
constexpr int DK     = 7;
constexpr int DV     = 8;
// Optional slots have fixed indices regardless of which features are enabled.
// Unused slots are simply not populated by host code — no slot remapping.
constexpr int BIAS    = 9;  // optional: present if bias_type != NONE
constexpr int DBIAS   = 10; // optional: present if has_bias_grad
constexpr int RANDVAL = 11; // optional: present if has_dropout

/// Minimum tensor slot count (max_used_index + 1) for a given config.
/// Slot indices are fixed (BIAS=9, DBIAS=10, RANDVAL=11) regardless of
/// which features are enabled — unused slots are simply not populated.
constexpr int requiredTensors(FmhaBwdDQDKDVSpec k)
{
    if(k.has_dropout)
        return RANDVAL + 1; // 12
    if(k.has_bias_grad)
        return DBIAS + 1; // 11
    if(k.bias_type != FmhaBiasType::NONE)
        return BIAS + 1; // 10
    return DV + 1;       // 9
}

// Scalar slots (indices into Args::scalars[])
constexpr int RAW_SCALE      = 0; // f32: attention scale (1/sqrt(hdim))
constexpr int SCALE          = 1; // f32: raw_scale * log2(e)
constexpr int NUM_HEAD_Q     = 2; // i32: number of Q heads
constexpr int NHEAD_RATIO_QK = 3; // i32: Q heads / K heads (for GQA/MQA)
constexpr int P_UNDROP       = 4; // f32: 1/(1-dropout_rate)
constexpr int RP_UNDROP      = 5; // f32: 1/p_undrop
constexpr int DROP_SEED      = 6; // u64: dropout RNG seed
constexpr int DROP_OFFSET    = 7; // u64: dropout RNG offset

/// Minimum scalar slot count (max_used_index + 1) for a given config.
constexpr int requiredScalars(FmhaBwdDQDKDVSpec k)
{
    if(k.has_dropout)
        return DROP_OFFSET + 1; // 8
    return NHEAD_RATIO_QK + 1;  // 4
}

} // namespace fmha_bwd_dqdkdv_slots

// ---------------------------------------------------------------------------
// make_spec — consteval validation
// ---------------------------------------------------------------------------

/// Validate config and produce a structural kernel descriptor.
/// Overload resolution: each kernel family has its own Config type,
/// so make_spec(FmhaBwdDQDKDVConfig) is unambiguous.
/// All compile-time constraints are checked here; invalid configs produce
/// a compile error with a descriptive message.
consteval FmhaBwdDQDKDVSpec make_spec(FmhaBwdDQDKDVConfig cfg)
{
    auto sig  = cfg.signature;
    auto algo = cfg.algorithm;

    // --- dtype validation ---
    if(sig.dtype != DataType::FP16 && sig.dtype != DataType::BF16)
        throw "FmhaBwdDQDKDV only supports FP16 or BF16"
              " (Q/K/V/dO types; LSE/D/dQ_acc are always float)";

    // --- head dimension validation ---
    if(sig.hdim_q != 32 && sig.hdim_q != 64 && sig.hdim_q != 96 && sig.hdim_q != 128 &&
       sig.hdim_q != 256)
        throw "hdim_q must be one of {32, 64, 96, 128, 256}";

    if(sig.hdim_v != 32 && sig.hdim_v != 64 && sig.hdim_v != 96 && sig.hdim_v != 128 &&
       sig.hdim_v != 256)
        throw "hdim_v must be one of {32, 64, 96, 128, 256}";

    // --- mode validation ---
    // Group mode uses variable-length sequences, which requires padding.
    // Note: unlike OGradDotO/ConvertDQ, DqDkDv does not have a separate
    // pad_seqlen_q flag — sequence padding is implied by pad_hdim_q/v.
    // pad_hdim_q/v must be nonzero for group mode to handle unaligned heads.
    if(sig.mode == FmhaMode::GROUP && algo.pad_hdim_q == 0 && algo.pad_hdim_v == 0)
        throw "group mode requires padding"
              " (pad_hdim_q and/or pad_hdim_v must be nonzero)";

    // --- feature flag validation ---
    if(algo.has_bias_grad && algo.bias_type == FmhaBiasType::NONE)
        throw "has_bias_grad requires bias_type != NONE";

    // --- padding validation ---
    if(algo.pad_hdim_q != 0 && algo.pad_hdim_q != 1 && algo.pad_hdim_q != 8)
        throw "pad_hdim_q must be 0, 1, or 8";

    if(algo.pad_hdim_v != 0 && algo.pad_hdim_v != 1 && algo.pad_hdim_v != 8)
        throw "pad_hdim_v must be 0, 1, or 8";

    // --- tile geometry (hardcoded for d128 gfx9 demo) ---
    // Config 4 from fmha_bwd.py: num_warps=4, warp_size=64, bn0=128.
    // Production would derive these from architecture + hdim.
    constexpr int demo_block_size = 256; // 4 warps * 64
    constexpr int demo_block_n0   = 128; // kN0 = bn0

    // --- block_per_cu default ---
    int resolved_block_per_cu = algo.block_per_cu;
    if(resolved_block_per_cu == -1)
        resolved_block_per_cu = 1; // d128 dQ/dK/dV is register-heavy

    if(resolved_block_per_cu <= 0)
        throw "block_per_cu must be positive (or -1 for auto)";

    // --- build the kernel descriptor ---
    FmhaBwdDQDKDVSpec k{
        .dtype            = sig.dtype,
        .hdim_q           = sig.hdim_q,
        .hdim_v           = sig.hdim_v,
        .mode             = sig.mode,
        .bias_type        = algo.bias_type,
        .has_bias_grad    = algo.has_bias_grad,
        .has_mask         = algo.has_mask,
        .has_dropout      = algo.has_dropout,
        .is_deterministic = algo.is_deterministic,
        .pad_hdim_q       = algo.pad_hdim_q,
        .pad_hdim_v       = algo.pad_hdim_v,
        .block_per_cu     = resolved_block_per_cu,
        .block_size       = demo_block_size,
        .block_n0         = demo_block_n0,
    };

    return k;
}

// Compile canary: dropout variant exercises bias/dropout/slot count paths.
// clang-format off
static_assert(fmha_bwd_dqdkdv_slots::requiredTensors(make_spec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 12);
// clang-format on

} // namespace rocm_ck
