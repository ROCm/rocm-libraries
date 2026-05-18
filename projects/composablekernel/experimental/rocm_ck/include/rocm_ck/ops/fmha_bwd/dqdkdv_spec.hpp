// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compile-time configuration and argument slot definitions for the FMHA BWD
// dQ/dK/dV kernel family -- the main backward kernel that computes query, key,
// and value gradients via 5 GEMMs.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types, consteval makeSpec() factory, and named slot
// constants. No runtime code, no HIP dependency.
//
// Compilation boundary:
//   _spec.hpp (this) -- consteval factory + slot constants (both passes)
//   _api.hpp         -- host-only helpers: grid_size (host pass only, #error on device)
//   _dev.hpp         -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#include <rocm_ck/ops/fmha_bwd/common.hpp>

#include <rocm_ck/arch_properties.hpp>
#include <rocm_ck/datatype_utils.hpp>

#include <cstdint>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Tile geometry config (consteval table)
// ---------------------------------------------------------------------------

/// Tile geometry for the dQ/dK/dV backward kernel.
/// All fields are plain integers; the device bridge converts them to
/// CK Tile sequence<> types. Source of truth: fmha_bwd.py get_dq_dk_dv_tiles().
struct FmhaBwdDQDKDVTileConfig
{
    int hdim_q; // head dimension for Q (lookup key)
    int hdim_v; // head dimension for V (lookup key)

    // Block tile: <bm0, bn0, bk0, bk1, bk2, bk3, bk4, hdim_q, hdim_v>
    int bm0, bn0, bk0, bk1, bk2, bk3, bk4;

    // GEMM0/2 (S = Q@K^T, dP = dO@V^T): block warps + warp tile
    int rm0, rn0, rk0;
    int wm0, wn0, wk0;

    // GEMM1/3 (dV = P^T@dO, dK = dS^T@Q): block warps + warp tile
    int rm1, rn1, rk1;
    int wm1, wn1, wk1;

    // GEMM4 (dQ = dS@K): block warps
    // GEMM4 warp tile = (wm0, wn0, min(wk0, bk4)) per fmha_bwd.py
    int rm2, rn2, rk2;

    // Tuning
    int occupancy; // target occupancy (-1 = auto)
    int max_seq_q; // maximum Q sequence length (0 = unlimited)

    // Derived quantities
    constexpr int num_warps() const { return rm0 * rn0 * rk0; }
    constexpr int block_size(GpuTarget target) const { return num_warps() * wavefrontSize(target); }
};

/// GFX9 tile configs for fp16/bf16, indexed by (hdim_q, hdim_v).
/// Source: fmha_bwd.py KernelComponentFactoryGfx9.get_dq_dk_dv_tiles("fp16", "f")
// clang-format off
inline constexpr FmhaBwdDQDKDVTileConfig GFX9_FP16_DQDKDV_TILES[] = {
    //                      hdq hdv  bm0  bn0  bk0  bk1  bk2  bk3  bk4  rm0 rn0 rk0  wm0 wn0 wk0  rm1 rn1 rk1  wm1 wn1 wk1  rm2 rn2 rk2  occ msq
    FmhaBwdDQDKDVTileConfig{ 32, 32,  32, 128,  32,  32,  32,  32,  64,   1,  4,  1,  16, 16, 32,   4,  1,  1,  16, 16, 16,   2,  2,  1,    1,  0},
    FmhaBwdDQDKDVTileConfig{ 64, 64,  32, 128,  64,  32,  64,  32,  32,   1,  4,  1,  16, 16, 32,   4,  1,  1,  16, 16, 16,   1,  4,  1,    1,  0},
    FmhaBwdDQDKDVTileConfig{ 96, 96,  32, 128,  96,  32,  96,  32,  32,   1,  4,  1,  16, 16, 32,   4,  1,  1,  16, 16, 16,   2,  2,  1,    1,  0},
    FmhaBwdDQDKDVTileConfig{128,128,  16, 128, 128,  16, 128,  16,  32,   1,  4,  1,  16, 16, 32,   4,  1,  1,  16, 16, 16,   1,  4,  1,    1,  0},
    FmhaBwdDQDKDVTileConfig{256,256,  16,  64, 256,  16, 256,  16,  32,   1,  4,  1,  16, 16, 32,   4,  1,  1,  16, 16, 16,   1,  4,  1,    1,  0},
};
// clang-format on

inline constexpr int GFX9_FP16_DQDKDV_TILES_COUNT =
    sizeof(GFX9_FP16_DQDKDV_TILES) / sizeof(GFX9_FP16_DQDKDV_TILES[0]);

/// Look up tile geometry for dQ/dK/dV given problem shape and target arch.
/// Returns the matching tile config. Throws at compile time if no config exists.
consteval FmhaBwdDQDKDVTileConfig
getTileConfig(int hdim_q, int hdim_v, DataType dtype, GpuTarget target)
{
    // GFX9 (gfx90a, gfx942): fp16/bf16 tile configs
    constexpr auto gfx9_targets = TargetSet::only(GpuTarget::gfx90a, GpuTarget::gfx942);
    if(gfx9_targets.contains(target) && (dtype == DataType::FP16 || dtype == DataType::BF16))
    {
        for(int i = 0; i < GFX9_FP16_DQDKDV_TILES_COUNT; ++i)
        {
            const auto& t = GFX9_FP16_DQDKDV_TILES[i];
            if(t.hdim_q == hdim_q && t.hdim_v == hdim_v)
                return t;
        }
    }

    throw "no tile config for this (hdim_q, hdim_v, dtype, arch) combination";
}

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
    // Feature flags -- which variation of the computation
    FmhaBiasType bias_type = FmhaBiasType::NONE;
    bool has_bias_grad     = false;
    bool has_mask          = false;
    bool has_dropout       = false;
    bool is_deterministic  = false;

    // Tuning -- padding and occupancy
    int pad_hdim_q   = 0;  // 0 (no pad), 1 (small pad), or 8 (full vec pad)
    int pad_hdim_v   = 0;  // 0, 1, or 8
    int block_per_cu = -1; // occupancy hint (-1 = auto, resolved in makeSpec)
};

/// Config: user-facing Signature + Algorithm pair.
struct FmhaBwdDQDKDVConfig
{
    FmhaBwdDQDKDVSignature signature;
    FmhaBwdDQDKDVAlgorithm algorithm;
};

/// Validated kernel descriptor -- structural type, safe for use as NTTP.
/// All optional/default values are resolved; no std::optional.
struct FmhaBwdDQDKDVSpec
{
    // From Signature
    DataType dtype;
    int hdim_q;
    int hdim_v;
    FmhaMode mode;

    // From Algorithm -- feature flags
    FmhaBiasType bias_type;
    bool has_bias_grad;
    bool has_mask;
    bool has_dropout;
    bool is_deterministic;

    // From Algorithm -- tuning
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
/// host code populates the same slots -- named constants prevent
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
// Unused slots are simply not populated by host code -- no slot remapping.
constexpr int BIAS    = 9;  // optional: present if bias_type != NONE
constexpr int DBIAS   = 10; // optional: present if has_bias_grad
constexpr int RANDVAL = 11; // optional: present if has_dropout

// Group-mode tensor slots (indices into Args::tensors[])
// These provide per-batch sequence start offsets and actual lengths
// for variable-length sequences.
constexpr int SEQSTART_Q = 12; // const int32_t*: Q-sequence start offsets [batch+1]
constexpr int SEQSTART_K = 13; // const int32_t*: K-sequence start offsets [batch+1]
constexpr int SEQLEN_Q   = 14; // const int32_t*: per-batch actual Q-lengths [batch]
constexpr int SEQLEN_K   = 15; // const int32_t*: per-batch actual K-lengths [batch]

/// Minimum tensor slot count (max_used_index + 1) for a given config.
/// Slot indices are fixed (BIAS=9, DBIAS=10, RANDVAL=11) regardless of
/// which features are enabled -- unused slots are simply not populated.
constexpr int requiredTensors(FmhaBwdDQDKDVSpec k)
{
    // Start with the highest optional feature slot used
    int n = DV + 1; // 9 (base: Q through DV)
    if(k.bias_type != FmhaBiasType::NONE)
        n = BIAS + 1; // 10
    if(k.has_bias_grad)
        n = DBIAS + 1; // 11
    if(k.has_dropout)
        n = RANDVAL + 1; // 12

    // Group mode adds seqstart/seqlen slots after all feature slots
    if(k.mode == FmhaMode::GROUP)
        n = SEQLEN_K + 1; // 16 (always the highest slot)

    return n;
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
// Mask scalar slots -- present only when has_mask=true.
// Indices are fixed regardless of dropout; unused slots are not populated.
constexpr int WINDOW_SIZE_LEFT  = 8;  // i32: left context window (-1 = unlimited)
constexpr int WINDOW_SIZE_RIGHT = 9;  // i32: right context window (0 = causal)
constexpr int MASK_TYPE         = 10; // i32: GenericAttentionMaskEnum cast to int
// Batch count -- present only when is_deterministic=true AND mode==BATCH.
// CK Tile's persistent kernel reads kargs.batch for total_jobs computation
// (fmha_bwd_kernel.hpp:752: total_heads = kargs.batch * kargs.num_head_q).
// In group mode the kernel derives per-batch counts from seqstart pointers,
// so this slot is unused there.
constexpr int BATCH_SIZE = 11; // i32: batch count (deterministic batch mode only)

} // namespace fmha_bwd_dqdkdv_slots

/// Single source of truth for "does this spec use the BATCH_SIZE scalar slot".
/// Used by requiredScalars(), validateArgs(), and the device bridge so the
/// predicate cannot drift between sites. CK Tile's persistent kernel
/// (kUsePersistent = is_deterministic && !is_group_mode) is the only path
/// that reads kargs.batch; group mode derives batch implicitly from seqstart.
constexpr bool usesBatchSizeSlot(FmhaBwdDQDKDVSpec k)
{
    return k.is_deterministic && k.mode == FmhaMode::BATCH;
}

namespace fmha_bwd_dqdkdv_slots {

/// Minimum scalar slot count (max_used_index + 1) for a given config.
constexpr int requiredScalars(FmhaBwdDQDKDVSpec k)
{
    if(usesBatchSizeSlot(k))
        return BATCH_SIZE + 1; // 12 (dominates mask/dropout slots)
    if(k.has_mask)
        return MASK_TYPE + 1; // 11 (covers dropout slots [4..7] since 11 > 8)
    if(k.has_dropout)
        return DROP_OFFSET + 1; // 8
    return NHEAD_RATIO_QK + 1;  // 4
}

} // namespace fmha_bwd_dqdkdv_slots

// ---------------------------------------------------------------------------
// makeSpec -- consteval validation
// ---------------------------------------------------------------------------

/// Validate config and produce a structural kernel descriptor.
/// Overload resolution: each kernel family has its own Config type,
/// so makeSpec(FmhaBwdDQDKDVConfig) is unambiguous.
/// All compile-time constraints are checked here; invalid configs produce
/// a compile error with a descriptive message.
consteval FmhaBwdDQDKDVSpec makeSpec(FmhaBwdDQDKDVConfig cfg)
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
    // pad_seqlen_q flag -- sequence padding is implied by pad_hdim_q/v.
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

    // --- tile geometry (from consteval lookup table) ---
    // getTileConfig() returns the architecture-specific tile geometry for
    // the given (hdim_q, hdim_v, dtype, target). Currently only GFX9 fp16/bf16
    // configs are populated.
    constexpr GpuTarget target = GpuTarget::gfx942;
    auto tile                  = getTileConfig(sig.hdim_q, sig.hdim_v, sig.dtype, target);

    // --- block_per_cu default ---
    int resolved_block_per_cu = algo.block_per_cu;
    if(resolved_block_per_cu == -1)
        resolved_block_per_cu = tile.occupancy;

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
        .block_size       = tile.block_size(target),
        .block_n0         = tile.bn0,
    };

    return k;
}

// Compile canaries: each variant exercises a distinct slot-count path so
// requiredTensors() / requiredScalars() drift is caught at build time.
// clang-format off

// Plain BATCH: Q..DV only -> 9 tensors, base scalars only -> 4 scalars.
static_assert(fmha_bwd_dqdkdv_slots::requiredTensors(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})) == 9);
static_assert(fmha_bwd_dqdkdv_slots::requiredScalars(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})) == 4);

// ALiBi BATCH: bias slot active -> 10 tensors.
static_assert(fmha_bwd_dqdkdv_slots::requiredTensors(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ALIBI,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 10);

// Elementwise bias + bias gradient: BIAS+DBIAS slots active -> 11 tensors.
static_assert(fmha_bwd_dqdkdv_slots::requiredTensors(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                      .has_bias_grad = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 11);

// Causal mask + deterministic BATCH: BATCH_SIZE dominates -> 12 scalars.
static_assert(fmha_bwd_dqdkdv_slots::requiredScalars(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.has_mask = true,
                      .is_deterministic = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 12);

// Plain deterministic BATCH (no mask): BATCH_SIZE still required -> 12 scalars.
static_assert(fmha_bwd_dqdkdv_slots::requiredScalars(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 12);

// Dropout BATCH: RANDVAL slot active -> 12 tensors, dropout scalars -> 8.
static_assert(fmha_bwd_dqdkdv_slots::requiredTensors(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 12);
static_assert(fmha_bwd_dqdkdv_slots::requiredScalars(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true,
                      .pad_hdim_q = 8, .pad_hdim_v = 8}})) == 8);

// GROUP mode always extends to SEQLEN_K slot -> 16 tensors regardless of
// optional features (the seqstart/seqlen tail dominates).
static_assert(fmha_bwd_dqdkdv_slots::requiredTensors(makeSpec(
    FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16,
                      .hdim_q = 128, .hdim_v = 128,
                      .mode = FmhaMode::GROUP},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})) == 16);
// clang-format on

} // namespace rocm_ck
