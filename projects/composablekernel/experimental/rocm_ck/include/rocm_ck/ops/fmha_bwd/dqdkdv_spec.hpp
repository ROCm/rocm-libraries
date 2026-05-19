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

#include <algorithm>
#include <cstdint>
#include <iterator>

namespace rocm_ck {

// ---------------------------------------------------------------------------
// Tile geometry config (consteval generation)
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

// ---------------------------------------------------------------------------
// Base tile table (compact tuning parameters)
// ---------------------------------------------------------------------------

/// Base tile geometry — the ONLY parameters that are empirically tuned
/// per (arch, dtype, head-dimension tier). All other tile fields are derived
/// from these via invariant rules documented below.
struct FmhaBwdBaseTile
{
    int bm0;       // M-dimension block tile
    int bn0;       // N-dimension block tile
    int bk4;       // GEMM4 K-unroll (must divide bn0)
    int occupancy; // target occupancy (-1 = auto)
    int max_seq_q; // maximum Q sequence length (0 = unlimited)
};

/// Entry in the base tile table: maps an effective head dimension to its
/// empirically tuned base tile geometry.
struct FmhaBwdBaseTileEntry
{
    int hdim;
    FmhaBwdBaseTile tile;
};

/// GFX9 fp16/bf16 base tiles, indexed by effective head dimension.
/// The effective hdim for a (hdim_q, hdim_v) pair is max(hdim_q, hdim_v),
/// because both GEMM0 (K-unroll=hdim_q) and GEMM2 (K-unroll=hdim_v) share
/// the same bm0 and bn0, so the resource constraint is the larger dimension.
// clang-format off
inline constexpr FmhaBwdBaseTileEntry
GFX9_FP16_DQDKDV_BASE_TILES[] = {
    { 32, { 32, 128, 64, 1, 0}},
    { 64, { 32, 128, 32, 1, 0}},
    { 96, { 32, 128, 32, 1, 0}},
    {128, { 16, 128, 32, 1, 0}},
    {256, { 16,  64, 32, 1, 0}},
};
// clang-format on

inline constexpr int GFX9_FP16_DQDKDV_BASE_TILES_COUNT =
    static_cast<int>(std::size(GFX9_FP16_DQDKDV_BASE_TILES));

// ---------------------------------------------------------------------------
// Tile config generation (consteval derivation)
// ---------------------------------------------------------------------------

/// Compute GEMM4 block warp distribution (rm2, rn2) from (bm0, hdim_q).
///
/// GEMM4 computes dQ = dS @ K with output shape (bm0 × hdim_q).
/// The 4 warps (rm2 * rn2 * rk2, rk2=1) must tile this output such that:
///   rm2 * wm0(=16) divides bm0
///   rn2 * wn0(=16) divides hdim_q
///
/// Preference: (1,4) > (2,2) > (4,1) — maximise N-parallelism for
/// large head dimensions, fall back to balanced or M-heavy splits.
struct Gemm4Warps
{
    int rm2;
    int rn2;
};

consteval Gemm4Warps computeGemm4Warps(int bm0, int hdim_q)
{
    if(bm0 <= 0 || hdim_q <= 0)
        throw "computeGemm4Warps: bm0 and hdim_q must be positive";

    constexpr int wm0 = 16;
    constexpr int wn0 = 16;

    // (1,4): best N-parallelism — when hdim_q is divisible by 64
    if(bm0 % (1 * wm0) == 0 && hdim_q % (4 * wn0) == 0)
        return {1, 4};
    // (2,2): balanced — when hdim_q is divisible by 32 but not 64
    if(bm0 % (2 * wm0) == 0 && hdim_q % (2 * wn0) == 0)
        return {2, 2};
    // (4,1): M-heavy fallback
    if(bm0 % (4 * wm0) == 0 && hdim_q % (1 * wn0) == 0)
        return {4, 1};

    throw "no valid GEMM4 warp distribution for this (bm0, hdim_q) combination";
}

/// Generate a complete tile config from (hdim_q, hdim_v) and a base tile.
///
/// Invariant derivation rules (from CK Tile pipeline — always true):
///   bk0 = hdim_q  — GEMM0 (Q@K^T) K-unroll = QK head dim
///   bk1 = bm0     — GEMM1 (P^T@dO) K-unroll = M-dim
///   bk2 = hdim_v  — GEMM2 (dO@V^T) K-unroll = V head dim
///   bk3 = bm0     — GEMM3 (dS^T@Q) K-unroll = M-dim
///
/// GFX9 fp16/bf16 constants (invariant across all head dims):
///   GEMM0/2: rm0=1, rn0=4, rk0=1, wm0=16, wn0=16, wk0=32
///   GEMM1/3: rm1=4, rn1=1, rk1=1, wm1=16, wn1=16, wk1=16
///   GEMM4:   rk2=1, warp tile = (wm0, wn0, min(wk0, bk4))
consteval FmhaBwdDQDKDVTileConfig
generateTileConfig(int hdim_q, int hdim_v, const FmhaBwdBaseTile base)
{
    auto warps = computeGemm4Warps(base.bm0, hdim_q);

    return FmhaBwdDQDKDVTileConfig{
        .hdim_q = hdim_q,
        .hdim_v = hdim_v,
        .bm0    = base.bm0,
        .bn0    = base.bn0,
        .bk0    = hdim_q,   // Rule: GEMM0 K-unroll = hdim_q
        .bk1    = base.bm0, // Rule: GEMM1 K-unroll = bm0
        .bk2    = hdim_v,   // Rule: GEMM2 K-unroll = hdim_v
        .bk3    = base.bm0, // Rule: GEMM3 K-unroll = bm0
        .bk4    = base.bk4,
        // GEMM0/2 warps (constant for GFX9 fp16/bf16)
        .rm0 = 1,
        .rn0 = 4,
        .rk0 = 1,
        .wm0 = 16,
        .wn0 = 16,
        .wk0 = 32,
        // GEMM1/3 warps (constant for GFX9 fp16/bf16)
        .rm1 = 4,
        .rn1 = 1,
        .rk1 = 1,
        .wm1 = 16,
        .wn1 = 16,
        .wk1 = 16,
        // GEMM4 warps (computed from bm0 and hdim_q)
        .rm2       = warps.rm2,
        .rn2       = warps.rn2,
        .rk2       = 1,
        .occupancy = base.occupancy,
        .max_seq_q = base.max_seq_q,
    };
}

/// Look up base tile for a given effective head dimension.
consteval FmhaBwdBaseTile getBaseTile(int effective_hdim, DataType dtype, GpuTarget target)
{
    constexpr auto gfx9_targets = TargetSet::only(GpuTarget::gfx90a, GpuTarget::gfx942);
    if(gfx9_targets.contains(target) && (dtype == DataType::FP16 || dtype == DataType::BF16))
    {
        for(int i = 0; i < GFX9_FP16_DQDKDV_BASE_TILES_COUNT; ++i)
        {
            if(GFX9_FP16_DQDKDV_BASE_TILES[i].hdim == effective_hdim)
                return GFX9_FP16_DQDKDV_BASE_TILES[i].tile;
        }
        throw "no base tile for this head dimension on this arch";
    }
    throw "unsupported (dtype, arch) combination for tile config lookup";
}

/// Look up tile geometry for dQ/dK/dV given problem shape and target arch.
/// Returns the matching tile config. Throws at compile time if no config exists.
///
/// Supports both symmetric (hdim_q == hdim_v) and asymmetric head dimensions.
/// Base tile is looked up by max(hdim_q, hdim_v); remaining fields are derived
/// via invariant rules. Some asymmetric combinations may fail at compile time
/// if no valid GEMM4 warp distribution exists (see computeGemm4Warps).
consteval FmhaBwdDQDKDVTileConfig
getTileConfig(int hdim_q, int hdim_v, DataType dtype, GpuTarget target)
{
    int effective_hdim = std::max(hdim_q, hdim_v);
    auto base          = getBaseTile(effective_hdim, dtype, target);
    return generateTileConfig(hdim_q, hdim_v, base);
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
