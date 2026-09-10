// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shape -> kernel-variant selection for the Flash-Attention 2 SDPA engine.
// Mirrors the pattern in asm_sdpa_engine, which ships ~290 kernel objects and
// carries per-variant launch geometry alongside the plan.
//
// WHY MORE THAN ONE VARIANT
// No single tiling is fastest across attention shapes. Measured on MI300X
// (gfx942, ROCm 7.2, FP16) over 21 shapes against the CK/AITER backend, the
// best SINGLE variant reaches 0.66x while selecting per shape reaches 1.043x.
// An exhaustive 18-variant x 21-shape sweep put the oracle ceiling at 0.998x,
// so almost all of that gap was selection rather than kernel quality.
//
// SELECTION KEY
// The number of CTAs the shape produces, compared against the device CU count
// -- NOT the sequence length. Shapes that underperformed with a sequence-length
// rule had nothing in common in S; they simply produced too few CTAs to fill
// the GPU once a CTA is 512 threads. ceil(S/qPerCta)*B*H captures batch, heads
// and sequence length together.
//
// VARIANT NAMING: w<waves>q<queryGroups>k<keyGroups>
//   queries/CTA = waves * queryGroups * 16
//   blockDim    = waves * 64
//   BK (keys per tile) = keyGroups * 16

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

namespace hip_flash2_engine
{

/// Launch geometry for one compiled kernel variant. blockDim and qPerCta MUST
/// match the values the corresponding .co was compiled with: launching a
/// 512-thread variant with 64 threads silently computes wrong results, and the
/// reverse fails with hipErrorLaunchFailure (719).
struct Flash2Variant
{
    const char* tag; ///< selects kernels/hip_flash2_fwd_<arch>_<tag>.co
    unsigned int blockDim; ///< threads per CTA (waves * 64)
    unsigned int qPerCta; ///< query rows per CTA (waves * queryGroups * 16)
};

// Variants shipped for gfx942. Keep in sync with the kernels/ directory and
// with the F2_WAVES / F2_QG / F2_KG values each object was built with.
constexpr Flash2Variant K_FLASH2_W4Q1K4{"w4q1k4", 256, 64}; // tiny grids
constexpr Flash2Variant K_FLASH2_W8Q1K4{"w8q1k4", 512, 128}; // short causal
constexpr Flash2Variant K_FLASH2_W8Q2K4{"w8q2k4", 512, 256}; // general case
constexpr Flash2Variant K_FLASH2_W8Q3K2{"w8q3k2", 512, 384}; // BK=32, long S
constexpr Flash2Variant K_FLASH2_W8Q3K4{"w8q3k4", 512, 384}; // BK=64, D=64 large batch

/// Legacy geometry: the historical single-kernel object is __launch_bounds__(64)
/// with a 64-query tile. Used when no per-variant object is installed.
constexpr Flash2Variant K_FLASH2_LEGACY{"", 64, 64};

/// Whether split-K is actually executed. buildPlan() currently forces
/// splitK = 1 because execute() has no merge-pass launch or workspace pointer,
/// so selection must NOT credit a split that will not happen: doing so shifts
/// the variant choice for shapes whose CTA count only clears the tiny-grid
/// threshold once multiplied by the split factor. Measured over a 560-shape
/// sweep, 90 D=128 shapes with ctas256 in [32, 96] change variant purely from
/// that credit, and the only such shape we have measured runs 0.478x.
///
/// Flip this to true in the same change that wires split-K through execute().
constexpr bool K_FLASH2_SPLITK_EXECUTES = false;

/// Result of variant selection, including any split-K decision.
struct Flash2Selection
{
    Flash2Variant variant = K_FLASH2_W8Q2K4;
    int splitK = 1; ///< 1 = single pass; >1 = partition the KV axis
};

/// Number of CTAs a given queries-per-CTA tiling produces for this shape.
inline long long flash2CtaCount(int batch, int numHeadsQ, int seqLenQ, unsigned int qPerCta)
{
    const long long tiles = (static_cast<long long>(seqLenQ) + qPerCta - 1) / qPerCta;
    return tiles * static_cast<long long>(batch) * static_cast<long long>(numHeadsQ);
}

/// Choose the kernel variant (and split factor) for a shape.
///
/// @param cuCount device compute-unit count. Thresholds were calibrated on a
///                304-CU part and are scaled linearly for other devices.
inline Flash2Selection selectFlash2Config(
    int batch, int numHeadsQ, int seqLenQ, int headDim, bool causal, int cuCount = 304)
{
    Flash2Selection sel;

    const long long ctas256 = flash2CtaCount(batch, numHeadsQ, seqLenQ, 256);
    const long long ctas384 = flash2CtaCount(batch, numHeadsQ, seqLenQ, 384);

    const auto scaled = [cuCount](long long v) -> long long {
        return (v * static_cast<long long>(cuCount)) / 304;
    };

    // --- Split-K: only when the base grid cannot fill the GPU ---------------
    // Partitioning the KV axis multiplies grid parallelism but costs an fp32
    // partial buffer plus a merge pass, so it pays only on starved grids.
    // Measured B=1 H=8 S=2048 noncausal: 83 -> 200 TFLOPS at splitK=4. On a
    // full grid it costs up to 19%, and splitK=8 never won on any shape.
    if(ctas256 < scaled(128) && seqLenQ >= 1024 && headDim == 128)
    {
        const long long denom = (ctas256 > 0) ? ctas256 : 1;
        const long long nsplit = std::clamp<long long>((scaled(256) + denom - 1) / denom, 2, 4);
        // Record the split factor but do NOT return: the variant still has to
        // be chosen by the rules below. Returning here pinned every starved
        // shape to w8q2k4 and suppressed the tiny-grid rule underneath it
        // (reported by S. Reeder). w8q2k4 was only ever the *partner* of
        // splitK=4, not the right single-pass choice for these shapes.
        sel.splitK = static_cast<int>(nsplit);
    }

    // --- Tiny grid: an 8-wave CTA cannot be filled at all -------------------
    // Note the direction: small grids want FEWER waves, not more/smaller CTAs.
    // Measured, the 8-wave variant still wins on merely-starved grids (185 vs
    // 110 TFLOPS) because the kernel is bandwidth-bound and smaller CTAs
    // multiply K/V re-reads. Only a genuinely tiny grid prefers 4 waves.
    // Test against the grid that will ACTUALLY run. Split-K multiplies CTA
    // count by splitK, so a shape that is starved single-pass may be full once
    // split -- B=1 H=8 S=2048 has 64 CTAs but 256 at splitK=4, and 256 CTAs
    // want the 8-wave variant (measured 191 TFLOPS). That reasoning only holds
    // when the split is executed; while it is not, crediting it picks the
    // 8-wave variant for a grid that stays starved. See
    // K_FLASH2_SPLITK_EXECUTES.
    const long long effectiveCtas = K_FLASH2_SPLITK_EXECUTES ? ctas256 * sel.splitK : ctas256;
    if(effectiveCtas < scaled(100))
    {
        sel.variant = K_FLASH2_W4Q1K4;
        return sel;
    }

    // --- head_dim 64 --------------------------------------------------------
    // Half the per-CTA working set, so the wider 384-query tile pays off, but
    // only once S is long enough to amortize it.
    if(headDim == 64)
    {
        sel.variant
            = (batch * numHeadsQ >= 64 && seqLenQ >= 4096) ? K_FLASH2_W8Q3K4 : K_FLASH2_W8Q2K4;
        return sel;
    }

    // --- causal -------------------------------------------------------------
    if(causal)
    {
        if(seqLenQ <= 1024 && ctas256 < scaled(200))
        {
            sel.variant = K_FLASH2_W8Q1K4;
        }
        else if(seqLenQ >= 1024 && ctas384 >= scaled(256))
        {
            sel.variant = K_FLASH2_W8Q3K2;
        }
        else
        {
            sel.variant = K_FLASH2_W8Q2K4;
        }
        return sel;
    }

    // --- non-causal ---------------------------------------------------------
    // Every KV tile is full, so the smaller BK=32 tile only pays at low head
    // count or very long sequences. The discriminator is HEAD COUNT, not
    // batch*heads: the win case (B=2 H=16) and the loss case (B=1 H=32) both
    // have batch*heads == 32, so a rule on the product cannot separate them.
    // Both conditions below select the same variant, so they are combined
    // into one branch rather than two identical bodies.
    if((numHeadsQ <= 16 && seqLenQ >= 3072) || seqLenQ >= 8192)
    {
        sel.variant = K_FLASH2_W8Q3K2;
    }
    else
    {
        sel.variant = K_FLASH2_W8Q2K4;
    }
    return sel;
}

/// Workspace bytes required for a split-K plan (0 when splitK == 1).
/// Layout: fp32 partial outputs [B, H, splitK, Sq, D] followed by per-split
/// softmax statistics m and l, each [B, H, splitK, Sq].
inline size_t flash2WorkspaceBytes(int batch, int numHeadsQ, int seqLenQ, int headDim, int splitK)
{
    if(splitK <= 1)
    {
        return 0;
    }
    const size_t rows = static_cast<size_t>(batch) * static_cast<size_t>(numHeadsQ)
                        * static_cast<size_t>(splitK) * static_cast<size_t>(seqLenQ);
    return rows * static_cast<size_t>(headDim) * sizeof(float) // partial O
           + 2 * rows * sizeof(float); // per-split m and l
}

} // namespace hip_flash2_engine
