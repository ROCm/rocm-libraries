// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU target properties, target set (consteval bitset), and wave tile validation.
// Pure C++20 constexpr/consteval. No runtime code, no CK deps.
//
// This header is the single source of truth for GPU target metadata across
// CK Tile, rocm_ck, and the dispatcher. It is intentionally free of device
// code, HIP runtime, and CK Tile dependencies so it can be used in both
// host-side consteval validation and device-side template selection.
//
// See also (verification anchor, not a source of truth):
//   CK Tile MMA selector specializations under
//   core/arch/mma/mfma/mfma_gfx9.hpp and wmma/wmma_gfx11.hpp
//   provide the authoritative list of supported wave tile shapes; the table
//   in isValidWaveTile() below mirrors those specializations.

#pragma once

#include <ck_common/datatype.hpp>
#include <ck_common/gpu_target.hpp>
#include <ck_common/platform.hpp>

#include <bit>
#include <cstdint>

namespace ck_common {

// ============================================================================
// Per-target properties
// ============================================================================

/// ISA architecture family.
///
/// CDNA = data-center compute (Instinct GPUs), MFMA instructions, wave64.
/// RDNA = gaming/workstation (Radeon GPUs), WMMA instructions, wave32.
enum struct ArchFamily
{
    CDNA,
    RDNA
};

/// Consteval properties for a single GPU target.
struct TargetProperties
{
    int wavefront_size;
    ArchFamily arch_family;
};

/// Single source of truth for per-target metadata.
constexpr TargetProperties properties(GpuTarget target)
{
    switch(target)
    {
    case GpuTarget::gfx90a: return {.wavefront_size = 64, .arch_family = ArchFamily::CDNA};
    case GpuTarget::gfx942: return {.wavefront_size = 64, .arch_family = ArchFamily::CDNA};
    case GpuTarget::gfx950: return {.wavefront_size = 64, .arch_family = ArchFamily::CDNA};
    case GpuTarget::gfx1100: return {.wavefront_size = 32, .arch_family = ArchFamily::RDNA};
    case GpuTarget::gfx1101: return {.wavefront_size = 32, .arch_family = ArchFamily::RDNA};
    case GpuTarget::gfx1102: return {.wavefront_size = 32, .arch_family = ArchFamily::RDNA};
    case GpuTarget::gfx1150: return {.wavefront_size = 32, .arch_family = ArchFamily::RDNA};
    case GpuTarget::gfx1151: return {.wavefront_size = 32, .arch_family = ArchFamily::RDNA};
    case GpuTarget::_count:
    default: CK_COMMON_TRAP();
    }
}

/// True if the target uses CDNA architecture (MFMA instructions, wave64).
constexpr bool isCDNA(GpuTarget target)
{
    return properties(target).arch_family == ArchFamily::CDNA;
}

/// True if the target uses RDNA architecture (WMMA instructions, wave32).
constexpr bool isRDNA(GpuTarget target)
{
    return properties(target).arch_family == ArchFamily::RDNA;
}

/// Wavefront size for a given GPU target.
constexpr int wavefrontSize(GpuTarget target) { return properties(target).wavefront_size; }

// ============================================================================
// TargetSet -- consteval bitset over GpuTarget values
// ============================================================================

/// Compile-time set of GPU targets. Structural type (usable as NTTP).
///
/// Storage: each GpuTarget maps to a bit in a uint64_t.
///
/// Named constructors express a 3-level target hierarchy:
///   Architecture: TargetSet::cdna(), TargetSet::rdna(), TargetSet::all()
///   Family:       TargetSet::family_gfx9(), family_gfx94(), family_gfx11(), family_gfx115()
///   Specific:     TargetSet::only(GpuTarget::gfx942)
///
/// See also (not authoritative -- CK Tile names may drift): these sets are
/// the ck_common analogue of CK Tile's enable_if_target_arch_*_t,
/// enable_if_target_family_*_t, and enable_if_target_id_t selectors and the
/// __gfx11__ / __gfx115__ preprocessor groupings.
struct TargetSet
{
    uint64_t bits = 0;

    // ---- Bit index mapping ------------------------------------------------
    // Bit positions equal GpuTarget enum values. Adding a target:
    //   1. Add enum value to GpuTarget before _count
    //   2. Add to architecture/family named constructors

    static constexpr int NUM_TARGETS = static_cast<int>(GpuTarget::_count);

    static constexpr int bitIndex(GpuTarget target)
    {
        if(target >= GpuTarget::_count)
        {
            // GpuTarget out of range -- value must be a valid enum member, not '_count'
            CK_COMMON_TRAP();
        }
        return static_cast<int>(target);
    }

    static constexpr GpuTarget targetAt(int index)
    {
        if(index < 0 || index >= NUM_TARGETS)
        {
            // TargetSet index out of range [0, NUM_TARGETS)
            CK_COMMON_TRAP();
        }
        return static_cast<GpuTarget>(index);
    }

    // ---- Constructors -----------------------------------------------------

    /// Default: empty set.
    constexpr TargetSet() = default;

    /// Construction from a single GpuTarget is explicit to avoid silent
    /// implicit conversion in mixed GpuTarget/TargetSet comparisons. Use
    /// TargetSet::of(target) for the named factory.
    explicit constexpr TargetSet(GpuTarget target) : bits(uint64_t{1} << bitIndex(target)) {}

    /// Named factory equivalent to the explicit single-target constructor.
    static constexpr TargetSet of(GpuTarget target) { return TargetSet{target}; }

    // ---- Named constructors: architecture level ---------------------------

    /// All real GPU targets.
    static constexpr TargetSet all() { return from_bits((uint64_t{1} << NUM_TARGETS) - 1); }

    /// All CDNA targets (MFMA, wave64): gfx90a, gfx942, gfx950.
    static constexpr TargetSet cdna()
    {
        return only(GpuTarget::gfx90a, GpuTarget::gfx942, GpuTarget::gfx950);
    }

    /// All RDNA targets (WMMA, wave32): gfx1100, gfx1101, gfx1102, gfx1150, gfx1151.
    static constexpr TargetSet rdna()
    {
        return only(GpuTarget::gfx1100, GpuTarget::gfx1101, GpuTarget::gfx1102)
            .union_with(only(GpuTarget::gfx1150, GpuTarget::gfx1151));
    }

    // ---- Named constructors: family level ---------------------------------
    // Matches CK Tile's __gfx9__, __gfx94__, __gfx11__ groupings.

    /// GFX9 family (CDNA): gfx90a, gfx942, gfx950.
    static constexpr TargetSet family_gfx9()
    {
        return only(GpuTarget::gfx90a, GpuTarget::gfx942, GpuTarget::gfx950);
    }

    /// GFX94 subfamily (CDNA 3+): gfx942, gfx950.
    static constexpr TargetSet family_gfx94() { return only(GpuTarget::gfx942, GpuTarget::gfx950); }

    /// GFX11 family (RDNA 3/3.5): gfx1100, gfx1101, gfx1102, gfx1150, gfx1151.
    /// Matches CK Tile's __gfx11__ preprocessor grouping.
    static constexpr TargetSet family_gfx11() { return rdna(); }

    /// GFX115 subfamily (RDNA 3.5): gfx1150, gfx1151.
    /// Matches CK Tile's __gfx115__ preprocessor grouping.
    static constexpr TargetSet family_gfx115()
    {
        return only(GpuTarget::gfx1150, GpuTarget::gfx1151);
    }

    // ---- Named constructors: specific targets -----------------------------

    /// Set containing exactly one target.
    static constexpr TargetSet only(GpuTarget t) { return from_bits(uint64_t{1} << bitIndex(t)); }

    /// Set containing exactly two targets.
    static constexpr TargetSet only(GpuTarget a, GpuTarget b)
    {
        return only(a).union_with(only(b));
    }

    /// Set containing exactly three targets.
    static constexpr TargetSet only(GpuTarget a, GpuTarget b, GpuTarget c)
    {
        return only(a).union_with(only(b)).union_with(only(c));
    }

    // ---- Set operations ---------------------------------------------------

    /// Remove one target from the set.
    constexpr TargetSet excluding(GpuTarget t) const
    {
        return from_bits(bits & ~(uint64_t{1} << bitIndex(t)));
    }

    /// Union: targets in either set.
    constexpr TargetSet union_with(TargetSet other) const { return from_bits(bits | other.bits); }

    /// Intersection: targets in both sets.
    constexpr TargetSet intersect_with(TargetSet other) const
    {
        return from_bits(bits & other.bits);
    }

    /// Difference: targets in this set but not in other.
    constexpr TargetSet minus(TargetSet other) const { return from_bits(bits & ~other.bits); }

    // ---- Operators (delegate to named methods) ----------------------------

    friend constexpr TargetSet operator|(TargetSet a, TargetSet b) { return a.union_with(b); }
    friend constexpr TargetSet operator&(TargetSet a, TargetSet b) { return a.intersect_with(b); }
    friend constexpr TargetSet operator-(TargetSet a, TargetSet b) { return a.minus(b); }
    friend constexpr bool operator==(TargetSet a, TargetSet b) { return a.bits == b.bits; }

    // ---- Queries ----------------------------------------------------------

    /// True if target is in the set.
    constexpr bool contains(GpuTarget t) const
    {
        return (bits & (uint64_t{1} << bitIndex(t))) != 0;
    }

    /// True if the set is empty.
    constexpr bool is_empty() const { return bits == 0; }

    /// True if the set contains exactly one target.
    constexpr bool is_single_target() const { return bits != 0 && (bits & (bits - 1)) == 0; }

    /// Number of targets in the set.
    constexpr int count() const { return std::popcount(bits); }

    /// Wavefront size, if uniform across all targets in the set.
    /// Compile error if targets disagree or set is empty.
    constexpr int wavefront_size() const
    {
        if(is_empty())
        {
            // wavefront_size() called on empty TargetSet
            CK_COMMON_TRAP();
        }

        int wf = -1;
        for(int i = 0; i < NUM_TARGETS; ++i)
        {
            if((bits & (uint64_t{1} << i)) == 0)
                continue;
            int target_wf = wavefrontSize(targetAt(i));
            if(wf == -1)
                wf = target_wf;
            else if(wf != target_wf)
            {
                // wavefront_size() requires all targets in the set to have the same
                // wavefront size -- this set mixes wave64 (CDNA) and wave32 (RDNA)
                // targets. Split with intersect_with(cdna()) or intersect_with(rdna()).
                CK_COMMON_TRAP();
            }
        }
        return wf;
    }

    /// True if all targets in the set are CDNA.
    constexpr bool is_all_cdna() const { return !is_empty() && minus(cdna()).is_empty(); }

    /// True if all targets in the set are RDNA.
    constexpr bool is_all_rdna() const { return !is_empty() && minus(rdna()).is_empty(); }

    /// Get the single target (compile error if set has != 1 target).
    constexpr GpuTarget single_target() const
    {
        if(!is_single_target())
        {
            // single_target() requires exactly one target in the set
            CK_COMMON_TRAP();
        }
        return targetAt(std::countr_zero(bits));
    }

    // ---- Iteration --------------------------------------------------------

    /// Call fn(GpuTarget) for each target in the set.
    template <typename Fn>
    constexpr void for_each(Fn fn) const
    {
        for(int i = 0; i < NUM_TARGETS; ++i)
            if(bits & (uint64_t{1} << i))
                fn(targetAt(i));
    }

    private:
    explicit constexpr TargetSet(uint64_t b) : bits(b) {}
    static constexpr TargetSet from_bits(uint64_t b) { return TargetSet{b}; }
};

// ============================================================================
// Wave tile validation -- single source of truth
// ============================================================================
// Based on CK Tile's WarpGemmDispatcher specializations.
// See: ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp (MFMA builtins)
//      ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp (WMMA builtins)
//      ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp (dispatch table)

/// Check if (a_dtype, m, n, k) maps to a valid wave instruction shape
/// on a specific target.
consteval bool isValidWaveTile(DataType a_dtype, int m, int n, int k, GpuTarget target)
{
    // RDNA targets: WMMA -- fixed 16x16x16 tile shape
    if(isRDNA(target))
    {
        if(m != 16 || n != 16 || k != 16)
            return false;
        // RDNA (gfx11xx) WMMA: fp16, bf16, int8 -- all targets share 16x16x16 tile
        return a_dtype == DataType::FP16 || a_dtype == DataType::BF16 || a_dtype == DataType::I8;
    }

    // CDNA MFMA tiles -- common across gfx90a, gfx942, gfx950
    if(a_dtype == DataType::FP32)
    {
        if(m == 16 && n == 16 && (k == 4 || k == 8 || k == 16))
            return true;
        if(m == 32 && n == 32 && (k == 4 || k == 8))
            return true;
    }
    if(a_dtype == DataType::FP16)
    {
        if(m == 16 && n == 16 && (k == 16 || k == 32))
            return true;
        if(m == 32 && n == 32 && (k == 8 || k == 16))
            return true;
    }
    if(a_dtype == DataType::BF16)
    {
        if(m == 16 && n == 16 && (k == 16 || k == 32))
            return true;
        if(m == 32 && n == 32 && (k == 8 || k == 16))
            return true;
    }

    // INT8 MFMA -- int8x int8->int32 accumulation
    if(a_dtype == DataType::I8)
    {
        if(m == 32 && n == 32 && k == 16)
            return true;
        if(m == 16 && n == 16 && k == 32)
            return true;
    }

    // FP8/BF8 MFMA -- architecture-dependent
    if(a_dtype == DataType::FP8_FNUZ || a_dtype == DataType::BF8_FNUZ)
    {
        // gfx90a: no FP8 MFMA support
        if(target == GpuTarget::gfx90a)
            return false;

        // gfx942+: base FP8 MFMA tiles
        if(m == 32 && n == 32 && k == 16)
            return true;
        if(m == 16 && n == 16 && k == 32)
            return true;

        // gfx942+: IterateK compositions of base FP8 MFMA
        if(m == 32 && n == 32 && (k == 32 || k == 64))
            return true;
        if(m == 16 && n == 16 && k == 64)
            return true;
    }

    // FP8_OCP/BF8_OCP -- not yet supported
    if(a_dtype == DataType::FP8_OCP || a_dtype == DataType::BF8_OCP)
        throw "FP8_OCP/BF8_OCP not yet supported in GEMM -- use FP8_FNUZ/BF8_FNUZ";

    return false;
}

/// Check if a tile is valid for ALL targets in a set (intersection semantics).
consteval bool isValidWaveTile(DataType a_dtype, int m, int n, int k, TargetSet targets)
{
    if(targets.is_empty())
        throw "isValidWaveTile called with empty TargetSet";

    for(int i = 0; i < TargetSet::NUM_TARGETS; ++i)
    {
        if(!(targets.bits & (uint64_t{1} << i)))
            continue;
        if(!isValidWaveTile(a_dtype, m, n, k, TargetSet::targetAt(i)))
            return false;
    }
    return true;
}

} // namespace ck_common
