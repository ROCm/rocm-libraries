// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — GPU target properties and target set (consteval bitset).
//
// UPSTREAM CANDIDATE: This header prototypes functionality that should
// eventually live in ck_tile/core/arch/arch_properties.hpp as the single
// source of truth for GPU target metadata. It is intentionally free of
// device code, HIP runtime, and CK Tile dependencies so it can be used
// in both host-side consteval validation and device-side template selection.
//
// When upstreaming, the valid tile table should be generated from or
// verified against CK Tile's MMA selector specializations
// (core/arch/mma/mfma/mfma_gfx9.hpp, wmma/wmma_gfx11.hpp, etc.).
//
// No runtime code, no CK deps. Pure C++20 constexpr/consteval.

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/types.hpp>

#include <cstdint>

namespace rocm_ck {

// ============================================================================
// Per-target properties
// ============================================================================

/// ISA architecture family.
///
/// CDNA = data-center compute (Instinct GPUs), MFMA instructions, wave64.
/// RDNA = gaming/workstation (Radeon GPUs), WMMA instructions, wave32.
enum class ArchFamily
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
    case GpuTarget::gfx90a: return {64, ArchFamily::CDNA};
    case GpuTarget::gfx942: return {64, ArchFamily::CDNA};
    case GpuTarget::gfx950: return {64, ArchFamily::CDNA};
    case GpuTarget::gfx1100: return {32, ArchFamily::RDNA};
    case GpuTarget::gfx1101: return {32, ArchFamily::RDNA};
    case GpuTarget::gfx1102: return {32, ArchFamily::RDNA};
    case GpuTarget::gfx1150: return {32, ArchFamily::RDNA};
    case GpuTarget::gfx1151: return {32, ArchFamily::RDNA};
    case GpuTarget::gfx1200: return {32, ArchFamily::RDNA};
    case GpuTarget::gfx1201: return {32, ArchFamily::RDNA};
    default: throw "invalid GpuTarget in properties";
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
// TargetSet — consteval bitset over GpuTarget values
// ============================================================================

/// Compile-time set of GPU targets. Structural type (usable as NTTP).
///
/// Storage: each GpuTarget maps to a bit in a uint64_t.
///
/// Named constructors express CK Tile's 3-level hierarchy:
///   Architecture: TargetSet::cdna(), TargetSet::rdna(), TargetSet::all()
///   Family:       TargetSet::family_gfx9(), family_gfx94(), family_gfx11(), family_gfx115(),
///                 family_gfx12()
///   Specific:     TargetSet::only(GpuTarget::gfx942)
///
/// CK Tile mapping:
///   TargetSet::cdna()                          → enable_if_target_arch_cdna_t
///   TargetSet::rdna()                          → enable_if_target_arch_rdna_t
///   TargetSet::family_gfx9()                   → enable_if_target_family_gfx9_t
///   TargetSet::family_gfx11()                  → __gfx11__ preprocessor grouping
///   TargetSet::family_gfx115()                 → __gfx115__ preprocessor grouping
///   TargetSet::family_gfx12()                  → __gfx12__ preprocessor grouping
///   TargetSet::only(gfx942, gfx950)            → enable_if_target_id_t<T, GFX942, GFX950>
///   TargetSet::cdna().excluding(gfx90a)        → is_any_value_of(T::TARGET_ID, GFX942, GFX950)
struct TargetSet
{
    uint64_t bits = 0;

    // ---- Bit index mapping ------------------------------------------------
    // Decoupled from GpuTarget enum values. Adding a target:
    //   1. Add enum value to GpuTarget
    //   2. Add case to bitIndex() and targetAt()
    //   3. Add to architecture/family constructors
    //   4. Bump kNumTargets

    /// Number of real targets (update when adding targets).
    static constexpr int kNumTargets = 10;

    /// Map a real GpuTarget to its bit index.
    static constexpr int bitIndex(GpuTarget target)
    {
        switch(target)
        {
        case GpuTarget::gfx90a: return 0;
        case GpuTarget::gfx942: return 1;
        case GpuTarget::gfx950: return 2;
        case GpuTarget::gfx1151: return 3;
        case GpuTarget::gfx1100: return 4;
        case GpuTarget::gfx1101: return 5;
        case GpuTarget::gfx1102: return 6;
        case GpuTarget::gfx1150: return 7;
        case GpuTarget::gfx1200: return 8;
        case GpuTarget::gfx1201: return 9;
        default: throw "invalid GpuTarget in bitIndex";
        }
    }

    /// Reverse map: bit index to GpuTarget.
    static constexpr GpuTarget targetAt(int index)
    {
        switch(index)
        {
        case 0: return GpuTarget::gfx90a;
        case 1: return GpuTarget::gfx942;
        case 2: return GpuTarget::gfx950;
        case 3: return GpuTarget::gfx1151;
        case 4: return GpuTarget::gfx1100;
        case 5: return GpuTarget::gfx1101;
        case 6: return GpuTarget::gfx1102;
        case 7: return GpuTarget::gfx1150;
        case 8: return GpuTarget::gfx1200;
        case 9: return GpuTarget::gfx1201;
        default: throw "invalid bit index in targetAt";
        }
    }

    // ---- Constructors -----------------------------------------------------

    /// Default: empty set.
    constexpr TargetSet() = default;

    /// Implicit conversion from a single GpuTarget.
    constexpr TargetSet(GpuTarget target) : bits(uint64_t{1} << bitIndex(target)) {}

    // ---- Named constructors: architecture level ---------------------------

    /// All real GPU targets.
    static constexpr TargetSet all() { return fromBits((uint64_t{1} << kNumTargets) - 1); }

    /// All CDNA targets (MFMA, wave64): gfx90a, gfx942, gfx950.
    static constexpr TargetSet cdna()
    {
        return only(GpuTarget::gfx90a, GpuTarget::gfx942, GpuTarget::gfx950);
    }

    /// All RDNA targets (WMMA, wave32): gfx11, gfx115, gfx12.
    static constexpr TargetSet rdna()
    {
        return only(GpuTarget::gfx1100, GpuTarget::gfx1101, GpuTarget::gfx1102)
            .union_with(only(GpuTarget::gfx1150, GpuTarget::gfx1151))
            .union_with(only(GpuTarget::gfx1200, GpuTarget::gfx1201));
    }

    // ---- Named constructors: family level ---------------------------------
    // Matches CK Tile's __gfx9__, __gfx94__, __gfx11__, __gfx115__, __gfx12__ groupings.

    /// GFX9 family (CDNA): gfx90a, gfx942, gfx950.
    static constexpr TargetSet family_gfx9()
    {
        return only(GpuTarget::gfx90a, GpuTarget::gfx942, GpuTarget::gfx950);
    }

    /// GFX94 subfamily (CDNA 3+): gfx942, gfx950.
    static constexpr TargetSet family_gfx94() { return only(GpuTarget::gfx942, GpuTarget::gfx950); }

    /// GFX11 family (RDNA 3/3.5): gfx1100, gfx1101, gfx1102, gfx1150, gfx1151.
    /// Matches CK Tile's __gfx11__ preprocessor grouping.
    static constexpr TargetSet family_gfx11()
    {
        return only(GpuTarget::gfx1100, GpuTarget::gfx1101, GpuTarget::gfx1102)
            .union_with(only(GpuTarget::gfx1150, GpuTarget::gfx1151));
    }

    /// GFX115 subfamily (RDNA 3.5): gfx1150, gfx1151.
    /// Matches CK Tile's __gfx115__ preprocessor grouping.
    static constexpr TargetSet family_gfx115()
    {
        return only(GpuTarget::gfx1150, GpuTarget::gfx1151);
    }

    /// GFX12 family (RDNA 4): gfx1200, gfx1201.
    /// Matches CK Tile's __gfx12__ preprocessor grouping.
    static constexpr TargetSet family_gfx12()
    {
        return only(GpuTarget::gfx1200, GpuTarget::gfx1201);
    }

    // ---- Named constructors: specific targets -----------------------------

    /// Set containing exactly one target.
    static constexpr TargetSet only(GpuTarget t) { return fromBits(uint64_t{1} << bitIndex(t)); }

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
        return fromBits(bits & ~(uint64_t{1} << bitIndex(t)));
    }

    /// Union: targets in either set.
    constexpr TargetSet union_with(TargetSet other) const { return fromBits(bits | other.bits); }

    /// Intersection: targets in both sets.
    constexpr TargetSet intersect_with(TargetSet other) const
    {
        return fromBits(bits & other.bits);
    }

    /// Difference: targets in this set but not in other.
    constexpr TargetSet minus(TargetSet other) const { return fromBits(bits & ~other.bits); }

    // ---- Operators (delegate to named methods) ----------------------------

    friend constexpr TargetSet operator|(TargetSet a, TargetSet b) { return a.union_with(b); }
    friend constexpr TargetSet operator&(TargetSet a, TargetSet b) { return a.intersect_with(b); }
    friend constexpr TargetSet operator-(TargetSet a, TargetSet b) { return a.minus(b); }
    friend constexpr bool operator==(TargetSet a, TargetSet b) { return a.bits == b.bits; }
    friend constexpr bool operator!=(TargetSet a, TargetSet b) { return a.bits != b.bits; }

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
    constexpr int count() const
    {
        uint64_t b = bits;
        int n      = 0;
        while(b)
        {
            n++;
            b &= b - 1;
        }
        return n;
    }

    /// Wavefront size, if uniform across all targets in the set.
    /// Compile error if targets disagree or set is empty.
    constexpr int wavefront_size() const
    {
        if(is_empty())
            throw "wavefront_size() called on empty TargetSet";

        int wf = -1;
        for(int i = 0; i < kNumTargets; ++i)
        {
            if((bits & (uint64_t{1} << i)) == 0)
                continue;
            int target_wf = wavefrontSize(targetAt(i));
            if(wf == -1)
                wf = target_wf;
            else if(wf != target_wf)
                throw "wavefront_size() requires all targets in the set to have "
                      "the same wavefront size — this set mixes wave64 (CDNA) and "
                      "wave32 (RDNA) targets. Split with intersect_with(cdna()) or "
                      "intersect_with(rdna()).";
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
            throw "single_target() requires exactly one target in the set";
        for(int i = 0; i < kNumTargets; ++i)
            if(bits & (uint64_t{1} << i))
                return targetAt(i);
        throw "unreachable";
    }

    // ---- Iteration --------------------------------------------------------

    /// Call fn(GpuTarget) for each target in the set.
    template <typename Fn>
    constexpr void for_each(Fn fn) const
    {
        for(int i = 0; i < kNumTargets; ++i)
            if(bits & (uint64_t{1} << i))
                fn(targetAt(i));
    }

    private:
    explicit constexpr TargetSet(uint64_t b) : bits(b) {}
    static constexpr TargetSet fromBits(uint64_t b) { return TargetSet{b}; }
};

// ============================================================================
// Wave tile validation — single source of truth
// ============================================================================
// Based on CK Tile's WarpGemmDispatcher specializations.
// See: ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp (MFMA builtins)
//      ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp (WMMA builtins)
//      ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp (dispatch table)

/// Check if (a_dtype, m, n, k) maps to a valid wave instruction shape
/// on a specific target.
consteval bool isValidWaveTile(DataType a_dtype, int m, int n, int k, GpuTarget target)
{
    // RDNA targets: WMMA — fixed 16x16x16 tile shape
    if(isRDNA(target))
    {
        if(m != 16 || n != 16 || k != 16)
            return false;
        // RDNA (gfx11xx) WMMA: fp16, bf16, int8 — all targets share 16×16×16 tile
        return a_dtype == DataType::FP16 || a_dtype == DataType::BF16 || a_dtype == DataType::I8;
    }

    // CDNA MFMA tiles — common across gfx90a, gfx942, gfx950
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

    // INT8 MFMA — int8x int8→int32 accumulation
    if(a_dtype == DataType::I8)
    {
        if(m == 32 && n == 32 && k == 16)
            return true;
        if(m == 16 && n == 16 && k == 32)
            return true;
    }

    // FP8/BF8 MFMA — architecture-dependent
    if(a_dtype == DataType::FP8_FNUZ || a_dtype == DataType::BF8_FNUZ)
    {
        // gfx90a: no FP8 MFMA support
        if(target == GpuTarget::gfx90a)
            return false;

        // gfx942 baseline: 32x32x16, 16x16x32
        if(m == 32 && n == 32 && k == 16)
            return true;
        if(m == 16 && n == 16 && k == 32)
            return true;

        // gfx950: extended FP8 tiles — 32x32x{32,64}, 16x16x64
        if(target == GpuTarget::gfx950)
        {
            if(m == 32 && n == 32 && (k == 32 || k == 64))
                return true;
            if(m == 16 && n == 16 && k == 64)
                return true;
        }
    }

    // FP8_OCP/BF8_OCP — not yet supported
    if(a_dtype == DataType::FP8_OCP || a_dtype == DataType::BF8_OCP)
        throw "FP8_OCP/BF8_OCP not yet supported in GEMM — use FP8_FNUZ/BF8_FNUZ";

    return false;
}

/// Check if a tile is valid for ALL targets in a set (intersection semantics).
consteval bool isValidWaveTile(DataType a_dtype, int m, int n, int k, TargetSet targets)
{
    if(targets.is_empty())
        throw "isValidWaveTile called with empty TargetSet";

    for(int i = 0; i < TargetSet::kNumTargets; ++i)
    {
        if(!(targets.bits & (uint64_t{1} << i)))
            continue;
        if(!isValidWaveTile(a_dtype, m, n, k, TargetSet::targetAt(i)))
            return false;
    }
    return true;
}

} // namespace rocm_ck
