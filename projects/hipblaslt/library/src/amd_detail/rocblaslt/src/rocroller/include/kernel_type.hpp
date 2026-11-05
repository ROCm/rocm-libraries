// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// A kernel's operands are described in the types hipBLASLt already uses for
// them: hipDataType for an element format, and the scaling format carried by
// RocblasltContractionProblem. Nothing here names a kernel generator's types,
// which is what lets the tile ranker build with no generator present.
#include "rocblaslt-types.h"

#include <hipblaslt/hipblaslt.h>

#include <compare>
#include <cstddef>
#include <functional>
#include <vector>

/**
 * @brief ScaleType
 *
 * Scale parameters for a matrix (A or B).
 * - format: how the operand's block scales are supplied. ScalingFormat::None
 *   means the operand is not scaled; isBlockScaling(format) means the scales
 *   arrive in a tensor of their own.
 * - preSwizzleTile: Pre-swizzle tile configuration {tileMN, tileK, subTileK}
 *   Similar to scaleShuffleTileA/B in rocRoller's TypeParameters.
 * - preTile: Pre-tile configuration {tileM, tileK} for A or {tileK, tileN} for B
 */
struct ScaleType
{
    RocblasltContractionProblem::ScalingFormat format
        = RocblasltContractionProblem::ScalingFormat::None;
    size_t              blockRowSize = 32u;
    size_t              blockColSize = 1u;
    hipDataType         type         = HIP_R_8F_UE8M0;
    std::vector<size_t> preSwizzleTile; // {tileMN, tileK, subTileK} for pre-swizzled scale data
    std::vector<size_t> preTile;

    auto operator<=>(const ScaleType& other) const = default;
};

namespace detail
{
    // boost::hash_combine implementation for combining hash values
    // - 0x9e3779b9: Golden ratio constant (2^32 / phi) for good bit distribution
    // - Bit shifts (<< 6, >> 2): Ensures element order matters in final hash
    // - XOR (^=): Combines new hash with accumulated value
    inline void hash_combine(size_t& seed, size_t value)
    {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
} // namespace detail

template <>
struct std::hash<ScaleType>
{
    size_t operator()(const ScaleType& s) const noexcept
    {
        size_t seed = 0;
        detail::hash_combine(seed,
                             std::hash<RocblasltContractionProblem::ScalingFormat>{}(s.format));
        detail::hash_combine(seed, std::hash<size_t>{}(s.blockRowSize));
        detail::hash_combine(seed, std::hash<size_t>{}(s.blockColSize));
        detail::hash_combine(seed, std::hash<hipDataType>{}(s.type));

        for(const auto& elem : s.preSwizzleTile)
        {
            detail::hash_combine(seed, std::hash<size_t>{}(elem));
        }

        for(const auto& elem : s.preTile)
        {
            detail::hash_combine(seed, std::hash<size_t>{}(elem));
        }

        return seed;
    }
};

/**
 * @brief KernelType
 *
 * All of the values required for different types of kernels.
 * This should not include any optimization flags.
 *
 * An element format the ranker does not handle is held as
 * HIPBLASLT_DATATYPE_INVALID, which matches no kernel.
 */
struct KernelType
{
    hipDataType typeA;
    hipDataType typeB;
    hipDataType typeC;
    hipDataType typeD;
    hipDataType typeAcc = HIP_R_32F;

    bool transA;
    bool transB;

    ScaleType scaleTypeA;
    ScaleType scaleTypeB;

    bool swizzleA = false;
    bool swizzleB = false;

    auto operator<=>(const KernelType& other) const = default;
};

template <>
struct std::hash<KernelType>
{
    size_t operator()(const KernelType& k) const noexcept
    {
        size_t seed = 0;
        detail::hash_combine(seed, std::hash<hipDataType>{}(k.typeA));
        detail::hash_combine(seed, std::hash<hipDataType>{}(k.typeB));
        detail::hash_combine(seed, std::hash<hipDataType>{}(k.typeC));
        detail::hash_combine(seed, std::hash<hipDataType>{}(k.typeD));
        detail::hash_combine(seed, std::hash<hipDataType>{}(k.typeAcc));
        detail::hash_combine(seed, std::hash<ScaleType>{}(k.scaleTypeA));
        detail::hash_combine(seed, std::hash<ScaleType>{}(k.scaleTypeB));
        detail::hash_combine(seed, std::hash<bool>{}(k.transA));
        detail::hash_combine(seed, std::hash<bool>{}(k.transB));
        detail::hash_combine(seed, std::hash<bool>{}(k.swizzleA));
        detail::hash_combine(seed, std::hash<bool>{}(k.swizzleB));
        return seed;
    }
};
