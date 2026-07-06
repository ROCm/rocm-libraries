// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command.hpp>

/**
 * @brief ScaleType
 *
 * Scale parameters for a matrix (A or B).
 * - preSwizzleTile: Pre-swizzle tile configuration {tileMN, tileK, subTileK}
 *   Similar to scaleShuffleTileA/B in rocRoller's TypeParameters.
 * - preTile: Pre-tile configuration {tileM, tileK} for A or {tileK, tileN} for B
 */
struct ScaleType
{
    rocRoller::Operations::ScaleMode mode;
    size_t                           blockRowSize = 32u;
    size_t                           blockColSize = 1u;
    rocRoller::DataType              type         = rocRoller::DataType::E8M0;
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
        detail::hash_combine(seed, std::hash<rocRoller::Operations::ScaleMode>{}(s.mode));
        detail::hash_combine(seed, std::hash<size_t>{}(s.blockRowSize));
        detail::hash_combine(seed, std::hash<size_t>{}(s.blockColSize));
        detail::hash_combine(seed, std::hash<rocRoller::DataType>{}(s.type));

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
 */
struct KernelType
{
    rocRoller::DataType typeA;
    rocRoller::DataType typeB;
    rocRoller::DataType typeC;
    rocRoller::DataType typeD;
    rocRoller::DataType typeAcc = rocRoller::DataType::Float;

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
        detail::hash_combine(seed, std::hash<rocRoller::DataType>{}(k.typeA));
        detail::hash_combine(seed, std::hash<rocRoller::DataType>{}(k.typeB));
        detail::hash_combine(seed, std::hash<rocRoller::DataType>{}(k.typeC));
        detail::hash_combine(seed, std::hash<rocRoller::DataType>{}(k.typeD));
        detail::hash_combine(seed, std::hash<rocRoller::DataType>{}(k.typeAcc));
        detail::hash_combine(seed, std::hash<ScaleType>{}(k.scaleTypeA));
        detail::hash_combine(seed, std::hash<ScaleType>{}(k.scaleTypeB));
        detail::hash_combine(seed, std::hash<bool>{}(k.transA));
        detail::hash_combine(seed, std::hash<bool>{}(k.transB));
        detail::hash_combine(seed, std::hash<bool>{}(k.swizzleA));
        detail::hash_combine(seed, std::hash<bool>{}(k.swizzleB));
        return seed;
    }
};
