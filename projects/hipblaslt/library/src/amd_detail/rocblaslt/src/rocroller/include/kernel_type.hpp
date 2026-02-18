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

template <>
struct std::hash<ScaleType>
{
    size_t operator()(const ScaleType& s) const noexcept
    {
        size_t modeHash         = std::hash<rocRoller::Operations::ScaleMode>{}(s.mode);
        size_t blockRowSizeHash = std::hash<size_t>{}(s.blockRowSize);
        size_t blockColSizeHash = std::hash<size_t>{}(s.blockColSize);
        size_t typeHash         = std::hash<rocRoller::DataType>{}(s.type);

        // Hash the preSwizzleTile vector by combining hashes of its elements
        // Uses boost::hash_combine technique:
        // - 0x9e3779b9: Golden ratio constant (2^32 / phi) for good bit distribution
        // - Bit shifts (<< 6, >> 2): Ensures element order matters in final hash
        // - XOR (^=): Combines new hash with accumulated value
        size_t preSwizzleTileHash = 0;
        for(const auto& elem : s.preSwizzleTile)
        {
            preSwizzleTileHash ^= std::hash<size_t>{}(elem) + 0x9e3779b9 + (preSwizzleTileHash << 6)
                                  + (preSwizzleTileHash >> 2);
        }

        size_t preTileHash = 0;
        for(const auto& elem : s.preTile)
        {
            preTileHash ^= std::hash<size_t>{}(elem) + 0x9e3779b9 + (preTileHash << 6)
                           + (preTileHash >> 2);
        }

        return modeHash ^ (blockRowSizeHash << 1) ^ (blockColSizeHash << 2) ^ (typeHash << 3)
               ^ (preSwizzleTileHash << 4) ^ (preTileHash << 5);
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

    auto operator<=>(const KernelType& other) const = default;
};

template <>
struct std::hash<KernelType>
{
    size_t operator()(const KernelType& k) const noexcept
    {
        size_t typeAHash      = std::hash<rocRoller::DataType>{}(k.typeA);
        size_t typeBHash      = std::hash<rocRoller::DataType>{}(k.typeB);
        size_t typeCHash      = std::hash<rocRoller::DataType>{}(k.typeC);
        size_t typeDHash      = std::hash<rocRoller::DataType>{}(k.typeD);
        size_t typeAccHash    = std::hash<rocRoller::DataType>{}(k.typeAcc);
        size_t scaleTypeAHash = std::hash<ScaleType>{}(k.scaleTypeA);
        size_t scaleTypeBHash = std::hash<ScaleType>{}(k.scaleTypeB);
        size_t transAHash     = std::hash<bool>{}(k.transA);
        size_t transBHash     = std::hash<bool>{}(k.transB);

        return typeAHash ^ (typeBHash << 1) ^ (typeCHash << 2) ^ (typeDHash << 3)
               ^ (typeAccHash << 4) ^ (scaleTypeAHash << 5) ^ (scaleTypeBHash << 6)
               ^ (transAHash << 7) ^ (transBHash << 8);
    }
};
