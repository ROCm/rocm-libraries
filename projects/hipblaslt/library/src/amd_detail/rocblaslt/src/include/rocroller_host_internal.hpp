/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*********************************************************
 * Use this for any header contents that you don't want exposed. *
 *********************************************************/

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <Tensile/analytical/Utils.hpp>

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

    hipblasOperation_t transA;
    hipblasOperation_t transB;

    rocRoller::Operations::ScaleMode scaleAMode;
    rocRoller::Operations::ScaleMode scaleBMode;

    size_t scaleABlockRowSize = 32u;
    size_t scaleABlockColSize = 1u;
    size_t scaleBBlockRowSize = 1u;
    size_t scaleBBlockColSize = 32u;

    auto operator<=>(const KernelType& other) const = default;
};

/**
 * @brief WorkGroupTileSize
 *
 * The size of a tile that will be executed by a work group.
 *
 */
struct WorkGroupTileSize
{
    int m;
    int n;
    int k;
};

/**
 * @brief MachineInstructionSize
 *
 * The machine instruction that will be used for matrix multiplication operations
 *
 */
struct MachineInstructionSize
{
    int m = -1;
    int n = -1;
    int k = -1;
    int b = -1;
};

/**
 **************************************************************************************************
 * This section defines the lists of macro-tile/matrix-instruction combinations so that they are
 * compile-time known.
 */

constexpr std::array<WorkGroupTileSize, 23> possibleTileSizes = {{
    {256, 256, 128}, {256, 128, 128}, {128, 256, 128}, {256, 64, 128}, {64, 256, 128},
    {128, 128, 128}, {256, 32, 128},  {32, 256, 128},  {128, 64, 128}, {64, 128, 128},
    {256, 16, 128},  {16, 256, 128},  {128, 32, 128},  {32, 128, 128}, {64, 64, 128},
    {64, 32, 128},   {32, 64, 128},   {64, 16, 128},   {16, 64, 128},  {32, 32, 64},
    {32, 16, 128},   {16, 32, 128},   {16, 16, 128}
}};

constexpr MachineInstructionSize pickMI(rocRoller::DataType typeA, rocRoller::DataType typeB, WorkGroupTileSize wgt) {
    if (typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16) {
        return {32, 32, 8, 1};
    } else if (typeA == rocRoller::DataType::Float) {
        return {32, 32, 2, 1};
    } else {
        if ((typeA == rocRoller::DataType::FP6 || typeA == rocRoller::DataType::BF6 ||
             typeB == rocRoller::DataType::FP6 || typeB == rocRoller::DataType::BF6) &&
            ((wgt.m == 256 && wgt.n == 64) || (wgt.m == 64 && wgt.n == 256))) {
            return {32, 32, 64, 1};
        } else if (wgt.k % 128 == 0) {
            return {16, 16, 128, 1};
        } else {
            return {32, 32, 64, 1};
        }
    }
}

template <rocRoller::DataType typeA, rocRoller::DataType typeB>
constexpr auto generateTileList() {
    std::array<TensileLite::analytical::TileTuple, possibleTileSizes.size()> tileList{};

    for (size_t i = 0; i < possibleTileSizes.size(); ++i) {
        const auto& wgt = possibleTileSizes[i];
        auto MI = pickMI(typeA, typeB, wgt);

        int wgtk = wgt.k;
        if (typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16 || typeA == rocRoller::DataType::Float) {
            wgtk = 32;
        }

        tileList[i] = std::make_tuple(
            wgt.m, wgt.n, wgtk,
            MI.m, MI.n, MI.k,
            1 // occupancy
        );
    }

    return tileList;
}

using TileListGeneratorFn = std::vector<TensileLite::analytical::TileTuple>(*)();

template <rocRoller::DataType A, rocRoller::DataType B>
std::vector<TensileLite::analytical::TileTuple> generateTileListWrapper() {
    constexpr auto arr = generateTileList<A, B>();
    return {arr.begin(), arr.end()};
}

#define INSTANTIATE_TILE_LIST(A, B) \
    { {rocRoller::DataType::A, rocRoller::DataType::B}, &generateTileListWrapper<rocRoller::DataType::A, rocRoller::DataType::B> }

#define INSTANTIATE_TILE_LIST_FOR(A) \
    INSTANTIATE_TILE_LIST(A, Half), \
    INSTANTIATE_TILE_LIST(A, Float), \
    INSTANTIATE_TILE_LIST(A, BFloat16), \
    INSTANTIATE_TILE_LIST(A, FP8), \
    INSTANTIATE_TILE_LIST(A, BF8), \
    INSTANTIATE_TILE_LIST(A, FP4), \
    INSTANTIATE_TILE_LIST(A, BF6), \
    INSTANTIATE_TILE_LIST(A, FP6)

const std::map<std::pair<rocRoller::DataType, rocRoller::DataType>, TileListGeneratorFn> tileListGenerators = {
    INSTANTIATE_TILE_LIST_FOR(Half),
    INSTANTIATE_TILE_LIST_FOR(Float),
    INSTANTIATE_TILE_LIST_FOR(BFloat16),
    INSTANTIATE_TILE_LIST_FOR(FP8),
    INSTANTIATE_TILE_LIST_FOR(BF8),
    INSTANTIATE_TILE_LIST_FOR(FP4),
    INSTANTIATE_TILE_LIST_FOR(BF6),
    INSTANTIATE_TILE_LIST_FOR(FP6)
};

std::vector<TensileLite::analytical::TileTuple> getTileListForKernelType(KernelType kernelType)
{
    auto key = std::make_pair(kernelType.typeA, kernelType.typeB);
    auto it = tileListGenerators.find(key);
    if (it != tileListGenerators.end())
        return it->second();
    throw std::runtime_error("Unsupported DataType combination");
}

/**
 *
 **************************************************************************************************
 */
