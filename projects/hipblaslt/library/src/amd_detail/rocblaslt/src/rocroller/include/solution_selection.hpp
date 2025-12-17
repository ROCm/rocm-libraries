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

#pragma once

#include "kernel_type.hpp"
#include "rocblaslt.h"

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

    auto operator<=>(const WorkGroupTileSize& other) const = default;
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
 * @brief SwizzleTileSize
 *
 * The swizzle tile size used for scale tensor swizzling.
 * For shuffle tile {tileMN, tileK, subTileK}:
 * - m, n = tileMN (from shuffleTile[0])
 * - k, l = tileK = 256 / tileMN (from shuffleTile[1])
 *
 * The relationship: tileMN * tileK = 256 (minimal swizzle tile elements)
 */
struct SwizzleTileSize
{
    int m = 64;  // For matrix A scale (M direction)
    int n = 64;  // For matrix A/B scale (common)
    int k = 4;   // For matrix A scale (K direction)
    int l = 4;   // For matrix B scale (L direction, same as K)
};

/**
 * @brief SolutionIndex Parameters
 *
 * All of the parameters that are used to generated a unique solution index.
 * There can be multiple kernels of the same KernelType that have different
 * SolutionIndexParameters.
 *
 */
struct SolutionIndexParameters
{
    WorkGroupTileSize workgroupTile;
    bool              workgroupMapping;
    bool              streamK;

    auto operator<=>(const SolutionIndexParameters& other) const = default;
};

int parametersToIndex(const SolutionIndexParameters& params);
SolutionIndexParameters indexToParameters(int index);

size_t maxNumberSolutions();

/**
 * @brief Pick machine instruction based on data types, workgroup tile, and shuffle tile
 *
 * When pre-swizzled scale data is used (shuffle tile is non-empty), always use
 * 16x16x128 MI instruction for compatibility.
 *
 * @param typeA Data type of matrix A
 * @param typeB Data type of matrix B
 * @param wgt Workgroup tile size
 * @param shuffleTileMN The tileMN value from shuffle tile (0 if no shuffle tile)
 * @return MachineInstructionSize The selected machine instruction dimensions
 */
inline MachineInstructionSize pickMI(rocRoller::DataType typeA,
                                     rocRoller::DataType typeB,
                                     WorkGroupTileSize wgt,
                                     size_t shuffleTileMN = 0) {
    if (typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16) {
        return {32, 32, 8, 1};
    } else if (typeA == rocRoller::DataType::Float) {
        return {32, 32, 2, 1};
    } else {
        assert((shuffleTileMN == 0 || shuffleTileMN == 64 || shuffleTileMN == 32) &&
               "shuffleTileMN must be 0, 64, or 32");
        // For pre-swizzled scale data, always use 16x16x128 MI
        if (shuffleTileMN != 0) {
            return {16, 16, 128, 1};
        }

        // Default selection logic when no shuffle tile constraint
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

constexpr int preferredUnrolling(rocRoller::DataType typeA, rocRoller::DataType typeB, WorkGroupTileSize wgt) {
    // Other datatypes run out of registers when prefetchInFlight is too
    // large.
    // There is an error with smaller tile sizes and larger prefetchInFlight.
    if (typeA == rocRoller::DataType::FP4 && typeB == rocRoller::DataType::FP4 && wgt.m > 32 && wgt.n > 32)
        return 4;
    else
        return 2;
}

/**
 * @brief Choose the SolutionIndexParameters to use for a given problem
 *
 * Examine the KernelType and problem size to determine the kernel to use
 * to compute the problem.
 *
 * Return a list of SolutionIndexParameters, in sorted order, based on how many kernels are requested.
 *
 * @param kernelType
 * @param prob
 * @return std::vector<SolutionIndexParameters>
 */
std::vector<SolutionIndexParameters> chooseSolutionIndexParameters(
    const KernelType& kernelType, const RocblasltContractionProblem& prob, int requestedAlgoCount);
