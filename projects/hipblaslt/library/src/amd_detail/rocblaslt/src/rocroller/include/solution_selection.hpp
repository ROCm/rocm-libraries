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
 * The swizzle tile size used for swizzling scale data in SolutionParameters.
 * Uses {m, k, n, l} order to match rocRoller's MKNLTuple.
 *
 * Constraints:
 * - Currently only supports m == n && k == l because the kernel assumes the swizzle tile size is the same for matrix A and matrix B.
 * - The swizzle tile size has to be compatible with the pre-swizzled scale data when using pre-swizzled scale data.
 */
struct SwizzleTileSize
{
    int m; // For matrix A scale (M direction)
    int k; // For matrix A scale (K direction)
    int n; // For matrix B scale (N direction)
    int l; // For matrix B scale (L/K direction)
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

int                     parametersToIndex(const SolutionIndexParameters& params);
SolutionIndexParameters indexToParameters(int index);

size_t maxNumberSolutions();

/**
 * @brief Pick machine instruction based on data types, workgroup tile, and pre-swizzle configuration
 *
 * When pre-swizzled scale data is used (preSwizzleTileMN != 0), the MI instruction
 * is constrained to 16x16x128 for compatibility with the pre-swizzled data layout.
 *
 * @param typeA Data type of matrix A
 * @param typeB Data type of matrix B
 * @param wgt Workgroup tile size
 * @param preSwizzleTileMN The tileMN value from preSwizzleTile (0 if no pre-swizzle)
 * @return MachineInstructionSize The selected machine instruction dimensions
 */
inline MachineInstructionSize pickMI(rocRoller::DataType typeA,
                                     rocRoller::DataType typeB,
                                     WorkGroupTileSize   wgt,
                                     size_t              preSwizzleTileMN = 0)
{
    if(typeA == rocRoller::DataType::Half || typeA == rocRoller::DataType::BFloat16)
    {
        return {32, 32, 8, 1};
    }
    else if(typeA == rocRoller::DataType::Float)
    {
        return {32, 32, 2, 1};
    }
    else
    {
        // For pre-swizzled scale data, MI instruction must be 16x16x128
        // This ensures subTileK = MI.k / scaleBlockSize = 128 / 32 = 4
        if(preSwizzleTileMN != 0)
        {
            assert(preSwizzleTileMN == 32 && "preSwizzleTileMN must be 32 for pre-swizzled data");
            return {16, 16, 128, 1};
        }

        // Default selection logic when no pre-swizzle constraint
        if((typeA == rocRoller::DataType::FP6 || typeA == rocRoller::DataType::BF6
            || typeB == rocRoller::DataType::FP6 || typeB == rocRoller::DataType::BF6)
           && ((wgt.m == 256 && wgt.n == 64) || (wgt.m == 64 && wgt.n == 256)))
        {
            return {32, 32, 64, 1};
        }
        else if(wgt.k % 128 == 0)
        {
            return {16, 16, 128, 1};
        }
        else
        {
            return {32, 32, 64, 1};
        }
    }
}

constexpr int
    preferredUnrolling(rocRoller::DataType typeA, rocRoller::DataType typeB, WorkGroupTileSize wgt)
{
    // Other datatypes run out of registers when prefetchInFlight is too
    // large.
    // There is an error with smaller tile sizes and larger prefetchInFlight.
    // Unroll is 2 when wgt.k is greater or equal to 256, otherwise 4
    if(typeA == rocRoller::DataType::FP4 && typeB == rocRoller::DataType::FP4 && wgt.m > 32
       && wgt.n > 32 && wgt.k < 256)
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
