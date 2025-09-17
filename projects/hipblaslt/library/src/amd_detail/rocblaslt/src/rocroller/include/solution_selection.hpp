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
    int               prefetchInFlight;
    bool              workgroupMapping;
};

int parametersToIndex(const SolutionIndexParameters& params);
SolutionIndexParameters indexToParameters(int index);

size_t maxNumberSolutions();

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
