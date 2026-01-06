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
#include "solution_selection.hpp"

#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Parameters/Solution/StreamK.hpp>


/**
 * @brief Solution Parameters
 *
 * Everything needed to generate a kernel
 *
 */
struct SolutionParameters
{
    // Datatype of inputs and outputs
    KernelType kernelType;

    // Workgroup Tile size
    WorkGroupTileSize workgroupTile;

    // Machine Instruction
    MachineInstructionSize machineInstruction;

    // Number of wave tiles to execute per workgroup
    uint wavefrontSize  = 64;
    int  workgroupSizeX = 2 * wavefrontSize;
    int  workgroupSizeY = 2;

    // Other options
    rocRoller::Parameters::Solution::LoadPath loadPathA
        = rocRoller::Parameters::Solution::LoadPath::BufferToLDS;
    rocRoller::Parameters::Solution::LoadPath loadPathB
        = rocRoller::Parameters::Solution::LoadPath::BufferToLDS;

    bool storeLDSD = false;

    bool prefetch          = true;
    int  prefetchInFlight  = 2;
    int  prefetchLDSFactor = 1;
    bool prefetchMixMemOps = true;
    bool betaInFma         = true;

    // Unroll Options
    unsigned int unrollX = 0;
    unsigned int unrollY = 0;

    std::string scheduler;

    bool streamK        = false;
    bool streamKTwoTile = false;

    // Scale options
    bool loadLDSScaleA = false;
    bool loadLDSScaleB = false;
    bool swizzleScale  = true;
    bool prefetchScale = true;

    // Workgroup Mapping
    int workgroupMappingDim = 0;
    bool workgroupRemapXCC = true;

    std::string toString() const;
};

/**
 * @brief Generate all of the solution parameters needed to create a kernel.
 *
 * This should only take into account the KernelType and SolutionIndexParameters
 * when deciding on the rest of the parameters to use for the kernel.
 *
 * @param kernelType
 * @param solutionIndexParameters
 * @return std::shared_ptr<SolutionParameters>
 */
std::shared_ptr<SolutionParameters>
    genSolutionParameters(const KernelType&              kernelType,
                          const SolutionIndexParameters& solutionIndexParameters);
