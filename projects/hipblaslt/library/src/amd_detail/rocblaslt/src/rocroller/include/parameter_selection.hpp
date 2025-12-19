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

#include <vector>


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
    rocRoller::Parameters::Solution::LoadPath loadPathAScale
        = rocRoller::Parameters::Solution::LoadPath::BufferToVGPR;
    rocRoller::Parameters::Solution::LoadPath loadPathBScale
        = rocRoller::Parameters::Solution::LoadPath::BufferToVGPR;
    bool swizzleScale  = true;
    bool prefetchScale = true;

    // Swizzle tile size for scale tensor swizzling
    // Uses {m, k, n, l} order to match rocRoller's MKNLTuple
    SwizzleTileSize swizzleTileSize;

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

/**
 * @brief Select the swizzle tile size in M/N dimension
 *
 * For pre-swizzled data, returns preSwizzleTileSize[0] (tileMN).
 * Otherwise, selects from {32, 64} based on divisibility constraints.
 *
 * @param workgroupTile Workgroup tile size {m, n, k}
 * @param mi Machine instruction size {m, n, k}
 * @param workgroupSizeX Workgroup size in X dimension
 * @param workgroupSizeY Workgroup size in Y dimension
 * @param preSwizzleTileSize Pre-swizzle tile configuration {tileMN, tileK, subTileK}, or empty if no pre-swizzle
 * @return Selected swizzle tile MN size, or -1 if no valid size found
 */
inline int selectSwizzleTileMN(
    const WorkGroupTileSize&      workgroupTile,
    const MachineInstructionSize& mi,
    int                           workgroupSizeX,
    int                           workgroupSizeY,
    const std::vector<size_t>&    preSwizzleTileSize)
{
    // For pre-swizzled data, return tileMN from preSwizzleTileSize
    if (preSwizzleTileSize.size() == 3)
    {
        return static_cast<int>(preSwizzleTileSize[0]);
    }

    // Validate inputs
    if (mi.m <= 0 || mi.n <= 0 || workgroupSizeX <= 0 || workgroupSizeY <= 0)
    {
        return 0;
    }

    // Number of waves: {workgroupSizeX / 64, workgroupSizeY}
    constexpr int wavefrontSize = 64;
    int numWavesX = workgroupSizeX / wavefrontSize;
    int numWavesY = workgroupSizeY;

    if (numWavesX <= 0 || numWavesY <= 0)
    {
        return 0;
    }

    // Compute number of tiles per wave in M and N dimensions
    int numMTilesPerWave = workgroupTile.m / mi.m / numWavesX;
    int numNTilesPerWave = workgroupTile.n / mi.n / numWavesY;

    // Possible swizzle tile MN values
    std::vector<int> possibleSwizzleTileMN = {32, 64};
    std::vector<int> validSwizzleTileMN;

    for (int swizzleTileMN : possibleSwizzleTileMN)
    {
        // Compute minimum number of tiles per wave
        int minMTilesPerWave = swizzleTileMN / mi.m;
        int minNTilesPerWave = swizzleTileMN / mi.n;

        std::cout << "numMTilesPerWave: " << numMTilesPerWave << std::endl;
        std::cout << "numNTilesPerWave: " << numNTilesPerWave << std::endl;
        std::cout << "minMTilesPerWave: " << minMTilesPerWave << std::endl;
        std::cout << "minNTilesPerWave: " << minNTilesPerWave << std::endl;

        // Check divisibility
        bool validM = (minMTilesPerWave > 0) && (numMTilesPerWave % minMTilesPerWave == 0);
        bool validN = (minNTilesPerWave > 0) && (numNTilesPerWave % minNTilesPerWave == 0);

        if (validM && validN)
        {
            std::cout << "validSwizzleTileMN: " << swizzleTileMN << std::endl;
            validSwizzleTileMN.push_back(swizzleTileMN);
        }
    }

    if (validSwizzleTileMN.empty())
    {
        return 0;  // Error: no valid swizzle tile size
    }

    // if (validSwizzleTileMN.size() == 1)
    // {
    //     return validSwizzleTileMN[0];
    // }

    // If both 32 and 64 are valid, prefer 64
    return validSwizzleTileMN.back();
}

/**
 * @brief Select the swizzle tile size in K dimension
 *
 * For pre-swizzled data, returns preSwizzleTileSize[1] (tileK).
 * Otherwise, selects from {4, 8, 16} based on divisibility constraints.
 *
 * @param workgroupTile Workgroup tile size {m, n, k}
 * @param mi Machine instruction size {m, n, k}
 * @param unroll Unroll factor (prefetchInFlight)
 * @param scaleBlockSize Scale block size (e.g., 32)
 * @param swizzleTileMN The selected swizzle tile MN value
 * @param preSwizzleTileSize Pre-swizzle tile configuration {tileMN, tileK, subTileK}, or empty if no pre-swizzle
 * @return Selected swizzle tile K size, or -1 if no valid size found
 */
inline int selectSwizzleTileK(
    const WorkGroupTileSize&      workgroupTile,
    const MachineInstructionSize& mi,
    int                           unroll,
    int                           scaleBlockSize,
    int                           swizzleTileMN,
    const std::vector<size_t>&    preSwizzleTileSize)
{
    // For pre-swizzled data, return tileK from preSwizzleTileSize
    if (preSwizzleTileSize.size() == 3)
    {
        if (swizzleTileMN == 32)
         return 8;
        else
         return 0;
    }

    // Validate inputs
    if (mi.k <= 0 || scaleBlockSize <= 0 || unroll <= 0)
    {
        return 0;
    }

    int miKScale = mi.k / scaleBlockSize;
    if (miKScale <= 0)
    {
        return 0;
    }

    // Compute number of K tiles per wave
    // numberOfKTilesPerWave = workgroupTile.k * unroll / scaleBlockSize / (mi.k / scaleBlockSize)
    int numKTilesPerWave = workgroupTile.k * unroll / scaleBlockSize / miKScale;

    // Possible swizzle tile K values
    std::vector<int> possibleSwizzleTileK = {4, 8, 16};
    std::vector<int> validSwizzleTileK;

    for (int swizzleTileK : possibleSwizzleTileK)
    {
        // Compute minimum K tiles per wave
        int minKTilesPerWave = swizzleTileK / miKScale;

        std::cout << "numKTilesPerWave: " << numKTilesPerWave << std::endl;
        std::cout << "minKTilesPerWave: " << minKTilesPerWave << std::endl;

        // Check divisibility
        if (minKTilesPerWave > 0 && numKTilesPerWave % minKTilesPerWave == 0)
        {
            std::cout << "validSwizzleTileK: " << swizzleTileK << std::endl;
            validSwizzleTileK.push_back(swizzleTileK);
        }
    }

    if (validSwizzleTileK.empty())
    {
        return 0;  // Error: no valid swizzle tile K size
    }

    return validSwizzleTileK.back();
}

/**
 * @brief Select the complete swizzle tile size {m, k, n, l}
 *
 * Combines selectSwizzleTileMN and selectSwizzleTileK to determine the full swizzle tile.
 * Currently assumes m == n and k == l (symmetric swizzle tile).
 *
 * @param workgroupTile Workgroup tile size {m, n, k}
 * @param mi Machine instruction size {m, n, k}
 * @param workgroupSizeX Workgroup size in X dimension
 * @param workgroupSizeY Workgroup size in Y dimension
 * @param unroll Unroll factor (prefetchInFlight)
 * @param scaleBlockSize Scale block size (e.g., 32)
 * @param preSwizzleTileSize Pre-swizzle tile configuration {tileMN, tileK, subTileK}, or empty if no pre-swizzle
 * @return SwizzleTileSize with valid values, or all -1 if no valid size found
 */
inline SwizzleTileSize selectSwizzleTileSize(
    const WorkGroupTileSize&      workgroupTile,
    const MachineInstructionSize& mi,
    int                           workgroupSizeX,
    int                           workgroupSizeY,
    int                           unroll,
    int                           scaleBlockSize,
    const std::vector<size_t>&    preSwizzleTileSize)
{
    std::cout << "selectSwizzleTileSize: workgroupTile: " << workgroupTile.m << "x" << workgroupTile.n << "x" << workgroupTile.k << std::endl;
    std::cout << "selectSwizzleTileSize: mi: " << mi.m << "x" << mi.n << "x" << mi.k << std::endl;
    std::cout << "selectSwizzleTileSize: workgroupSizeX: " << workgroupSizeX << std::endl;
    std::cout << "selectSwizzleTileSize: workgroupSizeY: " << workgroupSizeY << std::endl;
    std::cout << "selectSwizzleTileSize: unroll: " << unroll << std::endl;
    std::cout << "selectSwizzleTileSize: scaleBlockSize: " << scaleBlockSize << std::endl;
    std::cout << "selectSwizzleTileSize: preSwizzleTileSize.size(): " << preSwizzleTileSize.size() << std::endl;

    SwizzleTileSize result = {0, 0, 0, 0};

    int swizzleTileMN = selectSwizzleTileMN(workgroupTile, mi, workgroupSizeX, workgroupSizeY, preSwizzleTileSize);
    if (swizzleTileMN < 0)
    {
        return result;
    }

    int swizzleTileK = selectSwizzleTileK(workgroupTile, mi, unroll, scaleBlockSize, swizzleTileMN, preSwizzleTileSize);
    if (swizzleTileK < 0)
    {
        return result;
    }

    // Set symmetric swizzle tile (m == n, k == l)
    result.m = swizzleTileMN;
    result.k = swizzleTileK;
    result.n = swizzleTileMN;
    result.l = swizzleTileK;

    return result;
}
