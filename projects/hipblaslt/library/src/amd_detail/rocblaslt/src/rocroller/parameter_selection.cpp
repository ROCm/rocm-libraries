/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "parameter_selection.hpp"

const int SWIZZLE_BLOCK_SIZE           = 64;

std::string SolutionParameters::toString() const
{
    std::stringstream result;

    result << "WorkGroupTile:" << workgroupTile.m << "x" << workgroupTile.n << "x"
           << workgroupTile.k << std::endl;
    result << "MachineInstruction:" << machineInstruction.m << "x" << machineInstruction.n << "x"
           << machineInstruction.k << std::endl;
    result << "WorkgroupSize:" << workgroupSizeX << "x" << workgroupSizeY << std::endl;
    result << "LoadA: " << loadPathA << std::endl;
    result << "LoadB: " << loadPathB << std::endl;
    result << "LDS Usage";
    result << " D:" << (storeLDSD ? "On" : "Off") << std::endl;
    result << "Workgroup Mapping: Dim:" << workgroupMappingDim << " RemapXCC:" << workgroupRemapXCC << std::endl;
    result << "Prefetch:" << prefetch << " InFlight:" << prefetchInFlight
           << " LDSFactor:" << prefetchLDSFactor << " MixMemOps:" << prefetchMixMemOps << std::endl;
    result << "Block Scale Options:" << " Swizzle Scale:" << swizzleScale
           << " Prefetch Scale:" << prefetchScale << " loadLDS A:" << loadLDSScaleA
           << " loadLDS B:" << loadLDSScaleB << std::endl;

    return result.str();
}

std::pair<int, int> pickWorkgroupSize(std::shared_ptr<SolutionParameters> gemm)
{
    int x = 2;
    int y = 2;

    int requiredX = -1;
    int requiredY = -1;

    if(gemm->workgroupTile.m / gemm->machineInstruction.m == 1)
        requiredX = 1;
    if(gemm->workgroupTile.n / gemm->machineInstruction.n == 1)
        requiredY = 1;

    //Swizzle Scale only works with certain combinations of workgroup sizes
    if(gemm->swizzleScale && (gemm->workgroupTile.m / SWIZZLE_BLOCK_SIZE) % x != 0)
        requiredX = 1;
    if(gemm->swizzleScale && (gemm->workgroupTile.n / SWIZZLE_BLOCK_SIZE) % y != 0)
        requiredY = 1;

    if(requiredX != -1 && requiredY == -1)
    {
        x = requiredX;
        if (gemm->swizzleScale && (gemm->workgroupTile.n / SWIZZLE_BLOCK_SIZE) % 4 == 0)
            y = 4;
    }
    else if (requiredX == -1 && requiredY != -1)
    {
        y = requiredY;
        if (gemm->swizzleScale && (gemm->workgroupTile.m / SWIZZLE_BLOCK_SIZE) % 4 == 0)
            x = 4;
    }
    else if (requiredX != -1 && requiredY != -1)
    {
        x = requiredX;
        y = requiredY;
    }

    return {x * gemm->wavefrontSize, y};
}

std::shared_ptr<SolutionParameters>
    genSolutionParameters(const KernelType&              kernelType,
                          const SolutionIndexParameters& solutionIndexParameters)
{
    namespace SolutionParams = rocRoller::Parameters::Solution;
    auto gemm = std::make_shared<SolutionParameters>();

    gemm->kernelType = kernelType;

    gemm->workgroupTile = solutionIndexParameters.workgroupTile;

    gemm->machineInstruction = pickMI(gemm->kernelType.typeA, gemm->kernelType.typeB, gemm->workgroupTile);

    if(solutionIndexParameters.prefetchInFlight == 1)
    {
        gemm->prefetch = false;
    }
    else
    {
        gemm->prefetchInFlight = solutionIndexParameters.prefetchInFlight;
    }

    // Swizzle Scale only support in certain situations
    // Swizzle Scale also runs out of registers with FP8
    if (kernelType.scaleAMode != rocRoller::Operations::ScaleMode::Separate ||
        kernelType.scaleBMode != rocRoller::Operations::ScaleMode::Separate)
    {
        gemm->swizzleScale = false;
        gemm->prefetchScale = false;
        gemm->loadLDSScaleA = false;
        gemm->loadLDSScaleB = false;
    }
    else if(solutionIndexParameters.workgroupTile.m >= 128
        && solutionIndexParameters.workgroupTile.n >= 128)
    {
        gemm->swizzleScale  = true;
        gemm->loadLDSScaleA = false;
        gemm->loadLDSScaleB = false;
    }
    else
    {
        gemm->swizzleScale  = false;
        gemm->prefetchScale = false;
        gemm->loadLDSScaleA = true;
        gemm->loadLDSScaleB = true;
    }

    auto workgroupSize = pickWorkgroupSize(gemm);
    gemm->workgroupSizeX = workgroupSize.first;
    gemm->workgroupSizeY = workgroupSize.second;

    // Direct To LDS only supported in certain situations
    if(kernelType.typeA == rocRoller::DataType::FP6 || kernelType.typeA == rocRoller::DataType::BF6)
        gemm->loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
    if(kernelType.typeB == rocRoller::DataType::FP6 || kernelType.typeB == rocRoller::DataType::BF6)
        gemm->loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
    if((kernelType.typeA == rocRoller::DataType::FP4
        || kernelType.typeB == rocRoller::DataType::FP4)
       && (solutionIndexParameters.workgroupTile.m <= 64
           || solutionIndexParameters.workgroupTile.n <= 64))
    {
        gemm->loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm->loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
    }

    if(not(SolutionParams::IsBufferToLDS(gemm->loadPathA)
           and SolutionParams::IsBufferToLDS(gemm->loadPathB)))
    {
        gemm->prefetchLDSFactor = 2;
    }

    // LDS can only be used for scaling data with certain workgroup tile sizes
    auto workgroupSizeTotal = gemm->workgroupSizeX * gemm->workgroupSizeY;
    auto numScaleElementsA = 0;
    if(gemm->kernelType.scaleABlockRowSize * gemm->kernelType.scaleABlockColSize != 0)
    {
        numScaleElementsA = gemm->workgroupTile.m
          * (gemm->workgroupTile.k
             / (gemm->kernelType.scaleABlockRowSize * gemm->kernelType.scaleABlockColSize));
    }
    auto numScaleElementsB = 0;
    if(gemm->kernelType.scaleBBlockRowSize * gemm->kernelType.scaleBBlockColSize != 0)
    {
        numScaleElementsB = gemm->workgroupTile.n
          * (gemm->workgroupTile.k
             / (gemm->kernelType.scaleBBlockRowSize * gemm->kernelType.scaleBBlockColSize));
    }
    if(numScaleElementsA % workgroupSizeTotal != 0)
    {
        gemm->loadLDSScaleA     = false;
        gemm->prefetchMixMemOps = false;
    }
    if(numScaleElementsB % workgroupSizeTotal != 0)
    {
        gemm->loadLDSScaleB     = false;
        gemm->prefetchMixMemOps = false;
    }

    if(!solutionIndexParameters.workgroupMapping)
    {
        gemm->workgroupMappingDim = -1;
        gemm->workgroupRemapXCC = false;
    }
    else
    {
        gemm->workgroupMappingDim = 0;
        gemm->workgroupRemapXCC = true;
    }

    // TODO: StreamK is not currently working with prefetching or workgroup mapping
    if(gemm->streamK)
    {
        gemm->prefetch = false;
        gemm->workgroupMappingDim = -1;
        gemm->workgroupRemapXCC = false;
    }

    return gemm;
}
