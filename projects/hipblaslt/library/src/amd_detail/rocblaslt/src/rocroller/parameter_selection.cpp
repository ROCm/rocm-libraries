// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "parameter_selection.hpp"

const int SWIZZLE_BLOCK_SIZE = 64;

std::string SolutionParameters::toString() const
{
    std::stringstream result;

    result << "WorkGroupTile:" << workgroupTile.m << "x" << workgroupTile.n << "x"
           << workgroupTile.k << std::endl;
    result << "MachineInstruction:" << machineInstruction.m << "x" << machineInstruction.n << "x"
           << machineInstruction.k << std::endl;
    result << "WorkgroupSize:" << workgroupSizeX << "x" << workgroupSizeY << std::endl;
    result << "StreamK: " << streamK << std::endl;
    result << "LoadA: " << loadPathA << std::endl;
    result << "LoadB: " << loadPathB << std::endl;
    result << "StoreD:" << storePath << std::endl;
    result << "Workgroup Mapping: Dim:" << workgroupMappingDim << " RemapXCC:" << workgroupRemapXCC
           << std::endl;
    result << "Prefetch:" << prefetch << " InFlight:" << prefetchInFlight
           << " LDSFactor:" << prefetchLDSFactor << " MixMemOps:" << prefetchMixMemOps << std::endl;
    result << "Block Scale Options:" << " Swizzle Scale:" << swizzleScale
           << " Prefetch Scale:" << prefetchScale << " loadPathAScale:" << loadPathAScale
           << " loadPathBScale:" << loadPathBScale << std::endl;
    result << "SwizzleTileSize: M:" << swizzleTileSize.m << " K:" << swizzleTileSize.k
           << " N:" << swizzleTileSize.n << " L:" << swizzleTileSize.l << std::endl;

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

    auto isPreSwizzle = gemm->kernelType.scaleTypeA.preSwizzleTile.size() == 3 && gemm->kernelType.scaleTypeB.preSwizzleTile.size() == 3;

    auto swizzleBlockSize = isPreSwizzle ? gemm->kernelType.scaleTypeA.preSwizzleTile[0] : SWIZZLE_BLOCK_SIZE;

    //Swizzle Scale only works with certain combinations of workgroup sizes
    if(gemm->swizzleScale && (gemm->workgroupTile.m / swizzleBlockSize) % x != 0)
        requiredX = 1;
    if(gemm->swizzleScale && (gemm->workgroupTile.n / swizzleBlockSize) % y != 0)
        requiredY = 1;

    if(requiredX != -1 && requiredY == -1)
    {
        x = requiredX;
        if(gemm->swizzleScale && (gemm->workgroupTile.n / swizzleBlockSize) % 4 == 0)
            y = 4;
    }
    else if(requiredX == -1 && requiredY != -1)
    {
        y = requiredY;
        if(gemm->swizzleScale && (gemm->workgroupTile.m / swizzleBlockSize) % 4 == 0)
            x = 4;
    }
    else if(requiredX != -1 && requiredY != -1)
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
    auto gemm                = std::make_shared<SolutionParameters>();

    gemm->kernelType    = kernelType;
    gemm->workgroupTile = solutionIndexParameters.workgroupTile;

    // Check if pre-swizzled scale data is requested
    // preSwizzleTile format: {tileMN, tileK, subTileK} e.g., {32, 8, 4}
    bool hasPreSwizzleA = kernelType.scaleTypeA.preSwizzleTile.size() == 3;
    bool hasPreSwizzleB = kernelType.scaleTypeB.preSwizzleTile.size() == 3;
    bool hasPreSwizzle  = hasPreSwizzleA && hasPreSwizzleB;

    bool hasPreTileA = kernelType.scaleTypeA.preTile.size() == 2;
    bool hasPreTileB = kernelType.scaleTypeB.preTile.size() == 2;
    bool hasPreTile  = hasPreTileA && hasPreTileB;

    // Get preSwizzleTileMN for MI selection (0 if no pre-swizzle)
    size_t preSwizzleTileMN = 0;
    if(hasPreSwizzle)
    {
        preSwizzleTileMN = kernelType.scaleTypeA.preSwizzleTile[0];
    }

    gemm->machineInstruction = pickMI(
        gemm->kernelType.typeA, gemm->kernelType.typeB, gemm->workgroupTile, preSwizzleTileMN);

    gemm->prefetchInFlight
        = preferredUnrolling(kernelType.typeA, kernelType.typeB, gemm->workgroupTile, hasPreSwizzle, hasPreTile);
    if(gemm->prefetchInFlight <= 1)
        gemm->prefetch = false;

    // Check if the workgroup tile K dimension is 256
    bool isWorkgroupTileK256 = solutionIndexParameters.workgroupTile.k == 256;

    // Swizzle Scale only support in certain situations
    // Swizzle Scale also runs out of registers with FP8
    if(kernelType.scaleTypeA.mode != rocRoller::Operations::ScaleMode::Separate
       || kernelType.scaleTypeB.mode != rocRoller::Operations::ScaleMode::Separate)
    {
        gemm->swizzleScale   = false;
        gemm->prefetchScale  = false;
        gemm->loadPathAScale = SolutionParams::LoadPath::BufferToVGPR;
        gemm->loadPathBScale = SolutionParams::LoadPath::BufferToVGPR;
    }
    else if(isWorkgroupTileK256)
    {
        // For 256x256x256 tile, use BufferToLDS for scale loading to reduce register pressure
        gemm->swizzleScale   = true;
        gemm->prefetchScale  = true;

        auto isValidworkgroupTileMN = (solutionIndexParameters.workgroupTile.m >= 128 && solutionIndexParameters.workgroupTile.n >= 128)
                                      || (solutionIndexParameters.workgroupTile.m == 32 && solutionIndexParameters.workgroupTile.n == 32);

        if (isValidworkgroupTileMN)
        {
            gemm->loadPathAScale = SolutionParams::LoadPath::BufferToLDS;
            gemm->loadPathBScale = SolutionParams::LoadPath::BufferToLDS;
        }
        else
        {
            gemm->loadPathAScale = SolutionParams::LoadPath::BufferToVGPR;
            gemm->loadPathBScale = SolutionParams::LoadPath::BufferToVGPR;
        }
    }
    else if(solutionIndexParameters.workgroupTile.m >= 128
        && solutionIndexParameters.workgroupTile.n >= 128)
    {
        gemm->swizzleScale = true;
        // Use BufferToVGPR for swizzle scale (matches rocRoller client --loadScale_A BufferToVGPR)
        gemm->loadPathAScale = SolutionParams::LoadPath::BufferToVGPR;
        gemm->loadPathBScale = SolutionParams::LoadPath::BufferToVGPR;
    }
    else
    {
        gemm->swizzleScale  = false;
        gemm->prefetchScale = false;
        // Use BufferToLDSViaVGPR for LDS-based scale loads
        gemm->loadPathAScale = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm->loadPathBScale = SolutionParams::LoadPath::BufferToLDSViaVGPR;
    }

    auto workgroupSize   = pickWorkgroupSize(gemm);
    gemm->workgroupSizeX = workgroupSize.first;
    gemm->workgroupSizeY = workgroupSize.second;

    // Pass StreamK flag from solution index parameters
    if(solutionIndexParameters.streamK)
        gemm->streamK = rocRoller::StreamKMode::TwoTile;

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
        if (gemm->streamK == rocRoller::StreamKMode::None)
        {
            gemm->loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
            gemm->loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        }
        else
        {
            gemm->loadPathA = SolutionParams::LoadPath::BufferToVGPR;
            gemm->loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        }
    }

    if(not(SolutionParams::IsBufferToLDS(gemm->loadPathA)
           and SolutionParams::IsBufferToLDS(gemm->loadPathB)))
    {
        gemm->prefetchLDSFactor = 2;
    }

    auto scaleIsBufferToLDS = SolutionParams::IsBufferToLDS(gemm->loadPathAScale) and SolutionParams::IsBufferToLDS(gemm->loadPathBScale);
    auto is32MN = solutionIndexParameters.workgroupTile.m == 32 && solutionIndexParameters.workgroupTile.n == 32;
    if (scaleIsBufferToLDS and is32MN)
    {
        gemm->prefetchLDSFactor = 1;
    }

    // LDS can only be used for scaling data with certain workgroup tile sizes
    auto workgroupSizeTotal = gemm->workgroupSizeX * gemm->workgroupSizeY;
    auto numScaleElementsA  = 0;
    if(gemm->kernelType.scaleTypeA.blockRowSize * gemm->kernelType.scaleTypeA.blockColSize != 0)
    {
        numScaleElementsA = gemm->workgroupTile.m
                            * (gemm->workgroupTile.k
                               / (gemm->kernelType.scaleTypeA.blockRowSize
                                  * gemm->kernelType.scaleTypeA.blockColSize));
    }
    auto numScaleElementsB = 0;
    if(gemm->kernelType.scaleTypeB.blockRowSize * gemm->kernelType.scaleTypeB.blockColSize != 0)
    {
        numScaleElementsB = gemm->workgroupTile.n
                            * (gemm->workgroupTile.k
                               / (gemm->kernelType.scaleTypeB.blockRowSize
                                  * gemm->kernelType.scaleTypeB.blockColSize));
    }
        if(numScaleElementsA % workgroupSizeTotal != 0)
        {
            gemm->loadPathAScale    = SolutionParams::LoadPath::BufferToVGPR;
            gemm->prefetchMixMemOps = false;
        }
        if(numScaleElementsB % workgroupSizeTotal != 0)
        {
            gemm->loadPathBScale    = SolutionParams::LoadPath::BufferToVGPR;
            gemm->prefetchMixMemOps = false;
    }

    if(!solutionIndexParameters.workgroupMapping)
    {
        gemm->workgroupMappingDim = -1;
        gemm->workgroupRemapXCC   = false;
    }
    else
    {
        gemm->workgroupMappingDim = 0;
        gemm->workgroupRemapXCC   = true;
    }

    // Pass tailLoops flag from solution index parameters
    gemm->tailLoops = solutionIndexParameters.tailLoops;

    // StreamK is not currently working with workgroup mapping due to register pressure
    if(gemm->streamK != rocRoller::StreamKMode::None)
    {
        gemm->workgroupMappingDim = -1;
        gemm->workgroupRemapXCC   = false;
    }

    // Select swizzle tile size
    if(gemm->swizzleScale)
    {
        gemm->swizzleTileSize = selectSwizzleTileSize(
            gemm->workgroupTile,
            gemm->machineInstruction,
            gemm->workgroupSizeX,
            gemm->workgroupSizeY,
            gemm->prefetchInFlight,
            gemm->kernelType.scaleTypeA.blockRowSize * gemm->kernelType.scaleTypeA.blockColSize,
            gemm->kernelType.scaleTypeA.preSwizzleTile);
    }

    return gemm;
}
