// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The rocRoller-off path does not rank tiles itself. It calls
// chooseSolutionIndexParameters, which is what rocRoller calls before codegen.

#include "include/rocroller_selected.hpp"

#include "solution_selection.hpp"
#include "utility.hpp"

bool rocRollerSupportsProblem(const RocblasltContractionProblem& prob)
{
    if(prob.bias != nullptr || prob.batch_count != 1)
        return false;

    // Asked in the types hipBLASLt already uses, not a generator's: this target
    // builds without rocRoller's headers so that a checkout without
    // shared/rocroller still compiles it. normalizeOperandType answers
    // HIPBLASLT_DATATYPE_INVALID for a format no ranked kernel handles, which
    // is the "unset" case and not a refusal.
    const auto scaleType = normalizeOperandType(prob.scale_type);
    if(scaleType != HIPBLASLT_DATATYPE_INVALID && scaleType != HIP_R_32F)
        return false;

    return genKernelType(prob).typeAcc == HIP_R_32F;
}

rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles)
{
    tiles.clear();

    if(!rocRollerSupportsProblem(prob))
        return rocblaslt_status_invalid_value;

    const KernelType kernelType = genKernelType(prob);
    const auto ranked = chooseSolutionIndexParameters(kernelType, prob, -1);
    // Same inputs parameter_selection.cpp feeds preferredUnrolling. The name
    // records that unroll, so the match has to ask for it.
    const bool hasPreSwizzle = kernelType.scaleTypeA.preSwizzleTile.size() == 3
                               && kernelType.scaleTypeB.preSwizzleTile.size() == 3;
    const bool hasPreTile    = kernelType.scaleTypeA.preTile.size() == 2
                            && kernelType.scaleTypeB.preTile.size() == 2;
    const bool blockA = isBlockScaling(kernelType.scaleTypeA.format);
    const bool blockB = isBlockScaling(kernelType.scaleTypeB.format);
    tiles.reserve(ranked.size());
    for(const SolutionIndexParameters& params : ranked)
    {
        RocRollerSelectedTile tile;
        tile.m                = params.workgroupTile.m;
        tile.n                = params.workgroupTile.n;
        tile.k                = params.workgroupTile.k;
        tile.workgroupMapping = params.workgroupMapping;
        tile.streamK          = params.streamK;
        tile.nonTemporalA     = params.nonTemporalA;
        tile.nonTemporalB     = params.nonTemporalB;
        tile.unroll           = preferredUnrolling(kernelType.typeA,
                                         kernelType.typeB,
                                         params.workgroupTile,
                                         hasPreSwizzle,
                                         hasPreTile);
        if(blockA)
        {
            tile.scaleABlockRow = static_cast<int>(kernelType.scaleTypeA.blockRowSize);
            tile.scaleABlockCol = static_cast<int>(kernelType.scaleTypeA.blockColSize);
        }
        if(blockB)
        {
            tile.scaleBBlockCol = static_cast<int>(kernelType.scaleTypeB.blockColSize);
            tile.scaleBBlockRow = static_cast<int>(kernelType.scaleTypeB.blockRowSize);
        }
        tiles.push_back(tile);
    }
    return rocblaslt_status_success;
}
