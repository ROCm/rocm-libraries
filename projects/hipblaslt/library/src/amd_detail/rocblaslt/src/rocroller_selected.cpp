// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The rocRoller-off path does not rank tiles itself. It calls
// chooseSolutionIndexParameters, which is what rocRoller calls before codegen.

#include "include/rocroller_selected.hpp"
#include "include/rocroller_unbuildable.hpp"

#include "solution_selection.hpp"

rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles)
{
    tiles.clear();

    if(prob.bias != nullptr || prob.batch_count != 1)
        return rocblaslt_status_invalid_value;

    const KernelType kernelType = genKernelType(prob);
    const auto       scaleType  = normalizeOperandType(prob.scale_type);
    if(scaleType != HIPBLASLT_DATATYPE_INVALID && scaleType != HIP_R_32F)
        return rocblaslt_status_invalid_value;
    if(kernelType.typeAcc != HIP_R_32F)
        return rocblaslt_status_invalid_value;

    // rocRoller refuses every tile of such a problem.
    if(rocRollerOperandExceeds4GiB(
           kernelType.typeA, kernelType.typeB, kernelType.typeC, kernelType.typeD, prob.m, prob.n, prob.k))
        return rocblaslt_status_success;

    // Same inputs parameter_selection.cpp feeds preferredUnrolling.
    const bool hasPreSwizzle = kernelType.scaleTypeA.preSwizzleTile.size() == 3
                               && kernelType.scaleTypeB.preSwizzleTile.size() == 3;
    const bool hasPreTile
        = kernelType.scaleTypeA.preTile.size() == 2 && kernelType.scaleTypeB.preTile.size() == 2;

    const auto ranked = chooseSolutionIndexParameters(kernelType, prob, -1);
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
        tile.unroll           = preferredUnrolling(
            kernelType.typeA, kernelType.typeB, params.workgroupTile, hasPreSwizzle, hasPreTile);
        tile.unbuildable = rocRollerTileUnbuildable(kernelType, tile);
        tiles.push_back(tile);
    }
    return rocblaslt_status_success;
}
