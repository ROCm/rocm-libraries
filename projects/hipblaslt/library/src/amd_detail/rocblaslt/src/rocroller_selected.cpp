// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The rocRoller-off path does not rank tiles itself. It calls
// chooseSolutionIndexParameters, which is what rocRoller calls before codegen.

#include "include/rocroller_selected.hpp"

#include "solution_selection.hpp"

rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles)
{
    tiles.clear();

    if(prob.bias != nullptr || prob.batch_count != 1)
        return rocblaslt_status_invalid_value;

    const KernelType kernelType = genKernelType(prob);
    const auto       scaleType  = hipDataType_to_rocRoller_type(prob.scale_type);
    if(scaleType != rocRoller::DataType::None && scaleType != rocRoller::DataType::Float)
        return rocblaslt_status_invalid_value;
    if(kernelType.typeAcc != rocRoller::DataType::Float)
        return rocblaslt_status_invalid_value;

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
        tiles.push_back(tile);
    }
    return rocblaslt_status_success;
}
