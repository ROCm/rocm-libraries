// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The rocRoller-off path does not rank tiles itself. It calls
// chooseSolutionIndexParameters, which is what rocRoller calls before codegen.

#include "include/rocroller_selected.hpp"

#include "solution_selection.hpp"
#include "utility.hpp"

#include <iostream>

bool rocRollerSupportsProblem(const RocblasltContractionProblem& prob)
{
    if(prob.bias != nullptr)
    {
        std::cerr << "rocRoller does not support bias" << std::endl;
        return false;
    }

    if(prob.batch_count != 1)
    {
        std::cerr << "rocRoller only supports 1 batch_count not " << prob.batch_count << std::endl;
        return false;
    }

    // Asked in the types hipBLASLt already uses, not a generator's: this target
    // builds without rocRoller's headers so that a checkout without
    // shared/rocroller still compiles it. normalizeOperandType answers
    // HIPBLASLT_DATATYPE_INVALID for a format no ranked kernel handles, which
    // is the "unset" case and not a refusal. Comparing here also avoids
    // hip2RocRollerType, which throws on a format it cannot name.
    const auto scaleType = normalizeOperandType(prob.scale_type);
    if(scaleType != HIPBLASLT_DATATYPE_INVALID && scaleType != HIP_R_32F)
    {
        std::cerr << "rocRoller only supports F32 as scale type not "
                  << hipDataType_to_string(scaleType) << std::endl;
        return false;
    }

    const auto typeAcc = genKernelType(prob).typeAcc;
    if(typeAcc != HIP_R_32F)
    {
        std::cerr << "rocRoller only supports F32 accumulation, not "
                  << hipDataType_to_string(typeAcc) << std::endl;
        return false;
    }

    return true;
}

rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles)
{
    tiles.clear();

    if(!rocRollerSupportsProblem(prob))
        return rocblaslt_status_invalid_value;

    const KernelType kernelType = genKernelType(prob);
    const auto       ranked     = chooseSolutionIndexParameters(kernelType, prob, -1);
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
