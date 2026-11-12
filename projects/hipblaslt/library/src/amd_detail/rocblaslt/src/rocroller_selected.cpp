// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The rocRoller-off path does not rank tiles itself. It calls
// chooseSolutionIndexParameters, which is what rocRoller calls before codegen.

#include "include/rocroller_selected.hpp"

#include "solution_selection.hpp"

namespace
{
    // DataTypeInfo::Get().elementBits for the types rocRoller is handed.
    size_t elementBits(hipDataType type)
    {
        switch(type)
        {
        case HIP_R_4F_E2M1:
            return 4;
        case HIP_R_6F_E2M3:
        case HIP_R_6F_E3M2:
            return 6;
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E5M2:
            return 8;
        case HIP_R_16F:
        case HIP_R_16BF:
            return 16;
        case HIP_R_32F:
            return 32;
        case HIP_R_64F:
            return 64;
        default:
            return 0;
        }
    }

    // RocRollerGemmKernel::exceedsBufferAddressingLimit: one buffer descriptor
    // per tensor, whose 32-bit byte count wraps at 4 GiB.
    bool exceedsBufferAddressingLimit(const KernelType& kernelType, size_t m, size_t n, size_t k)
    {
        constexpr size_t maxBufferBytes = size_t(1) << 32;
        auto             bytesOf
            = [](size_t elems, hipDataType type) { return (elems * elementBits(type) + 7u) / 8u; };
        return bytesOf(m * k, kernelType.typeA) >= maxBufferBytes
               || bytesOf(k * n, kernelType.typeB) >= maxBufferBytes
               || bytesOf(m * n, kernelType.typeC) >= maxBufferBytes
               || bytesOf(m * n, kernelType.typeD) >= maxBufferBytes;
    }
} // namespace

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
    if(exceedsBufferAddressingLimit(kernelType, prob.m, prob.n, prob.k))
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
        tiles.push_back(tile);
    }
    return rocblaslt_status_success;
}
