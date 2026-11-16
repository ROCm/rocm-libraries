// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

// RocRollerSelectedTile, the kernel match and the walk, in headers that depend
// on nothing GPU-side so they can be unit-tested GPU-free.
#include "rocroller_kernel_select.hpp"

#include <cstddef>
#include <vector>

// RocRollerGemmKernel::exceedsBufferAddressingLimit: one buffer descriptor
// per tensor, whose 32-bit byte count wraps at 4 GiB. Bits per element match
// DataTypeInfo for the types rocRoller is handed; an unknown type is 0 bits
// and does not trip the limit.
inline bool rocRollerOperandExceeds4GiB(hipDataType typeA,
                                        hipDataType typeB,
                                        hipDataType typeC,
                                        hipDataType typeD,
                                        size_t      m,
                                        size_t      n,
                                        size_t      k)
{
    constexpr size_t maxBufferBytes = size_t(1) << 32;
    auto             bits           = [](hipDataType type) -> size_t {
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
    };
    auto bytesOf = [&](size_t elems, hipDataType type) { return (elems * bits(type) + 7u) / 8u; };
    return bytesOf(m * k, typeA) >= maxBufferBytes || bytesOf(k * n, typeB) >= maxBufferBytes
           || bytesOf(m * n, typeC) >= maxBufferBytes || bytesOf(m * n, typeD) >= maxBufferBytes;
}

// Same rejects and the same rank order as getRocRollerBestSolutions.
// Implemented by calling chooseSolutionIndexParameters. A problem with an
// operand of 4 GiB or more gets no tiles: rocRoller refuses it at every tile.
rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles);
