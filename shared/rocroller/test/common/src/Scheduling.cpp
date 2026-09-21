// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common/Scheduling.hpp"

namespace rocRoller
{
    std::vector<size_t>
        generateLDSAddresses(size_t count, size_t strideMultiplier, size_t instrDwords)
    {
        std::vector<size_t> addresses;
        for(size_t workitemId = 0; workitemId < count; ++workitemId)
        {
            size_t address = workitemId * (4 * strideMultiplier * instrDwords);
            addresses.push_back(address);
        }
        return addresses;
    }

    std::pair<size_t, size_t>
        getAlignedSubset(size_t totalRegs, size_t requestedRegCount, size_t position)
    {
        // If run out of registers, wrap around
        size_t num_complete_chunks = totalRegs / requestedRegCount;
        if(num_complete_chunks == 0)
        {
            return {0, 0};
        }
        size_t chunk_index = position % num_complete_chunks;
        size_t start       = chunk_index * requestedRegCount;
        return {start, start + requestedRegCount};
    }

} // namespace rocRoller
