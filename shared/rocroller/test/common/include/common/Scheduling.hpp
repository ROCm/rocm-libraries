// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <rocRoller/Utilities/Error.hpp>
#include <utility>
#include <vector>

namespace rocRoller
{
    /**
     * @brief Generates LDS addresses simulating bank conflicts between workitems
     *
     * Assumes bank entries are 4 bytes (1 dword) apart.
     *
     * E.g. for workgroupSize=2, strideMultiplier=2, instrDwords=1, returns [0, 8];
     * so workitem 0 accesses bank 0, workitem 1 accesses bank 2.
     *
     * @param workgroupSize Number of work items in the workgroup
     * @param strideMultiplier The degree of bank conflict (1 = every bank is used, 2 = every other bank used, etc.)
     * @param instrDwords Instruction size in dwords
     * @return std::vector<size_t> Vector of calculated LDS addresses
     */
    std::vector<size_t>
        generateLDSAddresses(size_t workgroupSize, size_t strideMultiplier, size_t instrDwords);

    /**
     * @brief Finds the median value of an odd-sized vector
     *
     * @tparam T Type of elements in the vector
     * @param values Vector with odd number of elements
     * @return T Median value
     */
    template <typename T>
    T MedianOfOddElements(std::vector<T> values)
    {
        AssertFatal(!values.empty(), "vector must not be empty");
        AssertFatal(values.size() % 2 == 1, "vector size must be odd");
        const auto n = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + n, values.end());
        return values[n];
    }

    /**
     * @brief Returns aligned register subset [start, end) with wraparound
     *
     * @param totalRegs Total number of registers available
     * @param requestedRegCount Number of registers requested
     * @param position Position index
     * @return std::pair<size_t, size_t> Start and end indices of the aligned subset
     */
    std::pair<size_t, size_t>
        getAlignedSubset(size_t totalRegs, size_t requestedRegCount, size_t position);

} // namespace rocRoller
