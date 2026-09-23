/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <array>
#include <vector>

namespace WorkgroupClustersDetail
{
    constexpr unsigned int MaxWorkgroupsPerCluster = 16;

    /**
     * @brief Checks if the given cluster size is valid w.r.t. the number of workgroups.
     *
     * A valid cluster size sepcification must:
     * - Have all dimensions greater than zero.
     * - Divide evenly into the corresponding number of workgroups in each dimension.
     * - Not exceed the maximum allowed workgroups per cluster.
     *
     * @param clusterSize The size of the cluster in each dimension (x, y, z).
     * @param numWorkgroups The number of workgroups in each dimension (x, y, z).
     * @return true if the cluster size is valid, false otherwise.
     */
    inline bool IsValidWorkgroupClusterSize(const std::array<unsigned int, 3>& clusterSize,
                                            const std::array<unsigned int, 3>& numWorkgroups)
    {
        return (clusterSize[0] > 0) and (clusterSize[1] > 0) and (clusterSize[2] > 0)
               and (numWorkgroups[0] % clusterSize[0] == 0)
               and (numWorkgroups[1] % clusterSize[1] == 0)
               and (numWorkgroups[2] % clusterSize[2] == 0)
               and (clusterSize[0] * clusterSize[1] * clusterSize[2] <= MaxWorkgroupsPerCluster);
    }

    /**
     * @brief Returns all valid workgroup cluster sizes w.r.t. the number of workgroups.
     *
     * @param numWorkgroups The number of workgroups in each dimension (x, y, z).
     * @return a vector of valid cluster sizes.
     */
    inline std::vector<std::array<unsigned int, 3>>
        ValidWorkgroupClusterSizes(const std::array<unsigned int, 3>& numWorkgroups)
    {
        unsigned int const limitX = std::min(numWorkgroups[0], MaxWorkgroupsPerCluster);
        unsigned int const limitY = std::min(numWorkgroups[1], MaxWorkgroupsPerCluster);
        unsigned int const limitZ = std::min(numWorkgroups[2], MaxWorkgroupsPerCluster);

        std::vector<std::array<unsigned int, 3>> valid;
        for(unsigned int x = 1; x <= limitX; ++x)
            for(unsigned int y = 1; y <= limitY; ++y)
                for(unsigned int z = 1; z <= limitZ; ++z)
                    if(IsValidWorkgroupClusterSize({x, y, z}, numWorkgroups))
                        valid.push_back({x, y, z});
        return valid;
    }
}
