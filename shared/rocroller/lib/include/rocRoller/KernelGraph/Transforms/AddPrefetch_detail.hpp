/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller::KernelGraph
{
    namespace AddPrefetchDetail
    {
        /**
         * @brief Container for info related to loading from Global
         * into LDS.
         */
        struct LDSOperationInfo
        {
            int user; // User coordinate
            int globalOperation; // LoadTiled/StoreTiled operation
            int ldsOperation; // StoreLDSTile/LoadLDSTile operation
            int globalChain; // LoadTiled/StoreTiled operation
            int ldsChain; // StoreLDStile/LoadLDSTile operation
        };

        /**
         * @brief Find loops (and loads in them) that can be prefetched.
         *
         * To find prefetch candidates:
         *
         * 1. Look for LoadTiled operations that have ForLoop
         *    dimensions in their associated coordinate transform.
         *
         * 2. Find their containing ForLoop operation and make sure
         *    the loops associated coordinate is contained in the set
         *    above.
         *
         * 3. Make sure there is a neighboring Unroll coordinate
         *    beside the ForLoop coordinate.
         *
         * 4. Make sure the size of the Unroll coordinate is
         *    consistent with the requested number of prefetches.
         */
        std::map<int, int> findPrefetch(KernelGraph const& kgraph);

        /**
         * @brief Is the LoadTile for an Exchange?
         *
         * Checks if the loads destination tile is associated with an
         * Exchange operation.
         *
         * The tile is either directly connected to an Exchange, or
         * it's connected through an Index to a tile that is directly
         * connected to an Exchange.
         */
        bool isLoadForExchange(int loadTag, KernelGraph const& graph);
    }
}
