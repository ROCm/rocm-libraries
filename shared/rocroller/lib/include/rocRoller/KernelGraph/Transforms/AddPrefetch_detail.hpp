// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
         * @brief Container for info for LoadTiled operations
         */
        struct LoadTiledInfo
        {
            int user; // User coordinate
            int tag; // LoadTiled operation tag
            int top; // Top SetCoordinate of in the chain for this load
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

        bool IsDirectLoadToVGPR(KernelGraph const& k, int loadTag);
    }
}
