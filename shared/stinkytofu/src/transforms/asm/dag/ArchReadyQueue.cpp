/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "ArchReadyQueue.hpp"

#include <iostream>

#define DEBUG_TYPE "StinkyDAGSchedulerPass"

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

// The only translation unit that includes an arch ready queue. Keeping it to one
// keeps those queues file-local, so no internal-linkage type reaches an exported
// interface.
#include "CDNA5.hpp"

namespace stinkytofu {
namespace dag {

std::unique_ptr<ReadyQueue> createArchReadyQueue(const PassContext& passCtx,
                                                 ArchReadyQueueOptions options) {
    if (passCtx.getGemmTileConfig().arch[0] == 12 && passCtx.getGemmTileConfig().arch[1] == 5) {
        PASS_DEBUG(std::cerr << "Using CDNA5ReadyQueue for scheduling\n");
        return std::make_unique<CDNA5ReadyQueue>(passCtx, options);
    }
    // Only CDNA5ReadyQueue models a co-issue window at all, so no other queue has
    // anything to apply these to.
    (void)options;
    // The SCC chain lock applyClusterBarrierSccRule sets up is carried in the node fields and
    // honoured only by CDNA5ReadyQueue's pick loop. ReadyQueueByDAGid pops by id and reads
    // none of them, so it would issue a handshake barrier straight through an open chain --
    // the clobber this rule exists to prevent, and silently.
    if (passCtx.getPassFeatureConfig().dagFeatures.clusterBarrier) {
        STINKY_UNREACHABLE("ClusterBarrier scheduling requires CDNA5ReadyQueue");
    }
    PASS_DEBUG(std::cerr << "Using Default ReadyQueue for scheduling\n");
    return std::make_unique<ReadyQueueByDAGid>(passCtx);
}

}  // namespace dag
}  // namespace stinkytofu
