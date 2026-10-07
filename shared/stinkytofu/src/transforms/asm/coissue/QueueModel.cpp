// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "QueueModel.hpp"

#include <algorithm>

namespace stinkytofu::coissue {

int QueueModel::issueAt(int at) const {
    const size_t depth = static_cast<size_t>(std::max(0, depth_));
    if (depth > 0 && ops_.size() >= depth) at = std::max(at, ops_[ops_.size() - depth].start);
    return at;
}

const PipeOp& QueueModel::push(int issue, int latency) {
    const int start = ops_.empty() ? issue : std::max(issue, ops_.back().end);
    ops_.push_back({issue, start, start + latency});
    return ops_.back();
}

int afterBarrierWait(SyncModel sync, int t, const QueueModel& queue) {
    // Conservative: the release comes no earlier than the end of the queued matrix work, so
    // every compute change before a barrier counts in full.
    if (sync == SyncModel::Conservative && !queue.ops().empty())
        return std::max(t, queue.pipeFree());
    return t;
}

}  // namespace stinkytofu::coissue
