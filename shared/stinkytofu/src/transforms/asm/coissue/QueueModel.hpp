// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// What turns issue cycles into matrix-pipe time (WAITCNT_COISSUE.md 7.7).
//   QueueModel: a matrix op leaves the issue stream as soon as the queue in front of the
//               pipe has room, and the pipe runs each op for its window latency. A window's
//               extra issue cycles cost something only once the wave's lead is used up.
//   SyncModel:  what a barrier wait does to queued matrix work (TimingProfile::sync).

#include <vector>

#include "TimingProfile.hpp"

namespace stinkytofu::coissue {

/// One matrix op in the pipe: when it left the issue stream, and when the pipe ran it.
struct PipeOp {
    int issue = 0;
    int start = 0;
    int end = 0;
};

class QueueModel {
   public:
    /// `depth` matrix ops can wait in front of the pipe; 0 means no queue.
    explicit QueueModel(int depth) : depth_(depth) {}

    int depth() const {
        return depth_;
    }
    /// The earliest cycle at or after `at` a matrix op can leave the issue stream at: once
    /// fewer than `depth` ops wait in front of the pipe.
    int issueAt(int at) const;
    /// A matrix op issued at `issue` starts once the pipe is free and runs `latency` cycles.
    const PipeOp& push(int issue, int latency);
    /// The cycle the pipe finishes everything queued so far.
    int pipeFree() const {
        return ops_.empty() ? 0 : ops_.back().end;
    }
    const std::vector<PipeOp>& ops() const {
        return ops_;
    }

   private:
    int depth_;
    std::vector<PipeOp> ops_;
};

/// The earliest cycle the wave issues again after an s_barrier_wait that ends at `t`.
int afterBarrierWait(SyncModel sync, int t, const QueueModel& queue);

/// What an order costs under one profile. The pipe's busy time is fixed by the loop's
/// matrix ops, so the steady trip is busy time plus idle; the issue length breaks ties.
struct TripCost {
    /// Matrix-pipe idle per steady trip, including the hand-over from the previous trip.
    int pipeIdle = 0;
    /// Issue length of the steady trip.
    int cycles = 0;

    auto operator<=>(const TripCost&) const = default;
    TripCost operator+(const TripCost& o) const {
        return {pipeIdle + o.pipeIdle, cycles + o.cycles};
    }
};

}  // namespace stinkytofu::coissue
