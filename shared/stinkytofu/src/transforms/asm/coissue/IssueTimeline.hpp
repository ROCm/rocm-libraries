// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The co-issue timeline: issue cycles of one wave's instruction stream, with every
// number taken from a TimingProfile. It keeps the rules of CDNA5ReadyQueue:
//   - a matrix op opens a window of its latency cycles at its issue (pickOneFromWMMA);
//     without a matrix queue it issues only once the previous window has ended;
//   - inside a window a VALU issues only on a co-issue cycle that is not blocked
//     (computeValuAdvanceCycles);
//   - the stream never rests on a blocked cycle (isBlockedWindowCycle, advanceTime);
//   - a consumer waits out its producer's latency (stampDataReady) and the cycle
//     hazard gaps of HWModel::hazards (kCdna5HazardRules);
// and adds what the scheduler does not see: wait instructions (issue + settle, then
// the hold until the counter is met; DS ops return in order), context-dependent issue
// costs, latency rules, and the matrix queue in front of the pipe.

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include "QueueModel.hpp"
#include "TimingProfile.hpp"

namespace stinkytofu {
struct StinkyInstruction;
}

namespace stinkytofu::coissue {

/// One instruction as the timeline sees it.
struct TimedInst {
    const StinkyInstruction* inst = nullptr;
    int opcode = -1;
    IssueClass kind = IssueClass::Any;  ///< never Any or Branch, except for labels
    bool isLabel = false;
    bool isBranch = false;
    bool isBarrierWait = false;
    bool isNop = false;      ///< s_nop N: N + 1 cycles
    bool isDsWait = false;   ///< s_wait_dscnt
    bool isWaitAlu = false;  ///< s_wait_alu
    bool fp4Operands = false;
    int issue = 1;
    int latency = 1;
    int count = -1;  ///< s_nop N, s_wait_*cnt N
    int winLatency = 0;
    uint16_t coIssueMask = 0;
    uint16_t blockedMask = 0;
    /// Register slots (regSlot) of every destination / source lane; pseudo registers
    /// are left out.
    std::vector<uint16_t> defs;
    std::vector<uint16_t> uses;
    /// The uses a gating producer of the stream can write (TimedInstCache::restrictUses);
    /// all of `uses` unless restricted. Only these can delay the instruction.
    std::vector<uint16_t> gatedUses;
    /// Bit r: the instruction is a producer / consumer of HWModel::hazards rule r.
    uint32_t hazardProducer = 0;
    uint32_t hazardConsumer = 0;
};

/// Dense index of a register lane; VCC and EXEC halves fold into one slot each, as one
/// condition and one mask. Returns -1 for a register the timeline does not track.
int regSlot(int regType, uint32_t index);
constexpr int kNumRegSlots = 4096;
/// The register class latency rules match on.
LatencyReg regClassOfSlot(int slot);

/// The timeline's view of `inst`. `hw` supplies the hazard rules.
TimedInst makeTimedInst(const StinkyInstruction& inst, const HWModel& hw);
/// The TimedInst of a label, which costs nothing and is skipped.
TimedInst makeLabel();

/// Timed instructions keyed by the IR instruction, built once.
class TimedInstCache {
   public:
    explicit TimedInstCache(const HWModel& hw);
    const TimedInst& get(const StinkyInstruction& inst);
    /// Only `gating[slot]` registers can hold a result that delays a consumer (every scalar
    /// and vector def of the streams to come): drop every other slot from gatedUses, now and
    /// for instructions built later.
    void restrictUses(std::vector<bool> gating);

   private:
    void restrict(TimedInst& t) const;

    const HWModel& hw_;
    // Open addressing on the instruction pointer; the instructions live in `store_`.
    std::vector<std::pair<const StinkyInstruction*, uint32_t>> table_;
    std::deque<TimedInst> store_;
    std::vector<bool> gating_;
};

struct Placement {
    int cycle = 0;    ///< issue cycle
    int window = -1;  ///< index of the matrix op whose window the instruction sits in
    int pos = 0;      ///< cycles after that matrix op's issue
    int stall = 0;    ///< cycles a wait held the wave beyond its own cost
};

class IssueTimeline {
   public:
    explicit IssueTimeline(const TimingProfile& profile);

    /// Start over at cycle 0, keeping the tables.
    void reset();

    /// Issue `inst` at the earliest cycle the rules allow, and no earlier than
    /// `notBefore`; advance the stream past it. Labels are skipped.
    Placement place(const TimedInst& inst, int notBefore = 0);

    /// The cycle the next instruction may issue at.
    int now() const {
        return t_;
    }
    const std::vector<PipeOp>& pipe() const {
        return queue_.ops();
    }
    const TimingProfile& profile() const {
        return profile_;
    }

   private:
    struct Producer {
        uint32_t gen = 0;
        int at = 0;
        int after = 0;
        const TimedInst* inst = nullptr;
    };

    bool inWindow(int c) const;
    bool blocked(int c) const;
    int roll(int c) const;
    int dataReady(const TimedInst& inst) const;
    int costOf(const TimedInst& inst) const;
    bool matches(const OpMatch& m, const TimedInst& inst) const;
    bool matchesPrev(const OpMatch& m) const;

    const TimingProfile& profile_;
    int t_ = 0;
    bool hasWindow_ = false;
    int winStart_ = 0;
    int winLatency_ = 0;
    uint16_t winMask_ = 0;
    uint16_t winBlocked_ = 0;
    int window_ = -1;
    std::vector<int> dsDone_;
    QueueModel queue_;
    std::vector<Producer> producers_;
    /// [gap][slot]: the cycle a consumer of that hazard gap may read the slot, valid for
    /// generation gen_.
    std::vector<std::vector<std::pair<uint32_t, int>>> hazardReady_;
    uint32_t gen_ = 1;
    bool hasPrev_ = false;
    IssueClass prevKind_ = IssueClass::Any;
    bool prevKindIsBranch_ = false;
    int prevOpcode_ = -1;
    bool prevIsWait_ = false;
};

/// The last of `trips` back-to-back iterations of a loop body.
struct TripTiming {
    /// Issue length: from the first instruction of the trip to the end of the stream.
    int cycles = 0;
    /// Matrix-pipe idle between the trip's matrix ops (timeline2.steady_trip).
    int pipeIdle = 0;
    /// The same, plus the hand-over from the previous trip's last matrix op.
    int pipeIdleWithHandover = 0;
    /// Placement of every body instruction in the trip (labels get cycle -1).
    std::vector<Placement> placements;
    /// The trip's matrix ops, in order.
    std::vector<PipeOp> pipe;
    /// The previous trip's last matrix op, if any.
    std::optional<PipeOp> handover;
    /// Timing stopped early: the pipe idle already exceeded the bound.
    bool overBound = false;

    TripCost cost() const {
        return {pipeIdleWithHandover, cycles};
    }
};

constexpr int kSteadyTrips = 2;

TripTiming steadyTrip(const std::vector<const TimedInst*>& body, const TimingProfile& profile,
                      int trips = kSteadyTrips);
/// The same on `timeline` (reset first), which keeps its tables between calls. Without
/// `placements`, TripTiming::placements stays empty. With `idleBound`, timing stops once the
/// last trip's pipe idle exceeds it (idle only grows): the result then has `overBound` set.
TripTiming steadyTrip(const std::vector<const TimedInst*>& body, IssueTimeline& timeline,
                      bool placements = true, int trips = kSteadyTrips,
                      std::optional<int> idleBound = std::nullopt);

}  // namespace stinkytofu::coissue
