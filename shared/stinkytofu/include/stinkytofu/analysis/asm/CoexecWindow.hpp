/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#pragma once

#include <cstdint>

#include "stinkytofu/analysis/asm/WmmaHideBudgetAnalysis.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

/// The co-execution window a matrix instruction opens, as a cycle timeline.
///
/// Answers only what the ISA says about one matrix op's latency shadow: which
/// cycles something can issue into, which of them a VALU can co-execute in, and
/// where the timeline currently stands. It holds no queue policy, no region
/// target and no memory model, which is what makes it shareable -- a scheduler
/// building a schedule and a pass repairing one disagree about nearly
/// everything else, but not about this.
///
/// Three pieces of per-opcode ISA data drive it, all read from the instruction:
///   - `latencyCycles`    how long the shadow is;
///   - `coIssueWindow`    bit i set = a VALU may co-execute at cycle i;
///   - `blockedScaleMask` cycles the hardware takes outright (the LD_SCALE of a
///                        scale WMMA), END-anchored so one mask stays correct
///                        across every per-format latency override.
///
/// Cycle positions are measured from the matrix op's own issue, so position 0 is
/// the cycle it issued in and \ref open leaves the timeline just past it.
///
/// On the name: co-execution is what the AMDGPU backend calls this -- see
/// GCNHazardRecognizer::fixWMMACoexecutionHazards. It reserves "co-issue" for a
/// different question, SIInstrInfo::isNeverCoissue, which asks whether two
/// instructions can share one issue slot (the VOPD dual-issue encoding) rather
/// than whether one can execute in another's shadow. Much of the surrounding
/// code here predates that distinction and still says co-issue, including the
/// `coIssueWindow` ISA field this class reads.
class CoexecWindow {
   public:
    /// Take \p matrixOp's window, positioned at its issue cycle.
    ///
    /// Separate from \ref open for a caller that keeps its own clock alongside
    /// this one: it needs the matrix op's issue cycles to land on both, so it
    /// resets here and advances through whatever advances them together.
    void reset(const StinkyInstruction& matrixOp) {
        coexecMask_ = matrixOp.coIssueWindow;
        latency_ = matrixOp.latencyCycles;
        const HwInstDesc* desc = matrixOp.getHwInstDesc();
        blockedMask_ = desc != nullptr ? desc->blockedScaleMask : 0;
        pos_ = 0;
    }

    /// Start the window \p matrixOp opens, positioned just past its own issue.
    void open(const StinkyInstruction& matrixOp) {
        reset(matrixOp);
        advance(matrixOp.issueCycles);
    }

    /// Close the window: nothing more can issue into this shadow.
    void close() {
        pos_ = latency_;
    }

    /// Move \p cycles down the timeline.
    ///
    /// Never comes to rest on a blocked cycle. Every caller reads the position to
    /// decide what may issue next and nothing may issue there, so the timeline
    /// rolls on to the next issuable cycle instead; the skipped cycles still
    /// elapse, the hardware is just spending them itself.
    void advance(int cycles) {
        int landing = pos_ + cycles;
        while (isBlockedCycle(landing)) ++landing;
        pos_ = landing;
    }

    /// Cycles elapsed since the matrix op issued.
    int position() const {
        return pos_;
    }

    /// The shadow's full length. Zero when no matrix op is active, which makes
    /// every "inside the window" test false without a separate flag.
    int latency() const {
        return latency_;
    }

    /// True once the shadow is spent.
    bool closed() const {
        return pos_ >= latency_;
    }

    /// Cycles of genuinely free shadow left: to the window's end, or to the first
    /// blocked cycle if one comes sooner.
    int freeSpace() const {
        for (int pos = pos_; pos < latency_; ++pos)
            if (isBlockedCycle(pos)) return pos - pos_;
        return latency_ > pos_ ? latency_ - pos_ : 0;
    }

    /// True when \p pos lands on a cycle the hardware reserves outright.
    bool isBlockedCycle(int pos) const {
        return isBlockedWindowCycle(pos, latency_, blockedMask_);
    }

    /// Whether a VALU op may co-execute at the current position. Outside the
    /// window a VALU has the pipe to itself, so there is nothing to permit.
    bool valuPickable() const {
        if (closed()) return true;
        return ((coexecMask_ >> pos_) & 1u) != 0u;
    }

    /// Elapsed cycles a VALU op costs from here.
    ///
    /// More than its issue cycles when the next co-issue slot is not adjacent:
    /// inside the window only the cycles `coIssueWindow` permits make VALU
    /// progress, so the op waits out the ones that do not.
    int valuAdvanceCycles(int issueCycles) const {
        if (issueCycles <= 0) return 0;
        if (closed()) return issueCycles;

        constexpr int kCoexecBits = static_cast<int>(sizeof(coexecMask_) * 8);
        int elapsed = 0;
        int issued = 0;
        while (issued < issueCycles) {
            const int pos = pos_ + elapsed;
            bool canIssue = true;
            if (pos < latency_) {
                canIssue = (pos < kCoexecBits) && (((coexecMask_ >> pos) & 1u) != 0u) &&
                           !isBlockedCycle(pos);
            }
            if (canIssue) issued++;
            elapsed++;
        }
        return elapsed;
    }

    /// Co-execution slots this window offers in total, spent or not.
    int slots() const {
        return __builtin_popcount(static_cast<unsigned>(coexecMask_));
    }

   private:
    uint16_t coexecMask_ = 0;
    uint16_t blockedMask_ = 0;
    int latency_ = 0;
    int pos_ = 0;
};

}  // namespace stinkytofu
