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
 * ************************************************************************ */

#pragma once

#include <array>
#include <map>
#include <vector>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace stinkytofu {

// Models the s_wait_<c>cnt instructions that StinkyWaitCntInsertionPass inserts
// AFTER scheduling, so the wait-free scheduler can reserve the issue slot each one
// costs. Without this, a wait that lands inside a WMMA co-issue window steals a slot
// the scheduler already handed to VALU work -> a 1-cycle bubble per inner-loop
// iteration (see CDNA5.hpp co-issue timeline).
//
// Structure follows LLVM's WaitcntBrackets (SIInsertWaitcnts.cpp): per hardware
// counter, a lower/upper score bracket [lb, ub]. Each producer bumps ub and stamps
// its dest regs with the new ub as their score. A consumer reading a reg whose score
// is still > lb forces a wait; applying it advances lb to that score, which drains
// that op AND every older op on the counter in one step (in-order retirement). The
// producer -> counter mapping reuses waitcnt::classifyMemOp so it always matches the
// pass that actually emits the waits.
//
// Assumption: each modeled counter carries a single event class, so retirement is
// in-order and the bracket is exact. (LLVM only falls back to conservative
// waitcnt(0) via counterOutOfOrder() when different event types share one counter;
// CK_DS = ds_read/ds_write/ds_atomic all retire on dscnt in order, so that does not
// apply here.)
class MemWaitCntModel {
   public:
    using CounterKind = waitcnt::CounterKind;

    // `reserved` selects which counters actually charge an issue cycle when a wait
    // fires. Producers on every counter are still recorded; non-reserved counters
    // just never contribute a reserved cycle. Default {CK_DS} scopes the behavior
    // change to the ds_read -> wmma dscnt case.
    explicit MemWaitCntModel(std::initializer_list<CounterKind> reserved = {waitcnt::CK_DS}) {
        for (CounterKind c : reserved) reserved_[c] = true;
    }

    // Register-file-aware key so vector / scalar / accumulator regs of the same index
    // never collide. Mirrors regDepKey in CDNA5.hpp; the model only ever compares its
    // own producer keys against its own consumer keys, so the two need not agree.
    static int regKey(RegType type, uint32_t idx) {
        return (static_cast<int>(type) << 20) | static_cast<int>(idx & 0xFFFFF);
    }

    // ---- instruction-level API (scheduler-facing) ----

    // Record a memory op as outstanding on its counter. No-op for non-producers.
    void addProducer(const StinkyInstruction& memOp) {
        const CounterKind c = waitcnt::classifyMemOp(memOp);
        if (c == waitcnt::CK_Count) return;
        recordProducer(c, destKeys(memOp));
    }

    // True if issuing `consumer` now would require an s_wait_<c>cnt on a reserved
    // counter (it reads a reg still outstanding there). Entry point for callers that
    // only want the yes/no without mutating state.
    bool isWaitNeeded(const StinkyInstruction& consumer) const {
        return waitCostForSrcKeys(srcKeys(consumer)) > 0;
    }

    // Apply the waits `consumer` needs and return the issue cycles to reserve (one per
    // reserved counter that fires). Drains the consumed op + all older ops on each
    // fired counter.
    int applyWait(const StinkyInstruction& consumer) {
        return applyWaitForSrcKeys(srcKeys(consumer));
    }

    void reset() {
        for (int c = 0; c < waitcnt::CK_Count; ++c) {
            brackets_[c] = Bracket{};
            regScore_[c].clear();
        }
    }

    // ---- key-level core (unit-test-facing; instruction API delegates here) ----

    void recordProducer(CounterKind c, const std::vector<int>& destRegKeys) {
        Bracket& b = brackets_[c];
        ++b.ub;
        for (int k : destRegKeys) regScore_[c][k] = b.ub;
    }

    // True if a consumer reading `srcRegKeys` would need a wait on any reserved counter.
    bool isWaitNeededForSrcKeys(const std::vector<int>& srcRegKeys) const {
        return waitCostForSrcKeys(srcRegKeys) > 0;
    }

    // Cycles a consumer reading `srcRegKeys` would reserve, without mutating state.
    int waitCostForSrcKeys(const std::vector<int>& srcRegKeys) const {
        int cost = 0;
        for (int c = 0; c < waitcnt::CK_Count; ++c) {
            if (!reserved_[c]) continue;
            if (maxOutstandingScore(static_cast<CounterKind>(c), srcRegKeys) > brackets_[c].lb)
                ++cost;
        }
        return cost;
    }

    // Apply and mutate: advance each fired counter's lb to the newest score the
    // consumer reads, draining that op and all older on the counter.
    int applyWaitForSrcKeys(const std::vector<int>& srcRegKeys) {
        int cost = 0;
        for (int c = 0; c < waitcnt::CK_Count; ++c) {
            const int newest = maxOutstandingScore(static_cast<CounterKind>(c), srcRegKeys);
            if (newest <= brackets_[c].lb) continue;  // nothing outstanding to wait on
            brackets_[c].lb = newest;                 // drain target + all older
            if (reserved_[c]) ++cost;
        }
        return cost;
    }

    // Remaining outstanding depth on a counter (ub - lb); the hook for a future model
    // that charges cycles proportional to depth instead of a flat 1.
    int outstanding(CounterKind c) const {
        return brackets_[c].ub - brackets_[c].lb;
    }

   private:
    struct Bracket {
        int lb = 0;
        int ub = 0;
    };

    // Highest score among `srcRegKeys` currently tracked on counter `c` (0 if none).
    int maxOutstandingScore(CounterKind c, const std::vector<int>& srcRegKeys) const {
        int newest = 0;
        for (int k : srcRegKeys) {
            auto it = regScore_[c].find(k);
            if (it != regScore_[c].end() && it->second > newest) newest = it->second;
        }
        return newest;
    }

    static std::vector<int> destKeys(const StinkyInstruction& inst) {
        return collectKeys(inst.getDestRegs());
    }
    static std::vector<int> srcKeys(const StinkyInstruction& inst) {
        return collectKeys(inst.getSrcRegs());
    }
    static std::vector<int> collectKeys(const std::vector<StinkyRegister>& regs) {
        std::vector<int> keys;
        for (const StinkyRegister& r : regs) {
            if (!r.isRegister() || isPseudoReg(r)) continue;
            for (unsigned off = 0; off < r.reg.num; ++off)
                keys.push_back(regKey(r.reg.type, r.reg.idx + off));
        }
        return keys;
    }

    std::array<Bracket, waitcnt::CK_Count> brackets_;
    std::array<std::map<int, int>, waitcnt::CK_Count> regScore_;  // per counter: key -> score
    std::array<bool, waitcnt::CK_Count> reserved_{};
};

}  // namespace stinkytofu
