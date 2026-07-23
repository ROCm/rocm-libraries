// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/transforms/asm/waitcnt/WaitCntValidator.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitCntDataflowUtils.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
namespace waitcnt {

using namespace utils;

namespace {

/// True iff `inst` is any s_wait_* counter instruction (register or tensor).
bool isWaitInst(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

/// Map an s_wait_* opcode to the counter it drains. Used as a fallback for
/// hand-written STIR whose waits carry only a literal keep operand and no
/// SWaitCntData / SWaitTensorCntData modifier.
CounterKind waitOpcodeCounter(const StinkyInstruction& inst) {
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_wait_dscnt:
            return CK_DS;
        case GFX::s_wait_loadcnt:
            return CK_Buffer;
        case GFX::s_wait_kmcnt:
            return CK_KM;
        case GFX::s_wait_tensorcnt:
            return CK_Tensor;
        default:
            return CK_Count;
    }
}

int firstLiteralIntSrc(const StinkyInstruction& inst) {
    for (const StinkyRegister& r : inst.getSrcRegs()) {
        if (r.dataType == StinkyRegister::Type::LiteralInt) return r.getLiteralInt();
    }
    return 0;
}

/// Read the per-counter keep values from an existing s_wait_* instruction.
void readWaitCounts(const StinkyInstruction& inst, int keep[CK_Count], bool present[CK_Count]) {
    for (int c = 0; c < CK_Count; ++c) present[c] = false;

    if (const auto* sw = inst.getModifier<SWaitCntData>()) {
        if (sw->dlcnt != -1) {
            keep[CK_DS] = sw->dlcnt;
            present[CK_DS] = true;
        }
        if (sw->vlcnt != -1) {
            keep[CK_Buffer] = sw->vlcnt;
            present[CK_Buffer] = true;
        }
        if (sw->kmcnt != -1) {
            keep[CK_KM] = sw->kmcnt;
            present[CK_KM] = true;
        }
    }
    if (const auto* tw = inst.getModifier<SWaitTensorCntData>()) {
        if (tw->tlcnt != -1) {
            keep[CK_Tensor] = tw->tlcnt;
            present[CK_Tensor] = true;
        }
    }

    bool any = present[CK_DS] || present[CK_Buffer] || present[CK_KM] || present[CK_Tensor];
    if (!any) {
        CounterKind c = waitOpcodeCounter(inst);
        if (c != CK_Count) {
            keep[c] = firstLiteralIntSrc(inst);
            present[c] = true;
        }
    }
}

const char* counterWaitOp(CounterKind c) {
    switch (c) {
        case CK_DS:
            return "s_wait_dscnt";
        case CK_Buffer:
            return "s_wait_loadcnt";
        case CK_KM:
            return "s_wait_kmcnt";
        case CK_Tensor:
            return "s_wait_tensorcnt";
        default:
            return "?";
    }
}

const char* mnemonicOf(const StinkyInstruction* inst) {
    if (inst == nullptr || inst->getHwInstDesc() == nullptr) return "?";
    return inst->getHwInstDesc()->mnemonic;
}

}  // namespace

std::vector<WaitValidationViolation> WaitCntValidator::validate(
    Function& /*func*/, const std::vector<BasicBlock*>& rpo) {
    std::vector<WaitValidationViolation> violations;

    std::array<RawWaitPredicate, CK_Count> rawNeedsWait;
    rawNeedsWait[CK_DS] = [](const StinkyInstruction&) { return true; };
    rawNeedsWait[CK_Buffer] = [](const StinkyInstruction&) { return true; };
    rawNeedsWait[CK_KM] = [](const StinkyInstruction&) { return true; };
    const unsigned nw = numWaves;
    rawNeedsWait[CK_Tensor] = [nw](const StinkyInstruction& i) {
        return isBarrier(i) || nw == 1;
    };

    std::unordered_set<int> reportedCounters;
    auto transfer = [&](BasicBlock& bb, DataflowState& state,
                        std::vector<WaitValidationViolation>* reportInto) {
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            if (isPhi(*inst)) continue;

            if (isWaitInst(*inst)) {
                int keep[CK_Count] = {0, 0, 0, 0};
                bool present[CK_Count] = {false, false, false, false};
                readWaitCounts(*inst, keep, present);
                for (int c = 0; c < CK_Count; ++c) {
                    if (present[c]) trimQueues(state.queues[c], keep[c]);
                }
                continue;
            }

            int required[CK_Count];
computeRequiredWaits(inst, state, rawNeedsWait, required);

            if (reportInto != nullptr) {
                for (int c = 0; c < CK_Count; ++c) {
                    if (required[c] == WaitCountSpec::kUnused) continue;
                    if (reportedCounters.count(c)) continue;
                    reportedCounters.insert(c);

                    StinkyInstruction* producer = nullptr;
                    int bestDepth = 0;
                    for (const auto& q : state.queues[static_cast<CounterKind>(c)]) {
                        const int qsize = static_cast<int>(q.ops.size());
                        for (int idx = 0; idx < qsize; ++idx) {
                            int depth = qsize - idx;
                            if (producer == nullptr || depth < bestDepth) {
                                producer = q.ops[idx];
                                bestDepth = depth;
                            }
                        }
                    }

                    WaitValidationViolation v;
                    v.counter = static_cast<CounterKind>(c);
                    v.consumer = inst;
                    v.producer = producer;
                    v.message = std::string("consumer '") + mnemonicOf(inst) + "' in ^" +
                                bb.getLabel() + " uses async producer '" + mnemonicOf(producer) +
                                "' still in flight (missing " + counterWaitOp(v.counter) + " " +
                                std::to_string(required[c]) + ")";
                    reportInto->push_back(std::move(v));
                }
            }

            CounterKind self = classifyMemOp(*inst);
            if (self != CK_Count) appendToAllPaths(state.queues[self], inst);
        }
    };

    std::unordered_map<const BasicBlock*, DataflowState> exitStates;
    for (BasicBlock* bb : rpo) exitStates[bb] = DataflowState();

    const unsigned n = static_cast<unsigned>(rpo.size());
    const unsigned floor = static_cast<unsigned>(kMaxInFlight) + 8u;
    const unsigned iterationCap = std::min<unsigned>(256u, std::max<unsigned>(floor, 2u * n));

    for (unsigned iter = 0; iter < iterationCap; ++iter) {
        bool changed = false;
        for (BasicBlock* bb : rpo) {
            DataflowState working;
seedQueuesFromPredecessors(*bb, exitStates, working);
            transfer(*bb, working, /*reportInto=*/nullptr);
            if (!(exitStates[bb] == working)) {
                exitStates[bb] = std::move(working);
                changed = true;
            }
        }
        if (!changed) break;
    }

    reportedCounters.clear();
    for (BasicBlock* bb : rpo) {
        DataflowState entry;
seedQueuesFromPredecessors(*bb, exitStates, entry);
        transfer(*bb, entry, &violations);
        if (reportedCounters.size() == CK_Count) break;
    }

    return violations;
}

}  // namespace waitcnt
}  // namespace stinkytofu
