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

#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "InsertWaitAluPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/RegHalfKeyer.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace {
using namespace stinkytofu;

// ---------------------------------------------------------------------------
// Mode 2 counters and events (VA_VDST, VM_VSRC).
// ---------------------------------------------------------------------------

enum CounterType : uint8_t {
    CT_VA_VDST = 0,
    CT_VM_VSRC = 1,
    NUM_COUNTERS = 2,
};

enum WaitEventType : uint8_t {
    // VA_VDST events: VALU VGPR-dest writes.
    EV_VGPR_CSMACC_WRITE = 0,  // core/side-MACC VALU (v_add_f32, v_mul_f32, v_mfma, ...)
    EV_VGPR_DPMACC_WRITE,      // double-precision MACC (v_add/mul/fma_f64, f64 cmp, v_cvt_u32_f64)
    EV_VGPR_TRANS_WRITE,       // transcendental VALU, 32- and 64-bit (v_rcp_f32, v_rcp_f64, ...)
    EV_VGPR_XDL_WRITE,         // XDL WMMA / SWMMAC
    // VM_VSRC events
    EV_VGPR_LDS_READ,   // ds_read / ds_write reading a VGPR source
    EV_VGPR_FLAT_READ,  // FLAT reading a VGPR source
    EV_VGPR_VMEM_READ,  // buffer / global / image / tensor reading a VGPR source
    EV_NUM,
};

// Compact bitset over WaitEventType; twoOrMore() flags "out-of-order" completion
// (more than one event class pending on a single counter ⇒ force wait(0)).
struct WaitEventSet {
    uint32_t mask = 0;
    void insert(WaitEventType e) {
        mask |= 1u << e;
    }
    bool contains(WaitEventType e) const {
        return mask & (1u << e);
    }
    bool containsAll(WaitEventSet o) const {
        return (mask & o.mask) == o.mask;
    }
    bool twoOrMore() const {
        return (mask & (mask - 1)) != 0;
    }
    WaitEventSet operator|(WaitEventSet o) const {
        return {mask | o.mask};
    }
    WaitEventSet operator&(WaitEventSet o) const {
        return {mask & o.mask};
    }
    WaitEventSet operator~() const {
        return {~mask};
    }
    bool operator==(WaitEventSet o) const {
        return mask == o.mask;
    }
};

inline CounterType counterFromEvent(WaitEventType e) {
    switch (e) {
        case EV_VGPR_CSMACC_WRITE:
        case EV_VGPR_DPMACC_WRITE:
        case EV_VGPR_TRANS_WRITE:
        case EV_VGPR_XDL_WRITE:
            return CT_VA_VDST;
        default:
            return CT_VM_VSRC;
    }
}

inline const char* counterName(CounterType c) {
    return c == CT_VA_VDST ? "va_vdst" : "vm_vsrc";
}

inline const char* eventName(WaitEventType e) {
    switch (e) {
        case EV_VGPR_CSMACC_WRITE:
            return "CSMACC_WRITE";
        case EV_VGPR_DPMACC_WRITE:
            return "DPMACC_WRITE";
        case EV_VGPR_TRANS_WRITE:
            return "TRANS_WRITE";
        case EV_VGPR_XDL_WRITE:
            return "XDL_WRITE";
        case EV_VGPR_LDS_READ:
            return "LDS_READ";
        case EV_VGPR_FLAT_READ:
            return "FLAT_READ";
        case EV_VGPR_VMEM_READ:
            return "VMEM_READ";
        default:
            return "?";
    }
}

inline WaitEventSet eventsForCounter(CounterType c) {
    WaitEventSet s;
    if (c == CT_VA_VDST) {
        s.insert(EV_VGPR_CSMACC_WRITE);
        s.insert(EV_VGPR_DPMACC_WRITE);
        s.insert(EV_VGPR_TRANS_WRITE);
        s.insert(EV_VGPR_XDL_WRITE);
    } else {
        s.insert(EV_VGPR_LDS_READ);
        s.insert(EV_VGPR_FLAT_READ);
        s.insert(EV_VGPR_VMEM_READ);
    }
    return s;
}

// Comma-separated list of pending event names in `ev`, e.g. "XDL_WRITE,CSMACC_WRITE".
// Used by the wait-hit debug print when ooo=1 fires, to make it clear *which*
// event classes are causing the conservative full-drain.
inline std::string pendingEventsStr(WaitEventSet ev) {
    std::string out;
    for (int i = 0; i < EV_NUM; ++i) {
        auto e = static_cast<WaitEventType>(i);
        if (ev.contains(e)) {
            if (!out.empty()) out += ",";
            out += eventName(e);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Instruction classifiers
// ---------------------------------------------------------------------------

// Map an instruction to its (single) mode2 event class, or none if it is
// neither a VALU producer nor a VMEM/LDS/FLAT consumer.
// VALU completion-class order: XDL -> TRANS -> DPMACC -> CSMACC. f64
// transcendentals carry both the TRANS and DPMACC properties; TRANS is matched
// first so they classify as TRANS.
std::optional<WaitEventType> classifyEvent(const StinkyInstruction& inst) {
    if (isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst)) {
        if (isXDLWMMA(inst)) return EV_VGPR_XDL_WRITE;
        if (isTranscendental(inst)) return EV_VGPR_TRANS_WRITE;  // 32- and 64-bit
        if (isDPMACC(inst)) return EV_VGPR_DPMACC_WRITE;
        return EV_VGPR_CSMACC_WRITE;
    }
    if (isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst)) return EV_VGPR_LDS_READ;
    if (isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst)) return EV_VGPR_FLAT_READ;
    // VMEM family. Stinkytofu does not yet flag scratch / image / sample / BVH
    // instructions; on archs that emit them they belong in this same bucket.
    if (isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) || isGLOBALLoad(inst) ||
        isGLOBALStore(inst) || isTensorLoad(inst))
        return EV_VGPR_VMEM_READ;
    return std::nullopt;
}

// VOP3PX2 / VOP3PX3 are software-only encodings of a back-to-back VOP3P pair
// (LD_SCALE + WMMA). Hardware decodes each as two separate VOP3P sub-issues,
// both bumping VA_VDST, so software must count 2.
inline bool hasMatrixScalePair(const StinkyInstruction& inst) {
    auto mc = inst.getHwInstDesc()->microcode;
    return mc == MicrocodeFormat::MC_VOP3PX2 || mc == MicrocodeFormat::MC_VOP3PX3;
}

// EXEC writes invalidate any non-zero VA_VDST wait (skipped VALUs don't bump
// the HW counter). Covers explicit destination and implicit destination via
// HW flag.
inline bool writesExec(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitWriteEXEC)) return true;
    for (const auto& d : inst.getDestRegs()) {
        if (d.dataType != StinkyRegister::Type::Register) continue;
        RegType t = d.reg.type;
        if (t == RegType::EXEC || t == RegType::EXEC_LO || t == RegType::EXEC_HI) return true;
    }
    return false;
}

inline bool isWaitAluInst(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_wait_alu;
}

// isReturn = kernel exit (s_endpgm). Used to drop mode2 before the wave exits.
// Note: function-call returns (s_setpc_b64 s[26:27]) are intentionally NOT
// handled here — mode2 is confined to the loop region.
inline bool isReturn(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_endpgm;
}

// Last real (non-pseudo) instruction in a block, or nullptr if the block is
// label-only. Walks back from the terminator, skipping trailing pseudos
// (label / asm directive).
inline StinkyInstruction* lastRealInst(BasicBlock& bb) {
    for (IRBase* n = bb.getTerminator(); n; n = n->getPrev()) {
        auto* inst = dyn_cast<StinkyInstruction>(n);
        if (inst && !isPseudoInst(inst)) return inst;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// True16 half-selectors
// ---------------------------------------------------------------------------

// True16 half-selector for dest operand index `destIdx` (only operand 0 and 1
// can have a True16 dst-half). Falls through to NONE without modifier.
inline HighBitSel destHalfSel(const True16Modifiers* mod, size_t destIdx) {
    if (!mod) return HighBitSel::NONE;
    if (destIdx == 0) return mod->getDst0();
    if (destIdx == 1) return mod->getDst1();
    return HighBitSel::NONE;
}

inline HighBitSel srcHalfSel(const True16Modifiers* mod, size_t srcIdx) {
    return mod ? mod->getSrc(srcIdx) : HighBitSel::NONE;
}

// ---------------------------------------------------------------------------
// Wait struct
// ---------------------------------------------------------------------------

// Sentinel: "don't emit this field" for a per-counter wait value.
constexpr unsigned kNoWait = ~0u;

struct Wait {
    std::array<unsigned, NUM_COUNTERS> counts = {kNoWait, kNoWait};
    unsigned get(CounterType c) const {
        return counts[c];
    }
    void set(CounterType c, unsigned v) {
        counts[c] = v;
    }
    bool hasAny() const {
        return counts[CT_VA_VDST] != kNoWait || counts[CT_VM_VSRC] != kNoWait;
    }
};

inline void setNoWait(Wait& w, CounterType c) {
    w.set(c, kNoWait);
}
inline bool isNoWait(const Wait& w, CounterType c) {
    return w.get(c) == kNoWait;
}

inline void addWait(Wait& w, CounterType c, unsigned v) {
    w.set(c, std::min(w.get(c), v));
}

// SWaitAluData field widths: va_vdst is 4 bits, vm_vsrc is 3 bits. The all-ones
// value of each field is reserved as the "no-wait" sentinel, so the largest
// emittable real wait is (1 << width) - 2.
inline unsigned encodingSentinel(CounterType c) {
    return c == CT_VA_VDST ? 15u : 7u;
}
inline unsigned maxEmittableWait(CounterType c) {
    return encodingSentinel(c) - 1;
}

// ---------------------------------------------------------------------------
// WaitcntBrackets — UB/LB scoreboard with per-VGPR per-counter scores
// ---------------------------------------------------------------------------

using PerCounterScores = std::array<unsigned, NUM_COUNTERS>;

class WaitcntBrackets {
   public:
    unsigned getScoreLB(CounterType c) const {
        return scoreLB[c];
    }
    unsigned getScoreUB(CounterType c) const {
        return scoreUB[c];
    }
    unsigned getScoreRange(CounterType c) const {
        return scoreUB[c] - scoreLB[c];
    }
    size_t scoresSize() const {
        return scores.size();
    }

    unsigned getVGPRScore(RegKey k, CounterType c) const {
        auto it = scores.find(k);
        return it == scores.end() ? 0u : it->second[c];
    }

    // Stamp scoreboard after instruction `inst` issues with event `ev`.
    // VA_VDST stamps each VGPR def, VM_VSRC stamps each VGPR src.
    void onProducer(WaitEventType ev, const StinkyInstruction& inst, const VGPRHalfKeyer& keyer) {
        CounterType ct = counterFromEvent(ev);
        unsigned inc = (ct == CT_VA_VDST && hasMatrixScalePair(inst)) ? 2u : 1u;
        unsigned curr = scoreUB[ct] + inc;
        scoreUB[ct] = curr;
        pendingEvents.insert(ev);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   stamp " << counterName(ct)
                             << " event=" << eventName(ev) << " inc=" << inc << " new_ub=" << curr
                             << " (mnemonic=" << inst.getHwInstDesc()->mnemonic << ")\n");

        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        if (ct == CT_VA_VDST) {
            size_t destIdx = 0;
            for (size_t i = 0; i < inst.getNumDestRegs(); ++i) {
                const auto& reg = inst.getDestReg(i);
                if (reg.dataType != StinkyRegister::Type::Register) continue;
                if (reg.reg.type != RegType::V) continue;
                HighBitSel half = destHalfSel(true16Mod, destIdx);
                ++destIdx;
                for (uint16_t off = 0; off < reg.reg.num; ++off) {
                    RegKey k = keyer.producerKey(reg.reg.idx + off, half);
                    scores[k][CT_VA_VDST] = curr;
                    // Per-VGPR stamp record so a later "wait hit ... score=N"
                    // can be grepped back to the producer instruction.
                    PASS_DEBUG(std::cerr << "[InsertWaitAlu]     score=" << curr << " on v" << k.idx
                                         << "(" << halfName(k.half) << ") " << eventName(ev)
                                         << "\n");
                }
            }
        } else {
            for (size_t i = 0; i < inst.getNumSrcRegs(); ++i) {
                const auto& reg = inst.getSrcReg(i);
                if (reg.dataType != StinkyRegister::Type::Register) continue;
                if (reg.reg.type != RegType::V) continue;
                // VM_VSRC tracks "in-flight VMEM read of this VGPR". The
                // potential WAR is against a 32-bit VALU write to the same
                // DWORD, so a full-DWORD key is correct.
                for (uint16_t off = 0; off < reg.reg.num; ++off) {
                    RegKey k{RegType::V, reg.reg.idx + off, RegHalf::NONE};
                    scores[k][CT_VM_VSRC] = curr;
                    PASS_DEBUG(std::cerr << "[InsertWaitAlu]     score=" << curr << " on v" << k.idx
                                         << "(" << halfName(k.half) << ") " << eventName(ev)
                                         << "\n");
                }
            }
        }
    }

    // For each VGPR src (RAW on VA_VDST) and each VGPR dst (WAW on VA_VDST,
    // WAR on VM_VSRC), probe the score map and accumulate the worst-case wait.
    void onConsumer(const StinkyInstruction& inst, const VGPRHalfKeyer& keyer, Wait& wait) const {
        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        size_t srcIdx = 0;
        for (size_t i = 0; i < inst.getNumSrcRegs(); ++i) {
            const auto& reg = inst.getSrcReg(i);
            if (reg.dataType != StinkyRegister::Type::Register) continue;
            if (reg.reg.type != RegType::V) continue;
            HighBitSel half = srcHalfSel(true16Mod, srcIdx);
            ++srcIdx;
            for (uint16_t off = 0; off < reg.reg.num; ++off) {
                keyer.forEachConsumerKey(reg.reg.idx + off, half, [&](RegKey k) {
                    determineWaitForScore(CT_VA_VDST, getVGPRScore(k, CT_VA_VDST), wait, k,
                                          "src(RAW)");
                });
            }
        }

        size_t destIdx = 0;
        for (size_t i = 0; i < inst.getNumDestRegs(); ++i) {
            const auto& reg = inst.getDestReg(i);
            if (reg.dataType != StinkyRegister::Type::Register) continue;
            if (reg.reg.type != RegType::V) continue;
            HighBitSel half = destHalfSel(true16Mod, destIdx);
            ++destIdx;
            for (uint16_t off = 0; off < reg.reg.num; ++off) {
                keyer.forEachConsumerKey(reg.reg.idx + off, half, [&](RegKey k) {
                    determineWaitForScore(CT_VA_VDST, getVGPRScore(k, CT_VA_VDST), wait, k,
                                          "dst(WAW)");
                });
                // WAR on VM_VSRC: writer-vs-in-flight-VMEM-read uses full DWORD.
                RegKey full{RegType::V, reg.reg.idx + off, RegHalf::NONE};
                determineWaitForScore(CT_VM_VSRC, getVGPRScore(full, CT_VM_VSRC), wait, full,
                                      "dst(WAR)");
            }
        }
    }

    void determineWaitForScore(CounterType c, unsigned score, Wait& wait, const RegKey& k,
                               const char* role) const {
        unsigned lb = scoreLB[c];
        unsigned ub = scoreUB[c];
        if (ub >= score && score > lb) {
            unsigned chosen;
            bool ooo = counterOutOfOrder(c);
            if (ooo) {
                chosen = 0;
                addWait(wait, c, 0);
            } else {
                chosen = std::min(ub - score, maxEmittableWait(c));
                addWait(wait, c, chosen);
            }
            // Include the consumer VGPR identity and the role (src/dst hazard
            // class) so the user can trace which operand triggered the wait
            // without re-reading the source IR by hand. When ooo=1 also dump
            // the pending event set that forced the full drain.
            PASS_DEBUG(
                std::cerr << "[InsertWaitAlu]     wait hit " << counterName(c) << " on v" << k.idx
                          << "(" << halfName(k.half) << "," << role << ")" << " score=" << score
                          << " lb=" << lb << " ub=" << ub << " ooo=" << ooo
                          << (ooo ? " events={" +
                                        pendingEventsStr(pendingEvents & eventsForCounter(c)) + "}"
                                  : std::string())
                          << " → wait=" << chosen << "\n");
        }
    }

    bool counterOutOfOrder(CounterType c) const {
        WaitEventSet ev = pendingEvents & eventsForCounter(c);
        return ev.twoOrMore();
    }

    // Advance LB after a wait is inserted. count==0 fully drains the counter
    // — clear its event bits so counterOutOfOrder() no longer flags it.
    void applyWaitcnt(CounterType c, unsigned count) {
        if (count == kNoWait) return;
        unsigned ub = scoreUB[c];
        unsigned oldLB = scoreLB[c];
        unsigned newLB = ub - std::min(count, ub - oldLB);
        if (newLB > scoreLB[c]) scoreLB[c] = newLB;
        if (count == 0) pendingEvents = pendingEvents & ~eventsForCounter(c);
        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     apply " << counterName(c) << "(" << count
                             << ") LB " << oldLB << "→" << scoreLB[c] << " UB=" << ub
                             << (count == 0 ? " [drain events]" : "") << "\n");
    }

    // Widen this entry state with a predecessor's exit. Returns true (strictDom)
    // when the other side contributed a tighter score or new event type.
    bool merge(const WaitcntBrackets& other) {
        bool strictDom = false;
        struct MergeInfo {
            unsigned oldLB, otherLB, myShift, otherShift;
        };
        std::array<MergeInfo, NUM_COUNTERS> mi{};

        for (int t = 0; t < NUM_COUNTERS; ++t) {
            CounterType ct = static_cast<CounterType>(t);
            unsigned myPending = scoreUB[ct] - scoreLB[ct];
            unsigned otherPending = other.scoreUB[ct] - other.scoreLB[ct];
            unsigned newUB = scoreLB[ct] + std::max(myPending, otherPending);

            mi[t].oldLB = scoreLB[ct];
            mi[t].otherLB = other.scoreLB[ct];
            mi[t].myShift = newUB - scoreUB[ct];
            mi[t].otherShift = newUB - other.scoreUB[ct];

            scoreUB[ct] = newUB;
        }

        for (const auto& [k, _] : other.scores) scores.try_emplace(k);

        for (auto& [k, mySc] : scores) {
            auto it = other.scores.find(k);
            for (int t = 0; t < NUM_COUNTERS; ++t) {
                unsigned otherVal = (it != other.scores.end()) ? it->second[t] : 0;
                unsigned myS = mySc[t] <= mi[t].oldLB ? 0 : mySc[t] + mi[t].myShift;
                unsigned otherS = otherVal <= mi[t].otherLB ? 0 : otherVal + mi[t].otherShift;
                if (otherS > myS) strictDom = true;
                mySc[t] = std::max(myS, otherS);
            }
        }

        if (!pendingEvents.containsAll(other.pendingEvents)) strictDom = true;
        pendingEvents = pendingEvents | other.pendingEvents;
        return strictDom;
    }

   private:
    std::array<unsigned, NUM_COUNTERS> scoreLB = {0, 0};
    std::array<unsigned, NUM_COUNTERS> scoreUB = {0, 0};
    WaitEventSet pendingEvents;
    std::unordered_map<RegKey, PerCounterScores, RegKeyHash> scores;
};

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

class InsertWaitAluPassImpl : public Pass {
    std::unordered_map<BasicBlock*, WaitcntBrackets> blockEntryState;
    GfxArchID archId = GfxArchID{};
    VGPRHalfKeyer keyer{};

    StinkyInstruction* emitWaitAlu(BasicBlock& bb, IRBase* insertBefore, const Wait& wait,
                                   int hold_cnt = -1) {
        AsmIRBuilder builder(bb, archId);
        StinkyInstruction* w = builder.create(getMCIDByUOp(GFX::s_wait_alu, archId), insertBefore);
        int va = isNoWait(wait, CT_VA_VDST) ? -1 : static_cast<int>(wait.get(CT_VA_VDST));
        int vm = isNoWait(wait, CT_VM_VSRC) ? -1 : static_cast<int>(wait.get(CT_VM_VSRC));
        w->addModifier<SWaitAluData>(SWaitAluData(va, /*va_sdst=*/-1, /*va_ssrc=*/-1, hold_cnt, vm,
                                                  /*va_vcc=*/-1,
                                                  /*sa_sdst=*/-1));
        return w;
    }

    // If the instruction immediately before `consumer` in `bb` is a hold_cnt-only
    // s_wait_alu survivor, return its hold_cnt value and erase the instruction
    // so the caller can fold the hold_cnt into a freshly-emitted merged wait.
    // Returns -1 if no such survivor is adjacent.
    //
    // Scope note: this only handles the hold_cnt-only shape because that is
    // the only pre-existing s_wait_alu RemoveWaitAluPass leaves behind. A
    // general per-field min merge across non-trivial va_vdst/vm_vsrc would
    // need more care and isn't needed today.
    int extractAdjacentHoldCnt(BasicBlock& bb, IRBase* consumer) {
        auto consumerIt = IRList::iterator(consumer);
        if (consumerIt == bb.begin()) return -1;
        auto prevIt = consumerIt;
        --prevIt;
        auto* prev = dyn_cast<StinkyInstruction>(prevIt.getNodePtr());
        if (!prev || !isWaitAluInst(*prev)) return -1;
        auto* data = prev->getModifier<SWaitAluData>();
        if (!data) return -1;
        if (!data->hasField(SWaitAluData::HOLD_CNT)) return -1;
        if (data->hasField(SWaitAluData::VA_VDST)) return -1;
        if (data->hasField(SWaitAluData::VM_VSRC)) return -1;
        int hold_cnt = static_cast<int>(data->getField(SWaitAluData::HOLD_CNT));
        bb.eraseIR(prevIt);
        return hold_cnt;
    }

    Wait computeWaitForInst(const StinkyInstruction& inst, const WaitcntBrackets& sb) const {
        Wait wait;

        // Step 1: scoreboard probes on every VGPR operand of `inst`.
        sb.onConsumer(inst, keyer, wait);

        // Step 2: skip VA_VDST for VALU consumers
        if (isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst)) {
            if (!isNoWait(wait, CT_VA_VDST))
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     suppress va_vdst (VALU consumer, was "
                                     << int(wait.get(CT_VA_VDST)) << ")\n");
            setNoWait(wait, CT_VA_VDST);
        }

        // Step 3: eager EXEC guard. If this instruction modifies EXEC and any
        // VA_VDST work is in flight, drain now — subsequent VALUs may be
        // EXEC-skipped at runtime and therefore won't bump VA_VDST_hw, leaving
        // any precomputed non-zero wait invalid. Must run AFTER Step 2 so that
        // v_cmpx_* (VALU + writes EXEC) gets the va_vdst(0) drain rather than
        // the VALU suppression.
        if (writesExec(inst) && sb.getScoreRange(CT_VA_VDST) > 0) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     drain va_vdst (EXEC writer, in-flight="
                                 << sb.getScoreRange(CT_VA_VDST) << ")\n");
            addWait(wait, CT_VA_VDST, 0);
        }

        return wait;
    }

    // Process one BB starting from its accumulated entry state.
    // emit=false → run scoreboard, return exit state for Phase 1 propagation.
    // emit=true → re-run with the converged entry state and insert s_wait_alu.
    WaitcntBrackets runOnBasicBlock(BasicBlock& bb, bool emit) {
        WaitcntBrackets sb = blockEntryState[&bb];

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] " << (emit ? "emit" : "analyze") << " bb=\""
                             << bb.getLabel()
                             << "\" entry=[va_vdst LB=" << sb.getScoreLB(CT_VA_VDST)
                             << " UB=" << sb.getScoreUB(CT_VA_VDST) << " sz=" << sb.scoresSize()
                             << "; vm_vsrc LB=" << sb.getScoreLB(CT_VM_VSRC)
                             << " UB=" << sb.getScoreUB(CT_VM_VSRC) << "]\n");

        for (auto it = bb.begin(); it != bb.end();) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst) {
                ++it;
                continue;
            }
            if (isPseudoInst(inst)) {
                ++it;
                continue;
            }

            // Pre-existing s_wait_alu: absorb its va_vdst/vm_vsrc into LB so the
            // rest of the BB sees the post-wait state, and leave the instruction
            // in place so the runtime drain actually happens. Today the only
            // realistic source is hold_cnt-only survivors from RemoveWaitAluPass
            // (their va_vdst/vm_vsrc are already kNoWait, so the absorb is a
            // no-op); the emit branch below merges fresh va_vdst/vm_vsrc into
            // such a survivor when it's the immediately-preceding instruction.
            if (isWaitAluInst(*inst)) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]   absorb existing s_wait_alu\n");
                if (const auto* data = inst->getModifier<SWaitAluData>()) {
                    if (data->hasField(SWaitAluData::VA_VDST))
                        sb.applyWaitcnt(CT_VA_VDST, data->getField(SWaitAluData::VA_VDST));
                    if (data->hasField(SWaitAluData::VM_VSRC))
                        sb.applyWaitcnt(CT_VM_VSRC, data->getField(SWaitAluData::VM_VSRC));
                }
                ++it;
                continue;
            }

            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   visit " << inst->getHwInstDesc()->mnemonic
                                 << "\n");

            // Function call (s_swappc): treat as an analysis barrier — reset the
            // scoreboard so pre-call producer scores don't leak into post-call
            // tracking as phantom dependencies.
            //
            // No drain / no callee-return handling is needed: mode2 is confined
            // to the loop region (see insertSchedModeLifecycle), and every
            // s_swappc lives in the mode0 epilogue (GlobalWriteBatch is the sole
            // call emitter). In mode0 the hardware auto-stalls on all hazards —
            // caller leftovers and callee outputs are all handled by HW, so the
            // pass needs to emit nothing around the call.
            if (isCall(*inst)) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]   call — reset brackets (mode0 epilogue, "
                                        "HW handles hazards)\n");
                sb.applyWaitcnt(CT_VA_VDST, 0);
                sb.applyWaitcnt(CT_VM_VSRC, 0);
                ++it;
                continue;
            }

            Wait wait = computeWaitForInst(*inst, sb);
            if (wait.hasAny()) {
                PASS_DEBUG(std::cerr
                           << "[InsertWaitAlu]   emit s_wait_alu va_vdst="
                           << (isNoWait(wait, CT_VA_VDST) ? -1 : int(wait.get(CT_VA_VDST)))
                           << " vm_vsrc="
                           << (isNoWait(wait, CT_VM_VSRC) ? -1 : int(wait.get(CT_VM_VSRC)))
                           << "\n");
                if (emit) {
                    // If the immediately-preceding instruction is a hold_cnt-only
                    // s_wait_alu survivor, fold its hold_cnt into our new wait
                    // so the constraint isn't lost and we don't emit two
                    // adjacent waits.
                    int holdCnt = extractAdjacentHoldCnt(bb, inst);
                    if (holdCnt >= 0)
                        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     fold hold_cnt=" << holdCnt
                                             << " from adjacent survivor\n");
                    emitWaitAlu(bb, inst, wait, holdCnt);
                    PASS_DEBUG(std::cerr << "[InsertWaitAlu]     inserted s_wait_alu before "
                                         << inst->getHwInstDesc()->mnemonic << "\n");
                }
                if (!isNoWait(wait, CT_VA_VDST)) sb.applyWaitcnt(CT_VA_VDST, wait.get(CT_VA_VDST));
                if (!isNoWait(wait, CT_VM_VSRC)) sb.applyWaitcnt(CT_VM_VSRC, wait.get(CT_VM_VSRC));
            }

            if (auto ev = classifyEvent(*inst)) sb.onProducer(*ev, *inst, keyer);

            ++it;
        }

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] end-of-bb \"" << bb.getLabel()
                             << "\" sb=[va_vdst LB=" << sb.getScoreLB(CT_VA_VDST)
                             << " UB=" << sb.getScoreUB(CT_VA_VDST) << " sz=" << sb.scoresSize()
                             << "; vm_vsrc LB=" << sb.getScoreLB(CT_VM_VSRC)
                             << " UB=" << sb.getScoreUB(CT_VM_VSRC) << "]\n");
        return sb;
    }

    // Build "s_setreg_imm32_b32 hwreg(SCHED_MODE, DEP_MODE), value"
    StinkyInstruction* makeSchedModeSetreg(BasicBlock& bb, IRBase* insertBefore, int value) {
        AsmIRBuilder builder(bb, archId);
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::s_setreg_IMM32_b32, archId), insertBefore);
        const HwReg::SubField depMode = HwReg::schedModeDepMode(archId);
        inst->addDestReg(
            StinkyRegister::Hwreg(HwReg::schedModeId(archId), depMode.offset, depMode.size));
        inst->addSrcReg(StinkyRegister(value));
        return inst;
    }

    void insertSchedModeLifecycle(Function& func) {
        BasicBlock* entry = func.getEntryBlock();
        if (!entry) return;

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 3: insert mode2 lifecycle setregs\n");

        // The wave can enter the compute region through two labels: the
        // kernarg-preload path jumps straight to label_Preload_Offset_Start
        // (skipping the +0..255 prologue), while the non-preload path enters at
        // label_ASM_Start (the main-body entry). A kernel may emit either or
        // both. Enable mode2 at EVERY entry label present so whichever path the
        // wave takes hits a setreg(SCHED_MODE)=2 — re-enabling is idempotent and
        // the span between the two labels is SALU kernarg processing (no
        // VALU/VMEM in flight), so a second enable is still drain-free.
        // If no entry label is found, fall back to the function entry block.
        std::vector<BasicBlock*> anchorBBs;
        for (BasicBlock& bb : func) {
            if (bb.getLabel() == "label_Preload_Offset_Start" ||
                bb.getLabel() == "label_ASM_Start") {
                anchorBBs.push_back(&bb);
            }
        }
        if (anchorBBs.empty()) anchorBBs.push_back(entry);

        // Drain-free: each anchor is a kernel entry (all DEPCTR counters zero,
        // SALU kernarg code follows), and mode2->mode0 needs no drain either.
        for (BasicBlock* anchorBB : anchorBBs) {
            // Skip leading labels / pseudo instructions so the setreg lands at
            // the first real instruction position after the label.
            auto anchorIt = anchorBB->begin();
            while (anchorIt != anchorBB->end()) {
                auto* inst = dyn_cast<StinkyInstruction>(anchorIt.getNodePtr());
                if (inst && isPseudoInst(inst)) {
                    ++anchorIt;
                    continue;
                }
                break;
            }
            IRBase* anchor = (anchorIt == anchorBB->end()) ? nullptr : anchorIt.getNodePtr();
            makeSchedModeSetreg(*anchorBB, anchor, /*value=*/2);
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   inserted setreg(SCHED_MODE)=2 at entry "
                                    "bb=\""
                                 << anchorBB->getLabel() << "\"\n");
        }

        // Disable mode2 before every return and every call. We do NOT re-enable
        // after a call: function calls only occur in the mode0 epilogue.
        struct Insertion {
            BasicBlock* bb;
            StinkyInstruction* anchor;
            int value;
            bool insertAfter;
        };
        std::vector<Insertion> work;
        std::unordered_set<BasicBlock*> bbsWithExitDisable;
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst) continue;
                if (isReturn(*inst)) {
                    work.push_back({&bb, inst, /*value=*/0, /*insertAfter=*/false});
                    bbsWithExitDisable.insert(&bb);
                } else if (isCall(*inst)) {
                    work.push_back({&bb, inst, /*value=*/0, /*insertAfter=*/false});
                    bbsWithExitDisable.insert(&bb);
                }
            }
        }

        // A BB containing no real (non-pseudo) instruction is an out-of-region
        // placeholder: CFGBuilder created it as the target of a branch whose real
        // destination lives outside the extracted scope.
        auto hasRealInst = [](BasicBlock& b) {
            for (auto it = b.begin(); it != b.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && !isPseudoInst(inst)) return true;
            }
            return false;
        };
        // A "scope-exit placeholder" is a label-only BB that CFGBuilder created
        // as the target of a branch whose real IR lives OUTSIDE the extracted
        // region: it has no real instructions AND no path back to real in-region
        // content (a leaf, or a chain of label-only BBs that dead-ends). This
        // must NOT match an in-region label-only BB that merely falls through to
        // its real body in the next BB (e.g. label_ActivationSetPCAddrEnd,
        // label_GW_B0_FD0_OptNLL_MB, the To_Activation_* arm targets) — those are
        // not exits and would otherwise collect spurious mode0 disables.
        auto isExitPlaceholder = [&](BasicBlock& start) {
            if (hasRealInst(start)) return false;
            std::unordered_set<BasicBlock*> seen;
            std::vector<BasicBlock*> stack{&start};
            while (!stack.empty()) {
                BasicBlock* b = stack.back();
                stack.pop_back();
                if (!seen.insert(b).second) continue;
                for (BasicBlock* s : b->getSuccessors()) {
                    if (!s) continue;
                    if (hasRealInst(*s)) return false;  // reaches real in-region code
                    stack.push_back(s);
                }
            }
            return true;  // no real in-region content reachable -> true exit
        };

        // Region-exit disables: one mode0 per edge leaving the mode2 region, none
        // on in-region edges. Never disable before a conditional branch (it would
        // clobber the in-region edge); conditional exits converge at the boundary label.
        std::unordered_set<BasicBlock*> coveredAtLabel;
        for (BasicBlock& bb : func) {
            if (bbsWithExitDisable.count(&bb)) continue;
            if (!bb.getSuccessors().empty()) continue;
            StinkyInstruction* tail = lastRealInst(bb);
            if (tail) {
                const bool tailExits =
                    isBranch(*tail) || tail->getUnifiedOpcode() == GFX::s_setpc_b64;
                work.push_back({&bb, tail, /*value=*/0, /*insertAfter=*/!tailExits});
                bbsWithExitDisable.insert(&bb);
                continue;
            }
            // Label-only BB. Skip if unreachable — every predecessor terminates
            // with a return (s_endpgm), e.g. a trailing `label_ASM_End:` after
            // `s_endpgm`, where the disable would be dead code.
            StinkyInstruction* labelAnchor = nullptr;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst && inst->getUnifiedOpcode() == GFX::LABEL) {
                    labelAnchor = inst;
                    break;
                }
            }
            if (labelAnchor) {
                bool reachable = bb.getPredecessors().empty();
                for (BasicBlock* pred : bb.getPredecessors()) {
                    StinkyInstruction* predTail = lastRealInst(*pred);
                    if (!predTail || !isReturn(*predTail)) {
                        reachable = true;
                        break;
                    }
                }
                if (reachable) {
                    work.push_back({&bb, labelAnchor, /*value=*/0, /*insertAfter=*/true});
                    bbsWithExitDisable.insert(&bb);
                    coveredAtLabel.insert(&bb);
                }
            }
        }
        // Pass B — unconditional out-transfer (s_branch / s_setpc) not already
        // covered by Pass A: disable before the terminator. Conditional branches
        // are never disabled here (their exit converges at the label, Pass A).
        for (BasicBlock& bb : func) {
            if (bbsWithExitDisable.count(&bb)) continue;
            StinkyInstruction* tail = lastRealInst(bb);
            if (!tail) continue;
            if (isConditionalBranch(*tail)) continue;
            BasicBlock* exitSucc = nullptr;
            for (BasicBlock* succ : bb.getSuccessors()) {
                if (succ && isExitPlaceholder(*succ)) {
                    exitSucc = succ;
                    break;
                }
            }
            if (!exitSucc) continue;
            if (coveredAtLabel.count(exitSucc)) continue;
            const bool tailTransfers =
                isBranch(*tail) || tail->getUnifiedOpcode() == GFX::s_setpc_b64;
            work.push_back({&bb, tail, /*value=*/0, /*insertAfter=*/!tailTransfers});
            bbsWithExitDisable.insert(&bb);
        }
        for (const auto& w : work) {
            IRBase* insertBefore = nullptr;
            if (w.insertAfter) {
                auto it = IRList::iterator(w.anchor);
                ++it;
                insertBefore = (it == w.bb->end()) ? nullptr : it.getNodePtr();
            } else {
                insertBefore = w.anchor;
            }
            makeSchedModeSetreg(*w.bb, insertBefore, w.value);
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   inserted setreg(SCHED_MODE)=" << w.value
                                 << " " << (w.insertAfter ? "after" : "before") << " "
                                 << w.anchor->getHwInstDesc()->mnemonic << " in bb=\""
                                 << w.bb->getLabel() << "\"\n");
        }
    }

   public:
    static char ID;
    const char* getName() const override {
        return "InsertWaitAluPass";
    }
    Pass::ID getPassID() const override {
        return &InsertWaitAluPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const auto* archInfo = ArchHelper::getInstance().getArchInfo(archId);
        const bool hasD16 = archInfo && archInfo->hasD16Writes32BitVgpr();
        keyer = VGPRHalfKeyer(hasD16);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << " hasD16Writes32BitVgpr=" << hasD16 << "\n");

        if (func.empty()) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu] empty function, nothing to do\n");
            return preserveCFGAnalyses();
        }

        const auto& bbIndex = AM.getResult<BBIndexAnalysis>(func);
        const auto& rpo = bbIndex.rpo;

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 1: fixed-point analysis (" << rpo.size()
                             << " BBs in RPO)\n");

        // Phase 1: fixed-point analysis using entry-state propagation.
        // Each BB starts from its accumulated entry state. After processing,
        // the exit state is merged into each successor's entry state. The
        // merge is monotonically widening, guaranteeing convergence.
        {
            std::vector<BasicBlock*> worklist;
            std::unordered_set<BasicBlock*> inWL;
            for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
                worklist.push_back(*it);
                inWL.insert(*it);
            }
            unsigned visits = 0;
            while (!worklist.empty()) {
                BasicBlock* bb = worklist.back();
                worklist.pop_back();
                inWL.erase(bb);
                ++visits;
                WaitcntBrackets exitState = runOnBasicBlock(*bb, /*emit=*/false);
                for (auto* succ : bb->getSuccessors()) {
                    if (blockEntryState[succ].merge(exitState)) {
                        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   entry widened for bb=\""
                                             << succ->getLabel() << "\" — queueing\n");
                        if (inWL.insert(succ).second) worklist.push_back(succ);
                    }
                }
            }
            PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 1 converged after " << visits
                                 << " BB visits\n");
        }

        // Phase 2: emit s_wait_alu using converged state.
        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 2: emit s_wait_alu instructions\n");
        for (auto* bb : rpo) runOnBasicBlock(*bb, /*emit=*/true);

        // Phase 3: mode2 lifecycle (setreg at entry / before calls+returns /
        // after calls).
        insertSchedModeLifecycle(func);

        blockEntryState.clear();
        return PreservedAnalyses::none();
    }
};

char InsertWaitAluPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertWaitAluPass() {
    return std::make_unique<InsertWaitAluPassImpl>();
}
}  // namespace stinkytofu
