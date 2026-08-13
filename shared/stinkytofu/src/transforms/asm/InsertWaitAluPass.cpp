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
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/RegHalfKeyer.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace {
using namespace stinkytofu;

// Gate for the ESM2 VALU source-operand VA_VDST stamp (the src-operand WAR hazard).
bool g_enableESM2TrackValuVsrc = false;

// TEMP HACK gate. When true, suppress the va_vdst wait for the VGPR-source (RAW)
// hazard of GLOBAL-family memory ops — the "valu writes VGPR, global op reads
// it" case. Only the global op's src RAW va_vdst is dropped; its dst WAW
// va_vdst, the vm_vsrc WAR, and every non-GLOBAL consumer are untouched.
constexpr bool g_enableESM2SuppressValuToGlobalVaVdst = true;

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
    EV_VGPR_VMEM_READ,  // buffer / global / image reading a VGPR source
    EV_NUM,
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

// The VALU pipes the single VA_VDST counter aggregates.
enum VaPipe : uint8_t {
    PIPE_CSMACC = 0,
    PIPE_DPMACC = 1,
    PIPE_TRANS = 2,
    PIPE_XDL = 3,
    NUM_VA_PIPE = 4,
};

// Guard the shared ordinals vaPipeOfEvent's cast relies on.
static_assert(static_cast<int>(EV_VGPR_CSMACC_WRITE) == PIPE_CSMACC);
static_assert(static_cast<int>(EV_VGPR_DPMACC_WRITE) == PIPE_DPMACC);
static_assert(static_cast<int>(EV_VGPR_TRANS_WRITE) == PIPE_TRANS);
static_assert(static_cast<int>(EV_VGPR_XDL_WRITE) == PIPE_XDL);

// Valid only for VA_VDST events.
inline VaPipe vaPipeOfEvent(WaitEventType e) {
    return static_cast<VaPipe>(e);
}

// The two VM_VSRC ordering FIFOs; flat_* enqueues into both.
enum VmFifo : uint8_t {
    FIFO_LDS = 0,
    FIFO_TEX = 1,
    NUM_VM_FIFOS = 2,
};

inline bool enqueuesFifoLds(WaitEventType e) {
    return e == EV_VGPR_LDS_READ || e == EV_VGPR_FLAT_READ;
}
inline bool enqueuesFifoTex(WaitEventType e) {
    return e == EV_VGPR_VMEM_READ || e == EV_VGPR_FLAT_READ;
}

inline const char* vaPipeName(VaPipe p) {
    switch (p) {
        case PIPE_CSMACC:
            return "CSMACC";
        case PIPE_DPMACC:
            return "DPMACC";
        case PIPE_TRANS:
            return "TRANS";
        case PIPE_XDL:
            return "XDL";
        case NUM_VA_PIPE:
            break;
    }
    return "?";
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
    if (isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) || isGLOBALOrAtomic(inst))
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

// Walk `regs`, skipping non-VGPR ones, and invoke fn(vgprIdx, half) for each
// VGPR. halfFn maps an operand's position (VGPR-only) to its True16 half
// selector, so fn may act at half-word (LOW/HIGH) granularity. Callers pass a
// single operand list (getSrcRegs or getDestRegs) — never both, since src and
// dst use different half selectors. Shared by producer stamping and consumer
// probing.
template <typename HalfFn, typename Fn>
inline void forEachVGPR(const std::vector<StinkyRegister>& regs, HalfFn&& halfFn, Fn&& fn) {
    size_t opIdx = 0;
    for (const auto& reg : regs) {
        if (reg.dataType != StinkyRegister::Type::Register) continue;
        if (reg.reg.type != RegType::V) continue;
        HighBitSel half = halfFn(opIdx);
        ++opIdx;
        for (uint16_t off = 0; off < reg.reg.num; ++off) fn(reg.reg.idx + off, half);
    }
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
// WaitcntBrackets — per-pipe/per-FIFO scoreboard with per-VGPR stamps
// ---------------------------------------------------------------------------

struct VgprStamp {
    std::array<unsigned, NUM_VA_PIPE> vaOrd = {};
    unsigned vmOrdLds = 0;
    unsigned vmOrdTex = 0;
    // Both vm ordinals from one flat_*.
    bool pairedFlat = false;
};

class WaitcntBrackets {
   public:
    // Aggregate views.
    unsigned getScoreLB(CounterType c) const {
        return c == CT_VA_VDST ? vaPipeSum(vaPipeLB) : vmLB;
    }
    unsigned getScoreUB(CounterType c) const {
        return c == CT_VA_VDST ? vaPipeSum(vaPipeUB) : vmUB;
    }
    unsigned getScoreRange(CounterType c) const {
        return getScoreUB(c) - getScoreLB(c);
    }
    size_t scoresSize() const {
        return scores.size();
    }

    // Stamp the scoreboard for producer `inst`.
    void onProducer(WaitEventType ev, const StinkyInstruction& inst, const VGPRHalfKeyer& keyer) {
        CounterType ct = counterFromEvent(ev);

        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        if (ct == CT_VA_VDST) {
            VaPipe pipe = vaPipeOfEvent(ev);
            unsigned inc = hasMatrixScalePair(inst) ? 2u : 1u;
            vaPipeUB[pipe] += inc;
            unsigned ord = vaPipeUB[pipe];

            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   stamp event=" << eventName(ev) << " inc="
                                 << inc << " [pipe=" << vaPipeName(pipe) << " ord=" << ord
                                 << " ub=" << vaPipeUB[pipe] << " lb=" << vaPipeLB[pipe] << "]"
                                 << " (mnemonic=" << inst.getHwInstDesc()->mnemonic << ")\n");

            auto stampVA = [&](unsigned idx, HighBitSel half) {
                RegKey k = keyer.producerKey(idx, half);
                VgprStamp& s = scores[k];
                s.vaOrd[pipe] = ord;
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     stamp va v" << k.idx << "("
                                     << halfName(k.half) << ") [pipe=" << vaPipeName(pipe)
                                     << " ord=" << ord << "]\n");
            };
            if (g_enableESM2TrackValuVsrc)
                forEachVGPR(
                    inst.getSrcRegs(), [&](size_t i) { return srcHalfSel(true16Mod, i); },
                    [&](unsigned idx, HighBitSel half) { stampVA(idx, half); });
            forEachVGPR(
                inst.getDestRegs(), [&](size_t i) { return destHalfSel(true16Mod, i); },
                [&](unsigned idx, HighBitSel half) { stampVA(idx, half); });
            return;
        }

        // VM_VSRC. flat_* bumps both FIFOs.
        ++vmUB;
        unsigned ordLds = 0, ordTex = 0;
        if (enqueuesFifoLds(ev)) ordLds = ++vmFifoUB[FIFO_LDS];
        if (enqueuesFifoTex(ev)) ordTex = ++vmFifoUB[FIFO_TEX];

        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   stamp vm event=" << eventName(ev)
                             << " [vm ub=" << vmUB << " lb=" << vmLB << "]"
                             << " [LDS ord=" << ordLds << " ub=" << vmFifoUB[FIFO_LDS]
                             << " lb=" << vmFifoLB[FIFO_LDS] << "]" << " [TEX ord=" << ordTex
                             << " ub=" << vmFifoUB[FIFO_TEX] << " lb=" << vmFifoLB[FIFO_TEX] << "]"
                             << " (mnemonic=" << inst.getHwInstDesc()->mnemonic << ")\n");

        auto stampVM = [&](unsigned idx, HighBitSel half) {
            RegKey k = keyer.producerKey(idx, half);
            VgprStamp& s = scores[k];
            // Set only the FIFO(s) this op enqueues into.
            if (enqueuesFifoLds(ev)) s.vmOrdLds = ordLds;
            if (enqueuesFifoTex(ev)) s.vmOrdTex = ordTex;
            // Paired when both ordinals came from this one flat_*.
            s.pairedFlat = enqueuesFifoLds(ev) && enqueuesFifoTex(ev);
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     stamp vm v" << k.idx << "("
                                 << halfName(k.half) << ") [LDS ord=" << s.vmOrdLds << "]"
                                 << " [TEX ord=" << s.vmOrdTex << " paired=" << s.pairedFlat
                                 << "]\n");
        };
        // VM_VSRC tracks in-flight VMEM reads, which are always full DWORD.
        forEachVGPR(
            inst.getSrcRegs(), [](size_t) { return HighBitSel::NONE; },
            [&](unsigned idx, HighBitSel half) { stampVM(idx, half); });
    }

    // Probe each VGPR operand for hazards and accumulate the wait.
    void onConsumer(const StinkyInstruction& inst, const VGPRHalfKeyer& keyer, Wait& wait) const {
        const True16Modifiers* true16Mod = inst.getModifier<True16Modifiers>();

        // TEMP HACK: optionally drop the src RAW va_vdst for valu->global.
        const bool suppressSrcVaVdst =
            g_enableESM2SuppressValuToGlobalVaVdst && isGLOBALOrAtomic(inst);
        if (!suppressSrcVaVdst) {
            forEachVGPR(
                inst.getSrcRegs(), [&](size_t i) { return srcHalfSel(true16Mod, i); },
                [&](unsigned idx, HighBitSel half) {
                    keyer.forEachConsumerKey(idx, half, [&](RegKey k) {
                        determineWait(CT_VA_VDST, k, wait, "src(RAW)");
                    });
                });
        }

        forEachVGPR(
            inst.getDestRegs(), [&](size_t i) { return destHalfSel(true16Mod, i); },
            [&](unsigned idx, HighBitSel half) {
                keyer.forEachConsumerKey(
                    idx, half, [&](RegKey k) { determineWait(CT_VA_VDST, k, wait, "dst(WAW)"); });
                // WAR on VM_VSRC: writer-vs-in-flight-VMEM-read uses full DWORD.
                RegKey full{RegType::V, idx, RegHalf::NONE};
                determineWait(CT_VM_VSRC, full, wait, "dst(WAR)");
            });
    }

    // Wait needed for this reg's VM readers.
    unsigned vmFollowers(const VgprStamp& s) const {
        bool liveLds = s.vmOrdLds && s.vmOrdLds > vmFifoLB[FIFO_LDS];
        bool liveTex = s.vmOrdTex && s.vmOrdTex > vmFifoLB[FIFO_TEX];
        unsigned fLds = liveLds ? vmFifoUB[FIFO_LDS] - s.vmOrdLds : 0u;
        unsigned fTex = liveTex ? vmFifoUB[FIFO_TEX] - s.vmOrdTex : 0u;
        // One flat_* retires from both FIFOs at once, so either proves it done.
        if (s.pairedFlat && liveLds && liveTex) return std::max(fLds, fTex);
        // Two distinct producers: must wait for both.
        unsigned f = ~0u;
        if (liveLds) f = std::min(f, fLds);
        if (liveTex) f = std::min(f, fTex);
        return f;
    }

    // Wait needed for this reg's VA producers.
    unsigned vaFollowers(const VgprStamp& s) const {
        unsigned f = ~0u;
        for (int p = 0; p < NUM_VA_PIPE; ++p) {
            if (s.vaOrd[p] && s.vaOrd[p] > vaPipeLB[p]) f = std::min(f, vaPipeUB[p] - s.vaOrd[p]);
        }
        return f;
    }

    // Accumulate the wait for the hazard on VGPR `k` against counter `c`.
    void determineWait(CounterType c, const RegKey& k, Wait& wait, const char* role) const {
        auto it = scores.find(k);
        if (it == scores.end()) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait " << counterName(c) << " on v"
                                 << k.idx << "(" << halfName(k.half) << "," << role
                                 << ") [no stamp]\n");
            return;
        }
        const VgprStamp& s = it->second;

        if (c == CT_VM_VSRC) {
            // Wait for live FIFO producers.
            bool liveLds = s.vmOrdLds && s.vmOrdLds > vmFifoLB[FIFO_LDS];
            bool liveTex = s.vmOrdTex && s.vmOrdTex > vmFifoLB[FIFO_TEX];
            if (!liveLds && !liveTex) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait vm_vsrc on v" << k.idx << "("
                                     << halfName(k.half) << "," << role << ")" << vmStateStr(&s)
                                     << " → drained (no wait)\n");
                return;
            }
            unsigned f = vmFollowers(s);
            unsigned chosen = (f > 0) ? std::min(f, maxEmittableWait(c)) : 0u;
            addWait(wait, c, chosen);
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     wait hit vm_vsrc on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ")" << vmStateStr(&s)
                                 << " f=" << f << " → wait=" << chosen << "\n");
            return;
        }

        // Wait for live pipe producers.
        unsigned f = vaFollowers(s);
        if (f == ~0u) {  // no live producer in any pipe
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait va_vdst on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ")" << vaStateStr(&s)
                                 << " → no live producer\n");
            return;
        }
        unsigned chosen = std::min(f, maxEmittableWait(c));
        addWait(wait, c, chosen);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     wait hit " << counterName(c) << " on v"
                             << k.idx << "(" << halfName(k.half) << "," << role << ")"
                             << vaStateStr(&s) << " f=" << f << " → wait=" << chosen << "\n");
    }

    void applyWaitcnt(CounterType c, unsigned count) {
        if (count == kNoWait) return;
        if (c == CT_VA_VDST) {
            // count bounds every pipe.
            for (int P = 0; P < NUM_VA_PIPE; ++P) {
                unsigned oldLB = vaPipeLB[P];
                unsigned newLB = vaPipeUB[P] >= count ? vaPipeUB[P] - count : 0u;
                if (newLB > vaPipeLB[P]) vaPipeLB[P] = newLB;
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     apply va_vdst(" << count << ") [pipe="
                                     << vaPipeName(static_cast<VaPipe>(P)) << " lb " << oldLB << "→"
                                     << vaPipeLB[P] << " ub=" << vaPipeUB[P] << "]\n");
            }
        } else {
            // count bounds the aggregate and each FIFO.
            unsigned newVmLB = vmUB >= count ? vmUB - count : 0u;
            if (newVmLB > vmLB) vmLB = newVmLB;
            for (int g = 0; g < NUM_VM_FIFOS; ++g) {
                unsigned newLB = vmFifoUB[g] >= count ? vmFifoUB[g] - count : 0u;
                if (newLB > vmFifoLB[g]) vmFifoLB[g] = newLB;
            }
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     apply vm_vsrc(" << count << ") [vm lb→"
                                 << vmLB << " ub=" << vmUB << "]" << " [LDS lb="
                                 << vmFifoLB[FIFO_LDS] << " ub=" << vmFifoUB[FIFO_LDS] << "]"
                                 << " [TEX lb=" << vmFifoLB[FIFO_TEX]
                                 << " ub=" << vmFifoUB[FIFO_TEX] << "]\n");
        }
    }

    // Widen this entry state with a predecessor's exit. Returns true (strictDom)
    // when the other side contributed a tighter score.
    bool merge(const WaitcntBrackets& other) {
        bool strictDom = false;
        std::array<unsigned, NUM_VA_PIPE> myShift{}, otherShift{}, myOldFloor{}, otherOldFloor{};

        for (int P = 0; P < NUM_VA_PIPE; ++P) {
            unsigned mineIF = vaPipeUB[P] - vaPipeLB[P];
            unsigned otherIF = other.vaPipeUB[P] - other.vaPipeLB[P];
            unsigned newUB = vaPipeLB[P] + std::max(mineIF, otherIF);
            myOldFloor[P] = vaPipeLB[P];
            otherOldFloor[P] = other.vaPipeLB[P];
            myShift[P] = newUB - vaPipeUB[P];
            otherShift[P] = newUB - other.vaPipeUB[P];
            vaPipeUB[P] = newUB;
        }

        {
            unsigned mineIF = vmUB - vmLB;
            unsigned otherIF = other.vmUB - other.vmLB;
            vmUB = vmLB + std::max(mineIF, otherIF);
        }

        std::array<unsigned, NUM_VM_FIFOS> fMyShift{}, fOtherShift{}, fMyOldFloor{},
            fOtherOldFloor{};
        for (int g = 0; g < NUM_VM_FIFOS; ++g) {
            unsigned mineIF = vmFifoUB[g] - vmFifoLB[g];
            unsigned otherIF = other.vmFifoUB[g] - other.vmFifoLB[g];
            unsigned newUB = vmFifoLB[g] + std::max(mineIF, otherIF);
            fMyOldFloor[g] = vmFifoLB[g];
            fOtherOldFloor[g] = other.vmFifoLB[g];
            fMyShift[g] = newUB - vmFifoUB[g];
            fOtherShift[g] = newUB - other.vmFifoUB[g];
            vmFifoUB[g] = newUB;
        }

        for (const auto& [k, _] : other.scores) scores.try_emplace(k);

        for (auto& [k, s] : scores) {
            auto it = other.scores.find(k);
            const VgprStamp* o = (it != other.scores.end()) ? &it->second : nullptr;
            // Merge each pipe's ordinal independently.
            for (int p = 0; p < NUM_VA_PIPE; ++p) {
                mergeSlotOrd(s.vaOrd[p], o ? o->vaOrd[p] : 0, myShift[p], otherShift[p],
                             myOldFloor[p], otherOldFloor[p], strictDom);
            }
            mergeSlotOrd(s.vmOrdLds, o ? o->vmOrdLds : 0, fMyShift[FIFO_LDS], fOtherShift[FIFO_LDS],
                         fMyOldFloor[FIFO_LDS], fOtherOldFloor[FIFO_LDS], strictDom);
            mergeSlotOrd(s.vmOrdTex, o ? o->vmOrdTex : 0, fMyShift[FIFO_TEX], fOtherShift[FIFO_TEX],
                         fMyOldFloor[FIFO_TEX], fOtherOldFloor[FIFO_TEX], strictDom);
            // Paired survives the join only if both paths agree.
            s.pairedFlat = s.pairedFlat && o && o->pairedFlat;
        }

        return strictDom;
    }

   public:
    // Debug-only VA snapshot.
    std::string vaStateStr(const VgprStamp* s) const {
        std::string out;
        for (int p = 0; p < NUM_VA_PIPE; ++p) {
            out += " [";
            out += vaPipeName(static_cast<VaPipe>(p));
            out += " ub=" + std::to_string(vaPipeUB[p]) + " lb=" + std::to_string(vaPipeLB[p]);
            if (s) {
                unsigned ord = s->vaOrd[p];
                bool live = ord && ord > vaPipeLB[p];
                out += " ord=" + std::to_string(ord) + " live=" + std::to_string(live);
                if (live) out += " f=" + std::to_string(vaPipeUB[p] - ord);
            }
            out += "]";
        }
        return out;
    }

    // Debug-only VM snapshot.
    std::string vmStateStr(const VgprStamp* s) const {
        std::string out = " [vm ub=" + std::to_string(vmUB) + " lb=" + std::to_string(vmLB) + "]";
        static const char* fifoName[NUM_VM_FIFOS] = {"LDS", "TEX"};
        for (int g = 0; g < NUM_VM_FIFOS; ++g) {
            out += " [";
            out += fifoName[g];
            out += " ub=" + std::to_string(vmFifoUB[g]) + " lb=" + std::to_string(vmFifoLB[g]);
            if (s) {
                unsigned ord = (g == FIFO_LDS) ? s->vmOrdLds : s->vmOrdTex;
                bool live = ord && ord > vmFifoLB[g];
                out += " ord=" + std::to_string(ord) + " live=" + std::to_string(live);
                if (live) out += " f=" + std::to_string(vmFifoUB[g] - ord);
            }
            out += "]";
        }
        if (s) out += " paired=" + std::to_string(s->pairedFlat);
        return out;
    }

   private:
    static unsigned vaPipeSum(const std::array<unsigned, NUM_VA_PIPE>& a) {
        unsigned n = 0;
        for (int P = 0; P < NUM_VA_PIPE; ++P) n += a[P];
        return n;
    }

    // Shift both into the widened frame, keep the later.
    static void mergeSlotOrd(unsigned& myOrd, unsigned oOrd, unsigned myShift, unsigned otherShift,
                             unsigned myOldFloor, unsigned otherOldFloor, bool& strictDom) {
        unsigned myS = (myOrd && myOrd > myOldFloor) ? myOrd + myShift : 0;
        unsigned oS = (oOrd && oOrd > otherOldFloor) ? oOrd + otherShift : 0;
        if (oS > myS) {
            myOrd = oS;
            strictDom = true;
        } else {
            myOrd = myS;
        }
    }

    // VA_VDST per-pipe UB/LB.
    std::array<unsigned, NUM_VA_PIPE> vaPipeUB = {};
    std::array<unsigned, NUM_VA_PIPE> vaPipeLB = {};
    // VM_VSRC aggregate UB/LB.
    unsigned vmUB = 0;
    unsigned vmLB = 0;
    // VM_VSRC per-FIFO UB/LB.
    std::array<unsigned, NUM_VM_FIFOS> vmFifoUB = {};
    std::array<unsigned, NUM_VM_FIFOS> vmFifoLB = {};
    std::unordered_map<RegKey, VgprStamp, RegKeyHash> scores;
};

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

class InsertWaitAluPassImpl : public Pass {
    std::unordered_map<BasicBlock*, WaitcntBrackets> blockEntryState;
    GfxArchID archId = GfxArchID{};
    VGPRHalfKeyer keyer{};

   public:
    explicit InsertWaitAluPassImpl(bool enableESM2TrackValuVsrc) {
        g_enableESM2TrackValuVsrc = enableESM2TrackValuVsrc;
    }

   private:
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
                             << bb.getLabel() << "\" entry sz=" << sb.scoresSize() << " va:"
                             << sb.vaStateStr(nullptr) << " vm:" << sb.vmStateStr(nullptr) << "\n");

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

            // Function call (s_swappc): drain both counters right after the call,
            // at the return-landing site. The callee may leave VALU/VMEM
            // instructions outstanding on VA_VDST/VM_VSRC, so the drain is
            // unconditional. The callee entry is drained separately
            // (insertCalleeEntryDrain).
            if (isCall(*inst)) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]   call — drain va_vdst(0)+vm_vsrc(0) "
                                        "after s_swappc (callee->caller bracket)\n");
                // nextIt is the instruction after the call. The drain is inserted
                // before it, so resuming at nextIt continues past the drain
                // instead of re-visiting it.
                auto nextIt = it;
                ++nextIt;
                if (emit) {
                    Wait drain;
                    addWait(drain, CT_VA_VDST, 0);
                    addWait(drain, CT_VM_VSRC, 0);
                    // Insert before the node after the call (append at BB end if
                    // the call is the last node) so the drain lands in the
                    // caller's own BB, bound to the return path.
                    IRBase* insertBefore = (nextIt == bb.end()) ? nullptr : nextIt.getNodePtr();
                    emitWaitAlu(bb, insertBefore, drain);
                }
                sb.applyWaitcnt(CT_VA_VDST, 0);
                sb.applyWaitcnt(CT_VM_VSRC, 0);
                it = nextIt;
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
                             << "\" sz=" << sb.scoresSize() << " va:" << sb.vaStateStr(nullptr)
                             << " vm:" << sb.vmStateStr(nullptr) << "\n");
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

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 3: insert mode2 enable setreg\n");

        // Whole-kernel mode2: enable at the kernel entry label(s). Mode2 stays
        // active across function calls and across the whole kernel body — it is
        // never switched back to mode0.

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
        // SALU kernarg code follows).
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
    }

   public:
    static char ID;
    const char* getName() const override {
        return "InsertWaitAluPass";
    }
    Pass::ID getPassID() const override {
        return &InsertWaitAluPassImpl::ID;
    }

   private:
    // Drain both counters at the callee entry to establish the zero DEPCTR start
    // the analysis assumes. Lands before the first real instruction, so it stays
    // after the entry label whether the callee is flat or split by CFGBuilder.
    void insertCalleeEntryDrain(Function& callee) {
        for (BasicBlock& bb : callee) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;

                Wait drain;
                addWait(drain, CT_VA_VDST, 0);
                addWait(drain, CT_VM_VSRC, 0);
                emitWaitAlu(bb, it.getNodePtr(), drain);
                PASS_DEBUG(std::cerr << "[InsertWaitAlu] callee \"" << callee.getName()
                                     << "\": entry drain va_vdst(0)+vm_vsrc(0)\n");
                return;
            }
        }
    }

    // True if any real instruction in func reads or writes a VGPR.
    static bool functionReadsOrWritesVGPR(Function& func) {
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) continue;
                for (const auto& r : inst->getSrcRegs())
                    if (r.dataType == StinkyRegister::Type::Register && r.reg.type == RegType::V)
                        return true;
                for (const auto& r : inst->getDestRegs())
                    if (r.dataType == StinkyRegister::Type::Register && r.reg.type == RegType::V)
                        return true;
            }
        }
        return false;
    }

    // Full fixed-point scoreboard analysis for one function. A callee is first
    // drained at entry to establish the zero start the analysis assumes.
    void runFullAnalysis(Function& func, AnalysisManager& AM, bool isCallee) {
        if (isCallee) {
            // A callee that never touches a VGPR (e.g. the "None" activation) has
            // no hazard to drain or analyze.
            if (!functionReadsOrWritesVGPR(func)) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu] callee \"" << func.getName()
                                     << "\": no VGPR use, skip\n");
                return;
            }
            insertCalleeEntryDrain(func);
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

        // Phase 2: emit s_wait_alu using converged state. Caller->callee bracket
        // drains are emitted here, in the isCall branch of runOnBasicBlock.
        PASS_DEBUG(std::cerr << "[InsertWaitAlu] Phase 2: emit s_wait_alu instructions\n");
        for (auto* bb : rpo) runOnBasicBlock(*bb, /*emit=*/true);

        // Phase 3: enable mode2 at entry label (never disabled thereafter).
        // Entry-only: mode2 is kernel-global, callees must not re-enable it.
        if (!isCallee) insertSchedModeLifecycle(func);

        blockEntryState.clear();
    }

    // Per-arch setup shared by every function run. Idempotent.
    void setupArch(PassContext& passCtx) {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const auto* archInfo = ArchHelper::getInstance().getArchInfo(archId);
        const bool hasD16 = archInfo && archInfo->hasD16Writes32BitVgpr();
        keyer = VGPRHalfKeyer(hasD16);
        PASS_DEBUG(std::cerr << "[InsertWaitAlu] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << " hasD16Writes32BitVgpr=" << hasD16 << "\n");
    }

   public:
    // Entry and callees both get full analysis; callees also get the entry drain.
    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        setupArch(passCtx);
        if (!func.empty()) runFullAnalysis(func, AM, /*isCallee=*/func.getIsCallable());
        return PreservedAnalyses::none();
    }

    // Full per-function analysis, exposed for the ModulePass driver.
    void runOnFunction(Function& func, PassContext& passCtx, AnalysisManager& AM, bool isCallee) {
        setupArch(passCtx);
        if (!func.empty()) runFullAnalysis(func, AM, isCallee);
    }
};

char InsertWaitAluPassImpl::ID = 0;

// Whole-kernel driver: full per-function analysis on entry and every callee. A
// real ModulePass so it owns cross-function iteration and reserves the seam for
// future callee<->caller analysis.
class InsertWaitAluModulePass : public ModulePass {
   public:
    explicit InsertWaitAluModulePass(bool enableESM2TrackValuVsrc)
        : enableESM2TrackValuVsrc(enableESM2TrackValuVsrc) {}

    const char* getName() const override {
        return "InsertWaitAluModulePass";
    }

    PreservedAnalyses run(StinkyAsmModule& M, PassContext& passCtx,
                          ModuleAnalysisManager& /*MAM*/) override {
        InsertWaitAluPassImpl impl(enableESM2TrackValuVsrc);
        AnalysisManager AM;
        registerAllAnalyses(AM);

        // Uniform per-function flow. The AM caches results by analysis type, not
        // by Function, so it must be cleared before each function's analysis.
        for (Function* fn : M.getFunctions()) {
            if (!fn || fn->empty()) continue;
            AM.clear();
            impl.runOnFunction(*fn, passCtx, AM, /*isCallee=*/fn->getIsCallable());
        }

        // Future: callee<->caller cross-function analysis. The whole module is
        // available here when a more aggressive policy is needed.

        return PreservedAnalyses::none();
    }

   private:
    bool enableESM2TrackValuVsrc;
};

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertWaitAluPass(bool enableESM2TrackValuVsrc) {
    return std::make_unique<InsertWaitAluPassImpl>(enableESM2TrackValuVsrc);
}
std::unique_ptr<ModulePass> createInsertWaitAluModulePass(bool enableESM2TrackValuVsrc) {
    return std::make_unique<InsertWaitAluModulePass>(enableESM2TrackValuVsrc);
}
}  // namespace stinkytofu
