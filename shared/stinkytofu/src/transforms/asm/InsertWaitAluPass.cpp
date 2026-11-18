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
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define DEBUG_TYPE "InsertWaitAluPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/OperandHalves.hpp"
#include "stinkytofu/ir/asm/RegHalfKeyer.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace {
using namespace stinkytofu;

// One pass instance's configuration: its options plus the arch's hide model, set per run.
struct WaitAluContext {
    InsertWaitAluOptions opts;
    const HWModel::WaitHide* waitHide = nullptr;
    // Saturation point for every stamp age, so the ages stay finite and the analysis converges.
    unsigned ageCap = 0;
};

// TEMP HACK gate. When true, suppress the va_vdst wait for the VGPR-source (RAW)
// hazard of GLOBAL-family memory ops and global_prefetch — the "valu writes VGPR,
// global op / prefetch reads it" case. global_prefetch does not carry IF_GLOBALLoad,
// so it is classified separately via isGlobalPrefetch(). Only the op's src RAW
// va_vdst is dropped (prefetch has no dst); the vm_vsrc WAR and every non-GLOBAL
// consumer are untouched. Safe because the DAG already spaces the vaddr producer
// >=32 cycles ahead (CDNA5 isVmemAddrHazardConsumer covers buffer loads and prefetch).
// Stores and atomics are not covered by that spacing guarantee, so their data
// operand still needs the real wait.
constexpr bool g_enableESM2SuppressValuToGlobalVaVdst = false;

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

// Render one WaitHide count for the debug banner; 0 reads as off.
inline std::string waitHideStr(int v) {
    return v > 0 ? std::to_string(v) : std::string("0(off)");
}

using Hide = HWModel::WaitHide::Hide;
using RoleHide = HWModel::WaitHide::RoleHide;
using RuleSet = HWModel::WaitHide::RuleSet;
using HideForm = HWModel::WaitHide::Form;

// A stamp's role: the producer wrote the VGPR (dst) or read it (src).
enum Role : uint8_t {
    ROLE_DST = 0,
    ROLE_SRC = 1,
    NUM_ROLES = 2,
};

inline const char* roleName(Role r) {
    return r == ROLE_DST ? "dst" : "src";
}

// Producer classes whose mixed rule carries its own "VALUs after the lead" counter.
enum HideClass : uint8_t {
    CLASS_XDL = 0,
    CLASS_CSMACC = 1,
    CLASS_DPMACC = 2,
    NUM_HIDE_CLASSES = 3,
};

inline HideClass hideClassOfPipe(VaPipe p) {
    switch (p) {
        case PIPE_XDL:
            return CLASS_XDL;
        case PIPE_DPMACC:
            return CLASS_DPMACC;
        default:
            return CLASS_CSMACC;
    }
}

inline VaPipe pipeOfHideClass(HideClass c) {
    switch (c) {
        case CLASS_XDL:
            return PIPE_XDL;
        case CLASS_DPMACC:
            return PIPE_DPMACC;
        default:
            return PIPE_CSMACC;
    }
}

inline const RoleHide& roleRules(const RuleSet& rs, Role r) {
    return (r == ROLE_SRC && rs.src.any()) ? rs.src : rs.dst;
}

// Row matching a form's cost latency, destination width and scaling, or nullptr if none. A
// scaled op with no scaled row takes the unscaled one.
inline const HideForm* waitHideForm(const HWModel::WaitHide& wh, int costLatency, int dstVgprs,
                                    bool scaled) {
    const HideForm* unscaled = nullptr;
    for (const auto& form : wh.forms) {
        if (form.costLatency != costLatency || form.dstVgprs != dstVgprs) continue;
        if (form.scaled == scaled) return &form;
        if (!form.scaled) unscaled = &form;
    }
    return scaled ? unscaled : nullptr;
}

// Visit every rule set of the table.
template <typename Fn>
inline void forEachRuleSet(const HWModel::WaitHide& wh, Fn&& fn) {
    for (const auto& form : wh.forms) {
        fn(form.xdl);
        fn(form.csmacc);
    }
    fn(wh.noLatchCsmacc);
    fn(wh.dpmacc);
}

template <typename Fn>
inline void forEachHide(const RoleHide& rh, Fn&& fn) {
    fn(rh.wmma);
    fn(rh.mix);
    fn(rh.valu);
}

// Saturation point of every age: past it every rule is met, so the analysis converges.
inline unsigned computeAgeCap(const HWModel::WaitHide& wh) {
    int maxHide = 0;
    forEachRuleSet(wh, [&](const RuleSet& rs) {
        for (const RoleHide* rh : {&rs.dst, &rs.src})
            forEachHide(*rh, [&](const Hide& h) { maxHide = std::max(maxHide, h.lead + h.tail); });
    });
    return 2 * static_cast<unsigned>(maxHide);
}

// Pipes whose producers may take their follower count from the shared VA order.
inline bool isCountablePipe(VaPipe p) {
    return p == PIPE_CSMACC || p == PIPE_DPMACC || p == PIPE_TRANS || p == PIPE_XDL;
}

// Pipes that can anchor a matrix producer's follower count.
inline bool isAnchorPipe(VaPipe p) {
    return p == PIPE_XDL || p == PIPE_TRANS;
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
    if (isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst) || isFLATPrefetch(inst))
        return EV_VGPR_FLAT_READ;
    // TEX path. Stinkytofu does not yet flag scratch / image / sample / BVH
    // instructions; on archs that emit them they belong in this same bucket.
    if (isVmemTex(inst)) return EV_VGPR_VMEM_READ;
    return std::nullopt;
}

// VOP3PX2 / VOP3PX3 are software-only encodings of a back-to-back VOP3P pair
// (LD_SCALE + WMMA). Hardware decodes each as two separate VOP3P sub-issues,
// both bumping VA_VDST, so software must count 2.
inline bool hasMatrixScalePair(const StinkyInstruction& inst) {
    auto mc = inst.getHwInstDesc()->microcode;
    return mc == MicrocodeFormat::MC_VOP3PX2 || mc == MicrocodeFormat::MC_VOP3PX3;
}

inline int wmmaDstVgprs(const StinkyInstruction& inst) {
    for (const auto& r : inst.getDestRegs())
        if (r.dataType == StinkyRegister::Type::Register && r.reg.type == RegType::V)
            return static_cast<int>(r.reg.num);
    return 0;
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

// One hazard role of a VGPR: its newest producer per pipe and the AgeClock readings taken when
// it stamped. The rules compare ages, which survive a merge.
struct RoleStamp {
    static constexpr unsigned kNotMet = ~0u;
    std::array<unsigned, NUM_VA_PIPE> ord = {};
    // This producer's ticket in the shared VA order.
    unsigned ordShared = 0;
    // Shared ticket just before the nearest anchor issued after this producer, or 0 until one has.
    unsigned nextAnchorShared = 0;
    // AgeClock readings when this producer stamped.
    unsigned matrixAt = 0;
    unsigned valuAt = 0;
    // VALU reading when the class's mixed-rule lead was met; set only by a join, otherwise the
    // clock's ring answers (valuAfterLead).
    std::array<unsigned, NUM_HIDE_CLASSES> valuAtLead = {kNotMet, kNotMet, kNotMet};

    // Whether the stamp names a producer at all, live or retired.
    bool hasProducer() const {
        for (int p = 0; p < NUM_VA_PIPE; ++p)
            if (ord[p] != 0) return true;
        return false;
    }
};

struct VgprStamp {
    std::array<RoleStamp, NUM_ROLES> role = {};
    unsigned vmOrdLds = 0;
    unsigned vmOrdTex = 0;
    // Both vm ordinals from one flat_*.
    bool pairedFlat = false;
    // Oldest op holding a ticket in both classes that issued after this producer; 0 when none.
    unsigned anchorLds = 0;
    unsigned anchorTex = 0;
};

// The counters ages are measured against; the ring keeps the VALU reading per matrix ordinal.
struct AgeClock {
    static constexpr unsigned kRing = 64;
    unsigned matrix = 0;
    unsigned valu = 0;
    std::array<unsigned, kRing> valuAtMatrix = {};

    void tickMatrix(unsigned inc) {
        for (unsigned u = matrix + 1; u <= matrix + inc; ++u) valuAtMatrix[u % kRing] = valu;
        matrix += inc;
    }
};

enum class HideKind { Wmma, Mix, Valu };

inline const Hide& hideOf(const RoleHide& rh, HideKind k) {
    switch (k) {
        case HideKind::Wmma:
            return rh.wmma;
        case HideKind::Mix:
            return rh.mix;
        default:
            return rh.valu;
    }
}

// A producer that still forces a consumer's va_vdst wait; collected for the tracker's bounds.
struct Blocker {
    const RoleStamp* rs;
    Role r;
    VaPipe p;
};
using Blockers = std::vector<Blocker>;

class WaitcntBrackets {
   public:
    explicit WaitcntBrackets(const WaitAluContext& ctx) : ctx(&ctx) {}

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

    // Whether `form` at step `inc` is the form already latched.
    bool sameXdlForm(unsigned inc, const HideForm* form) const {
        return inc == xdlInc && form == form_;
    }

    // Latch this matrix op's form, or note a second form so the form rules switch off.
    void latchXdlForm(const StinkyInstruction& inst, unsigned inc) {
        const HideForm* form = ctx->waitHide == nullptr
                                   ? nullptr
                                   : waitHideForm(*ctx->waitHide, inst.latencyCycles,
                                                  wmmaDstVgprs(inst), hasMatrixScalePair(inst));
        if (!xdlIncSeen) {
            // One kernel issues one form, so the first op's row stands for the kernel.
            xdlInc = inc;
            form_ = form;
            xdlIncSeen = true;
        } else if (!sameXdlForm(inc, form)) {
            // A second form: disable the form rules rather than pick one.
            xdlFormMixed = true;
        }
    }

    // The rules in force for producers in pipe `p`; nullptr = always wait. A CSMACC producer
    // takes the no-latch rules until a form is latched, a TRANS producer always.
    const RuleSet* rulesFor(VaPipe p) const {
        if (ctx->waitHide == nullptr) return nullptr;
        const bool formRules = xdlIncSeen && !xdlFormMixed && form_ != nullptr;
        switch (p) {
            case PIPE_XDL:
                return formRules ? &form_->xdl : nullptr;
            case PIPE_CSMACC:
                return formRules ? &form_->csmacc : &ctx->waitHide->noLatchCsmacc;
            case PIPE_DPMACC:
                return &ctx->waitHide->dpmacc;
            case PIPE_TRANS:
                return &ctx->waitHide->noLatchCsmacc;
            default:
                return nullptr;
        }
    }

    // Ages of a stamp, each saturating at the age cap so the analysis converges.
    unsigned cap(unsigned age) const {
        return std::min(ctx->ageCap, age);
    }
    unsigned matrixSince(const RoleStamp& rs) const {
        return cap(clock.matrix - rs.matrixAt);
    }
    unsigned valuSince(const RoleStamp& rs) const {
        return cap(clock.valu - rs.valuAt);
    }
    // VALUs issued after the lead (`leadUnits` matrix ordinals) was met; 0 before that. Past
    // the ring the cap stands in: by then every rule with a matrix count is met anyway.
    unsigned valuAfterLead(const RoleStamp& rs, HideClass c, unsigned leadUnits) const {
        if (leadUnits == 0) return valuSince(rs);
        const unsigned leadOrd = rs.matrixAt + leadUnits;
        if (clock.matrix < leadOrd) return 0;
        if (rs.valuAtLead[c] != RoleStamp::kNotMet) return cap(clock.valu - rs.valuAtLead[c]);
        if (clock.matrix - leadOrd < AgeClock::kRing)
            return cap(clock.valu - clock.valuAtMatrix[leadOrd % AgeClock::kRing]);
        return ctx->ageCap;
    }

    // Tail slots the VALUs issued after this producer fill for rule `t` of kind `k`: a mixed
    // rule counts them only after its lead, a no-order rule from the producer.
    unsigned tailCount(const RoleStamp& rs, HideKind k, const Hide& t, HideClass c) const {
        switch (k) {
            case HideKind::Wmma:
                return 0;
            case HideKind::Mix:
                return valuAfterLead(rs, c, static_cast<unsigned>(t.lead) * xdlInc);
            default:
                return valuSince(rs);
        }
    }

    // `t` is met: the lead matrix ops issued, then matrix ops (`slots` each) and VALUs fill the
    // tail; a VALU-only rule counts VALUs alone. Monotone in every age.
    bool hideSatisfied(const RoleStamp& rs, HideKind k, const Hide& t, HideClass c,
                       int slots) const {
        if (!t.on()) return false;
        const unsigned leadOrd = static_cast<unsigned>(t.lead) * xdlInc;
        const unsigned ms = matrixSince(rs);
        if (ms < leadOrd) return false;
        const unsigned matrixSlots =
            k == HideKind::Valu
                ? 0
                : (ms - leadOrd) / xdlInc * static_cast<unsigned>(std::max(slots, 0));
        return matrixSlots + tailCount(rs, k, t, c) >= static_cast<unsigned>(t.tail);
    }

    // Any rule of the role is met.
    bool roleHidden(const RoleStamp& rs, const RoleHide& rh, HideClass c, int slots) const {
        for (HideKind k : {HideKind::Wmma, HideKind::Mix, HideKind::Valu})
            if (hideSatisfied(rs, k, hideOf(rh, k), c, slots)) return true;
        return false;
    }

    // Fewest `op`s until some rule of `rh` hides `rs`; -1 if none ever does. Bounds
    // countUntilFree's scan; hideSatisfied decides.
    int stepsUntilHidden(const RoleStamp& rs, const RoleHide& rh, HideClass c, int slots,
                         WaitAluHideOp op) const {
        const bool wmma = op == WaitAluHideOp::Wmma;
        const unsigned step = xdlInc;
        const unsigned fill = static_cast<unsigned>(std::max(slots, 0));
        // Tail slots one op fills: a matrix op `fill`, a VALU one.
        const unsigned perStep = wmma ? fill : 1u;
        const unsigned ms = matrixSince(rs);
        int best = -1;
        auto consider = [&](int n) {
            if (n >= 0 && (best < 0 || n < best)) best = n;
        };
        auto ceilDiv = [](unsigned a, unsigned b) { return static_cast<int>((a + b - 1) / b); };
        auto remaining = [](unsigned have, unsigned need) {
            return need > have ? need - have : 0u;
        };
        if (rh.wmma.on() && wmma)
            consider(ceilDiv(remaining(ms, static_cast<unsigned>(rh.wmma.lead) * step), step));
        if (rh.mix.on()) {
            const Hide& t = rh.mix;
            const unsigned leadOrd = static_cast<unsigned>(t.lead) * step;
            const unsigned tail = static_cast<unsigned>(t.tail);
            if (ms >= leadOrd) {
                const unsigned have =
                    (ms - leadOrd) / step * fill + tailCount(rs, HideKind::Mix, t, c);
                if (have >= tail)
                    consider(0);
                else if (perStep > 0)
                    consider(ceilDiv(tail - have, perStep));
            } else if (wmma) {
                // To the lead first; the op that meets it and everything after fill the tail.
                const int toLead = ceilDiv(leadOrd - ms, step);
                const unsigned past = static_cast<unsigned>(toLead) * step - (leadOrd - ms);
                const unsigned have = past / step * fill;
                if (have >= tail)
                    consider(toLead);
                else if (perStep > 0)
                    consider(toLead + ceilDiv(tail - have, perStep));
            }
        }
        if (rh.valu.on()) {
            const unsigned need = remaining(valuSince(rs), static_cast<unsigned>(rh.valu.tail));
            if (need == 0)
                consider(0);
            else if (!wmma)
                consider(static_cast<int>(need));
        }
        return best;
    }

    // The mixed rule's remainder for `rs`: matrix ops short of the lead, then tail slots left
    // once it is met. {-1, 0} when the role has no mixed rule.
    std::pair<int, int> mixGap(const RoleStamp& rs, const RoleHide& rh, HideClass c,
                               int slots) const {
        if (!rh.mix.on()) return {-1, 0};
        const Hide& t = rh.mix;
        const unsigned step = xdlInc;
        const unsigned leadOrd = static_cast<unsigned>(t.lead) * step;
        const unsigned ms = matrixSince(rs);
        if (ms < leadOrd) return {static_cast<int>((leadOrd - ms + step - 1) / step), t.tail};
        const unsigned have = (ms - leadOrd) / step * static_cast<unsigned>(std::max(slots, 0)) +
                              tailCount(rs, HideKind::Mix, t, c);
        const unsigned tail = static_cast<unsigned>(t.tail);
        return {0, have >= tail ? 0 : static_cast<int>(tail - have)};
    }

    // Per-queue rebasing data for one join, filled before any stamp is touched.
    struct SlotFrame {
        std::array<unsigned, NUM_VM_FIFOS> myShift{}, otherShift{}, myFloor{}, otherFloor{};
    };

    // Join the paired flag and the anchor; gaining either weakens a wait, so it must propagate.
    static void mergeStampVm(VgprStamp& s, const VgprStamp* o, bool myVm, bool oVm,
                             const SlotFrame& vm, bool& strictDom) {
        // Paired survives only if every path that contributed a stamp got it from a paired op.
        const bool wasPaired = s.pairedFlat;
        const unsigned wasAnchorLds = s.anchorLds;
        s.pairedFlat = (!myVm || s.pairedFlat) && (!oVm || o->pairedFlat) && (myVm || oVm);
        // An anchor holds only if every path carrying a producer issued one.
        if (myVm && oVm) {
            if (s.anchorLds == 0 || o->anchorLds == 0) {
                s.anchorLds = 0;
                s.anchorTex = 0;
            } else {
                mergeSlotOrd(s.anchorLds, o->anchorLds, vm.myShift[FIFO_LDS],
                             vm.otherShift[FIFO_LDS], vm.myFloor[FIFO_LDS], vm.otherFloor[FIFO_LDS],
                             strictDom, "anchorLds");
                mergeSlotOrd(s.anchorTex, o->anchorTex, vm.myShift[FIFO_TEX],
                             vm.otherShift[FIFO_TEX], vm.myFloor[FIFO_TEX], vm.otherFloor[FIFO_TEX],
                             strictDom, "anchorTex");
            }
        } else if (oVm) {
            // The merged ordinals came from the other side, so its anchor comes too.
            s.anchorLds =
                rebase(o->anchorLds, vm.otherShift[FIFO_LDS], vm.otherFloor[FIFO_LDS], "anchorLds");
            s.anchorTex =
                rebase(o->anchorTex, vm.otherShift[FIFO_TEX], vm.otherFloor[FIFO_TEX], "anchorTex");
        } else if (s.anchorLds != 0) {
            s.anchorLds += vm.myShift[FIFO_LDS];
            s.anchorTex += vm.myShift[FIFO_TEX];
        }
        // Report any flip: a loss tightens the wait, so successors must be reprocessed too.
        if (wasPaired != s.pairedFlat || (wasAnchorLds == 0) != (s.anchorLds == 0))
            strictDom = true;
    }

    // Each hide class's mixed-rule lead per role, in matrix ordinals; 0 without a mixed rule.
    using LeadUnits = std::array<std::array<unsigned, NUM_ROLES>, NUM_HIDE_CLASSES>;
    LeadUnits leadUnits() const {
        LeadUnits leads{};
        for (int c = 0; c < NUM_HIDE_CLASSES; ++c) {
            const RuleSet* set = rulesFor(pipeOfHideClass(static_cast<HideClass>(c)));
            if (set == nullptr) continue;
            for (int r = 0; r < NUM_ROLES; ++r) {
                const Hide& mix = roleRules(*set, static_cast<Role>(r)).mix;
                if (mix.lead > 0 && mix.tail > 0)
                    leads[c][r] = static_cast<unsigned>(mix.lead) * xdlInc;
            }
        }
        return leads;
    }

    // One role stamp's ages as the rules read them; what a join compares and carries.
    struct Ages {
        unsigned matrix = 0;
        unsigned valu = 0;
        std::array<unsigned, NUM_HIDE_CLASSES> afterLead = {};
    };
    Ages agesOf(const RoleStamp& rs, Role r, const LeadUnits& leads) const {
        Ages a{matrixSince(rs), valuSince(rs), {}};
        for (int c = 0; c < NUM_HIDE_CLASSES; ++c)
            a.afterLead[c] =
                leads[c][r] == 0 ? 0 : valuAfterLead(rs, static_cast<HideClass>(c), leads[c][r]);
        return a;
    }
    // Store `a` as readings of clock `at`. A met lead keeps its reading (the ring it would read
    // predates the join); an unmet one stays unmet, its ordinal is still ahead.
    static void writeAges(RoleStamp& rs, const Ages& a, const AgeClock& at, Role r,
                          const LeadUnits& leads) {
        rs.matrixAt = at.matrix - std::min(a.matrix, at.matrix);
        rs.valuAt = at.valu - std::min(a.valu, at.valu);
        for (int c = 0; c < NUM_HIDE_CLASSES; ++c) {
            const unsigned lead = leads[c][r];
            rs.valuAtLead[c] = lead != 0 && a.matrix >= lead
                                   ? at.valu - std::min(a.afterLead[c], at.valu)
                                   : RoleStamp::kNotMet;
        }
    }

    // Join one role's ages: the smaller of each; only a smaller age reports strictDom.
    void mergeStampAge(RoleStamp& s, const RoleStamp* o, bool myVa, bool oVa, Role r,
                       const WaitcntBrackets& other, const AgeClock& merged, const LeadUnits& leads,
                       bool& strictDom) const {
        Ages mine = myVa ? agesOf(s, r, leads) : Ages{};
        if (oVa) {
            const Ages theirs = other.agesOf(*o, r, leads);
            if (!myVa) {
                // The ordinal merge reports an arriving producer; a stamp without one never
                // reads its ages.
                mine = theirs;
            } else {
                auto takeMin = [&](unsigned& m, unsigned t, const char* what) {
                    if (t < m) {
                        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   widen slot=" << what << " " << m
                                             << "->" << t << "\n");
                        m = t;
                        strictDom = true;
                    }
                };
                takeMin(mine.matrix, theirs.matrix, "matrixSince");
                takeMin(mine.valu, theirs.valu, "vSince");
                for (int c = 0; c < NUM_HIDE_CLASSES; ++c)
                    takeMin(mine.afterLead[c], theirs.afterLead[c], "valuAfterLead");
            }
        } else if (!myVa) {
            return;
        }
        writeAges(s, mine, merged, r, leads);
    }

    // Join the latched form; paths that disagree are a mixed kernel.
    // Either change only disables hides, so report it: successors must be reprocessed.
    void mergeXdlForm(const WaitcntBrackets& other, bool& strictDom) {
        const bool wasMixed = xdlFormMixed;
        const bool wasSeen = xdlIncSeen;
        xdlFormMixed = xdlFormMixed || other.xdlFormMixed ||
                       (xdlIncSeen && other.xdlIncSeen && !sameXdlForm(other.xdlInc, other.form_));
        if (!xdlIncSeen && other.xdlIncSeen) {
            xdlInc = other.xdlInc;
            form_ = other.form_;
        }
        xdlIncSeen = xdlIncSeen || other.xdlIncSeen;
        if (xdlFormMixed != wasMixed || xdlIncSeen != wasSeen) strictDom = true;
    }

    // Move the clock for this op; an anchor pipe also anchors the stamps that have none yet.
    void noteIssue(VaPipe pipe, unsigned inc) {
        switch (pipe) {
            case PIPE_XDL:
                clock.tickMatrix(inc);
                break;
            case PIPE_CSMACC:
            case PIPE_DPMACC:
                ++clock.valu;
                break;
            default:
                break;
        }
        if (!isAnchorPipe(pipe)) return;
        // vaUB counts this op, so vaUB - inc is the ticket before it. First anchor wins.
        for (auto& [k, s] : scores)
            for (RoleStamp& rs : s.role)
                if (rs.nextAnchorShared == 0) rs.nextAnchorShared = vaUB - inc;
    }

    // An op holding a ticket in both classes anchors every older producer. First anchor wins.
    void noteVmAnchor(unsigned ordLds, unsigned ordTex) {
        if (ordLds == 0 || ordTex == 0) return;
        for (auto& [k, s] : scores) {
            if (s.anchorLds != 0) continue;
            const bool olderLds = s.vmOrdLds != 0 && s.vmOrdLds < ordLds;
            const bool olderTex = s.vmOrdTex != 0 && s.vmOrdTex < ordTex;
            if (!olderLds && !olderTex) continue;
            s.anchorLds = ordLds;
            s.anchorTex = ordTex;
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   set anchor on v" << k.idx << " ["
                                 << s.vmOrdLds << "/" << s.vmOrdTex << " -> anchor " << ordLds
                                 << "/" << ordTex << "]\n");
        }
    }

    // Stamp the scoreboard for producer `inst`.
    void onProducer(WaitEventType ev, const StinkyInstruction& inst, const VGPRHalfKeyer& keyer) {
        CounterType ct = counterFromEvent(ev);

        if (ct == CT_VA_VDST) {
            VaPipe pipe = vaPipeOfEvent(ev);
            unsigned inc = hasMatrixScalePair(inst) ? 2u : 1u;
            vaPipeUB[pipe] += inc;
            vaUB += inc;
            unsigned ord = vaPipeUB[pipe];
            if (pipe == PIPE_XDL) latchXdlForm(inst, inc);
            noteIssue(pipe, inc);

            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   stamp event=" << eventName(ev) << " inc="
                                 << inc << " [pipe=" << vaPipeName(pipe) << " ord=" << ord
                                 << " ub=" << vaPipeUB[pipe] << " lb=" << vaPipeLB[pipe] << "]"
                                 << " (mnemonic=" << inst.getHwInstDesc()->mnemonic << ")\n");

            // A write stamps the dst role, a read the src role; the other role keeps its producer.
            auto stampVA = [&](unsigned idx, HighBitSel half, Role r) {
                RegKey k = keyer.producerKey(idx, half);
                RoleStamp& rs = scores[k].role[r];
                if (r == ROLE_DST) rs.ord.fill(0);
                rs.ord[pipe] = ord;
                rs.ordShared = vaUB;
                rs.nextAnchorShared = 0;
                rs.matrixAt = clock.matrix;
                rs.valuAt = clock.valu;
                rs.valuAtLead.fill(RoleStamp::kNotMet);
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     stamp va v" << k.idx << "("
                                     << halfName(k.half) << "," << roleName(r)
                                     << ") [pipe=" << vaPipeName(pipe) << " ord=" << ord << "]\n");
            };
            if (ctx->opts.enableESM2TrackValuVsrc)
                forEachVgprHalf(inst, OperandSide::Src, [&](unsigned idx, HighBitSel half) {
                    stampVA(idx, half, ROLE_SRC);
                });
            forEachVgprHalf(inst, OperandSide::Dst,
                            [&](unsigned idx, HighBitSel half) { stampVA(idx, half, ROLE_DST); });
            return;
        }

        // VM_VSRC. flat_* bumps both FIFOs.
        ++vmUB;
        unsigned ordLds = 0, ordTex = 0;
        if (enqueuesFifoLds(ev)) ordLds = ++vmFifoUB[FIFO_LDS];
        if (enqueuesFifoTex(ev)) ordTex = ++vmFifoUB[FIFO_TEX];
        noteVmAnchor(ordLds, ordTex);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   stamp vm event=" << eventName(ev)
                             << " [vm ub=" << vmUB << " lb=" << vmLB << "]"
                             << " [LDS ord=" << ordLds << " ub=" << vmFifoUB[FIFO_LDS]
                             << " lb=" << vmFifoLB[FIFO_LDS] << "]" << " [TEX ord=" << ordTex
                             << " ub=" << vmFifoUB[FIFO_TEX] << " lb=" << vmFifoLB[FIFO_TEX] << "]"
                             << " (mnemonic=" << inst.getHwInstDesc()->mnemonic << ")\n");

        auto stampVM = [&](unsigned idx, HighBitSel half) {
            RegKey k = keyer.producerKey(idx, half);
            VgprStamp& s = scores[k];
            const bool lds = enqueuesFifoLds(ev);
            const bool tex = enqueuesFifoTex(ev);
            // A paired stamp's other ordinal is already covered by this one, so drop it.
            if (s.pairedFlat && lds != tex) (lds ? s.vmOrdTex : s.vmOrdLds) = 0;
            // Set only the FIFO(s) this op enqueues into.
            if (lds) s.vmOrdLds = ordLds;
            if (tex) s.vmOrdTex = ordTex;
            // Paired when both ordinals came from this one flat_*.
            s.pairedFlat = lds && tex;
            s.anchorLds = 0;
            s.anchorTex = 0;
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     stamp vm v" << k.idx << "("
                                 << halfName(k.half) << ") [LDS ord=" << s.vmOrdLds << "]"
                                 << " [TEX ord=" << s.vmOrdTex << " paired=" << s.pairedFlat
                                 << "]\n");
        };
        // VM_VSRC tracks in-flight VMEM reads, which are always full DWORD.
        forEachVgprHalf(inst, OperandSide::Src,
                        [&](unsigned idx, HighBitSel) { stampVM(idx, HighBitSel::NONE); });
    }

    // Probe each VGPR operand for hazards and accumulate the wait.
    void onConsumer(const StinkyInstruction& inst, const VGPRHalfKeyer& keyer, Wait& wait,
                    Blockers* blockers = nullptr) const {
        // TEMP HACK: drop the src RAW va_vdst for valu->global_load and valu->global_prefetch
        // (prefetch lacks IF_GLOBALLoad, so classify it separately). Safe: the DAG already
        // spaces the vaddr producer >=32 cycles (CDNA5 isVmemAddrHazardConsumer covers it).
        const bool suppressSrcVaVdst = g_enableESM2SuppressValuToGlobalVaVdst &&
                                       (isGLOBALLoad(inst) || isGlobalPrefetch(inst));

        if (!suppressSrcVaVdst) {
            forEachVgprHalf(inst, OperandSide::Src, [&](unsigned idx, HighBitSel half) {
                keyer.forEachConsumerKey(idx, half, [&](RegKey k) {
                    determineVaWait(k, ROLE_DST, wait, "src(RAW)", blockers);
                });
            });
        }

        forEachVgprHalf(inst, OperandSide::Dst, [&](unsigned idx, HighBitSel half) {
            keyer.forEachConsumerKey(idx, half, [&](RegKey k) {
                determineVaWait(k, ROLE_DST, wait, "dst(WAW)", blockers);
                determineVaWait(k, ROLE_SRC, wait, "dst(WAR)", blockers);
            });
            // WAR on VM_VSRC: writer-vs-in-flight-VMEM-read uses full DWORD.
            RegKey full{RegType::V, idx, RegHalf::NONE};
            determineVmWait(full, wait, "dst(WAR)");
        });
    }

    // Same-class reads after this one satisfy its wait; clears a satisfied class's liveness.
    bool vmFollowerHides(const VgprStamp& s, unsigned fLds, unsigned fTex, bool& liveLds,
                         bool& liveTex) const {
        if (ctx->waitHide == nullptr) return false;
        const int reqLds = ctx->waitHide->vmVsrcLds;
        const int reqTex = ctx->waitHide->vmVsrcTex;
        const bool hidLds = liveLds && countMet(fLds, reqLds);
        const bool hidTex = liveTex && countMet(fTex, reqTex);
        if (s.pairedFlat && liveLds && liveTex) {
            if (!hidLds && !hidTex) return false;
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     skip vm_vsrc (flat) [LDS=" << fLds << "/"
                                 << reqLds << " TEX=" << fTex << "/" << reqTex << "]\n");
            return true;
        }
        if (hidLds) liveLds = false;
        if (hidTex) liveTex = false;
        if (liveLds || liveTex) return false;
        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     skip vm_vsrc [LDS=" << fLds << "/" << reqLds
                             << " TEX=" << fTex << "/" << reqTex << "]\n");
        return true;
    }

    // True when the anchor recorded for this producer has accumulated enough reads in
    // either class to prove it drained, which proves the producer drained too.
    bool bridgeHides(const VgprStamp& s, bool liveLds, bool liveTex) const {
        if (ctx->waitHide == nullptr || s.anchorLds == 0 || ctx->waitHide->vmVsrcBridge <= 0)
            return false;
        if (s.anchorLds <= vmFifoLB[FIFO_LDS] && s.anchorTex <= vmFifoLB[FIFO_TEX]) return false;
        // The anchor must sit after the producer in a class the producer is live in.
        const bool afterLds = liveLds && s.anchorLds > s.vmOrdLds;
        const bool afterTex = liveTex && s.anchorTex > s.vmOrdTex;
        if (!afterLds && !afterTex) return false;
        const unsigned fLds = vmFifoUB[FIFO_LDS] - s.anchorLds;
        const unsigned fTex = vmFifoUB[FIFO_TEX] - s.anchorTex;
        // Each class is tested against its own count.
        if (!countMet(fLds, ctx->waitHide->vmVsrcLds) && !countMet(fTex, ctx->waitHide->vmVsrcTex))
            return false;
        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     skip vm_vsrc (anchor LDS ord=" << s.anchorLds
                             << " TEX ord=" << s.anchorTex << ") [LDS=" << fLds << "/"
                             << ctx->waitHide->vmVsrcLds << " TEX=" << fTex << "/"
                             << ctx->waitHide->vmVsrcTex << "]\n");
        return true;
    }

    // Wait needed for this reg's VM readers. kNoWait when nothing live constrains it.
    unsigned vmFollowers(const VgprStamp& s) const {
        bool liveLds = s.vmOrdLds && s.vmOrdLds > vmFifoLB[FIFO_LDS];
        bool liveTex = s.vmOrdTex && s.vmOrdTex > vmFifoLB[FIFO_TEX];
        if (s.pairedFlat && !(liveLds && liveTex)) return kNoWait;
        unsigned fLds = liveLds ? vmFifoUB[FIFO_LDS] - s.vmOrdLds : 0u;
        unsigned fTex = liveTex ? vmFifoUB[FIFO_TEX] - s.vmOrdTex : 0u;

        if (vmFollowerHides(s, fLds, fTex, liveLds, liveTex)) return kNoWait;

        if (bridgeHides(s, liveLds, liveTex)) return kNoWait;

        // One flat_* retires from both FIFOs at once, so either proves it done.
        if (s.pairedFlat && liveLds && liveTex) return std::max(fLds, fTex);
        // Two distinct producers: must wait for both.
        unsigned f = ~0u;
        if (liveLds) f = std::min(f, fLds);
        if (liveTex) f = std::min(f, fTex);
        return f;
    }

    // Whether the shared VA order may supply this stamp's follower count.
    static bool sharedCountApplies(const RoleStamp& rs) {
        for (int p = 0; p < NUM_VA_PIPE; ++p)
            if (rs.ord[p] != 0 && !isCountablePipe(static_cast<VaPipe>(p))) return false;
        return true;
    }

    // Whether the stamp names a VM reader at all, live or retired.
    static bool hasVmProducer(const VgprStamp& s) {
        return s.vmOrdLds != 0 || s.vmOrdTex != 0;
    }

    // Follower count for this role's producer on pipe p; ~0u when a rule in force is met.
    unsigned vaPipeFollowers(const RoleStamp& rs, Role r, VaPipe p) const {
        unsigned followers = vaPipeUB[p] - rs.ord[p];
        if (ctx->waitHide == nullptr) return followers;
        const RuleSet* set = rulesFor(p);
        const HideClass c = hideClassOfPipe(p);
        const bool hidByRules =
            set != nullptr && roleHidden(rs, roleRules(*set, r), c, set->wmmaSlots);
        if (p == PIPE_XDL) {
            if (hidByRules) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     skip va_vdst [XDL " << roleName(r)
                                     << " matrix-ops=" << matrixSince(rs) << " valus="
                                     << tailCount(rs, HideKind::Mix, roleRules(*set, r).mix, c)
                                     << "]\n");
                return ~0u;
            }
            // Ops from the anchor onward are ordered behind this producer; live keeps it in frame.
            if (!xdlFormMixed && ctx->opts.xdlCountFromNextWmma && sharedCountApplies(rs) &&
                rs.nextAnchorShared > vaLB && vaUB - rs.nextAnchorShared > followers) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     va_vdst from next-wmma ["
                                     << (vaUB - rs.nextAnchorShared) << " vs per-pipe " << followers
                                     << "]\n");
                followers = vaUB - rs.nextAnchorShared;
            }
            return followers;
        }
        if (p == PIPE_CSMACC) {
            // Count from the producer's own ticket in the shared order.
            if (!xdlFormMixed && ctx->opts.sharedOrderCountFollowers && sharedCountApplies(rs) &&
                rs.ordShared != 0 && !hidByRules) {
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     va_vdst from shared order ["
                                     << (vaUB - rs.ordShared) << " vs per-pipe " << followers
                                     << "]\n");
                return vaUB - rs.ordShared;
            }
        }
        if (hidByRules) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     skip va_vdst [" << vaPipeName(p) << " "
                                 << roleName(r) << " matrix-ops=" << matrixSince(rs)
                                 << " valus=" << valuSince(rs) << "/"
                                 << tailCount(rs, HideKind::Mix, roleRules(*set, r).mix, c)
                                 << "]\n");
            return ~0u;
        }
        return followers;
    }

    // Wait needed for this role's producers; `blockers` collects the live ones no rule hides.
    unsigned vaFollowers(const RoleStamp& rs, Role r, Blockers* blockers) const {
        // The weaker count stops the per-pipe floor rising, so test the shared floor instead.
        if ((ctx->opts.sharedOrderCountFollowers || ctx->opts.xdlCountFromNextWmma) &&
            rs.hasProducer() && (rs.ordShared == 0 || rs.ordShared <= vaLB)) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     drained by shared order [ord="
                                 << rs.ordShared << " vaLB=" << vaLB << "]\n");
            return ~0u;
        }
        unsigned f = ~0u;
        for (int p = 0; p < NUM_VA_PIPE; ++p) {
            if (!rs.ord[p] || rs.ord[p] <= vaPipeLB[p]) continue;
            const unsigned pf = vaPipeFollowers(rs, r, static_cast<VaPipe>(p));
            if (pf != ~0u && blockers != nullptr)
                blockers->push_back({&rs, r, static_cast<VaPipe>(p)});
            f = std::min(f, pf);
        }
        return f;
    }

    // Accumulate the va_vdst wait for the hazard on VGPR `k` against its `r`-role producer.
    void determineVaWait(const RegKey& k, Role r, Wait& wait, const char* role,
                         Blockers* blockers) const {
        auto it = scores.find(k);
        if (it == scores.end()) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait va_vdst on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ") [no stamp]\n");
            return;
        }
        const RoleStamp& rs = it->second.role[r];
        if (!rs.hasProducer()) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait va_vdst on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ") [no " << roleName(r)
                                 << " producer]\n");
            return;
        }

        // Wait for live pipe producers.
        unsigned f = vaFollowers(rs, r, blockers);
        if (f == ~0u) {  // no live producer in any pipe
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait va_vdst on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ")" << vaStateStr(&rs, r)
                                 << " → no live producer\n");
            return;
        }
        unsigned chosen = std::min(f, maxEmittableWait(CT_VA_VDST));
        addWait(wait, CT_VA_VDST, chosen);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     wait hit va_vdst on v" << k.idx << "("
                             << halfName(k.half) << "," << role << ")" << vaStateStr(&rs, r)
                             << " f=" << f << " → wait=" << chosen << "\n");
    }

    // Accumulate the vm_vsrc wait for the hazard on VGPR `k`.
    void determineVmWait(const RegKey& k, Wait& wait, const char* role) const {
        auto it = scores.find(k);
        if (it == scores.end()) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait vm_vsrc on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ") [no stamp]\n");
            return;
        }
        const VgprStamp& s = it->second;

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
        if (f == kNoWait) {  // no live FIFO constrains it (drained or hidden by depth)
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]     no-wait vm_vsrc on v" << k.idx << "("
                                 << halfName(k.half) << "," << role << ")" << vmStateStr(&s)
                                 << " → hidden/drained (no wait)\n");
            return;
        }
        unsigned chosen = (f > 0) ? std::min(f, maxEmittableWait(CT_VM_VSRC)) : 0u;
        addWait(wait, CT_VM_VSRC, chosen);
        PASS_DEBUG(std::cerr << "[InsertWaitAlu]     wait hit vm_vsrc on v" << k.idx << "("
                             << halfName(k.half) << "," << role << ")" << vmStateStr(&s)
                             << " f=" << f << " → wait=" << chosen << "\n");
    }

    void applyWaitcnt(CounterType c, unsigned count) {
        if (count == kNoWait) return;
        if (c == CT_VA_VDST) {
            // count bounds the shared order and every pipe.
            unsigned newVaLB = vaUB >= count ? vaUB - count : 0u;
            if (newVaLB > vaLB) vaLB = newVaLB;
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

        mergeXdlForm(other, strictDom);

        // The clock after the join: the larger reading of each; the ring stays this side's.
        AgeClock merged = clock;
        merged.matrix = std::max(clock.matrix, other.clock.matrix);
        merged.valu = std::max(clock.valu, other.clock.valu);
        const LeadUnits leads = leadUnits();

        // Shared VA order: widen like a pipe, keeping the shift for the stamps below.
        unsigned myShiftShared = 0, otherShiftShared = 0;
        const unsigned myOldFloorShared = vaLB, otherOldFloorShared = other.vaLB;
        {
            unsigned mineIF = vaUB - vaLB;
            unsigned otherIF = other.vaUB - other.vaLB;
            unsigned newUB = vaLB + std::max(mineIF, otherIF);
            myShiftShared = newUB - vaUB;
            otherShiftShared = newUB - other.vaUB;
            vaUB = newUB;
        }

        {
            unsigned mineIF = vmUB - vmLB;
            unsigned otherIF = other.vmUB - other.vmLB;
            vmUB = vmLB + std::max(mineIF, otherIF);
        }

        SlotFrame vm;
        for (int g = 0; g < NUM_VM_FIFOS; ++g) {
            unsigned mineIF = vmFifoUB[g] - vmFifoLB[g];
            unsigned otherIF = other.vmFifoUB[g] - other.vmFifoLB[g];
            unsigned newUB = vmFifoLB[g] + std::max(mineIF, otherIF);
            vm.myFloor[g] = vmFifoLB[g];
            vm.otherFloor[g] = other.vmFifoLB[g];
            vm.myShift[g] = newUB - vmFifoUB[g];
            vm.otherShift[g] = newUB - other.vmFifoUB[g];
            vmFifoUB[g] = newUB;
        }

        for (const auto& [k, _] : other.scores) scores.try_emplace(k);

        for (auto& [k, s] : scores) {
            auto it = other.scores.find(k);
            const VgprStamp* o = (it != other.scores.end()) ? &it->second : nullptr;
            // Each role joins on its own; a dst stamp never meets a src stamp.
            for (int r = 0; r < NUM_ROLES; ++r) {
                RoleStamp& rs = s.role[r];
                const RoleStamp* ors = o ? &o->role[r] : nullptr;
                // Taken before the ordinal merge below, which can zero a stamp's last ordinal.
                const bool myVa = rs.hasProducer();
                const bool oVa = ors && ors->hasProducer();
                // Merge each pipe's ordinal independently.
                for (int p = 0; p < NUM_VA_PIPE; ++p) {
                    mergeSlotOrd(rs.ord[p], ors ? ors->ord[p] : 0, myShift[p], otherShift[p],
                                 myOldFloor[p], otherOldFloor[p], strictDom,
                                 vaPipeName(static_cast<VaPipe>(p)));
                }
                mergeSlotOrd(rs.ordShared, ors ? ors->ordShared : 0, myShiftShared,
                             otherShiftShared, myOldFloorShared, otherOldFloorShared, strictDom,
                             "ordShared");
                // An anchor holds only if every path carrying a producer issued one.
                const unsigned wasNextAnchor = rs.nextAnchorShared;
                if (myVa && oVa) {
                    if (rs.nextAnchorShared == 0 || ors->nextAnchorShared == 0) {
                        rs.nextAnchorShared = 0;
                    } else {
                        mergeSlotOrd(rs.nextAnchorShared, ors->nextAnchorShared, myShiftShared,
                                     otherShiftShared, myOldFloorShared, otherOldFloorShared,
                                     strictDom, "nextAnchorShared");
                    }
                } else if (oVa) {
                    // The merged ordinals came from the other side, so its anchor comes too.
                    rs.nextAnchorShared = rebase(ors->nextAnchorShared, otherShiftShared,
                                                 otherOldFloorShared, "nextAnchorShared");
                } else if (rs.nextAnchorShared != 0) {
                    rs.nextAnchorShared += myShiftShared;
                }
                // Report a loss: it tightens the wait, so successors must be reprocessed too.
                if ((wasNextAnchor == 0) != (rs.nextAnchorShared == 0)) strictDom = true;
                mergeStampAge(rs, ors, myVa, oVa, static_cast<Role>(r), other, merged, leads,
                              strictDom);
            }
            const bool myVm = hasVmProducer(s);
            const bool oVm = o && hasVmProducer(*o);
            mergeSlotOrd(s.vmOrdLds, o ? o->vmOrdLds : 0, vm.myShift[FIFO_LDS],
                         vm.otherShift[FIFO_LDS], vm.myFloor[FIFO_LDS], vm.otherFloor[FIFO_LDS],
                         strictDom, "vmLds");
            mergeSlotOrd(s.vmOrdTex, o ? o->vmOrdTex : 0, vm.myShift[FIFO_TEX],
                         vm.otherShift[FIFO_TEX], vm.myFloor[FIFO_TEX], vm.otherFloor[FIFO_TEX],
                         strictDom, "vmTex");
            mergeStampVm(s, o, myVm, oVm, vm, strictDom);
        }
        clock = merged;

        return strictDom;
    }

   public:
    // Debug-only VA snapshot of one role.
    std::string vaStateStr(const RoleStamp* rs, Role r = ROLE_DST) const {
        std::string out;
        for (int p = 0; p < NUM_VA_PIPE; ++p) {
            out += " [";
            out += vaPipeName(static_cast<VaPipe>(p));
            out += " ub=" + std::to_string(vaPipeUB[p]) + " lb=" + std::to_string(vaPipeLB[p]);
            if (rs) {
                unsigned ord = rs->ord[p];
                bool live = ord && ord > vaPipeLB[p];
                out += " ord=" + std::to_string(ord) + " live=" + std::to_string(live);
                if (live) out += " f=" + std::to_string(vaPipeUB[p] - ord);
            }
            out += "]";
        }
        if (rs) {
            const Ages a = agesOf(*rs, r, leadUnits());
            out += " [matrix=" + std::to_string(a.matrix);
            out += " v=" + std::to_string(a.valu);
            out += " afterLead=" + std::to_string(a.afterLead[CLASS_XDL]) + "/" +
                   std::to_string(a.afterLead[CLASS_CSMACC]) + "/" +
                   std::to_string(a.afterLead[CLASS_DPMACC]);
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
        if (s) {
            out += " paired=" + std::to_string(s->pairedFlat);
            out += " anchor=" + std::to_string(s->anchorLds) + "/" + std::to_string(s->anchorTex);
        }
        return out;
    }

    // One op of `op`'s kind that touches no VGPR, counted the way the walk counts it.
    void countBareOp(WaitAluHideOp op) {
        if (op == WaitAluHideOp::Wmma) {
            vaPipeUB[PIPE_XDL] += xdlInc;
            vaUB += xdlInc;
            noteIssue(PIPE_XDL, xdlInc);
            return;
        }
        if (op == WaitAluHideOp::Valu) {
            ++vaPipeUB[PIPE_CSMACC];
            ++vaUB;
            noteIssue(PIPE_CSMACC, 1);
            return;
        }
        ++vmUB;
        if (op != WaitAluHideOp::Tex) ++vmFifoUB[FIFO_LDS];
        if (op != WaitAluHideOp::Lds) ++vmFifoUB[FIFO_TEX];
    }

    // Take `o`'s latched WMMA form, if it has one.
    void copyFormFrom(const WaitcntBrackets& o) {
        xdlInc = o.xdlInc;
        form_ = o.form_;
        xdlIncSeen = o.xdlIncSeen;
        xdlFormMixed = o.xdlFormMixed;
    }
    bool formLatched() const {
        return xdlIncSeen;
    }

   private:
    static unsigned vaPipeSum(const std::array<unsigned, NUM_VA_PIPE>& a) {
        unsigned n = 0;
        for (int P = 0; P < NUM_VA_PIPE; ++P) n += a[P];
        return n;
    }

    // True when the follower count meets the arch entry; 0 is off.
    static bool countMet(unsigned count, int required) {
        return required > 0 && count >= static_cast<unsigned>(required);
    }

    // Shift both into the widened frame, keep the later.
    static void mergeSlotOrd(unsigned& myOrd, unsigned oOrd, unsigned myShift, unsigned otherShift,
                             unsigned myOldFloor, unsigned otherOldFloor, bool& strictDom,
                             const char* slot) {
        unsigned myS = (myOrd && myOrd > myOldFloor) ? myOrd + myShift : 0;
        unsigned oS = (oOrd && oOrd > otherOldFloor) ? oOrd + otherShift : 0;
        takeLater(myOrd, myS, oS, strictDom, slot);
    }

    // Move an ordinal into the widened frame; the clamp keeps a retired one from wrapping.
    static unsigned rebase(unsigned ord, unsigned shift, unsigned oldFloor,
                           const char* slot = "?") {
        if (ord == 0) return 0;
        PASS_DEBUG(if (shift > (~0u / 2) && ord < (0u - shift)) std::cerr
                   << "[InsertWaitAlu]   WRAP-AVOIDED slot=" << slot << " ord=" << ord << " shift=-"
                   << (0u - shift) << " floor=" << oldFloor << "\n");
        return std::max(ord, oldFloor) + shift;
    }

    static void takeLater(unsigned& myOrd, unsigned myS, unsigned oS, bool& strictDom,
                          const char* slot) {
        if (oS > myS) {
            PASS_DEBUG(std::cerr << "[InsertWaitAlu]   widen slot=" << slot << " " << myS << "->"
                                 << oS << "\n");
            myOrd = oS;
            strictDom = true;
        } else {
            myOrd = myS;
        }
    }

    // VA_VDST shared UB/LB.
    unsigned vaUB = 0;
    unsigned vaLB = 0;
    // VA_VDST per-pipe UB/LB.
    std::array<unsigned, NUM_VA_PIPE> vaPipeUB = {};
    std::array<unsigned, NUM_VA_PIPE> vaPipeLB = {};
    // The clock every stamp's ages are measured against.
    AgeClock clock;
    // This kernel's form row, latched on the first XDL op (nullptr: a form without a row).
    const HideForm* form_ = nullptr;
    // Ordinal step of the latched form, for scaling the matrix-op counts.
    unsigned xdlInc = 1;
    bool xdlIncSeen = false;
    // Set when a second matrix-op form issues; a mixed kernel runs without the form rules.
    bool xdlFormMixed = false;
    // VM_VSRC aggregate UB/LB.
    unsigned vmUB = 0;
    unsigned vmLB = 0;
    // VM_VSRC per-FIFO UB/LB.
    std::array<unsigned, NUM_VM_FIFOS> vmFifoUB = {};
    std::array<unsigned, NUM_VM_FIFOS> vmFifoLB = {};
    std::unordered_map<RegKey, VgprStamp, RegKeyHash> scores;
    // Owned by the pass instance, which outlives every bracket it builds.
    const WaitAluContext* ctx;
};

// Per-arch setup shared by the pass and WaitAluTracker. Idempotent.
GfxArchID setupWaitAluContext(WaitAluContext& ctx, VGPRHalfKeyer& keyer,
                              const PassContext& passCtx) {
    auto arch = passCtx.getGemmTileConfig().arch;
    const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archId);
    const bool hasD16 = archInfo && archInfo->hasD16Writes32BitVgpr();
    keyer = VGPRHalfKeyer(hasD16);
    ctx.waitHide = ctx.opts.ignoreWaitHide ? nullptr : &passCtx.getHWModel().waitHide;
    ctx.ageCap = ctx.waitHide == nullptr ? 0u : computeAgeCap(*ctx.waitHide);
    // Past the cap every rule with a matrix count is met, so the ring only has to reach it.
    assert(ctx.ageCap <= AgeClock::kRing && "wait-hide age cap exceeds the age ring");
    PASS_DEBUG(std::cerr << "[InsertWaitAlu] run arch=gfx" << arch[0] << arch[1] << arch[2]
                         << " hasD16Writes32BitVgpr=" << hasD16 << "\n");
    PASS_DEBUG(if (ctx.waitHide != nullptr) std::cerr
                   << "[InsertWaitAlu] waitHide" << " forms=" << ctx.waitHide->forms.size()
                   << " vmVsrcLds=" << waitHideStr(ctx.waitHide->vmVsrcLds)
                   << " vmVsrcTex=" << waitHideStr(ctx.waitHide->vmVsrcTex) << " vmVsrcBridge="
                   << waitHideStr(ctx.waitHide->vmVsrcBridge) << " [ageCap=" << ctx.ageCap << "]\n";
               else std::cerr << "[InsertWaitAlu] waitHide ignored (plain mode 2)\n");
    PASS_DEBUG(std::cerr << "[InsertWaitAlu] sharedOrder"
                         << " countFollowers=" << ctx.opts.sharedOrderCountFollowers
                         << " xdlFromNextWmma=" << ctx.opts.xdlCountFromNextWmma << "\n");
    return archId;
}

// `blockers`, when given, collects the producers whose va_vdst wait survives the rules.
Wait computeWaitForInst(const StinkyInstruction& inst, const WaitcntBrackets& sb,
                        const VGPRHalfKeyer& keyer, Blockers* blockers = nullptr) {
    Wait wait;

    // Step 1: scoreboard probes on every VGPR operand of `inst`.
    sb.onConsumer(inst, keyer, wait, blockers);

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

// How one instruction advances the walk, and what (if anything) is emitted for it.
enum class StepKind { Skip, AbsorbWait, Call, Normal };
struct Step {
    StepKind kind = StepKind::Skip;
    Wait wait;  // Normal: the s_wait_alu to emit before the instruction, if any
};

// Advance `sb` past `inst` exactly as the pass's walk does. The caller only emits: a Call
// gets its drain after the instruction, a Normal step with wait.hasAny() gets an
// s_wait_alu before it. Shared by the pass and WaitAluTracker so the two cannot diverge.
Step stepInstruction(WaitcntBrackets& sb, const StinkyInstruction& inst,
                     const VGPRHalfKeyer& keyer) {
    Step step;
    if (isPseudoInst(&inst)) return step;

    // Pre-existing s_wait_alu: absorb its va_vdst/vm_vsrc into LB so the
    // rest of the BB sees the post-wait state, and leave the instruction
    // in place so the runtime drain actually happens. Today the only
    // realistic source is hold_cnt-only survivors from RemoveWaitAluPass
    // (their va_vdst/vm_vsrc are already kNoWait, so the absorb is a
    // no-op); the emit branch merges fresh va_vdst/vm_vsrc into such a
    // survivor when it's the immediately-preceding instruction.
    if (isWaitAluInst(inst)) {
        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   absorb existing s_wait_alu\n");
        if (const auto* data = inst.getModifier<SWaitAluData>()) {
            if (data->hasField(SWaitAluData::VA_VDST))
                sb.applyWaitcnt(CT_VA_VDST, data->getField(SWaitAluData::VA_VDST));
            if (data->hasField(SWaitAluData::VM_VSRC))
                sb.applyWaitcnt(CT_VM_VSRC, data->getField(SWaitAluData::VM_VSRC));
        }
        step.kind = StepKind::AbsorbWait;
        return step;
    }

    PASS_DEBUG(std::cerr << "[InsertWaitAlu]   visit " << inst.getHwInstDesc()->mnemonic << "\n");

    // Function call (s_swappc): drain both counters right after the call, at the
    // return-landing site. The callee may leave VALU/VMEM instructions outstanding on
    // VA_VDST/VM_VSRC, so the drain is unconditional. The callee entry is drained
    // separately (insertCalleeEntryDrain).
    if (isCall(inst)) {
        PASS_DEBUG(std::cerr << "[InsertWaitAlu]   call — drain va_vdst(0)+vm_vsrc(0) "
                                "after s_swappc (callee->caller bracket)\n");
        sb.applyWaitcnt(CT_VA_VDST, 0);
        sb.applyWaitcnt(CT_VM_VSRC, 0);
        step.kind = StepKind::Call;
        return step;
    }

    step.kind = StepKind::Normal;
    step.wait = computeWaitForInst(inst, sb, keyer);
    if (step.wait.hasAny()) {
        PASS_DEBUG(
            std::cerr << "[InsertWaitAlu]   emit s_wait_alu va_vdst="
                      << (isNoWait(step.wait, CT_VA_VDST) ? -1 : int(step.wait.get(CT_VA_VDST)))
                      << " vm_vsrc="
                      << (isNoWait(step.wait, CT_VM_VSRC) ? -1 : int(step.wait.get(CT_VM_VSRC)))
                      << "\n");
        if (!isNoWait(step.wait, CT_VA_VDST))
            sb.applyWaitcnt(CT_VA_VDST, step.wait.get(CT_VA_VDST));
        if (!isNoWait(step.wait, CT_VM_VSRC))
            sb.applyWaitcnt(CT_VM_VSRC, step.wait.get(CT_VM_VSRC));
    }
    if (auto ev = classifyEvent(inst)) sb.onProducer(*ev, inst, keyer);
    return step;
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

class InsertWaitAluPassImpl : public Pass {
    WaitAluContext ctx_;
    std::unordered_map<BasicBlock*, WaitcntBrackets> blockEntryState;
    GfxArchID archId = GfxArchID{};
    VGPRHalfKeyer keyer{};

   public:
    explicit InsertWaitAluPassImpl(const InsertWaitAluOptions& opts) {
        ctx_.opts = opts;
    }

   private:
    WaitcntBrackets& entryState(BasicBlock* bb) {
        return blockEntryState.try_emplace(bb, ctx_).first->second;
    }

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

    // Process one BB starting from its accumulated entry state.
    // emit=false → run scoreboard, return exit state for Phase 1 propagation.
    // emit=true → re-run with the converged entry state and insert s_wait_alu.
    WaitcntBrackets runOnBasicBlock(BasicBlock& bb, bool emit) {
        WaitcntBrackets sb = entryState(&bb);

        PASS_DEBUG(std::cerr << "[InsertWaitAlu] " << (emit ? "emit" : "analyze") << " bb=\""
                             << bb.getLabel() << "\" entry sz=" << sb.scoresSize() << " va:"
                             << sb.vaStateStr(nullptr) << " vm:" << sb.vmStateStr(nullptr) << "\n");

        for (auto it = bb.begin(); it != bb.end();) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst) {
                ++it;
                continue;
            }
            // The walk itself lives in stepInstruction (shared with WaitAluTracker);
            // here only the emission happens.
            const Step step = stepInstruction(sb, *inst, keyer);
            if (step.kind == StepKind::Call) {
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
                it = nextIt;
                continue;
            }
            if (step.kind == StepKind::Normal && step.wait.hasAny() && emit) {
                // If the immediately-preceding instruction is a hold_cnt-only
                // s_wait_alu survivor, fold its hold_cnt into our new wait
                // so the constraint isn't lost and we don't emit two
                // adjacent waits.
                int holdCnt = extractAdjacentHoldCnt(bb, inst);
                if (holdCnt >= 0)
                    PASS_DEBUG(std::cerr << "[InsertWaitAlu]     fold hold_cnt=" << holdCnt
                                         << " from adjacent survivor\n");
                emitWaitAlu(bb, inst, step.wait, holdCnt);
                PASS_DEBUG(std::cerr << "[InsertWaitAlu]     inserted s_wait_alu before "
                                     << inst->getHwInstDesc()->mnemonic << "\n");
            }

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
                    if (entryState(succ).merge(exitState)) {
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
        archId = setupWaitAluContext(ctx_, keyer, passCtx);
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
    explicit InsertWaitAluModulePass(const InsertWaitAluOptions& opts) : opts(opts) {}

    const char* getName() const override {
        return "InsertWaitAluModulePass";
    }

    PreservedAnalyses run(StinkyAsmModule& M, PassContext& passCtx,
                          ModuleAnalysisManager& /*MAM*/) override {
        InsertWaitAluPassImpl impl(opts);
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
    InsertWaitAluOptions opts;
};

}  // namespace

namespace stinkytofu {
struct WaitAluTracker::Impl {
    WaitAluContext ctx;
    VGPRHalfKeyer keyer;
    WaitcntBrackets sb;
    Impl(const PassContext& passCtx, InsertWaitAluOptions opts) : sb(ctx) {
        ctx.opts = opts;
        setupWaitAluContext(ctx, keyer, passCtx);
    }
};

WaitAluTracker::WaitAluTracker(const PassContext& passCtx, InsertWaitAluOptions opts)
    : impl_(std::make_unique<Impl>(passCtx, opts)) {}

WaitAluTracker::~WaitAluTracker() = default;

WaitAluNeed WaitAluTracker::query(const StinkyInstruction& inst) const {
    WaitAluNeed need;
    if (isPseudoInst(&inst) || isWaitAluInst(inst) || isCall(inst)) return need;
    const Wait wait = computeWaitForInst(inst, impl_->sb, impl_->keyer);
    if (!isNoWait(wait, CT_VA_VDST)) need.vaVdst = static_cast<int>(wait.get(CT_VA_VDST));
    if (!isNoWait(wait, CT_VM_VSRC)) need.vmVsrc = static_cast<int>(wait.get(CT_VM_VSRC));
    return need;
}

// The most reads of `op`'s class the vm_vsrc rules need; past it, more of them hide nothing.
static int maxVmHideOps(const WaitAluContext& ctx, WaitAluHideOp op) {
    if (ctx.waitHide == nullptr) return 0;
    const HWModel::WaitHide& wh = *ctx.waitHide;
    const int lds = op == WaitAluHideOp::Tex ? 0 : wh.vmVsrcLds;
    const int tex = op == WaitAluHideOp::Lds ? 0 : wh.vmVsrcTex;
    return std::max({lds, tex, 0});
}

// Upper bound on countUntilFree's scan: the most `op`s any blocker needs; -1 when one never hides.
static int hideBound(const WaitcntBrackets& sb, const Blockers& blockers, WaitAluHideOp op) {
    int bound = 0;
    for (const Blocker& b : blockers) {
        const RuleSet* set = sb.rulesFor(b.p);
        if (set == nullptr) return -1;
        const int n = sb.stepsUntilHidden(*b.rs, roleRules(*set, b.r), hideClassOfPipe(b.p),
                                          set->wmmaSlots, op);
        if (n < 0) return -1;
        bound = std::max(bound, n);
    }
    return bound;
}

// The mixed hide covering every blocker: largest lead and tail, fewest slots; -1 when one has none.
static WaitAluMixGap mixGapBound(const WaitcntBrackets& sb, const Blockers& blockers) {
    WaitAluMixGap gap{0, 0, std::numeric_limits<int>::max()};
    for (const Blocker& b : blockers) {
        const RuleSet* set = sb.rulesFor(b.p);
        if (set == nullptr) return {};
        const auto [lead, tail] =
            sb.mixGap(*b.rs, roleRules(*set, b.r), hideClassOfPipe(b.p), set->wmmaSlots);
        if (lead < 0) return {};
        gap.leadWmmas = std::max(gap.leadWmmas, lead);
        gap.tailSlots = std::max(gap.tailSlots, tail);
        gap.wmmaSlots = std::min(gap.wmmaSlots, std::max(set->wmmaSlots, 0));
    }
    if (blockers.empty()) gap.wmmaSlots = 1;
    return gap;
}

// Latch `formWmma`'s form on `sb` the way the walk would on its first issue, if `sb` has none.
static void latchFormIfUnset(WaitcntBrackets& sb, const StinkyInstruction* formWmma) {
    if (sb.formLatched() || formWmma == nullptr) return;
    if (classifyEvent(*formWmma) != EV_VGPR_XDL_WRITE) return;
    sb.latchXdlForm(*formWmma, hasMatrixScalePair(*formWmma) ? 2u : 1u);
}

// Bare ops of `op`'s kind on a copy of `start` until `inst` needs no wait on their counter; -1
// when none does. `formWmma` as in latchFormIfUnset.
static int countUntilFree(const WaitcntBrackets& start, const StinkyInstruction& inst,
                          const WaitAluContext& ctx, const VGPRHalfKeyer& keyer, WaitAluHideOp op,
                          const StinkyInstruction* formWmma) {
    if (isPseudoInst(&inst) || isWaitAluInst(inst) || isCall(inst)) return 0;
    const bool va = op == WaitAluHideOp::Wmma || op == WaitAluHideOp::Valu;
    const CounterType c = va ? CT_VA_VDST : CT_VM_VSRC;
    if (isNoWait(computeWaitForInst(inst, start, keyer), c)) return 0;
    WaitcntBrackets sb = start;
    latchFormIfUnset(sb, formWmma);
    int maxOps = 0;
    if (!va) {
        maxOps = maxVmHideOps(ctx, op);
    } else {
        Blockers blockers;
        const Wait latched = computeWaitForInst(inst, sb, keyer, &blockers);
        if (isNoWait(latched, c)) {
            // The latch alone freed it; the first op confirms.
            maxOps = 1;
        } else if (blockers.empty()) {
            // A drain no bare op satisfies (an EXEC writer with work in flight).
            return -1;
        } else {
            maxOps = hideBound(sb, blockers, op);
            if (maxOps < 0) return -1;
            maxOps = std::max(maxOps, 1);
        }
    }
    for (int n = 1; n <= maxOps; ++n) {
        sb.countBareOp(op);
        if (isNoWait(computeWaitForInst(inst, sb, keyer), c)) return n;
    }
    return -1;
}

// The mixed hide `inst` still owes on a copy of `start`. Both counts 0 when it needs no va_vdst;
// {1, 0} when latching `formWmma`'s form alone frees it (that WMMA has to issue).
static WaitAluMixGap mixGapUntilFree(const WaitcntBrackets& start, const StinkyInstruction& inst,
                                     const VGPRHalfKeyer& keyer,
                                     const StinkyInstruction* formWmma) {
    if (isPseudoInst(&inst) || isWaitAluInst(inst) || isCall(inst)) return {0, 0, 1};
    if (isNoWait(computeWaitForInst(inst, start, keyer), CT_VA_VDST)) return {0, 0, 1};
    WaitcntBrackets sb = start;
    latchFormIfUnset(sb, formWmma);
    Blockers blockers;
    if (isNoWait(computeWaitForInst(inst, sb, keyer, &blockers), CT_VA_VDST)) return {1, 0, 1};
    if (blockers.empty()) return {};
    return mixGapBound(sb, blockers);
}

// A fresh scoreboard with `producer` just issued, the form taken from `live` or `formWmma`.
static WaitcntBrackets afterProducer(const WaitAluContext& ctx, const WaitcntBrackets& live,
                                     const VGPRHalfKeyer& keyer, const StinkyInstruction& producer,
                                     const StinkyInstruction* formWmma) {
    WaitcntBrackets sb(ctx);
    if (live.formLatched()) sb.copyFormFrom(live);
    latchFormIfUnset(sb, formWmma);
    stepInstruction(sb, producer, keyer);
    return sb;
}
int WaitAluTracker::untilFree(const StinkyInstruction& inst, WaitAluHideOp op,
                              const StinkyInstruction* formWmma) const {
    return countUntilFree(impl_->sb, inst, impl_->ctx, impl_->keyer, op, formWmma);
}

int WaitAluTracker::requiredGap(const StinkyInstruction& producer,
                                const StinkyInstruction& consumer, WaitAluHideOp op,
                                const StinkyInstruction* formWmma) const {
    return countUntilFree(afterProducer(impl_->ctx, impl_->sb, impl_->keyer, producer, formWmma),
                          consumer, impl_->ctx, impl_->keyer, op, nullptr);
}

WaitAluMixGap WaitAluTracker::untilFreeMixed(const StinkyInstruction& inst,
                                             const StinkyInstruction* formWmma) const {
    return mixGapUntilFree(impl_->sb, inst, impl_->keyer, formWmma);
}

WaitAluMixGap WaitAluTracker::requiredGapMixed(const StinkyInstruction& producer,
                                               const StinkyInstruction& consumer,
                                               const StinkyInstruction* formWmma) const {
    return mixGapUntilFree(afterProducer(impl_->ctx, impl_->sb, impl_->keyer, producer, formWmma),
                           consumer, impl_->keyer, nullptr);
}

void WaitAluTracker::commit(const StinkyInstruction& inst) {
    stepInstruction(impl_->sb, inst, impl_->keyer);
}

std::unique_ptr<Pass> createInsertWaitAluPass(InsertWaitAluOptions opts) {
    return std::make_unique<InsertWaitAluPassImpl>(opts);
}
std::unique_ptr<ModulePass> createInsertWaitAluModulePass(InsertWaitAluOptions opts) {
    return std::make_unique<InsertWaitAluModulePass>(opts);
}
}  // namespace stinkytofu
