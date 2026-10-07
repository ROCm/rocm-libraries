// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "IssueTimeline.hpp"

#include <algorithm>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace stinkytofu::coissue {
namespace {

constexpr int kVgprBase = 0, kVgprCount = 2048;
constexpr int kSgprBase = 2048, kSgprCount = 1024;
constexpr int kAgprBase = 3072, kAgprCount = 512;
constexpr int kSccSlot = 3584, kVccSlot = 3585, kExecSlot = 3586;
constexpr int kMBase = 3587, kMCount = 256;
static_assert(kMBase + kMCount <= kNumRegSlots, "register slots overflow");

bool isInsertedOpcode(int op) {
    return op == GFX::s_set_vgpr_msb || op == GFX::s_wait_alu || op == GFX::s_nop ||
           op == GFX::v_nop || op == GFX::s_delay_alu;
}

bool isScalarFormat(const HwInstDesc& desc) {
    switch (desc.microcode) {
        case MicrocodeFormat::MC_SOP1:
        case MicrocodeFormat::MC_SOP2:
        case MicrocodeFormat::MC_SOPC:
        case MicrocodeFormat::MC_SOPK:
        case MicrocodeFormat::MC_SOPP:
            return true;
        default:
            return false;
    }
}

IssueClass classify(const StinkyInstruction& inst) {
    if (isInsertedOpcode(inst.getUnifiedOpcode())) return IssueClass::Inserted;
    if (isMatrixInstruction(inst)) return IssueClass::Matrix;
    if (isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt)) return IssueClass::MemWait;
    if (isBarrier(inst)) return IssueClass::Barrier;
    if (isDSRead(inst)) return IssueClass::LdsLoad;
    if (isDSWrite(inst) || isDSAtomic(inst)) return IssueClass::LdsStore;
    if (isVectorALU(inst) || isTranscendental(inst)) return IssueClass::Valu;
    if (isScalarALU(inst) || isBranch(inst) || isConditionalBranch(inst) ||
        isScalarFormat(*inst.getHwInstDesc()))
        return IssueClass::Salu;
    return IssueClass::Memory;
}

int firstLiteral(const StinkyInstruction& inst) {
    for (const StinkyRegister& r : inst.getSrcRegs())
        if (r.dataType == StinkyRegister::Type::LiteralInt) return static_cast<int>(r.getLiteralInt());
    return -1;
}

void addSlots(const std::vector<StinkyRegister>& regs, std::vector<uint16_t>& out) {
    for (const StinkyRegister& r : regs) {
        if (!r.isRegister() || isPseudoReg(r)) continue;
        for (unsigned off = 0; off < r.reg.num; ++off) {
            const int slot = regSlot(static_cast<int>(r.reg.type), r.reg.idx + off);
            if (slot >= 0 && std::find(out.begin(), out.end(), slot) == out.end())
                out.push_back(static_cast<uint16_t>(slot));
        }
    }
}

bool classMatches(IssueClass cls, IssueClass kind, bool isBranchInst, bool branchIsSalu) {
    switch (cls) {
        case IssueClass::Any:
            return true;
        case IssueClass::Branch:
            return isBranchInst;
        case IssueClass::Salu:
            return kind == IssueClass::Salu && (branchIsSalu || !isBranchInst);
        default:
            return kind == cls;
    }
}

}  // namespace

int regSlot(int regType, uint32_t index) {
    const auto type = static_cast<RegType>(regType);
    const int i = static_cast<int>(index);
    switch (type) {
        case RegType::V:
            return i < kVgprCount ? kVgprBase + i : -1;
        case RegType::S:
            return i < kSgprCount ? kSgprBase + i : -1;
        case RegType::A:
        case RegType::ACC:
        case RegType::AGPR:
            return i < kAgprCount ? kAgprBase + i : -1;
        case RegType::SCC:
            return kSccSlot;
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
            return kVccSlot;
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return kExecSlot;
        case RegType::M:
            return i < kMCount ? kMBase + i : -1;
        default:
            return -1;
    }
}

LatencyReg regClassOfSlot(int slot) {
    if (slot < kSgprBase) return LatencyReg::Vgpr;
    if (slot < kAgprBase) return LatencyReg::Sgpr;
    if (slot == kSccSlot) return LatencyReg::Scc;
    if (slot == kVccSlot) return LatencyReg::Vcc;
    return LatencyReg::Any;
}

TimedInst makeLabel() {
    TimedInst t;
    t.isLabel = true;
    return t;
}

TimedInst makeTimedInst(const StinkyInstruction& inst, const HWModel& hw) {
    TimedInst t;
    t.inst = &inst;
    t.opcode = inst.getUnifiedOpcode();
    if (isPseudoInst(&inst)) {
        t.isLabel = true;
        return t;
    }
    t.kind = classify(inst);
    t.issue = inst.issueCycles;
    t.latency = inst.latencyCycles;
    t.isBranch = isBranch(inst) || isConditionalBranch(inst);
    t.isBarrierWait = isBarrierWait(inst);
    t.isNop = t.opcode == GFX::s_nop;
    t.isDsWait = t.opcode == GFX::s_wait_dscnt;
    t.isWaitAlu = t.opcode == GFX::s_wait_alu;
    if (t.isNop || t.kind == IssueClass::MemWait) t.count = firstLiteral(inst);
    if (t.kind == IssueClass::Matrix) {
        t.winLatency = inst.latencyCycles;
        t.coIssueMask = inst.coIssueWindow;
        t.blockedMask = inst.getHwInstDesc()->blockedScaleMask;
        if (const auto* fmt = inst.getModifier<MatrixFmtModifiers>())
            t.fp4Operands = fmt->fmtA == MatrixFmt::FP4 && fmt->fmtB == MatrixFmt::FP4;
    }
    addSlots(inst.getDestRegs(), t.defs);
    addSlots(inst.getSrcRegs(), t.uses);
    for (int r = 0; r < hw.hazards.numRules && r < 32; ++r) {
        const HazardRule& rule = hw.hazards.rules[r];
        if (rule.unit != HazardUnit::Cycles || rule.dir != HazardDir::WriteThenRead) continue;
        if (rule.isProducer(inst)) t.hazardProducer |= 1u << r;
        if (rule.isConsumer(inst)) t.hazardConsumer |= 1u << r;
    }
    return t;
}

const TimedInst& TimedInstCache::get(const StinkyInstruction& inst) {
    auto it = cache_.find(&inst);
    if (it != cache_.end()) return it->second;
    return cache_.emplace(&inst, makeTimedInst(inst, hw_)).first->second;
}

IssueTimeline::IssueTimeline(const TimingProfile& profile)
    : profile_(profile),
      producers_(kNumRegSlots),
      hazardReady_(profile.hazardGaps.size(), std::vector<int>(kNumRegSlots, 0)) {}

bool IssueTimeline::inWindow(int c) const {
    return hasWindow_ && c - winStart_ < winLatency_;
}

bool IssueTimeline::blocked(int c) const {
    if (!profile_.blockedCycleAtIssue || !inWindow(c)) return false;
    const int fromEnd = winLatency_ - 1 - (c - winStart_);
    return fromEnd < 16 && ((winBlocked_ >> fromEnd) & 1u) != 0u;
}

int IssueTimeline::roll(int c) const {
    while (blocked(c)) ++c;
    return c;
}

bool IssueTimeline::matches(const OpMatch& m, const TimedInst& inst) const {
    if (m.opcode >= 0) return inst.opcode == m.opcode;
    return classMatches(m.cls, inst.kind, inst.isBranch, /*branchIsSalu=*/false);
}

bool IssueTimeline::matchesPrev(const OpMatch& m) const {
    if (m.opcode >= 0) return prevOpcode_ == m.opcode;
    if (m.cls == IssueClass::Any) return true;
    if (!hasPrev_) return false;
    return classMatches(m.cls, prevKind_, prevKindIsBranch_, /*branchIsSalu=*/true);
}

int IssueTimeline::costOf(const TimedInst& inst) const {
    for (const CostRule& r : profile_.costRules)
        if (matches(r.inst, inst) && matchesPrev(r.after)) return r.cycles;
    return inst.kind == IssueClass::MemWait ? profile_.waitcntIssueCycles : inst.issue;
}

int IssueTimeline::dataReady(const TimedInst& inst) const {
    int ready = 0;
    for (uint16_t slot : inst.uses) {
        const Producer& p = producers_[slot];
        if (p.gen != gen_) continue;
        int lat = std::max(p.inst->latency, p.inst->issue);
        for (const LatencyRule& rule : profile_.latencyRules) {
            if (!classMatches(rule.producer, p.inst->kind, p.inst->isBranch, true)) continue;
            if (!classMatches(rule.consumer, inst.kind, inst.isBranch, false)) continue;
            if (rule.reg != LatencyReg::Any && rule.reg != regClassOfSlot(slot)) continue;
            lat = rule.cycles;
            break;
        }
        ready = std::max(ready, profile_.gapsFromIssueEnd
                                    ? p.after + std::max(0, lat - p.inst->issue)
                                    : p.at + lat);
    }
    if (inst.hazardConsumer != 0) {
        for (size_t g = 0; g < profile_.hazardGaps.size(); ++g) {
            const HazardGap& gap = profile_.hazardGaps[g];
            if (((inst.hazardConsumer >> gap.ruleIndex) & 1u) == 0u) continue;
            for (uint16_t slot : inst.uses)
                if (gap.reg == LatencyReg::Any || regClassOfSlot(slot) == gap.reg)
                    ready = std::max(ready, hazardReady_[g][slot]);
        }
    }
    return ready;
}

Placement IssueTimeline::place(const TimedInst& in, int notBefore) {
    Placement pl;
    if (in.isLabel) {
        pl.cycle = -1;
        pl.window = window_;
        return pl;
    }
    int at = 0;
    switch (in.kind) {
        case IssueClass::Matrix: {
            if (profile_.matrixQueueDepth == 0 && hasWindow_)
                t_ = std::max(t_, winStart_ + winLatency_);
            at = std::max({t_, dataReady(in), notBefore});
            const int depth = profile_.matrixQueueDepth;
            if (depth > 0 && static_cast<int>(pipe_.size()) >= depth)
                at = std::max(at, pipe_[pipe_.size() - depth].start);
            const int start = pipe_.empty() ? at : std::max(at, pipe_.back().end);
            pipe_.push_back({at, start, start + in.winLatency});
            hasWindow_ = true;
            winStart_ = at;
            winLatency_ = in.winLatency;
            winMask_ = in.coIssueMask;
            winBlocked_ = in.blockedMask;
            ++window_;
            t_ = roll(at + (profile_.matrixIssueCycles > 0 ? profile_.matrixIssueCycles : in.issue));
            break;
        }
        case IssueClass::Valu: {
            int c = roll(std::max({t_, dataReady(in), notBefore}));
            while (inWindow(c) && !(((winMask_ >> (c - winStart_)) & 1u) != 0u && !blocked(c))) ++c;
            at = c;
            t_ = roll(c + in.issue);
            break;
        }
        case IssueClass::MemWait: {
            at = std::max(t_, notBefore);
            const int settle = prevIsWait_ ? 0 : profile_.waitcntSettleCycles;
            int c = at + costOf(in) + settle;
            if (in.isDsWait && in.count >= 0 && static_cast<int>(dsDone_.size()) > in.count) {
                const int done = dsDone_[dsDone_.size() - in.count - 1];
                if (done > c) {
                    pl.stall = done - c;
                    c = done;
                }
            }
            t_ = roll(c);
            break;
        }
        case IssueClass::Barrier: {
            at = std::max(t_, notBefore);
            t_ = roll(at + std::max(in.issue, in.latency));
            if (profile_.sync == SyncModel::Conservative && in.isBarrierWait && !pipe_.empty())
                t_ = std::max(t_, pipe_.back().end);
            break;
        }
        default: {
            at = roll(std::max({t_, dataReady(in), notBefore}));
            t_ = roll(at + (in.isNop ? std::max(0, in.count) + 1 : costOf(in)));
            break;
        }
    }

    if (in.kind == IssueClass::LdsLoad || in.kind == IssueClass::LdsStore) {
        int done = at + in.latency;
        if (!dsDone_.empty()) done = std::max(done, dsDone_.back());
        dsDone_.push_back(done);
    }
    const bool gates = in.kind == IssueClass::Salu || in.kind == IssueClass::Valu ||
                       (profile_.latencyFromAllProducers && in.latency > in.issue);
    if (gates)
        for (uint16_t slot : in.defs) producers_[slot] = {gen_, at, t_, &in};
    if (in.hazardProducer != 0) {
        for (size_t g = 0; g < profile_.hazardGaps.size(); ++g) {
            const HazardGap& gap = profile_.hazardGaps[g];
            if (((in.hazardProducer >> gap.ruleIndex) & 1u) == 0u) continue;
            const int ready = (profile_.gapsFromIssueEnd ? t_ : at) + gap.cycles;
            for (uint16_t slot : in.defs)
                if (gap.reg == LatencyReg::Any || regClassOfSlot(slot) == gap.reg)
                    hazardReady_[g][slot] = ready;
        }
    }

    if (in.kind != IssueClass::Inserted) {
        hasPrev_ = true;
        prevKind_ = in.kind;
        prevKindIsBranch_ = in.isBranch;
    }
    prevOpcode_ = in.opcode;
    prevIsWait_ = in.kind == IssueClass::MemWait || in.isWaitAlu;

    pl.cycle = at;
    pl.window = window_;
    pl.pos = hasWindow_ ? at - winStart_ : at;
    return pl;
}

TripTiming steadyTrip(const std::vector<const TimedInst*>& body, const TimingProfile& profile,
                      int trips) {
    IssueTimeline tl(profile);
    TripTiming out;
    out.placements.resize(body.size());
    int t0 = -1;
    int matrixPerTrip = 0;
    for (const TimedInst* inst : body)
        if (!inst->isLabel && inst->kind == IssueClass::Matrix) ++matrixPerTrip;
    const int windowBase = (trips - 1) * matrixPerTrip;
    for (int k = 0; k < trips; ++k) {
        const bool last = k == trips - 1;
        for (size_t i = 0; i < body.size(); ++i) {
            Placement pl = tl.place(*body[i]);
            if (!last) continue;
            if (!body[i]->isLabel && t0 < 0) t0 = pl.cycle;
            pl.window -= windowBase;
            out.placements[i] = pl;
        }
    }
    out.cycles = t0 < 0 ? 0 : tl.now() - t0;
    const std::vector<PipeOp>& pipe = tl.pipe();
    const size_t n = static_cast<size_t>(matrixPerTrip);
    if (n > 0 && pipe.size() >= n) {
        out.pipe.assign(pipe.end() - static_cast<long>(n), pipe.end());
        for (size_t i = 1; i < out.pipe.size(); ++i)
            out.pipeIdle += std::max(0, out.pipe[i].start - out.pipe[i - 1].end);
        out.pipeIdleWithHandover = out.pipeIdle;
        if (pipe.size() > n)
            out.pipeIdleWithHandover +=
                std::max(0, out.pipe.front().start - pipe[pipe.size() - n - 1].end);
    }
    return out;
}

}  // namespace stinkytofu::coissue
