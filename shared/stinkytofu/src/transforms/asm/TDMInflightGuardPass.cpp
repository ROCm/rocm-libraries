// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/TDMInflightGuardPass.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

#define DEBUG_TYPE "TDMInflightGuardPass"

namespace {
using namespace stinkytofu;

/// Ceiling of the bound, above anything the 6-bit tensor counter can hold. A loop
/// that keeps issuing without a wait saturates here, which is what makes the
/// fixpoint terminate; the limit is clamped to it so the wait immediate fits.
constexpr int kCeiling = 64;

/// Instructions that increment this wave's tensor counter. A load whose descriptor
/// was nulled still issues and retires on the counter, so every one counts.
/// tensor_store_from_lds belongs here as well once it is modelled.
bool isTensorCounterOp(const StinkyInstruction& inst) {
    return isTensorLoad(inst);
}

/// Immediate of an s_wait_tensorcnt, or -1 when it cannot be decoded, in which case
/// the wait caps nothing. The literal operand wins over the modifier, as in
/// WaitDataflow's observedWaitDrains.
int tensorWaitImmediate(const StinkyInstruction& inst) {
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (src.dataType == StinkyRegister::Type::LiteralInt)
            return static_cast<int>(src.getLiteralInt());
    }
    const auto* data = inst.getModifier<SWaitTensorCntData>();
    return data != nullptr ? data->tlcnt : -1;
}

/// A straight-line run of the instruction stream: it starts at a label or right
/// after a branch, and only its last instruction can transfer control.
struct Segment {
    size_t begin = 0;
    size_t end = 0;
    std::vector<size_t> succs;
};

/// The CFG of a function's instruction stream in emission order, built from labels
/// and branch targets rather than from BasicBlock edges: RegionClonePass emits its
/// clone as one block that keeps the region's labels and branches but no edges for
/// them, and before CFGBuilderPass the whole kernel is such a block.
struct StreamCFG {
    std::vector<StinkyInstruction*> insts;
    std::vector<Segment> segs;
};

StreamCFG buildStreamCFG(Function& func) {
    StreamCFG cfg;
    std::unordered_map<std::string, std::vector<size_t>> labelPos;
    std::unordered_map<std::string, std::vector<size_t>> blockPos;
    for (BasicBlock& bb : func) {
        // .stir input and synthetic blocks name a branch target by the block alone,
        // with no LABEL instruction; such a name resolves to the block's first
        // instruction.
        if (!bb.getLabel().empty()) blockPos[bb.getLabel()].push_back(cfg.insts.size());
        for (IRBase& node : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == nullptr) continue;
            if (isLabel(*inst)) {
                if (const auto* ld = inst->getModifier<LabelData>())
                    labelPos[ld->label].push_back(cfg.insts.size());
            }
            cfg.insts.push_back(inst);
        }
    }
    for (auto& [name, positions] : blockPos) labelPos.try_emplace(name, std::move(positions));

    const size_t n = cfg.insts.size();
    if (n == 0) return cfg;

    std::vector<char> leader(n + 1, 0);
    leader[0] = 1;
    for (const auto& entry : labelPos)
        for (size_t p : entry.second) leader[p] = 1;
    for (size_t i = 0; i < n; ++i) {
        const StinkyInstruction& inst = *cfg.insts[i];
        if (isBranch(inst) || isEndOfFunction(inst)) leader[i + 1] = 1;
    }
    std::vector<size_t> segAt(n);
    for (size_t i = 0; i < n; ++i) {
        if (leader[i]) cfg.segs.push_back(Segment{i, i, {}});
        cfg.segs.back().end = i + 1;
        segAt[i] = cfg.segs.size() - 1;
    }

    // A branch to a target this stream does not define may land on any label.
    std::vector<size_t> anyLabel;
    for (const auto& entry : labelPos)
        for (size_t p : entry.second)
            if (p < n) anyLabel.push_back(segAt[p]);
    std::sort(anyLabel.begin(), anyLabel.end());
    anyLabel.erase(std::unique(anyLabel.begin(), anyLabel.end()), anyLabel.end());

    for (Segment& seg : cfg.segs) {
        const StinkyInstruction& last = *cfg.insts[seg.end - 1];
        bool fallsThrough = !isEndOfFunction(last);
        if (isBranch(last)) {
            fallsThrough = isConditionalBranch(last);
            // A register-target s_setpc_b64 returns from a callable function; in a
            // kernel it is a jump to an unknown target.
            const bool isReturn = isEndOfFunction(last) && func.getIsCallable();
            const std::vector<std::string> targets = getBranchTargets(last);
            std::vector<size_t> dests;
            bool resolved = !targets.empty();
            for (const std::string& target : targets) {
                const auto it = labelPos.find(target);
                if (it == labelPos.end()) {
                    resolved = false;
                    break;
                }
                for (size_t p : it->second)
                    if (p < n) dests.push_back(segAt[p]);
            }
            if (resolved)
                seg.succs.insert(seg.succs.end(), dests.begin(), dests.end());
            else if (!isReturn)
                seg.succs.insert(seg.succs.end(), anyLabel.begin(), anyLabel.end());
        }
        if (fallsThrough && seg.end < n) seg.succs.push_back(segAt[seg.end]);
    }
    return cfg;
}

class TDMInflightGuardPass : public Pass {
   public:
    static char ID;

    TDMInflightGuardPass(int limit, std::vector<Function*> functions)
        : limit_(std::min(limit, kCeiling)), functions_(std::move(functions)) {}

    const char* getName() const override {
        return "TDMInflightGuardPass";
    }

    PassID getPassID() const override {
        return &TDMInflightGuardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        if (limit_ <= 0) return PreservedAnalyses::all();

        const StreamCFG cfg = buildStreamCFG(func);
        if (cfg.segs.empty()) return PreservedAnalyses::all();
        const bool callsMayIssue = callsMayIssueTensorOps(func);

        // Forward dataflow taking the max at joins; -1 marks a segment no path
        // reaches. A kernel starts with nothing in flight; what a callable
        // function's caller left outstanding is unknown.
        std::vector<int> in(cfg.segs.size(), -1);
        in[0] = func.getIsCallable() ? kCeiling : 0;
        std::deque<size_t> work{0};
        std::vector<char> queued(cfg.segs.size(), 0);
        queued[0] = 1;
        const auto noGuard = [](size_t, int) {};
        while (!work.empty()) {
            const size_t s = work.front();
            work.pop_front();
            queued[s] = 0;
            const int out = runSegment(cfg, cfg.segs[s], in[s], callsMayIssue, noGuard);
            for (size_t t : cfg.segs[s].succs) {
                if (out <= in[t]) continue;
                in[t] = out;
                if (!queued[t]) {
                    queued[t] = 1;
                    work.push_back(t);
                }
            }
        }

        std::vector<StinkyInstruction*> guarded;
        int worstBound = 0;
        for (size_t s = 0; s < cfg.segs.size(); ++s) {
            if (in[s] < 0) continue;
            runSegment(cfg, cfg.segs[s], in[s], callsMayIssue, [&](size_t i, int bound) {
                guarded.push_back(cfg.insts[i]);
                worstBound = std::max(worstBound, bound);
            });
        }
        PASS_DEBUG(std::cerr << "[TDMInflightGuardPass] @" << func.getName()
                             << " segments=" << cfg.segs.size() << " guards=" << guarded.size()
                             << "\n");
        if (guarded.empty()) return PreservedAnalyses::all();

        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_wait_tensorcnt, archId);
        assert(waitDesc && "s_wait_tensorcnt opcode is not supported on this architecture");
        const int cap = limit_ - 1;
        const std::string comment =
            "at most " + std::to_string(limit_) + " TDM ops in flight per wave";
        for (StinkyInstruction* issue : guarded) {
            AsmIRBuilder irBuilder(*issue->getParent(), archId);
            StinkyInstruction* w = irBuilder.create(waitDesc, issue);
            w->addSrcReg(StinkyRegister(cap));
            SWaitTensorCntData d;
            d.tlcnt = static_cast<int8_t>(cap);
            w->addModifier<SWaitTensorCntData>(d);
            w->addModifier<CommentData>(CommentData{comment});
        }

        const std::string worst =
            worstBound >= kCeiling ? std::string("unbounded") : std::to_string(worstBound);
        emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, getName(), "TDMInflightGuard",
                             "@" + func.getName() + ": inserted " + std::to_string(guarded.size()) +
                                 " s_wait_tensorcnt " + std::to_string(cap) + " to keep at most " +
                                 std::to_string(limit_) +
                                 " TDM ops in flight per wave; worst bound before an issue: " +
                                 worst});
        return PreservedAnalyses::none();
    }

   private:
    /// The bound after cfg.insts[seg.begin, seg.end) when entered with \p bound.
    /// Every issue is evaluated as if a guard stood before it, and \p onGuard(index,
    /// bound) reports the issues where that guard lowers the bound. A guard at an
    /// issue already within the limit is the identity, so the fixpoint is the same
    /// as with guards at exactly the reported issues, and every set of guards that
    /// keeps all issues within the limit contains them.
    template <typename OnGuard>
    int runSegment(const StreamCFG& cfg, const Segment& seg, int bound, bool callsMayIssue,
                   OnGuard&& onGuard) const {
        const int cap = limit_ - 1;
        for (size_t i = seg.begin; i < seg.end; ++i) {
            const StinkyInstruction& inst = *cfg.insts[i];
            if (isTensorCounterOp(inst)) {
                if (bound > cap) onGuard(i, bound);
                bound = std::min(bound, cap) + 1;
            } else if (inst.is(InstFlag::IF_WaitTensorCnt)) {
                const int imm = tensorWaitImmediate(inst);
                if (imm >= 0) bound = std::min(bound, imm);
            } else if (callsMayIssue &&
                       (isCall(inst) ||
                        inst.getUnifiedOpcode() == GFX::FUNCTION_ASM_PLACEMENT_MARKER)) {
                // FlattenCalleesPass later splices a callable body in at its marker,
                // so falling through the marker runs that body.
                bound = kCeiling;
            }
        }
        return bound;
    }

    /// False only when the function list shows that no callable function issues a
    /// TDM op, so no call can add to this wave's count.
    bool callsMayIssueTensorOps(const Function& func) const {
        if (functions_.empty()) return true;
        for (Function* f : functions_) {
            if (f == nullptr || f == &func || !f->getIsCallable()) continue;
            for (BasicBlock& bb : *f) {
                for (IRBase& node : bb) {
                    auto* inst = dyn_cast<StinkyInstruction>(&node);
                    if (inst != nullptr && isTensorCounterOp(*inst)) return true;
                }
            }
        }
        return false;
    }

    const int limit_;
    const std::vector<Function*> functions_;
};

char TDMInflightGuardPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createTDMInflightGuardPass(int limit, std::vector<Function*> functions) {
    return std::make_unique<TDMInflightGuardPass>(limit, std::move(functions));
}
}  // namespace stinkytofu
