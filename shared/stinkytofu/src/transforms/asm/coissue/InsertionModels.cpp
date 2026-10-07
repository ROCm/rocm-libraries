// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "InsertionModels.hpp"

#include <algorithm>
#include <chrono>
#include <list>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/PrefetchBridgePlanner.hpp"
#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

namespace stinkytofu::coissue {
namespace {

using Sequence = std::vector<const StinkyInstruction*>;

std::vector<const StinkyInstruction*> irOrder(const BasicBlock& bb) {
    Sequence out;
    for (const IRBase& node : bb)
        if (const auto* inst = dyn_cast<StinkyInstruction>(&node)) out.push_back(inst);
    return out;
}

// --- VgprMsbModel: InsertVgprMsbPass ---------------------------------------------------

class VgprMsbModel : public InsertionModel {
   public:
    VgprMsbModel(VgprMsbMode mode, InstructionPool& pool) : mode_(mode), pool_(pool) {}
    const char* name() const override {
        return "VgprMsb";
    }
    void apply(PredictedBlock& block, Fidelity) override {
        if (mode_ == VgprMsbMode::None) return;
        for (Sequence& seq : block.seqs) {
            const std::vector<VgprMsbInsertion> plan = planVgprMsb(seq, mode_);
            if (plan.empty()) continue;
            Sequence out;
            out.reserve(seq.size() + 2 * plan.size());
            size_t next = 0;
            for (size_t i = 0; i < seq.size(); ++i) {
                for (; next < plan.size() && plan[next].before == i; ++next) {
                    if (plan[next].nopFirst) out.push_back(pool_.nop0());
                    out.push_back(pool_.bankSwitch(plan[next].immediate, plan[next].requiredMsb));
                }
                out.push_back(seq[i]);
            }
            seq.swap(out);
        }
    }

   private:
    VgprMsbMode mode_;
    InstructionPool& pool_;
};

// --- BridgeModel: PrefetchBridgeSubstitutionPass ---------------------------------------

bool isLdsFifoOp(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst) || isFLATLoad(inst) ||
           isFLATStore(inst) || isFLATAtomic(inst) || isFLATPrefetch(inst);
}

class BridgeModel : public InsertionModel {
   public:
    BridgeModel(Function& func, const std::vector<const BasicBlock*>& scope, int required,
                InstructionPool& pool)
        : required_(required), pool_(pool) {
        for (BasicBlock& bb : func) {
            blocks_.push_back(&bb);
            irOrders_.push_back(irOrder(bb));
        }
        for (size_t s = 0; s < scope.size(); ++s) scopeIndex_[scope[s]] = s;
        for (const Sequence& seq : irOrders_)
            for (const StinkyInstruction* inst : seq) {
                if (!isGlobalPrefetch(*inst)) continue;
                for (const StinkyRegister& r : inst->getSrcRegs()) {
                    if (!r.isRegister() || isPseudoReg(r) || r.reg.type != RegType::V) continue;
                    for (unsigned off = 0; off < r.reg.num; ++off) addrRegs_.insert(r.reg.idx + off);
                }
            }
        // The planner only reads the events, so every block keeps just those.
        for (Sequence& seq : irOrders_) {
            Sequence events;
            for (const StinkyInstruction* inst : seq)
                if (isEvent(*inst)) events.push_back(inst);
            seq.swap(events);
        }
    }
    const char* name() const override {
        return "PrefetchBridge";
    }

    void apply(PredictedBlock& block, Fidelity fidelity) override {
        if (required_ <= 0 || addrRegs_.empty()) return;
        // The plan only depends on where the prefetches, the LDS ops and the writes of a
        // prefetch address sit relative to each other; reuse it while those stay put.
        const bool movedNonEvent = block.movedFromBase != nullptr && basePlanned_ &&
                                   !isEvent(*block.movedFromBase);
        if (movedNonEvent) {
            if (!(planned_ && lastIsBase_)) {
                chosen_ = baseChosen_;
                lastEvents_ = baseEvents_;
                planned_ = true;
                lastIsBase_ = true;
            }
        }
        Sequence events;
        std::vector<Sequence> blockEvents;
        if (!movedNonEvent) {
            blockEvents.resize(block.seqs.size());
            for (size_t b = 0; b < block.seqs.size(); ++b)
                for (const StinkyInstruction* inst : block.seqs[b])
                    if (isEvent(*inst)) {
                        events.push_back(inst);
                        blockEvents[b].push_back(inst);
                    }
        }
        if (!movedNonEvent && (!planned_ || events != lastEvents_)) {
            std::vector<BlockOrder> layout;
            layout.reserve(blocks_.size());
            for (size_t b = 0; b < blocks_.size(); ++b) {
                auto it = scopeIndex_.find(blocks_[b]);
                layout.push_back({blocks_[b], it != scopeIndex_.end() ? &blockEvents[it->second]
                                                                      : &irOrders_[b]});
            }
            chosen_ = planPrefetchBridge(layout, required_);
            lastEvents_ = std::move(events);
            planned_ = true;
            lastIsBase_ = false;
        }
        if (fidelity == Fidelity::ExactBase) {
            baseChosen_ = chosen_;
            baseEvents_ = lastEvents_;
            basePlanned_ = true;
            lastIsBase_ = true;
        }
        if (chosen_.empty()) return;
        for (Sequence& seq : block.seqs)
            for (const StinkyInstruction*& inst : seq)
                if (chosen_.count(inst) != 0) inst = pool_.flatBridge(*inst);
    }

   private:
    bool isEvent(const StinkyInstruction& inst) {
        if (isLdsFifoOp(inst) || isGlobalPrefetch(inst)) return true;
        auto it = writesAddr_.find(&inst);
        if (it != writesAddr_.end()) return it->second;
        bool writes = false;
        for (const StinkyRegister& r : inst.getDestRegs()) {
            if (!r.isRegister() || isPseudoReg(r) || r.reg.type != RegType::V) continue;
            for (unsigned off = 0; off < r.reg.num && !writes; ++off)
                writes = addrRegs_.count(r.reg.idx + off) != 0;
        }
        writesAddr_[&inst] = writes;
        return writes;
    }

    int required_;
    InstructionPool& pool_;
    std::vector<const BasicBlock*> blocks_;
    std::vector<Sequence> irOrders_;
    std::unordered_map<const BasicBlock*, size_t> scopeIndex_;
    std::unordered_set<unsigned> addrRegs_;
    std::unordered_map<const StinkyInstruction*, bool> writesAddr_;
    bool planned_ = false;
    bool lastIsBase_ = false;
    Sequence lastEvents_;
    std::unordered_set<const StinkyInstruction*> chosen_;
    bool basePlanned_ = false;
    Sequence baseEvents_;
    std::unordered_set<const StinkyInstruction*> baseChosen_;
};

// --- WaitAluModel: InsertWaitAluPass ---------------------------------------------------

bool isHoldOnlyWaitAlu(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() != GFX::s_wait_alu) return false;
    const auto* data = inst.getModifier<SWaitAluData>();
    return data != nullptr && data->hasField(SWaitAluData::HOLD_CNT) &&
           !data->hasField(SWaitAluData::VA_VDST) && !data->hasField(SWaitAluData::VM_VSRC);
}

// Whether `inst` can change InsertWaitAlu's scoreboard or need one of its waits. A scalar
// instruction with no VGPR operand that leaves EXEC alone does neither, and neither do the
// bank switches and nops.
bool waitAluRelevant(const StinkyInstruction& inst) {
    if (isPseudoInst(&inst)) return false;
    const int op = inst.getUnifiedOpcode();
    if (op == GFX::s_set_vgpr_msb || op == GFX::s_nop || op == GFX::s_delay_alu) return false;
    if (op == GFX::s_wait_alu || isCall(inst)) return true;
    const bool scalar = isScalarALU(inst) || isBranch(inst) || isConditionalBranch(inst);
    if (!scalar) return true;
    for (const auto* regs : {&inst.getDestRegs(), &inst.getSrcRegs()})
        for (const StinkyRegister& r : *regs)
            if (r.isRegister() && (r.reg.type == RegType::V || r.reg.type == RegType::EXEC ||
                                   r.reg.type == RegType::EXEC_LO || r.reg.type == RegType::EXEC_HI))
                return true;
    return false;
}

class WaitAluModel : public InsertionModel {
   public:
    WaitAluModel(const PassContext& passCtx, InsertWaitAluOptions opts, InstructionPool& pool)
        : empty_(passCtx, opts), pool_(pool) {}
    const char* name() const override {
        return "WaitAlu";
    }

    // The pass's two phases over the scope: widen every block's entry state with its
    // in-scope predecessors' exits until nothing changes, then put an s_wait_alu in front of
    // every instruction that needs one. A loop header's back edge carries one trip's state
    // into the next: the warm-up trip.
    void apply(PredictedBlock& block, Fidelity fidelity) override {
        const size_t n = block.blocks.size();
        // A moved instruction the scoreboard does not see leaves every wait as it was.
        if (block.movedFromBase != nullptr && hasBase_ && !baseHoldSurvivor_ &&
            !waitAluRelevant(*block.movedFromBase)) {
            ++hits_;
            insertWaits(block, base_);
            return;
        }
        std::vector<std::vector<const StinkyInstruction*>> relevant(n);
        bool holdSurvivor = false;
        for (size_t b = 0; b < n; ++b)
            for (const StinkyInstruction* inst : block.seqs[b]) {
                if (!waitAluRelevant(*inst)) continue;
                relevant[b].push_back(inst);
                holdSurvivor |= isHoldOnlyWaitAlu(*inst);
            }
        // The scoreboard only sees the relevant instructions: when they keep their order, so
        // do the waits. A hold-only survivor folds by adjacency, so then it is recomputed.
        // The candidates of one round all start from the same order, so a few entries keep
        // that order's waits at hand.
        const Waits* waits = nullptr;
        if (!holdSurvivor) {
            for (auto it = cache_.begin(); it != cache_.end(); ++it)
                if (it->relevant == relevant) {
                    cache_.splice(cache_.begin(), cache_, it);
                    waits = &cache_.front().waits;
                    ++hits_;
                    break;
                }
        }
        if (waits == nullptr && fidelity == Fidelity::Screen && hasBase_ && !holdSurvivor) {
            ++screened_;
            waits = &base_;
        }
        if (waits == nullptr) {
            ++misses_;
            if (cache_.size() >= kCacheEntries) cache_.pop_back();
            cache_.push_front({std::move(relevant), computeWaits(block)});
            waits = &cache_.front().waits;
        }
        if (fidelity == Fidelity::ExactBase) {
            base_ = *waits;
            hasBase_ = true;
            baseHoldSurvivor_ = holdSurvivor;
        }
        insertWaits(block, *waits);
    }

    std::string stats() const override {
        return "waits reused " + std::to_string(hits_) + ", screened " + std::to_string(screened_) +
               ", computed " + std::to_string(misses_) + " in " + std::to_string(rounds_) + " rounds";
    }

   private:
    using Waits = std::unordered_map<const StinkyInstruction*, std::tuple<int, int, int>>;

    void insertWaits(PredictedBlock& block, const Waits& waits) {
        for (Sequence& seq : block.seqs) {
            Sequence out;
            out.reserve(seq.size() + 8);
            for (const StinkyInstruction* inst : seq) {
                auto it = waits.find(inst);
                if (it != waits.end()) {
                    const auto& [va, vm, hold] = it->second;
                    if (hold >= 0 && !out.empty() && isHoldOnlyWaitAlu(*out.back())) out.pop_back();
                    out.push_back(pool_.waitAlu(va, vm, hold));
                }
                out.push_back(inst);
            }
            seq.swap(out);
        }
    }
    struct Entry {
        std::vector<std::vector<const StinkyInstruction*>> relevant;
        Waits waits;
    };
    static constexpr size_t kCacheEntries = 4;

    Waits computeWaits(const PredictedBlock& block) {
        Waits waits_;
        const size_t n = block.blocks.size();
        std::unordered_map<const BasicBlock*, size_t> index;
        for (size_t b = 0; b < n; ++b) index[block.blocks[b]] = b;
        std::vector<WaitAluTracker> entry(n, empty_);
        constexpr int kMaxRounds = 32;
        for (int round = 0; round < kMaxRounds; ++round) {
            // A round that widens no entry walked every block from its final entry state, so
            // its queries are what the pass emits.
            bool changed = false;
            waits_.clear();
            for (size_t b = 0; b < n; ++b) {
                WaitAluTracker t = entry[b];
                const StinkyInstruction* prev = nullptr;
                for (const StinkyInstruction* inst : block.seqs[b]) {
                    const WaitAluNeed need = t.step(*inst);
                    if (need.any()) {
                        // A hold_cnt-only survivor right in front folds into the new wait.
                        int hold = -1;
                        if (prev != nullptr && isHoldOnlyWaitAlu(*prev))
                            hold = static_cast<int>(
                                prev->getModifier<SWaitAluData>()->getField(SWaitAluData::HOLD_CNT));
                        waits_[inst] = {need.vaVdst, need.vmVsrc, hold};
                    }
                    prev = inst;
                }
                for (const BasicBlock* succ : block.blocks[b]->getSuccessors()) {
                    auto it = index.find(succ);
                    if (it != index.end()) changed |= entry[it->second].merge(t);
                }
            }
            ++rounds_;
            if (!changed) return waits_;
        }
        return waits_;
    }

    WaitAluTracker empty_;
    InstructionPool& pool_;
    std::list<Entry> cache_;
    size_t hits_ = 0;
    size_t misses_ = 0;
    size_t rounds_ = 0;
    size_t screened_ = 0;
    Waits base_;
    bool hasBase_ = false;
    bool baseHoldSurvivor_ = false;
};

// --- CoexecNopModel: InsertCoexecHazardPass --------------------------------------------

class CoexecNopModel : public InsertionModel {
   public:
    CoexecNopModel(Function& func, const HWModel& hw, InstructionPool& pool) : hw_(hw), pool_(pool) {
        // Blocks outside the scope keep the v_nops the pass gives them for their IR order.
        std::vector<const BasicBlock*> all;
        for (BasicBlock& bb : func) {
            all.push_back(&bb);
            work_[&bb] = irOrder(bb);
        }
        planCoexecNops(hw_, all, work_, *pool_.vnop());
    }
    const char* name() const override {
        return "CoexecNop";
    }
    void apply(PredictedBlock& block, Fidelity fidelity) override {
        // A scalar instruction is no coexec filler, consumer or producer: when it is the one
        // that moved, every v_nop stays in front of the same instruction.
        const StinkyInstruction* moved = block.movedFromBase;
        if (moved != nullptr && hasBase_ && (isScalarALU(*moved) || isBranch(*moved)) &&
            !isVectorALU(*moved) && !isMatrixInstruction(*moved)) {
            insertNops(block, base_);
            return;
        }
        for (size_t b = 0; b < block.blocks.size(); ++b) work_[block.blocks[b]] = block.seqs[b];
        const auto plan = planCoexecNops(hw_, block.blocks, work_, *pool_.vnop());
        for (size_t b = 0; b < block.blocks.size(); ++b) block.seqs[b] = work_[block.blocks[b]];
        if (fidelity == Fidelity::ExactBase) {
            base_.clear();
            for (const auto& list : plan)
                for (const CoexecNopInsertion& i : list) base_[i.before] = i.count;
            hasBase_ = true;
        }
    }

   private:
    void insertNops(PredictedBlock& block, const std::unordered_map<const StinkyInstruction*, int>& nops) {
        if (nops.empty()) return;
        for (Sequence& seq : block.seqs) {
            Sequence out;
            out.reserve(seq.size() + 4);
            for (const StinkyInstruction* inst : seq) {
                auto it = nops.find(inst);
                if (it != nops.end()) out.insert(out.end(), static_cast<size_t>(it->second), pool_.vnop());
                out.push_back(inst);
            }
            seq.swap(out);
        }
    }

    const HWModel& hw_;
    InstructionPool& pool_;
    BlockSequences work_;
    std::unordered_map<const StinkyInstruction*, int> base_;
    bool hasBase_ = false;
};

}  // namespace

std::unique_ptr<InsertionModel> makeVgprMsbModel(VgprMsbMode mode, InstructionPool& pool) {
    return std::make_unique<VgprMsbModel>(mode, pool);
}

std::unique_ptr<InsertionModel> makeBridgeModel(Function& func,
                                                const std::vector<const BasicBlock*>& scope,
                                                int required, InstructionPool& pool) {
    return std::make_unique<BridgeModel>(func, scope, required, pool);
}

std::unique_ptr<InsertionModel> makeWaitAluModel(const PassContext& passCtx,
                                                 InsertWaitAluOptions opts, InstructionPool& pool) {
    return std::make_unique<WaitAluModel>(passCtx, opts, pool);
}

std::unique_ptr<InsertionModel> makeCoexecNopModel(Function& func, const HWModel& hw,
                                                   InstructionPool& pool) {
    return std::make_unique<CoexecNopModel>(func, hw, pool);
}

// --- InstructionPool -------------------------------------------------------------------

InstructionPool::InstructionPool(GfxArchID arch) : arch_(arch), scratch_("coissue-pool") {
    if (const auto* info = ArchHelper::getInstance().getArchInfo(arch)) {
        GemmTileConfig config = scratch_.getGemmTileConfig();
        config.arch = {static_cast<int>(info->major), static_cast<int>(info->minor),
                       static_cast<int>(info->stepping)};
        scratch_.setGemmTileConfig(config);
    }
    block_ = scratch_.createBasicBlock("pool");
}

StinkyInstruction* InstructionPool::create(int opcode) {
    AsmIRBuilder builder(*block_, arch_);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(static_cast<GFX>(opcode), arch_));
    owned_.insert(inst);
    return inst;
}

const StinkyInstruction* InstructionPool::bankSwitch(int immediate, int requiredMsb) {
    (void)requiredMsb;
    auto it = bankSwitches_.find(immediate);
    if (it != bankSwitches_.end()) return it->second;
    StinkyInstruction* inst = create(GFX::s_set_vgpr_msb);
    inst->addSrcReg(StinkyRegister(immediate));
    bankSwitches_[immediate] = inst;
    return inst;
}

const StinkyInstruction* InstructionPool::nop0() {
    if (nop0_ == nullptr) {
        StinkyInstruction* inst = create(GFX::s_nop);
        inst->addSrcReg(StinkyRegister(0));
        nop0_ = inst;
    }
    return nop0_;
}

const StinkyInstruction* InstructionPool::vnop() {
    if (vnop_ == nullptr) vnop_ = create(GFX::v_nop);
    return vnop_;
}

const StinkyInstruction* InstructionPool::waitAlu(int vaVdst, int vmVsrc, int holdCnt) {
    const auto key = std::make_tuple(vaVdst, vmVsrc, holdCnt);
    auto it = waitAlus_.find(key);
    if (it != waitAlus_.end()) return it->second;
    StinkyInstruction* inst = create(GFX::s_wait_alu);
    inst->addModifier<SWaitAluData>(
        SWaitAluData(vaVdst, /*va_sdst=*/-1, /*va_ssrc=*/-1, holdCnt, vmVsrc, /*va_vcc=*/-1,
                     /*sa_sdst=*/-1));
    waitAlus_[key] = inst;
    return inst;
}

const StinkyInstruction* InstructionPool::flatBridge(const StinkyInstruction& prefetch) {
    auto it = bridges_.find(&prefetch);
    if (it != bridges_.end()) return it->second;
    StinkyInstruction* inst = prefetch.clone();
    block_->appendIR(inst);
    inst->updateHwInstDesc(getMCIDByUOp(GFX::flat_prefetch_b8, arch_));
    std::vector<StinkyRegister> srcs;
    for (const StinkyRegister& src : inst->getSrcRegs()) {
        if (src.dataType == StinkyRegister::Type::LiteralString && src.literalValue == "off")
            continue;
        srcs.push_back(src);
    }
    inst->setSrcRegs(srcs);
    owned_.insert(inst);
    bridges_[&prefetch] = inst;
    originals_[inst] = &prefetch;
    return inst;
}

// --- InsertionPipeline -----------------------------------------------------------------

InsertionPipeline::InsertionPipeline(Function& func, std::vector<const BasicBlock*> scope,
                                     const PassContext& passCtx, const InsertionConfig& config)
    : scope_(std::move(scope)) {
    const auto triple = passCtx.getGemmTileConfig().arch;
    const GfxArchID arch = getGfxArchID(triple[0], triple[1], triple[2]);
    pool_ = std::make_unique<InstructionPool>(arch);
    const HWModel& hw = passCtx.getHWModel();
    models_.push_back(std::make_unique<VgprMsbModel>(config.msbMode, *pool_));
    if (config.esm2) {
        if (getMCIDByUOp(GFX::flat_prefetch_b8, arch) != nullptr)
            models_.push_back(
                std::make_unique<BridgeModel>(func, scope_, hw.waitHide.vmVsrcBridge, *pool_));
        models_.push_back(std::make_unique<WaitAluModel>(passCtx, config.waitAlu, *pool_));
    }
    models_.push_back(std::make_unique<CoexecNopModel>(func, hw, *pool_));
}

InsertionPipeline::~InsertionPipeline() = default;

PredictedBlock InsertionPipeline::predict(
    const std::vector<std::vector<const StinkyInstruction*>>& orders, Fidelity fidelity,
    const StinkyInstruction* movedFromBase) {
    PredictedBlock block;
    block.blocks = scope_;
    block.seqs = orders;
    block.movedFromBase = movedFromBase;
    modelSeconds_.resize(models_.size(), 0.0);
    for (size_t m = 0; m < models_.size(); ++m) {
        const auto t0 = std::chrono::steady_clock::now();
        models_[m]->apply(block, fidelity);
        modelSeconds_[m] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    for (const Sequence& seq : block.seqs) {
        for (const StinkyInstruction* inst : seq) {
            const int op = inst->getUnifiedOpcode();
            if (op == GFX::s_set_vgpr_msb) ++block.counts.bankSwitches;
            if (op == GFX::s_wait_alu) ++block.counts.waitAlus;
            if (op == GFX::s_nop || op == GFX::v_nop) ++block.counts.nops;
            if (op == GFX::flat_prefetch_b8 && pool_->owns(inst)) ++block.counts.bridges;
        }
    }
    return block;
}

}  // namespace stinkytofu::coissue
