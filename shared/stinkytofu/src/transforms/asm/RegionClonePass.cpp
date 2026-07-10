// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/RegionClonePass.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"

// Enable per-pass debug logging via PassManagerDebugConfig::addDebugOnly("RegionClonePass")
// or `StinkyTofuDebugPass: "RegionClonePass"` in YAML.
#define DEBUG_TYPE "RegionClonePass"

namespace {
using namespace stinkytofu;

// src2 (acc / src C) slot in WMMA/MFMA srcRegs. Update if MFMA layout changes.
constexpr size_t kAcc2SrcIdx = 2;

/// True for an MFMA whose src C (acc) is a real accumulator operand.
bool isMfmaWithAcc(const StinkyInstruction* inst) {
    return inst && inst->getModifier<MFMAModifiers>() && inst->getSrcRegs().size() > kAcc2SrcIdx;
}

/// True if bb's terminator branches to label (the loop back-edge to startLabel).
bool terminatorBranchesTo(BasicBlock* bb, const std::string& label) {
    auto* term = dyn_cast<StinkyInstruction>(bb->getTerminator());
    if (!term) return false;
    for (const auto& src : term->getSrcRegs()) {
        if (src.dataType == StinkyRegister::Type::LiteralString &&
            src.getLiteralString() == label) {
            return true;
        }
    }
    if (auto* ld = term->getModifier<LabelData>()) return ld->label == label;
    return false;
}

//----------------------------------------------------------------------
// Region discovery: collect startBBs, then compute the region end in-pass.
//----------------------------------------------------------------------

struct RegionRange {
    BasicBlock* startBB;
    BasicBlock* markerBB;
    StinkyInstruction* endInst;
};

/// Find every region matching the spec: collect BBs named startLabel, then
/// compute the end boundary as the last chain head — the last MFMA whose src C
/// (acc) first appears. The scan is bounded to the unrolled-loop body: it stops
/// at the back-edge (the terminator branching back to startLabel).
std::vector<RegionRange> findRegions(Function& func, const std::string& startLabel) {
    std::vector<RegionRange> out;

    std::vector<BasicBlock*> startBBs;
    for (BasicBlock& bb : func) {
        if (bb.getLabel() == startLabel) startBBs.push_back(&bb);
    }

    for (BasicBlock* startBB : startBBs) {
        std::unordered_set<StinkyRegister> seenAccs;
        BasicBlock* boundaryBB = nullptr;
        StinkyInstruction* boundaryInst = nullptr;
        for (BasicBlock* bb = startBB; bb; bb = bb->getNext()) {
            for (IRBase& node : *bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (!isMfmaWithAcc(inst)) continue;
                if (!seenAccs.insert(inst->getSrcRegs()[kAcc2SrcIdx]).second) continue;
                boundaryBB = bb;
                boundaryInst = inst;
            }
            if (terminatorBranchesTo(bb, startLabel)) break;
        }
        if (boundaryInst) out.push_back({startBB, boundaryBB, boundaryInst});
    }
    return out;
}

/// Collect BBs from startBB through endBB (inclusive) in physical order.
std::vector<BasicBlock*> collectRegionBBs(BasicBlock* startBB, BasicBlock* endBB) {
    std::vector<BasicBlock*> region;
    for (BasicBlock* bb = startBB; bb; bb = bb->getNext()) {
        region.push_back(bb);
        if (bb == endBB) return region;
    }
    return {};  // endBB not reachable from startBB; caller skips.
}

//----------------------------------------------------------------------
// BB / label / branch helpers
//----------------------------------------------------------------------

/// Split bb at splitPoint into (prefix, suffix); suffix inherits the
/// successors, prefix falls through to suffix.
std::pair<BasicBlock*, BasicBlock*> splitBBAt(Function& func, BasicBlock* bb,
                                              BasicBlock::iterator splitPoint,
                                              const std::string& newLabel) {
    BasicBlock* bb2 = func.createBasicBlockAfter(bb, newLabel);

    auto it = splitPoint;
    while (it != bb->end()) {
        IRBase* node = it.getNodePtr();
        auto next = std::next(it);
        bb->removeIR(node);
        bb2->appendIR(node);
        it = next;
    }

    // Transfer bb's successors to bb2, then add bb -> bb2 fall-through.
    const std::vector<BasicBlock*> oldSuccessors = bb->getSuccessors();
    func.removeSuccessorEdges(*bb);
    for (BasicBlock* succ : oldSuccessors) func.addEdge(bb2, succ);
    func.addEdge(bb, bb2);

    return {bb, bb2};
}

/// Emit a LABEL inst at the start of bb so AsmEmitter prints `labelName:` first.
void insertLabelAtStart(BasicBlock& bb, const std::string& labelName, GfxArchID archId) {
    AsmIRBuilder builder(bb, archId);
    StinkyInstruction* labelInst = builder.createLabel(labelName, /*alignment=*/1);
    if (bb.size() > 1) {
        bb.removeIR(labelInst);
        bb.insertIR(bb.begin(), labelInst);
    }
}

/// Append `s_branch targetLabel` at end of bb and connect the CFG edge.
void appendBranchTo(Function& func, BasicBlock& bb, BasicBlock* targetBB,
                    const std::string& targetLabel, GfxArchID archId) {
    AsmIRBuilder builder(bb, archId);
    StinkyInstruction* br = builder.create(getMCIDByUOp(GFX::s_branch, archId));
    br->addSrcReg(StinkyRegister(targetLabel));
    func.addEdge(&bb, targetBB);
}

//----------------------------------------------------------------------
// Region cloning
//----------------------------------------------------------------------

struct CloneResult {
    std::vector<BasicBlock*> clonedBBs;
    std::map<std::string, std::string> labelMap;  // origLabel -> clonedLabel
};

/// Build the cloned label name for an origin label.
std::string makeClonedLabel(const std::string& specName, const std::string& origLabel,
                            size_t jobIdx) {
    return "label_" + specName + "_" + origLabel + "_" + std::to_string(jobIdx);
}

/// Clone the region into ONE flat BB before insertBeforeBB. A multi-BB origin
/// (e.g. ClusterBarrier handshake) is flattened — safe because the clone runs
/// once and is never re-scheduled, and it avoids intra-clone edge maintenance.
/// Returns labelMap (origLabel -> clonedLabel) for the branch rewrite.
CloneResult cloneRegion(Function& func, const std::vector<BasicBlock*>& origBBs,
                        const std::string& specName, size_t jobIdx, BasicBlock* insertBeforeBB,
                        GfxArchID archId) {
    CloneResult res;
    // startBB must be named (we found it by name in findRegions).
    assert(!origBBs.empty() && !origBBs.front()->getLabel().empty() &&
           "region start BB must have a name");

    const std::string headerLabel = makeClonedLabel(specName, origBBs.front()->getLabel(), jobIdx);
    res.labelMap[origBBs.front()->getLabel()] = headerLabel;
    BasicBlock* clonedBB = func.createBasicBlockBefore(insertBeforeBB, headerLabel);
    insertLabelAtStart(*clonedBB, headerLabel, archId);

    AsmIRBuilder builder(*clonedBB, archId);
    for (size_t i = 0; i < origBBs.size(); ++i) {
        // BB[0]'s label is the header; later BBs' labels become internal
        // (renamed) labels so intra-region branches resolve.
        if (i > 0 && !origBBs[i]->getLabel().empty()) {
            const std::string cl = makeClonedLabel(specName, origBBs[i]->getLabel(), jobIdx);
            res.labelMap[origBBs[i]->getLabel()] = cl;
            builder.createLabel(cl, /*alignment=*/1);
        }
        for (IRBase& node : *origBBs[i]) {
            // Labels are re-emitted via getLabel() above; skip pseudo insts
            // (LABEL/PHI/FENCE) when copying the body.
            if (auto* inst = dyn_cast<StinkyInstruction>(&node)) {
                if (isPseudoInst(inst)) continue;
            }
            IRBase* cloned = node.clone();
            if (cloned) clonedBB->appendIR(cloned);
        }
    }
    res.clonedBBs.push_back(clonedBB);
    return res;
}

/// Redirect intra-region branches in the flat clone to the cloned labels
/// (escaping branches keep their targets). Scans every inst, not just the
/// terminator, because the flat clone can hold mid-block branches.
void rewriteInternalBranches(const std::vector<BasicBlock*>& clonedBBs,
                             const std::map<std::string, std::string>& labelMap) {
    for (BasicBlock* bb : clonedBBs) {
        for (IRBase& node : *bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            // Branch label may sit at any srcReg (e.g. SCC takes srcReg[0]).
            for (size_t s = 0; s < inst->getSrcRegs().size(); ++s) {
                const auto& src = inst->getSrcRegs()[s];
                if (src.dataType != StinkyRegister::Type::LiteralString) continue;
                auto it = labelMap.find(src.getLiteralString());
                if (it != labelMap.end()) {
                    inst->setSrcReg(s, StinkyRegister(it->second));
                }
            }
            // Some rocisa branches carry the label in a LabelData modifier.
            if (auto* ld = inst->getModifier<LabelData>()) {
                auto it = labelMap.find(ld->label);
                if (it != labelMap.end()) ld->label = it->second;
            }
        }
    }
}

/// Rewrite the terminator's label operand oldLabel -> newLabel in `bb`.
void rewriteLabelRefsInBB(BasicBlock& bb, const std::string& oldLabel,
                          const std::string& newLabel) {
    IRBase* term = bb.getTerminator();
    if (!term) return;
    auto* inst = dyn_cast<StinkyInstruction>(term);
    if (!inst) return;
    for (size_t s = 0; s < inst->getSrcRegs().size(); ++s) {
        const auto& src = inst->getSrcRegs()[s];
        if (src.dataType == StinkyRegister::Type::LiteralString &&
            src.getLiteralString() == oldLabel) {
            inst->setSrcReg(s, StinkyRegister(newLabel));
        }
    }
    if (auto* ld = inst->getModifier<LabelData>()) {
        if (ld->label == oldLabel) ld->label = newLabel;
    }
}

/// Redirect pre-region forward entries from origStartBB to firstClonedBB.
/// Loop-back branches (from BBs after origStartBB) are intentionally untouched.
void rerouteEntryPredToClonedRegion(Function& func, BasicBlock* origStartBB,
                                    BasicBlock* firstClonedBB, BasicBlock* prevPhysicalPred,
                                    const std::string& origStartLabel,
                                    const std::string& firstClonedLabel) {
    if (!prevPhysicalPred) return;

    for (BasicBlock* bb = prevPhysicalPred; bb; bb = bb->getPrev()) {
        rewriteLabelRefsInBB(*bb, origStartLabel, firstClonedLabel);
    }

    prevPhysicalPred->removeSuccessor(origStartBB);
    origStartBB->removePredecessor(prevPhysicalPred);
    func.addEdge(prevPhysicalPred, firstClonedBB);
}

//----------------------------------------------------------------------
// Per-spec post-clone transforms
//----------------------------------------------------------------------

/// InitCIterWmma: zero src C on each chain head (first MFMA whose src C acc is
/// seen); later writes to the same acc keep accumulating.
void initCIterWmma_zeroChainHeads(const std::vector<BasicBlock*>& clonedBBs) {
    std::unordered_set<StinkyRegister> seenAccs;
    for (BasicBlock* bb : clonedBBs) {
        for (IRBase& node : *bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!isMfmaWithAcc(inst)) continue;
            if (!seenAccs.insert(inst->getSrcRegs()[kAcc2SrcIdx]).second) continue;
            inst->setSrcReg(kAcc2SrcIdx, StinkyRegister(0));
        }
    }
}

using PostCloneFn = void (*)(const std::vector<BasicBlock*>&);

/// spec.name -> per-kind post-clone transform. Adding a new kind = add one entry.
PostCloneFn postCloneFor(const std::string& specName) {
    static const std::unordered_map<std::string, PostCloneFn> kRecipes = {
        {"InitCIterWmma", &initCIterWmma_zeroChainHeads},
    };
    auto it = kRecipes.find(specName);
    return it == kRecipes.end() ? nullptr : it->second;
}

//----------------------------------------------------------------------
// Per-region driver
//----------------------------------------------------------------------

/// Clone one region into a stage placed before it, then reroute pre-region
/// entries through the clone. Returns false if skipped (logged inline).
bool cloneOneRegion(Function& func, const CloneSpec& spec, size_t jobIdx, const RegionRange& region,
                    GfxArchID archId) {
    // 1. Re-derive boundaryBB (an earlier job's split may have moved this inst).
    BasicBlock* boundaryBB = region.endInst->getParent();
    if (!boundaryBB) return false;

    const std::string targetLabel = "label_" + spec.name + "_target_" + std::to_string(jobIdx);

    // 2. Split after the boundary; the suffix (targetLabel) is where the clone
    //    tail branches to (step 8), so first entry skips the origin region.
    BasicBlock::iterator splitPoint(region.endInst);
    ++splitPoint;
    auto [_unused, bb2] = splitBBAt(func, boundaryBB, splitPoint, targetLabel);
    (void)_unused;  // == boundaryBB
    insertLabelAtStart(*bb2, targetLabel, archId);

    // 3. Collect region BBs (startBB..boundaryBB).
    const auto origRegion = collectRegionBBs(region.startBB, boundaryBB);
    if (origRegion.empty()) {
        PASS_DEBUG(std::cerr << "  job " << jobIdx << " (" << spec.name
                             << "): empty region; skip\n");
        return false;
    }

    // 4. Snapshot pre-region predecessor + start label before cloning shifts
    //    physical order (getPrev would change).
    BasicBlock* prevPhysicalPred = region.startBB->getPrev();
    const std::string origStartLabel = region.startBB->getLabel();

    // 5. Clone the region into a flat BB placed before startBB.
    CloneResult cr = cloneRegion(func, origRegion, spec.name, jobIdx, region.startBB, archId);
    if (cr.clonedBBs.empty()) return false;

    // 6. Inside the clone, redirect intra-region branches to cloned labels.
    rewriteInternalBranches(cr.clonedBBs, cr.labelMap);

    // 7. Per-kind post-clone transform (e.g. InitCIterWmma zeroes chain heads).
    if (PostCloneFn fn = postCloneFor(spec.name)) {
        fn(cr.clonedBBs);
    }

    // 8. Branch the clone tail to targetLabel (skip origin on first entry).
    appendBranchTo(func, *cr.clonedBBs.back(), bb2, targetLabel, archId);

    // 9. Reroute pre-region forward entries to land in the clone.
    const std::string& firstClonedLabel = cr.labelMap.count(origStartLabel)
                                              ? cr.labelMap[origStartLabel]
                                              : cr.clonedBBs.front()->getLabel();
    rerouteEntryPredToClonedRegion(func, region.startBB, cr.clonedBBs.front(), prevPhysicalPred,
                                   origStartLabel, firstClonedLabel);

    PASS_DEBUG(std::cerr << "  job " << jobIdx << " (" << spec.name << "): cloned "
                         << origRegion.size() << " BB(s) (region from [" << origStartLabel
                         << "])\n");
    return true;
}

//----------------------------------------------------------------------
// Pass driver
//----------------------------------------------------------------------

class RegionClonePass : public StinkyInstPass {
   public:
    static char ID;

    explicit RegionClonePass(std::vector<CloneSpec> cloneList) : cloneList_(std::move(cloneList)) {}

    const char* getName() const override {
        return "RegionClonePass";
    }

    PassID getPassID() const override {
        return &RegionClonePass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        if (cloneList_.empty()) return preserveCFGAnalyses();

        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        bool mutated = false;
        size_t jobIdx = 0;
        for (const CloneSpec& spec : cloneList_) {
            const auto regions = findRegions(func, spec.startLabel);
            if (regions.empty()) continue;

            PASS_DEBUG(std::cerr << "[RegionClonePass] spec '" << spec.name << "' (from "
                                 << spec.startLabel << "): " << regions.size() << " region(s)\n");

            for (const auto& region : regions) {
                if (cloneOneRegion(func, spec, jobIdx++, region, archId)) {
                    mutated = true;
                }
            }
        }

        return mutated ? PreservedAnalyses::none() : preserveCFGAnalyses();
    }

   private:
    std::vector<CloneSpec> cloneList_;
};

char RegionClonePass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRegionClonePass(std::vector<CloneSpec> cloneList) {
    return std::make_unique<RegionClonePass>(std::move(cloneList));
}
}  // namespace stinkytofu
