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
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
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

/// InsertClusterBarrierPass names wave 0's head `<start>_CBWave0` and points
/// the latch there. The dispatch label itself is no longer the back-edge target.
bool terminatorClosesLoop(BasicBlock* bb, const std::string& startLabel) {
    return terminatorBranchesTo(bb, startLabel) ||
           terminatorBranchesTo(bb, startLabel + "_CBWave0");
}

constexpr int kClusterBarrierId = -3;
constexpr const char* kWaveIdxSymbol = "sgprWaveIdx";
constexpr const char* kSkipLabelPrefix = "label_skipCBPreSignal_";
constexpr const char* kWave0HeadSuffix = "_CBWave0";

StinkyRegister makeSymbolicSgpr(const std::string& symbolicName) {
    StinkyRegister reg(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
    reg.setSymbolicName(symbolicName);
    return reg;
}

bool isClusterSignalInst(const StinkyInstruction& inst) {
    if (!isBarrierSignal(inst)) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
           srcs[0].getLiteralInt() == kClusterBarrierId;
}

/// The entrance block of a wave-split loop: compare, branch to the other copy,
/// and the next block is the wave-0 head. It must not be part of the init clone,
/// or a non-zero wave would leave the zero-acc prefix immediately.
bool isWaveDispatchBlock(BasicBlock* bb, BasicBlock* next) {
    if (bb == nullptr || next == nullptr) return false;
    if (next->getLabel() != bb->getLabel() + kWave0HeadSuffix) return false;
    auto* term = dyn_cast<StinkyInstruction>(bb->getTerminator());
    if (term == nullptr || !isBranch(*term)) return false;
    return getBranchTarget(*term).rfind(kSkipLabelPrefix, 0) == 0;
}

BasicBlock* findBlockByLabel(Function& func, const std::string& label) {
    for (BasicBlock& block : func) {
        if (block.getLabel() == label) return &block;
    }
    return nullptr;
}

/// Real instructions, skipping the bare cluster signal that only wave 0 has,
/// so a wave-0 prefix and its copy stay in step.
bool countsForWaveAlign(const StinkyInstruction& inst) {
    if (isPseudoInst(&inst)) return false;
    return !isClusterSignalInst(inst);
}

/// Index of \p marker in the counted stream, or of the next counted instruction
/// when \p marker itself is a wave-0-only signal.
int continuationAlignIndex(BasicBlock* start, const StinkyInstruction* marker) {
    bool seen = false;
    int index = 0;
    for (BasicBlock* bb = start; bb != nullptr; bb = bb->getNext()) {
        for (IRBase& node : *bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == nullptr) continue;
            if (inst == marker) seen = true;
            if (!countsForWaveAlign(*inst)) continue;
            if (seen) return index;
            ++index;
        }
    }
    return -1;
}

StinkyInstruction* alignInstAt(BasicBlock* start, int index) {
    int seen = 0;
    for (BasicBlock* bb = start; bb != nullptr; bb = bb->getNext()) {
        for (IRBase& node : *bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == nullptr || !countsForWaveAlign(*inst)) continue;
            if (seen == index) return inst;
            ++seen;
        }
    }
    return nullptr;
}

/// Wrap each bare `s_barrier_signal -3` so the shared init clone, which every
/// wave executes, still lets only wave 0 post the signal.
bool gateBareClusterSignals(const std::vector<BasicBlock*>& blocks, GfxArchID archId,
                            int& gateSerial) {
    std::vector<StinkyInstruction*> bare;
    for (BasicBlock* bb : blocks) {
        StinkyInstruction* prevReal = nullptr;
        for (IRBase& node : *bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == nullptr || isPseudoInst(inst)) continue;
            if (isClusterSignalInst(*inst)) {
                const bool gated = prevReal != nullptr &&
                                   prevReal->getUnifiedOpcode() == GFX::s_cbranch_scc0 &&
                                   getBranchTarget(*prevReal).rfind(kSkipLabelPrefix, 0) == 0;
                if (!gated) bare.push_back(inst);
            }
            prevReal = inst;
        }
    }
    if (bare.empty()) return false;

    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    const HwInstDesc* cmpDesc = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
    const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc0, archId);
    if (cmpDesc == nullptr || brDesc == nullptr) return false;

    for (StinkyInstruction* signal : bare) {
        BasicBlock* parent = signal->getParent();
        if (parent == nullptr) continue;
        const std::string skip =
            std::string(kSkipLabelPrefix) + "init" + std::to_string(gateSerial++);
        AsmIRBuilder builder(*parent, archId);
        auto nextIt = std::next(BasicBlock::iterator(signal));
        IRBase* after = (nextIt == parent->end()) ? nullptr : nextIt.getNodePtr();
        StinkyInstruction* skipLbl =
            (after != nullptr) ? builder.create(&labelMCID, after) : builder.create(&labelMCID);
        skipLbl->addModifier<LabelData>(LabelData{skip, /*alignment=*/1});

        StinkyInstruction* brInst = builder.create(brDesc, signal);
        brInst->addSrcReg(StinkyRegister(skip));
        brInst->addModifier<LabelData>(LabelData{skip});
        brInst->addModifier<CommentData>(
            CommentData{"Execute cluster barrier signal for waveID 0"});

        StinkyInstruction* cmpInst = builder.create(cmpDesc, brInst);
        cmpInst->addDestReg(StinkyRegister::getSCCRegister());
        cmpInst->addSrcReg(makeSymbolicSgpr(kWaveIdxSymbol));
        cmpInst->addSrcReg(StinkyRegister(0));
        cmpInst->addModifier<CommentData>(CommentData{"Check for waveID 0"});
    }
    return true;
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
            // The entrance compare is not part of the cloned body.
            if (isWaveDispatchBlock(bb, bb->getNext())) {
                if (terminatorClosesLoop(bb, startLabel)) break;
                continue;
            }
            for (IRBase& node : *bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (!isMfmaWithAcc(inst)) continue;
                if (!seenAccs.insert(inst->getSrcRegs()[kAcc2SrcIdx]).second) continue;
                boundaryBB = bb;
                boundaryInst = inst;
            }
            if (terminatorClosesLoop(bb, startLabel)) break;
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
                    GfxArchID archId, bool& insertedSymbolic, int& gateSerial) {
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

    // 3. Collect region BBs (startBB..boundaryBB). A wave-split entrance is only
    // the compare/branch, so the clone starts at the wave-0 body. Leaving the
    // branch in the clone would send every non-zero wave out of the zero-acc
    // prefix.
    auto origRegion = collectRegionBBs(region.startBB, boundaryBB);
    BasicBlock* wave0BB = nullptr;
    BasicBlock* loop1BB = nullptr;
    if (origRegion.size() >= 2 && isWaveDispatchBlock(origRegion[0], origRegion[1])) {
        wave0BB = origRegion[1];
        if (auto* term = dyn_cast<StinkyInstruction>(origRegion[0]->getTerminator()))
            loop1BB = findBlockByLabel(func, getBranchTarget(*term));
        origRegion.erase(origRegion.begin());
    }
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

    // The shared init clone runs for every wave. A bare signal copied from the
    // wave-0 body has to grow its check back, and the tail has to send non-zero
    // waves into the other copy instead of back into wave 0.
    if (gateBareClusterSignals(cr.clonedBBs, archId, gateSerial)) insertedSymbolic = true;

    std::string waveNzTail;
    if (wave0BB != nullptr && loop1BB != nullptr) {
        StinkyInstruction* continuation = nullptr;
        for (IRBase& node : *bb2) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst != nullptr && !isPseudoInst(inst)) {
                continuation = inst;
                break;
            }
        }
        if (continuation == nullptr && bb2->getNext() != nullptr) {
            for (IRBase& node : *bb2->getNext()) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (inst != nullptr && !isPseudoInst(inst)) {
                    continuation = inst;
                    break;
                }
            }
        }
        const int index =
            (continuation != nullptr) ? continuationAlignIndex(wave0BB, continuation) : -1;
        StinkyInstruction* parallel = (index >= 0) ? alignInstAt(loop1BB, index) : nullptr;
        if (parallel != nullptr && parallel->getParent() != nullptr) {
            waveNzTail = targetLabel + "_waveNz";
            AsmIRBuilder nzBuilder(*parallel->getParent(), archId);
            static const HwInstDesc labelMCID{
                GFX::LABEL, GFX::LABEL, 0,       0,
                0,          0,          "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
            StinkyInstruction* nzLbl = nzBuilder.create(&labelMCID, parallel);
            nzLbl->addModifier<LabelData>(LabelData{waveNzTail, /*alignment=*/1});
            // The label sits mid-block until the next CFG build. Point the edge at
            // the block that contains it so the branch is not a dangling target.
            func.addEdge(cr.clonedBBs.back(), parallel->getParent());
        }
    }

    // 8. Branch the clone tail to targetLabel (skip origin on first entry).
    //    A wave-split loop sends wave 0 there and every other wave to the
    //    matching point in the copied body.
    BasicBlock& cloneTail = *cr.clonedBBs.back();
    if (!waveNzTail.empty()) {
        AsmIRBuilder tailBuilder(cloneTail, archId);
        const HwInstDesc* cmpDesc = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
        const HwInstDesc* brDesc = getMCIDByUOp(GFX::s_cbranch_scc0, archId);
        const HwInstDesc* jumpDesc = getMCIDByUOp(GFX::s_branch, archId);
        if (cmpDesc != nullptr && brDesc != nullptr && jumpDesc != nullptr) {
            StinkyInstruction* cmpInst = tailBuilder.create(cmpDesc);
            cmpInst->addDestReg(StinkyRegister::getSCCRegister());
            cmpInst->addSrcReg(makeSymbolicSgpr(kWaveIdxSymbol));
            cmpInst->addSrcReg(StinkyRegister(0));
            cmpInst->addModifier<CommentData>(CommentData{"Check for waveID 0"});

            StinkyInstruction* brInst = tailBuilder.create(brDesc);
            brInst->addSrcReg(StinkyRegister(waveNzTail));
            brInst->addModifier<LabelData>(LabelData{waveNzTail});
            brInst->addModifier<CommentData>(
                CommentData{"Execute cluster barrier signal for waveID 0"});

            StinkyInstruction* jump = tailBuilder.create(jumpDesc);
            jump->addSrcReg(StinkyRegister(targetLabel));
            jump->addModifier<LabelData>(LabelData{targetLabel});
            func.addEdge(&cloneTail, bb2);
            insertedSymbolic = true;
        } else {
            appendBranchTo(func, cloneTail, bb2, targetLabel, archId);
        }
    } else {
        appendBranchTo(func, cloneTail, bb2, targetLabel, archId);
    }

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
        bool insertedSymbolic = false;
        // Per-run, not process-lifetime: a static serial survives into the next
        // kernel compiled in this process and makes otherwise identical emits differ.
        int gateSerial = 0;
        size_t jobIdx = 0;
        for (const CloneSpec& spec : cloneList_) {
            const auto regions = findRegions(func, spec.startLabel);
            if (regions.empty()) continue;

            PASS_DEBUG(std::cerr << "[RegionClonePass] spec '" << spec.name << "' (from "
                                 << spec.startLabel << "): " << regions.size() << " region(s)\n");

            for (const auto& region : regions) {
                if (cloneOneRegion(func, spec, jobIdx++, region, archId, insertedSymbolic,
                                   gateSerial)) {
                    mutated = true;
                }
            }
        }

        if (insertedSymbolic) {
            std::vector<SymbolicOperandFix> fixes;
            resolveSymbolicOperands(func, fixes);
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
