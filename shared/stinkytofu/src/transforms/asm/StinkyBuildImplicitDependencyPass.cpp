/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
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
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

#define DEBUG_TYPE "StinkyBuildImplicitDependencyPass"

// Implicit dependency pass
// ========================
// Attaches implicit registers to instructions so that the def-use chain builder
// can see dependencies that are not encoded as explicit operands. Two kinds of
// implicit dependencies are handled:
//
// 1) Implicit special registers (SCC, VCC, EXEC) driven by HW flags
//    (Flags.def: IF_ImplicitRead/WriteSCC, IF_ImplicitReadVCC,
//     IF_ImplicitRead/WriteEXEC). The corresponding special register is added
//    to src/dest if not already present.
//
// 2) RegType::LDS pseudo-registers (keyed by MemTokenData token IDs). The
//    instruction type determines src vs dest placement:
//
//      tensor_load / ds_write                      →  LDS token to dest (LDS producer)
//      ds_read / global_store_async_from_lds_*      →  LDS token to src  (LDS consumer)
//      barrier / signal / wait                      →  LDS token to both (synchronization point)
//
//    The def-use chain builder then sees:
//      producer(def LDS[t]) → barrier(use+def LDS[t]) → consumer(use LDS[t])
//    which forces the scheduler to respect: producers → barrier → consumers.
//
//    Under the GIR frame contract an instruction carries no MemTokenData; the token is derived
//    here from GirFrameAnalysis. The token is the concrete storage id, one per (operand,
//    generation). Every later touch of that storage reuses it, in any frame. A GIR fence gets
//    the token of every hazard storage it encloses. The derived token then takes the normal
//    path below.

namespace {
using namespace stinkytofu;

static void addUniqueLdsDest(StinkyInstruction& inst, int tokenId) {
    for (const StinkyRegister& d : inst.getDestRegs())
        if (d.reg.type == RegType::LDS && d.reg.idx == static_cast<uint32_t>(tokenId)) return;
    inst.addDestReg(StinkyRegister(RegType::LDS, tokenId, 1));
}

static void addUniqueLdsSrc(StinkyInstruction& inst, int tokenId) {
    for (const StinkyRegister& s : inst.getSrcRegs())
        if (s.reg.type == RegType::LDS && s.reg.idx == static_cast<uint32_t>(tokenId)) return;
    inst.addSrcReg(StinkyRegister(RegType::LDS, tokenId, 1));
}

using GirSlotTokens = std::map<const StinkyInstruction*, std::set<int>>;

static const char* instMnemonic(const StinkyInstruction* inst) {
    if (!inst || !inst->getHwInstDesc() || !inst->getHwInstDesc()->mnemonic) return "?";
    return inst->getHwInstDesc()->mnemonic;
}

// Same generation GirFrameAnalysis feeds to concreteStorage.
static int occurrenceGeneration(const GirAccessSpec& access, const GirFrame& frame) {
    int generation = access.absoluteGeneration;
    if (generation < 0) generation = frame.phaseOf(access.genId) + access.gdelta;
    if (access.ring > 0) {
        generation %= access.ring;
        if (generation < 0) generation += access.ring;
    }
    return generation;
}

GirSlotTokens collectGirSlotTokens(Function& func, AnalysisManager& AM) {
    GirSlotTokens tokens;
    const auto& frames = AM.getResult<GirFrameAnalysis>(func);
    if (frames.empty()) return tokens;

    // Token id is the storage id. The first touch of a storage claims it; every later
    // touch of that storage, in any frame, reuses the same token.
    std::set<int> seenStorage;
    PASS_DEBUG(std::cerr << "[GirToken] occurrences=" << frames.occurrences.size() << "\n");
    for (const GirAccessOccurrence& occurrence : frames.occurrences) {
        const int tokenId = occurrence.storage;
        const bool shared = !seenStorage.insert(tokenId).second;
        tokens[occurrence.inst].insert(tokenId);
        const GirAccessSpec* access = occurrence.accessIndex < frames.contract.accesses.size()
                                          ? &frames.contract.accesses[occurrence.accessIndex]
                                          : nullptr;
        PASS_DEBUG(std::cerr << "[GirToken] " << instMnemonic(occurrence.inst)
                             << " bb=" << (occurrence.block ? occurrence.block->getLabel() : "?")
                             << " inst#" << occurrence.instructionIndex;
                   if (access) {
                       std::cerr << " action=" << access->actionId
                                 << (access->isWrite ? " write" : " read")
                                 << " operand=" << access->operand
                                 << " gen=" << occurrenceGeneration(*access, occurrence.frame);
                   } std::cerr
                   << " storage=" << occurrence.storage << (shared ? " slot=shared" : " slot=new")
                   << " token=" << tokenId << "\n";);
    }

    const auto& hazards = AM.getResult<GirFrameHazardAnalysis>(func);
    for (const GirFrameHazard& hazard : hazards.hazards) {
        for (const GirHazardFenceOwner& owner : enclosingFences(frames, hazard).owners) {
            std::vector<StinkyInstruction*> pieces{owner.anchor};
            if (const auto* action = owner.anchor->getModifier<GirActionData>()) {
                auto owned = frames.actionInstructions.find(action->actionId);
                if (owned != frames.actionInstructions.end()) pieces = owned->second;
            }
            const int tokenId = hazard.storage;
            PASS_DEBUG(std::cerr << "[GirToken] fence anchor=" << instMnemonic(owner.anchor)
                                 << " storage=" << hazard.storage << " token=" << tokenId << "\n");
            for (const StinkyInstruction* piece : pieces) tokens[piece].insert(tokenId);
        }
    }
    return tokens;
}

// "sync LDS0" or "sync LDS0, sync LDS1" starting at \p i. \p end is one past the run.
static bool matchSyncLdsRun(const std::string& text, size_t i, size_t& end) {
    auto one = [&](size_t at, size_t& out) {
        constexpr const char* kPrefix = "sync LDS";
        constexpr size_t kPrefixLen = 8;
        if (at + kPrefixLen > text.size() || text.compare(at, kPrefixLen, kPrefix) != 0)
            return false;
        size_t j = at + kPrefixLen;
        if (j >= text.size() || text[j] < '0' || text[j] > '9') return false;
        while (j < text.size() && text[j] >= '0' && text[j] <= '9') ++j;
        out = j;
        return true;
    };
    if (!one(i, end)) return false;
    for (;;) {
        size_t j = end;
        while (j < text.size() && text[j] == ' ') ++j;
        if (j >= text.size() || text[j] != ',') break;
        ++j;
        while (j < text.size() && text[j] == ' ') ++j;
        size_t next = 0;
        if (!one(j, next)) break;
        end = next;
    }
    return true;
}

// Keep the rest of the comment. A "sync LDS<n>" run is replaced by the GIR tokens;
// a comment with no such run gets the GIR text appended.
static void setGirSyncComment(StinkyInstruction& inst, const std::set<int>& tokenIds) {
    std::string gir;
    for (int tokenId : tokenIds) {
        if (!gir.empty()) gir += ", ";
        gir += "GIR to sync LDS";
        gir += std::to_string(tokenId);
    }

    auto* existing = inst.getModifier<CommentData>();
    if (!existing) {
        inst.addModifier<CommentData>(CommentData{gir});
        return;
    }

    std::string rewritten;
    bool replaced = false;
    for (size_t i = 0; i < existing->comment.size();) {
        size_t end = 0;
        if (matchSyncLdsRun(existing->comment, i, end)) {
            rewritten += gir;
            i = end;
            replaced = true;
        } else {
            rewritten.push_back(existing->comment[i]);
            ++i;
        }
    }
    if (!replaced) {
        if (!rewritten.empty() && rewritten.back() != ' ') rewritten.push_back(' ');
        rewritten += gir;
    }
    existing->comment = std::move(rewritten);
}

// The kinds the token branches below accept.
bool takesLdsToken(const StinkyInstruction& inst) {
    return isBarrier(inst) || inst.is(InstFlag::IF_WaitTensorCnt) || isTensorLoad(inst) ||
           isDSWrite(inst) || isDSRead(inst) || isGlobalStoreAsyncFromLds(inst);
}

// Barrier: LDS tokens to both src and dest.
// Returns false if this is a non-all-wave split barrier that should be skipped.
static bool processBarrier(StinkyInstruction& inst, const MemTokenData& mt,
                           [[maybe_unused]] const std::string& bbLabel) {
    if ((isBarrierSignal(inst) || isBarrierWait(inst)) && !isSplitBarrierAllWave(inst)) {
        PASS_DEBUG(std::cerr << "[BuildImplicitDep] non-all-wave split barrier BB label=\""
                             << bbLabel << "\" — skipping (workgroup-sync mode)\n");
        return false;
    }

    PASS_DEBUG(std::cerr << "[BuildImplicitDep] barrier BB label=\"" << bbLabel << "\" tokens=[";
               for (int t
                    : mt.tokens) std::cerr
               << t << " ";
               std::cerr << "]\n");

    for (int tokenId : mt.tokens) {
        addUniqueLdsDest(inst, tokenId);
        addUniqueLdsSrc(inst, tokenId);
    }
    return true;
}

// LDS writers (tensor_load, ds_write): LDS tokens to dest.
static void processLdsWriter(StinkyInstruction& inst, const MemTokenData& mt,
                             [[maybe_unused]] const std::string& bbLabel) {
    PASS_DEBUG(std::cerr << "[BuildImplicitDep] LDS writer ("
                         << (isTensorLoad(inst) ? "tensor_load" : "ds_write") << ") tokens=[";
               for (int t
                    : mt.tokens) std::cerr
               << t << " ";
               std::cerr << "] -> dest\n");

    for (int tokenId : mt.tokens) addUniqueLdsDest(inst, tokenId);
}

// LDS readers (ds_read, global_store_async_from_lds_*): LDS tokens to src.
static void processLdsReader(StinkyInstruction& inst, const MemTokenData& mt,
                             [[maybe_unused]] const std::string& bbLabel) {
    PASS_DEBUG(std::cerr << "[BuildImplicitDep] LDS reader ("
                         << (isDSRead(inst) ? "ds_read" : "global_store_async_from_lds")
                         << ") tokens=[";
               for (int t
                    : mt.tokens) std::cerr
               << t << " ";
               std::cerr << "] -> src\n");

    for (int tokenId : mt.tokens) addUniqueLdsSrc(inst, tokenId);
}

void setPseudoRegistersInBlock(BasicBlock& bb, PassContext& passCtx,
                               bool enableMemoryTokenDependencies, const GirSlotTokens& girTokens) {
    const uint32_t wavefrontSize = passCtx.getWavefrontSize();
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (!inst) continue;

        // Always attach implicit special registers (SCC/VCC/EXEC) declared by HW flags
        legalizeImplicitSpecialRegisters(inst, wavefrontSize);

        if (!enableMemoryTokenDependencies) continue;
        if (!inst->getModifier<MemTokenData>() && takesLdsToken(*inst)) {
            if (auto slots = girTokens.find(inst); slots != girTokens.end()) {
                inst->addModifier<MemTokenData>(
                    MemTokenData{std::vector<int>(slots->second.begin(), slots->second.end())});
                setGirSyncComment(*inst, slots->second);
            }
        }
        const MemTokenData* mt = inst->getModifier<MemTokenData>();
        if (!mt) continue;
        assert(!mt->tokens.empty() && "MemTokenData with empty tokens");

        // `s_wait_tensorcnt` is a synchronization point that cuts no scheduling region, so it is
        // the one the DAG cannot see without these pseudo-registers.
        if (isBarrier(*inst) || inst->is(InstFlag::IF_WaitTensorCnt))
            processBarrier(*inst, *mt, bb.getLabel());
        else if (isTensorLoad(*inst) || isDSWrite(*inst))
            processLdsWriter(*inst, *mt, bb.getLabel());
        else if (isDSRead(*inst) || isGlobalStoreAsyncFromLds(*inst))
            processLdsReader(*inst, *mt, bb.getLabel());
        else
            // StinkyWaitCntInsertionPass tags s_waitcnt with MemTokenData, which would hit this
            // assert — safe only because it runs strictly after this pass.
            assert(false &&
                   "instruction has MemTokenData but is not a barrier, fence, wait_tensorcnt, "
                   "tensor_load, ds_write, ds_read, or global_store_async_from_lds");
    }
}

class StinkyBuildImplicitDependencyPass : public StinkyInstPass {
   public:
    explicit StinkyBuildImplicitDependencyPass(bool enableMemoryTokenDependencies)
        : enableMemoryTokenDependencies(enableMemoryTokenDependencies) {}

    static char ID;

    const char* getName() const override {
        return "StinkyBuildImplicitDependencyPass";
    }

    PassID getPassID() const override {
        return &StinkyBuildImplicitDependencyPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        const GirSlotTokens girTokens = collectGirSlotTokens(func, AM);
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb))
                setPseudoRegistersInBlock(bb, passCtx, enableMemoryTokenDependencies, girTokens);
        }
        return preserveCFGAnalyses();
    }

   private:
    bool enableMemoryTokenDependencies;
};

char StinkyBuildImplicitDependencyPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyBuildImplicitDependencyPass(bool enableMemoryTokenDependencies) {
    return std::make_unique<StinkyBuildImplicitDependencyPass>(enableMemoryTokenDependencies);
}
}  // namespace stinkytofu
