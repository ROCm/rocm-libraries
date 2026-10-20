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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
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
//    here from one coherent path through the GirFrame graph. Its concrete storage id is the
//    scheduling token; predecessor-to-successor phase transitions preserve unrolled LDS rotation,
//    while other dynamic paths remain exclusively in GIR hazard/waitcnt analysis. A GIR fence gets
//    only selected-path endpoint tokens used in its own scheduling block.

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

const char* girHazardKindName(GirHazardKind kind) {
    switch (kind) {
        case GirHazardKind::RAW:
            return "RAW";
        case GirHazardKind::WAR:
            return "WAR";
        case GirHazardKind::WAW:
            return "WAW";
    }
    return "???";
}

// "gen0=0 gen5=1", the phase vector that names one frame.
std::string girFrameText(const GirFrame& frame) {
    std::string text;
    for (const auto& [gen, phase] : frame.phases) {
        if (!text.empty()) text += " ";
        text += "gen" + std::to_string(gen) + "=" + std::to_string(phase);
    }
    return text.empty() ? "-" : text;
}

// "ds_load_b128 label_LoopBeginL#10"
std::string girWhere(const StinkyInstruction* inst, const BasicBlock* block, size_t index);

size_t girIndexInBlock(const StinkyInstruction* inst) {
    size_t index = 0;
    if (!inst || !inst->getParent()) return index;
    for (IRBase& node : *const_cast<BasicBlock*>(inst->getParent())) {
        auto* here = dyn_cast<StinkyInstruction>(&node);
        if (!here) continue;
        if (here == inst) break;
        ++index;
    }
    return index;
}

std::string girWhere(const StinkyInstruction* inst, const BasicBlock* block, size_t index) {
    const char* mnemonic = inst && inst->getHwInstDesc() && inst->getHwInstDesc()->mnemonic
                               ? inst->getHwInstDesc()->mnemonic
                               : "?";
    return std::string(mnemonic) + " " + (block ? block->getLabel() : "?") + "#" +
           std::to_string(index);
}

using GirSlotTokens = std::map<const StinkyInstruction*, std::set<int>>;
using GirFencePieces = std::map<const StinkyInstruction*, std::vector<StinkyInstruction*>>;

// A materialized split fence has no GIR action tying its two physical instructions together.
// Index both halves so a hazard owned by either one can assign its complete token bridge to the
// pair directly.
GirFencePieces collectGirFencePieces(Function& func) {
    GirFencePieces pieces;
    for (BasicBlock& block : func) {
        StinkyInstruction* pendingSignal = nullptr;
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            if (isBarrierSignal(*inst) && isSplitBarrierAllWave(*inst)) {
                pendingSignal = inst;
                continue;
            }
            if (isBarrierWait(*inst) && isSplitBarrierAllWave(*inst) && pendingSignal) {
                std::vector<StinkyInstruction*> pair{pendingSignal, inst};
                pieces[pendingSignal] = pair;
                pieces[inst] = std::move(pair);
                pendingSignal = nullptr;
                continue;
            }
            if (isBarrier(*inst)) pendingSignal = nullptr;
        }
    }
    return pieces;
}

using GirSchedulingFrames = std::map<const BasicBlock*, GirFrame>;

// Select one coherent path through the frame graph. Picking each block's smallest phase
// independently resets every unrolled loop-copy block to phase zero and destroys the intended LDS
// rotation. A selected predecessor instead hands its exact successor frame to the next block.
GirSchedulingFrames collectGirSchedulingFrames(Function& func,
                                               const GirFrameAnalysis::Result& frames) {
    std::map<const BasicBlock*, std::vector<GirFrameNode>> candidates;
    const auto remember = [&](const GirFrameNode& node) {
        auto& list = candidates[node.block];
        if (std::find(list.begin(), list.end(), node) == list.end()) list.push_back(node);
    };
    for (const auto& [node, successors] : frames.edges) {
        remember(node);
        for (const GirFrameNode& successor : successors) remember(successor);
    }
    for (auto& [block, nodes] : candidates) {
        (void)block;
        std::sort(nodes.begin(), nodes.end(), [](const GirFrameNode& lhs, const GirFrameNode& rhs) {
            if (!(lhs.frame == rhs.frame)) return lhs.frame < rhs.frame;
            return lhs.incomingAction < rhs.incomingAction;
        });
    }

    GirSchedulingFrames selected;
    std::deque<GirFrameNode> work;
    const auto select = [&](const GirFrameNode& node) {
        if (!node.block || selected.contains(node.block)) return false;
        selected.emplace(node.block, node.frame);
        work.push_back(node);
        return true;
    };
    const auto propagate = [&] {
        while (!work.empty()) {
            const GirFrameNode node = work.front();
            work.pop_front();
            auto edges = frames.edges.find(node);
            if (edges == frames.edges.end()) continue;
            for (const GirFrameNode& successor : edges->second) select(successor);
        }
    };

    // Function order makes the entry path primary and gives mutually-exclusive, still-unselected
    // arms a deterministic feasible fallback.
    for (const BasicBlock& block : func) {
        if (selected.contains(&block)) continue;
        auto available = candidates.find(&block);
        if (available == candidates.end() || available->second.empty()) continue;
        select(available->second.front());
        propagate();
    }
    return selected;
}

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

    // Memory tokens constrain one static scheduling instance, not every dynamic frame. Map each
    // static access to the concrete storage on one coherent frame-graph path. The full graph
    // remains the authority for hazard and waitcnt analysis.
    const GirSchedulingFrames schedulingFrames = collectGirSchedulingFrames(func, frames);

    using StaticAccessKey = std::pair<const StinkyInstruction*, size_t>;
    using EndpointKey = std::tuple<const StinkyInstruction*, GirFrame, int>;
    std::map<StaticAccessKey, int> schedulingTokens;
    std::map<EndpointKey, std::set<size_t>> endpointAccesses;

    PASS_DEBUG(std::cerr << "[GirToken] occurrences=" << frames.occurrences.size() << "\n");
    for (const GirAccessOccurrence& occurrence : frames.occurrences) {
        endpointAccesses[EndpointKey{occurrence.inst, occurrence.frame, occurrence.storage}].insert(
            occurrence.accessIndex);

        auto selected = schedulingFrames.find(occurrence.block);
        if (selected == schedulingFrames.end() || !(occurrence.frame == selected->second)) continue;

        const StaticAccessKey accessKey{occurrence.inst, occurrence.accessIndex};
        auto [scheduled, fresh] = schedulingTokens.emplace(accessKey, occurrence.storage);
        if (!fresh && scheduled->second != occurrence.storage)
            STINKY_UNREACHABLE(
                "StinkyBuildImplicitDependencyPass: one static access maps to multiple "
                "single-frame tokens");
        tokens[occurrence.inst].insert(occurrence.storage);
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
                   << " storage=" << occurrence.storage
                   << " schedulingFrame=1 token=" << occurrence.storage << "\n";);
    }

    const auto& hazards = AM.getResult<GirFrameHazardAnalysis>(func);
    const GirFencePieces splitFencePieces = collectGirFencePieces(func);
    const auto tokensAtEndpoint = [&](const StinkyInstruction* inst, const GirFrame& frame,
                                      int storage) {
        auto access = endpointAccesses.find(EndpointKey{inst, frame, storage});
        if (access == endpointAccesses.end())
            STINKY_UNREACHABLE(
                "StinkyBuildImplicitDependencyPass: GIR hazard endpoint has no static access");
        std::set<int> result;
        for (size_t accessIndex : access->second) {
            auto token = schedulingTokens.find(StaticAccessKey{inst, accessIndex});
            if (token == schedulingTokens.end())
                STINKY_UNREACHABLE(
                    "StinkyBuildImplicitDependencyPass: GIR hazard endpoint has no single-frame "
                    "memory token");
            result.insert(token->second);
        }
        return result;
    };
    // Placement recorded what each fence covers and on which frame. Scheduling constrains ONE
    // path, so a fence takes the slots it covers on its block's selected frame and no others.
    const auto* placed = func.getStructMetaData<GirFenceCoverage>(kGirFenceCoverageKey);
    std::set<const StinkyInstruction*> answered;
    if (placed)
        for (const auto& [fence, byFrame] : placed->fences) {
            auto selected = schedulingFrames.find(fence->getParent());
            if (selected == schedulingFrames.end()) continue;
            auto covered = byFrame.find(selected->second);
            if (covered == byFrame.end() || covered->second.empty()) continue;
            std::vector<StinkyInstruction*> pieces{const_cast<StinkyInstruction*>(fence)};
            if (auto physical = splitFencePieces.find(fence); physical != splitFencePieces.end())
                pieces = physical->second;
            for (const StinkyInstruction* piece : pieces) {
                answered.insert(piece);
                for (const GirFrameHazard& hazard : covered->second)
                    if (hazard.storage >= 0) tokens[piece].insert(hazard.storage);
            }
            PASS_DEBUG({
                const char* mnemonic = fence->getHwInstDesc() && fence->getHwInstDesc()->mnemonic
                                           ? fence->getHwInstDesc()->mnemonic
                                           : "fence";
                std::cerr << "\n[GirCover] " << mnemonic << " at " << fence->getParent()->getLabel()
                          << " #" << girIndexInBlock(fence) << "  (" << byFrame.size()
                          << (byFrame.size() == 1 ? " frame)\n" : " frames)\n");
                for (const auto& [frame, hz] : byFrame) {
                    const bool isSelected = frame == selected->second;
                    std::vector<GirFrameHazard> sorted(hz.begin(), hz.end());
                    std::sort(sorted.begin(), sorted.end(),
                              [](const GirFrameHazard& a, const GirFrameHazard& b) {
                                  return std::tie(a.kind, a.storage) < std::tie(b.kind, b.storage);
                              });
                    std::cerr << (isSelected ? "  -> " : "     ") << girFrameText(frame)
                              << (isSelected ? "   [scheduled]\n" : "\n");
                    for (const GirFrameHazard& h : sorted)
                        std::cerr << "       " << girHazardKindName(h.kind) << " LDS" << h.storage
                                  << "  " << girWhere(h.producer, h.producerBlock, h.producerIndex)
                                  << " -> "
                                  << girWhere(h.consumer, h.consumerBlock, h.consumerIndex)
                                  << (h.gap == 0 ? "  (same node)\n"
                                                 : "  (" + std::to_string(h.gap) + " nodes on)\n");
                }
            });
        }

    const GirPlanFenceCoverage plan = buildGirPlanFenceCoverage(frames, hazards);
    for (size_t h = 0; h < hazards.hazards.size(); ++h) {
        const GirFrameHazard& hazard = hazards.hazards[h];
        for (const GirHazardFenceOwner& owner : plan.byHazard[h].owners) {
            BasicBlock* anchorBlock = owner.anchor ? owner.anchor->getParent() : nullptr;
            if (!anchorBlock) continue;
            if (answered.count(owner.anchor)) continue;

            std::vector<StinkyInstruction*> pieces{owner.anchor};
            if (auto physical = splitFencePieces.find(owner.anchor);
                physical != splitFencePieces.end()) {
                pieces = physical->second;
            } else if (const auto* action = owner.anchor->getModifier<GirActionData>()) {
                auto owned = frames.actionInstructions.find(action->actionId);
                if (owned != frames.actionInstructions.end()) pieces = owned->second;
            }
            std::set<int> bridge;
            // The scheduler never moves instructions across blocks. Keep only endpoint tokens that
            // can form a dependency in this barrier's block; dynamic frame occurrences collapse
            // onto the one selected-path storage of that static endpoint.
            if (hazard.producerBlock == anchorBlock) {
                const std::set<int> producer =
                    tokensAtEndpoint(hazard.producer, hazard.producerFrame, hazard.storage);
                bridge.insert(producer.begin(), producer.end());
            }
            if (hazard.consumerBlock == anchorBlock) {
                const std::set<int> consumer =
                    tokensAtEndpoint(hazard.consumer, hazard.consumerFrame, hazard.storage);
                bridge.insert(consumer.begin(), consumer.end());
            }
            if (bridge.empty()) continue;
            PASS_DEBUG({
                std::cerr << "[GirToken] fence anchor=" << instMnemonic(owner.anchor)
                          << " storage=" << hazard.storage << " schedulingTokens=";
                for (int token : bridge) std::cerr << token << ",";
                std::cerr << "\n";
            });
            for (const StinkyInstruction* piece : pieces) {
                tokens[piece].insert(bridge.begin(), bridge.end());
            }
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
