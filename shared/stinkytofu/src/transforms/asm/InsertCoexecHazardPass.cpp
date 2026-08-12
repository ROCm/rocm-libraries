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

#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "InsertCoexecHazardPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

// Per-arch co-execution hazard rules. WMMA V_NOP counts come from each producer's
// coIssueWindow bitmask at runtime; only arch-level rules live here.
struct CoexecHazardConfig {
    // TRANS -> TRANS and TRANS -> XDL WMMA spacing.
    int transToNonCoreSide = 0;
    bool hwHandlesTransToCoreSide = false;
};

constexpr CoexecHazardConfig kGfx1250Config = {
    /*transToNonCoreSide=*/1,
    /*hwHandlesTransToCoreSide=*/true,
};

// Bounds the backward scan. Max count on gfx1250 is 9. 18 to match LLVM's MaxVALULookAhead.
constexpr int kMaxSlotBudget = 18;

enum class ProducerKind { WMMA, TRANS, DGEMM, PERM };

inline bool isDGEMMProducer(const StinkyInstruction& inst) {
    return (isMatrixInstruction(inst) && !isXDLWMMA(inst)) || isDPMACC(inst);
}

// What the consumer is looking for during a backward scan.
struct ConsumerCtx {
    ProducerKind kind;
    bool consumerIsWmma;  // only meaningful for kind == WMMA
    const StinkyInstruction* consumer;
};

inline int popcount16(uint16_t v) {
    return __builtin_popcount(static_cast<unsigned>(v));
}

// Only VALU-pipe ops fill a coexec slot (incl. transcendental, matrix, bare v_nop).
inline bool isSlotFiller(const StinkyInstruction& inst) {
    return isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst) ||
           inst.getUnifiedOpcode() == GFX::v_nop;
}

inline bool isCoexecutableVALU(const StinkyInstruction& inst) {
    return (isVectorALU(inst) || isTranscendental(inst)) && !isMatrixInstruction(inst);
}

// WMMA producer D feeds a WMMA consumer's A/B (or SWMMAC index). D->C
// (accumulation) is intentionally NOT a hazard.
bool wmmaToWmmaOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    const auto& srcs = cons.getSrcRegs();
    if (srcs.size() > 0 && d.isOverlap(srcs[0])) return true;                   // A
    if (srcs.size() > 1 && d.isOverlap(srcs[1])) return true;                   // B
    if (isSWMMA(cons) && srcs.size() > 2 && d.isOverlap(srcs[2])) return true;  // index
    return false;
}

// WMMA producer D vs a co-executable VALU consumer: RAW (D->src), WAW (D->dst),
// WAR (producer A/B, or SWMMAC index, -> consumer dst).
bool wmmaToValuOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    for (const StinkyRegister& s : cons.getSrcRegs())
        if (d.isOverlap(s)) return true;  // RAW
    for (const StinkyRegister& cd : cons.getDestRegs())
        if (d.isOverlap(cd)) return true;  // WAW
    // WAR: a later VALU overwrites a register the WMMA still reads. Producer
    // inputs are A (src0), B (src1), and for SWMMAC the index (src2).
    const auto& psrc = prod.getSrcRegs();
    const size_t nWar = isSWMMA(prod) ? 3 : 2;
    for (size_t i = 0; i < psrc.size() && i < nWar; ++i)
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (psrc[i].isOverlap(cd)) return true;  // WAR
    return false;
}

// TRANS producer vs consumer: RAW/WAW on producer dst, WAR on producer src.
bool transOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    for (const StinkyRegister& d : prod.getDestRegs()) {
        for (const StinkyRegister& s : cons.getSrcRegs())
            if (d.isOverlap(s)) return true;  // RAW
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (d.isOverlap(cd)) return true;  // WAW
    }
    for (const StinkyRegister& ps : prod.getSrcRegs())
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (ps.isOverlap(cd)) return true;  // WAR
    return false;
}

class InsertCoexecHazardPass : public StinkyInstPass {
   public:
    static char ID;
    InsertCoexecHazardPass() = default;

    const char* getName() const override {
        return "InsertCoexecHazardPass";
    }

    PassID getPassID() const override {
        return &InsertCoexecHazardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        setupArch(passCtx);
        if (!func.empty()) processFunction(func);
        return preserveCFGAnalyses();
    }

   private:
    void setupArch(PassContext& passCtx) {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId_ = getGfxArchID(arch[0], arch[1], arch[2]);
        config_ = kGfx1250Config;
        PASS_DEBUG(std::cerr << "[InsertCoexecHazard] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << "\n");
    }

    // V_NOPs a consumer needs behind a matched producer.
    int required(ProducerKind kind, int slots, bool consumerIsWmma) const {
        if (kind == ProducerKind::TRANS) return config_.transToNonCoreSide;
        // DGEMM/SGEMM -> WMMA: a single spacer.
        if (kind == ProducerKind::DGEMM) return 1;
        // Tensor-LUT (perm_pk16): coexec slots.
        if (kind == ProducerKind::PERM) return slots;
        // WMMA producer: +1 on WMMA->WMMA is the D-writeback/pre-read pipeline spacer.
        return consumerIsWmma ? slots + 1 : slots;
    }

    // Does `prod` match what `ctx` is scanning for?
    bool matches(const StinkyInstruction& prod, const ConsumerCtx& ctx) const {
        if (ctx.kind == ProducerKind::WMMA) {
            if (!isXDLWMMA(prod)) return false;
            return ctx.consumerIsWmma ? wmmaToWmmaOverlap(prod, *ctx.consumer)
                                      : wmmaToValuOverlap(prod, *ctx.consumer);
        }
        if (ctx.kind == ProducerKind::DGEMM) {
            if (!isDGEMMProducer(prod)) return false;
            // Same RAW/WAW/WAR shape as the TRANS overlap (producer dst vs
            // consumer src/dst, producer src vs consumer dst).
            return transOverlap(prod, *ctx.consumer);
        }
        if (ctx.kind == ProducerKind::PERM) {
            if (!isTensorLUT(prod)) return false;
            return transOverlap(prod, *ctx.consumer);
        }
        if (!isTranscendental(prod)) return false;
        return transOverlap(prod, *ctx.consumer);
    }

    static std::vector<StinkyInstruction*> realInsts(BasicBlock& bb) {
        std::vector<StinkyInstruction*> out;
        for (auto& node : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst && !isPseudoInst(inst)) out.push_back(inst);
        }
        return out;
    }

    // Backward scan returning the max shortfall over every matching producer across
    // all predecessor paths; memo prunes re-entries.
    int scanBack(BasicBlock& bb, const StinkyInstruction* startBefore, int accExisting,
                 const ConsumerCtx& ctx, std::unordered_map<const BasicBlock*, int>& minExisting) {
        // Memoize predecessor entries on fewest fillers; prune when this arrival can't widen the
        // shortfall.
        if (!startBefore) {
            auto it = minExisting.find(&bb);
            if (it != minExisting.end() && it->second <= accExisting) return INT_MIN;
            minExisting[&bb] = accExisting;
        }

        int best = INT_MIN;
        int existing = accExisting;

        const std::vector<StinkyInstruction*> insts = realInsts(bb);
        int start = static_cast<int>(insts.size());
        if (startBefore) {
            for (int i = 0; i < static_cast<int>(insts.size()); ++i)
                if (insts[i] == startBefore) {
                    start = i;
                    break;
                }
        }

        for (int i = start - 1; i >= 0; --i) {
            StinkyInstruction& inst = *insts[i];

            // A call is a hard boundary: do not scan across it.
            if (isCall(inst)) return best;

            if (matches(inst, ctx)) {
                const bool hasWindow =
                    ctx.kind == ProducerKind::WMMA || ctx.kind == ProducerKind::PERM;
                const int slots = hasWindow ? popcount16(inst.getHwInstDesc()->coIssueWindow) : 0;
                const int need = required(ctx.kind, slots, ctx.consumerIsWmma);
                best = std::max(best, need - existing);
            }

            // Only the nearest preceding XDL WMMA can hazard the consumer; an earlier one is
            // separated by this WMMA, so stop here.
            if (ctx.kind == ProducerKind::WMMA && isXDLWMMA(inst)) return best;

            if (isSlotFiller(inst)) ++existing;
            if (existing > kMaxSlotBudget) return best;
        }

        // Reached the top of the BB with budget to spare: continue into every
        // predecessor, taking the max shortfall across them.
        for (BasicBlock* pred : bb.getPredecessors())
            best = std::max(best,
                            scanBack(*pred, /*startBefore=*/nullptr, existing, ctx, minExisting));
        return best;
    }

    void processFunction(Function& func) {
        // v_nops are stripped upstream by StinkyRemoveNopPass; this pass counts the
        // remaining (deliberate, scheduler-placed) fillers and tops up the shortfall.
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) {
                    ++it;
                    continue;
                }

                int toInsert = 0;
                if (isXDLWMMA(*inst)) {
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::WMMA,
                                                            /*consumerIsWmma=*/true));
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::TRANS,
                                                            /*consumerIsWmma=*/false));
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::DGEMM,
                                                            /*consumerIsWmma=*/true));
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::PERM,
                                                            /*consumerIsWmma=*/false));
                } else if (isCoexecutableVALU(*inst)) {
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::WMMA,
                                                            /*consumerIsWmma=*/false));
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::PERM,
                                                            /*consumerIsWmma=*/false));
                    // TRANS -> core/side is HW-handled; only a TRANS consumer needs
                    // the TRANS -> TRANS spacing.
                    if (isTranscendental(*inst))
                        toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::TRANS,
                                                                /*consumerIsWmma=*/false));
                }

                if (toInsert > 0) {
                    insertVNops(bb, it.getNodePtr(), toInsert);
                    PASS_DEBUG(std::cerr << "[InsertCoexecHazard]   inserted " << toInsert
                                         << " v_nop before " << inst->getHwInstDesc()->mnemonic
                                         << " in bb \"" << bb.getLabel() << "\"\n");
                }
                ++it;
            }
        }
    }

    int hazardFor(BasicBlock& bb, const StinkyInstruction& consumer, ProducerKind kind,
                  bool consumerIsWmma) {
        ConsumerCtx ctx{kind, consumerIsWmma, &consumer};
        // minExisting bounds re-entries: a block is re-scanned only on a strictly smaller filler
        // count.
        std::unordered_map<const BasicBlock*, int> minExisting;
        const int r = scanBack(bb, /*startBefore=*/&consumer, /*accExisting=*/0, ctx, minExisting);
        return r > 0 ? r : 0;
    }

    void insertVNops(BasicBlock& bb, IRBase* insertBefore, int n) {
        AsmIRBuilder builder(bb, archId_);
        for (int i = 0; i < n; ++i) builder.create(getMCIDByUOp(GFX::v_nop, archId_), insertBefore);
    }

    GfxArchID archId_ = GfxArchID{};
    CoexecHazardConfig config_;
};

char InsertCoexecHazardPass::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertCoexecHazardPass() {
    return std::make_unique<InsertCoexecHazardPass>();
}
}  // namespace stinkytofu
