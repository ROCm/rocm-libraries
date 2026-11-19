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
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "InsertCoexecHazardPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace {
using namespace stinkytofu;

// Per-arch co-execution hazard rules now live in HWModel::Coexec (see
// stinkytofu/hardware/HWModel.hpp), reached via passCtx.getHWModel(). WMMA V_NOP
// counts still come from each producer's coIssueWindow bitmask at runtime.

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

//===----------------------------------------------------------------------===//
// In-place WMMA accumulation (gfx1250). A WMMA that takes the previous WMMA's D
// through the C fast-forward path must write a different accumulator, unless its
// type allows the same one under InPlaceRule. "Previous" means the instruction
// right before: any instruction in between breaks the chain. A broken rule stops
// codegen: switching the accumulator needs a free register range that this pass
// cannot pick.
//===----------------------------------------------------------------------===//

enum class InPlaceRule {
    Forbidden,
    DifferentAB,  // K=128 f8xf8, f8xf4, f4xf8, f6xf4, f4xf6
    Paired,       // f16, bf16, iu8, 16x16x128 f4xf4: WMMAs pair up, one 2xK tile each
    Allowed,      // 32x16x128 f4
};

enum class SmallFloat { F8, F6, F4 };

SmallFloat smallFloat(MatrixFmt fmt) {
    if (fmt == MatrixFmt::FP6 || fmt == MatrixFmt::BF6) return SmallFloat::F6;
    if (fmt == MatrixFmt::FP4) return SmallFloat::F4;
    return SmallFloat::F8;  // NONE selects FP8
}

InPlaceRule inPlaceRule(const StinkyInstruction& wmma) {
    const std::string_view m(wmma.getHwInstDesc()->mnemonic);
    if (isSWMMA(wmma)) return InPlaceRule::Forbidden;
    if (m.ends_with("_32x16x128_f4")) return InPlaceRule::Allowed;
    if (m.ends_with("_16x16x128_f8f6f4")) {
        const auto* fmt = wmma.getModifier<MatrixFmtModifiers>();
        const SmallFloat a = fmt ? smallFloat(fmt->fmtA) : SmallFloat::F8;
        const SmallFloat b = fmt ? smallFloat(fmt->fmtB) : SmallFloat::F8;
        if (a == SmallFloat::F4 && b == SmallFloat::F4) return InPlaceRule::Paired;
        const bool hasF6 = a == SmallFloat::F6 || b == SmallFloat::F6;
        const bool hasF4 = a == SmallFloat::F4 || b == SmallFloat::F4;
        return hasF6 && !hasF4 ? InPlaceRule::Forbidden : InPlaceRule::DifferentAB;
    }
    if (m.find("_16x16x128_") != std::string_view::npos)  // fp8/bf8 x fp8/bf8
        return InPlaceRule::DifferentAB;
    if (m.ends_with("_f16") || m.ends_with("_bf16") || m.ends_with("_iu8"))
        return InPlaceRule::Paired;
    return InPlaceRule::Forbidden;
}

// `cons` takes `prod`'s D as C and writes the same D. SWMMAC's C is its D.
bool accumulatesInPlace(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty() || cons.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    if (!(d == cons.getDestRegs()[0])) return false;
    if (isSWMMA(cons)) return true;
    const auto& srcs = cons.getSrcRegs();
    return srcs.size() > 2 && d == srcs[2];
}

bool abOverlap(const StinkyInstruction& wmma) {
    const auto& s = wmma.getSrcRegs();
    return s.size() > 1 && s[0].isOverlap(s[1]);
}

bool differentAB(const StinkyInstruction& x, const StinkyInstruction& y) {
    const auto& xs = x.getSrcRegs();
    const auto& ys = y.getSrcRegs();
    return xs.size() > 1 && ys.size() > 1 && !xs[0].isOverlap(ys[0]) && !xs[1].isOverlap(ys[1]);
}

// The first real instruction from `node` on, walking in one direction.
const StinkyInstruction* firstInst(IRBase* node, bool forward) {
    for (; node; node = forward ? node->getNext() : node->getPrev()) {
        auto* inst = dyn_cast<StinkyInstruction>(node);
        if (inst && !isPseudoInst(inst)) return inst;
    }
    return nullptr;
}

const StinkyInstruction* adjacentWmmaInBlock(const StinkyInstruction& from, bool forward) {
    const StinkyInstruction* inst = firstInst(forward ? from.getNext() : from.getPrev(), forward);
    return inst && isXDLWMMA(*inst) ? inst : nullptr;
}

// The XDL WMMAs right before `from` on every path; from the block end when null.
// A branch ending a predecessor sits in between, so only fallthrough blocks chain.
void collectPrevWmmas(BasicBlock& bb, const StinkyInstruction* from,
                      std::unordered_set<const BasicBlock*>& visited,
                      std::vector<const StinkyInstruction*>& prev) {
    if (const StinkyInstruction* inst =
            firstInst(from ? from->getPrev() : bb.getTerminator(), /*forward=*/false)) {
        if (isXDLWMMA(*inst)) prev.push_back(inst);
        return;
    }
    for (BasicBlock* pred : bb.getPredecessors())
        if (visited.insert(pred).second) collectPrevWmmas(*pred, nullptr, visited, prev);
}

std::vector<const StinkyInstruction*> prevWmmas(BasicBlock& bb, const StinkyInstruction& wmma) {
    std::unordered_set<const BasicBlock*> visited;
    std::vector<const StinkyInstruction*> prev;
    collectPrevWmmas(bb, &wmma, visited, prev);
    return prev;
}

// The in-place chain ending at `last`, oldest first. Empty when it continues from
// another block, where its pairing cannot be checked.
std::vector<const StinkyInstruction*> inBlockChain(BasicBlock& bb, const StinkyInstruction& last) {
    std::vector<const StinkyInstruction*> chain{&last};
    for (;;) {
        const StinkyInstruction& head = *chain.back();
        const StinkyInstruction* prev = adjacentWmmaInBlock(head, /*forward=*/false);
        if (!prev) {
            for (const StinkyInstruction* p : prevWmmas(bb, head))
                if (accumulatesInPlace(*p, head)) return {};
            break;
        }
        if (!accumulatesInPlace(*prev, head)) break;
        chain.push_back(prev);
    }
    std::reverse(chain.begin(), chain.end());
    return chain;
}

// Null reason: allowed. Unconfirmed: a paired chain from another block, which the
// check cannot follow.
struct Verdict {
    const char* reason = nullptr;
    bool confirmed = true;
};

// Pairs are chain positions (0,1), (2,3), ... `wmma` needs different A and B from
// the rest of its pair and from the previous pair, and a pair needs both halves.
Verdict pairedVerdict(BasicBlock& bb, const StinkyInstruction& wmma) {
    const auto chain = inBlockChain(bb, wmma);
    if (chain.size() < 2)
        return {"the pairs continue from another block and cannot be checked", false};
    const size_t pos = chain.size() - 1;
    const size_t pairStart = pos - pos % 2;
    for (size_t i = pairStart < 2 ? 0 : pairStart - 2; i < pos; ++i)
        if (!differentAB(*chain[i], wmma))
            return {"needs different A and B from its pair and the previous pair"};
    if (pos % 2 == 0) {
        const StinkyInstruction* next = adjacentWmmaInBlock(wmma, /*forward=*/true);
        if (!next || !accumulatesInPlace(wmma, *next)) return {"must be paired with the next WMMA"};
    }
    return {};
}

Verdict inPlaceVerdict(BasicBlock& bb, const StinkyInstruction& prev,
                       const StinkyInstruction& wmma) {
    const InPlaceRule rule = inPlaceRule(wmma);
    if (rule == InPlaceRule::Forbidden || rule != inPlaceRule(prev))
        return {"switch the accumulator"};
    if (rule == InPlaceRule::Allowed) return {};
    if (abOverlap(prev) || abOverlap(wmma)) return {"A and B of one WMMA must not overlap"};
    if (rule == InPlaceRule::DifferentAB)
        return differentAB(prev, wmma) ? Verdict{} : Verdict{"needs different A and B"};
    return pairedVerdict(bb, wmma);
}

void checkInPlaceAccumulation(const Function& func, BasicBlock& bb, const StinkyInstruction& wmma) {
    for (const StinkyInstruction* prev : prevWmmas(bb, wmma)) {
        if (!accumulatesInPlace(*prev, wmma)) continue;
        const Verdict verdict = inPlaceVerdict(bb, *prev, wmma);
        if (!verdict.reason) continue;
        std::cerr << "[InsertCoexecHazardPass] " << (verdict.confirmed ? "error" : "warning")
                  << ": WMMA in function \"" << func.getName() << "\" block \"" << bb.getLabel()
                  << "\" accumulates the previous WMMA's D in place: " << verdict.reason << "\n";
        prev->dump(std::cerr);
        wmma.dump(std::cerr);
        if (verdict.confirmed)
            report_fatal_error("InsertCoexecHazardPass: in-place WMMA accumulation in '" +
                               func.getName() + "': " + verdict.reason);
        return;
    }
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
        hw_ = &passCtx.getHWModel();
        PASS_DEBUG(std::cerr << "[InsertCoexecHazard] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << "\n");
    }

    // V_NOPs a consumer needs behind a matched producer.
    int required(ProducerKind kind, int slots, bool consumerIsWmma) const {
        if (kind == ProducerKind::TRANS) return hw_->coexec.transToNonCoreSide;
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

    // Backward scan returning the max shortfall over every matching producer across
    // all predecessor paths; memo prunes re-entries. Gives up after maxSlotBudget
    // fillers.
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

        // Start just before the consumer, or at the block's last instruction when
        // scanning a predecessor. Skip pseudo nodes.
        IRBase* node = startBefore ? startBefore->getPrev() : bb.getTerminator();
        for (; node; node = node->getPrev()) {
            auto* instPtr = dyn_cast<StinkyInstruction>(node);
            if (!instPtr || isPseudoInst(instPtr)) continue;
            StinkyInstruction& inst = *instPtr;

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
            if (existing > hw_->coexec.maxSlotBudget) return best;
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
                    checkInPlaceAccumulation(func, bb, *inst);
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
    const HWModel* hw_ = nullptr;
};

char InsertCoexecHazardPass::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertCoexecHazardPass() {
    return std::make_unique<InsertCoexecHazardPass>();
}
}  // namespace stinkytofu
