// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/CoexecNopPlanner.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <optional>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "InsertCoexecHazardPass"

namespace stinkytofu {
namespace {

// Per-arch co-execution hazard rules live in HWModel::Coexec (see
// stinkytofu/hardware/HWModel.hpp). WMMA V_NOP counts come from each producer's
// coIssueWindow bitmask.

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

// Blocks a scan entered, and on how few fillers: the memo of one consumer's scan.
using EntryMemo = std::vector<std::pair<const BasicBlock*, int>>;

class Planner {
   public:
    // A kind no instruction of the sequences can produce never needs a scan.
    Planner(const HWModel& hw, BlockSequences& seqs, const CoexecProducers& producers)
        : hw_(hw),
          seqs_(seqs),
          hasTrans_(producers.trans),
          hasDgemm_(producers.dgemm),
          hasPerm_(producers.perm) {}

    // V_NOPs a consumer needs behind a matched producer.
    int required(ProducerKind kind, int slots, bool consumerIsWmma) const {
        if (kind == ProducerKind::TRANS) return hw_.coexec.transToNonCoreSide;
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
    int scanBack(const BasicBlock* bb, std::optional<size_t> startBefore, int accExisting,
                 const ConsumerCtx& ctx, EntryMemo& minExisting) {
        // Memoize predecessor entries on fewest fillers; prune when this arrival can't widen the
        // shortfall.
        if (!startBefore) {
            auto it = std::find_if(minExisting.begin(), minExisting.end(),
                                   [bb](const auto& e) { return e.first == bb; });
            if (it != minExisting.end() && it->second <= accExisting) return INT_MIN;
            if (it != minExisting.end())
                it->second = accExisting;
            else
                minExisting.emplace_back(bb, accExisting);
        }

        int best = INT_MIN;
        int existing = accExisting;

        // Start just before the consumer, or at the block's last instruction when
        // scanning a predecessor. Skip pseudo nodes.
        const std::vector<const StinkyInstruction*>& seq = seqs_.at(bb);
        for (size_t i = startBefore ? *startBefore : seq.size(); i-- > 0;) {
            const StinkyInstruction& inst = *seq[i];
            if (isPseudoInst(&inst)) continue;

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
            if (existing > hw_.coexec.maxSlotBudget) return best;
        }

        // Reached the top of the BB with budget to spare: continue into every
        // predecessor, taking the max shortfall across them.
        for (const BasicBlock* pred : bb->getPredecessors())
            best = std::max(best, scanBack(pred, std::nullopt, existing, ctx, minExisting));
        return best;
    }

    int hazardFor(const BasicBlock* bb, size_t index, ProducerKind kind, bool consumerIsWmma) {
        if ((kind == ProducerKind::TRANS && !hasTrans_) || (kind == ProducerKind::DGEMM && !hasDgemm_) ||
            (kind == ProducerKind::PERM && !hasPerm_))
            return 0;
        ConsumerCtx ctx{kind, consumerIsWmma, seqs_.at(bb)[index]};
        // minExisting bounds re-entries: a block is re-scanned only on a strictly smaller filler
        // count.
        minExisting_.clear();
        const int r = scanBack(bb, index, /*accExisting=*/0, ctx, minExisting_);
        return r > 0 ? r : 0;
    }

    // v_nops are stripped upstream by StinkyRemoveNopPass; this counts the remaining
    // (deliberate, scheduler-placed) fillers and tops up the shortfall.
    std::vector<CoexecNopInsertion> planBlock(const BasicBlock* bb, const StinkyInstruction& vnop) {
        std::vector<CoexecNopInsertion> out;
        auto& seq = seqs_.at(bb);
        for (size_t i = 0; i < seq.size(); ++i) {
            const StinkyInstruction& inst = *seq[i];
            if (isPseudoInst(&inst) || &inst == &vnop) continue;

            int toInsert = 0;
            if (isXDLWMMA(inst)) {
                toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::WMMA, true));
                toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::TRANS, false));
                toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::DGEMM, true));
                toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::PERM, false));
            } else if (isCoexecutableVALU(inst)) {
                toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::WMMA, false));
                toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::PERM, false));
                // TRANS -> core/side is HW-handled; only a TRANS consumer needs
                // the TRANS -> TRANS spacing.
                if (isTranscendental(inst))
                    toInsert = std::max(toInsert, hazardFor(bb, i, ProducerKind::TRANS, false));
            }

            if (toInsert > 0) {
                seq.insert(seq.begin() + static_cast<long>(i), static_cast<size_t>(toInsert), &vnop);
                i += static_cast<size_t>(toInsert);
                out.push_back({&inst, toInsert});
                PASS_DEBUG(std::cerr << "[InsertCoexecHazard]   inserted " << toInsert
                                     << " v_nop before " << inst.getHwInstDesc()->mnemonic
                                     << " in bb \"" << bb->getLabel() << "\"\n");
            }
        }
        return out;
    }

   private:
    const HWModel& hw_;
    BlockSequences& seqs_;
    EntryMemo minExisting_;
    bool hasTrans_;
    bool hasDgemm_;
    bool hasPerm_;
};

}  // namespace

CoexecProducers coexecProducersIn(const BlockSequences& seqs) {
    CoexecProducers p;
    for (const auto& [bb, seq] : seqs) {
        (void)bb;
        for (const StinkyInstruction* inst : seq) {
            p.trans |= isTranscendental(*inst);
            p.dgemm |= isDGEMMProducer(*inst);
            p.perm |= isTensorLUT(*inst);
        }
    }
    return p;
}

std::vector<std::vector<CoexecNopInsertion>> planCoexecNops(
    const HWModel& hw, const std::vector<const BasicBlock*>& blocks, BlockSequences& seqs,
    const StinkyInstruction& vnop, const CoexecProducers* producers) {
    Planner planner(hw, seqs, producers != nullptr ? *producers : coexecProducersIn(seqs));
    std::vector<std::vector<CoexecNopInsertion>> out;
    out.reserve(blocks.size());
    for (const BasicBlock* bb : blocks) out.push_back(planner.planBlock(bb, vnop));
    return out;
}

}  // namespace stinkytofu
