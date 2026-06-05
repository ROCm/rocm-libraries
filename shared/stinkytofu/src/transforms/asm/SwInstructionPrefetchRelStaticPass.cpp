/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated software files (the "Software"), to deal
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
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelStaticPass.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"

namespace {
using namespace stinkytofu;

/// Global byte offsets for SW prefetch (128*255, then every 32*128), per plan.
constexpr int64_t kSwPrefetchFirstGlobalByte = int64_t(128) * 255;
constexpr int64_t kSwPrefetchSpacingBytes = int64_t(32) * 128;

/// **PC-rel chain after `s_getpc_b64`.**  The instruction records the address
/// of the *next* in-stream instruction in an SGPR pair.  Downstream scalars
/// (often `s_add_i32` / `s_add_u32` / `s_addc_u32` with label relocations)
/// combine that pair with addends to build a final PC (branch target, table
/// address, etc.).  The hardware and relocations assume a contiguous encoding
/// from getpc through those adds; inserting **`s_prefetch_inst_pc_rel`**
/// **between** getpc and the fixups shifts layout and invalidates the address
/// arithmetic.  While the forward window is open, a naive “insert before this
/// instruction” for any of its **N** real insns (`s_getpc_b64` plus **N−1**
/// followers) would interpose bytes in that chain (worst case: **between**
/// getpc and the first fixup).  The pass **redirects** such P(k): emit prefetch
/// **before** `s_getpc_b64` and re-walk byte sizes.
///
/// **Window size (constants below).**  **N** =
/// `kSwPrefetchForwardWindowRealInsnCount` counts real Stinky insns with
/// PHI/LABEL skipped; the guard after getpc is **N−1** =
/// `kSwPrefetchForwardWindowInsnsAfterGetpc`.  N is a conservative bound for
/// typical getpc + low/high PC materialization—raise it only if codegen emits
/// longer unbroken post-getpc chains.
constexpr unsigned kSwPrefetchForwardWindowRealInsnCount = 5u;
constexpr unsigned kSwPrefetchForwardWindowInsnsAfterGetpc =
    kSwPrefetchForwardWindowRealInsnCount - 1u;

/// Identify `s_getpc_b64` via unified opcode (same HwInstDesc row as mnemonic
/// `s_getpc_b64`).
bool instructionIsSGetpcB64(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_getpc_b64;
}

bool swPrefetchLabelNameExists(BasicBlock& bb, const std::string& name) {
    for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;
        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() != GFX::LABEL) continue;
        if (const LabelData* ld = inst.getModifier<LabelData>()) {
            if (ld->label == name) return true;
        }
    }
    return false;
}

/// klength simm5: 31 => 32 instruction cache lines (128 B each), same as CK `INST_PREFETCH`.
constexpr int32_t kSwPrefetchPcRelKlengthImm = 31;

static void addSwPrefetchInstPcRelOperands(StinkyInstruction& prefetchInst) {
    prefetchInst.addSrcReg(StinkyRegister(0));
    prefetchInst.addSrcReg(StinkyRegister("null"));
    prefetchInst.addSrcReg(StinkyRegister(kSwPrefetchPcRelKlengthImm));
}

/// \brief Emit **`s_prefetch_inst_pc_rel`** (`koffset=0`, `null`, `klength=31`)
///        **before** \p anchorIt.
///
/// **Sizing.**  Uses **`getEffectiveBaseSizeInBytes`** /
/// **`getLiteralExtraBytes`** at **`blockGlobalByteOffset +
/// blockLocalByteOffsetWherePrefetchStarts`**. \p labelOff / \p asmSetSymbols are
/// passed through for PC-relative literal rules.
///
/// \param blockLocalByteOffsetWherePrefetchStarts  Block-local offset to the
///        prefetch’s first byte before this insert—usually the walk’s
///        **`totalBytes`** at \p anchorIt, or **`nextPrefetchLocal`** when
///        stacking redirects before **`s_getpc_b64`**.
/// \param walkTotalBytes  Incremented by the prefetch size; pass **`dummyWalk`**
///        when the caller re-walks **`totalBytes`** separately.
///
/// \return Prefetch size in bytes, or **0** if **`s_prefetch_inst_pc_rel`** has no
///         **`HwInstDesc`** for \p archId.
int64_t insertSwPrefetchInstPcRelBefore(
    BasicBlock& bb, IRList::iterator anchorIt, GfxArchID archId, AsmIRBuilder& builder,
    std::unordered_map<std::string, int64_t>* labelOff, int64_t blockGlobalByteOffset,
    int64_t blockLocalByteOffsetWherePrefetchStarts, int64_t& walkTotalBytes,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    const HwInstDesc* pfMc = getMCIDByUOp(GFX::s_prefetch_inst_pc_rel, archId);
    if (pfMc == nullptr) return 0;

    StinkyInstruction* prefetchInst = builder.create(pfMc);
    addSwPrefetchInstPcRelOperands(*prefetchInst);

    const int64_t gPf = blockGlobalByteOffset + blockLocalByteOffsetWherePrefetchStarts;
    const int pfB = getEffectiveBaseSizeInBytes(*prefetchInst) +
                    getLiteralExtraBytes(*prefetchInst, labelOff, gPf, asmSetSymbols);

    walkTotalBytes += pfB;

    bb.insertIR(anchorIt, prefetchInst);
    return static_cast<int64_t>(pfB);
}

/// One prefetch before \p anchorIt (`s_getpc_b64`), chaining \p nextPrefetchLocal,
/// then re-walk \p windowIters so \p totalBytes matches IR (same as end-of-window
/// batch flush, but per P(k) hit inside the forward window).
void insertPrefetchBeforeGetpcAndRewalkWindow(
    BasicBlock& bb, IRList::iterator anchorIt, const std::vector<IRList::iterator>& windowIters,
    int64_t totalBytesAtGetpcStart, int64_t& nextPrefetchLocal, int64_t blockGlobalByteOffset,
    GfxArchID archId, AsmIRBuilder& builder, std::unordered_map<std::string, int64_t>& labelOff,
    int64_t& totalBytes, std::ostream* dbgOut,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    int64_t dummyWalk = 0;
    const int64_t d = insertSwPrefetchInstPcRelBefore(bb, anchorIt, archId, builder, &labelOff,
                                                      blockGlobalByteOffset, nextPrefetchLocal,
                                                      dummyWalk, asmSetSymbols);
    nextPrefetchLocal += d;
    // Block-local offset at start of s_getpc_b64 after stacked prefetches.
    totalBytes = nextPrefetchLocal;
    const int64_t bytesInsertedBeforeGetpc = nextPrefetchLocal - totalBytesAtGetpcStart;
    for (const IRList::iterator& wit : windowIters) {
        StinkyInstruction& winst = getStinkyInst(wit);
        const int64_t gBefore = blockGlobalByteOffset + totalBytes;
        const int baseSz = getEffectiveBaseSizeInBytes(winst);
        const int litEx = getLiteralExtraBytes(winst, &labelOff, gBefore, asmSetSymbols);
        totalBytes += baseSz + litEx;
    }

    if (dbgOut != nullptr) {
        *dbgOut << "[SwInstructionPrefetchRelStaticPass] getpc-window redirect insert: bb=\""
                << bb.getLabel() << "\", window_insn_count=" << windowIters.size()
                << ", cumulative_bytes_before_getpc=" << bytesInsertedBeforeGetpc << "\n";
    }
}

void appendSwPrefetchInstPcRel(BasicBlock& bb, GfxArchID archId, AsmIRBuilder& builder,
                               std::unordered_map<std::string, int64_t>* labelOff,
                               int64_t blockGlobalByteOffset, int64_t& totalBytes,
                               const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    const HwInstDesc* pfMc = getMCIDByUOp(GFX::s_prefetch_inst_pc_rel, archId);
    if (pfMc == nullptr) return;

    StinkyInstruction* prefetchInst = builder.create(pfMc);
    addSwPrefetchInstPcRelOperands(*prefetchInst);

    const int64_t gPf = blockGlobalByteOffset + totalBytes;
    const int pfB = getEffectiveBaseSizeInBytes(*prefetchInst) +
                    getLiteralExtraBytes(*prefetchInst, labelOff, gPf, asmSetSymbols);
    totalBytes += pfB;

    bb.appendIR(prefetchInst);
}

/// \brief Place software prefetch (`s_prefetch_inst_pc_rel`) at
/// fixed **global**
///        byte boundaries P(k), using one forward IR walk per basic block.
///
/// **Thresholds.**  P(k) = `kSwPrefetchFirstGlobalByte` + k *
/// `kSwPrefetchSpacingBytes` (equivalently 128*255 + k*(32*128) bytes from the
/// start of the linked program image).  Index `k` advances monotonically
/// (`kNext`) whenever a boundary is consumed (matched or flushed).
///
/// **Walk state.**  `totalBytes` is the block-local byte size accumulated so
/// far (same spirit as
/// **`stinkytofu::accumulateInstructionSize`** (same TU as the accumulate
/// pass): effective opcode size plus literal extras, labels in `labelOff`). PHI
/// nodes and LABELs do not advance `totalBytes`.  For each other Stinky
/// instruction, let `globalPcBefore` / `globalPcAfter` be its start/end in
/// **global** space using `blockGlobalByteOffset + totalBytes` (+ instruction
/// size for the end).
///
/// **Where P(k) lands.**  If `globalPcBefore < P(k) <= globalPcAfter`, the
/// boundary falls strictly *inside* this instruction's encoding footprint
/// (end-inclusive, start-exclusive).  Prefetch IR is inserted immediately
/// **before** that instruction so the stream byte at P(k) still belongs to the
/// same logical instruction as before the insert.  The pseudo-label name
/// `label_SWprefetch_<k>` is recorded in `labelOff` at `globalPcAfter` (first
/// byte after the host instruction) for literal/PC-relative sizing of the new
/// mov/prefetch pair.
///
/// **`s_getpc_b64` window.**  PC-relative lowering expects `s_getpc_b64` to
/// stay adjacent to the next few real instructions.  After each `s_getpc_b64`,
/// a guard counts the next `kSwPrefetchForwardWindowInsnsAfterGetpc` real
/// instructions (so the protected region is
/// `kSwPrefetchForwardWindowRealInsnCount` real insns: getpc plus the following
/// N-1).  While the guard is active, a P(k) that would normally insert *before*
/// the current instruction is redirected: prefetch goes **before** the queued
/// `s_getpc_b64`, the window is re-walked from that getpc so `totalBytes` and
/// literal layout stay consistent, and the current instruction's size is
/// absorbed by that rewalk when applicable.  When the guard expires (or the BB
/// ends), `getpcWindowIters` and the guard are cleared.
///
/// **Post-walk flush.**  Any P(k) that lies in `[blockGlobalByteOffset,
/// blockEndGlobal]` but never fell strictly inside some instruction's
/// `(globalPcBefore, globalPcAfter]` (e.g. only labels, alignment, or P exactly
/// at a boundary with no spanning op) is satisfied by appending prefetch at the
/// **end** of the block.
///
/// \param allowSwPrefetchInsertion  If false, perform the identical walk
/// (including getpc-window
///        logic and `kNext` updates) so sizes and decisions match the inserting
///        path, but do not emit prefetch IR or mutate the BB.  The pass
///        uses this with loop detection
///        (`findLoopForBB` / `detectLoops`) to skip natural loop bodies while
///        keeping global layout accounting aligned.  Compiler "unrolled" loops
///        are not tagged separately here; use label heuristics or other
///        metadata if you need unroll-specific behavior.
void insertSwPrefetchLabels(BasicBlock& bb, int64_t blockGlobalByteOffset, GfxArchID archId,
                            std::ostream* dbgOut,
                            const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                            bool allowSwPrefetchInsertion = true) {
    std::unordered_map<std::string, int64_t> labelOff;
    int64_t totalBytes = 0;
    int64_t kNext = 0;
    /// While >0, a prefetch that would have been placed before the current insn
    /// is redirected before the saved `s_getpc_b64` instead (see forward window).
    unsigned getpcPcRelChainGuardRemaining = 0;
    /// Block-local offset G at the active `s_getpc_b64` (unchanged for the
    /// window; rewalk baseline).
    int64_t totalBytesAtGetpcStart = 0;
    /// Program order: getpc then next (N-1) real insns; |vector| ≤ N.
    std::vector<IRList::iterator> getpcWindowIters;
    /// Prefetch encoding start before getpc; equals G at window open, then grows
    /// per stacked prefetch at that anchor.
    int64_t nextPrefetchBlockOffsetBeforeGetpc = 0;

    AsmIRBuilder builder(bb, archId);

    for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        addAlignmentPaddingFromDirectiveNode(node, blockGlobalByteOffset, totalBytes, dbgOut);
        if (node->getType() == IRBase::IRType::StinkyAsmDirective) continue;
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;

        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() == GFX::PHI) continue;
        if (inst.getUnifiedOpcode() == GFX::LABEL) {
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                addAlignmentPaddingForLabelInstruction(inst, blockGlobalByteOffset, totalBytes,
                                                       dbgOut);
                labelOff[ld->label] = blockGlobalByteOffset + totalBytes;
            }
            continue;
        }

        const bool isGetpc = instructionIsSGetpcB64(inst);

        // Forward window: N real insns = s_getpc_b64 + next (N-1). Open on getpc
        // (before P(k)); queue each following window insn here. Inside the window,
        // P(k) inserts before getpc immediately (see else-if anchor below);
        // outside, insert before current insn.
        if (isGetpc) {
            getpcWindowIters.clear();
            getpcWindowIters.push_back(it);
            totalBytesAtGetpcStart = totalBytes;
            nextPrefetchBlockOffsetBeforeGetpc = totalBytesAtGetpcStart;
            getpcPcRelChainGuardRemaining = kSwPrefetchForwardWindowInsnsAfterGetpc;
        } else if (!getpcWindowIters.empty() && getpcPcRelChainGuardRemaining > 0u) {
            if (getpcWindowIters.size() <
                static_cast<size_t>(kSwPrefetchForwardWindowRealInsnCount))
                getpcWindowIters.push_back(it);
        }

        /// Walk offset at the **start** of this real instruction (before inner P
        /// loop).
        const int64_t walkOffsetAtInstStart = totalBytes;

        const int64_t globalPcBefore = blockGlobalByteOffset + totalBytes;
        const int baseSize = getEffectiveBaseSizeInBytes(inst);
        const int literalExtra =
            getLiteralExtraBytes(inst, &labelOff, globalPcBefore, asmSetSymbols);
        const int instBytes = baseSize + literalExtra;
        const int64_t globalPcAfter = globalPcBefore + instBytes;
        // TODO: Revise for last basic block
        // Boundaries with globalPcBefore < P <= globalPcAfter (ascending k) before
        // this insn.
        bool redirectRewalkAbsorbedCurrentInstSizes = false;
        for (;;) {
            const int64_t P = kSwPrefetchFirstGlobalByte + kNext * kSwPrefetchSpacingBytes;
            if (P <= blockGlobalByteOffset) {
                ++kNext;
                continue;
            }
            if (P > globalPcAfter) break;
            if (P < globalPcBefore) {
                ++kNext;
                continue;
            }
            // TODO: check if this is correct
            if (P <= globalPcBefore) break;

            const std::string name = std::string("label_SWprefetch_") + std::to_string(kNext);
            if (!swPrefetchLabelNameExists(bb, name)) {
                if (allowSwPrefetchInsertion) {
                    // StinkyInstruction* lbl = builder.createLabel(name, 1);
                    // bb.insertIR(it, lbl);
                    labelOff[name] = globalPcAfter;
                    if (getpcPcRelChainGuardRemaining == 0u) {
                        (void)insertSwPrefetchInstPcRelBefore(
                            bb, it, archId, builder, &labelOff, blockGlobalByteOffset,
                            walkOffsetAtInstStart, totalBytes, asmSetSymbols);
                    } else if (!getpcWindowIters.empty()) {
                        insertPrefetchBeforeGetpcAndRewalkWindow(
                            bb, getpcWindowIters.front(), getpcWindowIters, totalBytesAtGetpcStart,
                            nextPrefetchBlockOffsetBeforeGetpc, blockGlobalByteOffset, archId,
                            builder, labelOff, totalBytes, dbgOut, asmSetSymbols);
                        redirectRewalkAbsorbedCurrentInstSizes = true;
                    }
                }
            }
            ++kNext;
        }

        if (!redirectRewalkAbsorbedCurrentInstSizes) totalBytes += instBytes;

        if (!isGetpc && getpcPcRelChainGuardRemaining > 0u) {
            --getpcPcRelChainGuardRemaining;
            if (getpcPcRelChainGuardRemaining == 0u) getpcWindowIters.clear();
        }
    }
    // Flush remaining P(k) with P <= block end that never matched any instruction
    // span (globalPcBefore < P <= globalPcAfter), e.g. empty block / only
    // PHI+LABEL, or L == P(k) with no op whose (start, end] contains P. Not
    // specific to the "last instruction" — any uncovered boundary is appended
    // after the IR walk.
    // TODO: pass it to next BasicBlock
    getpcWindowIters.clear();
    // TODO: check if this is needed
    const int64_t blockEndGlobal = blockGlobalByteOffset + totalBytes;
    for (;;) {
        const int64_t P = kSwPrefetchFirstGlobalByte + kNext * kSwPrefetchSpacingBytes;
        if (P < blockGlobalByteOffset) {
            ++kNext;
            continue;
        }
        // TODO: check if this is correct
        if (P > blockEndGlobal) break;

        const std::string name = std::string("label_SWprefetch_") + std::to_string(kNext);
        if (!swPrefetchLabelNameExists(bb, name)) {
            if (allowSwPrefetchInsertion) {
                // StinkyInstruction* lbl = builder.createLabel(name, 1);
                // bb.appendIR(lbl);
                labelOff[name] = blockEndGlobal;
                appendSwPrefetchInstPcRel(bb, archId, builder, &labelOff, blockGlobalByteOffset,
                                          totalBytes, asmSetSymbols);
            }
        }
        ++kNext;
    }
}

/// Debug-only: list P(k) grid boundaries that fall in this basic block.
/// Per-instruction placement (including emitted `s_prefetch_inst_pc_rel`) is
/// already printed by **`accumulateInstructionSize`** above (global **`total=`**).
void debugPrintSwPrefetchGrid(std::ostream& os, const std::string& bbLabel,
                              int64_t blockGlobalStart, int64_t blockBytes) {
    const int64_t blockEndGlobal = blockGlobalStart + blockBytes;

    os << "[SwInstructionPrefetchRelStaticPass] SW prefetch grid (pseudo labels "
          "label_SWprefetch_<k>), "
          "basic block \""
       << bbLabel << "\" blockGlobalStart=" << blockGlobalStart << " blockSize=" << blockBytes
       << " P(k)=" << kSwPrefetchFirstGlobalByte << "+k*" << kSwPrefetchSpacingBytes << "\n";

    if (blockEndGlobal < kSwPrefetchFirstGlobalByte) {
        os << "  (none: block end " << blockEndGlobal << " < first threshold "
           << kSwPrefetchFirstGlobalByte << ")\n";
        return;
    }

    for (int64_t k = 0;; ++k) {
        const int64_t P = kSwPrefetchFirstGlobalByte + k * kSwPrefetchSpacingBytes;
        if (P > blockEndGlobal) break;
        if (P < blockGlobalStart) continue;

        os << "  [label_SWprefetch_" << k << " k=" << k << " global_P=" << P
           << " local_P=" << (P - blockGlobalStart)
           << " (see accumulate dump: insn with total=" << P << ")]\n";
    }
}

class SwInstructionPrefetchRelStaticPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SwInstructionPrefetchRelStaticPass";
    }

    PassID getPassID() const override {
        return &SwInstructionPrefetchRelStaticPass::ID;
    }

    /// When true, `insertSwPrefetchLabels` walks each BB as usual but does not
    /// insert prefetch IR in basic blocks that belong to a natural loop
    /// (`detectLoops`). Default false.
    void setSkipSwPrefetchInNaturalLoopBodies(bool skip) {
        m_skipSwPrefetchInNaturalLoopBodies = skip;
    }

    bool getSkipSwPrefetchInNaturalLoopBodies() const {
        return m_skipSwPrefetchInNaturalLoopBodies;
    }

    void runOnBasicBlock(BasicBlock& bb, PassContext& passCtx) {
        const auto& archArr = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId =
            getGfxArchID(static_cast<uint32_t>(archArr[0]), static_cast<uint32_t>(archArr[1]),
                         static_cast<uint32_t>(archArr[2]));

        const int64_t blockGlobalStart = m_byteOffsetBase;
        const bool allowIns =
            !m_skipSwPrefetchInNaturalLoopBodies || (findLoopForBB(m_loops, &bb) == nullptr);
        insertSwPrefetchLabels(bb, blockGlobalStart, archId, m_debug ? m_debugStream : nullptr,
                               &m_asmSetSymbols, allowIns);

        int blockCount = 0;
        int64_t blockBytes = 0;
        if (m_debug) {
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] BasicBlock: " << bb.getLabel()
                           << "\n";
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] IR after SW prefetch label "
                              "insertion:\n";
            // bb.dump(*m_debugStream);
            *m_debugStream << "\n";
        }
        m_totalCycles +=
            accumulateInstructionSize(bb, m_labelByteOffset, m_debug ? m_debugStream : nullptr,
                                      &blockCount, &blockBytes, m_byteOffsetBase, &m_asmSetSymbols);
        if (m_debug)
            debugPrintSwPrefetchGrid(*m_debugStream, bb.getLabel(), blockGlobalStart, blockBytes);
        m_totalInstructionCount += blockCount;
        m_totalBytes += blockBytes;
        m_byteOffsetBase += blockBytes;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        m_totalCycles = 0;
        m_totalInstructionCount = 0;
        m_totalBytes = 0;
        m_labelByteOffset.clear();
        m_byteOffsetBase = 0;
        m_loops = detectLoops(func);
        collectAsmSetSymbolValues(func, m_asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        int totalBlocksInFunction = 0;
        for ([[maybe_unused]] const BasicBlock& bb : func) totalBlocksInFunction++;

        int blocksProcessed = 0;
        if (m_debug) {
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] processAllBlocks="
                           << (m_processAllBlocks ? "true" : "false") << ", function has "
                           << totalBlocksInFunction << " basic block(s)\n";
            dumpAsmSetSymbolMap(*m_debugStream, m_asmSetSymbols);
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] blocks to process:\n";
        }

        for (BasicBlock& bb : func) {
            bool processThis = m_processAllBlocks || passCtx.shouldProcessBasicBlock(bb);
            if (m_debug) {
                *m_debugStream << "  - BasicBlock \"" << bb.getLabel() << "\" "
                               << (processThis ? "[PROCESSING]" : "[SKIPPED by filter]") << "\n";
            }
            if (processThis) {
                runOnBasicBlock(bb, passCtx);
                blocksProcessed++;
            }
        }

        if (m_debug) {
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] processed " << blocksProcessed
                           << " / " << totalBlocksInFunction << " basic block(s)\n";
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] total instruction count = "
                           << m_totalInstructionCount << "\n";
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] total cycles = "
                           << m_totalCycles << "\n";
            *m_debugStream << "[SwInstructionPrefetchRelStaticPass] total size = " << m_totalBytes
                           << " bytes\n";
            if (!m_labelByteOffset.empty()) {
                *m_debugStream << "[SwInstructionPrefetchRelStaticPass] label -> byte offset:\n";
                for (const auto& kv : m_labelByteOffset)
                    *m_debugStream << "  \"" << kv.first << "\" -> " << kv.second << " bytes\n";
            }
            if (m_debugFile.is_open()) m_debugFile.close();
        }
        return PreservedAnalyses::none();
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

   private:
    int64_t m_totalCycles = 0;
    int m_totalInstructionCount = 0;
    int64_t m_totalBytes = 0;
    int64_t m_byteOffsetBase = 0;
    std::vector<Loop> m_loops;  // natural loops (see detectLoops) for optional insertion gating
    std::unordered_map<std::string, int64_t> m_labelByteOffset;
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    /// When set, do not insert SW prefetch in BBs that belong to any
    /// `detectLoops` body.
    bool m_skipSwPrefetchInNaturalLoopBodies = false;
    bool m_processAllBlocks = true;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchRelStaticPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createSwInstructionPrefetchRelStaticPass(const std::string& debugOutputPath) {
    auto p = std::make_unique<SwInstructionPrefetchRelStaticPass>();
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchRelStaticPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchRelStaticPass>();
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        constexpr const char* kSwPrefetchPassDumpLeaf = "sw_prefetch_pass.txt";
        const std::string path = (dir / kSwPrefetchPassDumpLeaf).string();
        p->setDebugOutputPath(path);
        p->setDebug(true);
    }
    return p;
}
}  // namespace stinkytofu
