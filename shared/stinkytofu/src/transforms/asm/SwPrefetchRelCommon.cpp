// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"

namespace {
using namespace stinkytofu;

/// Grid constants `kSwPrefetchFirstGlobalByte` / `kSwPrefetchSpacingBytes` live in
/// SwPrefetchRelCommon.hpp (128*255, then every 32*128).

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

    // XNACK safety: drain outstanding address translations before the prefetch. Relative hints are
    // scattered (one per grid point, separated by real code), so a single upstream drain would be
    // stale by the next hint — hence one s_wait_xcnt 0 per prefetch. Gated by the toggle + opcode
    // availability (no-op off gfx1250). Its bytes MUST join the walk so downstream P(k) placement
    // and PC-relative literal/label offsets stay aligned with the emitted layout.
    StinkyInstruction* xcntInst = nullptr;
    int xcntB = 0;
    if (kSwPrefetchEmitXnackWait) {
        if (const HwInstDesc* xcntMc = getMCIDByUOp(GFX::s_wait_xcnt, archId)) {
            xcntInst = builder.create(xcntMc);
            xcntInst->addSrcReg(StinkyRegister(0));  // xcnt = 0
            const int64_t gXcnt = blockGlobalByteOffset + blockLocalByteOffsetWherePrefetchStarts;
            xcntB = getEffectiveBaseSizeInBytes(*xcntInst) +
                    getLiteralExtraBytes(*xcntInst, labelOff, gXcnt, asmSetSymbols);
        }
    }

    StinkyInstruction* prefetchInst = builder.create(pfMc);
    addSwPrefetchInstPcRelOperands(*prefetchInst);

    // Prefetch now follows the wait, so its first byte is xcntB further along.
    const int64_t gPf = blockGlobalByteOffset + blockLocalByteOffsetWherePrefetchStarts + xcntB;
    const int pfB = getEffectiveBaseSizeInBytes(*prefetchInst) +
                    getLiteralExtraBytes(*prefetchInst, labelOff, gPf, asmSetSymbols);

    walkTotalBytes += xcntB + pfB;

    if (xcntInst != nullptr) bb.insertIR(anchorIt, xcntInst);  // wait first
    bb.insertIR(anchorIt, prefetchInst);                       // then prefetch
    return static_cast<int64_t>(xcntB) + static_cast<int64_t>(pfB);
}

/// One prefetch before \p anchorIt (`s_getpc_b64`), chaining \p nextPrefetchLocal,
/// then re-walk \p windowIters so \p totalBytes matches IR (same as end-of-window
/// batch flush, but per P(k) hit inside the forward window).
void insertPrefetchBeforeGetpcAndRewalkWindow(
    BasicBlock& bb, IRList::iterator anchorIt, const std::vector<IRList::iterator>& windowIters,
    int64_t totalBytesAtGetpcStart, int64_t& nextPrefetchLocal, int64_t blockGlobalByteOffset,
    GfxArchID archId, AsmIRBuilder& builder, std::unordered_map<std::string, int64_t>& labelOff,
    int64_t& totalBytes, std::ostream* dbgOut, const char* debugPassTag,
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

    if (dbgOut != nullptr && debugPassTag != nullptr) {
        *dbgOut << "[" << debugPassTag << "] getpc-window redirect insert: bb=\"" << bb.getLabel()
                << "\", window_insn_count=" << windowIters.size()
                << ", cumulative_bytes_before_getpc=" << bytesInsertedBeforeGetpc << "\n";
    }
}

/// True when insn span ends on or past P(0)=32640 (touch or fully post-CP). Used for `pathEntered`.
bool insnSpanEndsOnOrPastCpBoundary(int64_t layoutBefore, int64_t instBytes) {
    return layoutBefore + instBytes >= kSwPrefetchFirstGlobalByte;
}

/// Bytes strictly past P(0) in layout space (0 when `layoutAfter == 32640`; no sentinel +1).
int64_t postCpBytesForInstructionSpan(int64_t layoutBefore, int64_t instBytes) {
    const int64_t layoutAfter = layoutBefore + instBytes;
    if (layoutAfter < kSwPrefetchFirstGlobalByte) return 0;
    if (layoutBefore >= kSwPrefetchFirstGlobalByte) return instBytes;
    return layoutAfter - kSwPrefetchFirstGlobalByte;
}

int64_t cfgAccumBeforeGlobal(int64_t bbEntryAccum, int64_t postCpCumulBefore) {
    return kSwPrefetchFirstGlobalByte + bbEntryAccum + postCpCumulBefore;
}

int64_t cfgAccumAfterGlobal(int64_t bbEntryAccum, int64_t postCpCumulAfter) {
    return kSwPrefetchFirstGlobalByte + bbEntryAccum + postCpCumulAfter;
}

/// Anchor interval for cfgGate: layout span until first post-CP byte is accumulated in this BB;
/// then `32640 + accumExit` for subsequent insns.
void cfgGateIntervalBounds(int64_t layoutBefore, int64_t layoutAfter, int64_t bbEntryAccum,
                           int64_t postCpCumulBefore, int64_t postCpCumulAfter, int64_t& outBefore,
                           int64_t& outAfter) {
    if (postCpCumulBefore == 0) {
        outBefore = layoutBefore;
        outAfter = layoutAfter;
    } else {
        outBefore = cfgAccumBeforeGlobal(bbEntryAccum, postCpCumulBefore);
        outAfter = cfgAccumAfterGlobal(bbEntryAccum, postCpCumulAfter);
    }
}

/// Pure CFG global coords: `32640 + bbEntryAccum + postCpCumul` (path progress, not gate interval).
void debugPrintCfgAccumGlobals(std::ostream& os, int64_t bbEntryAccum, int64_t postCpCumulBefore,
                               int64_t postCpCumulAfter) {
    os << "accumBeforeGlobal=" << cfgAccumBeforeGlobal(bbEntryAccum, postCpCumulBefore)
       << " accumAfterGlobal=" << cfgAccumAfterGlobal(bbEntryAccum, postCpCumulAfter);
}

struct CfgEdge {
    BasicBlock* from = nullptr;
    BasicBlock* to = nullptr;
    bool operator==(const CfgEdge& other) const {
        return from == other.from && to == other.to;
    }
};

struct CfgEdgeHash {
    size_t operator()(const CfgEdge& edge) const {
        const auto h1 = std::hash<BasicBlock*>{}(edge.from);
        const auto h2 = std::hash<BasicBlock*>{}(edge.to);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

std::unordered_set<CfgEdge, CfgEdgeHash> buildCfgBackEdgeSet(Function& func) {
    std::unordered_set<CfgEdge, CfgEdgeHash> backEdges;
    for (const Loop& loop : detectLoops(func)) {
        if (loop.latchBB != nullptr && loop.headerBB != nullptr)
            backEdges.insert(CfgEdge{loop.latchBB, loop.headerBB});
    }
    return backEdges;
}

bool isCfgBackEdge(BasicBlock* pred, BasicBlock* succ,
                   const std::unordered_set<CfgEdge, CfgEdgeHash>& backEdges) {
    return backEdges.contains(CfgEdge{pred, succ});
}

/// Per-BB layout walk: `layoutGlobal` for real insns, `blockLocalBytes`, `blockLocalBytesPostCp`.
void walkBlockLayoutAndPostCp(BasicBlock& bb, int64_t blockGlobalByteOffset,
                              std::unordered_map<std::string, int64_t>& labelOff,
                              const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                              std::unordered_map<StinkyInstruction*, int64_t>& layoutGlobal,
                              int64_t& blockLocalBytes, int64_t& blockLocalBytesPostCp,
                              int64_t& firstPostCpLayoutByte) {
    blockLocalBytes = 0;
    blockLocalBytesPostCp = 0;
    firstPostCpLayoutByte = kSwPrefetchNoPerBbGridAnchor;
    int64_t totalBytes = 0;

    for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        addAlignmentPaddingFromDirectiveNode(node, blockGlobalByteOffset, totalBytes, nullptr);
        if (node->getType() == IRBase::IRType::StinkyAsmDirective) continue;
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;

        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() == GFX::PHI) continue;
        if (inst.getUnifiedOpcode() == GFX::LABEL) {
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                addAlignmentPaddingForLabelInstruction(inst, blockGlobalByteOffset, totalBytes,
                                                       nullptr);
                labelOff[ld->label] = blockGlobalByteOffset + totalBytes;
            }
            continue;
        }

        const int64_t layoutBefore = blockGlobalByteOffset + totalBytes;
        const int baseSize = getEffectiveBaseSizeInBytes(inst);
        const int literalExtra = getLiteralExtraBytes(inst, &labelOff, layoutBefore, asmSetSymbols);
        const int instBytes = baseSize + literalExtra;

        layoutGlobal[&inst] = layoutBefore;
        const int64_t postCpInsn = postCpBytesForInstructionSpan(layoutBefore, instBytes);
        if (postCpInsn > 0 && firstPostCpLayoutByte == kSwPrefetchNoPerBbGridAnchor) {
            // First byte of this insn that lies in the post-CP zone (clamped to P(0)). This is the
            // true per-BB anchor: it accounts for alignment gaps that move the real post-CP insn
            // far past layoutStart, unlike the layoutStart-based contiguous estimate.
            firstPostCpLayoutByte = std::max(layoutBefore, kSwPrefetchFirstGlobalByte);
        }
        blockLocalBytesPostCp += postCpInsn;
        totalBytes += instBytes;
    }
    blockLocalBytes = totalBytes;
}

/// Debug after CFG RPO: per-instruction layout + running post-CP + entry `accumByte` / `accumExit`.
void debugPrintPhase1PerInstruction(BasicBlock& bb, int64_t blockGlobalByteOffset,
                                    int64_t bbEntryAccumByte,
                                    const std::unordered_map<std::string, int64_t>& labelOff,
                                    const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                    std::ostream& os) {
    int64_t totalBytes = 0;
    int64_t postCpCumul = 0;

    for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        addAlignmentPaddingFromDirectiveNode(node, blockGlobalByteOffset, totalBytes, nullptr);
        if (node->getType() == IRBase::IRType::StinkyAsmDirective) continue;
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;

        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() == GFX::PHI) continue;
        if (inst.getUnifiedOpcode() == GFX::LABEL) {
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                addAlignmentPaddingForLabelInstruction(inst, blockGlobalByteOffset, totalBytes,
                                                       nullptr);
                const int64_t labelAddr = blockGlobalByteOffset + totalBytes;
                os << "    [LABEL name=\"" << ld->label << "\" layoutGlobal=" << labelAddr
                   << " bytes, accumByte=" << bbEntryAccumByte
                   << " accumExit=" << (bbEntryAccumByte + postCpCumul) << " ";
                debugPrintCfgAccumGlobals(os, bbEntryAccumByte, postCpCumul, postCpCumul);
                os << "]\n";
            } else {
                os << "    [LABEL (no LabelData)]\n";
            }
            continue;
        }

        const int64_t layoutBefore = blockGlobalByteOffset + totalBytes;
        const int baseSize = getEffectiveBaseSizeInBytes(inst);
        const int literalExtra = getLiteralExtraBytes(inst, &labelOff, layoutBefore, asmSetSymbols);
        const int instBytes = baseSize + literalExtra;
        const int64_t postCpInsn = postCpBytesForInstructionSpan(layoutBefore, instBytes);
        const int64_t postCpCumulBefore = postCpCumul;
        const int64_t postCpCumulAfter = postCpCumul + postCpInsn;
        postCpCumul = postCpCumulAfter;
        totalBytes += instBytes;

        const int64_t layoutAfter = layoutBefore + instBytes;
        int64_t gateBefore = 0;
        int64_t gateAfter = 0;
        cfgGateIntervalBounds(layoutBefore, layoutAfter, bbEntryAccumByte, postCpCumulBefore,
                              postCpCumulAfter, gateBefore, gateAfter);
        os << "    [layoutGlobal=" << layoutBefore << " size=" << baseSize;
        if (literalExtra != 0) {
            os << "+" << literalExtra << "(literal)=" << instBytes << " bytes";
        } else {
            os << " bytes";
        }
        os << ", layoutAfter=" << layoutAfter << ", blockLocal=" << totalBytes
           << " bytes, postCpInsn=" << postCpInsn << " blockLocalBytesPostCp=" << postCpCumul
           << " accumByte=" << bbEntryAccumByte << " accumExit=" << (bbEntryAccumByte + postCpCumul)
           << " ";
        debugPrintCfgAccumGlobals(os, bbEntryAccumByte, postCpCumulBefore, postCpCumulAfter);
        os << " gateBefore=" << gateBefore << " gateAfter=" << gateAfter
           << ", opcode=" << inst.getUnifiedOpcode() << " (isa=" << inst.getISAOpcode() << ")] ";
        inst.dump(os);
        os << "\n";
    }
}

void debugDumpInsnRef(std::ostream& os, const StinkyInstruction& inst) {
    os << "opcode=" << inst.getUnifiedOpcode() << " (isa=" << inst.getISAOpcode() << ")] ";
    inst.dump(os);
}

bool cfgPathEnteredPostCp(int64_t layoutBefore, int64_t instBytes) {
    return insnSpanEndsOnOrPastCpBoundary(layoutBefore, instBytes);
}

bool cfgPathEnteredPostCpAtBlockExit(int64_t bbEntryAccum, int64_t postCpCumulAtExit) {
    return bbEntryAccum + postCpCumulAtExit > 0;
}

bool cfgIntervalContainsP(int64_t P, int64_t intervalBefore, int64_t intervalAfter) {
    return intervalBefore < P && P <= intervalAfter;
}

bool cfgGateQualifies(int64_t P, int64_t layoutBefore, int64_t layoutAfter, int64_t instBytes,
                      int64_t bbEntryAccum, int64_t postCpCumulBefore, int64_t postCpCumulAfter) {
    if (!cfgPathEnteredPostCp(layoutBefore, instBytes)) return false;
    int64_t gateBefore = 0;
    int64_t gateAfter = 0;
    cfgGateIntervalBounds(layoutBefore, layoutAfter, bbEntryAccum, postCpCumulBefore,
                          postCpCumulAfter, gateBefore, gateAfter);
    return cfgIntervalContainsP(P, gateBefore, gateAfter);
}

/// Per-BB anchored grid (`P_bb = bbGridAnchorGlobal + localK * step`): grid points are **layout**
/// offsets, so match each candidate against the insn's **layout** span `(layoutBefore,
/// layoutAfter]` (not the CFG-accum remap — that is only for the global grid). This keeps the 4 KiB
/// steps landing on interior insns. When \p P equals both the BB anchor and this insn's
/// `layoutBefore` (e.g. a fully post-CP BB with `A == layoutStart`), use a **closed** left bound so
/// the prefetch emits
/// **before** that first insn at the anchor.
bool cfgGateQualifiesPerBbAnchor(int64_t P, int64_t layoutBefore, int64_t layoutAfter,
                                 int64_t instBytes, int64_t bbGridAnchorGlobal) {
    if (!cfgPathEnteredPostCp(layoutBefore, instBytes)) return false;
    if (P == bbGridAnchorGlobal && P == layoutBefore)
        return (layoutBefore <= P && P <= layoutAfter);
    return cfgIntervalContainsP(P, layoutBefore, layoutAfter);
}

void debugPrintInsertSiteInsnContext(std::ostream& os, int64_t /*layoutBefore*/, int baseSize,
                                     int literalExtra, int instBytes, int64_t /*layoutAfter*/,
                                     int64_t blockLocal, int64_t postCpInsn,
                                     int64_t postCpCumulAfter, int64_t bbEntryAccum) {
    os << "size=" << baseSize;
    if (literalExtra != 0) {
        os << "+" << literalExtra << "(literal)=" << instBytes << " bytes";
    } else {
        os << " bytes";
    }
    os << " blockLocal=" << blockLocal << " postCpInsn=" << postCpInsn
       << " blockLocalBytesPostCp=" << postCpCumulAfter << " accumByte=" << bbEntryAccum
       << " accumExit=" << (bbEntryAccum + postCpCumulAfter);
}

void appendSwPrefetchInstPcRel(BasicBlock& bb, GfxArchID archId, AsmIRBuilder& builder,
                               std::unordered_map<std::string, int64_t>* labelOff,
                               int64_t blockGlobalByteOffset, int64_t& totalBytes,
                               const std::unordered_map<std::string, int64_t>* asmSetSymbols);

struct SwPrefetchGridWalkResult {
    int64_t kNext = 0;
    int insertCount = 0;
    int planInsert = 0;
    int skipCount = 0;
};

/// Shared CFG-gated grid walk (plan preview and/or IR insert).
SwPrefetchGridWalkResult walkSwPrefetchRelGridInBlock(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t kNextIn,
    std::unordered_map<std::string, int64_t>& labelOff,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols, GfxArchID archId,
    bool allowMutate, bool allowSwPrefetchInsertion, std::ostream* planOs,
    std::ostream* insertDbgOut, const char* debugPassTag) {
    SwPrefetchGridWalkResult result;
    result.kNext = kNextIn;
    while (swPrefetchGridOffset(result.kNext) <= blockGlobalByteOffset) ++result.kNext;

    std::unique_ptr<AsmIRBuilder> builder;
    if (allowMutate) builder = std::make_unique<AsmIRBuilder>(bb, archId);

    int64_t totalBytes = 0;
    int64_t postCpCumul = 0;
    unsigned getpcPcRelChainGuardRemaining = 0;
    int64_t totalBytesAtGetpcStart = 0;
    int64_t nextPrefetchBlockOffsetBeforeGetpc = 0;
    std::vector<IRList::iterator> getpcWindowIters;

    for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        addAlignmentPaddingFromDirectiveNode(node, blockGlobalByteOffset, totalBytes, insertDbgOut);
        if (node->getType() == IRBase::IRType::StinkyAsmDirective) continue;
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;

        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() == GFX::PHI) continue;
        if (inst.getUnifiedOpcode() == GFX::LABEL) {
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                addAlignmentPaddingForLabelInstruction(inst, blockGlobalByteOffset, totalBytes,
                                                       insertDbgOut);
                labelOff[ld->label] = blockGlobalByteOffset + totalBytes;
            }
            continue;
        }

        const bool isGetpc = instructionIsSGetpcB64(inst);
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

        const int64_t walkOffsetAtInstStart = totalBytes;
        const int64_t globalPcBefore = blockGlobalByteOffset + totalBytes;
        const int baseSize = getEffectiveBaseSizeInBytes(inst);
        const int literalExtra =
            getLiteralExtraBytes(inst, &labelOff, globalPcBefore, asmSetSymbols);
        const int instBytes = baseSize + literalExtra;
        const int64_t globalPcAfter = globalPcBefore + instBytes;
        const int64_t postCpInsn = postCpBytesForInstructionSpan(globalPcBefore, instBytes);
        const int64_t postCpCumulBefore = postCpCumul;
        const int64_t postCpCumulAfter = postCpCumul + postCpInsn;
        int64_t gateBefore = 0;
        int64_t gateAfter = 0;
        cfgGateIntervalBounds(globalPcBefore, globalPcAfter, bbEntryAccum, postCpCumulBefore,
                              postCpCumulAfter, gateBefore, gateAfter);

        bool redirectRewalkAbsorbedCurrentInstSizes = false;
        for (;;) {
            const int64_t P = swPrefetchGridOffset(result.kNext);
            if (P <= blockGlobalByteOffset) {
                ++result.kNext;
                continue;
            }
            if (P > gateAfter) break;
            if (P < gateBefore) {
                ++result.kNext;
                continue;
            }

            const bool layoutGate = P >= kSwPrefetchFirstGlobalByte;
            const bool cfgGate =
                cfgGateQualifies(P, globalPcBefore, globalPcAfter, instBytes, bbEntryAccum,
                                 postCpCumulBefore, postCpCumulAfter);
            const bool wouldInsert = layoutGate && cfgGate;
            const bool getpcRedirect =
                getpcPcRelChainGuardRemaining > 0u && !getpcWindowIters.empty();
            const bool pathEntered = cfgPathEnteredPostCp(globalPcBefore, instBytes);

            if (planOs != nullptr) {
                *planOs << "  [insert-site k=" << result.kNext << " P=" << P
                        << " label=label_SWprefetch_" << result.kNext << " BB=\"" << bb.getLabel()
                        << "\" insertPoint="
                        << (getpcRedirect ? "before_getpc_redirect" : "before_insn")
                        << " layoutGate=" << (layoutGate ? "yes" : "no")
                        << " cfgGate=" << (cfgGate ? "yes" : "no")
                        << " pathEntered=" << (pathEntered ? "yes" : "no")
                        << " action=" << (wouldInsert ? "PLAN_INSERT" : "SKIP")
                        << " gateBefore=" << gateBefore << " gateAfter=" << gateAfter
                        << " layoutBefore=" << globalPcBefore << " layoutAfter=" << globalPcAfter
                        << " ";
                debugPrintCfgAccumGlobals(*planOs, bbEntryAccum, postCpCumulBefore,
                                          postCpCumulAfter);
                *planOs << " ";
                debugPrintInsertSiteInsnContext(*planOs, globalPcBefore, baseSize, literalExtra,
                                                instBytes, globalPcAfter, totalBytes + instBytes,
                                                postCpInsn, postCpCumulAfter, bbEntryAccum);
                *planOs << " ";
                if (getpcRedirect) {
                    *planOs << "insertBefore=";
                    debugDumpInsnRef(*planOs, getStinkyInst(getpcWindowIters.front()));
                } else {
                    *planOs << "insertBefore=";
                    debugDumpInsnRef(*planOs, inst);
                }
                *planOs << " anchor=";
                debugDumpInsnRef(*planOs, inst);
                *planOs << "\n";
            }

            if (wouldInsert) {
                if (planOs != nullptr) ++result.planInsert;
                if (allowMutate && allowSwPrefetchInsertion && builder != nullptr) {
                    const std::string name =
                        std::string("label_SWprefetch_") + std::to_string(result.kNext);
                    if (!swPrefetchLabelNameExists(bb, name)) {
                        labelOff[name] = globalPcAfter;
                        if (getpcRedirect) {
                            insertPrefetchBeforeGetpcAndRewalkWindow(
                                bb, getpcWindowIters.front(), getpcWindowIters,
                                totalBytesAtGetpcStart, nextPrefetchBlockOffsetBeforeGetpc,
                                blockGlobalByteOffset, archId, *builder, labelOff, totalBytes,
                                insertDbgOut, debugPassTag, asmSetSymbols);
                            redirectRewalkAbsorbedCurrentInstSizes = true;
                        } else {
                            (void)insertSwPrefetchInstPcRelBefore(
                                bb, it, archId, *builder, &labelOff, blockGlobalByteOffset,
                                walkOffsetAtInstStart, totalBytes, asmSetSymbols);
                        }
                        ++result.insertCount;
                    }
                }
            } else if (planOs != nullptr) {
                ++result.skipCount;
            }
            ++result.kNext;
        }

        if (!redirectRewalkAbsorbedCurrentInstSizes) totalBytes += instBytes;
        postCpCumul = postCpCumulAfter;

        if (!isGetpc && getpcPcRelChainGuardRemaining > 0u) {
            --getpcPcRelChainGuardRemaining;
            if (getpcPcRelChainGuardRemaining == 0u) getpcWindowIters.clear();
        }
    }

    getpcWindowIters.clear();
    const int64_t blockEndGlobal = blockGlobalByteOffset + totalBytes;
    const int64_t accumAfterGlobalEnd = cfgAccumAfterGlobal(bbEntryAccum, postCpCumul);
    for (;;) {
        const int64_t P = swPrefetchGridOffset(result.kNext);
        if (P < blockGlobalByteOffset) {
            ++result.kNext;
            continue;
        }
        if (P > blockEndGlobal) break;

        const bool layoutGate = P >= kSwPrefetchFirstGlobalByte;
        const bool pathEntered = cfgPathEnteredPostCpAtBlockExit(bbEntryAccum, postCpCumul);
        const bool tailInterval = accumAfterGlobalEnd < P && P <= blockEndGlobal;
        const bool cfgGate = pathEntered && tailInterval;
        const bool wouldInsert = layoutGate && cfgGate;

        if (planOs != nullptr) {
            *planOs << "  [insert-site k=" << result.kNext << " P=" << P
                    << " label=label_SWprefetch_" << result.kNext << " BB=\"" << bb.getLabel()
                    << "\" insertPoint=bb_end_append"
                    << " layoutGate=" << (layoutGate ? "yes" : "no")
                    << " cfgGate=" << (cfgGate ? "yes" : "no")
                    << " pathEntered=" << (pathEntered ? "yes" : "no")
                    << " action=" << (wouldInsert ? "PLAN_INSERT" : "SKIP")
                    << " gateBefore=" << accumAfterGlobalEnd << " gateAfter=" << blockEndGlobal
                    << " layoutBefore=" << blockEndGlobal << " layoutAfter=" << blockEndGlobal
                    << " ";
            debugPrintCfgAccumGlobals(*planOs, bbEntryAccum, postCpCumul, postCpCumul);
            *planOs << " anchorLayoutGlobal=" << blockEndGlobal << "]\n";
        }

        if (wouldInsert) {
            if (planOs != nullptr) ++result.planInsert;
            if (allowMutate && allowSwPrefetchInsertion && builder != nullptr) {
                const std::string name =
                    std::string("label_SWprefetch_") + std::to_string(result.kNext);
                if (!swPrefetchLabelNameExists(bb, name)) {
                    labelOff[name] = blockEndGlobal;
                    appendSwPrefetchInstPcRel(bb, archId, *builder, &labelOff,
                                              blockGlobalByteOffset, totalBytes, asmSetSymbols);
                    ++result.insertCount;
                }
            }
            ++result.kNext;
            while (true) {
                const int64_t Pcoalesced = swPrefetchGridOffset(result.kNext);
                if (Pcoalesced > blockEndGlobal) break;
                if (Pcoalesced <= accumAfterGlobalEnd) break;
                if (planOs != nullptr) {
                    *planOs << "  [insert-site k=" << result.kNext << " P=" << Pcoalesced
                            << " BB=\"" << bb.getLabel()
                            << "\" insertPoint=bb_end_append action=SKIP tail_coalesced"
                            << " (first tail PLAN_INSERT at blockEnd=" << blockEndGlobal << ")]\n";
                }
                ++result.skipCount;
                ++result.kNext;
            }
            break;
        }
        if (planOs != nullptr) ++result.skipCount;
        ++result.kNext;
    }

    return result;
}

std::string swPrefetchPerBbRelLabelName(int64_t blockGlobalByteOffset, int64_t localK) {
    return std::string("label_SWprefetch_bbrel_") + std::to_string(blockGlobalByteOffset) + "_" +
           std::to_string(localK);
}

/// Same as `walkSwPrefetchRelGridInBlock`, but grid lines are
/// `bbGridAnchorGlobal + localK * kSwPrefetchSpacingBytes` (per-BB post-CP anchor). Dual gate and
/// getpc redirect unchanged. `result.kNext` is the **per-BB local** index (restart at 0 each BB).
///
/// **Anchor at BB start:** advance with `P < layoutStart` (not `<=`) so `P_bb(0)==A==layoutStart`
/// is not skipped. **`cfgGateQualifiesPerBbAnchor`** uses a closed left bound when
/// `P==A==layoutBefore` so the prefetch can insert **before** the first real insn at the anchor.
SwPrefetchGridWalkResult walkSwPrefetchRelGridInBlockPerBbAnchor(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t bbGridAnchorGlobal,
    int64_t kLocalNextIn, std::unordered_map<std::string, int64_t>& labelOff,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols, GfxArchID archId,
    bool allowMutate, bool allowSwPrefetchInsertion, std::ostream* planOs,
    std::ostream* insertDbgOut, const char* debugPassTag) {
    SwPrefetchGridWalkResult result;
    result.kNext = kLocalNextIn;
    if (bbGridAnchorGlobal == kSwPrefetchNoPerBbGridAnchor) return result;

    while (swPrefetchPerBbAnchorGridOffset(result.kNext, bbGridAnchorGlobal) <
           blockGlobalByteOffset)
        ++result.kNext;

    std::unique_ptr<AsmIRBuilder> builder;
    if (allowMutate) builder = std::make_unique<AsmIRBuilder>(bb, archId);

    int64_t totalBytes = 0;
    int64_t postCpCumul = 0;
    unsigned getpcPcRelChainGuardRemaining = 0;
    int64_t totalBytesAtGetpcStart = 0;
    int64_t nextPrefetchBlockOffsetBeforeGetpc = 0;
    std::vector<IRList::iterator> getpcWindowIters;

    for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
        IRBase* node = it.getNodePtr();
        addAlignmentPaddingFromDirectiveNode(node, blockGlobalByteOffset, totalBytes, insertDbgOut);
        if (node->getType() == IRBase::IRType::StinkyAsmDirective) continue;
        if (node->getType() != IRBase::IRType::StinkyTofu) continue;

        StinkyInstruction& inst = getStinkyInst(it);
        if (inst.getUnifiedOpcode() == GFX::PHI) continue;
        if (inst.getUnifiedOpcode() == GFX::LABEL) {
            if (const LabelData* ld = inst.getModifier<LabelData>()) {
                addAlignmentPaddingForLabelInstruction(inst, blockGlobalByteOffset, totalBytes,
                                                       insertDbgOut);
                labelOff[ld->label] = blockGlobalByteOffset + totalBytes;
            }
            continue;
        }

        const bool isGetpc = instructionIsSGetpcB64(inst);
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

        const int64_t walkOffsetAtInstStart = totalBytes;
        const int64_t globalPcBefore = blockGlobalByteOffset + totalBytes;
        const int baseSize = getEffectiveBaseSizeInBytes(inst);
        const int literalExtra =
            getLiteralExtraBytes(inst, &labelOff, globalPcBefore, asmSetSymbols);
        const int instBytes = baseSize + literalExtra;
        const int64_t globalPcAfter = globalPcBefore + instBytes;
        const int64_t postCpInsn = postCpBytesForInstructionSpan(globalPcBefore, instBytes);
        const int64_t postCpCumulBefore = postCpCumul;
        const int64_t postCpCumulAfter = postCpCumul + postCpInsn;
        // Per-BB anchor grid is in layout coordinates: gate each grid point against this insn's
        // layout span so 4 KiB steps land on interior insns (no CFG-accum remap).
        const int64_t gateBefore = globalPcBefore;
        const int64_t gateAfter = globalPcAfter;

        bool redirectRewalkAbsorbedCurrentInstSizes = false;
        for (;;) {
            const int64_t P = swPrefetchPerBbAnchorGridOffset(result.kNext, bbGridAnchorGlobal);
            if (P < blockGlobalByteOffset) {
                ++result.kNext;
                continue;
            }
            if (P > gateAfter) break;
            if (P < gateBefore) {
                ++result.kNext;
                continue;
            }

            const bool layoutGate = P >= kSwPrefetchFirstGlobalByte;
            const bool cfgGate = cfgGateQualifiesPerBbAnchor(P, globalPcBefore, globalPcAfter,
                                                             instBytes, bbGridAnchorGlobal);
            const bool wouldInsert = layoutGate && cfgGate;
            const bool getpcRedirect =
                getpcPcRelChainGuardRemaining > 0u && !getpcWindowIters.empty();
            const bool pathEntered = cfgPathEnteredPostCp(globalPcBefore, instBytes);

            if (planOs != nullptr) {
                *planOs << "  [insert-site grid=per_bb_anchor bbAnchor=" << bbGridAnchorGlobal
                        << " localK=" << result.kNext << " P=" << P << " label="
                        << swPrefetchPerBbRelLabelName(blockGlobalByteOffset, result.kNext)
                        << " BB=\"" << bb.getLabel() << "\" insertPoint="
                        << (getpcRedirect ? "before_getpc_redirect" : "before_insn")
                        << " layoutGate=" << (layoutGate ? "yes" : "no")
                        << " cfgGate=" << (cfgGate ? "yes" : "no")
                        << " pathEntered=" << (pathEntered ? "yes" : "no")
                        << " action=" << (wouldInsert ? "PLAN_INSERT" : "SKIP")
                        << " gateBefore=" << gateBefore << " gateAfter=" << gateAfter
                        << " layoutBefore=" << globalPcBefore << " layoutAfter=" << globalPcAfter
                        << " ";
                debugPrintCfgAccumGlobals(*planOs, bbEntryAccum, postCpCumulBefore,
                                          postCpCumulAfter);
                *planOs << " ";
                debugPrintInsertSiteInsnContext(*planOs, globalPcBefore, baseSize, literalExtra,
                                                instBytes, globalPcAfter, totalBytes + instBytes,
                                                postCpInsn, postCpCumulAfter, bbEntryAccum);
                *planOs << " ";
                if (getpcRedirect) {
                    *planOs << "insertBefore=";
                    debugDumpInsnRef(*planOs, getStinkyInst(getpcWindowIters.front()));
                } else {
                    *planOs << "insertBefore=";
                    debugDumpInsnRef(*planOs, inst);
                }
                *planOs << " anchor=";
                debugDumpInsnRef(*planOs, inst);
                *planOs << "\n";
            }

            if (wouldInsert) {
                if (planOs != nullptr) ++result.planInsert;
                if (allowMutate && allowSwPrefetchInsertion && builder != nullptr) {
                    const std::string name =
                        swPrefetchPerBbRelLabelName(blockGlobalByteOffset, result.kNext);
                    if (!swPrefetchLabelNameExists(bb, name)) {
                        labelOff[name] = globalPcAfter;
                        if (getpcRedirect) {
                            insertPrefetchBeforeGetpcAndRewalkWindow(
                                bb, getpcWindowIters.front(), getpcWindowIters,
                                totalBytesAtGetpcStart, nextPrefetchBlockOffsetBeforeGetpc,
                                blockGlobalByteOffset, archId, *builder, labelOff, totalBytes,
                                insertDbgOut, debugPassTag, asmSetSymbols);
                            redirectRewalkAbsorbedCurrentInstSizes = true;
                        } else {
                            (void)insertSwPrefetchInstPcRelBefore(
                                bb, it, archId, *builder, &labelOff, blockGlobalByteOffset,
                                walkOffsetAtInstStart, totalBytes, asmSetSymbols);
                        }
                        ++result.insertCount;
                    }
                }
            } else if (planOs != nullptr) {
                ++result.skipCount;
            }
            ++result.kNext;
        }

        if (!redirectRewalkAbsorbedCurrentInstSizes) totalBytes += instBytes;
        postCpCumul = postCpCumulAfter;

        if (!isGetpc && getpcPcRelChainGuardRemaining > 0u) {
            --getpcPcRelChainGuardRemaining;
            if (getpcPcRelChainGuardRemaining == 0u) getpcWindowIters.clear();
        }
    }

    getpcWindowIters.clear();
    const int64_t blockEndGlobal = blockGlobalByteOffset + totalBytes;
    // Per-BB grid is layout-based: interior insns already cover `(A, blockEnd]`, so the tail append
    // only fires for a grid point strictly past the last insn's layout span (normally none).
    const int64_t tailLowerLayout = blockEndGlobal;
    for (;;) {
        const int64_t P = swPrefetchPerBbAnchorGridOffset(result.kNext, bbGridAnchorGlobal);
        if (P < blockGlobalByteOffset) {
            ++result.kNext;
            continue;
        }
        if (P > blockEndGlobal) break;

        const bool layoutGate = P >= kSwPrefetchFirstGlobalByte;
        const bool pathEntered = cfgPathEnteredPostCpAtBlockExit(bbEntryAccum, postCpCumul);
        const bool tailInterval = tailLowerLayout < P && P <= blockEndGlobal;
        const bool cfgGate = pathEntered && tailInterval;
        const bool wouldInsert = layoutGate && cfgGate;

        if (planOs != nullptr) {
            *planOs << "  [insert-site grid=per_bb_anchor bbAnchor=" << bbGridAnchorGlobal
                    << " localK=" << result.kNext << " P=" << P
                    << " label=" << swPrefetchPerBbRelLabelName(blockGlobalByteOffset, result.kNext)
                    << " BB=\"" << bb.getLabel() << "\" insertPoint=bb_end_append"
                    << " layoutGate=" << (layoutGate ? "yes" : "no")
                    << " cfgGate=" << (cfgGate ? "yes" : "no")
                    << " pathEntered=" << (pathEntered ? "yes" : "no")
                    << " action=" << (wouldInsert ? "PLAN_INSERT" : "SKIP")
                    << " gateBefore=" << tailLowerLayout << " gateAfter=" << blockEndGlobal
                    << " layoutBefore=" << blockEndGlobal << " layoutAfter=" << blockEndGlobal
                    << " ";
            debugPrintCfgAccumGlobals(*planOs, bbEntryAccum, postCpCumul, postCpCumul);
            *planOs << " anchorLayoutGlobal=" << blockEndGlobal << "]\n";
        }

        if (wouldInsert) {
            if (planOs != nullptr) ++result.planInsert;
            if (allowMutate && allowSwPrefetchInsertion && builder != nullptr) {
                const std::string name =
                    swPrefetchPerBbRelLabelName(blockGlobalByteOffset, result.kNext);
                if (!swPrefetchLabelNameExists(bb, name)) {
                    labelOff[name] = blockEndGlobal;
                    appendSwPrefetchInstPcRel(bb, archId, *builder, &labelOff,
                                              blockGlobalByteOffset, totalBytes, asmSetSymbols);
                    ++result.insertCount;
                }
            }
            ++result.kNext;
            while (true) {
                const int64_t Pcoalesced =
                    swPrefetchPerBbAnchorGridOffset(result.kNext, bbGridAnchorGlobal);
                if (Pcoalesced > blockEndGlobal) break;
                if (Pcoalesced <= tailLowerLayout) break;
                if (planOs != nullptr) {
                    *planOs << "  [insert-site localK=" << result.kNext << " P=" << Pcoalesced
                            << " BB=\"" << bb.getLabel()
                            << "\" insertPoint=bb_end_append action=SKIP tail_coalesced"
                            << " (first tail PLAN_INSERT at blockEnd=" << blockEndGlobal << ")]\n";
                }
                ++result.skipCount;
                ++result.kNext;
            }
            break;
        }
        if (planOs != nullptr) ++result.skipCount;
        ++result.kNext;
    }

    return result;
}

GfxArchID gfxArchFromBasicBlock(const BasicBlock& bb) {
    const Function* func = bb.getParentFunc();
    if (func == nullptr) return getGfxArchID(12, 5, 0);
    const auto& archArr = func->getGemmTileConfig().arch;
    return getGfxArchID(static_cast<uint32_t>(archArr[0]), static_cast<uint32_t>(archArr[1]),
                        static_cast<uint32_t>(archArr[2]));
}

/// Dry-run grid walk (phase 2 preview): CFG-interval anchor + dual gate.
int64_t debugPlanInsertSitesInBlock(BasicBlock& bb, int64_t blockGlobalByteOffset,
                                    int64_t bbEntryAccum, int64_t kNextIn,
                                    const std::unordered_map<std::string, int64_t>& labelOff,
                                    const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                    std::ostream& os, int& outPlanInsert, int& outSkip) {
    std::unordered_map<std::string, int64_t> localLabelOff = labelOff;
    const SwPrefetchGridWalkResult result = walkSwPrefetchRelGridInBlock(
        bb, blockGlobalByteOffset, bbEntryAccum, kNextIn, localLabelOff, asmSetSymbols,
        gfxArchFromBasicBlock(bb), false, true, &os, nullptr, nullptr);
    outPlanInsert += result.planInsert;
    outSkip += result.skipCount;
    return result.kNext;
}

int64_t debugPlanInsertSitesInBlockPerBbAnchor(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t bbGridAnchorGlobal,
    const std::unordered_map<std::string, int64_t>& labelOff,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols, std::ostream& os,
    int& outPlanInsert, int& outSkip) {
    if (bbGridAnchorGlobal == kSwPrefetchNoPerBbGridAnchor) return 0;
    std::unordered_map<std::string, int64_t> localLabelOff = labelOff;
    const SwPrefetchGridWalkResult result = walkSwPrefetchRelGridInBlockPerBbAnchor(
        bb, blockGlobalByteOffset, bbEntryAccum, bbGridAnchorGlobal, 0, localLabelOff,
        asmSetSymbols, gfxArchFromBasicBlock(bb), false, true, &os, nullptr, nullptr);
    outPlanInsert += result.planInsert;
    outSkip += result.skipCount;
    return result.kNext;
}

void debugPrintPhase1PlannedInsertSites(
    Function& func, const SwPrefetchRelPhase1Accum& phase1,
    const std::unordered_map<std::string, int64_t>& labelOff,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols, std::ostream& os,
    const char* tag, bool usePerBbAnchorPreview) {
    os << "[" << tag << "] Phase 1 planned insert sites (phase 2 preview"
       << (usePerBbAnchorPreview ? ", per-BB anchor grid" : ", global P(k) grid")
       << "), P(0)=" << kSwPrefetchFirstGlobalByte
       << " totalLayoutBytes=" << phase1.totalLayoutBytes << "\n";

    if (phase1.totalLayoutBytes <= kSwPrefetchFirstGlobalByte) {
        os << "  (none: totalLayoutBytes=" << phase1.totalLayoutBytes
           << " <= P(0)=" << kSwPrefetchFirstGlobalByte << ")\n";
        return;
    }

    int planInsert = 0;
    int skip = 0;
    for (BasicBlock& bb : func) {
        BasicBlock* bp = &bb;
        if (usePerBbAnchorPreview) {
            // Preview walks pre-insert layout (blockGlobalByteOffset == layoutStart), so the
            // recorded first post-CP byte is already in the right coordinate space (no drift
            // adjustment).
            const int64_t anchor = phase1.firstPostCpLayoutByte.at(bp);
            (void)debugPlanInsertSitesInBlockPerBbAnchor(bb, phase1.layoutStart.at(bp),
                                                         phase1.accumByte.at(bp), anchor, labelOff,
                                                         asmSetSymbols, os, planInsert, skip);
        } else {
            // Per-BB kNextIn=0: same P(k) may PLAN_INSERT in multiple branch BBs.
            (void)debugPlanInsertSitesInBlock(bb, phase1.layoutStart.at(bp),
                                              phase1.accumByte.at(bp), 0, labelOff, asmSetSymbols,
                                              os, planInsert, skip);
        }
    }

    os << "[" << tag << "] Phase 1 planned insert sites summary: PLAN_INSERT=" << planInsert
       << " SKIP=" << skip
       << (usePerBbAnchorPreview ? " (per-BB anchor preview)\n"
                                 : " (per-BB kNextIn=0 multi-arm sweep)\n");
}

void appendSwPrefetchInstPcRel(BasicBlock& bb, GfxArchID archId, AsmIRBuilder& builder,
                               std::unordered_map<std::string, int64_t>* labelOff,
                               int64_t blockGlobalByteOffset, int64_t& totalBytes,
                               const std::unordered_map<std::string, int64_t>* asmSetSymbols) {
    const HwInstDesc* pfMc = getMCIDByUOp(GFX::s_prefetch_inst_pc_rel, archId);
    if (pfMc == nullptr) return;

    // XNACK safety: one s_wait_xcnt 0 before the tail-flush prefetch (see insert-before variant).
    if (kSwPrefetchEmitXnackWait) {
        if (const HwInstDesc* xcntMc = getMCIDByUOp(GFX::s_wait_xcnt, archId)) {
            StinkyInstruction* xcntInst = builder.create(xcntMc);
            xcntInst->addSrcReg(StinkyRegister(0));  // xcnt = 0
            const int64_t gXcnt = blockGlobalByteOffset + totalBytes;
            const int xcntB = getEffectiveBaseSizeInBytes(*xcntInst) +
                              getLiteralExtraBytes(*xcntInst, labelOff, gXcnt, asmSetSymbols);
            totalBytes += xcntB;
            bb.appendIR(xcntInst);
        }
    }

    StinkyInstruction* prefetchInst = builder.create(pfMc);
    addSwPrefetchInstPcRelOperands(*prefetchInst);

    const int64_t gPf = blockGlobalByteOffset + totalBytes;
    const int pfB = getEffectiveBaseSizeInBytes(*prefetchInst) +
                    getLiteralExtraBytes(*prefetchInst, labelOff, gPf, asmSetSymbols);
    totalBytes += pfB;

    bb.appendIR(prefetchInst);
}
}  // namespace

namespace stinkytofu {

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
                            bool allowSwPrefetchInsertion, const char* debugPassTag) {
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
            const int64_t P = swPrefetchGridOffset(kNext);
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
                            builder, labelOff, totalBytes, dbgOut, debugPassTag, asmSetSymbols);
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
        const int64_t P = swPrefetchGridOffset(kNext);
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
                ++kNext;
                while (swPrefetchGridOffset(kNext) <= blockEndGlobal) ++kNext;
                break;
            }
        }
        ++kNext;
    }
}

/// Debug-only: list P(k) grid boundaries that fall in this basic block.
/// Per-instruction placement (including emitted `s_prefetch_inst_pc_rel`) is
/// already printed by **`accumulateInstructionSize`** above (global **`total=`**).
void debugPrintSwPrefetchGrid(std::ostream& os, const std::string& bbLabel,
                              int64_t blockGlobalStart, int64_t blockBytes,
                              const char* debugPassTag) {
    const int64_t blockEndGlobal = blockGlobalStart + blockBytes;
    const char* tag = debugPassTag != nullptr ? debugPassTag : "SwInstructionPrefetchRelStaticPass";

    os << "[" << tag
       << "] SW prefetch grid (pseudo labels label_SWprefetch_<k>), "
          "basic block \""
       << bbLabel << "\" blockGlobalStart=" << blockGlobalStart << " blockSize=" << blockBytes
       << " P(k)=" << kSwPrefetchFirstGlobalByte << "+k*" << kSwPrefetchSpacingBytes << "\n";

    if (blockEndGlobal < kSwPrefetchFirstGlobalByte) {
        os << "  (none: block end " << blockEndGlobal << " < first threshold "
           << kSwPrefetchFirstGlobalByte << ")\n";
        return;
    }

    for (int64_t k = 0;; ++k) {
        const int64_t P = swPrefetchGridOffset(k);
        if (P > blockEndGlobal) break;
        if (P < blockGlobalStart) continue;

        os << "  [label_SWprefetch_" << k << " k=" << k << " global_P=" << P
           << " local_P=" << (P - blockGlobalStart)
           << " (see accumulate dump: insn with total=" << P << ")]\n";
    }
}

int insertSwPrefetchLabelsDynamic(BasicBlock& bb, int64_t blockGlobalByteOffset,
                                  int64_t bbEntryAccum, int64_t kNextIn, GfxArchID archId,
                                  std::ostream* dbgOut,
                                  const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                  bool allowSwPrefetchInsertion, const char* debugPassTag) {
    std::unordered_map<std::string, int64_t> labelOff;
    const SwPrefetchGridWalkResult result = walkSwPrefetchRelGridInBlock(
        bb, blockGlobalByteOffset, bbEntryAccum, kNextIn, labelOff, asmSetSymbols, archId,
        allowSwPrefetchInsertion, allowSwPrefetchInsertion, nullptr, dbgOut, debugPassTag);
    return result.insertCount;
}

int insertSwPrefetchLabelsDynamicPerBbAnchor(
    BasicBlock& bb, int64_t blockGlobalByteOffset, int64_t bbEntryAccum, int64_t bbGridAnchorGlobal,
    int64_t kLocalNextIn, GfxArchID archId, std::ostream* dbgOut,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols, bool allowSwPrefetchInsertion,
    const char* debugPassTag) {
    std::unordered_map<std::string, int64_t> labelOff;
    if (bbGridAnchorGlobal == kSwPrefetchNoPerBbGridAnchor) return 0;
    const SwPrefetchGridWalkResult result = walkSwPrefetchRelGridInBlockPerBbAnchor(
        bb, blockGlobalByteOffset, bbEntryAccum, bbGridAnchorGlobal, kLocalNextIn, labelOff,
        asmSetSymbols, archId, allowSwPrefetchInsertion, allowSwPrefetchInsertion, nullptr, dbgOut,
        debugPassTag);
    return result.insertCount;
}

void computeSwPrefetchRelPhase1Accum(Function& func,
                                     const std::unordered_map<std::string, int64_t>* asmSetSymbols,
                                     SwPrefetchRelPhase1Accum& out, std::ostream* dbgOut,
                                     const char* debugPassTag, bool phase2UsesPerBbAnchorGrid) {
    const char* tag =
        debugPassTag != nullptr ? debugPassTag : "SwInstructionPrefetchRelDynamicPass";

    out.layoutStart.clear();
    out.blockLocalBytes.clear();
    out.blockLocalBytesPostCp.clear();
    out.firstPostCpLayoutByte.clear();
    out.accumByte.clear();
    out.accumExit.clear();
    out.layoutGlobal.clear();
    out.totalLayoutBytes = 0;

    std::unordered_map<std::string, int64_t> labelOff;
    int64_t layoutBase = 0;

    for (BasicBlock& bb : func) {
        out.layoutStart[&bb] = layoutBase;
        int64_t localBytes = 0;
        int64_t localPostCp = 0;
        int64_t firstPostCp = kSwPrefetchNoPerBbGridAnchor;
        walkBlockLayoutAndPostCp(bb, layoutBase, labelOff, asmSetSymbols, out.layoutGlobal,
                                 localBytes, localPostCp, firstPostCp);
        out.blockLocalBytes[&bb] = localBytes;
        out.blockLocalBytesPostCp[&bb] = localPostCp;
        out.firstPostCpLayoutByte[&bb] = firstPostCp;
        // Default for unreachable BBs (RPO only visits entry-reachable blocks).
        out.accumByte[&bb] = 0;
        out.accumExit[&bb] = localPostCp;
        layoutBase += localBytes;
    }
    out.totalLayoutBytes = layoutBase;

    const std::unordered_set<CfgEdge, CfgEdgeHash> backEdges = buildCfgBackEdgeSet(func);
    BasicBlock* entry = func.getEntryBlock();

    traverseCFGInRPO(func, [&](BasicBlock* bb) {
        if (bb == entry) {
            out.accumByte[bb] = 0;
        } else {
            std::vector<BasicBlock*> frontPreds;
            frontPreds.reserve(bb->getPredecessors().size());
            for (BasicBlock* pred : bb->getPredecessors()) {
                if (!isCfgBackEdge(pred, bb, backEdges)) frontPreds.push_back(pred);
            }

            if (frontPreds.empty()) {
                out.accumByte[bb] = 0;
            } else if (frontPreds.size() == 1u) {
                BasicBlock* pred = frontPreds.front();
                const int64_t predExit = out.accumByte[pred] + out.blockLocalBytesPostCp[pred];
                out.accumByte[bb] = predExit;
            } else {
                int64_t maxPredExit = 0;
                for (BasicBlock* pred : frontPreds) {
                    const int64_t predExit = out.accumByte[pred] + out.blockLocalBytesPostCp[pred];
                    maxPredExit = std::max(maxPredExit, predExit);
                }
                out.accumByte[bb] = maxPredExit;
            }
        }

        out.accumExit[bb] = out.accumByte[bb] + out.blockLocalBytesPostCp[bb];
    });

    if (dbgOut == nullptr) return;

    debugPrintPhase1PlannedInsertSites(func, out, labelOff, asmSetSymbols, *dbgOut, tag,
                                       phase2UsesPerBbAnchorGrid);

    *dbgOut << "[" << tag << "] Phase 1 accumulate (no insert), P(0)=" << kSwPrefetchFirstGlobalByte
            << " totalLayoutBytes=" << out.totalLayoutBytes << "\n";
    for (BasicBlock& bb : func) {
        BasicBlock* bp = &bb;
        *dbgOut << "  BB \"" << bb.getLabel() << "\" layoutStart=" << out.layoutStart[bp]
                << " accumByte(entry)=" << out.accumByte[bp] << "\n";
        debugPrintPhase1PerInstruction(bb, out.layoutStart[bp], out.accumByte[bp], labelOff,
                                       asmSetSymbols, *dbgOut);
        *dbgOut << "  BB \"" << bb.getLabel() << "\" summary layoutStart=" << out.layoutStart[bp]
                << " blockLocalBytes=" << out.blockLocalBytes[bp]
                << " blockLocalBytesPostCp=" << out.blockLocalBytesPostCp[bp]
                << " firstPostCpLayoutByte=" << out.firstPostCpLayoutByte[bp]
                << " accumByte=" << out.accumByte[bp] << " accumExit=" << out.accumExit[bp] << "\n";
    }
}
}  // namespace stinkytofu
