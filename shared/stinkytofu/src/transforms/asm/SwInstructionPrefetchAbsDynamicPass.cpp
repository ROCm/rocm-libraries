// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// SwInstructionPrefetchAbsDynamicPass — absolute SW prefetch for large gfx1250 kernels,
/// targeting the blocks the kernel branches to at run time (not a fixed byte grid).
///
/// Applies to the post-CP region (totalLayoutBytes > P(0)=32640); the abs-static pass owns
/// (32640, 65536] and this pass owns > 65536. TensileLite-specific: it keys off TensileLite's
/// global-write label names (GW_B0_MB / GW_B0_GSU1 / GW_B1_GSU1, etc.).
///
/// When debug output is enabled, run() first performs a read-only analysis, detectAndDumpD0(),
/// which classifies the 3-case GSU/beta global-write dispatch (A=GW_B0_{MB,MBSK} / B=GW_B0_GSU1 /
/// C=GW_B1_GSU1), applies the CP-window filter, and logs it to the debug file. Then, when
/// totalLayoutBytes > 65536 and a reserved baseSgpr is available, it emits a predicated prefetch
/// ladder (getpc + `s_add_i32 label,4` + N×`s_prefetch_inst`) immediately after
/// `label_MultiGemmEnd`, one arm per GSU/beta case. The ladder is skipped for Stream-K, GSU0
/// (undefined sgprGSU), and no-beta kernels; MBSK-reduction / activation are logged but not
/// handled.
///
/// Optional near-boundary cover (enabled in production via the module factory): an unconditional
/// burst of dynamic width `coverN` prepended before the ladder, covering the once-through fast
/// path just past the CP window ([P(0), P(0)+coverN*4096)). coverN is sized to the fast-path
/// boundary (label_TailLoopBegin* / label_OptNLL_End / shallowest GW_*>P(0)) and traded against
/// the per-arm width armN under coverN + armN <= 8 (armFloor=4 => coverN in {0..4}); coverN=0
/// (boundary already CP-resident) emits nothing. It targets a new `label_SW_PrefetchAbs_CpBoundary`
/// anchored (post-insertion re-accumulate) at the last insn <= P(0). The cover is independent of
/// the GSU/beta condition (it needs neither), so it also emits on GSU0/no-beta kernels; Stream-K
/// still bails both.

#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsDynamicPass.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

namespace stinkytofu {

namespace {
// Return the entry-BB insertion point AFTER the gfx1250 hardware-entrypoint prologue
// (`global_prefetch_b8` + `v_nop`, inserted by InsertInitialUnclausedVmemPass) so the entry
// CP-cover burst is placed after the prologue and the prologue stays the kernel's first executed
// instruction(s). Falls back to begin() when no prologue is present (isolated unit tests /
// non-gfx1250), preserving prior behavior and the empty-entry-BB APPEND (begin()==end()).
// Returns {BB, iterator} for the entry CP-cover insertion point: immediately AFTER the gfx1250
// hardware-entrypoint prologue (global_prefetch_b8 [+ v_nop]) wherever it lives in the function
// (it may be in a later BB than getEntryBlock(), e.g. after an empty preamble + label_ASM_Start),
// so the prologue stays first. Falls back to {entryBB, begin()} when no prologue is present,
// preserving prior behavior and the empty-entry-BB APPEND (begin()==end()).
std::pair<BasicBlock*, IRList::iterator> entryBurstInsertPoint(Function& func) {
    for (BasicBlock& bb : func) {
        for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
            IRBase* node = it.getNodePtr();
            if (node->getType() != IRBase::IRType::StinkyTofu) continue;  // asm directives
            const StinkyInstruction* inst = static_cast<const StinkyInstruction*>(node);
            // Skip pseudo nodes using the SAME predicate InsertInitialUnclausedVmemPass uses to
            // find the "first real instruction" (LABEL/PHI/FENCE/placement marker), so both passes
            // agree on where the prologue sits.
            if (isPseudoInst(inst)) continue;
            if (inst->getUnifiedOpcode() == GFX::global_prefetch_b8) {
                IRList::iterator after = it;
                ++after;  // past global_prefetch_b8
                if (after != bb.end()) {
                    IRBase* n2 = after.getNodePtr();
                    if (n2->getType() == IRBase::IRType::StinkyTofu &&
                        static_cast<const StinkyInstruction*>(n2)->getUnifiedOpcode() == GFX::v_nop)
                        ++after;  // past v_nop
                }
                return {&bb, after};
            }
            BasicBlock* entry = func.getEntryBlock();
            if (entry == nullptr) entry = &bb;
            return {entry, entry->begin()};
        }
    }
    BasicBlock* entry = func.getEntryBlock();
    if (entry == nullptr) {
        auto it = func.begin();
        if (it == func.end()) return {nullptr, IRList::iterator{}};
        entry = &(*it);
    }
    return {entry, entry->begin()};
}
}  // namespace

class SwInstructionPrefetchAbsDynamicPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SwInstructionPrefetchAbsDynamicPass";
    }

    PassID getPassID() const override {
        return &SwInstructionPrefetchAbsDynamicPass::ID;
    }

    void setBaseSgpr(int baseSgpr) {
        m_baseSgpr = baseSgpr;
    }

    int getBaseSgpr() const {
        return m_baseSgpr;
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

    /// Enable the CP-range-extend near-boundary cover (unconditional dynamic-width burst
    /// prepended to the ladder, covering [P(0), P(0)+N*4096)). Runtime-settable so tests can toggle
    /// it and assert the default-off behavior; production wiring flips it via the module factory.
    void setCpBoundaryCoverEnabled(bool enable) {
        m_cpBoundaryCover = enable;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        m_asmSetSymbols.clear();
        collectAsmSetSymbolValues(func, m_asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        // Compute layout (no IR mutation) to classify the kernel size regime.
        SwPrefetchRelPhase1Accum phase1;
        computeSwPrefetchRelPhase1Accum(func, &m_asmSetSymbols, phase1,
                                        m_debug ? m_debugStream : nullptr, getName());

        // Gate: CFG-target prefetch is about the POST-CP region, not the 64 KiB I-cache split.
        // Run whenever any code lives past P(0)=32640 (CP preload covers only [0, P(0))); the
        // semantic GW targets are chosen by branch logic regardless of total kernel size.
        // Only no-op when the whole kernel fits the CP window.
        if (phase1.totalLayoutBytes <= kSwPrefetchFirstGlobalByte) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] no-op: totalLayoutBytes ("
                               << phase1.totalLayoutBytes
                               << ") <= P(0)=" << kSwPrefetchFirstGlobalByte
                               << " (CP preload only; no post-CP region)\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }

        // Post-CP region exists. When debug is enabled, run the read-only analysis/debug dump
        // (covers the whole post-CP region). Basic 3-case model (GSU/beta) only; Stream-K /
        // MBSK-reduction / activation are flagged but not handled.
        if (m_debug) detectAndDumpD0(func, phase1);

        // Emission gate (regime split with abs-static): emit the prefetch ladder only in the
        // dynamic regime (total > 64 KiB) and only when a reserved SGPR base is available. Static
        // handles (32640, 65536]; this avoids both passes co-mutating the same kernel. The analysis
        // above still runs for all post-CP kernels.
        //
        // The emit site is [label_MultiGemmEnd, defineVariableSgprs) — the only +0-cost safe window
        // (multi-agent + fleet verified): KernelWriter._initKernel now DEFERS the abs-base triple's
        // checkIn to label_MultiGemmEnd (right before defineVariableSgprs reclaims those slots), so
        // the ladder can use the triple there without clobbering body values. Reserving any later
        // (ShadowInitStart/openLoopL) would overflow MaxSgpr by +3.
        // Set to false to ship detector-only (e.g. if the KernelWriter checkIn-defer is reverted).
        constexpr bool kD1LadderEmissionEnabled = true;
        bool mutated = false;
        if (kD1LadderEmissionEnabled &&
            phase1.totalLayoutBytes > kSwPrefetchAbsStaticIcacheSizeBytes && m_baseSgpr >= 0) {
            mutated = emitVariant1Ladder(func, passCtx, phase1);
        } else if (m_debug && phase1.totalLayoutBytes > kSwPrefetchAbsStaticIcacheSizeBytes) {
            *m_debugStream << "[" << getName() << "] D1 emit skipped: "
                           << (kD1LadderEmissionEnabled ? "baseSgpr unset (-1)"
                                                        : "emission gated off")
                           << " — detector-only\n";
        }

        closeDebugFile();
        return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

   private:
    void closeDebugFile() {
        if (m_debugFile.is_open()) m_debugFile.close();
    }

    /// CP preload window: bytes < P(0)=32640 are CP-resident; targets at/below are dropped.
    static constexpr int64_t kCpWindowBytes = kSwPrefetchFirstGlobalByte;
    /// A single fixed coverage count is used (fixed N=6 is safe).
    static constexpr int kFixedPrefetchN = 6;
    /// CP-range-extend cover: I-cache budget past the CP window = (65536-32640)/4096 = 8 hint
    /// blocks, shared between the cover and the ONE resident predicated arm: coverN + armN <= 8.
    static constexpr int kPostCpHintBudget = 8;
    /// Arm floor: reserve >=4 blocks for the resident ladder arm, so coverN is clamped to
    /// 8-4 = 4 (armFloor=4 is the accepted budget policy; keeps the deep hot arm >= 4*4096 B).
    static constexpr int kCpBoundaryArmFloor = 4;
    /// Conservative one-shot shift: an UPPER BOUND on the bytes inserted BEFORE the
    /// boundary (cover <=52 + ladder <=~244). Adding it to the pre-insertion boundary offset makes
    /// the computed coverN never under-cover (over-cover <= 1 block), with NO fixed-point
    /// iteration. +16 when the XNACK wait is on: up to 4 pre-boundary s_wait_xcnt 0 (1 cover burst
    /// + 3 ladder arms, all laid out before the CP boundary) x 4 B. Reverts to 320 when off, so
    /// coverN is bit-identical to before (the +16 stays within the same 4 KiB floor bucket except
    /// in the rare boundary-straddle case, where the extra block is genuinely needed).
    static constexpr int64_t kCpCoverInsertUpperBoundBytes = 320 + 4 * kSwPrefetchXnackWaitBytes;
    /// Target label anchored (post-insertion) at the final-layout CP boundary (offset <= P(0)).
    static constexpr const char* kCpBoundaryLabel = "label_SW_PrefetchAbs_CpBoundary";

    struct CaseTarget {
        const char* caseId;   // "A" / "B" / "C"
        std::string label;    // resolved anchor label name ("" if absent)
        int64_t offset = -1;  // global layout byte offset (-1 if absent)
        int64_t blockSize = 0;
    };

    /// Build label -> global layout byte offset from Phase-1 `layoutGlobal`. A label is 0 bytes,
    /// so its offset == the layout offset of the first real instruction at/after it (labels with
    /// no following real insn take the block-end offset). No IR mutation.
    static std::unordered_map<std::string, int64_t> buildLabelOffsets(
        Function& func, const SwPrefetchRelPhase1Accum& phase1) {
        std::unordered_map<std::string, int64_t> labelOffset;
        for (BasicBlock& bb : func) {
            std::vector<std::string> pending;  // labels awaiting the next real insn's offset
            for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
                IRBase* node = it.getNodePtr();
                if (node->getType() != IRBase::IRType::StinkyTofu) continue;
                StinkyInstruction& inst = getStinkyInst(it);
                const auto op = inst.getUnifiedOpcode();
                if (op == GFX::PHI) continue;
                if (op == GFX::LABEL) {
                    if (const LabelData* ld = inst.getModifier<LabelData>())
                        pending.push_back(ld->label);
                    continue;
                }
                if (pending.empty()) continue;
                const auto f = phase1.layoutGlobal.find(&inst);
                if (f == phase1.layoutGlobal.end())
                    continue;  // no layout for this insn; keep waiting
                for (const std::string& name : pending) labelOffset[name] = f->second;
                pending.clear();
            }
            if (!pending.empty()) {
                // Trailing labels with no following real insn in this BB: block-end offset.
                const auto ls = phase1.layoutStart.find(&bb);
                const auto lb = phase1.blockLocalBytes.find(&bb);
                if (ls != phase1.layoutStart.end() && lb != phase1.blockLocalBytes.end()) {
                    const int64_t end = ls->second + lb->second;
                    for (const std::string& name : pending) labelOffset[name] = end;
                }
            }
        }
        return labelOffset;
    }

    /// True if any non-LABEL instruction's branch/target modifier points at \p targetLabel
    /// (the "anchor on the taken branch/`s_setpc` target" rule; arch-independent).
    static bool hasBranchTarget(Function& func, const std::string& targetLabel) {
        for (BasicBlock& bb : func) {
            for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
                IRBase* node = it.getNodePtr();
                if (node->getType() != IRBase::IRType::StinkyTofu) continue;
                StinkyInstruction& inst = getStinkyInst(it);
                if (inst.getUnifiedOpcode() == GFX::LABEL) continue;
                if (const LabelData* ld = inst.getModifier<LabelData>())
                    if (ld->label == targetLabel) return true;
            }
        }
        return false;
    }

    /// Boundary for the CP-extend cover width — the end of the once-through fast path that
    /// lives PAST the CP window. Selection with fallbacks (all pre-insertion, phase-1 frame):
    ///   1. shallowest `label_TailLoopBegin*` (prefix match, MIN offset). Emitted name is
    ///      `label_TailLoopBeginL` (no underscore / nta-ntb suffix — verified fleet-wide); absent
    ///      only under NoTailLoop (ASEM % DepthU == 0).
    ///   2. `label_OptNLL_End` (present under isOptNLL; orthogonal to NoTailLoop).
    ///   3. shallowest `label_GW_*` with offset > P(0) (GW_B0_GSU1 is universal).
    /// Returns -1 if none — an intended `coverN = 0` no-op (nothing once-through lives past CP).
    static int64_t computeCoverBoundary(const std::unordered_map<std::string, int64_t>& labelOff) {
        int64_t tail = -1;
        for (const auto& kv : labelOff)
            if (kv.first.rfind("label_TailLoopBegin", 0) == 0 && (tail < 0 || kv.second < tail))
                tail = kv.second;
        if (tail >= 0) return tail;
        const auto f = labelOff.find("label_OptNLL_End");
        if (f != labelOff.end()) return f->second;
        int64_t gw = -1;
        for (const auto& kv : labelOff)
            if (kv.first.rfind("label_GW_", 0) == 0 && kv.second > kCpWindowBytes &&
                (gw < 0 || kv.second < gw))
                gw = kv.second;
        return gw;
    }

    /// CP-extend cover width. 0 when the boundary is absent or already CP-
    /// resident (<= P(0)); else the INCLUSIVE hint count `floor((boundary + INSERT_UB -
    /// P0)/4096)+1` (conservative one-shot: no iteration, never under-covers), clamped to `budget -
    /// armFloor`.
    static int computeCoverN(int64_t boundary) {
        // kCpWindowBytes >= 0, so `boundary <= kCpWindowBytes` already covers negatives.
        if (boundary <= kCpWindowBytes) return 0;
        const int64_t gap = boundary + kCpCoverInsertUpperBoundBytes - kCpWindowBytes;
        int raw = static_cast<int>(gap / kSwPrefetchSpacingBytes) + 1;  // floor+1 (gap > 0)
        const int maxCover = kPostCpHintBudget - kCpBoundaryArmFloor;   // = 4
        if (raw < 0) raw = 0;
        if (raw > maxCover) raw = maxCover;
        return raw;
    }

    /// Read-only CFG-target analysis + debug dump (no IR mutation). Basic 3-case (GSU/beta) only.
    void detectAndDumpD0(Function& func, const SwPrefetchRelPhase1Accum& phase1) {
        std::ostream& os = *m_debugStream;
        const std::unordered_map<std::string, int64_t> labelOff = buildLabelOffsets(func, phase1);

        auto offsetOf = [&](const std::string& name) -> int64_t {
            const auto f = labelOff.find(name);
            return f == labelOff.end() ? -1 : f->second;
        };
        auto has = [&](const std::string& n) { return labelOff.count(n) != 0; };

        // Step-1 anchors (name-based; KernelWriterAssembly "GW_B%u_%s" scheme).
        const std::string kB = "label_GW_B0_GSU1";  // Case B (universal)
        const std::string kC = "label_GW_B1_GSU1";  // Case C (only if UseBeta)
        const bool hasMBSK = has("label_GW_B0_MBSK");
        const bool hasMB = has("label_GW_B0_MB");
        const std::string kA = hasMBSK ? "label_GW_B0_MBSK" : (hasMB ? "label_GW_B0_MB" : "");

        const bool hasB = has(kB);
        const bool hasC = has(kC);
        const bool hasA = !kA.empty();

        os << "[" << getName()
           << "] D0 CFG-target detector (no IR mutation), P(0)=" << kCpWindowBytes
           << " totalLayoutBytes=" << phase1.totalLayoutBytes << " fixedN=" << kFixedPrefetchN
           << "\n";

        if (!hasB) {
            os << "  SKIP: label_GW_B0_GSU1 not found — kernel does not match the GSU1 beta-split "
                  "model (Stream-K / custom epilogue?); D0 analyzes nothing.\n";
            return;
        }

        // Case model: 3-case if a GSU>1 (MB/MBSK) arm exists, else 2-case (B/C).
        os << "  caseModel=" << (hasA ? "3-case (A=MB/MBSK present)" : "2-case (no MB/MBSK arm)")
           << " UseBeta=" << (hasC ? "true (B1_GSU1 present)" : "false (no B1_GSU1)") << "\n";

        // Assemble present targets in layout order to size each block to the next anchor.
        std::vector<CaseTarget> targets;
        if (hasA) targets.push_back({"A", kA, offsetOf(kA), 0});
        targets.push_back({"B", kB, offsetOf(kB), 0});
        if (hasC) targets.push_back({"C", kC, offsetOf(kC), 0});

        // blockSize is sized to the next *case anchor* (not the next emitted label), so it can be
        // marginally larger than the reference table. Coverage uses fixed N, not blockSize,
        // so this only affects the dumped value, never behavior.
        std::vector<int64_t> boundaries;
        boundaries.reserve(targets.size() + 1);
        for (const CaseTarget& t : targets) boundaries.push_back(t.offset);
        boundaries.push_back(phase1.totalLayoutBytes);
        std::sort(boundaries.begin(), boundaries.end());
        for (CaseTarget& t : targets) {
            int64_t next = phase1.totalLayoutBytes;
            for (int64_t b : boundaries)
                if (b > t.offset) {
                    next = b;
                    break;
                }
            t.blockSize = next - t.offset;
        }

        // Step-3 CP filter + per-case dump.
        const std::string selected = hasC ? kC : kB;  // default hot target: C if UseBeta else B
        for (const CaseTarget& t : targets) {
            const bool pastCp = t.offset > kCpWindowBytes;
            const int64_t covEnd = t.offset + int64_t(kFixedPrefetchN) * kSwPrefetchSpacingBytes;
            // DROP tag is A-only by design: B/C are always past CP on the gated fleet,
            // so only the GSU>1 (MB/MBSK) arm is the realistic drop candidate when inside the
            // window.
            const bool droppable = (std::string(t.caseId) == "A") && !pastCp;
            os << "  case=" << t.caseId << " target=" << t.label << " offset=" << t.offset
               << " blockSize=" << t.blockSize << " pastCP=" << (pastCp ? "yes" : "no")
               << " N=" << kFixedPrefetchN << " coverage=[" << t.offset << "," << covEnd << ")"
               << (droppable ? " [DROP: inside CP window]" : "")
               << (t.label == selected ? "  <== DEFAULT SELECTED" : "") << "\n";
        }

        // Step-2 beta selector (branch-target rule; arch-independent).
        if (hasC) {
            os << "  betaSelector: branch/setpc target -> " << kC << " "
               << (hasBranchTarget(func, kC) ? "FOUND" : "NOT FOUND (unexpected)") << "\n";
        }

        // Step-4 liveness gate (informational for emission; not computed in the analysis).
        os << "  D1-note: emit site must be AFTER the Beta kernarg load + s_waitcnt (not byte "
              "~308); "
              "GSU is live from its single prolog restore.\n";

        // Deferred families (NOT analyzed; flagged only).
        std::vector<std::string> deferred;
        if (hasMBSK) deferred.push_back("MBSK (reduction block precedes GW_B0_MBSK)");
        for (const auto& kv : labelOff) {
            const std::string& n = kv.first;
            if (n.rfind("label_Reduction_Start", 0) == 0)
                deferred.push_back("MBSK reduction block");
            else if (n.find("Fixup") != std::string::npos || n.rfind("label_SK", 0) == 0)
                deferred.push_back("Stream-K fixup/partials");
            // Match a genuine deferred-activation BODY (label_Activation_<func>_VW*), NOT the
            // ubiquitous SetPC address markers (label_ActivationSetPCAddrEnd*) that appear in
            // almost every kernel — the trailing underscore excludes the "ActivationSetPC..." form.
            else if (n.rfind("label_Activation_", 0) == 0)
                deferred.push_back("Activation deferred block");
        }
        std::sort(deferred.begin(), deferred.end());
        deferred.erase(std::unique(deferred.begin(), deferred.end()), deferred.end());
        if (!deferred.empty()) {
            os << "  DEFERRED (D2+, not handled by D0/D1 basic version):";
            for (const std::string& d : deferred) os << " [" << d << "]";
            os << "\n";
        }
    }

    /// Single-dword symbolic SGPR reference, emitted as `s[<name>]` (e.g. s[sgprGSU]).
    static StinkyRegister symbolicSgpr(const std::string& name) {
        StinkyRegister reg(RegType::S, /*regIdx=*/0u, /*regNum=*/1u);
        reg.setSymbolicName(name);
        return reg;
    }

    /// Predicated ladder emission: a GSU→beta branch ladder whose arms are verbatim abs-static
    /// bursts (getpc + `s_add_i32 label,4` + carry adds + N×`s_prefetch_inst`). Inserted once
    /// immediately AFTER `label_MultiGemmEnd` (the ArgType-merge join, before defineVariableSgprs),
    /// where sgprGSU/sgprBeta are live and the abs base triple is still reserved, and the whole
    /// main loop after it hides the fetch latency. Read-only on math; only adds scalar prefetch
    /// hints. Returns true iff IR was mutated. Basic 3-case (GSU/beta) only; falls back to an
    /// unconditional burst of the default target for non-3-arm shapes.
    bool emitVariant1Ladder(Function& func, PassContext& passCtx,
                            const SwPrefetchRelPhase1Accum& phase1) {
        const auto& archArr = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId =
            getGfxArchID(static_cast<uint32_t>(archArr[0]), static_cast<uint32_t>(archArr[1]),
                         static_cast<uint32_t>(archArr[2]));

        const std::unordered_map<std::string, int64_t> labelOff = buildLabelOffsets(func, phase1);
        auto has = [&](const std::string& n) { return labelOff.count(n) != 0; };
        const std::string kB = "label_GW_B0_GSU1";
        const std::string kC = "label_GW_B1_GSU1";
        const bool hasMBSK = has("label_GW_B0_MBSK");
        const bool hasMB = has("label_GW_B0_MB");
        const std::string kA = hasMBSK ? "label_GW_B0_MBSK" : (hasMB ? "label_GW_B0_MB" : "");
        const bool hasC = has(kC);
        const bool hasA = !kA.empty();

        // Dynamic CP-extend width: size the entry cover to reach the once-through fast-path
        // boundary (label_TailLoopBegin* / OptNLL_End / shallowest GW_*>P0). `armN` (below, once we
        // know the cover will actually emit) then splits the 8-block post-CP I-cache budget as
        // coverN + armN <= 8 (armFloor=4). coverN==0 => no cover emitted.
        const int64_t coverBoundary = computeCoverBoundary(labelOff);
        const int coverN = computeCoverN(coverBoundary);

        // Stream-K bails BOTH the ladder and the CP cover: its abs base triple is NOT reserved
        // to label_MultiGemmEnd (the Tensile checkIn is immediate for Stream-K), so emitting
        // anything at that site would clobber a live register. Deferred family.
        const bool isStreamK = m_asmSetSymbols.count("sgprSrdWS") != 0 ||
                               m_asmSetSymbols.count("sgprSynchronizer") != 0;
        if (isStreamK) {
            if (m_debug)
                *m_debugStream
                    << "[" << getName()
                    << "] D1 emit skip: Stream-K (abs base triple not reserved to MGE)\n";
            return false;
        }

        // The 3-arm LADDER needs the GSU1 beta-split anchor + defined GSU/beta selectors (GSU0 /
        // no-beta kernels omit the GSU restore, so `s[sgprGSU]` would be an UNDEFINED asm symbol).
        // The CP COVER needs NEITHER — it is unconditional and references only the base triple
        // + its own label — so it is DECOUPLED and still emits on GSU0/no-beta kernels. Bail only
        // when NEITHER can emit.
        const bool gsuDefined = m_asmSetSymbols.count("sgprGSU") != 0;
        const bool betaDefined = m_asmSetSymbols.count("sgprBeta") != 0;
        const bool wantLadder = has(kB) && gsuDefined && betaDefined;
        const bool wantCover = m_cpBoundaryCover;
        if (!wantLadder && !wantCover) {
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] D1 emit skip: ladder unsupported (hasB=" << has(kB)
                               << " gsuDefined=" << gsuDefined << " betaDefined=" << betaDefined
                               << ") and CP cover disabled\n";
            return false;
        }

        // Opcodes. The CP cover needs {getpc, add_i32, add_u32, addc_u32, prefetch}; the LADDER
        // also needs {and, mov, cmp, cbranch_scc0, branch}. Resolve all, then disable whichever
        // half lacks its opcodes (all resolve on gfx1250).
        const HwInstDesc* dAnd = getMCIDByUOp(GFX::s_and_b32, archId);
        const HwInstDesc* dMov = getMCIDByUOp(GFX::s_mov_b32, archId);
        const HwInstDesc* dCmp = getMCIDByUOp(GFX::s_cmp_eq_u32, archId);
        const HwInstDesc* dBr0 = getMCIDByUOp(GFX::s_cbranch_scc0, archId);
        const HwInstDesc* dBr = getMCIDByUOp(GFX::s_branch, archId);
        const HwInstDesc* dGetpc = getMCIDByUOp(GFX::s_getpc_b64, archId);
        const HwInstDesc* dAddI = getMCIDByUOp(GFX::s_add_i32, archId);
        const HwInstDesc* dAddU = getMCIDByUOp(GFX::s_add_u32, archId);
        const HwInstDesc* dAddC = getMCIDByUOp(GFX::s_addc_u32, archId);
        const HwInstDesc* dPf = getMCIDByUOp(GFX::s_prefetch_inst, archId);
        // XNACK safety wait; optional (toggle + arch), so NOT part of the cover/ladder opcode gate.
        const HwInstDesc* dXcnt = getMCIDByUOp(GFX::s_wait_xcnt, archId);
        const bool coverOpcodesOk = dGetpc && dAddI && dAddU && dAddC && dPf;
        const bool ladderOpcodesOk = coverOpcodesOk && dAnd && dMov && dCmp && dBr0 && dBr;
        const bool emitCover = wantCover && coverOpcodesOk;
        const bool emitLadder = wantLadder && ladderOpcodesOk;
        if (!emitCover && !emitLadder) {
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] D1 emit skip: opcode unavailable for arch\n";
            return false;
        }

        // Budget split. The arm width shrinks ONLY to fund a cover that will ACTUALLY emit
        // (emitCover && coverN>0). If the cover is disabled/absent/zero-width, the arms keep the
        // full kFixedPrefetchN=6 (no phantom budget reservation). armFloor=4 ⇒ armN in {4..6}.
        const int effectiveCoverN = (emitCover && coverN > 0) ? coverN : 0;
        const int armN = std::min(kFixedPrefetchN, kPostCpHintBudget - effectiveCoverN);

        // Site: immediately AFTER label_MultiGemmEnd (the ArgType-merge join). This is the unique
        // window that satisfies BOTH coupled constraints (multi-agent verified):
        //   - value liveness: it post-dominates the ArgType split, so sgprGSU/sgprBeta are live on
        //     every path (both arms branch TO this label — hence we insert AFTER it, not before).
        //   - SGPR safety: the abs base triple (s[base..base+2]) is re-allocated as ShadowLimitA/B
        //     by defineVariableSgprs right after this label, so [MultiGemmEnd, defineVariableSgprs)
        //     is the only spot where the triple is still free AND all paths have merged. Reserving
        //     it any later (ShadowInitStart/openLoopL) bumps the persistent block +3 → MaxSgpr
        //     overflow. The whole main loop still runs after this point ⇒ max issue latency.
        // Requires the Tensile-side checkIn-defer (KernelWriter._initKernel) to keep the triple
        // reserved across this window.
        BasicBlock* siteBB = nullptr;
        IRBase* siteAnchor =
            nullptr;  // first node AFTER the label (insert-before lands post-label)
        for (BasicBlock& bb : func) {
            for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
                IRBase* node = it.getNodePtr();
                if (node->getType() != IRBase::IRType::StinkyTofu) continue;
                StinkyInstruction& inst = getStinkyInst(it);
                if (inst.getUnifiedOpcode() != GFX::LABEL) continue;
                const LabelData* ld = inst.getModifier<LabelData>();
                if (ld == nullptr || ld->label != "label_MultiGemmEnd") continue;
                IRList::iterator nx = it;
                ++nx;  // anchor = instruction right after the label (so ladder runs on all arms)
                if (nx != bb.end()) {
                    siteBB = &bb;
                    siteAnchor = nx.getNodePtr();
                }
                break;
            }
            if (siteAnchor != nullptr) break;
        }
        if (siteAnchor == nullptr) {
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] D1 emit skip: no usable label_MultiGemmEnd site found\n";
            return false;
        }

        // GSU mask: gfx1250 fleet uses 0x3fff (KernArgsVersion<3); 0x0fff (>=3) is untested here.
        constexpr int kGsuMask = 0x3fff;

        AsmIRBuilder b(*siteBB, archId);
        const uint32_t lo = static_cast<uint32_t>(m_baseSgpr);
        const uint32_t hi = static_cast<uint32_t>(m_baseSgpr + 1);
        const uint32_t tmp = static_cast<uint32_t>(m_baseSgpr + 2);
        static const HwInstDesc labelMCID{
            GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

        auto emitLabel = [&](const std::string& name) {
            StinkyInstruction* l = b.create(&labelMCID, siteAnchor);
            l->addModifier<LabelData>(LabelData{name, /*alignment=*/1});
        };
        auto emitBranch = [&](const HwInstDesc* desc, const std::string& target) {
            StinkyInstruction* br = b.create(desc, siteAnchor);
            br->addSrcReg(StinkyRegister(target));
            br->addModifier<LabelData>(LabelData{target});
        };
        // One abs-static burst inserted at `at` in block `bb`: base = address(target), then `n`
        // prefetch hints at k*4096. SHARED by the 3-arm ladder (bb=*siteBB,
        // at=iterator(siteAnchor), n=armN) and the CP cover (bb=*entryBB,
        // at=entryBB->begin(), n=coverN). It creates DETACHED nodes (IRBase::createIR)
        // and inserts each before a FIXED `at` via insertIR: (1) for the arms this is IDENTICAL to
        // the old b.create(desc, siteAnchor) — same StinkyInstruction, inserted before siteAnchor,
        // in creation order; (2) for the cover it lets the SAME helper emit into the entry BB
        // INCLUDING the empty-CFG-stub case real kernels hit — insertIR(begin, x) APPENDS when
        // begin()==end(), so the cover is never silently skipped (an earlier create-before-anchor
        // form needed a non-null anchor and skipped empty stubs). Inserting successive nodes before
        // a fixed `at` preserves order (each lands just ahead of `at`).
        auto emitBurst = [&](BasicBlock& bb, IRList::iterator at, const std::string& target,
                             int n = kFixedPrefetchN) {
            auto ins = [&](const HwInstDesc* d) {
                StinkyInstruction* x = IRBase::createIR<StinkyInstruction>(d);
                bb.insertIR(at, x);
                return x;
            };
            StinkyInstruction* g = ins(dGetpc);
            g->addDestReg(StinkyRegister("s", lo, 2));
            StinkyInstruction* a0 = ins(dAddI);
            a0->addDestReg(StinkyRegister("s", tmp, 1));
            a0->addSrcReg(StinkyRegister(target));
            a0->addSrcReg(StinkyRegister(4));
            StinkyInstruction* a1 = ins(dAddU);
            a1->addDestReg(StinkyRegister("s", lo, 1));
            a1->addSrcReg(StinkyRegister("s", lo, 1));
            a1->addSrcReg(StinkyRegister("s", tmp, 1));
            StinkyInstruction* a2 = ins(dAddC);
            a2->addDestReg(StinkyRegister("s", hi, 1));
            a2->addSrcReg(StinkyRegister("s", hi, 1));
            a2->addSrcReg(StinkyRegister(0));
            // XNACK safety: one s_wait_xcnt 0 before this burst's contiguous prefetch group.
            // emitBurst is shared by the CP cover + all 3 ladder arms + the fallback, so this one
            // spot gates every group. No-op when the toggle is off or the opcode is unavailable;
            // the bytes of the pre-boundary waits are bounded by kCpCoverInsertUpperBoundBytes.
            if (kSwPrefetchEmitXnackWait && dXcnt != nullptr) {
                StinkyInstruction* w = ins(dXcnt);
                w->addSrcReg(StinkyRegister(0));  // xcnt = 0
            }
            for (int k = 0; k < n; ++k) {
                StinkyInstruction* p = ins(dPf);
                p->addSrcReg(StinkyRegister("s", lo, 2));
                p->addSrcReg(StinkyRegister(k * static_cast<int>(kSwPrefetchSpacingBytes)));
                p->addSrcReg(StinkyRegister("null"));
                p->addSrcReg(StinkyRegister(kSwPrefetchPcRelKlengthImm));
            }
        };

        const char* kSel = "label_Do_SW_PrefetchAbs_sel";
        const char* kCaseA = "label_Do_PF_caseA";
        const char* kCaseC = "label_Do_PF_caseC";
        const char* kEnd = "label_Do_PF_end";

        // CP-range-extend cover: ONE unconditional near-boundary burst (dynamic `coverN`,
        // computed above; 0 => skipped entirely) at the KERNEL ENTRY (entryBB->begin()), mirroring
        // the abs-static entry burst for MAXIMAL issue-latency lead. Covers [P(0),
        // P(0)+coverN*4096) — the once-through fast path the CP just misses (the ladder targets
        // only the deep GW blocks). SGPR-safe: emission is gated on label_MultiGemmEnd being
        // present (the MGE site was resolved above); per Tensile's PreLoop checkIn-defer, MGE
        // present <=> the abs base triple is reserved across the WHOLE prolog (incl. entry-begin) —
        // the same reservation abs-static relies on. The burst is self-contained (triple dead after
        // the prefetch). The target label is created post-insertion; the bare-label operand
        // is layout-safe (always +4 / FK_PCRel_4) even before the label exists.
        bool coverEmitted = false;
        if (emitCover && coverN > 0) {
            // Insert the CP cover AFTER the gfx1250 prologue (global_prefetch_b8 + v_nop), wherever
            // it lives, so the prologue stays the kernel's first executed instruction(s). Falls
            // back to the entry BB begin() (empty-CFG-stub APPEND preserved) when no prologue is
            // present.
            auto [coverBB, coverAt] = entryBurstInsertPoint(func);
            if (coverBB != nullptr) {
                emitBurst(*coverBB, coverAt, std::string(kCpBoundaryLabel), coverN);
                coverEmitted = true;
            } else if (m_debug) {
                *m_debugStream << "[" << getName() << "] CP cover skip: no entry BB\n";
            }
        } else if (emitCover && coverN == 0 && m_debug) {
            *m_debugStream << "[" << getName()
                           << "] CP cover skip: coverN=0 (once-through fast path is CP-resident;"
                              " boundary="
                           << coverBoundary << " <= P(0)=" << kCpWindowBytes << ")\n";
        }

        bool ladderEmitted = false;
        if (emitLadder && hasA && hasC) {
            // Full 3-arm ladder: GSU>1 -> A; GSU==1 & beta!=0 -> C; else -> B.
            emitLabel(kSel);
            StinkyInstruction* aGsu = b.create(dAnd, siteAnchor);
            aGsu->addDestReg(StinkyRegister("s", tmp, 1));
            aGsu->addSrcReg(symbolicSgpr("sgprGSU"));
            aGsu->addSrcReg(StinkyRegister(kGsuMask));
            StinkyInstruction* cGsu = b.create(dCmp, siteAnchor);
            cGsu->addSrcReg(StinkyRegister("s", tmp, 1));
            cGsu->addSrcReg(StinkyRegister(1));
            cGsu->addModifier<CommentData>(CommentData{"GSU == 1 ?"});
            emitBranch(dBr0, kCaseA);  // GSU != 1 (scc0) -> Case A (MB/MBSK)
            StinkyInstruction* mZero = b.create(dMov, siteAnchor);
            mZero->addDestReg(StinkyRegister("s", tmp, 1));
            mZero->addSrcReg(StinkyRegister(0));
            StinkyInstruction* cBeta = b.create(dCmp, siteAnchor);
            cBeta->addSrcReg(symbolicSgpr("sgprBeta"));
            cBeta->addSrcReg(StinkyRegister("s", tmp, 1));
            cBeta->addModifier<CommentData>(CommentData{"Beta == 0 ?"});
            emitBranch(dBr0, kCaseC);  // Beta != 0 (scc0) -> Case C (B1_GSU1)
            emitBurst(*siteBB, IRList::iterator(siteAnchor), kB,
                      armN);  // fall-through: Beta == 0 -> Case B (B0_GSU1)
            emitBranch(dBr, kEnd);
            emitLabel(kCaseA);
            emitBurst(*siteBB, IRList::iterator(siteAnchor), kA, armN);
            emitBranch(dBr, kEnd);
            emitLabel(kCaseC);
            emitBurst(*siteBB, IRList::iterator(siteAnchor), kC, armN);
            emitLabel(kEnd);
            ladderEmitted = true;
        } else if (emitLadder) {
            // Non-3-arm shape (rare): unconditionally prefetch the default hot target.
            emitLabel(kSel);
            emitBurst(*siteBB, IRList::iterator(siteAnchor), hasC ? kC : kB, armN);
            ladderEmitted = true;
        }

        // Anchor the CP-boundary target label on POST-insertion layout: re-accumulate (the
        // ladder+cover shifted following code forward ~N*8+... bytes, so CP now covers less
        // original code), then place the 0-byte label before the LAST real instruction whose
        // post-insertion offset is <= P(0). This lands the prefetch base at/just before the final
        // CP boundary with no gap. Mirrors SwInstructionPrefetchAbsStaticPass.cpp:321-372. (total >
        // 65536 guarantees an instruction at offset 0 <= P(0), so the anchor always resolves — no
        // dangling label.)
        if (coverEmitted) {
            SwPrefetchRelPhase1Accum phase2;
            computeSwPrefetchRelPhase1Accum(func, &m_asmSetSymbols, phase2,
                                            m_debug ? m_debugStream : nullptr, getName());
            BasicBlock* tBB = nullptr;
            IRList::iterator tIt;
            bool passedP0 = false;
            for (BasicBlock& bb2 : func) {
                for (IRList::iterator it = bb2.begin(); it != bb2.end(); ++it) {
                    IRBase* node = it.getNodePtr();
                    if (node->getType() != IRBase::IRType::StinkyTofu) continue;
                    StinkyInstruction& inst = getStinkyInst(it);
                    const auto op = inst.getUnifiedOpcode();
                    if (op == GFX::PHI || op == GFX::LABEL) continue;
                    const auto f = phase2.layoutGlobal.find(&inst);
                    if (f == phase2.layoutGlobal.end()) continue;
                    if (f->second > kCpWindowBytes) {  // layout monotonic in BB/list order
                        passedP0 = true;
                        break;
                    }
                    tBB = &bb2;
                    tIt = it;
                }
                if (passedP0) break;
            }
            if (tBB != nullptr) {
                AsmIRBuilder tb(*tBB, archId);
                StinkyInstruction* lbl = tb.createLabel(kCpBoundaryLabel, 1);
                tBB->insertIR(tIt, lbl);
            } else if (m_debug) {
                *m_debugStream << "[" << getName()
                               << "] CP cover: no anchor <= P(0) post-insert; label skipped\n";
            }
        }

        if (m_debug)
            *m_debugStream << "[" << getName() << "] D1 emitted after label_MultiGemmEnd"
                           << " baseSgpr=" << m_baseSgpr << " armN=" << armN
                           << (ladderEmitted ? (hasA && hasC ? " ladder(3-arm)" : " ladder(uncond)")
                                             : " ladder(none)")
                           << (coverEmitted ? " +CPcover(coverN=" + std::to_string(coverN) +
                                                  " boundary=" + std::to_string(coverBoundary) + ")"
                                            : "")
                           << "\n";
        return coverEmitted || ladderEmitted;
    }

    int m_baseSgpr = -1;
    bool m_cpBoundaryCover = false;  // CP-range-extend cover (default off; staged rollout)
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchAbsDynamicPass::ID = 0;

std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(int baseSgpr,
                                                                const std::string& debugOutputPath,
                                                                bool cpBoundaryCover) {
    auto p = std::make_unique<SwInstructionPrefetchAbsDynamicPass>();
    p->setBaseSgpr(baseSgpr);
    p->setCpBoundaryCoverEnabled(cpBoundaryCover);
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchAbsDynamicPass>();
    // Reserved even-aligned SGPR pair + scratch (base, base+1, base+2), auto-allocated in
    // Tensile KernelWriter._initKernel and passed via the module option (same source the abs
    // static pass reads). -1 ⇒ emission no-ops (analysis-only); the static pass owns the
    // burst in that case.
    p->setBaseSgpr(module.getModuleOptions().SwInstructionPrefetchAbsBaseSgpr);
    // CP-range-extend cover: ENABLED in production for the dynamic regime (unconditional
    // near-boundary burst of DYNAMIC width coverN, sized to the fast-path boundary and clamped to
    // coverN <= 4 via armFloor=4). Covers [P(0), P(0)+coverN*4096) — the once-through fast path the
    // CP just misses (OptNLL / tail-loop dispatch), which the 3-arm ladder does not target;
    // coverN=0 (fast path CP-resident) emits nothing. Pending the hardware gate (gfx1250
    // device-lib assemble-clean + numeric gtest). Set to false to ship the detector + 3-arm ladder
    // only.
    constexpr bool kCpBoundaryCoverEnabled = true;
    p->setCpBoundaryCoverEnabled(kCpBoundaryCoverEnabled);
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        constexpr const char* kDumpLeaf = "sw_prefetch_abs_dynamic_pass.txt";
        p->setDebugOutputPath((dir / kDumpLeaf).string());
        p->setDebug(true);
    }
    return p;
}

}  // namespace stinkytofu
