/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

/// SwInstructionPrefetchAbsDynamicPass — abs SW prefetch, CFG-target (post-CP) policy.
///
/// Policy: choose the prefetch target by the global-write BRANCH logic (GSU / beta
/// dispatch) instead of a byte grid. Gate is the POST-CP region (totalLayoutBytes >
/// P(0)=32640), NOT the 64 KiB I-cache split. See SwInstructionPrefetchAbsDynamic-
/// CfgTarget-Design.md §10–§12.
///
/// Status — D1 (enabled): DETECTOR + Variant-1 EMISSION. run() computes the layout, always runs
/// detectAndDumpD0() (read-only debug dump) which classifies the 3-case GSU/beta dispatch
/// (A=GW_B0_{MB,MBSK} / B=GW_B0_GSU1 / C=GW_B1_GSU1), applies the CP filter, and flags deferred
/// families. Then, when totalLayoutBytes > 65536 and a reserved baseSgpr is available, it EMITS
/// the Variant-1 predicated ladder (getpc + `s_add_i32 label,4` + N×`s_prefetch_inst`) immediately
/// after `label_MultiGemmEnd`. It bails (detector-only, no IR mutation) for Stream-K, GSU0
/// (undefined sgprGSU), and no-beta kernels; MBSK-reduction / f64 / activation remain D2+.

#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsDynamicPass.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
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

        // Post-CP region exists. Always run the §10.3 CFG-target DETECTOR debug dump (read-only;
        // covers the whole post-CP region). Basic 3-case model (GSU/beta) only; Stream-K /
        // MBSK-reduction / f64 / activation are flagged but not handled (D2+; see §11.2/§12).
        if (m_debug) detectAndDumpD0(func, phase1);

        // D1 emission gate (regime split with abs-static): emit the Variant-1 ladder only in the
        // dynamic regime (total > 64 KiB) and only when a reserved SGPR base is available. Static
        // handles (32640, 65536]; this avoids both passes co-mutating the same kernel. Detection
        // above still runs for all post-CP kernels.
        //
        // The emit site is [label_MultiGemmEnd, defineVariableSgprs) — the only +0-cost safe window
        // (multi-agent + fleet verified): KernelWriter._initKernel now DEFERS the abs-base triple's
        // checkIn to label_MultiGemmEnd (right before defineVariableSgprs reclaims those slots), so
        // the ladder can use the triple there without clobbering body values. Reserving any later
        // (ShadowInitStart/openLoopL) would overflow MaxSgpr by +3. See design §13/§14.
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
    /// D0 uses a single fixed coverage count (per design §10.5 "fixed N=6 is also safe").
    static constexpr int kFixedPrefetchN = 6;

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
    /// (the §10.3 "anchor on the taken branch/`s_setpc` target" rule; arch-independent).
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

    /// D0 CFG-target detector + debug dump (no IR mutation). Basic 3-case (GSU/beta) only.
    void detectAndDumpD0(Function& func, const SwPrefetchRelPhase1Accum& phase1) {
        std::ostream& os = *m_debugStream;
        const std::unordered_map<std::string, int64_t> labelOff = buildLabelOffsets(func, phase1);

        auto offsetOf = [&](const std::string& name) -> int64_t {
            const auto f = labelOff.find(name);
            return f == labelOff.end() ? -1 : f->second;
        };
        auto has = [&](const std::string& n) { return labelOff.count(n) != 0; };

        // §10.3 Step-1 anchors (name-based; KernelWriterAssembly "GW_B%u_%s" scheme).
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
        // marginally larger than the §10.2 reference table. Coverage uses fixed N, not blockSize,
        // so this only affects the dumped value, never behavior.
        std::vector<int64_t> boundaries;
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

        // §10.3 Step-3 CP filter + per-case dump.
        const std::string selected = hasC ? kC : kB;  // default hot target: C if UseBeta else B
        for (const CaseTarget& t : targets) {
            const bool pastCp = t.offset > kCpWindowBytes;
            const int64_t covEnd = t.offset + int64_t(kFixedPrefetchN) * kSwPrefetchSpacingBytes;
            // DROP tag is A-only by design: §12.4 shows B/C are always past CP on the gated fleet,
            // so only the GSU>1 (MB/MBSK) arm is the realistic drop candidate when inside the
            // window.
            const bool droppable = (std::string(t.caseId) == "A") && !pastCp;
            os << "  case=" << t.caseId << " target=" << t.label << " offset=" << t.offset
               << " blockSize=" << t.blockSize << " pastCP=" << (pastCp ? "yes" : "no")
               << " N=" << kFixedPrefetchN << " coverage=[" << t.offset << "," << covEnd << ")"
               << (droppable ? " [DROP: inside CP window]" : "")
               << (t.label == selected ? "  <== DEFAULT SELECTED" : "") << "\n";
        }

        // §10.3 Step-2 beta selector (branch-target rule; arch-independent).
        if (hasC) {
            os << "  betaSelector: branch/setpc target -> " << kC << " "
               << (hasBranchTarget(func, kC) ? "FOUND" : "NOT FOUND (unexpected)") << "\n";
        }

        // §10.4 Step-4 liveness gate (informational for D1; not computed in D0).
        os << "  D1-note: emit site must be AFTER the Beta kernarg load + s_waitcnt (not byte "
              "~308); "
              "GSU is live from its single prolog restore.\n";

        // Deferred families (NOT analyzed in D0; flagged only — see design §11.2/§12).
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

    /// D1 Variant-1 emission: a GSU→beta branch ladder whose arms are verbatim abs-static bursts
    /// (getpc + `s_add_i32 label,4` + carry adds + N×`s_prefetch_inst`). Inserted once immediately
    /// AFTER `label_MultiGemmEnd` (the ArgType-merge join, before defineVariableSgprs), where
    /// sgprGSU/sgprBeta are live and the abs base triple is still reserved, and the whole main loop
    /// after it hides the fetch latency. Read-only on math; only adds scalar prefetch hints.
    /// Returns true iff IR was mutated. Basic 3-case (GSU/beta) only; falls back to an
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
        if (!has(kB)) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] D1 emit skip: no " << kB
                               << " (not the GSU1 beta-split model)\n";
            return false;
        }

        // Supported-dispatch guard (basic D1). Bail when:
        //   - sgprGSU is not defined (GSU0 kernels omit the GSU restore, so `s[sgprGSU]` in the
        //     ladder would be an UNDEFINED asm symbol → "expected absolute expression"), or
        //   - sgprBeta is not defined (no beta split), or
        //   - this is a Stream-K kernel (different synchronizer/skTiles dispatch, and its abs base
        //     triple is NOT reserved to MGE — the Tensile checkIn is immediate for Stream-K — so
        //     emitting here would clobber a live register). Stream-K is a §11.2 deferred family.
        const bool gsuDefined = m_asmSetSymbols.count("sgprGSU") != 0;
        const bool betaDefined = m_asmSetSymbols.count("sgprBeta") != 0;
        const bool isStreamK = m_asmSetSymbols.count("sgprSrdWS") != 0 ||
                               m_asmSetSymbols.count("sgprSynchronizer") != 0;
        if (!gsuDefined || !betaDefined || isStreamK) {
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] D1 emit skip: unsupported dispatch (gsuDefined=" << gsuDefined
                               << " betaDefined=" << betaDefined << " streamK=" << isStreamK
                               << ")\n";
            return false;
        }
        const bool hasC = has(kC);
        const bool hasA = !kA.empty();

        // Opcodes (all must resolve for this arch, else skip safely).
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
        if (!dAnd || !dMov || !dCmp || !dBr0 || !dBr || !dGetpc || !dAddI || !dAddU || !dAddC ||
            !dPf) {
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] D1 emit skip: opcode unavailable for arch\n";
            return false;
        }

        // Site: immediately AFTER label_MultiGemmEnd (the ArgType-merge join). This is the unique
        // window that satisfies BOTH coupled constraints (multi-agent verified, design §14.2/§13):
        //   - value liveness: it post-dominates the ArgType split, so sgprGSU/sgprBeta are live on
        //     every path (both arms branch TO this label — hence we insert AFTER it, not before).
        //   - SGPR safety: the abs base triple (s[base..base+2]) is re-allocated as ShadowLimitA/B
        //     by defineVariableSgprs right after this label, so [MultiGemmEnd, defineVariableSgprs)
        //     is the only spot where the triple is still free AND all paths have merged. Reserving
        //     it any later (ShadowInitStart/openLoopL) bumps the persistent block +3 → MaxSgpr
        //     overflow. The whole main loop still runs after this point ⇒ max issue latency.
        // Requires the Tensile-side checkIn-defer (KernelWriter._initKernel) to keep the triple
        // reserved across this window; see design §13.
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
        const int N = kFixedPrefetchN;  // D1 uses a single fixed N (per §10.5).

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
        // One abs-static burst: base = address(target), then N prefetch hints at k*4096.
        auto emitBurst = [&](const std::string& target) {
            StinkyInstruction* g = b.create(dGetpc, siteAnchor);
            g->addDestReg(StinkyRegister("s", lo, 2));
            StinkyInstruction* a0 = b.create(dAddI, siteAnchor);
            a0->addDestReg(StinkyRegister("s", tmp, 1));
            a0->addSrcReg(StinkyRegister(target));
            a0->addSrcReg(StinkyRegister(4));
            StinkyInstruction* a1 = b.create(dAddU, siteAnchor);
            a1->addDestReg(StinkyRegister("s", lo, 1));
            a1->addSrcReg(StinkyRegister("s", lo, 1));
            a1->addSrcReg(StinkyRegister("s", tmp, 1));
            StinkyInstruction* a2 = b.create(dAddC, siteAnchor);
            a2->addDestReg(StinkyRegister("s", hi, 1));
            a2->addSrcReg(StinkyRegister("s", hi, 1));
            a2->addSrcReg(StinkyRegister(0));
            for (int k = 0; k < N; ++k) {
                StinkyInstruction* p = b.create(dPf, siteAnchor);
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

        if (hasA && hasC) {
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
            emitBurst(kB);             // fall-through: Beta == 0 -> Case B (B0_GSU1)
            emitBranch(dBr, kEnd);
            emitLabel(kCaseA);
            emitBurst(kA);
            emitBranch(dBr, kEnd);
            emitLabel(kCaseC);
            emitBurst(kC);
            emitLabel(kEnd);
        } else {
            // Non-3-arm shape (rare): unconditionally prefetch the default hot target.
            emitLabel(kSel);
            emitBurst(hasC ? kC : kB);
        }

        if (m_debug)
            *m_debugStream << "[" << getName()
                           << "] D1 emitted Variant-1 ladder after label_MultiGemmEnd"
                           << " baseSgpr=" << m_baseSgpr << " N=" << N
                           << (hasA && hasC ? " (3-arm)" : " (unconditional)") << "\n";
        return true;
    }

    int m_baseSgpr = -1;
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchAbsDynamicPass::ID = 0;

std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(
    int baseSgpr, const std::string& debugOutputPath) {
    auto p = std::make_unique<SwInstructionPrefetchAbsDynamicPass>();
    p->setBaseSgpr(baseSgpr);
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchAbsDynamicPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchAbsDynamicPass>();
    // Reserved even-aligned SGPR pair + scratch (base, base+1, base+2), auto-allocated in
    // Tensile KernelWriter._initKernel and passed via the module option (same source the abs
    // static pass reads). -1 ⇒ D1 emission no-ops (detector-only); the static pass owns the
    // burst in that case.
    p->setBaseSgpr(module.getModuleOptions().SwInstructionPrefetchAbsBaseSgpr);
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
