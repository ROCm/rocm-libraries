// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// SwInstructionPrefetchAbsStaticPass — Phase P1 of abs SW prefetch.
///
/// Policy: single-label + koffset burst.
///
/// One site label + getpc burst at the very beginning of the entry BB.
/// One target label (label_SW_PrefetchAbs_0) at the first real instruction
/// at or after P(0) = 32640.  N `s_prefetch_inst` with koffsets 0, 4096, ...
/// cover all needed 4 KiB grid steps from the same base.
///
/// Emitted burst (3 SGPRs: even-aligned base pair s[base:base+1] + scratch s[base+2]):
///   label_Do_SW_PrefetchAbs_entry:
///   s_getpc_b64 s[base:base+1]
///   s_add_i32   s[base+2], label_SW_PrefetchAbs_0, 4   ; PC-rel offset (+4 getpc correction)
///   s_add_u32   s[base],   s[base],   s[base+2]
///   s_addc_u32  s[base+1], s[base+1], 0
///   s_prefetch_inst s[base:base+1], k*4096, null, 0x1f   ; for k = 0 .. N-1
///
/// Notes:
///   - Address uses a bare label + temp SGPR (the rocisa long-branch idiom), proven on amdhsa.
///     `@pc` is an INVALID variant (assembler rejects it). The 2-SGPR `@rel32@lo/@hi` alternative
///     was tried and reverted — keep the proven 3-SGPR temp idiom.
///   - klength uses the simm5 immediate (0x1f); slength = null. No SGPR for the length.
///   - The base pair + temp (3 SGPRs) are auto-allocated in Tensile (KernelWriter._initKernel),
///     reserved through the prolog, and freed at label_MultiGemmEnd (before defineVariableSgprs
///     reclaims them) — body reuses them, so net ~0 SGPR pressure.
///   - Minimum-SGPR alternative (2 SGPRs, no temp): @rel32@lo+4 / @rel32@hi+12 on the two adds.
///
/// Purpose of single-label + koffset vs per-k labels:
///   - ONE getpc + ONE s_add chain builds the base address.
///   - N `s_prefetch_inst` with koffsets 0, 4096, 8192... emit N hints.
///   - vs per-k: N × (getpc + add + prefetch) = 3× more scalar instructions.
///   - Correct when layout is contiguous: koffset k*4096 lands on grid step P(k)
///     only if there is no .align padding between anchors that would shift the
///     physical addresses. For the static regime (≤64 KiB, whole-kernel burst)
///     this is acceptable; per-k targets remain the safer choice when alignment
///     gaps exist.

#include "stinkytofu/transforms/asm/SwInstructionPrefetchAbsStaticPass.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/SwPrefetchRelCommon.hpp"

namespace stinkytofu {

namespace {
// True when \p it points at a real instruction in \p bb with unified opcode \p uop.
// False at end(), or on an asm directive (non-StinkyTofu node).
bool isUnifiedOpcodeAt(BasicBlock& bb, IRList::iterator it, uint16_t uop) {
    if (it == bb.end()) return false;
    IRBase* node = it.getNodePtr();
    if (node->getType() != IRBase::IRType::StinkyTofu) return false;
    return static_cast<const StinkyInstruction*>(node)->getUnifiedOpcode() == uop;
}

// Return {BB, iterator} for the entry prefetch-burst insertion point: immediately AFTER the
// gfx1250 hardware-entrypoint prologue (`s_mov_b64 s[64:65], 0` + `v_nop` + `global_prefetch_b8`,
// inserted by InsertInitialUnclausedVmemPass) so the prologue stays the kernel's first executed
// instruction(s). The prologue is the function's first real (non label/phi/directive) instruction
// and may sit in a later BB than getEntryBlock() (e.g. after an empty preamble + label_ASM_Start),
// so we scan the whole function. Falls back to {entryBB, begin()} when no prologue is present
// (isolated unit tests / non-gfx1250), preserving prior behavior and the empty-entry-BB APPEND
// (begin()==end()).
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
            // First real instruction of the function. Walk the prologue shape
            //   s_mov_b64 s[64:65], 0 ; v_nop ; global_prefetch_b8 v0, s[64:65]
            // leading pieces are tolerated as absent (isolated unit tests), but the
            // global_prefetch_b8 must be there for this to be the prologue.
            {
                IRList::iterator after = it;
                if (isUnifiedOpcodeAt(bb, after, GFX::s_mov_b64)) ++after;
                if (isUnifiedOpcodeAt(bb, after, GFX::v_nop)) ++after;
                if (isUnifiedOpcodeAt(bb, after, GFX::global_prefetch_b8)) {
                    ++after;  // past global_prefetch_b8
                    return {&bb, after};
                }
            }
            // First real insn isn't the prologue -> no prologue; fall back to entry begin().
            BasicBlock* entry = func.getEntryBlock();
            if (entry == nullptr) entry = &bb;
            return {entry, entry->begin()};
        }
    }
    // No real instruction anywhere (empty stub): entry begin() so insertIR appends.
    BasicBlock* entry = func.getEntryBlock();
    if (entry == nullptr) {
        auto it = func.begin();
        if (it == func.end()) return {nullptr, IRList::iterator{}};
        entry = &(*it);
    }
    return {entry, entry->begin()};
}
}  // namespace

class SwInstructionPrefetchAbsStaticPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "SwInstructionPrefetchAbsStaticPass";
    }

    PassID getPassID() const override {
        return &SwInstructionPrefetchAbsStaticPass::ID;
    }

    void setBaseSgpr(int baseSgpr) {
        m_baseSgpr = baseSgpr;
    }

    int getBaseSgpr() const {
        return m_baseSgpr;
    }

    int getTotalPrefetchInserted() const {
        return m_totalPrefetchInserted;
    }

    void setDebug(bool enable) {
        m_debug = enable;
    }

    void setDebugOutputPath(const std::string& path) {
        m_debugOutputPath = path;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        m_asmSetSymbols.clear();
        m_totalPrefetchInserted = 0;
        m_totalBytes = 0;
        m_byteOffsetBase = 0;
        m_labelByteOffset.clear();
        collectAsmSetSymbolValues(func, m_asmSetSymbols);

        if (m_debug) {
            if (!m_debugOutputPath.empty()) {
                m_debugFile.open(m_debugOutputPath);
                m_debugStream = m_debugFile.is_open() ? &m_debugFile : &std::cerr;
            } else {
                m_debugStream = &std::cerr;
            }
        }

        // Phase 1: compute layout (pre-insert) to get totalLayoutBytes and
        // per-instruction global layout offsets.
        SwPrefetchRelPhase1Accum phase1;
        computeSwPrefetchRelPhase1Accum(func, &m_asmSetSymbols, phase1,
                                        m_debug ? m_debugStream : nullptr, getName());

        // Gate 1: totalLayoutBytes must be in (P(0), 64 KiB].
        // <= P(0): CP preload covers everything; no software prefetch needed.
        // > 64 KiB: dynamic regime — handled by SwInstructionPrefetchAbsDynamicPass (CFG-target).
        if (phase1.totalLayoutBytes <= kSwPrefetchFirstGlobalByte) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] no-op: totalLayoutBytes ("
                               << phase1.totalLayoutBytes
                               << ") <= P(0)=" << kSwPrefetchFirstGlobalByte
                               << " (CP preload only)\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }
        // Regime split: > 64 KiB is the dynamic regime, handled by the CFG-target
        // SwInstructionPrefetchAbsDynamicPass (now enabled — emits the predicated ladder after
        // label_MultiGemmEnd). The static entry-burst grid owns only (32640, 65536], so it no-ops
        // here to avoid co-mutating the same kernel / contending for the shared
        // SwInstructionPrefetchAbsBaseSgpr. (Re-disable in lockstep with kD1LadderEmissionEnabled
        // if ladder emission is ever turned back off.)
        if (phase1.totalLayoutBytes > kSwPrefetchAbsStaticIcacheSizeBytes) {
            if (m_debug)
                *m_debugStream
                    << "[" << getName() << "] no-op: totalLayoutBytes (" << phase1.totalLayoutBytes
                    << ") > I-cache limit " << kSwPrefetchAbsStaticIcacheSizeBytes
                    << " — dynamic regime, handled by SwInstructionPrefetchAbsDynamicPass\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }

        // Gate 2: base SGPR pair must be reserved.
        if (m_baseSgpr < 0) {
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] no-op: baseSgpr not configured (pass -1)\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }

        const auto& archArr = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId =
            getGfxArchID(static_cast<uint32_t>(archArr[0]), static_cast<uint32_t>(archArr[1]),
                         static_cast<uint32_t>(archArr[2]));

        // Count N (fixed point). The burst we insert at entry-begin grows the kernel by
        //   I = kBurstFixedBytes (s_getpc_b64=4 + s_add_i32(+label literal)=8 + s_add_u32=4
        //                         + s_addc_u32=4 = 20, plus the optional XNACK s_wait_xcnt 0 = 4)
        //     + N * kPrefetchInstBytes (each s_prefetch_inst)
        // bytes, shifting the whole body down by I. That can push one extra 4 KiB grid step past
        // the (now larger) tail, so N must be solved against the POST-insertion total. Because
        // I (<= ~hundreds of bytes) is far smaller than the 4 KiB grid step, two iterations
        // converge. When the XNACK wait is off, this reverts to 20 (byte-identical to before).
        constexpr int64_t kBurstFixedBytes = 20 + kSwPrefetchXnackWaitBytes;
        constexpr int64_t kPrefetchInstBytes = 8;
        const int64_t P0 = kSwPrefetchFirstGlobalByte;
        auto countN = [](int64_t total) {
            int n = 0;
            for (; swPrefetchGridOffset(n) < total; ++n)  // cover [P0, total)
                ;
            return n;
        };
        int N = countN(phase1.totalLayoutBytes);
        for (int iter = 0; iter < 2; ++iter) {
            const int64_t burstBytes = kBurstFixedBytes + int64_t(N) * kPrefetchInstBytes;
            N = countN(phase1.totalLayoutBytes + burstBytes);
        }
        // No ++N: the fixed-point already solves N against (preTotal + I) which is the
        // post-insertion total. The anchor is placed at the last instruction <= P0 (overlap
        // bias, ≤ one instruction below P0, typically < 12 B). Since I >= kBurstFixedBytes =
        // 20 >> 12 B, the fixed-point N absorbs that slack. An unconditional ++N would
        // over-provision on minimal-tail kernels (preTotal just above P0) where N=1 already
        // covers everything.

        // Cap N to the I-cache-resident window. The CP keeps [0, P0) resident; the I-cache holds
        // kSwPrefetchAbsStaticIcacheSizeBytes total, so the static burst must prefetch at most the
        // post-CP slice that still fits — (I-cache - P0) / spacing steps. Prefetching beyond that
        // would pull in code that cannot stay resident and would evict the CP-resident head, so
        // for > 64 KiB kernels we deliberately under-cover the far tail (that is the dynamic
        // pass's job) and keep [base, base + N*spacing) within the I-cache.
        constexpr int kMaxStaticPrefetchN =
            static_cast<int>((kSwPrefetchAbsStaticIcacheSizeBytes - kSwPrefetchFirstGlobalByte) /
                             kSwPrefetchSpacingBytes);
        if (N > kMaxStaticPrefetchN) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] capping N from " << N << " to I-cache max "
                               << kMaxStaticPrefetchN << " (post-CP window = "
                               << (kSwPrefetchAbsStaticIcacheSizeBytes - kSwPrefetchFirstGlobalByte)
                               << " B)\n";
            N = kMaxStaticPrefetchN;
        }

        if (m_debug) {
            *m_debugStream << "[" << getName() << "] Phase 2 abs-static insert: totalLayoutBytes="
                           << phase1.totalLayoutBytes << " baseSgpr=" << m_baseSgpr
                           << " N_prefetches=" << N << "\n";
        }

        const std::string targetLabelName = std::string(kSwPrefetchAbsTargetLabelBase) + "0";

        // Resolve the burst site: AFTER the gfx1250 hardware-entrypoint prologue
        // (s_mov_b64 + v_nop + global_prefetch_b8) wherever it lives in the function, so the
        // prologue stays the kernel's first executed instruction(s). May be a later BB than
        // getEntryBlock() (empty preamble + label_ASM_Start). Falls back to entry begin() when
        // no prologue is present.
        auto [burstBB, insertAt] = entryBurstInsertPoint(func);
        if (burstBB == nullptr) {
            if (m_debug)
                *m_debugStream << "[" << getName() << "] warning: no entry BB found; skip burst\n";
            closeDebugFile();
            return PreservedAnalyses::all();
        }

        // Insert the site burst FORWARD-referencing label_SW_PrefetchAbs_0 by name. The label
        // object is created afterwards, once its anchor is known from the post-insertion layout (a
        // bare-label operand always costs +4 regardless of whether the label is resolved yet, so
        // this forward reference is layout-safe). insertIR(insertAt, X) puts X before insertAt, so
        // the final order is: [prologue] siteLabel, getpc, add*, pf_0..pf_{N-1}, [original body].
        {
            AsmIRBuilder builder(*burstBB, archId);
            const uint32_t baseLo = static_cast<uint32_t>(m_baseSgpr);
            const uint32_t baseHi = static_cast<uint32_t>(m_baseSgpr + 1);
            // Scratch SGPR (base+2): holds the PC-relative offset for the address computation
            // (rocisa long-branch idiom). Dead after s_add_u32. klength uses the simm5 immediate
            // (0x1f), so no separate length register is needed.
            // To drop this scratch entirely (2 SGPRs total) switch the two adds to the
            // @rel32@lo+4 / @rel32@hi+12 relocation form (offset encoded in the instruction).
            const uint32_t tmp = static_cast<uint32_t>(m_baseSgpr + 2);

            // label_Do_SW_PrefetchAbs_entry
            StinkyInstruction* siteLabel =
                builder.createLabel(std::string(kSwPrefetchAbsSiteLabel), 1);
            burstBB->insertIR(insertAt, siteLabel);

            // s_getpc_b64 s[base:base+1]   ; PC of the next instruction
            StinkyInstruction* getpc = builder.create(getMCIDByUOp(GFX::s_getpc_b64, archId));
            getpc->addDestReg(StinkyRegister("s", baseLo, 2));
            burstBB->insertIR(insertAt, getpc);

            // s_add_i32 s[tmp], label_SW_PrefetchAbs_0, 4
            //   Bare label operand -> assembler emits a PC-relative relocation; +4 corrects for
            //   s_getpc_b64 returning the address of the following instruction. Same idiom as the
            //   rocisa long-branch sequence (proven on amdhsa). NOT @pc (invalid variant).
            StinkyInstruction* addOff = builder.create(getMCIDByUOp(GFX::s_add_i32, archId));
            addOff->addDestReg(StinkyRegister("s", tmp, 1));
            addOff->addSrcReg(StinkyRegister(targetLabelName));
            addOff->addSrcReg(StinkyRegister(4));
            burstBB->insertIR(insertAt, addOff);

            // s_add_u32 s[base], s[base], s[tmp]
            StinkyInstruction* addLo = builder.create(getMCIDByUOp(GFX::s_add_u32, archId));
            addLo->addDestReg(StinkyRegister("s", baseLo, 1));
            addLo->addSrcReg(StinkyRegister("s", baseLo, 1));
            addLo->addSrcReg(StinkyRegister("s", tmp, 1));
            burstBB->insertIR(insertAt, addLo);

            // s_addc_u32 s[base+1], s[base+1], 0
            StinkyInstruction* addHi = builder.create(getMCIDByUOp(GFX::s_addc_u32, archId));
            addHi->addDestReg(StinkyRegister("s", baseHi, 1));
            addHi->addSrcReg(StinkyRegister("s", baseHi, 1));
            addHi->addSrcReg(StinkyRegister(0));
            burstBB->insertIR(insertAt, addHi);

            // XNACK safety: one s_wait_xcnt 0 before the contiguous prefetch burst (Method 2 — a
            // single drain covers the whole back-to-back group). No-op when the toggle is off or
            // the opcode is unavailable. Its 4 bytes are modeled in kBurstFixedBytes above and
            // absorbed for real by the post-insertion re-accumulate (phase2 below).
            if (kSwPrefetchEmitXnackWait) {
                if (const HwInstDesc* xcntDesc = getMCIDByUOp(GFX::s_wait_xcnt, archId)) {
                    StinkyInstruction* xcnt = builder.create(xcntDesc);
                    xcnt->addSrcReg(StinkyRegister(0));  // xcnt = 0
                    burstBB->insertIR(insertAt, xcnt);
                }
            }

            // N × s_prefetch_inst s[base:base+1], k*kSpacing, null, 0x1f
            //   length = ((slength=null=0) + (klength imm=31)) & 31 + 1 = 32 lines = 4096 B.
            //   klength is the simm5 immediate (0x1f); no SGPR needed for the length.
            const HwInstDesc* pfDesc = getMCIDByUOp(GFX::s_prefetch_inst, archId);
            if (pfDesc == nullptr) {
                if (m_debug)
                    *m_debugStream << "[" << getName()
                                   << "] warning: s_prefetch_inst not found for arch; skip\n";
            } else {
                for (int k = 0; k < N; ++k) {
                    StinkyInstruction* pf = builder.create(pfDesc);
                    pf->addSrcReg(StinkyRegister("s", baseLo, 2));  // sbase pair
                    pf->addSrcReg(
                        StinkyRegister(k * static_cast<int>(kSwPrefetchSpacingBytes)));  // koffset
                    pf->addSrcReg(StinkyRegister("null"));  // slength = null (0)
                    pf->addSrcReg(
                        StinkyRegister(kSwPrefetchPcRelKlengthImm));  // klength = 31 (0x1f)
                    burstBB->insertIR(insertAt, pf);
                    ++m_totalPrefetchInserted;
                }
            }
        }

        // Re-accumulate to obtain REAL post-insertion addresses (absorbs the burst shift and any
        // .align padding uniformly). Phase 1 does not mutate the IR, so this is safe to run again.
        SwPrefetchRelPhase1Accum phase2;
        computeSwPrefetchRelPhase1Accum(func, &m_asmSetSymbols, phase2,
                                        m_debug ? m_debugStream : nullptr, getName());

        // Anchor label_SW_PrefetchAbs_0 at the LAST real instruction whose post-insertion offset
        // is <= P0. This puts the prefetch base at or just before the CP boundary, so SW coverage
        // [base, base + N*4096) overlaps the CP window by < one instruction (a harmless redundant
        // hint) and leaves NO gap at P0. Using post-insertion layout makes this robust to both the
        // burst shift and internal alignment padding.
        BasicBlock* targetBB = nullptr;
        IRList::iterator targetIt;
        int64_t anchorOff = -1;
        bool passedP0 = false;
        for (BasicBlock& bb : func) {
            for (IRList::iterator it = bb.begin(); it != bb.end(); ++it) {
                IRBase* node = it.getNodePtr();
                if (node->getType() != IRBase::IRType::StinkyTofu) continue;
                const StinkyInstruction& inst = *static_cast<const StinkyInstruction*>(node);
                if (inst.getUnifiedOpcode() == GFX::PHI || inst.getUnifiedOpcode() == GFX::LABEL)
                    continue;
                const auto found = phase2.layoutGlobal.find(const_cast<StinkyInstruction*>(&inst));
                if (found == phase2.layoutGlobal.end()) continue;
                if (found->second > P0) {  // layout is monotonic in BB/list order
                    passedP0 = true;
                    break;
                }
                targetBB = &bb;
                targetIt = it;
                anchorOff = found->second;
            }
            if (passedP0) break;
        }

        if (targetBB == nullptr) {
            // No instruction <= P0 post-insertion. Cannot happen given totalLayoutBytes > P0 and
            // the multi-KiB body, but guard: a dangling forward label ref would fail to assemble.
            if (m_debug)
                *m_debugStream << "[" << getName()
                               << "] warning: no anchor <= P0 found post-insert; label skipped\n";
            closeDebugFile();
            return PreservedAnalyses::none();
        }

        // Insert label_SW_PrefetchAbs_0 immediately before the anchor. A label is 0 bytes, so the
        // post-insertion layout computed above is unchanged; prefetch base = anchorOff <= P0.
        {
            AsmIRBuilder targetBuilder(*targetBB, archId);
            StinkyInstruction* targetLabel = targetBuilder.createLabel(targetLabelName, 1);
            targetBB->insertIR(targetIt, targetLabel);
        }

        if (m_debug) {
            *m_debugStream << "[" << getName() << "] Phase 2 abs-static complete: inserted " << N
                           << " s_prefetch_inst in BB \"" << burstBB->getLabel() << "\"\n";
            *m_debugStream << "[" << getName() << "]   site: " << kSwPrefetchAbsSiteLabel << "\n";
            *m_debugStream << "[" << getName() << "]   target: " << targetLabelName << " @ layout "
                           << anchorOff << " (P0=" << P0 << ", base<=P0 => no gap) in BB \""
                           << targetBB->getLabel() << "\"\n";
            *m_debugStream << "[" << getName() << "]   SW coverage = [" << anchorOff << ", "
                           << (anchorOff + int64_t(N) * kSwPrefetchSpacingBytes) << ")\n";
        }

        // Post-insert accumulateInstructionSize (mirrors static pass).
        // The burst's `s_add_i32 s[tmp], label_SW_PrefetchAbs_0, 4` forward-references
        // a label defined later (in targetBB). That is costed correctly on a single
        // forward walk: a label operand is always a FK_PCRel_4 relocation and is
        // unconditionally counted as +4(literal) by getLiteralExtraBytes, independent
        // of whether the label address is known yet (see InstructionSizeCosting.cpp).
        for (BasicBlock& bb : func) {
            int blockCount = 0;
            int64_t blockBytes = 0;
            accumulateInstructionSize(bb, m_labelByteOffset, m_debug ? m_debugStream : nullptr,
                                      &blockCount, &blockBytes, m_byteOffsetBase, &m_asmSetSymbols);
            m_totalBytes += blockBytes;
            m_byteOffsetBase += blockBytes;
        }

        if (m_debug) {
            *m_debugStream << "[" << getName() << "] total size (post-insert) = " << m_totalBytes
                           << " bytes\n";
        }

        closeDebugFile();
        return PreservedAnalyses::none();
    }

   private:
    void closeDebugFile() {
        if (m_debugFile.is_open()) m_debugFile.close();
    }

    int m_baseSgpr = -1;
    int m_totalPrefetchInserted = 0;
    int64_t m_totalBytes = 0;
    int64_t m_byteOffsetBase = 0;
    std::unordered_map<std::string, int64_t> m_labelByteOffset;
    std::unordered_map<std::string, int64_t> m_asmSetSymbols;
    bool m_debug = false;
    std::string m_debugOutputPath;
    std::ofstream m_debugFile;
    std::ostream* m_debugStream = &std::cerr;
};

char SwInstructionPrefetchAbsStaticPass::ID = 0;

std::unique_ptr<Pass> createSwInstructionPrefetchAbsStaticPass(int baseSgpr,
                                                               const std::string& debugOutputPath) {
    auto p = std::make_unique<SwInstructionPrefetchAbsStaticPass>();
    p->setBaseSgpr(baseSgpr);
    p->setDebugOutputPath(debugOutputPath);
    if (!debugOutputPath.empty()) p->setDebug(true);
    return p;
}

std::unique_ptr<Pass> createSwInstructionPrefetchAbsStaticPass(StinkyAsmModule& module) {
    auto p = std::make_unique<SwInstructionPrefetchAbsStaticPass>();
    // Base SGPR pair is auto-allocated in Tensile (KernelWriter._initKernel) and passed
    // via the module option SwInstructionPrefetchAbsBaseSgpr (-1 = off → pass no-ops).
    p->setBaseSgpr(module.getModuleOptions().SwInstructionPrefetchAbsBaseSgpr);
    if (!module.getOutputDir().empty()) {
        const std::string costBasename =
            module.getOutputName().empty() ? module.getName() : module.getOutputName();
        std::filesystem::path dir = std::filesystem::path(module.getOutputDir()) / costBasename;
        std::filesystem::create_directories(dir);
        constexpr const char* kDumpLeaf = "sw_prefetch_abs_static_pass.txt";
        p->setDebugOutputPath((dir / kDumpLeaf).string());
        p->setDebug(true);
    }
    return p;
}

}  // namespace stinkytofu
