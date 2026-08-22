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

#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

// Per-arch logical name -> ASM mnemonic (same data as Rocisa LogicalToArchMap; gives correct
// ds_read vs ds_load etc.)
#include "stinkytofu/ir/LogicalToAsmMappings_generated.inc"

// For ArchHelper access
using namespace stinkytofu;
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace stinkytofu;

/**
 * @brief Generate mnemonic for MFMA instructions
 *
 * Follows rocisa mnemonic format from rocisa/include/instruction/mfma.hpp:preStr()
 * CDNA (gfx942, gfx950): v_mfma_{accType}_{m}x{n}x{k}[_{blocks}b]_{instType}[_1k]
 * RDNA (gfx1250): v_wmma_{accType}_{m}x{n}x{k}_{instType}[_1k]
 */
std::string generateMFMAMnemonic(const std::string& accType, int m, int n, int k, int blocks,
                                 const std::string& instType, bool mfma1k, GfxArchID arch,
                                 bool scaled = false) {
    std::string variantStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

    // RDNA architectures (gfx12+) use v_wmma instead of v_mfma
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(arch);
    bool isRDNA = (archInfo && archInfo->major >= 12);

    if (isRDNA) {
        // RDNA: v_wmma[_scale]_{accType}_{m}x{n}x{k}_{instType}[_1k]
        // gfx1250 forceScaledWMMA emits the scale-instruction encoding for
        // low-precision (f8f6f4/f4) WMMA (rocisa mfma.hpp:forceScaledWMMA()).
        std::string wmmaPrefix = scaled ? "v_wmma_scale_" : "v_wmma_";
        std::string mfma1kSuffix = mfma1k ? "_1k" : "";
        return wmmaPrefix + accType + "_" + variantStr + "_" + instType + mfma1kSuffix;
    } else {
        // CDNA: v_mfma_{accType}_{m}x{n}x{k}[_{blocks}b]_{instType}[_1k]
        std::string blocksSuffix = (blocks > 1) ? std::to_string(blocks) + "b_" : "";
        std::string mfma1kSuffix = mfma1k ? "_1k" : "";
        return "v_mfma_" + accType + "_" + variantStr + "_" + blocksSuffix + instType +
               mfma1kSuffix;
    }
}

/**
 * @brief Generate mnemonic for SMFMA (Sparse MFMA) instructions
 *
 * CDNA (gfx942, gfx950): v_smfmac_{accType}_{m}x{n}x{k}_{instType}
 * RDNA (gfx1250): v_swmmac_{accType}_{m}x{n}x{k}_{instType}
 * gfx1250 has no MFMA/SMFMA; the sparse matrix op is SWMMAC (v_swmmac_*),
 * mirroring how generateMFMAMnemonic maps MFMA->WMMA for RDNA.
 * Note: blocks parameter is NOT part of the mnemonic, it's an instruction modifier/operand
 */
std::string generateSMFMAMnemonic(const std::string& accType, int m, int n, int k, int blocks,
                                  const std::string& instType, GfxArchID arch) {
    std::string variantStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

    // RDNA architectures (gfx12+) use v_swmmac instead of v_smfmac (blocks is
    // not part of the mnemonic for either family).
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(arch);
    bool isRDNA = (archInfo && archInfo->major >= 12);

    const char* prefix = isRDNA ? "v_swmmac_" : "v_smfmac_";
    return prefix + accType + "_" + variantStr + "_" + instType;
}

/**
 * @brief Generate mnemonic for MXMFMA (Mixed-precision scaled WMMA) instructions
 *
 * Follows rocisa mnemonic format from rocisa/include/instruction/mfma.hpp:preStr()
 * Format: v_wmma_scale[16]_f32_{m}x{n}x{k}_{instType}
 * Note: Uses "v_wmma_scale" prefix, not "v_mxmfma"
 */
std::string generateMXMFMAMnemonic(int m, int n, int k, int block, const std::string& instType) {
    std::string variantStr = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    std::string blockStr = (block == 16) ? "16" : "";

    return "v_wmma_scale" + blockStr + "_f32_" + variantStr + "_" + instType;
}

// Helper to create assembly instruction from IR
StinkyInstruction* createAsmFromIR(LogicalInstruction* irInst, GfxArchID arch) {
    const char* logicalName = irInst->getLogicalName();

    // Get the architecture-specific mnemonic
    uint16_t isaOpcode = 0;
    std::string mnemonic;

    // ====================================================================
    // Special Instructions: MFMA, SMFMA, MXMFMA
    // ====================================================================
    // Generate mnemonics dynamically from instruction metadata
    if (irInst->getOpcode() == logical::MFMA) {
        const MFMAData* data = irInst->asMFMA();
        if (!data) {
            STINKY_UNREACHABLE("MFMA instruction has no MFMAData");
            return nullptr;
        }
        mnemonic = generateMFMAMnemonic(data->accType, data->m, data->n, data->k, data->blocks,
                                        data->instType, data->mfma1k, arch, data->scaled);
    } else if (irInst->getOpcode() == logical::SMFMA) {
        const SMFMAData* data = irInst->asSMFMA();
        if (!data) {
            STINKY_UNREACHABLE("SMFMA instruction has no SMFMAData");
            return nullptr;
        }
        mnemonic = generateSMFMAMnemonic(data->accType, data->m, data->n, data->k, data->blocks,
                                         data->instType, arch);
    } else if (irInst->getOpcode() == logical::MXMFMA) {
        const MXMFMAData* data = irInst->asMXMFMA();
        if (!data) {
            STINKY_UNREACHABLE("MXMFMA instruction has no MXMFMAData");
            return nullptr;
        }
        mnemonic = generateMXMFMAMnemonic(data->m, data->n, data->k, data->block, data->instType);
    }
    // ====================================================================
    // Special Instructions: SWaitAlu, SchedulingFence
    // ====================================================================
    else if (irInst->getOpcode() == logical::SWaitAlu) {
        const SWaitAluLogicalData* data = irInst->asSWaitAlu();
        if (!data) {
            STINKY_UNREACHABLE("SWaitAlu instruction has no SWaitAluLogicalData");
            return nullptr;
        }
        isaOpcode = getMnemonicToIsaOpcode("s_wait_alu", arch);
        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, arch);
        if (!desc) {
            STINKY_UNREACHABLE("SWaitAlu: s_wait_alu not supported on this architecture");
            return nullptr;
        }
        StinkyInstruction* asmInst = IRBase::createIR<StinkyInstruction>(desc);
        SWaitAluData waitAluData(data->va_vdst, data->va_sdst, data->va_ssrc, data->hold_cnt,
                                 data->vm_vsrc, data->va_vcc, data->sa_sdst);
        asmInst->addModifier<SWaitAluData>(waitAluData);
        if (!irInst->comment.empty()) {
            asmInst->addModifier(CommentData(irInst->comment));
        }
        return asmInst;
    } else if (irInst->getOpcode() == logical::SchedulingFence) {
        static const HwInstDesc fenceMCID{
            GFX::FENCE, GFX::FENCE, 0, 0, 0, "FENCE", makeFlagSet({InstFlag::IF_HasSideEffect})};
        StinkyInstruction* asmInst = IRBase::createIR<StinkyInstruction>(&fenceMCID);
        if (!irInst->comment.empty()) {
            asmInst->addModifier(CommentData(irInst->comment));
        }
        return asmInst;
    }
    // ====================================================================
    // Regular instructions: per-arch map only (LogicalToAsmMappings_generated.inc)
    // Every lowering for each arch must be in the map; no fallback.
    // ====================================================================
    else {
        const char* archMnemonic = getMnemonicForLogicalOnArch(logicalName, arch);
        if (!archMnemonic) {
            STINKY_UNREACHABLE(
                ("ToStinkyAsmPass: No mapping for logical instruction '" +
                 std::string(logicalName) +
                 "' on this architecture; add to per-arch LogicalToArchMap (Gfx*.cpp).")
                    .c_str());
            return nullptr;
        }
        mnemonic = archMnemonic;
    }

    // Get the ISA opcode for this mnemonic on the target architecture
    isaOpcode = getMnemonicToIsaOpcode(mnemonic, arch);
    if (isaOpcode == GFX::INVALID) {
        // getMCIDByIsaOp indexes the MCID table directly, so an INVALID opcode
        // would read out of bounds and yield a non-null bogus pointer that the
        // !desc check below cannot catch (this used to segfault). Fail cleanly.
        STINKY_UNREACHABLE(
            ("ToStinkyAsmPass: No ISA opcode for mnemonic '" + mnemonic +
             "' on this architecture (missing hardware .def entry or wrong mnemonic).")
                .c_str());
        return nullptr;
    }
    const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, arch);

    if (!desc) {
        STINKY_UNREACHABLE(
            ("ToStinkyAsmPass: Instruction not supported on architecture: " + mnemonic).c_str());
        return nullptr;
    }

    // Create the assembly instruction
    StinkyInstruction* asmInst = IRBase::createIR<StinkyInstruction>(desc);

    // Copy operands from IR to assembly.
    // Handle the store-instruction mismatch: LogicalInstruction may place vdata
    // in dests (it's the "output" of the Python instruction), but the HW format
    // (e.g. MUBUF_STORE) defines ALL operand fields as sources (S0..S3).  When
    // the HW descriptor has zero dest fields but irInst has dests, prepend them
    // to srcRegs so that collectVgprMsbSlots lines up registers with HW fields.
    bool hwHasDestField = false;
    if (desc) {
        for (const auto& f : desc->operandFields) {
            if (f.isDest || f.isReadWrite) {
                hwHasDestField = true;
                break;
            }
        }
    }

    if (!irInst->dests.empty() && !hwHasDestField) {
        // HW has no dest field — logical dests are really src operands.
        std::vector<StinkyRegister> merged;
        merged.reserve(irInst->dests.size() + irInst->srcs.size());
        merged.insert(merged.end(), irInst->dests.begin(), irInst->dests.end());
        merged.insert(merged.end(), irInst->srcs.begin(), irInst->srcs.end());
        asmInst->setSrcRegs(merged);
    } else {
        std::vector<StinkyRegister> destRegs = irInst->dests;
        std::vector<StinkyRegister> srcRegs = irInst->srcs;

        // Read-write operands encoded as extra dest-position fields.
        // v_swap_b32 / v_permlane16_swap_b32 model BOTH exchanged vgprs as RW
        // dest fields (D0, D1), yet the logical form carries the second vgpr as
        // a *source*. The generic mapping (logical dests->destRegs,
        // logical srcs->srcRegs) then leaves destRegs one register short, and
        // the emitter — which prints one operand per dest field and never emits
        // RW fields as sources (emitSrcCount counts only non-dest fields) —
        // silently drops it, producing e.g. "v_swap_b32 v0" (missing operand).
        // Mirror rocisa's VSwapB32::getDstParams/getSrcParams: every RW operand
        // must appear in BOTH destRegs (for emission) and srcRegs (so use-def
        // tracking still sees the read).
        size_t numDestFields = 0;
        bool hasRWDest = false;
        for (const auto& f : desc->operandFields) {
            if (f.isDest) {
                numDestFields++;
                if (f.isReadWrite) hasRWDest = true;
            }
        }
        if (hasRWDest && numDestFields > destRegs.size()) {
            size_t need = numDestFields - destRegs.size();
            for (size_t k = 0; k < need && k < srcRegs.size(); ++k) {
                destRegs.push_back(srcRegs[k]);
            }
            // RW dest operands are also reads; keep them in srcRegs for
            // dependency/use-def tracking (not re-emitted: emitSrcCount == 0).
            for (const auto& d : irInst->dests) {
                srcRegs.push_back(d);
            }
        }

        if (!destRegs.empty()) {
            asmInst->setDestRegs(destRegs);
        }
        if (!srcRegs.empty()) {
            asmInst->setSrcRegs(srcRegs);
        }
    }

    // gfx1250 forceScaledWMMA: the scale-instruction encoding (v_wmma_scale_*)
    // carries two extra scale source operands. rocisa emits them as literal 0
    // (no actual scaling), matching MXWMMA_SCALE fields S3/S4. Append them so
    // the operand list becomes acc, a, b, acc2, 0, 0.
    if (irInst->getOpcode() == logical::MFMA) {
        const MFMAData* mfmaData = irInst->asMFMA();
        if (mfmaData && mfmaData->scaleOperands) {
            asmInst->addSrcReg(StinkyRegister(0));
            asmInst->addSrcReg(StinkyRegister(0));
        }
    }

    // Copy comment
    if (!irInst->comment.empty()) {
        asmInst->addModifier(CommentData(irInst->comment));
    }

    // Copy instruction modifiers from logical IR to assembly IR
    if (irInst->ds.has_value()) {
        asmInst->addModifier<DSModifiers>(irInst->ds.value());
    }
    if (irInst->mubuf.has_value()) {
        asmInst->addModifier<MUBUFModifiers>(irInst->mubuf.value());
    }
    if (irInst->dpp.has_value()) {
        asmInst->addModifier<DPPModifiers>(irInst->dpp.value());
    }
    if (irInst->sdwa.has_value()) {
        asmInst->addModifier<SDWAModifiers>(irInst->sdwa.value());
    }
    if (irInst->vop3.has_value()) {
        asmInst->addModifier<VOP3PModifiers>(irInst->vop3.value());
    }

    // MFMA/SMFMA/MXMFMA: attach MFMAModifiers so downstream passes
    // (RegionClonePass, SetMatrixReusePass) can identify these instructions.
    if (irInst->getOpcode() == logical::MFMA || irInst->getOpcode() == logical::SMFMA ||
        irInst->getOpcode() == logical::MXMFMA) {
        MFMAModifiers mod;
        if (irInst->getOpcode() == logical::MFMA) {
            const MFMAData* data = irInst->asMFMA();
            if (data && data->neg) {
                mod.negBits.negLo = {1, 1, 0};
                mod.negBits.numSrcs = 2;
            }
            // gfx1250 f8f6f4-family WMMA carries per-matrix input formats
            // (matrix_a_fmt:MATRIX_FMT_FP6 ...). Emit them via MatrixFmtModifiers.
            if (data && (!data->matrixAFmt.empty() || !data->matrixBFmt.empty())) {
                MatrixFmtModifiers fmts;
                if (!data->matrixAFmt.empty()) fmts.fmtA = parseMatrixFmt(data->matrixAFmt);
                if (!data->matrixBFmt.empty()) fmts.fmtB = parseMatrixFmt(data->matrixBFmt);
                asmInst->addModifier<MatrixFmtModifiers>(fmts);
            }
        } else if (irInst->getOpcode() == logical::SMFMA) {
            const SMFMAData* data = irInst->asSMFMA();
            if (data && data->neg) {
                mod.negBits.negLo = {1, 1, 0};
                mod.negBits.numSrcs = 2;
            }
        } else if (irInst->getOpcode() == logical::MXMFMA) {
            const MXMFMAData* data = irInst->asMXMFMA();
            if (data) {
                mod.reuseA = data->reuseA;
                mod.reuseB = data->reuseB;
                // gfx1250 f8f6f4-family scaled WMMA carries per-matrix input
                // formats (matrix_a_fmt:MATRIX_FMT_FP4 ...) and per-matrix scale
                // numeric formats (matrix_a_scale_fmt:N). Without the input format
                // the assembler assumes FP8 and rejects the FP4 register tuple
                // size; without the scale format the hardware misinterprets the
                // MX scale operands and produces numerically wrong results.
                // rocisa MXMFMAInstruction maps the scale datatype: f8 -> E4M3(2),
                // e5m3 -> E5M3(1), e8/other -> no modifier.
                auto scaleFmtFromStr = [](const std::string& s) {
                    if (s == "fp8" || s == "f8") return MatrixScaleFmt::E4M3;
                    if (s == "e5m3") return MatrixScaleFmt::E5M3;
                    return MatrixScaleFmt::NONE;
                };
                MatrixScaleFmt scaleFmtA = scaleFmtFromStr(data->mxScaleATypeStr);
                MatrixScaleFmt scaleFmtB = scaleFmtFromStr(data->mxScaleBTypeStr);
                if (!data->matrixAFmt.empty() || !data->matrixBFmt.empty() ||
                    scaleFmtA != MatrixScaleFmt::NONE || scaleFmtB != MatrixScaleFmt::NONE) {
                    MatrixFmtModifiers fmts;
                    if (!data->matrixAFmt.empty()) fmts.fmtA = parseMatrixFmt(data->matrixAFmt);
                    if (!data->matrixBFmt.empty()) fmts.fmtB = parseMatrixFmt(data->matrixBFmt);
                    fmts.scaleFmtA = scaleFmtA;
                    fmts.scaleFmtB = scaleFmtB;
                    asmInst->addModifier<MatrixFmtModifiers>(fmts);
                }
            }
        }
        asmInst->addModifier<MFMAModifiers>(mod);
    }

    return asmInst;
}

/// Implementation of ToStinkyAsmPass using unified Pass infrastructure
class ToStinkyAsmPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "ToStinkyAsmPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        // Process all basic blocks
        for (BasicBlock& bb : func) {
            // Skip filtered basic blocks
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            lowerToAsm(bb, arch);
        }
        return PreservedAnalyses::none();
    }

   private:
    void lowerToAsm(BasicBlock& bb, GfxArchID arch) {
        // Builder used to legalize instructions that have no direct hardware
        // encoding on the target arch (e.g. ds_*_b192 on gfx1250).
        AsmIRBuilder irBuilder(bb, arch);

        // Use iterators to allow insertion/removal during traversal
        auto it = bb.begin();
        while (it != bb.end()) {
            IRBase* irNode = &(*it);

            if (irNode->getType() == IRBase::IRType::LogicalIR) {
                LogicalInstruction* logicalInst = cast<LogicalInstruction>(irNode);

                // Lower to assembly
                StinkyInstruction* asmInst = createAsmFromIR(logicalInst, arch);

                if (asmInst) {
                    // Insert assembly instruction before the logical instruction
                    bb.insertIR(it, asmInst);

                    // Remove the logical instruction from IRList
                    auto toRemove = it;
                    ++it;  // Move to next before removing
                    bb.removeIR(&(*toRemove));

                    logicalInst->safeErase();

                    // gfx1250 (and other RDNA) have no ds_*_b192 encoding. Match
                    // rocisa's DSStoreB192/DSLoadB192::toString(), which always splits
                    // into a b128 + b64 pair. The rocisa->stinky conversion path handles
                    // this in ToStinkyTofuUtils::legalizeInstruction; the logical->asm
                    // path (adaptor / PyLogicalModule) must do the same here. VGPR MSB
                    // is materialized later by InsertVgprMsbPass, so pass hasVgprMsb=false.
                    if (asmInst->getUnifiedOpcode() == GFX::ds_store_b192) {
                        legalizeDSStoreB192(asmInst, irBuilder, arch, /*hasVgprMsb=*/false);
                    } else if (asmInst->getUnifiedOpcode() == GFX::ds_load_b192) {
                        legalizeDSLoadB192(asmInst, irBuilder, arch, /*hasVgprMsb=*/false);
                    } else if (asmInst->getUnifiedOpcode() == GFX::s_barrier) {
                        // gfx1250 has no plain s_barrier; it must split into
                        // s_barrier_signal -1 / s_barrier_wait -1. The rocisa->stinky
                        // conversion path does this in ToStinkyTofuUtils::legalizeInstruction
                        // (GFX::s_barrier -> legalizeBarrier); the logical->asm path
                        // (adaptor / PyLogicalModule) must do the same here. Doing it now
                        // (before the asm pipeline) is also required so the workgroup
                        // s_barrier_wait -1 exists as a distinct instruction for
                        // InsertClusterBarrierPass to anchor its Rule 3/4 handshakes on.
                        legalizeBarrier(asmInst, irBuilder, arch);
                    }
                    continue;
                }
            }

            ++it;
        }
    }
};

char ToStinkyAsmPassImpl::ID = 0;

}  // anonymous namespace

namespace stinkytofu {
std::unique_ptr<Pass> createToStinkyAsmPass() {
    return std::make_unique<ToStinkyAsmPassImpl>();
}

}  // namespace stinkytofu
