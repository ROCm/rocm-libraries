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
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "AllHwMappings.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

namespace nb = nanobind;

namespace {
using namespace rocisa;
using namespace stinkytofu;

StinkyRegister toStinkyRegister(const Container* container, bool hasVgprMsb, GfxArchID arch);
StinkyRegister toStinkyRegister(const InstructionInput& input, bool hasVgprMsb, GfxArchID arch);

std::string itemToString(const rocisa::Item* item) {
    return item->toString();
}

// Forward decls (definitions appear below; convertFLATModifiers references them).
stinkytofu::MUBUFScope convertMUBUFScope(rocisa::CacheScope scope);
stinkytofu::TemporalHint convertTemporalHint(rocisa::TemporalHint th);

// Helper functions to convert rocisa modifiers to stinkytofu modifiers
stinkytofu::DSModifiers convertDSModifiers(const rocisa::DSModifiers& rocMod) {
    return stinkytofu::DSModifiers(rocMod.na, rocMod.offset, rocMod.offset0, rocMod.offset1,
                                   rocMod.gds);
}

stinkytofu::FLATModifiers convertFLATModifiers(const rocisa::FLATModifiers& rocMod,
                                               const std::map<std::string, int>& asmCaps) {
    bool hasGLCModifier = asmCaps.count("HasGLCModifier") && asmCaps.at("HasGLCModifier");
    bool hasSC0Modifier = asmCaps.count("HasSC0Modifier") && asmCaps.at("HasSC0Modifier");
    return stinkytofu::FLATModifiers(
        rocMod.offset12, rocMod.glc, rocMod.slc, rocMod.lds, rocMod.isStore, hasGLCModifier,
        hasSC0Modifier, convertMUBUFScope(rocMod.scope), convertTemporalHint(rocMod.th));
}

stinkytofu::MUBUFScope convertMUBUFScope(rocisa::CacheScope scope) {
    switch (scope) {
        case rocisa::CacheScope::SCOPE_CU:
            return stinkytofu::MUBUFScope::SCOPE_CU;
        case rocisa::CacheScope::SCOPE_SE:
            return stinkytofu::MUBUFScope::SCOPE_SE;
        case rocisa::CacheScope::SCOPE_DEV:
            return stinkytofu::MUBUFScope::SCOPE_DEV;
        case rocisa::CacheScope::SCOPE_SYS:
            return stinkytofu::MUBUFScope::SCOPE_SYS;
        default:
            return stinkytofu::MUBUFScope::SCOPE_NONE;
    }
}

// rocisa and StinkyTofu use the same ISA TH[2:0] encoding, but keep distinct enum
// types; map explicitly rather than static_cast so the two stay decoupled. Note
// TH_WB aliases TH_LU and TH_NT_WB aliases TH_RESERVED (same encoding), so only the
// canonical labels appear here.
stinkytofu::TemporalHint convertTemporalHint(rocisa::TemporalHint th) {
    switch (th) {
        case rocisa::TemporalHint::TH_RT:
            return stinkytofu::TemporalHint::TH_RT;
        case rocisa::TemporalHint::TH_NT:
            return stinkytofu::TemporalHint::TH_NT;
        case rocisa::TemporalHint::TH_HT:
            return stinkytofu::TemporalHint::TH_HT;
        case rocisa::TemporalHint::TH_LU:
            return stinkytofu::TemporalHint::TH_LU;
        case rocisa::TemporalHint::TH_NT_RT:
            return stinkytofu::TemporalHint::TH_NT_RT;
        case rocisa::TemporalHint::TH_RT_NT:
            return stinkytofu::TemporalHint::TH_RT_NT;
        case rocisa::TemporalHint::TH_NT_HT:
            return stinkytofu::TemporalHint::TH_NT_HT;
        case rocisa::TemporalHint::TH_RESERVED:
            return stinkytofu::TemporalHint::TH_RESERVED;
        default:
            return stinkytofu::TemporalHint::TH_NONE;
    }
}

stinkytofu::MUBUFModifiers convertMUBUFModifiers(const rocisa::MUBUFModifiers& rocMod,
                                                 const std::map<std::string, int>& asmCaps) {
    bool hasMUBUFConst = asmCaps.count("HasMUBUFConst") && asmCaps.at("HasMUBUFConst");
    bool hasGLCModifier = asmCaps.count("HasGLCModifier") && asmCaps.at("HasGLCModifier");
    bool hasSC0Modifier = asmCaps.count("HasSC0Modifier") && asmCaps.at("HasSC0Modifier");
    stinkytofu::MUBUFScope scope = convertMUBUFScope(rocMod.scope);
    stinkytofu::TemporalHint th = convertTemporalHint(rocMod.th);
    return stinkytofu::MUBUFModifiers(rocMod.offen, rocMod.offset12, rocMod.glc, rocMod.slc,
                                      rocMod.nt, rocMod.lds, rocMod.isStore, hasMUBUFConst,
                                      hasGLCModifier, hasSC0Modifier, scope, th);
}

/// Returns true when vaddr is the MUBUF "off" keyword.
bool isOffVaddrContainer(const rocisa::Container* vaddr) {
    if (auto* regCont = dynamic_cast<const rocisa::RegisterContainer*>(vaddr)) {
        return regCont->isOff;
    }
    return false;
}

/// Build modifiers matching rocisa's MUBUF address form: real vaddr operands
/// use `offen`, while `off` keeps the no-vaddr form.
stinkytofu::MUBUFModifiers buildMUBUFModifiersForBufferOp(
    const std::optional<rocisa::MUBUFModifiers>& rocMubuf, const rocisa::Container* vaddr,
    const std::map<std::string, int>& asmCaps) {
    bool hasMUBUFConst = asmCaps.count("HasMUBUFConst") && asmCaps.at("HasMUBUFConst");
    bool hasGLCModifier = asmCaps.count("HasGLCModifier") && asmCaps.at("HasGLCModifier");
    bool hasSC0Modifier = asmCaps.count("HasSC0Modifier") && asmCaps.at("HasSC0Modifier");

    stinkytofu::MUBUFModifiers mod = rocMubuf.has_value()
                                         ? convertMUBUFModifiers(rocMubuf.value(), asmCaps)
                                         : stinkytofu::MUBUFModifiers(
                                               /*offen=*/false, /*offset12=*/0, /*glc=*/false,
                                               /*slc=*/false, /*nt=*/false, /*lds=*/false,
                                               /*isStore=*/false, hasMUBUFConst, hasGLCModifier,
                                               hasSC0Modifier, stinkytofu::MUBUFScope::SCOPE_NONE);

    if (!mod.offen && !isOffVaddrContainer(vaddr)) {
        mod.offen = 1;
    }
    return mod;
}

stinkytofu::SMEMModifiers convertSMEMModifiers(const rocisa::SMEMModifiers& rocMod,
                                               const std::map<std::string, int>& asmCaps) {
    bool hasSCOPEModifier = asmCaps.count("HasSCOPEModifier") && asmCaps.at("HasSCOPEModifier");
    return stinkytofu::SMEMModifiers(rocMod.glc, rocMod.nv != rocisa::NonVolatile::NV_NONE,
                                     rocMod.offset, hasSCOPEModifier);
}

stinkytofu::SDelayAluData convertSDelayAluData(const rocisa::SDelayAlu* delayAluInst) {
    // Convert DelayALUType to SDelayAluData::InstType
    auto convertType = [](rocisa::DelayALUType type) -> SDelayAluData::InstType {
        switch (type) {
            case rocisa::DelayALUType::VALU:
                return SDelayAluData::InstType::VALU;
            case rocisa::DelayALUType::SALU:
                return SDelayAluData::InstType::SALU;
            case rocisa::DelayALUType::TRANS:
                return SDelayAluData::InstType::TRANS;
            default:
                return SDelayAluData::InstType::NO_DEP;
        }
    };

    auto params = delayAluInst->getParams();

    assert(params.size() >= 2 && "s_delay_alu should have at least 2 parameters");

    int instid0TypeInt = std::get<int>(params[0]);
    int instid0Cnt = std::get<int>(params[1]);

    SDelayAluData::InstType instid0Type =
        convertType(static_cast<rocisa::DelayALUType>(instid0TypeInt));

    if (!delayAluInst->hasInstID1()) {
        // Single dependency: {instid0type, instid0cnt}
        assert(params.size() == 2 && "s_delay_alu single dependency should have 2 parameters");
        return SDelayAluData(instid0Type, static_cast<int8_t>(instid0Cnt));
    }

    // Dual dependency: {instid0type, instid0cnt, instskipCnt, instid1type, instid1cnt}
    assert(params.size() == 5 && "s_delay_alu has 5 parameters");

    int instskipCnt = std::get<int>(params[2]);
    int instid1TypeInt = std::get<int>(params[3]);
    int instid1Cnt = std::get<int>(params[4]);

    SDelayAluData::InstType instid1Type =
        convertType(static_cast<rocisa::DelayALUType>(instid1TypeInt));

    return SDelayAluData(instid0Type, static_cast<int8_t>(instid0Cnt),
                         static_cast<int8_t>(instskipCnt), instid1Type,
                         static_cast<int8_t>(instid1Cnt));
}

stinkytofu::VOP3PModifiers convertVOP3PModifiers(const rocisa::VOP3PModifiers& rocMod) {
    return stinkytofu::VOP3PModifiers(rocMod.op_sel, rocMod.op_sel_hi, rocMod.byte_sel);
}

stinkytofu::DPPModifiers convertDPPModifiers(const rocisa::DPPModifiers& rocMod) {
    // rocisa's WaveSplitK reduction is the only producer; it sends
    //   row_shr   in {1, 2, 4, 8}      -> DppCtrl::ROW_SHR0 + n
    //   row_bcast in {15, 31}          -> DppCtrl::BCAST15 / BCAST31
    //   bound_ctrl in {0, 1}
    // Other fields stay at the rocisa "unset" sentinel (-1).
    constexpr int kUnset = -1;
    constexpr int kBcastLane15 = 15;  // row_bcast:15 -> DppCtrl::BCAST15
    constexpr int kBcastLane31 = 31;  // row_bcast:31 -> DppCtrl::BCAST31

    auto ctrl = stinkytofu::DppCtrl::NONE;
    if (rocMod.row_shr != kUnset) {
        auto candidate = stinkytofu::dppRowShr(rocMod.row_shr);
        if (candidate <= stinkytofu::DppCtrl::ROW_SHR_LAST) ctrl = candidate;
    } else if (rocMod.row_bcast == kBcastLane15) {
        ctrl = stinkytofu::DppCtrl::BCAST15;
    } else if (rocMod.row_bcast == kBcastLane31) {
        ctrl = stinkytofu::DppCtrl::BCAST31;
    } else if (rocMod.quad_perm.size() == 4) {
        ctrl = stinkytofu::dppQuadPerm(rocMod.quad_perm[0], rocMod.quad_perm[1],
                                       rocMod.quad_perm[2], rocMod.quad_perm[3]);
    }

    // bound_ctrl is {0, 1}. Anything else (unset -1, or stray) -> default 0.
    uint8_t boundCtrl = (rocMod.bound_ctrl == 1) ? 1 : 0;
    // rocisa has no row_mask/bank_mask field; pass default 0xF (all rows/banks active).
    return stinkytofu::DPPModifiers(ctrl, /*rowMask=*/0xF, /*bankMask=*/0xF, boundCtrl);
}

stinkytofu::SDWAModifiers convertSDWAModifiers(const rocisa::SDWAModifiers& rocMod) {
    return stinkytofu::SDWAModifiers(
        static_cast<stinkytofu::SDWAModifiers::SelectBit>(rocMod.dst_sel),
        static_cast<stinkytofu::SDWAModifiers::UnusedBit>(rocMod.dst_unused),
        static_cast<stinkytofu::SDWAModifiers::SelectBit>(rocMod.src0_sel),
        static_cast<stinkytofu::SDWAModifiers::SelectBit>(rocMod.src1_sel));
}

Legalized legalizeInstruction(StinkyInstruction* inst, rocisa::Instruction* rocisaInst,
                              AsmIRBuilder& irBuilder, GfxArchID archId,
                              const std::map<std::string, int>& asmCaps,
                              const std::map<std::string, int>& archCaps, bool hasVgprMsb) {
    // Attach implicit special registers (SCC/VCC/`EXEC) declared by HW flags
    // (Flags.def) to the instruction.
    legalizeImplicitSpecialRegisters(inst, getWaveFrontSize(archId));

    if (isBranch(*inst)) {
        // Handle branch instructions
        rocisa::BranchInstruction* branchInst =
            dynamic_cast<rocisa::BranchInstruction*>(rocisaInst);
        assert(branchInst != nullptr && "This should be a rocisa Branch.");

        // SSetPCB64 records its long-branch target in a dedicated longBranchLabel
        // field (populated by the SLongBranch* helpers in rocisa extension.hpp)
        if (auto* setpc = dynamic_cast<rocisa::SSetPCB64*>(rocisaInst)) {
            if (!setpc->longBranchLabel.empty()) {
                inst->addModifier<LabelData>(LabelData{setpc->longBranchLabel});
            }
            return {nullptr, nullptr};
        }

        inst->addModifier<LabelData>(LabelData{branchInst->labelName});
        return {nullptr, nullptr};
    }

    if (inst->is(InstFlag::IF_VCmpX)) {
        return legalizeVCmpX(inst, irBuilder, archId, archCaps);
    }

    switch (inst->getUnifiedOpcode()) {
        case GFX::v_nop:
            return legalizeVNop(inst, irBuilder, archId);

        case GFX::ds_load_b192:
            return legalizeDSLoadB192(inst, irBuilder, archId, hasVgprMsb);

        case GFX::ds_store_b192:
            return legalizeDSStoreB192(inst, irBuilder, archId, hasVgprMsb);

        case GFX::s_waitcnt:
            return legalizeWaitCnt(inst, irBuilder, archId);

        case GFX::s_barrier:
            return legalizeBarrier(inst, irBuilder, archId);

        default:
            break;
    }
    return {nullptr, nullptr};
}

/// Create StinkyInstruction from rocisa instruction
StinkyInstruction* createStinkyInstructionFromRocisa(rocisa::Instruction& inst,
                                                     std::type_index rocisaTy,
                                                     AsmIRBuilder& irBuilder, GfxArchID archId) {
    const HwInstDesc* hwInstDesc = getRocisaToMCID(rocisaTy, archId);
    StinkyInstruction* stinkyInst = nullptr;

    if (hwInstDesc != nullptr) {
        // Direct mapping exists - create instruction directly
        stinkyInst = irBuilder.create(hwInstDesc);
    } else {
        // Need conversion function
        ConvertRocisaToHwInstFunc convFn = getConvertRocisaToHwInstFunc(rocisaTy, archId);
        if (convFn == nullptr) {
            return nullptr;
        }

        // Call conversion function
        std::vector<StinkyInstruction*> stinkyInsts = convFn(inst, irBuilder);

        if (stinkyInsts.empty()) {
            return nullptr;
        }

        // For now, only handle single instruction conversions
        stinkyInst = stinkyInsts[0];
    }

    return stinkyInst;
}

/// Add source and destination registers to StinkyInstruction
void addRegistersToInstruction(StinkyInstruction* stinkyInst, const rocisa::Instruction* inst,
                               const std::map<std::string, int>& asmCaps, bool hasVgprMsb,
                               GfxArchID archId) {
    // Skip adding registers for SDelayAlu - it uses SDelayAluData modifier instead
    if (stinkyInst->getUnifiedOpcode() == GFX::s_delay_alu) {
        return;
    }

    // Add destination registers
    for (const InstructionInput& dst : inst->getDstParams()) {
        StinkyRegister reg = toStinkyRegister(dst, hasVgprMsb, archId);
        if (reg.isValid()) {
            stinkyInst->addDestReg(reg);
        }
    }

    // Add source registers
    std::vector<InstructionInput> srcParams = inst->getSrcParams();

    // Adjust source parameters for VLShiftLeftAddU32 CompositeInstruction
    // VLShiftLeftAddU32 stores parameters as: {src0, src1, shift}
    // _VLShiftLeftAddU32 stores parameters as: {src0, shift, src1}
    // Assembly format is: v_lshl_add_u32 dst, src0, shift, src1
    if (typeid(*inst) == typeid(rocisa::VLShiftLeftAddU32)) {
        auto it = asmCaps.find("HasAddLshl");
        bool hasAddLshl = (it != asmCaps.end() && it->second);
        if (hasAddLshl) {
            // VLShiftLeftAddU32 order is src0, src1, shift
            // Need to swap to get: src0, shift, src1
            std::swap(srcParams[1], srcParams[2]);
        }
    }

    for (size_t i = 0; i < srcParams.size(); ++i) {
        StinkyRegister reg = toStinkyRegister(srcParams[i], hasVgprMsb, archId);
        if (reg.isValid()) {
            stinkyInst->addSrcReg(reg);
        }
    }

#ifndef NDEBUG
    // Verify: read-write operands must exist in both destRegs and srcRegs.
    {
        const auto& fields = stinkyInst->getHwInstDesc()->operandFields;
        for (const auto& field : fields) {
            if (!field.isReadWrite) continue;

            const auto& destRegs = stinkyInst->getDestRegs();
            const auto& srcRegs = stinkyInst->getSrcRegs();
            unsigned dIdx = 0, sIdx = 0;
            for (const auto& f : fields) {
                if (&f == &field) break;
                if (f.isDest || f.isReadWrite)
                    dIdx++;
                else
                    sIdx++;
            }

            const StinkyRegister* reg = nullptr;
            if (field.isDest && dIdx < destRegs.size())
                reg = &destRegs[dIdx];
            else if (sIdx < srcRegs.size())
                reg = &srcRegs[sIdx];

            if (reg && reg->dataType == StinkyRegister::Type::Register) {
                if (std::find(destRegs.begin(), destRegs.end(), *reg) == destRegs.end()) {
                    std::cerr << "Read-write operand missing from destRegs\n"
                              << "  Instruction: " << stinkyInst->getHwInstDesc()->mnemonic << ": ";
                    stinkyInst->dump(std::cerr);
                    std::cerr << "  Register: ";
                    reg->dump(std::cerr);
                    assert(false && "Read-write operand missing from destRegs");
                }
                if (std::find(srcRegs.begin(), srcRegs.end(), *reg) == srcRegs.end()) {
                    std::cerr << "Read-write operand missing from srcRegs\n"
                              << "  Instruction: " << stinkyInst->getHwInstDesc()->mnemonic << ": ";
                    stinkyInst->dump(std::cerr);
                    std::cerr << "  Register: ";
                    reg->dump(std::cerr);
                    assert(false && "Read-write operand missing from srcRegs");
                }
            }
        }
    }
#endif
}

/// Helper to extract neg_lo/neg_hi modifiers from instruction string
///
/// Searches for patterns like:
///   neg_lo:[x,x] or neg_lo:[x,x,x]
///   neg_hi:[x,x] or neg_hi:[x,x,x]
/// where x is a digit
///
/// \param instString The instruction string to search
/// \return MFMANegBits with per-source negLo/negHi arrays and numSrcs populated
static MFMANegBits extractNegModifiers(const std::string& instString) {
    MFMANegBits bits;

    auto parseArray = [&](const std::string& prefix, std::array<uint8_t, 3>& arr) -> int {
        size_t pos = instString.find(prefix);
        if (pos == std::string::npos) return 0;
        size_t start = instString.find('[', pos);
        size_t end = instString.find(']', pos);
        if (start == std::string::npos || end == std::string::npos || end <= start) return 0;
        auto inner = std::string_view(instString).substr(start + 1, end - start - 1);
        int count = 0;
        for (size_t i = 0; i < inner.size() && count < 3; ++i) {
            if (inner[i] >= '0' && inner[i] <= '1') {
                arr[count++] = static_cast<uint8_t>(inner[i] - '0');
            }
        }
        return count;
    };

    int loSrcs = parseArray("neg_lo:", bits.negLo);
    int hiSrcs = parseArray("neg_hi:", bits.negHi);
    bits.numSrcs = static_cast<uint8_t>(std::max(loSrcs, hiSrcs));

    return bits;
}

/// Helper to extract matrix/scale formats from instruction string.
/// Matches: matrix_a_fmt:xxx matrix_b_fmt:yyy [matrix_a_scale_fmt:z] [matrix_b_scale_fmt:w]
///      or: matrix_a_scale_fmt:z [matrix_b_scale_fmt:w]
static MatrixFmtModifiers extractMatrixFormats(std::string_view instString) {
    MatrixFmtModifiers fmts;

    auto extractValue = [&](const char* prefix) -> std::string_view {
        size_t pos = instString.find(prefix);
        if (pos == std::string_view::npos) return {};
        size_t valStart = pos + std::string_view(prefix).size();
        size_t valEnd = instString.find(' ', valStart);
        if (valEnd == std::string_view::npos) valEnd = instString.size();
        return instString.substr(valStart, valEnd - valStart);
    };

    auto aFmtVal = extractValue("matrix_a_fmt:");
    auto bFmtVal = extractValue("matrix_b_fmt:");
    if (!aFmtVal.empty()) fmts.fmtA = parseMatrixFmt(aFmtVal);
    if (!bFmtVal.empty()) fmts.fmtB = parseMatrixFmt(bFmtVal);

    auto aScaleVal = extractValue("matrix_a_scale_fmt:");
    auto bScaleVal = extractValue("matrix_b_scale_fmt:");
    if (!aScaleVal.empty()) fmts.scaleFmtA = parseMatrixScaleFmt(aScaleVal);
    if (!bScaleVal.empty()) fmts.scaleFmtB = parseMatrixScaleFmt(bScaleVal);

    return fmts;
}

/// Helper to handle MXMFMA instruction modifiers
void handleMXMFMAModifiers(StinkyInstruction* stinkyInst, const std::string& instString) {
    // MXMFMA does not support neg_lo/neg_hi modifiers; only matrix formats.
    auto fmts = extractMatrixFormats(instString);
    if (!fmts.empty()) stinkyInst->addModifier<MatrixFmtModifiers>(fmts);
    stinkyInst->addModifier<MFMAModifiers>(MFMAModifiers{});
}

/// Helper to handle MFMA instruction modifiers
void handleMFMAModifiers(StinkyInstruction* stinkyInst, const std::string& instString) {
    auto fmts = extractMatrixFormats(instString);
    if (!fmts.empty()) stinkyInst->addModifier<MatrixFmtModifiers>(fmts);

    MFMAModifiers mod;
    mod.negBits = extractNegModifiers(instString);
    stinkyInst->addModifier<MFMAModifiers>(mod);
}

/// Helper to handle SMFMA instruction modifiers
void handleSMFMAModifiers(StinkyInstruction* stinkyInst, const std::string& instString) {
    MFMAModifiers mod;
    mod.negBits = extractNegModifiers(instString);
    stinkyInst->addModifier<MFMAModifiers>(mod);
}

/// Helper to handle global_prefetch_b8 (gl2-prefetch) temporal-hint / cache-scope
/// modifiers. Read directly from rocisa's GLOBALModifiers (via getModifier()).
void handleGlobalPrefetchModifier(StinkyInstruction* stinkyInst,
                                  const rocisa::GlobalPrefetchB8* inst) {
    const std::optional<rocisa::GLOBALModifiers>& gm = inst->getModifier();
    if (!gm) return;

    stinkytofu::TemporalHint th = convertTemporalHint(gm->th);
    stinkytofu::MUBUFScope scope = convertMUBUFScope(gm->scope);
    if (th == stinkytofu::TemporalHint::TH_NONE && scope == stinkytofu::MUBUFScope::SCOPE_NONE &&
        gm->offset == 0) {
        return;
    }
    stinkyInst->addModifier<stinkytofu::GLOBALModifiers>(
        stinkytofu::GLOBALModifiers(gm->offset, th, scope));
}

/// Helper to handle SWaitCnt instruction modifiers
void handleSWaitCntModifiers(StinkyInstruction* stinkyInst, const rocisa::SWaitCnt* waitCntInst,
                             const std::map<std::string, int>& asmCaps) {
    auto itLgkm = asmCaps.find("MaxLgkmcnt");
    int maxLgkmcnt = itLgkm != asmCaps.end() ? itLgkm->second : -1;
    auto itVm = asmCaps.find("MaxVmcnt");
    int maxVmcnt = itVm != asmCaps.end() ? itVm->second : -1;
    SWaitCntData waitCntData(waitCntInst->vlcnt, waitCntInst->vscnt, -1, waitCntInst->dscnt,
                             waitCntInst->kmcnt, maxLgkmcnt, maxVmcnt);
    stinkyInst->addModifier<SWaitCntData>(waitCntData);
}

/// Helper to handle _SWaitDscnt instruction modifiers
void handleSWaitDscntModifiers(StinkyInstruction* stinkyInst,
                               const rocisa::_SWaitDscnt* waitCntInst,
                               const std::map<std::string, int>& asmCaps) {
    auto it = asmCaps.find("MaxDscnt");
    int maxDscnt = it != asmCaps.end() ? it->second : waitCntInst->getDscnt();
    int dscnt = std::min(waitCntInst->getDscnt(), maxDscnt);
    if (auto sWaitCntData = stinkyInst->getModifier<SWaitCntData>()) {
        sWaitCntData->dscnt = dscnt;
    } else {
        SWaitCntData waitCntData;
        waitCntData.dscnt = dscnt;
        stinkyInst->addModifier<SWaitCntData>(waitCntData);
    }
}

/// Helper to handle _SWaitLoadcnt instruction modifiers
void handleSWaitLoadcntModifiers(StinkyInstruction* stinkyInst,
                                 const rocisa::_SWaitLoadcnt* waitLoadcntInst,
                                 const std::map<std::string, int>& asmCaps) {
    auto it = asmCaps.find("MaxLoadcnt");
    int maxLoadcnt = it != asmCaps.end() ? it->second : waitLoadcntInst->getLoadcnt();
    int loadcnt = std::min(waitLoadcntInst->getLoadcnt(), maxLoadcnt);
    if (auto sWaitCntData = stinkyInst->getModifier<SWaitCntData>()) {
        sWaitCntData->vlcnt = loadcnt;
    } else {
        SWaitCntData waitCntData;
        waitCntData.vlcnt = loadcnt;
        stinkyInst->addModifier<SWaitCntData>(waitCntData);
    }
}

/// Helper to handle VCvt instruction True16 modifiers
void handleVCvtTrue16Modifiers(StinkyInstruction* stinkyInst,
                               const rocisa::VCvtInstruction* vcvtInst) {
    if (vcvtInst->true16.empty()) {
        return;
    }

    // Convert rocisa::True16Modifiers to stinkytofu True16Modifiers
    // rocisa uses indices: DST=0, DST1=1, SRC0=2, SRC1=3, ...
    stinkytofu::HighBitSel dst0 = stinkytofu::HighBitSel::NONE;
    stinkytofu::HighBitSel dst1 = stinkytofu::HighBitSel::NONE;
    std::vector<stinkytofu::HighBitSel> srcs;

    for (size_t i = 0; i < vcvtInst->true16.size(); ++i) {
        stinkytofu::HighBitSel highBit =
            static_cast<stinkytofu::HighBitSel>(static_cast<int>(vcvtInst->true16[i].high_bit));

        if (i == 0)  // DST
        {
            dst0 = highBit;
        } else if (i == 1)  // DST1
        {
            dst1 = highBit;
        } else  // SRC0, SRC1, ...
        {
            srcs.push_back(highBit);
        }
    }

    // Assert that source count is within the 2-bit encoding limit (max 6 sources)
    assert(srcs.size() <= 6 &&
           "True16Modifiers: source count must be <= 6 for uint16_t 2-bit encoding");

    stinkyInst->addModifier<stinkytofu::True16Modifiers>(
        stinkytofu::True16Modifiers(dst0, dst1, srcs));
}

/// Add modifiers to StinkyInstruction (DS, FLAT, MUBUF, SMEM, WaitCnt, DelayAlu)
void addModifiersToInstruction(StinkyInstruction* stinkyInst, const rocisa::Instruction* inst,
                               const std::map<std::string, int>& asmCaps) {
#define TRY_ADD_MOD(RocisaInstType, modField, StinkyModType, converter)                 \
    if (auto typed = dynamic_cast<const RocisaInstType*>(inst)) {                       \
        if (typed->modField.has_value()) {                                              \
            stinkyInst->addModifier<StinkyModType>(converter(typed->modField.value())); \
        }                                                                               \
    }

#define HANDLE_INST_TYPE(RocisaInstType, handlerCall)                 \
    if (auto typedInst = dynamic_cast<const RocisaInstType*>(inst)) { \
        handlerCall;                                                  \
    }

    // clang-format off
        // Chain all memory instruction types (mutually exclusive)
        TRY_ADD_MOD(DSLoadInstruction, ds, stinkytofu::DSModifiers, convertDSModifiers)
        else TRY_ADD_MOD(DSStoreInstruction, ds, stinkytofu::DSModifiers, convertDSModifiers)
        else TRY_ADD_MOD(FLATReadInstruction, flat, stinkytofu::FLATModifiers,
            [&](const auto& mod) { return convertFLATModifiers(mod, asmCaps); })
        else TRY_ADD_MOD(FLATStoreInstruction, flat, stinkytofu::FLATModifiers,
            [&](const auto& mod) { return convertFLATModifiers(mod, asmCaps); })
        else if (auto typed = dynamic_cast<const MUBUFReadInstruction*>(inst)) {
            stinkyInst->addModifier<stinkytofu::MUBUFModifiers>(
                buildMUBUFModifiersForBufferOp(typed->mubuf, typed->vaddr.get(), asmCaps));
        }
        else if (auto typed = dynamic_cast<const MUBUFStoreInstruction*>(inst)) {
            stinkyInst->addModifier<stinkytofu::MUBUFModifiers>(
                buildMUBUFModifiersForBufferOp(typed->mubuf, typed->vaddr.get(), asmCaps));
        }
        else TRY_ADD_MOD(SMemLoadInstruction, smem, stinkytofu::SMEMModifiers,
            [&](const auto& mod) { return convertSMEMModifiers(mod, asmCaps); })
        else TRY_ADD_MOD(SMemStoreInstruction, smem, stinkytofu::SMEMModifiers,
            [&](const auto& mod) { return convertSMEMModifiers(mod, asmCaps); })
        else TRY_ADD_MOD(SMemAtomicDecInstruction, smem, stinkytofu::SMEMModifiers,
            [&](const auto& mod) { return convertSMEMModifiers(mod, asmCaps); })
        else
        {
            // No memory modifier matched
            TRY_ADD_MOD(CommonInstruction, vop3, stinkytofu::VOP3PModifiers, convertVOP3PModifiers)
            TRY_ADD_MOD(CommonInstruction, sdwa, stinkytofu::SDWAModifiers, convertSDWAModifiers)
            TRY_ADD_MOD(CommonInstruction, dpp, stinkytofu::DPPModifiers, convertDPPModifiers)

            // VOP/SOP instructions - these can overlap with CommonInstruction base class
            HANDLE_INST_TYPE(rocisa::MXMFMAInstruction, handleMXMFMAModifiers(stinkyInst, itemToString(inst)))
            else HANDLE_INST_TYPE(rocisa::MFMAInstruction, handleMFMAModifiers(stinkyInst, itemToString(inst)))
            else HANDLE_INST_TYPE(rocisa::SMFMAInstruction, handleSMFMAModifiers(stinkyInst, itemToString(inst)))
            else HANDLE_INST_TYPE(rocisa::VCvtInstruction, handleVCvtTrue16Modifiers(stinkyInst, typedInst))

            // Control/Synchronization instructions, separate from VOP/SOP
            else HANDLE_INST_TYPE(rocisa::SDelayAlu,
                                stinkyInst->addModifier<SDelayAluData>(convertSDelayAluData(typedInst)))
            else HANDLE_INST_TYPE(rocisa::SWaitCnt, handleSWaitCntModifiers(stinkyInst, typedInst, asmCaps))
            else HANDLE_INST_TYPE(rocisa::_SWaitDscnt, handleSWaitDscntModifiers(stinkyInst, typedInst, asmCaps))
            else HANDLE_INST_TYPE(rocisa::_SWaitLoadcnt, handleSWaitLoadcntModifiers(stinkyInst, typedInst, asmCaps))

            // global_wb / global_inv: device-scope memory fences with no
            // register operands. The only encoded modifier is the cache
            // scope, which we attach as a dedicated CacheScopeModifiers so
            // MUBUFModifiers can evolve independently without affecting
            // fence emission.
            else HANDLE_INST_TYPE(rocisa::GlobalWb,
                                stinkyInst->addModifier<stinkytofu::CacheScopeModifiers>(
                                    stinkytofu::CacheScopeModifiers(
                                        convertMUBUFScope(typedInst->scope))))
            else HANDLE_INST_TYPE(rocisa::GlobalInv,
                                stinkyInst->addModifier<stinkytofu::CacheScopeModifiers>(
                                    stinkytofu::CacheScopeModifiers(
                                        convertMUBUFScope(typedInst->scope))))

            // global_prefetch_b8 (gl2-prefetch): temporal hint + cache scope.
            else HANDLE_INST_TYPE(rocisa::GlobalPrefetchB8,
                                handleGlobalPrefetchModifier(stinkyInst, typedInst))
        }
    // clang-format on

#undef TRY_ADD_MOD
#undef HANDLE_INST_TYPE

    // Always add comment if present
    if (!inst->comment.empty()) {
        stinkyInst->addModifier<CommentData>(CommentData{inst->comment});
    }
}

/// Get MSB value from StinkyRegister if it's a VGPR. Returns -1 for non-VGPR.
int getMsbFromStinkyVgpr(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V) return -1;
    return static_cast<int>(reg.reg.idx) / 256;
}

int getMsbOffsetFromStinkyVgpr(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register || reg.reg.type != RegType::V) return 0;
    return getMsbFromStinkyVgpr(reg) * (-256);
}

/// Convert a rocisa::Container to StinkyRegister
///
/// This function takes a rocisa::Container pointer and converts it to a
/// StinkyRegister. It handles RegisterContainer types by extracting the
/// register type, index, and number of registers.
///
/// \param container Pointer to rocisa::Container to convert
/// \param hasVgprMsb Whether VGPR MSB is supported (affects register offset for VGPRs > 255)
/// \return StinkyRegister representing the container, or invalid register if conversion fails
StinkyRegister toStinkyRegister(const rocisa::Container* container, bool hasVgprMsb,
                                GfxArchID arch) {
    if (const rocisa::RegisterContainer* regCont =
            dynamic_cast<const rocisa::RegisterContainer*>(container)) {
        // isOff=true signals the MUBUF "off" keyword (no address register).
        // rocisa emits "off" for this case; produce a literal string so the
        // emitter writes "off" instead of treating it as a named VGPR.
        if (regCont->isOff) {
            return StinkyRegister("off");
        }

        RegType regType = stringToRegType(regCont->regType);

        int physicalIdx = regCont->regIdx;
        if (regCont->regName && regType == RegType::V) {
            physicalIdx = regCont->regName->getTotalIdx();
        }

        StinkyRegister reg{regType, static_cast<uint32_t>(physicalIdx),
                           static_cast<uint16_t>(regCont->regNum)};

        reg.reg.isMinus = regCont->isMinus ? 1 : 0;
        reg.reg.isAbs = regCont->isAbs ? 1 : 0;

        // TODO: This is a hack to set the offset of the register for use case such as msb, etc.
        // Set offset for VGPR MSB when supported (use case: vgpr > 255)
        if (hasVgprMsb) reg.reg.offset = static_cast<int16_t>(getMsbOffsetFromStinkyVgpr(reg));

        // Capture symbolic register name if available
        // In rocisa, the symbolic name includes the type prefix and all offsets
        // (e.g., "vgprLocalWriteAddrA+0" or "vgprValuA_X0_I0+4")
        if (regCont->regName.has_value()) {
            // regName->toString() includes the base name and all offsets
            std::string fullName = regCont->getCompleteRegNameWithType();
            reg.setSymbolicName(fullName);
        }

        return reg;
    }
    if (const rocisa::VCC* vccCont = dynamic_cast<const rocisa::VCC*>(container)) {
        RegType regType = stringToRegType(vccCont->toString());
        return StinkyRegister(regType, 0, 1);
    }
    if (const rocisa::EXEC* execCont = dynamic_cast<const rocisa::EXEC*>(container)) {
        RegType regType = stringToRegType(execCont->toString());
        return StinkyRegister(regType, 0, 1);
    }
    if (const rocisa::HWRegContainer* hwregContainer =
            dynamic_cast<const rocisa::HWRegContainer*>(container)) {
        uint16_t id = HwReg::parseId(arch, hwregContainer->reg).value_or(0);
        uint16_t offset =
            hwregContainer->value.size() > 0 ? static_cast<uint16_t>(hwregContainer->value[0]) : 0;
        uint16_t size =
            hwregContainer->value.size() > 1 ? static_cast<uint16_t>(hwregContainer->value[1]) : 32;
        return StinkyRegister::Hwreg(id, offset, size);
    }
    return StinkyRegister{};
}

/// Convert a rocisa::InstructionInput to StinkyRegister
///
/// This overload handles InstructionInput variants which can contain:
/// - shared_ptr<Container> (converted via RegisterContainer)
/// - int literals
/// - double literals
/// - string literals
///
/// \param input The InstructionInput variant to convert
/// \param hasVgprMsb Whether VGPR MSB is supported
/// \return StinkyRegister representing the input value
StinkyRegister toStinkyRegister(const InstructionInput& input, bool hasVgprMsb, GfxArchID arch) {
    if (auto pptr = std::get_if<std::shared_ptr<rocisa::Container>>(&input)) {
        return toStinkyRegister(pptr->get(), hasVgprMsb, arch);
    } else if (const int* literalInt = std::get_if<int>(&input)) {
        return StinkyRegister(*literalInt);
    } else if (const double* literalDouble = std::get_if<double>(&input)) {
        return StinkyRegister(*literalDouble);
    } else if (const std::string* literalString = std::get_if<std::string>(&input)) {
        // Try to convert numeric strings to integers
        if (!literalString->empty()) {
            size_t start = 0;

            // Check for optional leading minus sign
            if ((*literalString)[0] == '-') {
                start = 1;
                if (literalString->length() == 1) {
                    return StinkyRegister(*literalString);
                }
            }

            // Check if all remaining characters are digits
            for (size_t i = start; i < literalString->length(); ++i)
                if (!std::isdigit(static_cast<unsigned char>((*literalString)[i])))
                    return StinkyRegister(*literalString);

            int value = std::atoi(literalString->c_str());
            return StinkyRegister(value);
        }
        return StinkyRegister(*literalString);
    }

    return StinkyRegister{};
}

/// Convert rocisa::SignatureValueKind to stinkytofu::SignatureValueKind
stinkytofu::SignatureValueKind convertSignatureValueKind(rocisa::SignatureValueKind kind) {
    switch (kind) {
        case rocisa::SignatureValueKind::SIG_GLOBALBUFFER:
            return stinkytofu::SignatureValueKind::SIG_GLOBALBUFFER;
        case rocisa::SignatureValueKind::SIG_VALUE:
            return stinkytofu::SignatureValueKind::SIG_VALUE;
        default:
            return stinkytofu::SignatureValueKind::SIG_VALUE;
    }
}

/// Convert rocisa::SignatureBase to stinkytofu::SignatureBase
std::shared_ptr<stinkytofu::SignatureBase> toStinkySignature(const rocisa::SignatureBase& rocisaSig,
                                                             const std::array<int, 3>& isaVersion,
                                                             int wavefrontSize) {
    // Extract kernel descriptor info
    const auto& kd = rocisaSig.kernelDescriptor;

    // Extract code metadata info
    const auto& cm = rocisaSig.codeMeta;

    // Use wavefrontSize passed from Python (kernel["WavefrontSize"])

    // Create stinkytofu signature
    auto stinkySig = std::make_shared<stinkytofu::SignatureBase>(
        rocisaSig.name, isaVersion, cm.kernArgsVersion, cm.codeObjectVersion, kd.groupSegSize,
        kd.sgprWorkGroup, kd.vgprWorkItem, cm.flatWgSize, wavefrontSize, kd.originalTotalVgprs,
        kd.totalAgprs, kd.totalSgprs, kd.numSgprPreload);

    // Convert arguments
    for (const auto& arg : cm.argList) {
        stinkySig->addArg(arg.name, convertSignatureValueKind(arg.valueKind), arg.valueType,
                          arg.addrSpaceQual);
    }

    // Note: Optimization config (ThreadTile, WaveGroup, VectorWidth, etc.) is now passed
    // directly from Python via toStinkyTofuModule's parameters and set via setOptimizationConfig()

    return stinkySig;
}

/**
 * @brief Visitor function type for processing each item in the rocisa::Module
 * @param item The current item being processed
 * @param moduleNames The hierarchical module names of the current item
 */
using ItemVisitor =
    std::function<void(rocisa::Item*, const std::vector<const std::string*>& moduleNames)>;

/**
 * @brief traversal rocisa::Module with DFS path and process each item
 * @param module The rocisa::Module to traverse
 * @param parentModuleNames The hierarchical module names of the parent items
 * @param visitor The visitor to process each item
 */
void traverseModule(const rocisa::Module& module,
                    const std::vector<const std::string*>& parentModuleNames, ItemVisitor visitor) {
    std::vector<const std::string*> moduleNames(parentModuleNames);
    moduleNames.push_back(&module.name);
    for (auto& item : module.itemList) {
        if (const auto subModule = dynamic_cast<const rocisa::Module*>(item.get())) {
            traverseModule(*subModule, moduleNames, visitor);
        } else {
            visitor(item.get(), moduleNames);
        }
    }
}

}  // anonymous namespace

namespace stinkytofu {
static std::shared_ptr<StinkyAsmModule> toStinkyTofuModule(
    const rocisa::Module& module, std::array<int, 3> arch, const std::string& moduleName,
    const StinkyAsmModule::ModuleOptions& moduleOptions) {
    // Get GfxArchID from architecture array
    GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

    // VgprMsbMode is auto-probed by Backend::configurePassManager() when it
    // sees VgprMsbMode::None, so no need to read it from rocisa caps here.
    StinkyAsmModule stinkyAsmModule(moduleName, arch, moduleOptions);

    // Add instruction groups registered by the target backend.
    if (auto* pipeline = BackendRegistry::getArchPipeline(arch)) {
        for (const auto& groupName : pipeline->groupNames) {
            stinkyAsmModule.addGroup(groupName);
        }
    }

    // TODO: We can create BasicBlocks when visiting Labels.
    BasicBlock* currentBB = stinkyAsmModule.getFunction().getEntryBlock();

    // Create IRBuilder for lower-level instruction creation
    AsmIRBuilder irBuilder(*currentBB, archId);

    // Process each item
    std::map<std::string, int> asmCaps = rocisa::rocIsa::getInstance().getAsmCaps();
    std::map<std::string, int> archCaps = rocisa::rocIsa::getInstance().getArchCaps();
    bool hasVgprMsb = asmCaps.count("HasVgprMSB") && asmCaps.at("HasVgprMSB");

    auto processItem = [&](rocisa::Item* item, const std::vector<const std::string*>& moduleNames) {
        const auto instsCountBefore = currentBB->size();

        // Handle text blocks
        if (rocisa::TextBlock* textBlock = dynamic_cast<rocisa::TextBlock*>(item)) {
            AsmDirective* directive = IRBase::createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = textBlock->text;
            currentBB->appendIR(directive);
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle labels
        if (rocisa::Label* rocLabel = dynamic_cast<rocisa::Label*>(item)) {
            StinkyInstruction* labelInst =
                irBuilder.createLabel(rocLabel->getLabelName(), rocLabel->alignment);

            // Add comment if present
            if (!rocLabel->comment.empty()) {
                labelInst->addModifier<CommentData>(CommentData{rocLabel->comment});
            }

            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle ValueSet directives
        if (rocisa::ValueSet* valueSet = dynamic_cast<rocisa::ValueSet*>(item)) {
            AsmDirective* directive = IRBase::createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::SET;
            directive->name = ".set";
            directive->symbol = valueSet->name;
            std::string itemString = itemToString(item);

            // get the last value after the last comma
            size_t pos = itemString.rfind(',');
            if (pos != std::string::npos) {
                directive->value = itemString.substr(pos + 1);
                directive->value.erase(0, directive->value.find_first_not_of(" \t\n\r"));
                directive->value.erase(directive->value.find_last_not_of(" \t\n\r") + 1);
            }

            currentBB->appendIR(directive);
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle macro directives
        if (rocisa::Macro* macro = dynamic_cast<rocisa::Macro*>(item)) {
            AsmDirective* directive = IRBase::createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::MACRO;
            directive->name = ".macro";
            directive->symbol = macro->name;
            directive->value = itemToString(item);
            currentBB->appendIR(directive);
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle ValueIf directives
        if (rocisa::ValueIf* valueIf = dynamic_cast<rocisa::ValueIf*>(item)) {
            AsmDirective* directive = IRBase::createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::IF;
            directive->name = ".if";
            directive->symbol = valueIf->value;
            directive->value = itemToString(item);
            currentBB->appendIR(directive);
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle ValueEndif directives
        if (rocisa::ValueEndif* valueEndif = dynamic_cast<rocisa::ValueEndif*>(item)) {
            AsmDirective* directive = IRBase::createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::ENDIF;
            directive->name = ".endif";
            directive->comment = valueEndif->comment;
            directive->value = itemToString(item);
            currentBB->appendIR(directive);
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle macro instruction calls
        if (auto* macroInst = dynamic_cast<rocisa::MacroInstruction*>(item)) {
            AsmDirective* directive = IRBase::createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->comment = macroInst->comment;
            directive->value = itemToString(item);
            currentBB->appendIR(directive);
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
            return;
        }

        // Handle instructions
        rocisa::Instruction* inst = dynamic_cast<rocisa::Instruction*>(item);
        if (inst == nullptr) {
            // TODO: Remove this once we have a better way to handle non-instruction items
            std::cout << "Skipping non-instruction item: " << itemToString(item) << std::endl;
            return;
        }
        assert(dynamic_cast<rocisa::SSetVgprMsb*>(inst) == nullptr &&
               "SSetVgprMsb should not be created directly in TensileLite");

        // Create StinkyInstruction from rocisa instruction
        std::type_index rocisaTy = std::type_index(typeid(*inst));
        StinkyInstruction* stinkyInst =
            createStinkyInstructionFromRocisa(*inst, rocisaTy, irBuilder, archId);

        assert(stinkyInst != nullptr &&
               "Failed to create StinkyInstruction from rocisa instruction");

        // Add registers (sources and destinations) to the instruction
        addRegistersToInstruction(stinkyInst, inst, asmCaps, hasVgprMsb, archId);

        // Add modifiers (DS, FLAT, MUBUF, SMEM, WaitCnt, comments)
        addModifiersToInstruction(stinkyInst, inst, asmCaps);

        if (auto memToken = inst->getMemToken()) {
            stinkyInst->addModifier<MemTokenData>(MemTokenData{memToken->tokens});
        }

        Legalized legalizedInsts =
            legalizeInstruction(stinkyInst, inst, irBuilder, archId, asmCaps, archCaps, hasVgprMsb);

        if (legalizedInsts.first != nullptr) {
            StinkyInstruction* currentStinkyInst = legalizedInsts.first;
            while (currentStinkyInst != legalizedInsts.last->getNext()) {
                stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
                currentStinkyInst = static_cast<StinkyInstruction*>(currentStinkyInst->getNext());
            }
        } else {
            stinkyAsmModule.updateInstructionGroups(moduleNames, instsCountBefore);
        }
    };

    // Check whether a rocisa Instruction is a global/buffer/flat load or tensor load.
    // Excludes SMemLoadInstruction (s_load) which also inherits from GlobalReadInstruction.
    auto isPrefetchLoadInst = [](const rocisa::Instruction* inst) -> bool {
        // NOLINTNEXTLINE(misc-redundant-expression)
        return dynamic_cast<const rocisa::MUBUFReadInstruction*>(inst) ||
               dynamic_cast<const rocisa::GLOBALLoadInstruction*>(inst) ||
               dynamic_cast<const rocisa::FLATReadInstruction*>(inst) ||
               dynamic_cast<const rocisa::TensorLoadToLds*>(inst);
    };

    // Recursively check whether an item contains a prefetch load instruction.
    std::function<bool(const rocisa::Item*)> containsPrefetchLoad =
        [&](const rocisa::Item* item) -> bool {
        if (const auto* inst = dynamic_cast<const rocisa::Instruction*>(item))
            return isPrefetchLoadInst(inst);
        if (const auto* mod = dynamic_cast<const rocisa::Module*>(item)) {
            for (const auto& child : mod->itemList)
                if (containsPrefetchLoad(child.get())) return true;
        }
        return false;
    };

    // Auto-detect the loopWithPrefetch region: from the first global read or
    // tensor load item up to and including Module("loopBody").
    int pgrStartIdx = -1;
    int loopBodyIdx = -1;
    for (int i = 0; i < static_cast<int>(module.itemList.size()); ++i) {
        const auto& item = module.itemList[i];
        if (pgrStartIdx == -1 && containsPrefetchLoad(item.get())) {
            pgrStartIdx = i;
        }
        if (const auto* subMod = dynamic_cast<const rocisa::Module*>(item.get())) {
            if (subMod->name == "loopBody") {
                loopBodyIdx = i;
                break;
            }
        }
    }

    const bool hasPGR = (pgrStartIdx != -1 && loopBodyIdx != -1 && pgrStartIdx <= loopBodyIdx);
    static const std::string kPGR = "loopWithPrefetch";

    // Recursively check whether an item's subtree contains a Label with \p name.
    std::function<bool(const rocisa::Item*, const std::string&)> containsLabel =
        [&](const rocisa::Item* item, const std::string& name) -> bool {
        if (const auto* lbl = dynamic_cast<const rocisa::Label*>(item))
            return lbl->getLabelName() == name;
        if (const auto* mod = dynamic_cast<const rocisa::Module*>(item)) {
            for (const auto& child : mod->itemList)
                if (containsLabel(child.get(), name)) return true;
        }
        return false;
    };

    // Recursively check whether an item is, or contains, a Module named \p name.
    std::function<bool(const rocisa::Item*, const std::string&)> containsModule =
        [&](const rocisa::Item* item, const std::string& name) -> bool {
        if (const auto* mod = dynamic_cast<const rocisa::Module*>(item)) {
            if (mod->name == name) return true;
            for (const auto& child : mod->itemList)
                if (containsModule(child.get(), name)) return true;
        }
        return false;
    };

    // Auto-detect the expertScheduleMode2 region: from the top-level item whose
    // subtree contains label_Preload_Offset_Start (kernel-body entry, before the
    // first VGPR producer) through Module("noLoadLoopBody"). This is the region
    // the wait-alu / mode2 ScopeAdaptor operates on — it deliberately excludes
    // the epilogue (Global Write), where all activation calls live and which must
    // stay in mode0.
    // Anchor the region start at label_ASM_Start, the main-body entry. It
    // precedes label_Preload_Offset_Start in program order, so starting here
    // keeps BOTH entry labels inside the region: the non-preload path enters at
    // label_ASM_Start and falls through; the kernarg-preload path jumps to
    // label_Preload_Offset_Start (further in). InsertWaitAluPass then enables
    // mode2 at each entry label it finds, covering both paths. Fall back to
    // label_Preload_Offset_Start if label_ASM_Start is somehow absent.
    int scopeStartIdx = -1;
    int scopeEndIdx = -1;
    for (int i = 0; i < static_cast<int>(module.itemList.size()); ++i) {
        const auto& item = module.itemList[i];
        if (scopeStartIdx == -1 && containsLabel(item.get(), "label_ASM_Start")) {
            scopeStartIdx = i;
        }
        if (containsModule(item.get(), "noLoadLoopBody")) scopeEndIdx = i;
    }
    if (scopeStartIdx == -1) {
        for (int i = 0; i < static_cast<int>(module.itemList.size()); ++i) {
            if (containsLabel(module.itemList[i].get(), "label_Preload_Offset_Start")) {
                scopeStartIdx = i;
                break;
            }
        }
    }
    const bool hasScope =
        (scopeStartIdx != -1 && scopeEndIdx != -1 && scopeStartIdx <= scopeEndIdx);
    static const std::string kScope = "expertScheduleMode2";

    // Traverse top-level items, injecting the loopWithPrefetch group name
    // for items in the detected prefetch region [pgrStartIdx, loopBodyIdx].
    for (int i = 0; i < static_cast<int>(module.itemList.size()); ++i) {
        const auto& item = module.itemList[i];
        const bool inPGR = hasPGR && (i >= pgrStartIdx && i <= loopBodyIdx);
        const bool inScope = hasScope && (i >= scopeStartIdx && i <= scopeEndIdx);

        std::vector<const std::string*> base;
        if (inScope) base.push_back(&kScope);
        if (inPGR) base.push_back(&kPGR);
        base.push_back(&module.name);

        if (const auto* subMod = dynamic_cast<const rocisa::Module*>(item.get())) {
            traverseModule(*subMod, base, processItem);
        } else {
            processItem(item.get(), base);
        }
    }

    return std::make_shared<StinkyAsmModule>(std::move(stinkyAsmModule));
}

}  // namespace stinkytofu

namespace {
/**
 * @brief Convert a Python sequence (tuple or list) to a std::array<int, 3>
 * @param arch_obj The Python sequence (tuple or list)
 * @return The std::array<int, 3> [major, minor, stepping]
 */
std::array<int, 3> convertArch(nb::object arch_obj) {
    // Convert Python sequence (tuple or list) to std::array
    if (!nb::isinstance<nb::sequence>(arch_obj)) {
        throw std::invalid_argument(
            "arch must be a tuple or list of 3 integers [major, minor, stepping]");
    }

    auto arch_seq = nb::cast<nb::sequence>(arch_obj);
    if (nb::len(arch_seq) != 3) {
        throw std::invalid_argument("arch must have exactly 3 elements [major, minor, stepping]");
    }

    return {nb::cast<int>(arch_seq[0]), nb::cast<int>(arch_seq[1]), nb::cast<int>(arch_seq[2])};
}

}  // anonymous namespace

/// Initialize StinkyTofu Python bindings
///
/// This function binds the rocisa to StinkyTofu utilities to Python, allowing
/// Python code to convert rocisa to StinkyTofu IR.
///
/// \param m The nanobind module to add bindings to
void init_stinkytofu(nb::module_ m) {  // NOLINT(misc-use-internal-linkage)
    // Pipeline extension point enum
    nb::enum_<PipelineExtensionPoint>(m, "PipelineExtensionPoint")
        .value("BeforeRegionPasses", PipelineExtensionPoint::BeforeRegionPasses)
        .value("InnerRegionBegin", PipelineExtensionPoint::InnerRegionBegin)
        .value("InnerRegionEnd", PipelineExtensionPoint::InnerRegionEnd)
        .value("AfterRegionPasses", PipelineExtensionPoint::AfterRegionPasses);

    m.def("loadPlugin", &PassBuilder::loadPlugin, nb::arg("path"),
          "Load a plugin shared library (.so/.dll) that exports registerPlugin()");
    m.def("loadPluginsFromDirectory", &PassBuilder::loadPluginsFromDirectory, nb::arg("dirPath"),
          "Load all plugin shared libraries (.so/.dll) from a directory");
    m.def("stinkytofuExamplePluginPath", &PassBuilder::examplePluginPath,
          "Absolute path to StinkyTofu's bundled example plugin, or \"\" if it was not built. "
          "For tests/demos; consumers with their own plugins pass their path to loadPlugin().");

    // Bind isSupportedByStinkyTofu to check if the architecture is supported by StinkyTofu
    m.def(
        "isSupportedByStinkyTofu",
        [](nb::object arch_obj) {
            std::array<int, 3> archArray = convertArch(arch_obj);
            return BackendRegistry::getArchPipeline(archArray) != nullptr;
        },
        nb::arg("arch"), "Check if the architecture is supported by StinkyTofu");
    m.def("getRegisteredArchKeys", &BackendRegistry::getRegisteredArchKeys,
          "Return a list of arch name strings for all registered StinkyTofu backends (e.g. "
          "[\"gfx1250\"]).");
    // Wrapper class to add signature support to StinkyAsmModule
    class StinkyAsmModuleWithSignature {
       private:
        std::shared_ptr<StinkyAsmModule> module_;
        std::shared_ptr<stinkytofu::SignatureBase> signature_;

       public:
        StinkyAsmModuleWithSignature(std::shared_ptr<StinkyAsmModule> module,
                                     std::shared_ptr<stinkytofu::SignatureBase> signature)
            : module_(module), signature_(signature) {}

        // Forward all StinkyAsmModule methods
        void runOptimizationPipeline() {
            module_->runOptimizationPipeline();
        }

        std::string getName() const {
            return module_->getName();
        }

        void setOutputName(const std::string& name) {
            module_->setOutputName(name);
        }

        std::string getOutputName() const {
            return module_->getOutputName();
        }

        void setOutputDir(const std::string& dir) {
            module_->setOutputDir(dir);
        }

        std::string getOutputDir() const {
            return module_->getOutputDir();
        }

        // Override emitAssembly to include signature
        std::string emitAssembly() const {
            std::string result;
            if (signature_) {
                int64_t totalBytes = module_->getTotalInstructionBytes();
                if (totalBytes >= 0) signature_->setTotalInstructionBytes(totalBytes);
                result = signature_->toString();
            }
            result += module_->emitAssembly();
            return result;
        }

        // Plugin data forwarding
        void setPluginDataI64(const std::string& key, int64_t value) {
            module_->setPluginDataI64(key, value);
        }
        int64_t getPluginDataI64(const std::string& key, int64_t defaultVal = 0) const {
            return module_->getPluginDataI64(key, defaultVal);
        }
        void setPluginDataStr(const std::string& key, const std::string& value) {
            module_->setPluginDataStr(key, value);
        }
        std::string getPluginDataStr(const std::string& key,
                                     const std::string& defaultVal = "") const {
            return module_->getPluginDataStr(key, defaultVal);
        }

        void registerPassAtExtensionPoint(PipelineExtensionPoint ep, const std::string& passName) {
            module_->getPassBuilder().registerAtExtensionPoint(
                ep, [passName](PassManager& PM, StinkyAsmModule& module) {
                    auto pass = PassBuilder::createPassByName(passName, module);
                    if (pass) PM.addPass(std::move(pass));
                });
        }

        // Provide access to underlying module if needed
        std::shared_ptr<StinkyAsmModule> getModule() const {
            return module_;
        }
    };

    // Bind the wrapper class
    nb::class_<StinkyAsmModuleWithSignature>(m, "StinkyAsmModule")
        .def("runOptimizationPipeline", &StinkyAsmModuleWithSignature::runOptimizationPipeline)
        .def("emitAssembly", &StinkyAsmModuleWithSignature::emitAssembly)
        .def("getName", &StinkyAsmModuleWithSignature::getName)
        .def("setOutputName", &StinkyAsmModuleWithSignature::setOutputName,
             "Set full kernel name for output files (e.g. cost file); should match .o basename")
        .def("getOutputName", &StinkyAsmModuleWithSignature::getOutputName)
        .def("setOutputDir", &StinkyAsmModuleWithSignature::setOutputDir,
             "Set output dir for cost file: comparison_output/<yaml_name>; file at "
             "<dir>/<kernel_name>/aggregated_instruction_cost.txt")
        .def("getOutputDir", &StinkyAsmModuleWithSignature::getOutputDir)
        .def("getModule", &StinkyAsmModuleWithSignature::getModule)
        .def("setPluginDataI64", &StinkyAsmModuleWithSignature::setPluginDataI64, nb::arg("key"),
             nb::arg("value"), "Set an integer plugin data value accessible by plugin passes")
        .def("getPluginDataI64", &StinkyAsmModuleWithSignature::getPluginDataI64, nb::arg("key"),
             nb::arg("defaultVal") = 0, "Get an integer plugin data value")
        .def("setPluginDataStr", &StinkyAsmModuleWithSignature::setPluginDataStr, nb::arg("key"),
             nb::arg("value"), "Set a string plugin data value accessible by plugin passes")
        .def("getPluginDataStr", &StinkyAsmModuleWithSignature::getPluginDataStr, nb::arg("key"),
             nb::arg("defaultVal") = "", "Get a string plugin data value")
        .def("registerPassAtExtensionPoint",
             &StinkyAsmModuleWithSignature::registerPassAtExtensionPoint, nb::arg("extensionPoint"),
             nb::arg("passName"), "Register a named C++ pass at a pipeline extension point");

    // Bind toStinkyTofuModule with signature support
    m.def(
        "toStinkyTofuModule",
        [](const rocisa::Module& module, nb::object arch_obj, const std::string& moduleName,
           const rocisa::SignatureBase& signature, nb::object options_obj) {
            // Convert architecture to std::array<int, 3> [major, minor, stepping]
            std::array<int, 3> archArray = convertArch(arch_obj);

            // Override with options dict if provided
            StinkyAsmModule::ModuleOptions moduleOptions{};
            // Sentinel: <0 means use legacy default scratch SGPR in SwPrefetchInsertionPass (102).
            moduleOptions.SwPrefetchScratchSgpr = -1;
            if (nb::isinstance<nb::dict>(options_obj)) {
                nb::dict options = nb::cast<nb::dict>(options_obj);

                bool hasSetOptions = false;

            // Set stinky module options from valid options in the options dict
#define SET_MODULE_OPTION(name, type) \
    hasSetOptions |=                  \
        (options.contains(#name) && nb::try_cast<type>(options[#name], moduleOptions.name));

#define DEBUG_SET_MODULE_OPTION(name, type)                                                  \
    if (options.contains(#name) && nb::try_cast<type>(options[#name], moduleOptions.name)) { \
        std::cout << "Setting " << #name << " to " << moduleOptions.name << std::endl;       \
        hasSetOptions = true;                                                                \
    }

                MODULE_OPTIONS_LIST(SET_MODULE_OPTION)
#undef SET_MODULE_OPTION
#undef DEBUG_SET_MODULE_OPTION
            }

            // Convert module to StinkyAsmModule (StinkyAsmModule ctor sets
            // EnableSwPrefetchInsertion from Sgpr != -1)
            auto stinkyModule =
                stinkytofu::toStinkyTofuModule(module, archArray, moduleName, moduleOptions);

            // Convert signature to StinkyTofu format, using the wavefrontSize passed from Python
            auto stinkySig = toStinkySignature(signature, archArray, moduleOptions.wavefrontSize);

            // Expose per-wave VGPR allocation on the Function.
            stinkyModule->getFunction().setMetaData(
                kSigTotalVgprsMetaKey,
                static_cast<uint64_t>(stinkySig->kernelDescriptor.totalVgprs));

            // Set optimization config
            std::array<int, 2> tt = {moduleOptions.TileA0, moduleOptions.TileB0};
            std::array<int, 2> sg = {moduleOptions.SubGroup0, moduleOptions.SubGroup1};
            std::array<int, 2> wg = {moduleOptions.WaveGroup0, moduleOptions.WaveGroup1};

            stinkySig->setOptimizationConfig(
                tt, sg, wg, moduleOptions.VectorWidthA, moduleOptions.VectorWidthB,
                moduleOptions.GlobalReadVectorWidthA, moduleOptions.GlobalReadVectorWidthB,
                moduleOptions.DirectToLdsA, moduleOptions.DirectToLdsB,
                moduleOptions.UseSgprForGRO);

            // Create and return wrapper with both
            return std::make_shared<StinkyAsmModuleWithSignature>(stinkyModule, stinkySig);
        },
        nb::arg("module"), nb::arg("arch"), nb::arg("moduleName") = "", nb::arg("signature"),
        nb::arg("options") = nb::none(),
        "Convert a rocisa.Module to a StinkyTofu StinkyAsmModule with signature support. "
        "The returned object's emitAssembly() will include both signature and instructions. "
        "Options may be passed via the 'options' dict (keys: TileA0, TileB0, TileM0, NumGRA, "
        "NumGRB, NumGRM, wavefrontSize, etc.); dict values override individual parameters.");
}
