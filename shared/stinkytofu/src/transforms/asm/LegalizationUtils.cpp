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

#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

#include <limits>
#include <regex>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {
Legalized legalizeVNop(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId) {
    assert(inst->getUnifiedOpcode() == GFX::v_nop && "Invalid v_nop instruction");
    // Check if this is a v_nop with count > 1
    const auto& srcRegs = inst->getSrcRegs();

    int count = inst->getSrcReg(0).literalInt;
    if (count <= 1) {
        inst->erase();
        return {nullptr, nullptr};
    }

    // Get the instruction descriptor and comment
    const HwInstDesc* desc = inst->getHwInstDesc();
    const CommentData* comment = inst->getModifier<CommentData>();

    StinkyInstruction* firstInst = nullptr;
    StinkyInstruction* lastInst = nullptr;

    // Create 'count' separate nop instructions before inst
    for (int i = 0; i < count; ++i) {
        StinkyInstruction* newInst = irBuilder.create(desc, inst);

        // v_nop has no operands - just add comment if present
        if (comment) {
            newInst->addModifier<CommentData>(*comment);
        }

        if (!firstInst) firstInst = newInst;
        lastInst = newInst;
    }

    // Remove the original instruction
    inst->erase();

    return {firstInst, lastInst};
}

Legalized legalizeVCmpX(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId,
                        const std::map<std::string, int>& archCaps) {
    // Check if this architecture supports CMPX writes to SGPR
    auto it = archCaps.find("CMPXWritesSGPR");
    if (it != archCaps.end() && it->second) {
        // No lowering needed on this architecture
        return {nullptr, nullptr};
    }

    // Get instruction details
    const HwInstDesc* desc = inst->getHwInstDesc();
    assert(desc != nullptr && "Instruction descriptor is not supported on this architecture");
    assert(desc->mnemonic != nullptr && "Missing mnemonic in instruction descriptor");

    std::string mnemonic(desc->mnemonic);
    size_t pos = mnemonic.find("_cmpx_");
    if (pos == std::string::npos) return {nullptr, nullptr};  // Not a v_cmpx instruction

    // Check if destination is EXEC
    const auto& destRegs = inst->getDestRegs();
    if (destRegs.empty()) return {nullptr, nullptr};

    // Get wavefront size
    int wavefrontSize = getWaveFrontSize(archId);

    // Replace v_cmpx with v_cmp
    mnemonic.replace(pos, 6, "_cmp_");

    uint16_t cmpOpcode = getMnemonicToIsaOpcode(mnemonic, archId);
    const HwInstDesc* cmpDesc = getMCIDByIsaOp(cmpOpcode, archId);

    assert(cmpDesc != nullptr && "v_cmp_* is not supported on this architecture");

    StinkyInstruction* cmpInst = irBuilder.create(cmpDesc, inst);

    // Replace EXEC destination with VCC
    StinkyRegister destReg = destRegs[0];
    if (destReg.reg.type == RegType::EXEC || destReg.reg.type == RegType::EXEC_LO) {
        destReg = StinkyRegister::getVCCRegister(wavefrontSize);
    }
    cmpInst->addDestReg(destReg);

    // Copy source registers
    for (const auto& srcReg : inst->getSrcRegs()) {
        cmpInst->addSrcReg(srcReg);
    }

    // Copy modifiers (comment, etc.)
    const CommentData* comment = inst->getModifier<CommentData>();
    if (comment) {
        cmpInst->addModifier<CommentData>(*comment);
    }

    // Create s_mov exec, vcc instruction
    const HwInstDesc* movDesc;
    StinkyRegister execReg;
    if (wavefrontSize == 32) {
        movDesc = getMCIDByUOp(GFX::s_mov_b32, archId);
        execReg = StinkyRegister(RegType::EXEC_LO, 0, 1);
    } else {
        movDesc = getMCIDByUOp(GFX::s_mov_b64, archId);
        execReg = StinkyRegister(RegType::EXEC, 0, 1);
    }

    assert(movDesc != nullptr && "s_mov_b32 or s_mov_b64 is not supported on this architecture");

    StinkyInstruction* movInst = irBuilder.create(movDesc, inst);
    movInst->addDestReg(execReg);
    movInst->addSrcReg(destReg);
    if (comment) {
        movInst->addModifier<CommentData>(*comment);
    }

    // Remove the original v_cmpx instruction
    inst->erase();

    return {cmpInst, movInst};
}

Legalized legalizeWaitCnt(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId) {
    // Only legalize on gfx1250
    // TODO: Support other SeparateVMcnt + SeparateLGKMcnt archs (e.g. gfx1200,
    // gfx1201). Need to verify whether they also support combined instructions
    // (s_wait_loadcnt_dscnt, s_wait_storecnt_dscnt).
    if (archId != GfxArchID::Gfx1250) return {nullptr, nullptr};

    // Get wait count data from modifiers
    const SWaitCntData* waitData = inst->getModifier<SWaitCntData>();
    if (!waitData) return {nullptr, nullptr};

    // Get comment if present
    const CommentData* commentMod = inst->getModifier<CommentData>();
    std::string comment = commentMod ? commentMod->comment : "";

    // Check which counters are present
    bool hasVlcnt = (waitData->vlcnt != -1);
    bool hasVscnt = (waitData->vscnt != -1);
    bool hasDlcnt = (waitData->dlcnt != -1);
    bool hasDscnt = (waitData->dscnt != -1);
    bool hasKmcnt = (waitData->kmcnt != -1);

    // Combine dlcnt and dscnt into a single dscnt for gfx1250
    int8_t combinedDscnt = -1;
    if (hasDlcnt && hasDscnt) {
        combinedDscnt = std::min(waitData->dlcnt, waitData->dscnt);
    } else if (hasDlcnt) {
        combinedDscnt = waitData->dlcnt;
    } else if (hasDscnt) {
        combinedDscnt = waitData->dscnt;
    }

    bool hasCombinedDs = (combinedDscnt != -1);

    StinkyInstruction* firstInst = nullptr;
    StinkyInstruction* lastInst = nullptr;

    // Helper to create single wait instruction before inst
    auto createSingleWait = [&](GFX opcode, int8_t count) -> StinkyInstruction* {
        const HwInstDesc* desc = getMCIDByUOp(opcode, archId);
        StinkyInstruction* waitInst = irBuilder.create(desc, inst);
        waitInst->addSrcReg(StinkyRegister(static_cast<int>(count)));
        if (!comment.empty()) {
            waitInst->addModifier<CommentData>(CommentData{comment});
        }
        return waitInst;
    };

    // Helper to create combined wait instruction before inst
    auto createCombinedWait = [&](GFX opcode, int8_t count1, int8_t count2) -> StinkyInstruction* {
        const HwInstDesc* desc = getMCIDByUOp(opcode, archId);
        StinkyInstruction* waitInst = irBuilder.create(desc, inst);
        uint16_t combinedCount = ((count1 & 0xFF) << 8) | (count2 & 0xFF);
        waitInst->addSrcReg(StinkyRegister(static_cast<int>(combinedCount)));
        if (!comment.empty()) {
            waitInst->addModifier<CommentData>(CommentData{comment});
        }
        return waitInst;
    };

    // Case 1: Both VMEM load and DS operations
    if (hasVlcnt && hasCombinedDs) {
        lastInst = createCombinedWait(GFX::s_wait_loadcnt_dscnt, waitData->vlcnt, combinedDscnt);
        if (!firstInst) firstInst = lastInst;
        hasVlcnt = false;
        hasCombinedDs = false;
    }
    // Case 2: Both VMEM store and DS operations
    else if (hasVscnt && hasCombinedDs) {
        lastInst = createCombinedWait(GFX::s_wait_storecnt_dscnt, waitData->vscnt, combinedDscnt);
        if (!firstInst) firstInst = lastInst;
        hasVscnt = false;
        hasCombinedDs = false;
    }

    // Case 3: Separate wait instructions for remaining counters
    if (hasVlcnt) {
        lastInst = createSingleWait(GFX::s_wait_loadcnt, waitData->vlcnt);
        if (!firstInst) firstInst = lastInst;
    }
    if (hasVscnt) {
        lastInst = createSingleWait(GFX::s_wait_storecnt, waitData->vscnt);
        if (!firstInst) firstInst = lastInst;
    }
    if (hasCombinedDs) {
        lastInst = createSingleWait(GFX::s_wait_dscnt, combinedDscnt);
        if (!firstInst) firstInst = lastInst;
    }
    if (hasKmcnt) {
        lastInst = createSingleWait(GFX::s_wait_kmcnt, waitData->kmcnt);
        if (!firstInst) firstInst = lastInst;
    }

    lastInst->addModifier<SWaitCntData>(*waitData);

    // Remove the original s_waitcnt instruction
    if (firstInst) inst->erase();

    return {firstInst, lastInst};
}

Legalized legalizeBarrier(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId) {
    // Only legalize on gfx1250
    if (archId != GfxArchID::Gfx1250) return {nullptr, nullptr};

    // Get modifiers to transfer
    const CommentData* commentMod = inst->getModifier<CommentData>();
    std::string comment = commentMod ? commentMod->comment : "";
    const MemTokenData* memTokenMod = inst->getModifier<MemTokenData>();

    // Create s_barrier_signal -1 (signal global barrier)
    const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
    StinkyInstruction* signalInst = irBuilder.create(signalDesc, inst);
    signalInst->addSrcReg(StinkyRegister(-1));  // -1 = global barrier

    // Create s_barrier_wait -1 (wait on global barrier)
    const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
    StinkyInstruction* waitInst = irBuilder.create(waitDesc, inst);
    waitInst->addSrcReg(StinkyRegister(-1));  // -1 = global barrier
    if (!comment.empty()) {
        waitInst->addModifier<CommentData>(CommentData{comment});
    }
    if (memTokenMod) {
        signalInst->addModifier<MemTokenData>(MemTokenData{memTokenMod->tokens});
        waitInst->addModifier<MemTokenData>(MemTokenData{memTokenMod->tokens});
    }

    // Remove the original s_barrier instruction
    inst->erase();

    return {signalInst, waitInst};
}

/// Helper: compute MSB offset for VGPR when hasVgprMsb is true.
/// When regIdx crosses 256 boundary (e.g., 256, 512), the offset differs from the base.
static int16_t getVgprMsbOffsetForIdx(RegType type, uint32_t regIdx, bool hasVgprMsb) {
    if (!hasVgprMsb || type != RegType::V) return 0;
    return static_cast<int16_t>((regIdx / 256) * (-256));
}

/// Helper function to adjust symbolic register name for split instructions
std::string adjustSymbolicRegName(const std::string& symbolicName, int offsetAdjust = 0) {
    if (symbolicName.empty()) return "";

    // Pattern: ${name}+${digitBase}
    static const std::regex pattern(R"(^(.+)\+(\d+)$)");
    std::smatch match;

    if (!std::regex_match(symbolicName, match, pattern)) return "";  // Format doesn't match

    // Extract captured groups
    std::string baseName = match[1].str();      // Group 1: name
    int digitBase = std::stoi(match[2].str());  // Group 2: digitBase

    // Calculate new base offset
    int newDigitBase = digitBase + offsetAdjust;

    // Build adjusted name: ${baseName}+${newDigitBase}
    return baseName + "+" + std::to_string(newDigitBase);
}

Legalized legalizeDSLoadB192(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId,
                             bool hasVgprMsb) {
    // DSLoadB192: ds_load_b192 v[a:a+5], v[b] offset:X
    // →
    // ds_load_b128 v[a:a+3], v[b] offset:X
    // ds_load_b64  v[a+4:a+5], v[b] offset:X+16

    // Get the original destination and source registers
    assert(inst->getNumDestRegs() == 1 && inst->getNumSrcRegs() == 1 &&
           "invalid ds_load_b192 format.");

    StinkyRegister origDst = inst->getDestReg(0);  // Loaded data destination
    StinkyRegister origSrc = inst->getSrcReg(0);   // Address source

    // Get the original DS modifiers (offset, etc.)
    const DSModifiers* origDSMod = inst->getModifier<DSModifiers>();
    int baseOffset = origDSMod ? origDSMod->offset : 0;

    // Get comment if present
    const CommentData* commentMod = inst->getModifier<CommentData>();
    std::string comment = commentMod ? commentMod->comment : "";

    // Create ds_load_b128 v[a:a+3], v[b] offset:X
    const HwInstDesc* desc1 = getMCIDByUOp(GFX::ds_load_b128, archId);
    StinkyInstruction* load1 = irBuilder.create(desc1, inst);

    StinkyRegister dstData1{origDst.reg.type, origDst.reg.idx,
                            4,  // 4 registers
                            origDst.reg.offset};

    if (!origDst.getSymbolicName().empty()) {
        dstData1.setSymbolicName(origDst.getSymbolicName());
    }
    load1->addDestReg(dstData1);
    load1->addSrcReg(origSrc);

    DSModifiers dsModifiers1;
    if (origDSMod) {
        dsModifiers1 = *origDSMod;
    }
    dsModifiers1.offset = baseOffset;
    load1->addModifier<DSModifiers>(dsModifiers1);

    if (!comment.empty()) {
        load1->addModifier<CommentData>(CommentData{comment});
    }

    // Create ds_load_b64 v[a+4:a+5], v[b] offset:X+16
    const HwInstDesc* desc2 = getMCIDByUOp(GFX::ds_load_b64, archId);
    StinkyInstruction* load2 = irBuilder.create(desc2, inst);

    assert(origDst.reg.idx + 4 < std::numeric_limits<uint16_t>::max() &&
           "Invalid destination register index");

    uint32_t dstData2Idx = static_cast<uint16_t>(origDst.reg.idx + 4);
    int16_t dstData2Offs = (hasVgprMsb && origDst.reg.type == RegType::V)
                               ? getVgprMsbOffsetForIdx(origDst.reg.type, dstData2Idx, hasVgprMsb)
                               : origDst.reg.offset;

    StinkyRegister dstData2{origDst.reg.type, dstData2Idx, 2, dstData2Offs};

    if (!origDst.getSymbolicName().empty()) {
        std::string adjustedName = adjustSymbolicRegName(origDst.getSymbolicName(), 4);
        if (!adjustedName.empty()) {
            dstData2.setSymbolicName(adjustedName);
        }
    }
    load2->addDestReg(dstData2);
    load2->addSrcReg(origSrc);

    DSModifiers dsModifiers2;
    if (origDSMod) {
        dsModifiers2 = *origDSMod;
    }
    dsModifiers2.offset = baseOffset + 16;
    load2->addModifier<DSModifiers>(dsModifiers2);

    if (!comment.empty()) {
        load2->addModifier<CommentData>(CommentData{comment});
    }

    // Remove the original ds_load_b192 instruction
    inst->erase();

    return {load1, load2};
}

Legalized legalizeDSStoreB192(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId,
                              bool hasVgprMsb) {
    // DSStoreB192: ds_store_b192 v[addr], v[b:b+5] offset:X
    // →
    // ds_store_b128 v[addr], v[b:b+3] offset:X
    // ds_store_b64  v[addr], v[b+4:b+5] offset:X+16

    // Get the original address and source data registers
    assert(inst->getNumDestRegs() == 0 && inst->getNumSrcRegs() == 2 &&
           "invalid ds_store_b192 format.");

    StinkyRegister origDstAddr = inst->getSrcReg(0);  // Address (first source)
    StinkyRegister origSrcData = inst->getSrcReg(1);  // Data registers (second source)

    // Get the original DS modifiers (offset, etc.)
    const DSModifiers* origDSMod = inst->getModifier<DSModifiers>();
    int baseOffset = origDSMod ? origDSMod->offset : 0;

    // Get comment if present
    const CommentData* commentMod = inst->getModifier<CommentData>();
    std::string comment = commentMod ? commentMod->comment : "";

    // Create ds_store_b128 v[a], v[b:b+3] offset:X
    const HwInstDesc* desc1 = getMCIDByUOp(GFX::ds_store_b128, archId);
    StinkyInstruction* store1 = irBuilder.create(desc1, inst);

    store1->addSrcReg(origDstAddr);

    StinkyRegister srcData1{origSrcData.reg.type, origSrcData.reg.idx,
                            4,  // 4 registers
                            origSrcData.reg.offset};

    if (!origSrcData.getSymbolicName().empty()) {
        srcData1.setSymbolicName(origSrcData.getSymbolicName());
    }
    store1->addSrcReg(srcData1);

    DSModifiers dsModifiers1;
    if (origDSMod) {
        dsModifiers1 = *origDSMod;
    }
    dsModifiers1.offset = baseOffset;
    store1->addModifier<DSModifiers>(dsModifiers1);

    if (!comment.empty()) {
        store1->addModifier<CommentData>(CommentData{comment});
    }

    // Create ds_store_b64 v[a], v[b+4:b+5] offset:X+16
    const HwInstDesc* desc2 = getMCIDByUOp(GFX::ds_store_b64, archId);
    StinkyInstruction* store2 = irBuilder.create(desc2, inst);

    store2->addSrcReg(origDstAddr);

    assert(origSrcData.reg.idx + 4 < std::numeric_limits<uint16_t>::max() &&
           "Invalid source register index");

    uint32_t srcData2Idx = static_cast<uint16_t>(origSrcData.reg.idx + 4);
    int16_t srcData2Offs =
        (hasVgprMsb && origSrcData.reg.type == RegType::V)
            ? getVgprMsbOffsetForIdx(origSrcData.reg.type, srcData2Idx, hasVgprMsb)
            : origSrcData.reg.offset;

    StinkyRegister srcData2{origSrcData.reg.type, srcData2Idx, 2, srcData2Offs};

    if (!origSrcData.getSymbolicName().empty()) {
        std::string adjustedName = adjustSymbolicRegName(origSrcData.getSymbolicName(), 4);
        if (!adjustedName.empty()) {
            srcData2.setSymbolicName(adjustedName);
        }
    }
    store2->addSrcReg(srcData2);

    DSModifiers dsModifiers2;
    if (origDSMod) {
        dsModifiers2 = *origDSMod;
    }
    dsModifiers2.offset = baseOffset + 16;
    store2->addModifier<DSModifiers>(dsModifiers2);

    if (!comment.empty()) {
        store2->addModifier<CommentData>(CommentData{comment});
    }

    // Remove the original ds_store_b192 instruction
    inst->erase();

    return {store1, store2};
}

Legalized legalizeDSStoreB256(StinkyInstruction* inst, AsmIRBuilder& irBuilder, GfxArchID archId,
                              bool hasVgprMsb) {
    // DSStoreB256: ds_store_b256 v[addr], v[b:b+7] offset:X
    // →
    // ds_write_b128/ds_store_b128 v[addr], v[b:b+3] offset:X
    // ds_write_b128/ds_store_b128 v[addr], v[b+4:b+7] offset:X+16

    // Get the original address and source data registers
    assert(inst->getNumDestRegs() == 0 && inst->getNumSrcRegs() == 2 &&
           "invalid ds_store_b256 format.");

    StinkyRegister origDstAddr = inst->getSrcReg(0);  // Address (first source)
    StinkyRegister origSrcData = inst->getSrcReg(1);  // Data registers (second source)

    // Get the original DS modifiers (offset, etc.)
    const DSModifiers* origDSMod = inst->getModifier<DSModifiers>();
    int baseOffset = origDSMod ? origDSMod->offset : 0;

    // Get comment if present
    const CommentData* commentMod = inst->getModifier<CommentData>();
    std::string comment = commentMod ? commentMod->comment : "";

    const HwInstDesc* dsB128Desc = getMCIDByUOp(GFX::ds_store_b128, archId);

    // Create ds_write_b128/ds_store_b128 v[a], v[b:b+3] offset:X
    StinkyInstruction* store1 = irBuilder.create(dsB128Desc, inst);

    store1->addSrcReg(origDstAddr);

    StinkyRegister srcData1{origSrcData.reg.type, origSrcData.reg.idx,
                            4,  // 4 registers
                            origSrcData.reg.offset};

    if (!origSrcData.getSymbolicName().empty()) {
        srcData1.setSymbolicName(origSrcData.getSymbolicName());
    }
    store1->addSrcReg(srcData1);

    DSModifiers dsModifiers1;
    if (origDSMod) {
        dsModifiers1 = *origDSMod;
    }
    dsModifiers1.offset = baseOffset;
    store1->addModifier<DSModifiers>(dsModifiers1);

    if (!comment.empty()) {
        store1->addModifier<CommentData>(CommentData{comment});
    }

    // Create ds_write_b128/ds_store_b128 v[a], v[b+4:b+7] offset:X+16
    StinkyInstruction* store2 = irBuilder.create(dsB128Desc, inst);

    store2->addSrcReg(origDstAddr);

    assert(origSrcData.reg.idx + 4 < std::numeric_limits<uint16_t>::max() &&
           "Invalid source register index");

    uint32_t srcData2Idx = static_cast<uint16_t>(origSrcData.reg.idx + 4);
    int16_t srcData2Offs =
        (hasVgprMsb && origSrcData.reg.type == RegType::V)
            ? getVgprMsbOffsetForIdx(origSrcData.reg.type, srcData2Idx, hasVgprMsb)
            : origSrcData.reg.offset;

    StinkyRegister srcData2{origSrcData.reg.type, srcData2Idx, 4, srcData2Offs};

    if (!origSrcData.getSymbolicName().empty()) {
        std::string adjustedName = adjustSymbolicRegName(origSrcData.getSymbolicName(), 4);
        if (!adjustedName.empty()) {
            srcData2.setSymbolicName(adjustedName);
        }
    }
    store2->addSrcReg(srcData2);

    DSModifiers dsModifiers2;
    if (origDSMod) {
        dsModifiers2 = *origDSMod;
    }
    dsModifiers2.offset = baseOffset + 16;
    store2->addModifier<DSModifiers>(dsModifiers2);

    if (!comment.empty()) {
        store2->addModifier<CommentData>(CommentData{comment});
    }

    // Remove the original ds_store_b256 instruction
    inst->erase();

    return {store1, store2};
}

}  // namespace stinkytofu
