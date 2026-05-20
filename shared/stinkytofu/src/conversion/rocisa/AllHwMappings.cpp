/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * THE SOFTWARE IS PROVIDED "AS IS") WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "AllHwMappings.hpp"

#include <cassert>
#include <unordered_map>

#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

std::vector<StinkyInstruction*> lowerRocisaMFMA(rocisa::Instruction& inst,
                                                AsmIRBuilder& irBuilder) {
    std::string mnemonic = inst.preStr();
    uint16_t opcode = getMnemonicToIsaOpcode(mnemonic, irBuilder.arch);

    StinkyInstruction* stinkyInst = irBuilder.create(getMCIDByIsaOp(opcode, irBuilder.arch));

    return {stinkyInst};
}

std::vector<StinkyInstruction*> lowerRocisaWaitCnt(rocisa::Instruction& inst,
                                                   AsmIRBuilder& irBuilder) {
    auto asmCaps = rocisa::rocIsa::getInstance().getAsmCaps();
    auto itLgkm = asmCaps.find("MaxLgkmcnt");
    auto itVm = asmCaps.find("MaxVmcnt");
    int maxLgkmcnt = itLgkm != asmCaps.end() ? itLgkm->second : -1;
    int maxVmcnt = itVm != asmCaps.end() ? itVm->second : -1;

    SWaitCntData waitCntData;
    waitCntData.maxLgkmcnt = maxLgkmcnt;
    waitCntData.maxVmcnt = maxVmcnt;
    if (rocisa::_SWaitCnt* waitCntInst = dynamic_cast<rocisa::_SWaitCnt*>(&inst)) {
        // _SWaitCnt carries pre-computed lgkmcnt (= dscnt + kmcnt) since gfx942
        // hardware uses a single lgkmcnt counter. Store in dlcnt here; the opt
        // pass will clear and re-analyze all counts independently.
        // gfx942:  getLgkmcnt() = dscnt + kmcnt, getVmcnt() = vlcnt + vscnt
        // gfx1030: getLgkmcnt() = dscnt + kmcnt, getVmcnt() = vlcnt (vscnt separate)
        waitCntData.dlcnt = waitCntInst->getLgkmcnt();
        waitCntData.vlcnt = waitCntInst->getVmcnt();
    } else if (rocisa::_SWaitLoadcnt* waitCntInst = dynamic_cast<rocisa::_SWaitLoadcnt*>(&inst)) {
        waitCntData.vlcnt = waitCntInst->getLoadcnt();
    } else if (const rocisa::_SWaitDscnt* waitCntInst =
                   dynamic_cast<const rocisa::_SWaitDscnt*>(&inst)) {
        waitCntData.dscnt = waitCntInst->getDscnt();
    } else if (const rocisa::_SWaitKMcnt* waitCntInst =
                   dynamic_cast<const rocisa::_SWaitKMcnt*>(&inst)) {
        waitCntData.kmcnt = waitCntInst->getKmcnt();
    } else if (const rocisa::_SWaitCntVscnt* waitCntInst =
                   dynamic_cast<const rocisa::_SWaitCntVscnt*>(&inst)) {
        waitCntData.vscnt = waitCntInst->getVscnt();
    } else {
        assert(false && "Internal error: WaitCntInstruction expected");
    }

    StinkyInstruction* stinkyInst = irBuilder.create(getMCIDByUOp(GFX::s_waitcnt, irBuilder.arch));

    stinkyInst->addModifier<SWaitCntData>(waitCntData);

    return {stinkyInst};
}

std::vector<StinkyInstruction*> lowerRocisaWaitTensorcnt(rocisa::Instruction& inst,
                                                         AsmIRBuilder& irBuilder) {
    SWaitTensorCntData waitTensorCntData;
    if (rocisa::SWaitTensorcnt* waitTensorCntInst = dynamic_cast<rocisa::SWaitTensorcnt*>(&inst)) {
        if (const int* tensorcnt = std::get_if<int>(&waitTensorCntInst->getSrcParams().front())) {
            waitTensorCntData.tlcnt = *tensorcnt;
        }
    } else {
        assert(false && "Internal error: WaitTensorCntInstruction expected");
    }

    StinkyInstruction* stinkyInst =
        irBuilder.create(getMCIDByUOp(GFX::s_wait_tensorcnt, irBuilder.arch));

    stinkyInst->addModifier<SWaitTensorCntData>(waitTensorCntData);

    return {stinkyInst};
}

std::vector<StinkyInstruction*> lowerRocisaStoreWaitCnt(rocisa::Instruction& inst,
                                                        AsmIRBuilder& irBuilder) {
    SWaitStoreCntData waitStoreCntData;
    if (rocisa::_SWaitStorecnt* waitStoreCntInst = dynamic_cast<rocisa::_SWaitStorecnt*>(&inst)) {
        if (const int* storecnt = std::get_if<int>(&waitStoreCntInst->getSrcParams().front())) {
            waitStoreCntData.storecnt = *storecnt;
        }
    } else {
        assert(false && "Internal error: WaitStoreCntInstruction expected");
    }

    StinkyInstruction* stinkyInst =
        irBuilder.create(getMCIDByUOp(GFX::s_wait_storecnt, irBuilder.arch));

    stinkyInst->addModifier<SWaitStoreCntData>(waitStoreCntData);

    return {stinkyInst};
}

// implement lowerRocisaWaitAlu
std::vector<StinkyInstruction*> lowerRocisaWaitAlu(rocisa::Instruction& inst,
                                                   AsmIRBuilder& irBuilder) {
    rocisa::SWaitAlu* waitAluInst = dynamic_cast<rocisa::SWaitAlu*>(&inst);

    assert(waitAluInst != nullptr && "Internal error: WaitAluInstruction expected");

    SWaitAluData waitAluData(waitAluInst->getVaVdst(), waitAluInst->getVaSdst(),
                             waitAluInst->getVaSsrc(), waitAluInst->getHoldCnt(),
                             waitAluInst->getVmVsrc(), waitAluInst->getVaVcc(),
                             waitAluInst->getSaSdst());

    StinkyInstruction* stinkyInst = irBuilder.create(getMCIDByUOp(GFX::s_wait_alu, irBuilder.arch));

    stinkyInst->addModifier<SWaitAluData>(waitAluData);

    return {stinkyInst};
}

std::vector<StinkyInstruction*> lowerRocisaSchedulingFence(rocisa::Instruction& /*inst*/,
                                                           AsmIRBuilder& irBuilder) {
    return {irBuilder.createFence()};
}

};  // anonymous namespace

// Include all Rocisa hardware mapping tables
#include "RocisaArchInfo.hpp"

namespace stinkytofu {
struct HwInstDescConversionMapping {
    const std::unordered_map<std::type_index, ConvertHwInstToRocisaFunc>* convHwInstToRocisaFMap =
        nullptr;
};

static const std::unordered_map<std::type_index, uint16_t>* getRocisaToHwInstMap(GfxArchID arch) {
    return vecRocisaToHwInstMap[static_cast<size_t>(arch)]();
}

static const std::unordered_map<std::type_index, ConvertRocisaToHwInstFunc>*
getConvRocisaToHwInstFMap(GfxArchID arch) {
    return vecRocisaToHwInstLoweringMap[static_cast<size_t>(arch)]();
}

const HwInstDesc* getRocisaToMCID(std::type_index type, GfxArchID arch) {
    const std::unordered_map<std::type_index, uint16_t>* rocisaToHwInstMap =
        getRocisaToHwInstMap(arch);

    auto it = rocisaToHwInstMap->find(type);
    if (it == rocisaToHwInstMap->end()) return nullptr;

    return getMCIDByIsaOp(it->second, arch);
}

ConvertRocisaToHwInstFunc getConvertRocisaToHwInstFunc(std::type_index type, GfxArchID arch) {
    auto convMap = getConvRocisaToHwInstFMap(arch);

    auto it = convMap->find(type);

    if (it == convMap->end()) {
        std::cerr << "Error: No conversion entry for rocisa " << type.name() << " in arch "
                  << getArchName(arch) << "\n";

        return nullptr;
    }

    ConvertRocisaToHwInstFunc convFn = it->second;

    if (convFn == nullptr) {
        std::cerr << "Error (TODO): conversion for rocisa " << type.name() << " in arch "
                  << getArchName(arch) << " is not implemented yet\n";

        return nullptr;
    }

    return convFn;
}

}  // namespace stinkytofu
