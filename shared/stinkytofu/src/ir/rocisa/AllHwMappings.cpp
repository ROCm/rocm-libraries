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

#include <cassert>
#include <unordered_map>

//#include "code.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"

#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/rocisa/AllHwMappings.hpp"

namespace stinkytofu
{
    std::vector<StinkyInstruction*>
        lowerRocisaSMFMA(rocisa::Instruction& inst, StinkyInstIRBuilder& irBuilder, IRList& insts)
    {
        rocisa::SMFMAInstruction* smfmaInst = dynamic_cast<rocisa::SMFMAInstruction*>(&inst);
        assert(smfmaInst != nullptr && "Internal error: SMFMAInstruction expected");

        std::string mnemonic = inst.preStr();
        uint16_t    opcode   = getMnemonicToIsaOpcode(mnemonic, irBuilder.arch);

        StinkyInstruction* stinkyInst
            = irBuilder.createStinkyInstBefore(insts.end(), getMCIDByIsaOp(opcode, irBuilder.arch));

        return {stinkyInst};
    }

    std::vector<StinkyInstruction*>
        lowerRocisaMFMA(rocisa::Instruction& inst, StinkyInstIRBuilder& irBuilder, IRList& insts)
    {
        rocisa::MFMAInstruction* mfmaInst = dynamic_cast<rocisa::MFMAInstruction*>(&inst);
        assert(mfmaInst != nullptr && "Internal error: MFMAInstruction expected");

        std::string mnemonic = inst.preStr();
        uint16_t    opcode   = getMnemonicToIsaOpcode(mnemonic, irBuilder.arch);

        StinkyInstruction* stinkyInst
            = irBuilder.createStinkyInstBefore(insts.end(), getMCIDByIsaOp(opcode, irBuilder.arch));

        return {stinkyInst};
    }

    std::vector<StinkyInstruction*>
        lowerRocisaWaitCnt(rocisa::Instruction& inst, StinkyInstIRBuilder& irBuilder, IRList& insts)
    {
        SWaitCntData waitCntData;
        if(rocisa::_SWaitCnt* waitCntInst = dynamic_cast<rocisa::_SWaitCnt*>(&inst))
        {
            waitCntData.dlcnt = waitCntInst->getLgkmcnt();
            waitCntData.vlcnt = waitCntInst->getVmcnt();
        }
        else if(rocisa::_SWaitLoadcnt* waitCntInst = dynamic_cast<rocisa::_SWaitLoadcnt*>(&inst))
        {
            if(const int* dlcnt = std::get_if<int>(&waitCntInst->getParams().front()))
            {
                waitCntData.dlcnt = *dlcnt;
            }
        }
        else if(const rocisa::_SWaitDscnt* waitCntInst
                = dynamic_cast<const rocisa::_SWaitDscnt*>(&inst))
        {
            if(const int* dscnt = std::get_if<int>(&waitCntInst->getParams().front()))
            {
                waitCntData.dscnt = *dscnt;
            }
        }
        else
        {
            assert(false && "Internal error: WaitCntInstruction expected");
        }

        StinkyInstruction* stinkyInst = irBuilder.createStinkyInstBefore(
            insts.end(), getMCIDByUOp(GFX::s_waitcnt, irBuilder.arch));

        stinkyInst->addModifier<SWaitCntData>(waitCntData);

        return {stinkyInst};
    }

    std::vector<StinkyInstruction*> lowerRocisaWaitTensorcnt(rocisa::Instruction& inst,
                                                             StinkyInstIRBuilder& irBuilder,
                                                             IRList&              insts)
    {
        SWaitTensorCntData waitTensorCntData;
        if(rocisa::SWaitTensorcnt* waitTensorCntInst = dynamic_cast<rocisa::SWaitTensorcnt*>(&inst))
        {
            if(const int* tensorcnt = std::get_if<int>(&waitTensorCntInst->getSrcParams().front()))
            {
                waitTensorCntData.tlcnt = *tensorcnt;
            }
        }
        else
        {
            assert(false && "Internal error: WaitTensorCntInstruction expected");
        }

        StinkyInstruction* stinkyInst = irBuilder.createStinkyInstBefore(
            insts.end(), getMCIDByUOp(GFX::s_wait_tensorcnt, irBuilder.arch));

        stinkyInst->addModifier<SWaitTensorCntData>(waitTensorCntData);

        return {stinkyInst};
    }

    struct HwInstDescConversionMapping
    {
        const std::unordered_map<std::type_index, ConvertHwInstToRocisaFunc>* convHwInstToRocisaFMap
            = nullptr;
    };

    const std::unordered_map<std::type_index, uint16_t>* getRocisaToHwInstMap(GfxArchID arch)
    {
        switch(arch)
        {
        case GfxArchID::gfx942:
        {
#define GET_ROCISA_HW_MAPPING_TABLE
#include "ir/rocisa/Rocisagfx942Mappings.inc"
            return &rocisaToHwInstMap;
        }
        case GfxArchID::gfx950:
        {
#define GET_ROCISA_HW_MAPPING_TABLE
#include "ir/rocisa/Rocisagfx950Mappings.inc"
            return &rocisaToHwInstMap;
        }
        case GfxArchID::gfx1250:
        {
#define GET_ROCISA_HW_MAPPING_TABLE
#include "ir/rocisa/Rocisagfx1250Mappings.inc"
            return &rocisaToHwInstMap;
        }
        default:
            assert(false && "Internal error: Unsupported architecture");
            return nullptr;
        }
    }

    const std::unordered_map<std::type_index, ConvertRocisaToHwInstFunc>*
        getConvRocisaToHwInstFMap(GfxArchID arch)
    {
        switch(arch)
        {
        case GfxArchID::gfx942:
        {
#define GET_ROCISA_TO_HW_CONVERSION_TABLE
#include "ir/rocisa/Rocisagfx942Mappings.inc"
            return &convertRocisaToHwInstFunc;
        }
        case GfxArchID::gfx950:
        {
#define GET_ROCISA_TO_HW_CONVERSION_TABLE
#include "ir/rocisa/Rocisagfx950Mappings.inc"
            return &convertRocisaToHwInstFunc;
        }
        case GfxArchID::gfx1250:
        {
#define GET_ROCISA_TO_HW_CONVERSION_TABLE
#include "ir/rocisa/Rocisagfx1250Mappings.inc"
            return &convertRocisaToHwInstFunc;
        }
        default:
            assert(false && "Internal error: Unsupported architecture");
            return nullptr;
        }
    }

    //     const std::unordered_map<std::type_index, ConvertHwInstToRocisaFunc>*
    //         getConvHwInstToRocisaFMap(GfxArchID arch)
    //     {
    //         switch(arch)
    //         {
    //         case GfxArchID::gfx942:
    //         {
    // #define GET_HW_TO_ROCISA_CONVERSION_TABLE
    // #include "ir/rocisa/Rocisagfx942Mappings.inc"
    //             return &convertHwInstToRocisaFunc;
    //         }
    //         case GfxArchID::gfx950:
    //         {
    // #define GET_HW_TO_ROCISA_CONVERSION_TABLE
    // #include "ir/rocisa/Rocisagfx950Mappings.inc"
    //             return &convertHwInstToRocisaFunc;
    //         }
    //         default:
    //             assert(false && "Internal error: Unsupported architecture");
    //             return nullptr;
    //         }
    //     }

    const HwInstDesc* getRocisaToMCID(std::type_index type, GfxArchID arch)
    {
        const std::unordered_map<std::type_index, uint16_t>* rocisaToHwInstMap
            = getRocisaToHwInstMap(arch);

        auto it = rocisaToHwInstMap->find(type);
        if(it == rocisaToHwInstMap->end())
            return nullptr;

        return getMCIDByIsaOp(it->second, arch);
    }

    ConvertRocisaToHwInstFunc getConvertRocisaToHwInstFunc(std::type_index type, GfxArchID arch)
    {
        auto convMap = getConvRocisaToHwInstFMap(arch);

        auto it = convMap->find(type);

        if(it == convMap->end())
        {
            std::cerr << "Error: No conversion entry for rocisa " << type.name() << " in arch gfx"
                      << std::to_string((int)arch) << "\n";

            return nullptr;
        }

        ConvertRocisaToHwInstFunc convFn = it->second;

        if(convFn == nullptr)
        {
            std::cerr << "Error (TODO): conversion for rocisa " << type.name() << " in arch gfx"
                      << std::to_string((int)arch) << " is not implemented yet\n";

            return nullptr;
        }

        return convFn;
    }

    // ConvertHwInstToRocisaFunc getConvertHwInstToRocisaFunc(std::type_index type, GfxArchID arch)
    // {
    //     auto convMap = getConvHwInstToRocisaFMap(arch);
    //     auto it      = convMap->find(type);

    //     if(it == convMap->end())
    //     {
    //         std::cerr << "Error: No conversion entry for hw instruction " << type.name()
    //                   << " in arch gfx" << std::to_string((int)arch) << "\n";

    //         return nullptr;
    //     }

    //     ConvertHwInstToRocisaFunc convFn = it->second;

    //     if(convFn == nullptr)
    //     {
    //         std::cerr << "Error (TODO): conversion for hw instruction " << type.name()
    //                   << " in arch gfx" << std::to_string((int)arch) << " is not implemented yet\n";

    //         return nullptr;
    //     }

    //     return convFn;
    // }
} // namespace stinkytofu
