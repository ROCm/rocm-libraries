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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <cstdint>
#include <ostream>

#include <iostream> // TODO: don't use iostream.

#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu
{
    //----------------------------------------------------------------------
    // StinkyRegister implementation
    //----------------------------------------------------------------------
    void StinkyRegister::dump(std::ostream& out, const std::string& prefix) const
    {
        out << prefix;
        RegisterPrinter printer(out);
        printer.print(*this);
    }

    //----------------------------------------------------------------------
    // StinkyInstruction implementation
    //----------------------------------------------------------------------
    void StinkyInstruction::dump(bool printDetails, const std::string& prefix) const
    {
        dump(std::cerr, printDetails, prefix);
    }

    void StinkyInstruction::dump(std::ostream&      out,
                                 bool               printDetails,
                                 const std::string& prefix) const
    {
        AsmPrinter printer(out);
        if(!prefix.empty())
            out << prefix;
        printer.print(*this);
    }

    //----------------------------------------------------------------------
    // StinkyInstIRBuilder implementation
    //----------------------------------------------------------------------
    IRBuilder::ID StinkyInstIRBuilder::ID = &StinkyInstIRBuilder::ID;

    StinkyInstruction* StinkyInstIRBuilder::createStinkyLabel(IRList::iterator   pos,
                                                              const std::string& label)
    {
        static const HwInstDesc labelMCID{
            GFX::LABEL, GFX::LABEL, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

        StinkyInstruction* labelInst = createStinkyInstBefore(pos, &labelMCID);
        labelInst->addModifier<LabelData>(LabelData{Modifier::Type::LABEL_NAME, label});
        return labelInst;
    }

    void StinkyInstIRBuilder::erase(StinkyInstruction* stinkyInst)
    {
        irlist->remove(stinkyInst);
        delete stinkyInst;
    }

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/gfx942Isa.inc"

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/gfx950Isa.inc"

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/gfx1250Isa.inc"

    static const HwInstDesc* getMCIDTable(GfxArchID arch)
    {
        switch(arch)
        {
        case GfxArchID::gfx942:
        {
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/gfx942Isa.inc"
            return MCIDTable;
        }
        case GfxArchID::gfx950:
        {
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/gfx950Isa.inc"
            return MCIDTable;
        }
        case GfxArchID::gfx1250:
        {
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/gfx1250Isa.inc"
            return MCIDTable;
        }
        default:
            assert(false && "Internal error: Unsupported architecture");
            return nullptr;
        }
    }

    const HwInstDesc* getMCIDByUOp(GFX unifiedOpcode, GfxArchID arch)
    {
        IsaOpcode isaOpcode = GFX::INVALID;

        switch(arch)
        {
        case GfxArchID::gfx942:
        {
            isaOpcode = getgfx942Opcode(unifiedOpcode);
            break;
        }
        case GfxArchID::gfx950:
        {
            isaOpcode = getgfx950Opcode(unifiedOpcode);
            break;
        }
        case GfxArchID::gfx1250:
        {
            isaOpcode = getgfx1250Opcode(unifiedOpcode);
            break;
        }
        default:
            assert(false && "Internal error: Unsupported architecture");
            return nullptr;
        }

        if(isaOpcode == GFX::INVALID)
        {
            assert(false && "Internal error: Unsupported unified opcode");
            return nullptr;
        }
        return &getMCIDTable(arch)[isaOpcode];
    }

    const HwInstDesc* getMCIDByIsaOp(IsaOpcode isaOpcode, GfxArchID arch)
    {
        return &getMCIDTable(arch)[isaOpcode];
    }

    uint16_t getMnemonicToIsaOpcode(const std::string& mnemonic, GfxArchID arch)
    {
        auto get = [&](const std::unordered_map<std::string, uint16_t>& map,
                       const std::string&                               mnemonic) -> uint16_t {
            auto it = map.find(mnemonic);
#ifndef NDEBUG
            if(it == map.end())
            {
                std::cerr << "Error: No ISA opcode found for mnemonic " << mnemonic
                          << " in arch gfx" << std::to_string((int)arch) << "\n";
                return GFX::INVALID;
            }
#endif
            return it->second;
        };

        switch(arch)
        {
        case GfxArchID::gfx942:
        {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/gfx942Isa.inc"
            return get(MnemonicToIsaOpcodeMap, mnemonic);
        }
        case GfxArchID::gfx950:
        {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/gfx950Isa.inc"
            return get(MnemonicToIsaOpcodeMap, mnemonic);
        }
        case GfxArchID::gfx1250:
        {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/gfx1250Isa.inc"
            return get(MnemonicToIsaOpcodeMap, mnemonic);
        }
        default:
            assert(false && "Internal error: Unsupported architecture");
            return GFX::INVALID;
        }
    }

    //----------------------------------------------------------------------
    // StinkyInstruction utilities
    //----------------------------------------------------------------------

    uint32_t getBytesPerGlobalLoad(const StinkyInstruction& inst)
    {
        auto op = inst.getUnifiedOpcode();

        // FIXME: Should we add metadata for the buffer load inst in tablegen to
        //        query the bytes per load?
        switch(op)
        {
        case GFX::buffer_load_ubyte:
            return 1;
        case GFX::buffer_load_ubyte_d16:
        case GFX::buffer_load_ubyte_d16_hi:
        case GFX::buffer_load_ushort:
        case GFX::buffer_load_short_d16_hi:
            return 2;
        case GFX::buffer_load_dword:
            return 4;
        case GFX::buffer_load_dwordx2:
            return 8;
        case GFX::buffer_load_dwordx4:
            return 16;

        default:
            assert("Calling getBytesPerGlobalLoad but input is not GlobalRead");
            return 0;
        }
    }
} // namespace stinkytofu
