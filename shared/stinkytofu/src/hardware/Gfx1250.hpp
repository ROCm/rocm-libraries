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
#pragma once

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"

#include <mutex>

namespace
{

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/Gfx1250Isa.inc"

}

using namespace stinkytofu;

struct Gfx1250ArchInfo : public ArchHelper::ArchInfo
{
    Gfx1250ArchInfo()
        : ArchInfo(12, 5, 0, 32)
    {
    }

    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override
    {
        return getGfx1250Opcode(unifiedOpcode);
    }

    const HwInstDesc* getMCIDTable() const override
    {
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/Gfx1250Isa.inc"
#include "hardware/generated/Gfx1250_operands.inc"

        static std::once_flag once;
        std::call_once(once, [] {
            for(const auto& req : instRequirements)
            {
                for(size_t i = 0; i < sizeof(MCIDTable) / sizeof(MCIDTable[0]); ++i)
                {
                    if(MCIDTable[i].mnemonic && std::string(MCIDTable[i].mnemonic) == req.mnemonic)
                    {
                        const_cast<HwInstDesc&>(MCIDTable[i]).operandWidths = req.requirements;
                        break;
                    }
                }
            }
        });

        return MCIDTable;
    }

    const std::unordered_map<std::string, uint16_t>& getMnemonicToIsaOpcodeMap() const override
    {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/Gfx1250Isa.inc"
        return MnemonicToIsaOpcodeMap;
    }
};
