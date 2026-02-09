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

#include "ir/asm/StinkyAsmIR.hpp"
#include "isa/ArchHelper.hpp"

#include <mutex>

//===============================================================================
// INSTRUCTION METADATA FOR GFX950
//===============================================================================
//
// This file contains:
//   - Operand REQUIREMENTS (register width/type) - see getMCIDTable() below
//
// Related metadata in other files:
//   - Instruction COSTS (cycle, latency) -> hardware/src/gfx/Gfx950.cpp
//   - Instruction DEFINITIONS (DEF_T calls) -> hardware/src/gfx/Gfx950.cpp
//
// To modify an instruction:
//   - Definitions, costs, operand requirements: edit hardware/defs/Gfx950Instructions.def; rebuild.
//   - Update costs: Open hardware/src/gfx/Gfx950.cpp (GFX950_COSTS[])
//   - Update definition: hardware/src/gfx/Gfx950.cpp (DEF_T calls)
//
//===============================================================================

namespace
{

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/Gfx950Isa.inc"

}

using namespace stinkytofu;

struct Gfx950ArchInfo : public ArchHelper::ArchInfo
{
    Gfx950ArchInfo()
        : ArchInfo(9, 5, 0, 64 /* waveFrontSize */)
    {
    }

    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override
    {
        return getGfx950Opcode(unifiedOpcode);
    }

    const HwInstDesc* getMCIDTable() const override
    {
// Implementation to return the MCID table for Gfx950
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/Gfx950Isa.inc"

#include "hardware/generated/Gfx950_operands.inc"

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
#include "hardware/Gfx950Isa.inc"
        return MnemonicToIsaOpcodeMap;
    }
};
