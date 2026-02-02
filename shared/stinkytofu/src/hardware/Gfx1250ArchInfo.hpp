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

#include "ir/asm/StinkyAsmIR.hpp" // For RegType enum
#include "isa/ArchHelper.hpp"
#include "isa/gfx/GfxIsa.hpp"

namespace
{

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/Gfx1250Isa.inc"

}

using namespace stinkytofu;

struct Gfx1250ArchInfo : public ArchHelper::ArchInfo
{
    Gfx1250ArchInfo()
        : ArchInfo(12, 5, 0, 32 /* waveFrontSize */)
    {
    }

    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override
    {
        return getGfx1250Opcode(unifiedOpcode);
    }

    const HwInstDesc* getMCIDTable() const override
    {
// Implementation to return the MCID table for Gfx1250
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/Gfx1250Isa.inc"

        //=======================================================================
        // INSTRUCTION METADATA FOR GFX1250
        //=======================================================================
        //
        // Operand requirements (register width/type) are defined HERE
        // Instruction costs (cycle/latency) -> hardware/src/gfx/Gfx1250.cpp
        //
        // To modify an instruction:
        //   - Update requirements: Scroll down in THIS file
        //   - Update costs: Open hardware/src/gfx/Gfx1250.cpp (GFX1250_COSTS[])
        //   - Update definition: hardware/src/gfx/Gfx1250.cpp (DEF_T calls)
        //
        //=======================================================================

        // Initialize architecture-specific register requirements
        // This is done once when the table is first accessed
        static bool initialized = false;
        if(!initialized)
        {
            //-------------------------------------------------------------------
            // Define instruction operand requirements (width + type)
            //-------------------------------------------------------------------

            // tensor_load_to_lds: src[0]=4 SGPRs, src[1]=8 SGPRs
            static constexpr HwInstDesc::OperandWidth tensorLoadToLdsReqs[] = {
                {0, 4, false, RegType::S}, // src[0]: 4 SGPRs
                {1, 8, false, RegType::S}, // src[1]: 8 SGPRs
            };

            // Add more instruction requirements here as needed:
            // static constexpr HwInstDesc::OperandWidth vAddF32Reqs[] = {
            //     {0, 1, true,  RegType::V},  // dest: 1 VGPR
            //     {0, 1, false, RegType::V},  // src0: 1 VGPR
            //     {1, 1, false, RegType::V},  // src1: 1 VGPR
            // };

            //-------------------------------------------------------------------
            // Table of instructions with register requirements
            //-------------------------------------------------------------------

            struct InstRequirement
            {
                const char*                          mnemonic;
                span<const HwInstDesc::OperandWidth> requirements;
            };

            static constexpr InstRequirement instRequirements[] = {
                {"tensor_load_to_lds", tensorLoadToLdsReqs},
                // Add more: {"v_add_f32", vAddF32Reqs},
            };

            //-------------------------------------------------------------------
            // Apply requirements to MCIDTable
            //-------------------------------------------------------------------

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

            initialized = true;
        }

        return MCIDTable;
    }

    const std::unordered_map<std::string, uint16_t>& getMnemonicToIsaOpcodeMap() const override
    {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/Gfx1250Isa.inc"
        return MnemonicToIsaOpcodeMap;
    }
};
