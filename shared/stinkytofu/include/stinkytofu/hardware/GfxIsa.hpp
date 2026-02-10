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
#include "stinkytofu/support/Span.hpp"
#include <bitset>
#include <cassert>
#include <cstdint>

namespace stinkytofu
{
    // Forward declaration of RegType (defined in ir/asm/StinkyAsmIR.hpp)
    enum class RegType;

    enum class GfxArchID : uint32_t
    {
#define STINKYTOFU_ARCH(archName) archName,
#include "Config/Archs.def"
    };

    enum InstFlag : uint8_t
    {
#define MACRO(flag) flag,
#include "stinkytofu/hardware/Flags.def"

#undef MACRO

        // Total count of flags.
        // Order does matter, define count immediately after the last flag.
        IF_COUNT,

        // The beginning of the flags is always 0.
        IF_BEGIN = 0, // = IF_MUBUFLoad
    };

    // Note: Try to keep the capacity <= 64.
    // If gfx instruction needs more than 64 flags, there are two options:
    // 1. Implement a custom bitset type that can support constexpr
    //    initialization for more than 64 bits.
    //
    // 2. (simpler) Just add another std::bitset<32> flag.
    constexpr size_t flagCapacity = 32;

    // Helper function to convert flags to a bit pattern at compile time
    constexpr uint64_t makeFlagBits(std::initializer_list<InstFlag> flags)
    {
        static_assert(flagCapacity <= 64, "flagCapacity exceeds uint64_t bit width");

        uint64_t bits = 0;
        for(auto f : flags)
            bits |= (1ULL << static_cast<size_t>(f));
        return bits;
    }

    constexpr std::bitset<flagCapacity> makeFlagSet(std::initializer_list<InstFlag> flags)
    {
        return std::bitset<flagCapacity>(makeFlagBits(flags));
    }

    using IsaOpcode     = uint16_t;
    using UnifiedOpcode = uint16_t;
    using InstFlagSet   = std::bitset<flagCapacity>;

    // General hardware instruction description for all ISAs.
    struct HwInstDesc
    {
        // instruction opcode in specific ISA.
        IsaOpcode isaOpcode = 0;

        // unified instruction opcode for all ISAs.
        UnifiedOpcode unifiedOpcode = 0;

        // issue cycles and latency cycles in specific ISA.
        uint16_t issue   = 0;
        uint16_t latency = 0;

        // mnemonic string for the instruction.
        const char* mnemonic = nullptr;

        std::bitset<flagCapacity> flags;

        /// @brief Register width and type requirement for an operand
        ///
        /// Specifies that a particular operand must use:
        /// 1. A specific number of consecutive registers (width)
        /// 2. A specific register type (V, S, A, etc.)
        ///
        /// Example: tensor_load_to_lds requires:
        ///   - src[0]: 4 consecutive SGPRs (e.g., s[0:3])
        ///   - src[1]: 8 consecutive SGPRs (e.g., s[4:11])
        ///
        /// Example: v_add_f32 requires:
        ///   - dest[0]: 1 VGPR (e.g., v0)
        ///   - src[0]: 1 VGPR (e.g., v1)
        ///   - src[1]: 1 VGPR (e.g., v2)
        struct OperandWidth
        {
            uint8_t operandIndex; ///< 0-based operand index
            uint8_t width; ///< Expected register count (e.g., 4 = 4 consecutive regs)
            bool    isDest; ///< true = destination operand, false = source operand
            RegType expectedType; ///< Expected register type (RegType::UNKNOWN = any type allowed)
        };

        /// Register width requirements for this instruction's operands
        /// nullptr if no width requirements (most instructions)
        /// Points to static constexpr array in hardware definition files
        span<const OperandWidth> operandWidths;

        bool has(InstFlag f) const
        {
            return flags.test((size_t)f);
        }
    };

} // namespace stinkytofu
