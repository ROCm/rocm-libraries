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

#pragma once
#include <bitset>
#include <cassert>
#include <cstdint>

namespace stinkytofu
{
    enum class GfxArchID : uint32_t
    {
        gfx942  = 942,
        gfx950  = 950,
        gfx1250 = 1250,
    };

    inline GfxArchID getGfxArchID(uint32_t major, uint32_t minor, uint32_t stepping)
    {
        uint32_t arch = major * 100 + minor * 10 + stepping;
        assert(arch == 942 || arch == 950 || arch == 1250 && "Unsupported GfxArchID");
        return static_cast<GfxArchID>(arch);
    }

    enum InstFlag : uint8_t
    {
#define MACRO(flag) flag,
#include "isa/gfx/Flags.def"

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

        bool has(InstFlag f) const
        {
            return flags.test((size_t)f);
        }
    };

} // namespace stinkytofu
