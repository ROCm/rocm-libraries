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

#include <cstdlib>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace HwReg {

// Resolve a hwreg id string: try symbolic HW_REG_* first, fall back to
// numeric (decimal or 0x). Returns std::nullopt on failure.
inline std::optional<uint16_t> parseId(GfxArchID arch, std::string_view idStr) {
    Id sym{};
    if (nameToId(arch, idStr, sym)) return static_cast<uint16_t>(sym);
    std::string s(idStr);
    if (s.empty()) return std::nullopt;
    char* end = nullptr;
    unsigned long v = std::strtoul(s.c_str(), &end, 0);
    if (end != s.c_str() + s.size() || v > 0xFFFFu) return std::nullopt;
    return static_cast<uint16_t>(v);
}

// Print a HwReg-variant operand as `hwreg(id[,offset[,size]])`. Defaults
// (offset=0, size=32) are suppressed.
inline void printOperand(std::ostream& os, const StinkyRegister& reg) {
    os << "hwreg(" << reg.hwreg.id;
    if (reg.hwreg.offset != 0 || reg.hwreg.size != 32) {
        os << "," << reg.hwreg.offset;
        if (reg.hwreg.size != 32) os << "," << reg.hwreg.size;
    }
    os << ")";
}

// Match an s_setreg whose hwreg operand targets (id, sub.offset, sub.size).
inline bool isSetregTo(const StinkyInstruction& inst, uint16_t id, SubField sub) {
    if (id == 0 || sub.size == 0) return false;
    const uint16_t op = inst.getUnifiedOpcode();
    if (op != GFX::s_setreg_IMM32_b32 && op != GFX::s_setreg_b32) return false;
    for (const auto& r : inst.getDestRegs()) {
        if (r.dataType != StinkyRegister::Type::HwReg) continue;
        return r.hwreg.id == id && r.hwreg.offset == sub.offset && r.hwreg.size == sub.size;
    }
    return false;
}

}  // namespace HwReg
}  // namespace stinkytofu
