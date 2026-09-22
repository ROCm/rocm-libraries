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

#include <cstdint>
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

inline std::string_view trimHwregToken(std::string_view s) {
    const auto begin = s.find_first_not_of(" \t");
    if (begin == std::string_view::npos) return {};
    const auto end = s.find_last_not_of(" \t");
    return s.substr(begin, end - begin + 1);
}

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

// Parse `hwreg(id[,offset[,size]])` into a structured HwReg operand.
// `id` may be a symbolic HW_REG_* name (looked up in the per-arch DEF_HWREG
// table) or a numeric literal. Whitespace around tokens is ignored.
// Returns std::nullopt if the text is not an hwreg() operand or the id is
// unknown — callers should leave the original LiteralString in that case.
inline std::optional<StinkyRegister> tryParseOperand(GfxArchID arch, std::string_view text) {
    text = trimHwregToken(text);
    constexpr std::string_view kPrefix = "hwreg(";
    if (text.size() < kPrefix.size() + 1 || text.substr(0, kPrefix.size()) != kPrefix ||
        text.back() != ')') {
        return std::nullopt;
    }
    const std::string_view inner = text.substr(kPrefix.size(), text.size() - kPrefix.size() - 1);

    std::string_view parts[3];
    int n = 0;
    size_t start = 0;
    for (size_t i = 0; i <= inner.size(); ++i) {
        if (i != inner.size() && inner[i] != ',') continue;
        if (n >= 3) return std::nullopt;
        parts[n++] = trimHwregToken(inner.substr(start, i - start));
        start = i + 1;
    }
    if (n == 0 || parts[0].empty()) return std::nullopt;

    const auto id = parseId(arch, parts[0]);
    if (!id) return std::nullopt;

    auto parseU16 = [](std::string_view s, uint16_t& out) -> bool {
        if (s.empty()) return false;
        std::string tmp(s);
        char* end = nullptr;
        unsigned long v = std::strtoul(tmp.c_str(), &end, 0);
        if (end != tmp.c_str() + tmp.size() || v > 0xFFFFu) return false;
        out = static_cast<uint16_t>(v);
        return true;
    };

    uint16_t offset = 0;
    uint16_t size = 32;
    if (n >= 2 && !parseU16(parts[1], offset)) return std::nullopt;
    if (n >= 3 && !parseU16(parts[2], size)) return std::nullopt;
    return StinkyRegister::Hwreg(*id, offset, size);
}

// If `reg` is a LiteralString hwreg(...) that parses on `arch`, return the
// structured HwReg form (so emitAssembly prints a numeric id). Otherwise
// return `reg` unchanged.
inline StinkyRegister canonicalizeOperand(GfxArchID arch, const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::LiteralString) return reg;
    if (auto parsed = tryParseOperand(arch, reg.getLiteralString())) return *parsed;
    return reg;
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
