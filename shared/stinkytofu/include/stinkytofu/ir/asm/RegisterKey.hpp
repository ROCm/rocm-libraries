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

#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
/// Per-DWORD register key used to track individual register components
/// across definitions and uses.
struct RegKey {
    RegType type;
    unsigned idx;

    bool operator==(const RegKey& o) const noexcept {
        return type == o.type && idx == o.idx;
    }
};

struct RegKeyHash {
    size_t operator()(const RegKey& k) const noexcept {
        size_t h = std::hash<int>{}(static_cast<int>(k.type));
        h ^= std::hash<unsigned>{}(k.idx) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

template <typename V>
using RegKeyMap = std::unordered_map<RegKey, V, RegKeyHash>;

using RegKeySet = std::unordered_set<RegKey, RegKeyHash>;

inline RegKey toRegKey(const StinkyRegister& reg, unsigned offset = 0) {
    return {reg.reg.type, reg.reg.idx + offset};
}

/// Invoke fn(RegKey) for each DWORD in a register operand.
/// Skips non-register operands (literals, immediates).
template <typename Fn>
void forEachRegUnit(const StinkyRegister& reg, Fn&& fn) {
    if (reg.dataType != StinkyRegister::Type::Register) return;
    for (unsigned i = 0; i < reg.reg.num; ++i) {
        fn(toRegKey(reg, i));
    }
}

}  // namespace stinkytofu
