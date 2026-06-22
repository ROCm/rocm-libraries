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

#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {

inline RegHalf toRegHalf(HighBitSel sel) {
    return static_cast<RegHalf>(static_cast<int8_t>(sel));
}

/// Producer/consumer VGPR key generator with arch-aware half collapsing.
class VGPRHalfKeyer {
   public:
    VGPRHalfKeyer() = default;
    explicit VGPRHalfKeyer(bool d16Writes32BitVgpr) : d16_(d16Writes32BitVgpr) {}

    /// Producer-side key for a single VGPR write.
    ///   half == NONE                 → {V, idx, NONE}  (32-bit write)
    ///   d16Writes32BitVgpr target    → {V, idx, NONE}  (ECC forces 32-bit
    ///                                                   write regardless of half)
    ///   otherwise (per-half archs)   → {V, idx, half}  (independent halves)
    RegKey producerKey(unsigned regIdx, HighBitSel half) const {
        if (half == HighBitSel::NONE || d16_) return {RegType::V, regIdx, RegHalf::NONE};
        return {RegType::V, regIdx, toRegHalf(half)};
    }

    /// Consumer-side keys for a single VGPR read. Always emits NONE first so
    /// 32-bit producers and all D16-collapsed producers match in one lookup. Then:
    ///   half == NONE → also emit LOW and HIGH: a 32-bit consumer depends on any
    ///                  half writer on per-half archs.
    ///   half != NONE → also emit `half` only — on per-half archs the two
    ///                  halves are independent.
    ///
    /// On d16Writes32BitVgpr targets every producer stamps NONE, so the
    /// second/third keys never match — emitting them is harmless.
    template <typename Fn>
    void forEachConsumerKey(unsigned regIdx, HighBitSel half, Fn&& fn) const {
        fn(RegKey{RegType::V, regIdx, RegHalf::NONE});
        if (half == HighBitSel::NONE) {
            fn(RegKey{RegType::V, regIdx, RegHalf::LOW});
            fn(RegKey{RegType::V, regIdx, RegHalf::HIGH});
        } else {
            fn(RegKey{RegType::V, regIdx, toRegHalf(half)});
        }
    }

   private:
    bool d16_ = false;
};

}  // namespace stinkytofu
