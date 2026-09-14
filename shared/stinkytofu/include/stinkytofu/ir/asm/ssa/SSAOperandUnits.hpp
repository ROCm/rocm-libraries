/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

// How a physical operand maps onto attached SSA values.
//
// Lifting, SSA destruction, and the ssaForm printer all walk srcRegs/destRegs
// alongside AttachedSSA, so they must agree on how many value slots each
// operand consumes. Disagreeing by one silently shifts every later operand onto
// the wrong value, so the rule lives here once instead of in each of them.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "stinkytofu/ir/asm/StinkyRegister.hpp"

namespace stinkytofu {

/// Classes whose registers the lifter can turn into SSA values at all, in a
/// stable order.
///
/// A capability, not a choice: a class outside this list cannot be lifted
/// soundly, so lifting rejects it rather than skipping it. Which of them a given
/// caller asked for is a RegClassSet. One list, so nothing that enumerates
/// classes can drift from what lifting actually supports.
///
/// VCC and EXEC are their own register types rather than SGPR indices, so
/// widening to SGPRs cannot make a scalar operand alias a special register.
/// Accumulators stay out until their VGPR aliasing rules are modelled: on some
/// architectures an AGPR and a VGPR name the same storage, and two SSA values
/// over one physical register would be unsound.
inline constexpr std::array<RegType, 2> kLiftableRegClasses{RegType::V, RegType::S};

inline bool isLiftableRegClass(RegType type) {
    for (RegType liftable : kLiftableRegClasses) {
        if (type == liftable) return true;
    }
    return false;
}

/// A set of register classes, always a subset of the ones lifting can model.
///
/// Two scopes are expressed with this. A lift records which classes it turned
/// into SSA values, and an allocation records which of those it may move; a class
/// outside either keeps whatever the producer chose.
///
/// Capability and scope are separate questions, and conflating them is a bug:
/// a class the lifter cannot model is an error, while a class deliberately left
/// physical is simply not lifted. Its operands keep their immediate payload and
/// no pass rewrites them, exactly as literals and special registers behave.
class RegClassSet {
   public:
    /// Every class the lifter can model. The default for a lift, so a caller that
    /// does not care sees the behaviour that predates scoping.
    static RegClassSet all() {
        RegClassSet classes;
        for (RegType type : kLiftableRegClasses) classes.add(type);
        return classes;
    }

    /// A default-constructed value is the empty set, which lifting rejects.
    static RegClassSet only(RegType type) {
        RegClassSet classes;
        classes.add(type);
        return classes;
    }

    /// Adds \p type when lifting can model it and ignores it otherwise, so the
    /// set is always a subset of the capability and no other method has to
    /// re-check.
    RegClassSet& add(RegType type) {
        if (isLiftableRegClass(type)) mask_ |= bitFor(type);
        return *this;
    }

    bool contains(RegType type) const {
        return (mask_ & bitFor(type)) != 0u;
    }

    bool empty() const {
        return mask_ == 0u;
    }

    /// True when every class in this set is also in \p other.
    bool isSubsetOf(const RegClassSet& other) const {
        return (mask_ & ~other.mask_) == 0u;
    }

    bool operator==(const RegClassSet& other) const = default;

    /// Comma-separated class names, or "none". For stamps and diagnostics.
    std::string toString() const {
        std::string text;
        for (RegType type : kLiftableRegClasses) {
            if (!contains(type)) continue;
            if (!text.empty()) text += ",";
            text += regTypeToString(type);
        }
        return text.empty() ? "none" : text;
    }

   private:
    static uint32_t bitFor(RegType type) {
        return 1u << static_cast<unsigned>(type);
    }

    uint32_t mask_ = 0;
};

/// Number of SSA value slots \p reg contributes to attached SSA, one per DWORD.
///
/// Zero means the operand is not lifted, and carries an immediate payload
/// instead: a literal, a hwreg, a special or pseudo register, an unresolved
/// template virtual register, a class lifting cannot model, or a class this lift
/// left physical.
///
/// \p classes is required rather than defaulted so a walker cannot keep the old
/// answer while the others move on. Read it from SSAArena::liftedClasses().
inline size_t liftedSSAUnits(const StinkyRegister& reg, const RegClassSet& classes) {
    if (!reg.isRegister() || reg.isVirtualReg()) return 0;
    if (isPseudoReg(reg)) return 0;
    if (!isAllocatableReg(reg.reg.type)) return 0;
    if (!classes.contains(reg.reg.type)) return 0;
    return reg.reg.num;
}

}  // namespace stinkytofu
