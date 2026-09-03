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

// Which architecture's rules apply, keyed on the ISA triple.
//
// Keyed on the triple rather than on GfxArchID for the two reasons
// adding-architecture.md gives for pipelines: a stepping-only build may not
// have the enumerator at all, and keying on the triple is what gives a new
// stepping its parent's rules with no edit and no duplicate row.
//
// Capabilities are a factory argument, not a key: the same chip can be
// configured two ways, and the triple cannot tell those apart.
//
// To give an architecture rules, add one TU that builds a table and registers
// it, then name its anchor in registerAll(). An unregistered triple yields an
// empty table.

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"

namespace stinkytofu {

class STINKYTOFU_EXPORT AllocationRulesRegistry {
   public:
    using Factory = std::function<AllocationRules(const AsmCapsConfig&)>;

    /// Register the rules for \p arch. A second registration replaces the first.
    static void setArch(const std::array<int, 3>& arch, Factory factory);

    /// Rules for \p arch under \p caps, by value. An unregistered triple yields
    /// an empty table, so there is no null and no special case.
    static AllocationRules forArch(const std::array<int, 3>& arch, const AsmCapsConfig& caps);

    /// True when \p arch has rules registered, even if every one is Off.
    static bool hasArch(const std::array<int, 3>& arch);

    /// Remove the registration for \p arch. Mainly for tests, which register a
    /// table for one triple and clear it afterwards.
    static void clearArch(const std::array<int, 3>& arch);

    /// Registered arch keys, sorted (form `gfxMNS`).
    static std::vector<std::string> archKeys();

    /// Exists so a static build keeps the self-registering per-arch TUs, whose
    /// anchors its body names. Registers nothing itself.
    static void registerAll();

   private:
    AllocationRulesRegistry() = default;

    struct Registry;
    static Registry& registry();
};

}  // namespace stinkytofu
