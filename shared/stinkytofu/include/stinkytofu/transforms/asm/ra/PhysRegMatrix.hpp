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

// Which live range currently occupies each physical register.
//
// This is where interference is answered. Two values conflict when their live
// ranges overlap on the same physical unit, so the question is asked of a
// register rather than of a pair of values, and no interference graph is built.
//
// A utility, deliberately not part of the allocator interface: a linear-scan or
// graph-colouring policy may want a different occupancy structure, and should
// not have to pay for this one.
//
// Only classes the target calls allocatable have storage here. EXEC, VCC, SCC,
// M0, literals, and memory tokens are not SSA values, so they never occupy a
// unit; VCC and EXEC are their own RegType in this IR, so colouring SGPRs cannot
// alias them by index.
//
// Ranges are referenced, not copied: SSALiveIntervals owns them and must outlive
// the matrix.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"

namespace stinkytofu {

class PhysRegMatrix {
   public:
    /// \p target must outlive the matrix.
    explicit PhysRegMatrix(const AsmTargetRegisters& target);

    /// True when \p idx may hold \p range: allocatable, and nothing already
    /// bound there is live at a point \p range covers.
    bool available(RegType regClass, uint32_t idx, const LiveRange& range) const;

    /// True when every unit of [base, base + width) may hold \p range. False when
    /// the run leaves the class, which is what stops a tuple running off the end.
    bool runAvailable(RegType regClass, uint32_t base, uint32_t width,
                      const LiveRange& range) const;

    /// Lowest base where \p width consecutive units are available, if any.
    /// No alignment is applied, because no operand alignment is modelled yet.
    std::optional<uint32_t> findFreeRun(RegType regClass, uint32_t width,
                                        const LiveRange& range) const;

    /// Values bound to \p idx whose range overlaps \p range, appended to \p out.
    /// This is what an evicting policy needs: not whether there is a conflict,
    /// but who it is with.
    void collectConflicts(RegType regClass, uint32_t idx, const LiveRange& range,
                          std::vector<SSAValueID>& out) const;

    void bind(RegType regClass, uint32_t idx, SSAValueID value, const LiveRange& range);

    /// Binding keeps a pointer to \p range, so a temporary would dangle. Deleted
    /// rather than left to fail at runtime; the query methods above take a
    /// temporary safely, because they only read during the call.
    void bind(RegType regClass, uint32_t idx, SSAValueID value, LiveRange&& range) = delete;

    /// Release \p value from \p idx. Silent when it does not hold it, so undoing
    /// a partly applied tuple needs no bookkeeping.
    void unbind(RegType regClass, uint32_t idx, SSAValueID value);

    size_t bindingCount() const;

    /// Highest index bound in \p regClass, or std::nullopt when none is. This is
    /// the number a resource descriptor cares about, not the peak pressure.
    std::optional<uint32_t> highestBound(RegType regClass) const;

    std::string toString() const;

   private:
    struct Binding {
        SSAValueID value = kInvalidSSAValueID;
        const LiveRange* range = nullptr;
    };

    /// Bindings for one class, indexed by register index.
    struct ClassUnits {
        RegType regClass = RegType::UNKNOWN;
        std::vector<std::vector<Binding>> byIndex;
    };

    const ClassUnits* unitsFor(RegType regClass) const;
    ClassUnits* unitsFor(RegType regClass);

    const AsmTargetRegisters* target_;
    std::vector<ClassUnits> classes_;
};

}  // namespace stinkytofu
