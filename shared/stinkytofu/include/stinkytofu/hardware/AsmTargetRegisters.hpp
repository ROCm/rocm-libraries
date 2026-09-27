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

// Which physical registers an allocator may hand out.
//
// Every number comes from the architecture's own DEF_ARCH block in
// <Arch>Formats.def, reached through ArchHelper. Nothing is keyed on an
// architecture here, so supporting a target means editing that target's .def and
// nothing in this file.
//
// Deliberately small for a first allocator: the classes lifting produces, how
// many indexes each has, which of those are reserved, and the granule occupancy
// is measured in. Alignment, AGPR/VGPR aliasing, VGPR-MSB encoding, and
// call-clobber sets are all absent, because nothing models them yet and
// inventing values would be worse than refusing to answer.
//
// Note that indexCount() is the range an operand can encode, DEF_ARCH's maxVGPR,
// not the physical file size totalPerSimd() reports. On gfx1250 those are 256 and
// 1024: reaching the rest of the file needs VGPR-MSB, which is not modelled, so a
// kernel whose pressure exceeds the addressable range has no colouring here.
//
// No reserved range is built in. Which registers the late passes and each ABI
// mode actually reserve is an open question, so a caller that knows adds them
// with reserve() rather than reading a guess from here.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"

namespace stinkytofu {

class Function;

class STINKYTOFU_EXPORT AsmTargetRegisters {
   public:
    /// Half-open [first, first + count) hole in a class.
    struct ReservedRange {
        RegType regClass = RegType::UNKNOWN;
        uint32_t first = 0;
        uint32_t count = 0;
    };

    /// Description for \p arch, with no reserved ranges.
    ///
    /// \p classes narrows which register classes may be handed out, and defaults
    /// to every class lifting can model. Pass the scope a function was lifted
    /// with, or use forFunction(), so a class left physical is never coloured.
    static AsmTargetRegisters forArch(GfxArchID arch,
                                      const RegClassSet& classes = RegClassSet::all());

    /// As above, for the architecture recorded on \p function and the register
    /// classes its SSA was lifted for.
    static AsmTargetRegisters forFunction(const Function& function);

    /// True for a class this target may hand out: one the lifter can model, and
    /// one in scope. A class left physical has no SSA values, so colouring it
    /// would produce assignments nothing can apply.
    bool isAllocatableClass(RegType regClass) const;

    /// Every allocatable class, in a stable order. The one place the set is
    /// enumerated, so a consumer cannot restate a shorter list and silently skip
    /// a class this target allows.
    std::span<const RegType> allocatableClasses() const {
        return classes_;
    }

    /// Indexes an operand can encode in \p regClass, from DEF_ARCH. 0 when the
    /// class is not allocatable or the architecture did not declare it.
    uint32_t indexCount(RegType regClass) const;

    /// Registers per allocation granule, which is the step occupancy moves in.
    /// 0 when unknown for this architecture.
    uint32_t allocationGranule(RegType regClass) const;

    /// Registers of \p regClass one SIMD has, for occupancy. 0 when unknown.
    uint32_t totalPerSimd(RegType regClass) const;

    /// Withhold [first, first + count) in \p regClass from allocation.
    void reserve(RegType regClass, uint32_t first, uint32_t count);

    bool isReserved(RegType regClass, uint32_t idx) const;

    /// True when \p idx may hold a value: allocatable class, in range, not reserved.
    bool isAllocatable(RegType regClass, uint32_t idx) const;

    std::span<const ReservedRange> reservedRanges() const {
        return reserved_;
    }

    GfxArchID arch() const {
        return arch_;
    }

    std::string toString() const;

   private:
    GfxArchID arch_{};
    std::vector<RegType> classes_;
    uint32_t vgprCount_ = 0;
    uint32_t sgprCount_ = 0;
    uint32_t vgprGranule_ = 0;
    uint32_t vgprPerSimd_ = 0;
    std::vector<ReservedRange> reserved_;
};

}  // namespace stinkytofu
