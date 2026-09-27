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
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"

#include <algorithm>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"

namespace stinkytofu {

AsmTargetRegisters AsmTargetRegisters::forArch(GfxArchID arch, const RegClassSet& classes) {
    AsmTargetRegisters target;
    target.arch_ = arch;

    for (RegType regClass : kLiftableRegClasses) {
        if (classes.contains(regClass)) target.classes_.push_back(regClass);
    }

    // Every number here is declared by the architecture in DEF_ARCH, so adding a
    // target means editing its own <Arch>Formats.def and nothing else. Zero
    // means the architecture did not declare it, which leaves the class with no
    // allocatable index rather than a guessed one.
    target.vgprCount_ = getMaxVGPR(arch);
    target.sgprCount_ = getMaxSGPR(arch);
    target.vgprGranule_ = getVgprAllocGranule(arch);
    target.vgprPerSimd_ = getTotalVgprPerSimd(arch);
    return target;
}

AsmTargetRegisters AsmTargetRegisters::forFunction(const Function& function) {
    const std::array<int, 3>& isa = function.getGemmTileConfig().arch;
    // Allocatable follows what this function's SSA was actually lifted for. A
    // class left physical has no values, and offering it would invite a colouring
    // for registers nothing can rewrite.
    return forArch(getGfxArchID(static_cast<uint32_t>(isa[0]), static_cast<uint32_t>(isa[1]),
                                static_cast<uint32_t>(isa[2])),
                   function.ssaArena().liftedClasses());
}

bool AsmTargetRegisters::isAllocatableClass(RegType regClass) const {
    return std::find(classes_.begin(), classes_.end(), regClass) != classes_.end();
}

uint32_t AsmTargetRegisters::indexCount(RegType regClass) const {
    switch (regClass) {
        case RegType::V:
            return vgprCount_;
        case RegType::S:
            return sgprCount_;
        default:
            return 0;
    }
}

uint32_t AsmTargetRegisters::allocationGranule(RegType regClass) const {
    return regClass == RegType::V ? vgprGranule_ : 0;
}

uint32_t AsmTargetRegisters::totalPerSimd(RegType regClass) const {
    return regClass == RegType::V ? vgprPerSimd_ : 0;
}

void AsmTargetRegisters::reserve(RegType regClass, uint32_t first, uint32_t count) {
    if (count == 0) return;
    reserved_.push_back({regClass, first, count});
}

bool AsmTargetRegisters::isReserved(RegType regClass, uint32_t idx) const {
    for (const ReservedRange& range : reserved_) {
        if (range.regClass != regClass) continue;
        if (idx >= range.first && idx - range.first < range.count) return true;
    }
    return false;
}

bool AsmTargetRegisters::isAllocatable(RegType regClass, uint32_t idx) const {
    if (!isAllocatableClass(regClass)) return false;
    if (idx >= indexCount(regClass)) return false;
    return !isReserved(regClass, idx);
}

std::string AsmTargetRegisters::toString() const {
    std::string text = "arch=" + getArchName(arch_) + " v=" + std::to_string(vgprCount_) +
                       " s=" + std::to_string(sgprCount_) +
                       " vgprGranule=" + std::to_string(vgprGranule_) +
                       " vgprPerSimd=" + std::to_string(vgprPerSimd_);
    for (const ReservedRange& range : reserved_) {
        text += "\nreserved " + regTypeToString(range.regClass) + std::to_string(range.first) +
                ".." + std::to_string(range.first + range.count - 1);
    }
    return text;
}

}  // namespace stinkytofu
