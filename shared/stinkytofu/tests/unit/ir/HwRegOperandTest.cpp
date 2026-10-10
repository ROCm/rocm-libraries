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

#include <gtest/gtest.h>

#include <sstream>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"

using namespace stinkytofu;

namespace {
std::string printed(const StinkyRegister& reg) {
    std::ostringstream os;
    HwReg::printOperand(os, reg);
    return os.str();
}
}  // namespace

// Symbolic ids resolve per arch; the canonical form drops default fields. A .s
// round trip prints the operand as written (raw_asm_spelling_hwreg.s), so these
// are what a pass-created or pass-changed hwreg operand looks like.
TEST(HwRegOperandTest, SymbolicIdsAndCanonicalForm) {
    const GfxArchID arch = getGfxArchID(12, 5, 0);
    EXPECT_EQ(HwReg::parseId(arch, "HW_REG_WAVE_MODE"), std::optional<uint16_t>(1));
    EXPECT_EQ(HwReg::parseId(arch, "HW_REG_WAVE_SCHED_MODE"), std::optional<uint16_t>(26));
    EXPECT_EQ(HwReg::parseId(arch, "0x1a"), std::optional<uint16_t>(26));
    EXPECT_FALSE(HwReg::parseId(arch, "HW_REG_NO_SUCH").has_value());

    EXPECT_EQ(printed(StinkyRegister::Hwreg(1, 0, 32)), "hwreg(1)");
    EXPECT_EQ(printed(StinkyRegister::Hwreg(2, 4, 32)), "hwreg(2,4)");
    EXPECT_EQ(printed(StinkyRegister::Hwreg(26, 0, 2)), "hwreg(26,0,2)");
    EXPECT_NE(HwReg::spellingKey(StinkyRegister::Hwreg(26, 0, 2)),
              HwReg::spellingKey(StinkyRegister::Hwreg(26, 0, 3)));
}
