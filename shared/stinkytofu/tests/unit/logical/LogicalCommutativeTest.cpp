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

#include <gtest/gtest.h>

#include "stinkytofu/ir/logical/LogicalInstructions.hpp"

using namespace stinkytofu;

/**
 * @brief Test that commutative operations are properly marked
 */
TEST(CommutativeTest, VectorArithmeticCommutative) {
    StinkyRegister dst = StinkyRegister::Virtual(0);
    StinkyRegister src0 = StinkyRegister::Virtual(1);
    StinkyRegister src1 = StinkyRegister::Virtual(2);

    // Test vector add operations (should be commutative)
    {
        auto* add = VAddF32(dst, src0, src1);
        EXPECT_TRUE(add->isCommutative()) << "VAddF32 should be commutative";
        add->safeErase();
    }

    {
        auto* add = VAddF16(dst, src0, src1);
        EXPECT_TRUE(add->isCommutative()) << "VAddF16 should be commutative";
        add->safeErase();
    }

    {
        auto* add = VAddU32(dst, src0, src1);
        EXPECT_TRUE(add->isCommutative()) << "VAddU32 should be commutative";
        add->safeErase();
    }

    {
        auto* add = VAddI32(dst, src0, src1);
        EXPECT_TRUE(add->isCommutative()) << "VAddI32 should be commutative";
        add->safeErase();
    }

    // Test vector mul operations (should be commutative)
    {
        auto* mul = VMulF32(dst, src0, src1);
        EXPECT_TRUE(mul->isCommutative()) << "VMulF32 should be commutative";
        mul->safeErase();
    }

    {
        auto* mul = VMulF16(dst, src0, src1);
        EXPECT_TRUE(mul->isCommutative()) << "VMulF16 should be commutative";
        mul->safeErase();
    }
}

/**
 * @brief Test that min/max operations are properly marked as commutative
 */
TEST(CommutativeTest, VectorMinMaxCommutative) {
    StinkyRegister dst = StinkyRegister::Virtual(0);
    StinkyRegister src0 = StinkyRegister::Virtual(1);
    StinkyRegister src1 = StinkyRegister::Virtual(2);

    // Test vector min operations (should be commutative)
    {
        auto* min = VMinF32(dst, src0, src1);
        EXPECT_TRUE(min->isCommutative()) << "VMinF32 should be commutative";
        min->safeErase();
    }

    {
        auto* min = VMinF16(dst, src0, src1);
        EXPECT_TRUE(min->isCommutative()) << "VMinF16 should be commutative";
        min->safeErase();
    }

    // Test vector max operations (should be commutative)
    {
        auto* max = VMaxF32(dst, src0, src1);
        EXPECT_TRUE(max->isCommutative()) << "VMaxF32 should be commutative";
        max->safeErase();
    }

    {
        auto* max = VMaxF16(dst, src0, src1);
        EXPECT_TRUE(max->isCommutative()) << "VMaxF16 should be commutative";
        max->safeErase();
    }
}

/**
 * @brief Test that bitwise operations are properly marked as commutative
 */
TEST(CommutativeTest, VectorBitwiseCommutative) {
    StinkyRegister dst = StinkyRegister::Virtual(0);
    StinkyRegister src0 = StinkyRegister::Virtual(1);
    StinkyRegister src1 = StinkyRegister::Virtual(2);

    // Test vector bitwise operations (should be commutative)
    {
        auto* and_op = VAndB32(dst, src0, src1);
        EXPECT_TRUE(and_op->isCommutative()) << "VAndB32 should be commutative";
    }

    {
        auto* or_op = VOrB32(dst, src0, src1);
        EXPECT_TRUE(or_op->isCommutative()) << "VOrB32 should be commutative";
    }

    {
        auto* xor_op = VXorB32(dst, src0, src1);
        EXPECT_TRUE(xor_op->isCommutative()) << "VXorB32 should be commutative";
    }
}

/**
 * @brief Test that scalar operations are properly marked as commutative
 */
TEST(CommutativeTest, ScalarCommutative) {
    StinkyRegister dst = StinkyRegister::VirtualSGPR(0);
    StinkyRegister src0 = StinkyRegister::VirtualSGPR(1);
    StinkyRegister src1 = StinkyRegister::VirtualSGPR(2);

    // Test scalar add operations (should be commutative)
    {
        auto* add = SAddU32(dst, src0, src1);
        EXPECT_TRUE(add->isCommutative()) << "SAddU32 should be commutative";
    }

    {
        auto* add = SAddI32(dst, src0, src1);
        EXPECT_TRUE(add->isCommutative()) << "SAddI32 should be commutative";
    }

    // Test scalar mul operations (should be commutative)
    {
        auto* mul = SMulI32(dst, src0, src1);
        EXPECT_TRUE(mul->isCommutative()) << "SMulI32 should be commutative";
    }

    // Test scalar min/max operations (should be commutative)
    {
        auto* min = SMinU32(dst, src0, src1);
        EXPECT_TRUE(min->isCommutative()) << "SMinU32 should be commutative";
    }

    {
        auto* max = SMaxU32(dst, src0, src1);
        EXPECT_TRUE(max->isCommutative()) << "SMaxU32 should be commutative";
    }

    // Test scalar bitwise operations (should be commutative)
    {
        auto* and_op = SAndB32(dst, src0, src1);
        EXPECT_TRUE(and_op->isCommutative()) << "SAndB32 should be commutative";
    }

    {
        auto* or_op = SOrB32(dst, src0, src1);
        EXPECT_TRUE(or_op->isCommutative()) << "SOrB32 should be commutative";
    }

    {
        auto* xor_op = SXorB32(dst, src0, src1);
        EXPECT_TRUE(xor_op->isCommutative()) << "SXorB32 should be commutative";
    }
}

/**
 * @brief Test that non-commutative operations are NOT marked as commutative
 */
TEST(CommutativeTest, NonCommutativeOperations) {
    StinkyRegister dst = StinkyRegister::Virtual(0);
    StinkyRegister src0 = StinkyRegister::Virtual(1);
    StinkyRegister src1 = StinkyRegister::Virtual(2);

    // Test subtraction (NOT commutative)
    {
        auto* sub = VSubF32(dst, src0, src1);
        EXPECT_FALSE(sub->isCommutative()) << "VSubF32 should NOT be commutative";
    }

    {
        auto* sub = VSubI32(dst, src0, src1);
        EXPECT_FALSE(sub->isCommutative()) << "VSubI32 should NOT be commutative";
    }

    // Test shift operations (NOT commutative)
    {
        auto* lsl = VLShiftLeftB32(dst, src0, src1);
        EXPECT_FALSE(lsl->isCommutative()) << "VLShiftLeftB32 should NOT be commutative";
    }

    {
        auto* lsr = VLShiftRightB32(dst, src0, src1);
        EXPECT_FALSE(lsr->isCommutative()) << "VLShiftRightB32 should NOT be commutative";
    }

    // Test move operation (NOT commutative, only has 1 operand)
    {
        auto* mov = VMovB32(dst, src0);
        EXPECT_FALSE(mov->isCommutative()) << "VMovB32 should NOT be commutative";
        mov->safeErase();
    }
}
