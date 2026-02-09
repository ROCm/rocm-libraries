/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cmath> // For std::sinf, std::cosf, std::isinf, std::isnan
#include <cstdint> // For uint32_t
#include <gtest/gtest.h> // Include Google Test framework
#include <iomanip> // For std::hex, std::fixed, std::setprecision
#include <limits>

namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE
{
    // These are already in rocwmma_xfloat32.hpp, just ensuring the namespace exists.
} // namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE

// Need to define these if __HIPCC_RTC__ is not defined, which is the case for host unit tests.
#ifndef FLT_EPSILON
#include <cfloat> // Provides FLT_EPSILON, FLT_MAX, FLT_MIN, HUGE_VALF
#endif

// --- End Mocks ---

// Include the file to be tested
#include <rocwmma/internal/rocwmma_xfloat32.hpp>

// Define a small epsilon for float comparisons
const float FLOAT_EPSILON = 1e-3f; // Adjust as needed for FP8 precision

// Re-defining for testing private static methods.
// In a real project, consider using friend classes for testing private methods,
// or ensuring they are fully covered by public API tests.
// These are only visible to this translation unit for testing purposes.
// This is done to explicitly test the private static helper functions
// that are critical for the `rocwmma_xfloat32` constructors.
float float_to_xfloat32(float f);
float truncate_float_to_xfloat32(float f);

// Helper functions (can be placed in an anonymous namespace or a utility class for tests)
// These helpers facilitate common operations within the tests.
uint32_t get_float_bits(float f)
{
    rocwmma::detail::Bits32 b(f);
    return b.u32;
}

float float_from_bits(uint32_t u)
{
    rocwmma::detail::Bits32 b(u);
    return b.fp32;
}

// Define a test fixture if common setup/teardown is needed across multiple tests.
// For these tests, a simple fixture is used, though not strictly necessary for all tests,
// it's good practice for scalability and organization.
class XFloat32Test : public ::testing::Test
{
protected:
    // Any setup code for the tests can go here.
    // For example:
    // virtual void SetUp() override {
    //     // Initialize common test data or resources
    // }

    // Any teardown code for the tests can go here.
    // virtual void TearDown() override {
    //     // Clean up resources
    // }
};

// --- Unit Tests using Google Test Macros ---

// Test Case: Constructors and Assignments
// Verifies correct initialization and assignment behavior of rocwmma_xfloat32.
TEST_F(XFloat32Test, ConstructorsAndAssignments)
{
    // float constructor using truncation
    // Verifies that converting a float to xfloat32 by truncation works as expected.
    rocwmma_xfloat32 xf2(123.456f);
    EXPECT_FLOAT_EQ(truncate_float_to_xfloat32(123.456f), static_cast<float>(xf2))
        << "Float constructor (truncating)";

    // float constructor using rounding (round_up enum)
    // Verifies that converting a float to xfloat32 by rounding works as expected.
    rocwmma_xfloat32 xf3(1.23456f, rocwmma_xfloat32::round_up);
    EXPECT_FLOAT_EQ(float_to_xfloat32(1.23456f), static_cast<float>(xf3))
        << "Float constructor (rounding)";

    // Copy constructor
    // Ensures that one rocwmma_xfloat32 can be correctly copied to another.
    rocwmma_xfloat32 xf4 = xf2;
    EXPECT_FLOAT_EQ(static_cast<float>(xf2), static_cast<float>(xf4)) << "Copy constructor";

    // Copy assignment operator
    // Ensures correct assignment behavior from one rocwmma_xfloat32 to another.
    rocwmma_xfloat32 xf5;
    xf5 = xf3;
    EXPECT_FLOAT_EQ(static_cast<float>(xf3), static_cast<float>(xf5)) << "Copy assignment";

    // Move constructor
    // Tests efficient transfer of resources (though for a simple float wrapper, it's trivial).
    rocwmma_xfloat32 xf6 = std::move(xf4);
    EXPECT_FLOAT_EQ(truncate_float_to_xfloat32(123.456f), static_cast<float>(xf6))
        << "Move constructor";

    // Move assignment operator
    // Tests efficient assignment from a temporary or moved object.
    rocwmma_xfloat32 xf7;
    xf7 = std::move(xf5);
    EXPECT_FLOAT_EQ(float_to_xfloat32(1.23456f), static_cast<float>(xf7)) << "Move assignment";
}

// Test Case: Float Conversion Logic (Truncate & Round)
// Deeply tests the bit manipulation logic for float to xfloat32 conversions.
TEST_F(XFloat32Test, FloatConversionLogic)
{
    // Test truncate_float_to_xfloat32
    // Verifies that lower bits are correctly zeroed out during truncation.
    EXPECT_EQ(0x3F800000, get_float_bits(rocwmma_xfloat32(1.0f)))
        << "Truncate 1.0f"; // 1.0f = 0x3F800000, truncated = 0x3F800000
    EXPECT_EQ(0x40000000, get_float_bits(rocwmma_xfloat32(2.0f)))
        << "Truncate 2.0f"; // 2.0f = 0x40000000, truncated = 0x40000000

    // Test truncation with non-zero lower bits
    float f_trunc_test_1 = float_from_bits(0x3F801234); // 1.0f + a little bit
    EXPECT_EQ(0x3F800000, get_float_bits(rocwmma_xfloat32(f_trunc_test_1)))
        << "Truncate 0x3F801234";

    float f_trunc_test_2 = float_from_bits(0x4000F000); // 2.0f + more bits
    EXPECT_EQ(0x4000E000, get_float_bits(rocwmma_xfloat32(f_trunc_test_2)))
        << "Truncate 0x4000F000"; // 0x4000F000 & 0xffffe000 = 0x4000E000

    // Test float_to_xfloat32 (rounding logic)
    // Basic rounding down (lower 13 bits < 0x1000)
    EXPECT_EQ(
        0x3F800000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(0x3F800FFF), rocwmma_xfloat32::round_up)))
        << "Round down (0FFF)";
    EXPECT_EQ(
        0x3F800000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(0x3F8007FF), rocwmma_xfloat32::round_up)))
        << "Round down (07FF)";

    // Basic rounding up (lower 13 bits > 0x1000)
    EXPECT_EQ(
        0x3F802000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(0x3F801001), rocwmma_xfloat32::round_up)))
        << "Round up (1001)";
    EXPECT_EQ(
        0x3F802000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(0x3F801FFF), rocwmma_xfloat32::round_up)))
        << "Round up (1FFF)";

    // Round to nearest, ties to even (lower 13 bits == 0x1000)
    // float_from_bits(0x3F801000) represents a value exactly halfway between two representable xfloat32 values.
    // Its mantissa (shifted) is even (0x1FC00), so it should round down.
    EXPECT_EQ(
        0x3F800000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(0x3F801000), rocwmma_xfloat32::round_up)))
        << "Round to even (even mantissa)";

    // float_from_bits(0x3F803000) also represents a value exactly halfway.
    // Its mantissa (shifted) is odd (0x1FC01), so it should round up.
    EXPECT_EQ(
        0x3F804000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(0x3F803000), rocwmma_xfloat32::round_up)))
        << "Round to even (odd mantissa)";

    // Test Inf / NaN handling for rounding
    float inf_val = std::numeric_limits<float>::infinity();
    EXPECT_EQ(get_float_bits(inf_val),
              get_float_bits(rocwmma_xfloat32(inf_val, rocwmma_xfloat32::round_up)))
        << "Round Inf: should remain Inf";

    float qnan_val = std::numeric_limits<float>::quiet_NaN();
    // A standard QNaN (0x7FC00000) should remain 0x7FC00000 after rounding.
    EXPECT_EQ(0x7FC00000, get_float_bits(rocwmma_xfloat32(qnan_val, rocwmma_xfloat32::round_up)))
        << "Round QNaN: should remain QNaN";

    // Signaling NaN conversion: 0x7FA00001 (example sNaN with lowest bit set)
    // The logic `u.u32 |= 0x2000` in the original code is intended to convert
    // signaling NaNs to quiet NaNs by setting the most significant bit of the truncated mantissa.
    uint32_t sNaN_bits = 0x7FA00001; // Example signaling NaN
    EXPECT_EQ(
        0x7FA02000,
        get_float_bits(rocwmma_xfloat32(float_from_bits(sNaN_bits), rocwmma_xfloat32::round_up)))
        << "Round SNaN: should convert to QNaN by setting 0x2000 bit";
}

// Test Case: Conversion Operators
// Verifies explicit and implicit conversions to other numeric types.
TEST_F(XFloat32Test, ConversionOperators)
{
    rocwmma_xfloat32 xf_val(42.5f); // Note: this will be truncated/rounded by the constructor
    rocwmma_xfloat32 xf_zero(0.0f);

    // operator float() - Implicit conversion
    EXPECT_FLOAT_EQ(static_cast<float>(xf_val), float(xf_val)) << "operator float() (implicit)";

    // operator bool() - Explicit conversion
    EXPECT_TRUE(static_cast<bool>(xf_val)) << "operator bool() for non-zero";
    EXPECT_FALSE(static_cast<bool>(xf_zero)) << "operator bool() for zero";

    // operator uint32_t() - Explicit conversion
    // Note: conversion from float to integer types truncates towards zero.
    EXPECT_EQ(static_cast<uint32_t>(static_cast<float>(xf_val)), static_cast<uint32_t>(xf_val))
        << "operator uint32_t()";
    // EXPECT_EQ(0xFFFFFFF0, static_cast<uint32_t>(rocwmma_xfloat32(0xFFFFFFF0.0f))) << "operator uint32_t() large value";

    // operator long() - Explicit conversion
    EXPECT_EQ(static_cast<long>(static_cast<float>(xf_val)), static_cast<long>(xf_val))
        << "operator long()";
    EXPECT_EQ(-123L, static_cast<long>(rocwmma_xfloat32(-123.45f)))
        << "operator long() negative value";

    // operator double() - Explicit conversion
    EXPECT_DOUBLE_EQ(static_cast<double>(static_cast<float>(xf_val)), static_cast<double>(xf_val))
        << "operator double()";
}

// Test Case: from_bits factory
// Ensures that `rocwmma_xfloat32::from_bits` correctly creates a value from a raw bit pattern.
TEST_F(XFloat32Test, FromBitsFactory)
{
    uint32_t         test_bits = 0x3F800000; // Represents 1.0f
    rocwmma_xfloat32 xf        = rocwmma_xfloat32::from_bits(test_bits);
    EXPECT_EQ(test_bits, get_float_bits(xf))
        << "from_bits(0x3F800000) should correctly set bits for 1.0f";

    uint32_t neg_one_bits = 0xBF800000; // Represents -1.0f
    xf                    = rocwmma_xfloat32::from_bits(neg_one_bits);
    EXPECT_EQ(neg_one_bits, get_float_bits(xf))
        << "from_bits(0xBF800000) should correctly set bits for -1.0f";
}

// Test Case: Unary Operators
// Verifies the behavior of unary plus and unary minus.
TEST_F(XFloat32Test, UnaryOperators)
{
    rocwmma_xfloat32 a(5.0f);
    rocwmma_xfloat32 b(-3.0f);

    // Unary plus should return the value itself.
    EXPECT_FLOAT_EQ(5.0f, static_cast<float>(+a)) << "Unary plus on positive";
    EXPECT_FLOAT_EQ(-3.0f, static_cast<float>(+b)) << "Unary plus on negative";

    // Unary minus should flip the sign bit.
    EXPECT_FLOAT_EQ(-5.0f, static_cast<float>(-a)) << "Unary minus on positive";
    EXPECT_FLOAT_EQ(3.0f, static_cast<float>(-b)) << "Unary minus on negative";

    // Detailed bit check for unary minus: it should just flip the sign bit (MSB).
    uint32_t a_bits = get_float_bits(static_cast<float>(a));
    EXPECT_EQ(a_bits ^ 0x80000000, get_float_bits(-a)) << "Unary minus flips sign bit";
}

// Test Case: Binary Arithmetic Operators
// Checks standard arithmetic operations between two rocwmma_xfloat32 values.
TEST_F(XFloat32Test, BinaryArithmeticOperators)
{
    rocwmma_xfloat32 a(1.0f);
    rocwmma_xfloat32 b(3.0f);
    rocwmma_xfloat32 c(-2.0f);

    // Addition
    EXPECT_FLOAT_EQ(4.0f, static_cast<float>(a + b)) << "Addition (positive + positive)";
    EXPECT_FLOAT_EQ(-1.0f, static_cast<float>(a + c)) << "Addition (positive + negative)";

    // Subtraction
    EXPECT_FLOAT_EQ(-2.0f, static_cast<float>(a - b)) << "Subtraction (positive - positive)";
    EXPECT_FLOAT_EQ(3.0f, static_cast<float>(a - c)) << "Subtraction (positive - negative)";

    // Multiplication
    EXPECT_FLOAT_EQ(3.0f, static_cast<float>(a * b)) << "Multiplication (positive * positive)";
    EXPECT_FLOAT_EQ(-2.0f, static_cast<float>(a * c)) << "Multiplication (positive * negative)";

    // Division
    EXPECT_NEAR(0.333333f, static_cast<float>(a / b), FLOAT_EPSILON)
        << "Division (positive / positive)"; // Using ASSERT_NEAR for floating point comparison with tolerance
    EXPECT_FLOAT_EQ(-.50f, static_cast<float>(a / c)) << "Division (positive / negative)";
    // Division by zero should result in infinity.
    EXPECT_TRUE(std::isinf(static_cast<float>(a / rocwmma_xfloat32(0.0f))))
        << "Division by zero is Inf";
}

// Test Case: Comparison Operators
// Verifies relational and equality operators.
TEST_F(XFloat32Test, ComparisonOperators)
{
    rocwmma_xfloat32 a(5.0f);
    rocwmma_xfloat32 b(10.0f);
    rocwmma_xfloat32 c(5.0f);

    EXPECT_LT(a, b) << "Less than";
    EXPECT_GT(b, a) << "Greater than";
    EXPECT_EQ(a, c) << "Equal to";
    EXPECT_LE(a, c) << "Less than or equal to (equal case)";
    EXPECT_LE(a, b) << "Less than or equal to (less case)";
    EXPECT_NE(a, b) << "Not equal to";
    EXPECT_GE(b, a) << "Greater than or equal to (greater case)";
    EXPECT_GE(a, c) << "Greater than or equal to (equal case)";

    // NaN comparison behavior: NaN is never equal to, less than, or greater than anything (including itself).
    rocwmma_xfloat32 nan_xf = std::numeric_limits<rocwmma_xfloat32>::quiet_NaN();
    EXPECT_FALSE(nan_xf == nan_xf) << "NaN == NaN is false";
    EXPECT_FALSE(nan_xf < a) << "NaN < any is false";
    EXPECT_FALSE(nan_xf > a) << "NaN > any is false";
    // EXPECT_FALSE(nan_xf <= a) << "NaN <= any is false";
    // EXPECT_FALSE(nan_xf >= a) << "NaN >= any is false";
    EXPECT_TRUE(nan_xf != a) << "NaN != any is true";
}

// Test Case: Compound Assignment Operators
// Checks shorthand assignment operators like `+=`, `-=`, etc.
TEST_F(XFloat32Test, CompoundAssignmentOperators)
{
    rocwmma_xfloat32 xf_val(10.0f);

    xf_val += rocwmma_xfloat32(5.0f);
    EXPECT_FLOAT_EQ(15.0f, static_cast<float>(xf_val)) << "Add assignment";

    xf_val -= rocwmma_xfloat32(2.0f);
    EXPECT_FLOAT_EQ(13.0f, static_cast<float>(xf_val)) << "Subtract assignment";

    xf_val *= rocwmma_xfloat32(2.0f);
    EXPECT_FLOAT_EQ(26.0f, static_cast<float>(xf_val)) << "Multiply assignment";

    xf_val /= rocwmma_xfloat32(4.0f);
    EXPECT_FLOAT_EQ(6.5f, static_cast<float>(xf_val)) << "Divide assignment";
}

// Test Case: Increment/Decrement Operators
// Verifies prefix and postfix increment/decrement behavior.
TEST_F(XFloat32Test, IncrementDecrementOperators)
{
    rocwmma_xfloat32 xf_val(5.0f);

    // Prefix increment (`++a`): increments then returns the new value.
    rocwmma_xfloat32 result_pre_inc = ++xf_val;
    EXPECT_FLOAT_EQ(6.0f, static_cast<float>(xf_val)) << "Prefix increment value";
    EXPECT_FLOAT_EQ(6.0f, static_cast<float>(result_pre_inc)) << "Prefix increment return value";

    // Postfix increment (`a++`): returns the old value then increments.
    rocwmma_xfloat32 result_post_inc = xf_val++;
    EXPECT_FLOAT_EQ(7.0f, static_cast<float>(xf_val)) << "Postfix increment value";
    EXPECT_FLOAT_EQ(6.0f, static_cast<float>(result_post_inc)) << "Postfix increment return value";

    // Prefix decrement (`--a`): decrements then returns the new value.
    rocwmma_xfloat32 result_pre_dec = --xf_val;
    EXPECT_FLOAT_EQ(6.0f, static_cast<float>(xf_val)) << "Prefix decrement value";
    EXPECT_FLOAT_EQ(6.0f, static_cast<float>(result_pre_dec)) << "Prefix decrement return value";

    // Postfix decrement (`a--`): returns the old value then decrements.
    rocwmma_xfloat32 result_post_dec = xf_val--;
    EXPECT_FLOAT_EQ(5.0f, static_cast<float>(xf_val)) << "Postfix decrement value";
    EXPECT_FLOAT_EQ(6.0f, static_cast<float>(result_post_dec)) << "Postfix decrement return value";
}

// Test Case: std:: functions
// Verifies that custom overloads for standard library functions work as expected.
TEST_F(XFloat32Test, StdFunctions)
{
    rocwmma_xfloat32 zero(0.0f);
    rocwmma_xfloat32 positive(10.0f);
    rocwmma_xfloat32 neg_one(-1.0f);
    rocwmma_xfloat32 inf  = std::numeric_limits<rocwmma_xfloat32>::infinity();
    rocwmma_xfloat32 qnan = std::numeric_limits<rocwmma_xfloat32>::quiet_NaN();

    // std::isinf
    EXPECT_TRUE(std::isinf(inf)) << "std::isinf(infinity)";
    EXPECT_FALSE(std::isinf(positive)) << "std::isinf(positive) is false";
    EXPECT_FALSE(std::isinf(zero)) << "std::isinf(zero) is false";
    EXPECT_FALSE(std::isinf(qnan)) << "std::isinf(NaN) is false";

    // std::isnan
    EXPECT_TRUE(std::isnan(qnan)) << "std::isnan(QNaN)";
    EXPECT_FALSE(std::isnan(inf)) << "std::isnan(infinity) is false";
    EXPECT_FALSE(std::isnan(positive)) << "std::isnan(positive) is false";
    EXPECT_FALSE(std::isnan(zero)) << "std::isnan(zero) is false";

    // std::iszero (custom extension)
    EXPECT_TRUE(std::iszero(zero)) << "std::iszero(0.0)";
    EXPECT_FALSE(std::iszero(positive)) << "std::iszero(positive) is false";

    // std::sin
    EXPECT_NEAR(std::sinf(static_cast<float>(positive)),
                static_cast<float>(std::sin(positive)),
                FLOAT_EPSILON)
        << "std::sin(positive)";
    EXPECT_NEAR(std::sinf(static_cast<float>(neg_one)),
                static_cast<float>(std::sin(neg_one)),
                FLOAT_EPSILON)
        << "std::sin(negative)";

    // std::cos
    EXPECT_NEAR(std::cosf(static_cast<float>(positive)),
                static_cast<float>(std::cos(positive)),
                FLOAT_EPSILON)
        << "std::cos(positive)";
    EXPECT_NEAR(std::cosf(static_cast<float>(neg_one)),
                static_cast<float>(std::cos(neg_one)),
                FLOAT_EPSILON)
        << "std::cos(negative)";

    // std::real (custom extension, likely for complex numbers but implemented for xfloat32)
    EXPECT_NEAR(
        static_cast<float>(positive), static_cast<float>(std::real(positive)), FLOAT_EPSILON)
        << "std::real";
}

// Test Case: std::numeric_limits<rocwmma_xfloat32>
// Verifies that the specialized numeric_limits provides correct floating-point properties.
TEST_F(XFloat32Test, NumericLimits)
{
    auto nl = std::numeric_limits<rocwmma_xfloat32>();

    EXPECT_FLOAT_EQ(FLT_EPSILON, static_cast<float>(nl.epsilon())) << "epsilon()";
    EXPECT_FLOAT_EQ(HUGE_VALF, static_cast<float>(nl.infinity())) << "infinity()";
    EXPECT_FLOAT_EQ(-FLT_MAX, static_cast<float>(nl.lowest())) << "lowest()";
    EXPECT_FLOAT_EQ(FLT_MAX, static_cast<float>(nl.max())) << "max()";
    EXPECT_FLOAT_EQ(FLT_MIN, static_cast<float>(nl.min())) << "min()";

    // Specific bit patterns for NaNs should match the expected values.
    EXPECT_EQ(0x7FF80000, get_float_bits(nl.quiet_NaN())) << "quiet_NaN() bits";
    EXPECT_TRUE(std::isnan(nl.quiet_NaN())) << "quiet_NaN() is NaN";

    EXPECT_EQ(0x7FF00000, get_float_bits(nl.signaling_NaN())) << "signaling_NaN() bits";
    EXPECT_TRUE(std::isnan(nl.signaling_NaN())) << "signaling_NaN() is NaN";
}

// Re-defining for testing private static methods.
// These functions are critical private implementation details of `rocwmma_xfloat32`'s constructors.
// For direct testing, they are re-declared here in the global namespace (or could be in an anonymous namespace)
// to make them accessible to the test functions. In a production codebase, a 'friend class'
// approach or testing purely via public API methods would be more common for private methods.
float float_to_xfloat32(float f)
{
    auto u = rocwmma::detail::Bits32{f};
    if(~u.u32 & 0x7f800000)
    {
        // When the exponent bits are not all 1s, the value is zero, normal, or subnormal.
        // Round the xfloat32 mantissa up by adding 0xFFF, plus 1 if the least significant bit
        // of the xfloat32 mantissa is 1 (odd). This implements round-to-nearest-even.
        u.u32 += 0xfff + ((u.u32 >> 13) & 1); // Round to nearest, round to even
    }
    else if(u.u32 & 0x1fff)
    {
        // When all of the exponent bits are 1, the value is Inf or NaN.
        // If any of the lower 13 bits of the mantissa are 1, set the least significant bit
        // of the xfloat32 mantissa (bit 13), to convert signaling NaNs to quiet NaNs.
        u.u32 |= 0x2000; // Preserve signaling NaN (by making it quiet)
    }
    u.u32 &= 0xffffe000; // Clear the lower 13 bits
    return u.fp32;
}

float truncate_float_to_xfloat32(float f)
{
    auto u = rocwmma::detail::Bits32{f};
    u.u32  = u.u32 & 0xffffe000; // Clear the lower 13 bits (truncation)
    return u.fp32;
}
