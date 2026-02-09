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
#include "gtest/gtest.h"

// For testing purposes, we need to mock or include the necessary HIP headers
// and utility headers that float8.hpp depends on.
// In a real scenario, these would be provided by your build system.

// Mock necessary types and macros from hip/hip_fp8.h and ROCWMMA headers
// IMPORTANT: These mocks are for demonstration and may not reflect the exact
// behavior or bit representations of real HIP types.
// You MUST replace these with actual includes if compiling in a HIP environment.

// Mock __hip_fp8_storage_t
using __hip_fp8_storage_t = uint8_t;

// Mock ROCWMMA defines and utility headers
#define ROCWMMA_HOST_DEVICE
#define ROCWMMA_HOST
#define ROCWMMA_ARCH_GFX942 0 // For this test, assume not on GFX942
#define ROCWMMA_ARCH_GFX950 0 // For this test, assume not on GFX950
#define ROCWMMA_ARCH_GFX12 0 // For this test, assume not on GFX12
#define ROCWMMA_ARCH_HOST 1 // Assume we are on host

// Basic type_traits mock
namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE
{
    template <typename T>
    struct is_arithmetic : std::false_type
    {
    };
    template <typename T>
    struct is_floating_point : std::false_type
    {
    };
    template <typename T>
    struct is_signed : std::false_type
    {
    };
    struct true_type : std::true_type
    {
    };
    struct false_type : std::false_type
    {
    };
} // namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE

// Basic numeric_limits mock for round_to_nearest
namespace ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE
{
    enum float_round_style
    {
        round_indeterminate       = -1,
        round_toward_zero         = 0,
        round_to_nearest          = 1,
        round_toward_infinity     = 2,
        round_toward_neg_infinity = 3
    };
} // namespace ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE

// Dummy uint8_t if not provided by <cstdint>
namespace rocwmma
{
    using uint8_t = std::uint8_t;
}

// Now include the float8.hpp header
#include <rocwmma/internal/float8.hpp>

// Define a small epsilon for float comparisons
const float FLOAT_EPSILON = 1e-4f; // Adjust as needed for FP8 precision

// Utility to create fp8 types from float values for testing
// This is a simplification; a real implementation would handle bit packing.
hip_fp8_e4m3 float_to_e4m3(float val)
{
    return hip_fp8_e4m3(val);
}

hip_fp8_e5m2 float_to_e5m2(float val)
{
    return hip_fp8_e5m2(val);
}

hip_fp8_e4m3_fnuz float_to_e4m3_fnuz(float val)
{
    return hip_fp8_e4m3_fnuz(val);
}

hip_fp8_e5m2_fnuz float_to_e5m2_fnuz(float val)
{
    return hip_fp8_e5m2_fnuz(val);
}

// --- Test fixture for common setup (optional, but good practice) ---
class Float8Test : public ::testing::Test
{
protected:
    // You can set up common data or operations here if needed
    void SetUp() override
    {
        // Code here will be run immediately after the constructor (right before each test).
    }

    void TearDown() override
    {
        // Code here will be run immediately after each test (right before the destructor).
    }
};

// --- Tests for Type Traits ---
TEST_F(Float8Test, TypeTraitsE4M3)
{
    using namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE;
    EXPECT_TRUE(is_arithmetic<hip_fp8_e4m3>::value);
    EXPECT_TRUE(is_floating_point<hip_fp8_e4m3>::value);
    EXPECT_TRUE(is_signed<hip_fp8_e4m3>::value);
}

TEST_F(Float8Test, TypeTraitsE5M2)
{
    using namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE;
    EXPECT_TRUE(is_arithmetic<hip_fp8_e5m2>::value);
    EXPECT_TRUE(is_floating_point<hip_fp8_e5m2>::value);
    EXPECT_TRUE(is_signed<hip_fp8_e5m2>::value);
}

TEST_F(Float8Test, TypeTraitsE4M3Fnuz)
{
    using namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE;
    EXPECT_TRUE(is_arithmetic<hip_fp8_e4m3_fnuz>::value);
    EXPECT_TRUE(is_floating_point<hip_fp8_e4m3_fnuz>::value);
    EXPECT_TRUE(is_signed<hip_fp8_e4m3_fnuz>::value);
}

TEST_F(Float8Test, TypeTraitsE5M2Fnuz)
{
    using namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE;
    EXPECT_TRUE(is_arithmetic<hip_fp8_e5m2_fnuz>::value);
    EXPECT_TRUE(is_floating_point<hip_fp8_e5m2_fnuz>::value);
    EXPECT_TRUE(is_signed<hip_fp8_e5m2_fnuz>::value);
}

// --- Tests for make_hip_fp8_from_bits functions ---
TEST_F(Float8Test, MakeHipFp8E4M3FromBits)
{
    // Test a specific bit pattern and conversion
    hip_fp8_e4m3 val = make_hip_fp8_e4m3_from_bits(0x30); // Example: 0.5 in E4M3
    EXPECT_EQ(val.__x, 0x30);
    EXPECT_NEAR(static_cast<float>(val), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MakeHipFp8E5M2FromBits)
{
    // Test a specific bit pattern and conversion
    hip_fp8_e5m2 val = make_hip_fp8_e5m2_from_bits(0x34); // Example: 0.25 in E5M2
    EXPECT_EQ(val.__x, 0x34);
    EXPECT_NEAR(static_cast<float>(val), 0.25f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MakeHipFp8E4M3FnuzFromBits)
{
    hip_fp8_e4m3_fnuz val = make_hip_fp8_e4m3_fnuz_from_bits(0x28);
    EXPECT_EQ(val.__x, 0x28);
    EXPECT_NEAR(static_cast<float>(val), 0.125f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MakeHipFp8E5M2FnuzFromBits)
{
    hip_fp8_e5m2_fnuz val = make_hip_fp8_e5m2_fnuz_from_bits(0x3c);
    EXPECT_EQ(val.__x, 0x3c);
    EXPECT_NEAR(static_cast<float>(val), 0.5f, FLOAT_EPSILON);
}

// --- Tests for Unary Sign Inversion ---
TEST_F(Float8Test, UnarySignInversionE4M3)
{
    hip_fp8_e4m3 a     = float_to_e4m3(0.5f); // 0x30
    hip_fp8_e4m3 neg_a = -a;
    EXPECT_NEAR(static_cast<float>(neg_a), -0.5f, FLOAT_EPSILON);
    EXPECT_EQ(neg_a.__x, (0x30 ^ 0x80)); // Check bit flip
}

TEST_F(Float8Test, UnarySignInversionE5M2)
{
    hip_fp8_e5m2 a     = float_to_e5m2(0.25f); // 0x34
    hip_fp8_e5m2 neg_a = -a;
    EXPECT_NEAR(static_cast<float>(neg_a), -0.25f, FLOAT_EPSILON);
    EXPECT_EQ(neg_a.__x, (0x34 ^ 0x80)); // Check bit flip
}

TEST_F(Float8Test, UnarySignInversionE4M3Fnuz)
{
    hip_fp8_e4m3_fnuz a_pos  = float_to_e4m3_fnuz(0.125f);
    hip_fp8_e4m3_fnuz a_neg  = float_to_e4m3_fnuz(-0.125f);
    hip_fp8_e4m3_fnuz a_zero = float_to_e4m3_fnuz(0.0f);

    EXPECT_NEAR(static_cast<float>(-a_pos), -0.125f, FLOAT_EPSILON);
    EXPECT_NEAR(static_cast<float>(-a_neg), 0.125f, FLOAT_EPSILON);
    EXPECT_NEAR(static_cast<float>(-a_zero), 0.0f, FLOAT_EPSILON); // Zero remains zero
}

TEST_F(Float8Test, UnarySignInversionE5M2Fnuz)
{
    hip_fp8_e5m2_fnuz a_pos  = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz a_neg  = float_to_e5m2_fnuz(-0.5f);
    hip_fp8_e5m2_fnuz a_zero = float_to_e5m2_fnuz(0.0f);

    EXPECT_NEAR(static_cast<float>(-a_pos), -0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(static_cast<float>(-a_neg), 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(static_cast<float>(-a_zero), 0.0f, FLOAT_EPSILON); // Zero remains zero
}

// --- Tests for Addition Operators ---
TEST_F(Float8Test, AddE4M3Float)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f); // Example value
    float        b = 0.5f;
    EXPECT_NEAR(a + b, 1.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b + a, 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE5M2Float)
{
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    float        b = 0.5f;
    EXPECT_NEAR(a + b, 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE4M3E5M2Mixed)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_NEAR(a + b, 1.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b + a, 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE4M3E4M3)
{
    hip_fp8_e4m3 a      = float_to_e4m3(0.5f);
    hip_fp8_e4m3 b      = float_to_e4m3(0.5f);
    hip_fp8_e4m3 result = a + b;
    EXPECT_NEAR(static_cast<float>(result), 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE5M2E5M2)
{ // New test
    hip_fp8_e5m2 a      = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b      = float_to_e5m2(0.5f);
    hip_fp8_e5m2 result = a + b;
    EXPECT_NEAR(static_cast<float>(result), 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddAssignE4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    a += b;
    EXPECT_NEAR(static_cast<float>(a), 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddAssignE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    a += b;
    EXPECT_NEAR(static_cast<float>(a), 1.5f, FLOAT_EPSILON);
}

// FNUZ types
TEST_F(Float8Test, AddE4M3FnuzFloat)
{
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    float             b = 0.125f;
    EXPECT_NEAR(a + b, 1.125f, FLOAT_EPSILON);
    EXPECT_NEAR(b + a, 1.125f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE5M2FnuzFloat)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    float             b = 0.5f;
    EXPECT_NEAR(float(a) + b, 1.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b + float(a), 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE4M3FnuzE5M2FnuzMixed)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_NEAR(a + b, 1.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b + a, 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE4M3FnuzE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a      = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz b      = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz result = a + b;
    EXPECT_NEAR(static_cast<float>(result), 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddE5M2FnuzE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a      = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b      = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz result = a + b;
    EXPECT_NEAR(static_cast<float>(result), 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddAssignE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    a += b;
    EXPECT_NEAR(static_cast<float>(a), 1.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, AddAssignE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    a += b;
    EXPECT_NEAR(static_cast<float>(a), 1.5f, FLOAT_EPSILON);
}

// --- Tests for Subtraction Operators ---
TEST_F(Float8Test, SubtractE4M3Float)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    float        b = 0.5f;
    EXPECT_NEAR(a - b, 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b - a, -0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE5M2Float)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    float        b = 0.5f;
    EXPECT_NEAR(a - b, 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b - a, -0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE4M3E5M2Mixed)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_NEAR(a - b, 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b - a, -0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE4M3E4M3)
{
    hip_fp8_e4m3 a      = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b      = float_to_e4m3(0.5f);
    hip_fp8_e4m3 result = a - b;
    EXPECT_NEAR(static_cast<float>(result), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE5M2E5M2)
{ // New test
    hip_fp8_e5m2 a      = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b      = float_to_e5m2(0.5f);
    hip_fp8_e5m2 result = a - b;
    EXPECT_NEAR(static_cast<float>(result), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractAssignE4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    a -= b;
    EXPECT_NEAR(static_cast<float>(a), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractAssignE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    a -= b;
    EXPECT_NEAR(static_cast<float>(a), 0.5f, FLOAT_EPSILON);
}

// FNUZ types
TEST_F(Float8Test, SubtractE4M3FnuzFloat)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    float             b = 0.5f;
    EXPECT_NEAR(a - b, 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b - a, -0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE5M2FnuzFloat)
{
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    float             b = 0.5f;
    EXPECT_NEAR(a - b, 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b - a, -0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE4M3FnuzE5M2FnuzMixed)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_NEAR(a - b, 0.5f, FLOAT_EPSILON);
    EXPECT_NEAR(b - a, -0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE4M3FnuzE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a      = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b      = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz result = a - b;
    EXPECT_NEAR(static_cast<float>(result), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractE5M2FnuzE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a      = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b      = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz result = a - b;
    EXPECT_NEAR(static_cast<float>(result), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractAssignE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    a -= b;
    EXPECT_NEAR(static_cast<float>(a), 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, SubtractAssignE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    a -= b;
    EXPECT_NEAR(static_cast<float>(a), 0.5f, FLOAT_EPSILON);
}

// --- Tests for Multiplication Operators ---
TEST_F(Float8Test, MultiplyE4M3E4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(0.5f);
    hip_fp8_e4m3 b = float_to_e4m3(2.0f);
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyE5M2E5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_NEAR(a * b, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyFloatE4M3)
{
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    float        a = 2.0f;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b * a, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyFloatE5M2)
{ // New test
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    float        a = 2.0f;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b * a, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyIntE4M3)
{
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    int32_t      a = 2;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyIntE5M2)
{ // New test
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    int32_t      a = 2;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyDoubleE4M3)
{ // New test
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    double       a = 2.0;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyDoubleE5M2)
{ // New test
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    double       a = 2.0;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyE4M3E5M2Mixed)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(2.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b * a, 1.0f, FLOAT_EPSILON);
}

// FNUZ types
TEST_F(Float8Test, MultiplyE4M3FnuzE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(2.0f);
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyE5M2FnuzE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_NEAR(a * b, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyFloatE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    float             a = 2.0f;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b * a, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyFloatE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    float             a = 2.0f;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b * a, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyIntE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    int32_t           a = 2;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyIntE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    int32_t           a = 2;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyDoubleE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    double            a = 2.0;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyDoubleE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    double            a = 2.0;
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, MultiplyE4M3FnuzE5M2FnuzMixed)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(2.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_NEAR(a * b, 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b * a, 1.0f, FLOAT_EPSILON);
}

// --- Tests for Division Operators ---
TEST_F(Float8Test, DivideE4M3E4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideE5M2E5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideFloatE4M3)
{
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    float        a = 1.0f;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b / a, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideFloatE5M2)
{ // New test
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    float        a = 1.0f;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b / a, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideIntE4M3)
{ // New test
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    int32_t      a = 1;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideIntE5M2)
{ // New test
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    int32_t      a = 1;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideDoubleE4M3)
{ // New test
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    double       a = 1.0;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideDoubleE5M2)
{ // New test
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    double       a = 1.0;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideE4M3E5M2Mixed)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b / a, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideAssignE4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    a /= b;
    EXPECT_NEAR(static_cast<float>(a), 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideAssignE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    a /= b;
    EXPECT_NEAR(static_cast<float>(a), 2.0f, FLOAT_EPSILON);
}

// FNUZ types
TEST_F(Float8Test, DivideE4M3FnuzE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideE5M2FnuzE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideFloatE4M3Fnuz)
{
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.25f);
    float             a = 1.0f;
    EXPECT_NEAR(a / b, 4.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b / a, 0.25f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideFloatE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    float             a = 1.0f;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b / a, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideIntE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    int32_t           a = 1;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideIntE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    int32_t           a = 1;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideDoubleE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    double            a = 1.0;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideDoubleE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    double            a = 1.0;
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideE4M3FnuzE5M2FnuzMixed)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_NEAR(a / b, 2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(b / a, 0.5f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideAssignE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    a /= b;
    EXPECT_NEAR(static_cast<float>(a), 2.0f, FLOAT_EPSILON);
}

TEST_F(Float8Test, DivideAssignE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    a /= b;
    EXPECT_NEAR(static_cast<float>(a), 2.0f, FLOAT_EPSILON);
}

// --- Tests for Comparison Operators ---
TEST_F(Float8Test, ComparisonEqualE4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(0.5f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    hip_fp8_e4m3 c = float_to_e4m3(1.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(Float8Test, ComparisonEqualE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(0.5f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    hip_fp8_e5m2 c = float_to_e5m2(1.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(Float8Test, ComparisonNotEqualE4M3)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(0.5f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    hip_fp8_e4m3 c = float_to_e4m3(1.0f);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(Float8Test, ComparisonNotEqualE5M2)
{
    hip_fp8_e5m2 a = float_to_e5m2(0.5f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    hip_fp8_e5m2 c = float_to_e5m2(1.0f);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(Float8Test, ComparisonLessThanE4M3)
{
    hip_fp8_e4m3 a = float_to_e4m3(0.5f);
    hip_fp8_e4m3 b = float_to_e4m3(1.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST_F(Float8Test, ComparisonLessThanE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(0.5f);
    hip_fp8_e5m2 b = float_to_e5m2(1.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST_F(Float8Test, ComparisonGreaterThanE4M3)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(Float8Test, ComparisonGreaterThanE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(Float8Test, ComparisonLessThanOrEqualE4M3)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(0.5f);
    hip_fp8_e4m3 b = float_to_e4m3(1.0f);
    hip_fp8_e4m3 c = float_to_e4m3(0.5f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TEST_F(Float8Test, ComparisonLessThanOrEqualE5M2)
{ // New test
    hip_fp8_e5m2 a = float_to_e5m2(0.5f);
    hip_fp8_e5m2 b = float_to_e5m2(1.0f);
    hip_fp8_e5m2 c = float_to_e5m2(0.5f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TEST_F(Float8Test, ComparisonGreaterThanOrEqualE4M3)
{ // New test
    hip_fp8_e4m3 a = float_to_e4m3(1.0f);
    hip_fp8_e4m3 b = float_to_e4m3(0.5f);
    hip_fp8_e4m3 c = float_to_e4m3(1.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

TEST_F(Float8Test, ComparisonGreaterThanOrEqualE5M2)
{
    hip_fp8_e5m2 a = float_to_e5m2(1.0f);
    hip_fp8_e5m2 b = float_to_e5m2(0.5f);
    hip_fp8_e5m2 c = float_to_e5m2(1.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

// Repeat for FNUZ types
TEST_F(Float8Test, ComparisonEqualE4M3Fnuz)
{
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    EXPECT_TRUE(a == b);
}

TEST_F(Float8Test, ComparisonEqualE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_TRUE(a == b);
}

TEST_F(Float8Test, ComparisonNotEqualE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(1.0f);
    EXPECT_TRUE(a != b);
}

TEST_F(Float8Test, ComparisonNotEqualE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(1.0f);
    EXPECT_TRUE(a != b);
}

TEST_F(Float8Test, ComparisonLessThanE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(1.0f);
    EXPECT_TRUE(a < b);
}

TEST_F(Float8Test, ComparisonLessThanE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(1.0f);
    EXPECT_TRUE(a < b);
}

TEST_F(Float8Test, ComparisonGreaterThanE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    EXPECT_TRUE(a > b);
}

TEST_F(Float8Test, ComparisonGreaterThanE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    EXPECT_TRUE(a > b);
}

TEST_F(Float8Test, ComparisonLessThanOrEqualE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz c = float_to_e4m3_fnuz(0.5f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
}

TEST_F(Float8Test, ComparisonLessThanOrEqualE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz c = float_to_e5m2_fnuz(0.5f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
}

TEST_F(Float8Test, ComparisonGreaterThanOrEqualE4M3Fnuz)
{ // New test
    hip_fp8_e4m3_fnuz a = float_to_e4m3_fnuz(1.0f);
    hip_fp8_e4m3_fnuz b = float_to_e4m3_fnuz(0.5f);
    hip_fp8_e4m3_fnuz c = float_to_e4m3_fnuz(1.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
}

TEST_F(Float8Test, ComparisonGreaterThanOrEqualE5M2Fnuz)
{ // New test
    hip_fp8_e5m2_fnuz a = float_to_e5m2_fnuz(1.0f);
    hip_fp8_e5m2_fnuz b = float_to_e5m2_fnuz(0.5f);
    hip_fp8_e5m2_fnuz c = float_to_e5m2_fnuz(1.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
}

// --- Tests for Numeric Limits (hip_fp8_e4m3) ---
TEST_F(Float8Test, NumericLimitsE4M3)
{
    using NL = ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE::numeric_limits<hip_fp8_e4m3>;
    EXPECT_TRUE(NL::is_specialized);
    EXPECT_TRUE(NL::is_signed);
    EXPECT_FALSE(NL::is_integer);
    EXPECT_FALSE(NL::has_infinity);
    EXPECT_TRUE(NL::has_quiet_NaN);
    EXPECT_TRUE(NL::has_signaling_NaN);
    EXPECT_TRUE(NL::has_denorm);
    EXPECT_TRUE(NL::has_denorm_loss);
    EXPECT_EQ(NL::round_style, ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE::round_to_nearest);
    EXPECT_FALSE(NL::is_iec559);
    EXPECT_TRUE(NL::is_bounded);
    EXPECT_FALSE(NL::is_modulo);
    EXPECT_EQ(NL::digits, 4);
    EXPECT_EQ(NL::digits10, 0);
    EXPECT_EQ(NL::max_digits10, 3);
    EXPECT_EQ(NL::radix, 2);
    EXPECT_EQ(NL::min_exponent, -5);
    EXPECT_EQ(NL::min_exponent10, -1);
    EXPECT_EQ(NL::max_exponent, 8);
    EXPECT_EQ(NL::max_exponent10, 2);
    EXPECT_FALSE(NL::traps);
    EXPECT_FALSE(NL::tinyness_before);

    // Test specific values using their bit representations
    EXPECT_EQ(NL::min().__x, 0x08);
    EXPECT_EQ(NL::lowest().__x, 0xFE);
    EXPECT_EQ(NL::max().__x, 0x7E);
    EXPECT_EQ(NL::epsilon().__x, 0x20);
    EXPECT_EQ(NL::round_error().__x, 0x30);
    EXPECT_EQ(NL::quiet_NaN().__x, 0x7F);
    EXPECT_EQ(NL::signaling_NaN().__x, 0x7F);
    EXPECT_EQ(NL::denorm_min().__x, 0x01);
}

// --- Tests for Numeric Limits (hip_fp8_e5m2) ---
TEST_F(Float8Test, NumericLimitsE5M2)
{
    using NL = ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE::numeric_limits<hip_fp8_e5m2>;
    EXPECT_TRUE(NL::is_specialized);
    EXPECT_TRUE(NL::is_signed);
    EXPECT_FALSE(NL::is_integer);
    EXPECT_TRUE(NL::has_infinity);
    EXPECT_TRUE(NL::has_quiet_NaN);
    EXPECT_TRUE(NL::has_signaling_NaN);
    EXPECT_TRUE(NL::has_denorm);
    EXPECT_TRUE(NL::has_denorm_loss);
    EXPECT_EQ(NL::round_style, ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE::round_to_nearest);
    EXPECT_FALSE(NL::is_iec559);
    EXPECT_TRUE(NL::is_bounded);
    EXPECT_FALSE(NL::is_modulo);
    EXPECT_EQ(NL::digits, 3);
    EXPECT_EQ(NL::digits10, 0);
    EXPECT_EQ(NL::max_digits10, 2);
    EXPECT_EQ(NL::radix, 2);
    EXPECT_EQ(NL::min_exponent, -13);
    EXPECT_EQ(NL::min_exponent10, -4);
    EXPECT_EQ(NL::max_exponent, 16);
    EXPECT_EQ(NL::max_exponent10, 4);
    EXPECT_FALSE(NL::traps);
    EXPECT_FALSE(NL::tinyness_before);

    // Test specific values using their bit representations
    EXPECT_EQ(NL::min().__x, 0x04);
    EXPECT_EQ(NL::max().__x, 0x7B);
    EXPECT_EQ(NL::lowest().__x, 0xFB);
    EXPECT_EQ(NL::epsilon().__x, 0x34);
    EXPECT_EQ(NL::round_error().__x, 0x38);
    EXPECT_EQ(NL::infinity().__x, 0x7C);
    EXPECT_EQ(NL::quiet_NaN().__x, 0x7F);
    EXPECT_EQ(NL::signaling_NaN().__x, 0x7F);
    EXPECT_EQ(NL::denorm_min().__x, 0x01);
}

// --- Tests for Numeric Limits (hip_fp8_e4m3_fnuz) ---
TEST_F(Float8Test, NumericLimitsE4M3Fnuz)
{
    using NL = ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE::numeric_limits<hip_fp8_e4m3_fnuz>;

    // Test specific values using their bit representations
    EXPECT_EQ(NL::epsilon().__x, static_cast<uint8_t>(0x28));
    // FNUZ types might have infinity/NaN represented differently (often 0x80 for NaN/Inf)
    EXPECT_EQ(NL::infinity().__x, static_cast<uint8_t>(0x80));
    EXPECT_EQ(NL::lowest().__x, static_cast<uint8_t>(0xFF));
    EXPECT_EQ(NL::max().__x, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(NL::min().__x, static_cast<uint8_t>(0x01));
    EXPECT_EQ(NL::quiet_NaN().__x, static_cast<uint8_t>(0x80));
    EXPECT_EQ(NL::signaling_NaN().__x, static_cast<uint8_t>(0x80));
}

// --- Tests for Numeric Limits (hip_fp8_e5m2_fnuz) ---
TEST_F(Float8Test, NumericLimitsE5M2Fnuz)
{
    using NL = ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE::numeric_limits<hip_fp8_e5m2_fnuz>;

    // Test specific values using their bit representations
    EXPECT_EQ(NL::epsilon().__x, static_cast<uint8_t>(0x38));
    EXPECT_EQ(NL::infinity().__x, static_cast<uint8_t>(0x80));
    EXPECT_EQ(NL::lowest().__x, static_cast<uint8_t>(0xFF));
    EXPECT_EQ(NL::max().__x, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(NL::min().__x, static_cast<uint8_t>(0x01));
    EXPECT_EQ(NL::quiet_NaN().__x, static_cast<uint8_t>(0x80));
    EXPECT_EQ(NL::signaling_NaN().__x, static_cast<uint8_t>(0x80));
}
