/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

// Checks of the helpers the contraction tests use to validate and report results:
// compareContractionOutput must look at every element of the output, both parts of every
// complex element included, and the GFLOP counts must follow the modes of the problem
// rather than assume an order. The expected counts are worked out by hand from the extents.
//
// The second test suite also checks that the shared test main passes Google Test flags to
// Google Test: ctest runs this binary with --gtest_filter=ContractionFlopsTest.* and fails
// the run if a ContractionOutputCompareTest case shows up in the output.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "contraction_test_utils.hpp"

namespace
{
    struct DeviceBuffer
    {
        void* ptr = nullptr;
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
    };

    template <typename T>
    void upload(DeviceBuffer& buffer, std::vector<T> const& host)
    {
        ASSERT_EQ(hipMalloc(&buffer.ptr, host.size() * sizeof(T)), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(buffer.ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
            hipSuccess);
    }

    // Small integers, so a change of 1 in any value is a large relative error.
    template <typename T>
    std::vector<T> integerValues(std::size_t count)
    {
        std::vector<T> values(count);
        for(std::size_t i = 0; i < count; i++)
        {
            values[i] = static_cast<T>(static_cast<int>(i % 7) - 3);
        }
        return values;
    }

    constexpr double tolerance = 1e-5;

    // Compares `elements` elements of `dataType`, stored as `scalarsPerElement` values of T
    // each, with one value of the output changed by `delta` (no change for a negative index).
    template <typename T>
    bool compareWithChangedValue(hiptensorDataType_t          dataType,
                                 hiptensorComputeDescriptor_t computeType,
                                 std::size_t                  elements,
                                 std::size_t                  scalarsPerElement,
                                 int64_t                      changedScalar,
                                 T                            delta)
    {
        auto reference = integerValues<T>(elements * scalarsPerElement);
        auto output    = reference;
        if(changedScalar >= 0)
        {
            output[static_cast<std::size_t>(changedScalar)] += delta;
        }

        DeviceBuffer devOutput, devReference;
        upload(devOutput, output);
        upload(devReference, reference);
        if(::testing::Test::HasFatalFailure())
        {
            return true; // the failure is already reported
        }

        return hiptensor::compareContractionOutput(
                   dataType, devOutput.ptr, devReference.ptr, elements, computeType, tolerance)
            .first;
    }

    constexpr std::size_t elements = 1000;

} // namespace

TEST(ContractionOutputCompareTest, EqualOutputsMatch)
{
    EXPECT_TRUE(compareWithChangedValue<float>(
        HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, elements, 1, -1, 0.0f));
    EXPECT_TRUE(compareWithChangedValue<double>(
        HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F, elements, 1, -1, 0.0));
    EXPECT_TRUE(compareWithChangedValue<float>(
        HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, elements, 2, -1, 0.0f));
    EXPECT_TRUE(compareWithChangedValue<double>(
        HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F, elements, 2, -1, 0.0));
}

TEST(ContractionOutputCompareTest, RealMismatchInLastElementIsDetected)
{
    EXPECT_FALSE(compareWithChangedValue<float>(
        HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, elements, 1, elements - 1, 1.0f));
    EXPECT_FALSE(compareWithChangedValue<double>(
        HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F, elements, 1, elements - 1, 1.0));
}

// A complex tensor of N elements holds 2N real values. Comparing only N of them leaves the
// second half of the tensor unchecked.
TEST(ContractionOutputCompareTest, ComplexMismatchInSecondHalfIsDetected)
{
    for(int64_t changed : {int64_t(elements), int64_t(elements + 1), int64_t(2 * elements - 1)})
    {
        EXPECT_FALSE(compareWithChangedValue<float>(
            HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, elements, 2, changed, 1.0f))
            << "c32 value " << changed;
        EXPECT_FALSE(compareWithChangedValue<double>(
            HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F, elements, 2, changed, 1.0))
            << "c64 value " << changed;
    }
}

TEST(ContractionOutputCompareTest, ComplexNaNInLastElementIsDetected)
{
    EXPECT_FALSE(compareWithChangedValue<float>(HIPTENSOR_C_32F,
                                                HIPTENSOR_COMPUTE_DESC_C32F,
                                                elements,
                                                2,
                                                2 * elements - 1,
                                                std::numeric_limits<float>::quiet_NaN()));
    EXPECT_FALSE(compareWithChangedValue<double>(HIPTENSOR_C_64F,
                                                 HIPTENSOR_COMPUTE_DESC_C64F,
                                                 elements,
                                                 2,
                                                 2 * elements - 1,
                                                 std::numeric_limits<double>::quiet_NaN()));
}

TEST(ContractionOutputCompareTest, ComplexMismatchInFirstHalfIsDetected)
{
    EXPECT_FALSE(compareWithChangedValue<float>(
        HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, elements, 2, 1, 1.0f));
    EXPECT_FALSE(compareWithChangedValue<double>(
        HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F, elements, 2, 1, 1.0));
}

// D[m,n] = A[m,k] * B[n,k], M = 4, N = 3, K = 5.
TEST(ContractionFlopsTest, CanonicalReal)
{
    EXPECT_DOUBLE_EQ(hiptensor::contractionGFlops(
                         {'m', 'k'}, {4, 5}, {'n', 'k'}, {3, 5}, HIPTENSOR_COMPUTE_DESC_32F),
                     2.0 * 4 * 3 * 5 * 1e-9);
}

// A's modes interleave free and contracted modes: A[m0,k0,m1,k1], B[n0,n1,k0,k1],
// D[m0,m1,n0,n1]. M = 2 * 7, N = 3 * 4, K = 5 * 6. Reading A's last two extents as K gives
// 7 * 6 instead.
TEST(ContractionFlopsTest, InterleavedModes)
{
    EXPECT_DOUBLE_EQ(hiptensor::contractionGFlops({'a', 'x', 'b', 'y'},
                                                  {2, 5, 7, 6},
                                                  {'p', 'q', 'x', 'y'},
                                                  {3, 4, 5, 6},
                                                  HIPTENSOR_COMPUTE_DESC_32F),
                     2.0 * (2 * 7) * (3 * 4) * (5 * 6) * 1e-9);
}

// The batch mode l is carried by A, B and D and counts once: 2 * M * N * K * L.
TEST(ContractionFlopsTest, BatchModeCountsOnce)
{
    EXPECT_DOUBLE_EQ(
        hiptensor::contractionGFlops(
            {'m', 'k', 'l'}, {4, 5, 6}, {'n', 'k', 'l'}, {3, 5, 6}, HIPTENSOR_COMPUTE_DESC_64F),
        2.0 * 4 * 3 * 5 * 6 * 1e-9);
}

// A complex multiply-add is four real multiplies and four real adds.
TEST(ContractionFlopsTest, ComplexCountsEightPerPoint)
{
    EXPECT_DOUBLE_EQ(hiptensor::contractionGFlops(
                         {'m', 'k'}, {4, 5}, {'n', 'k'}, {3, 5}, HIPTENSOR_COMPUTE_DESC_C32F),
                     8.0 * 4 * 3 * 5 * 1e-9);
    EXPECT_DOUBLE_EQ(hiptensor::contractionGFlops(
                         {'m', 'k'}, {4, 5}, {'n', 'k'}, {3, 5}, HIPTENSOR_COMPUTE_DESC_C64F),
                     8.0 * 4 * 3 * 5 * 1e-9);
}

// E[m,s] = A[m,k] * B[n,k] * C[s,n], M = 4, K = 5, N = 3, S = 6, run as T[m,n] = A * B
// (2 * M * N * K) and E = T * C (2 * M * S * N).
TEST(ContractionFlopsTest, TrinaryCountsBothSteps)
{
    EXPECT_DOUBLE_EQ(hiptensor::trinaryContractionGFlops({'m', 'k'},
                                                         {4, 5},
                                                         {'n', 'k'},
                                                         {3, 5},
                                                         {'s', 'n'},
                                                         {6, 3},
                                                         {'m', 's'},
                                                         HIPTENSOR_COMPUTE_DESC_32F),
                     (2.0 * 4 * 3 * 5 + 2.0 * 4 * 6 * 3) * 1e-9);
}

// The mode l carried by A, B and E is not contracted in the first step, so T[m,l,n] keeps it:
// 2 * M * L * N * K, then 2 * M * L * S * N.
TEST(ContractionFlopsTest, TrinaryKeepsModesOfTheOutput)
{
    EXPECT_DOUBLE_EQ(hiptensor::trinaryContractionGFlops({'m', 'l', 'k'},
                                                         {4, 2, 5},
                                                         {'n', 'k', 'l'},
                                                         {3, 5, 2},
                                                         {'s', 'n'},
                                                         {6, 3},
                                                         {'m', 's', 'l'},
                                                         HIPTENSOR_COMPUTE_DESC_32F),
                     (2.0 * 4 * 2 * 3 * 5 + 2.0 * 4 * 2 * 6 * 3) * 1e-9);
}
