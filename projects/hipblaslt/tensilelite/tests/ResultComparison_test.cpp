/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <DataInitialization.hpp>
#include <ResultComparison.hpp>
#include <Tensile/DataTypes.hpp>

using TensileLite::BFloat16;
using TensileLite::BFloat8;
using TensileLite::BFloat8_fnuz;
using TensileLite::Float8;
using TensileLite::Float8_fnuz;
using TensileLite::Half;
using TensileLite::Client::FastPointwiseComparison;
using TensileLite::Client::PointwiseComparison;

namespace
{
    // Captures stdout while executing a callable.
    template <typename F>
    std::string captureStdout(F&& func)
    {
        std::ostringstream captured;
        std::streambuf*    oldBuf = std::cout.rdbuf(captured.rdbuf());

        // RAII guard: restores std::cout's buffer when leaving scope,
        // even if func() throws.
        struct StreamGuard
        {
            std::streambuf* orig;
            ~StreamGuard()
            {
                std::cout.rdbuf(orig);
            }
        } guard{oldBuf};

        func();
        return captured.str();
    }

    // Per-type construction. Going through float covers every type the
    // validator dispatches on (integer, half/bf16/fp8 variants, real fp).
    // std::complex specializes to put the value in the real part.
    template <typename T>
    T fromInt(int v)
    {
        return static_cast<T>(static_cast<float>(v));
    }

    template <>
    double fromInt<double>(int v)
    {
        return static_cast<double>(v);
    }

    template <>
    std::complex<float> fromInt<std::complex<float>>(int v)
    {
        return std::complex<float>(static_cast<float>(v), 0.0f);
    }

    template <>
    std::complex<double> fromInt<std::complex<double>>(int v)
    {
        return std::complex<double>(static_cast<double>(v), 0.0);
    }

    template <typename T>
    std::vector<T> makeVec(std::initializer_list<int> ints)
    {
        std::vector<T> out;
        out.reserve(ints.size());
        for(int v : ints)
            out.push_back(fromInt<T>(v));
        return out;
    }

    // Simulates the OLD single-pass validation flow (before this PR).
    // One PointwiseComparison does both counting and printing, then report().
    template <typename T>
    std::string runSinglePassFlow(const T* ref,
                                  const T* res,
                                  size_t   count,
                                  bool     printValids,
                                  size_t   printMax,
                                  double   threshold)
    {
        return captureStdout([&]() {
            PointwiseComparison<T> compare(printValids, printMax, printMax > 0, threshold);
            for(size_t i = 0; i < count; i++)
                compare(ref[i], res[i], i, i);
            compare.report();
        });
    }

    // Simulates the NEW two-pass validation flow (this PR).
    // FastPointwiseComparison counts, then PointwiseComparison prints if needed.
    template <typename T>
    std::string runTwoPassFlow(const T* ref,
                               const T* res,
                               size_t   count,
                               bool     printValids,
                               size_t   printMax,
                               double   threshold)
    {
        return captureStdout([&]() {
            FastPointwiseComparison<T> fast(printMax > 0, threshold);
            for(size_t i = 0; i < count; i++)
                fast(ref[i], res[i], i, i);

            if((fast.errorCount() > 0 || printValids) && printMax > 0)
            {
                PointwiseComparison<T> print(printValids, printMax, false, threshold);
                for(size_t i = 0; i < count; i++)
                    print(ref[i], res[i], i, i);
            }
            fast.report();
        });
    }
} // namespace

template <typename T>
struct ResultComparisonTest : public ::testing::Test
{
};

using SupportedTypes = ::testing::Types<float,
                                        double,
                                        std::complex<float>,
                                        std::complex<double>,
                                        Half,
                                        BFloat16,
                                        Float8,
                                        BFloat8,
                                        Float8_fnuz,
                                        BFloat8_fnuz,
                                        int8_t,
                                        int32_t>;

TYPED_TEST_SUITE(ResultComparisonTest, SupportedTypes);

// Test: mismatches present, printing enabled — output should be identical.
TYPED_TEST(ResultComparisonTest, WithErrors_SameOutput)
{
    auto ref = makeVec<TypeParam>({1, 2, 3, 4, 5});
    auto res = makeVec<TypeParam>({1, 9, 3, 8, 5});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_NE(oldOutput, ""); // should have printed something
}

// Test: mismatches with printValids=true — output should be identical.
TYPED_TEST(ResultComparisonTest, WithErrors_PrintValids_SameOutput)
{
    auto ref = makeVec<TypeParam>({1, 2, 3, 4});
    auto res = makeVec<TypeParam>({1, 9, 3, 4});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), true, 10, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), true, 10, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_NE(oldOutput, "");
}

// Test: no mismatches — both flows should produce no output.
TYPED_TEST(ResultComparisonTest, NoErrors_SameOutput)
{
    auto ref = makeVec<TypeParam>({1, 2, 3});
    auto res = makeVec<TypeParam>({1, 2, 3});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_EQ(oldOutput, "");
}

// Test: mismatches present but printMax=0 — both flows should produce no output.
TYPED_TEST(ResultComparisonTest, PrintMaxZero_SameOutput)
{
    auto ref = makeVec<TypeParam>({1, 2, 3});
    auto res = makeVec<TypeParam>({1, 9, 3});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), false, 0, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), false, 0, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_EQ(oldOutput, "");
}
