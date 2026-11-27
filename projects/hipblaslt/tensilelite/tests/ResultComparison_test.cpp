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
#include <tuple>
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
using TensileLite::Client::InvalidComparison;
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

template <typename Tuple>
struct ResultComparisonTest : public ::testing::Test
{
    using TypeParam = typename std::tuple_element<0, Tuple>::type;
};

// Wrap each type in std::tuple so gtest emits typeinfo for _Float16 (Half).
using SupportedTypes = ::testing::Types<std::tuple<float>,
                                        std::tuple<double>,
                                        std::tuple<std::complex<float>>,
                                        std::tuple<std::complex<double>>,
                                        std::tuple<Half>,
                                        std::tuple<BFloat16>,
                                        std::tuple<Float8>,
                                        std::tuple<BFloat8>,
                                        std::tuple<Float8_fnuz>,
                                        std::tuple<BFloat8_fnuz>,
                                        std::tuple<int8_t>,
                                        std::tuple<int32_t>>;

TYPED_TEST_SUITE(ResultComparisonTest, SupportedTypes);

// Test: mismatches present, printing enabled — output should be identical.
TYPED_TEST(ResultComparisonTest, WithErrors_SameOutput)
{
    using T = typename TestFixture::TypeParam;
    auto ref = makeVec<T>({1, 2, 3, 4, 5});
    auto res = makeVec<T>({1, 9, 3, 8, 5});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_NE(oldOutput, ""); // should have printed something
}

// Test: mismatches with printValids=true — output should be identical.
TYPED_TEST(ResultComparisonTest, WithErrors_PrintValids_SameOutput)
{
    using T = typename TestFixture::TypeParam;
    auto ref = makeVec<T>({1, 2, 3, 4});
    auto res = makeVec<T>({1, 9, 3, 4});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), true, 10, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), true, 10, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_NE(oldOutput, "");
}

// Test: no mismatches — both flows should produce no output.
TYPED_TEST(ResultComparisonTest, NoErrors_SameOutput)
{
    using T = typename TestFixture::TypeParam;
    auto ref = makeVec<T>({1, 2, 3});
    auto res = makeVec<T>({1, 2, 3});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), false, 10, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_EQ(oldOutput, "");
}

// Test: mismatches present but printMax=0 — both flows should produce no output.
TYPED_TEST(ResultComparisonTest, PrintMaxZero_SameOutput)
{
    using T = typename TestFixture::TypeParam;
    auto ref = makeVec<T>({1, 2, 3});
    auto res = makeVec<T>({1, 9, 3});

    auto oldOutput = runSinglePassFlow(ref.data(), res.data(), ref.size(), false, 0, -1.0);
    auto newOutput = runTwoPassFlow(ref.data(), res.data(), ref.size(), false, 0, -1.0);

    EXPECT_EQ(oldOutput, newOutput);
    EXPECT_EQ(oldOutput, "");
}

// Test: mismatches present but printMax=0 — both comparators should still fail.
TYPED_TEST(ResultComparisonTest, PrintMaxZero_MismatchStillFails)
{
    using T = typename TestFixture::TypeParam;
    auto ref = makeVec<T>({1, 2, 3});
    auto res = makeVec<T>({1, 9, 3});

    PointwiseComparison<T>     slow(false, 0, false, -1.0);
    FastPointwiseComparison<T> fast(false, -1.0);
    for(size_t i = 0; i < ref.size(); i++)
    {
        slow(ref[i], res[i], i, i);
        fast(ref[i], res[i], i, i);
    }
    slow.report();
    fast.report();

    EXPECT_TRUE(slow.error());
    EXPECT_TRUE(fast.error());
}

namespace
{
    size_t countOccurrences(const std::string& text, const std::string& word)
    {
        size_t count = 0;
        for(size_t pos = text.find(word); pos != std::string::npos; pos = text.find(word, pos + 1))
            count++;
        return count;
    }
} // namespace

// Test: printMax caps the index lines of each guard region; every overwritten value still counts.
TYPED_TEST(ResultComparisonTest, InvalidComparison_PrintMaxCapsIndexLines)
{
    using T = typename TestFixture::TypeParam;
    const T              written = fromInt<T>(1);
    InvalidComparison<T> compare(2, true);

    auto output = captureStdout([&]() {
        for(size_t i = 0; i < 5; i++)
        {
            compare.before(written, i, 5);
            compare.inside(written, i, 5);
            compare.after(written, i, 5);
        }
        compare.report();
    });

    EXPECT_EQ(countOccurrences(output, "Index "), 6u);
    EXPECT_EQ(countOccurrences(output, "found 5 errors in 5 values checked."), 3u);
    EXPECT_TRUE(compare.error());
}

// Test: printMax=0 prints nothing for the guard regions and still fails.
TYPED_TEST(ResultComparisonTest, InvalidComparison_PrintMaxZero_StillFails)
{
    using T = typename TestFixture::TypeParam;
    const T              written = fromInt<T>(1);
    InvalidComparison<T> compare(0, false);

    auto output = captureStdout([&]() {
        compare.before(written, 0, 1);
        compare.inside(written, 0, 1);
        compare.after(written, 0, 1);
        compare.report();
    });

    EXPECT_EQ(output, "");
    EXPECT_TRUE(compare.error());
}
