/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

// Smoke tests for the public C API accessors (tensor queries, padding-mode
// getters, solver-name round-trip, debug flags). Host-only: exercises the C
// entry points and their exception/bad-parm boundaries, no GPU compute required.

#include <gtest/gtest.h>

#include <miopen/miopen.h>

#include <array>
#include <cstring>
#include <vector>

namespace {

class TensorDescGuard
{
public:
    TensorDescGuard() { EXPECT_EQ(miopenCreateTensorDescriptor(&desc), miopenStatusSuccess); }
    ~TensorDescGuard() { miopenDestroyTensorDescriptor(desc); }
    miopenTensorDescriptor_t get() const { return desc; }

private:
    miopenTensorDescriptor_t desc = nullptr;
};

} // namespace

TEST(CPU_PublicApiAccessors_NONE, TensorQueries)
{
    TensorDescGuard td;
    ASSERT_EQ(miopenSet4dTensorDescriptor(td.get(), miopenFloat, 2, 3, 4, 5), miopenStatusSuccess);

    miopenTensorLayout_t layout = miopenTensorNCHWc4;
    EXPECT_EQ(miopenGetTensorLayout(td.get(), &layout), miopenStatusSuccess);
    EXPECT_EQ(layout, miopenTensorNCHW);

    size_t elementSpace = 0;
    EXPECT_EQ(miopenGetTensorElementSpace(td.get(), &elementSpace), miopenStatusSuccess);
    EXPECT_EQ(elementSpace, static_cast<size_t>(2 * 3 * 4 * 5));

    bool isPacked = false;
    EXPECT_EQ(miopenIsTensorPacked(td.get(), &isPacked), miopenStatusSuccess);
    EXPECT_TRUE(isPacked);

    size_t vectorLength = 0;
    EXPECT_EQ(miopenGetTensorVectorLength(td.get(), &vectorLength), miopenStatusSuccess);
    EXPECT_EQ(vectorLength, 1u);

    // null-arg paths return bad-parm (exercises deref/guard branch).
    EXPECT_EQ(miopenGetTensorLayout(td.get(), nullptr), miopenStatusBadParm);
    EXPECT_EQ(miopenGetTensorElementSpace(td.get(), nullptr), miopenStatusBadParm);
    EXPECT_EQ(miopenIsTensorPacked(td.get(), nullptr), miopenStatusBadParm);
    EXPECT_EQ(miopenGetTensorVectorLength(td.get(), nullptr), miopenStatusBadParm);
    EXPECT_EQ(miopenGetTensorLayout(nullptr, &layout), miopenStatusBadParm);
}

#ifdef MIOPEN_BETA_API
TEST(CPU_PublicApiAccessors_NONE, GetTensorDescriptorV2RoundTrip)
{
    TensorDescGuard td;
    ASSERT_EQ(miopenSet4dTensorDescriptor(td.get(), miopenFloat, 2, 3, 4, 5), miopenStatusSuccess);

    int size = 0;
    ASSERT_EQ(miopenGetTensorDescriptorSize(td.get(), &size), miopenStatusSuccess);
    ASSERT_EQ(size, 4);

    miopenDataType_t dtypeInt = miopenHalf;
    std::vector<int> dimsInt(size, 0);
    std::vector<int> stridesInt(size, 0);
    ASSERT_EQ(miopenGetTensorDescriptor(td.get(), &dtypeInt, dimsInt.data(), stridesInt.data()),
              miopenStatusSuccess);

    miopenDataType_t dtypeV2 = miopenHalf;
    std::vector<size_t> dimsV2(size, 0);
    std::vector<size_t> stridesV2(size, 0);
    ASSERT_EQ(miopenGetTensorDescriptorV2(td.get(), &dtypeV2, dimsV2.data(), stridesV2.data()),
              miopenStatusSuccess);

    EXPECT_EQ(dtypeV2, dtypeInt);
    for(int i = 0; i < size; ++i)
    {
        EXPECT_EQ(dimsV2[i], static_cast<size_t>(dimsInt[i]));
        EXPECT_EQ(stridesV2[i], static_cast<size_t>(stridesInt[i]));
    }

    EXPECT_EQ(miopenGetTensorDescriptorV2(nullptr, &dtypeV2, dimsV2.data(), stridesV2.data()),
              miopenStatusBadParm);
}
#endif

TEST(CPU_PublicApiAccessors_NONE, ConvolutionPaddingMode)
{
    miopenConvolutionDescriptor_t conv = nullptr;
    ASSERT_EQ(miopenCreateConvolutionDescriptor(&conv), miopenStatusSuccess);
    ASSERT_EQ(miopenInitConvolutionDescriptor(conv, miopenConvolution, 1, 1, 1, 1, 1, 1),
              miopenStatusSuccess);

    miopenPaddingMode_t mode = miopenPaddingSame;
    EXPECT_EQ(miopenGetConvolutionPaddingMode(conv, &mode), miopenStatusSuccess);
    EXPECT_TRUE(mode == miopenPaddingDefault || mode == miopenPaddingSame ||
                mode == miopenPaddingValid);

    EXPECT_EQ(miopenGetConvolutionPaddingMode(conv, nullptr), miopenStatusBadParm);
    EXPECT_EQ(miopenGetConvolutionPaddingMode(nullptr, &mode), miopenStatusBadParm);

    miopenDestroyConvolutionDescriptor(conv);
}

TEST(CPU_PublicApiAccessors_NONE, PoolingPaddingMode)
{
    miopenPoolingDescriptor_t pool = nullptr;
    ASSERT_EQ(miopenCreatePoolingDescriptor(&pool), miopenStatusSuccess);
    ASSERT_EQ(miopenSet2dPoolingDescriptor(pool, miopenPoolingMax, 2, 2, 0, 0, 2, 2),
              miopenStatusSuccess);

    miopenPaddingMode_t mode = miopenPaddingSame;
    EXPECT_EQ(miopenGetPoolingPaddingMode(pool, &mode), miopenStatusSuccess);
    EXPECT_TRUE(mode == miopenPaddingDefault || mode == miopenPaddingSame ||
                mode == miopenPaddingValid);

    EXPECT_EQ(miopenGetPoolingPaddingMode(pool, nullptr), miopenStatusBadParm);
    EXPECT_EQ(miopenGetPoolingPaddingMode(nullptr, &mode), miopenStatusBadParm);

    miopenDestroyPoolingDescriptor(pool);
}

TEST(CPU_PublicApiAccessors_NONE, SolverNameRoundTrip)
{
    const char* known = "ConvDirectNaiveConvFwd";

    uint64_t id = 0;
    ASSERT_EQ(miopenGetSolverIdByName(known, &id), miopenStatusSuccess);
    EXPECT_NE(id, 0u);

    std::array<char, 256> buf{};
    ASSERT_EQ(miopenGetSolverName(id, buf.data(), buf.size()), miopenStatusSuccess);
    EXPECT_STREQ(buf.data(), known);

    // buffer-too-small returns bad-parm.
    std::array<char, 4> tiny{};
    EXPECT_EQ(miopenGetSolverName(id, tiny.data(), tiny.size()), miopenStatusBadParm);

    // null / zero-length paths return bad-parm.
    EXPECT_EQ(miopenGetSolverName(id, nullptr, buf.size()), miopenStatusBadParm);
    EXPECT_EQ(miopenGetSolverName(id, buf.data(), 0), miopenStatusBadParm);
    EXPECT_EQ(miopenGetSolverIdByName(nullptr, &id), miopenStatusBadParm);
    EXPECT_EQ(miopenGetSolverIdByName(known, nullptr), miopenStatusBadParm);

    // unknown name yields the invalid id (0).
    uint64_t unknownId = 12345;
    ASSERT_EQ(miopenGetSolverIdByName("ThisSolverDoesNotExist", &unknownId), miopenStatusSuccess);
    EXPECT_EQ(unknownId, 0u);

    // an invalid solver id (the reserved 0 and an out-of-range value) is rejected
    // rather than returning the INVALID_SOLVER_ID_* sentinel string.
    EXPECT_EQ(miopenGetSolverName(0, buf.data(), buf.size()), miopenStatusBadParm);
    EXPECT_EQ(miopenGetSolverName(~uint64_t{0}, buf.data(), buf.size()), miopenStatusBadParm);
}

#ifdef MIOPEN_BETA_API
TEST(CPU_PublicApiAccessors_NONE, DebugFlags)
{
    const std::array<miopenDebugFlag_t, 4> flags = {miopenDebugLoggingQuiet,
                                                    miopenDebugFindEnforceDisable,
                                                    miopenDebugIsWarmupOngoing,
                                                    miopenDebugAlwaysEnableConvDirectNaive};

    for(auto flag : flags)
    {
        bool original = false;
        ASSERT_EQ(miopenGetDebugFlag(flag, &original), miopenStatusSuccess);

        ASSERT_EQ(miopenSetDebugFlag(flag, true), miopenStatusSuccess);
        bool value = false;
        ASSERT_EQ(miopenGetDebugFlag(flag, &value), miopenStatusSuccess);
        EXPECT_TRUE(value);

        ASSERT_EQ(miopenSetDebugFlag(flag, false), miopenStatusSuccess);
        ASSERT_EQ(miopenGetDebugFlag(flag, &value), miopenStatusSuccess);
        EXPECT_FALSE(value);

        // restore original global state so other tests are unaffected.
        ASSERT_EQ(miopenSetDebugFlag(flag, original), miopenStatusSuccess);
    }

    // out-of-range flag returns bad-parm.
    const auto badFlag = static_cast<miopenDebugFlag_t>(999);
    bool v             = false;
    EXPECT_EQ(miopenSetDebugFlag(badFlag, true), miopenStatusBadParm);
    EXPECT_EQ(miopenGetDebugFlag(badFlag, &v), miopenStatusBadParm);
    EXPECT_EQ(miopenGetDebugFlag(miopenDebugLoggingQuiet, nullptr), miopenStatusBadParm);
}
#endif
