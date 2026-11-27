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

// Argument and buffer size checks of the attribute accessors:
// hiptensorOperationDescriptorSetAttribute / GetAttribute, hiptensorPlanPreferenceSetAttribute,
// and hiptensorPlanGetAttribute. No kernel is launched.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

namespace
{
    class AttributeTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);

            // Permutation with a rank-3 output B.
            int64_t const lensA[] = {2, 3, 4};
            int64_t const lensB[] = {4, 2, 3};
            int32_t const modeA[] = {'a', 'b', 'c'};
            int32_t const modeB[] = {'c', 'a', 'b'};
            ASSERT_EQ(createDesc(mDescA, 3, lensA), HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(createDesc(mDescB, 3, lensB), HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hiptensorCreatePermutation(mHandle,
                                                 &mPermutation,
                                                 mDescA,
                                                 modeA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 mDescB,
                                                 modeB,
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);

            // Reduction of A to a rank-1 output D.
            int64_t const lensD[] = {2};
            int32_t const modeD[] = {'a'};
            ASSERT_EQ(createDesc(mDescD, 1, lensD), HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hiptensorCreateReduction(mHandle,
                                               &mReduction,
                                               mDescA,
                                               modeA,
                                               HIPTENSOR_OP_IDENTITY,
                                               mDescD,
                                               modeD,
                                               HIPTENSOR_OP_IDENTITY,
                                               mDescD,
                                               modeD,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);

            ASSERT_EQ(hiptensorCreatePlanPreference(
                          mHandle, &mPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                      HIPTENSOR_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(mPlan != nullptr)
                hiptensorDestroyPlan(mPlan);
            if(mPref != nullptr)
                hiptensorDestroyPlanPreference(mPref);
            if(mReduction != nullptr)
                hiptensorDestroyOperationDescriptor(mReduction);
            if(mPermutation != nullptr)
                hiptensorDestroyOperationDescriptor(mPermutation);
            for(auto desc : {mDescA, mDescB, mDescD})
            {
                if(desc != nullptr)
                    hiptensorDestroyTensorDescriptor(desc);
            }
            if(mHandle != nullptr)
                hiptensorDestroy(mHandle);
        }

        hiptensorStatus_t
            createDesc(hiptensorTensorDescriptor_t& desc, uint32_t numModes, int64_t const* lens)
        {
            return hiptensorCreateTensorDescriptor(
                mHandle, &desc, numModes, lens, nullptr, HIPTENSOR_R_32F, 0);
        }

        hiptensorStatus_t set(hiptensorOperationDescriptor_t          desc,
                              hiptensorOperationDescriptorAttribute_t attr,
                              void const*                             buf,
                              size_t                                  size)
        {
            return hiptensorOperationDescriptorSetAttribute(mHandle, desc, attr, buf, size);
        }

        hiptensorStatus_t get(hiptensorOperationDescriptor_t          desc,
                              hiptensorOperationDescriptorAttribute_t attr,
                              void*                                   buf,
                              size_t                                  size)
        {
            return hiptensorOperationDescriptorGetAttribute(mHandle, desc, attr, buf, size);
        }

        // The scalar attributes keep the values they were given.
        void expectScalarAttributes(hiptensorOperationDescriptor_t desc,
                                    uint32_t                       tag,
                                    float                          flops,
                                    float                          movedBytes)
        {
            uint32_t            gotTag        = ~tag;
            hiptensorDataType_t gotScalarType = HIPTENSOR_R_64F;
            float               gotFlops = -1.0f, gotMovedBytes = -1.0f;
            EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &gotTag, sizeof(gotTag)),
                      HIPTENSOR_STATUS_SUCCESS);
            EXPECT_EQ(gotTag, tag);
            EXPECT_EQ(get(desc,
                          HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                          &gotScalarType,
                          sizeof(gotScalarType)),
                      HIPTENSOR_STATUS_SUCCESS);
            EXPECT_EQ(gotScalarType, HIPTENSOR_R_32F);
            EXPECT_EQ(get(desc, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS, &gotFlops, sizeof(gotFlops)),
                      HIPTENSOR_STATUS_SUCCESS);
            EXPECT_EQ(gotFlops, flops);
            EXPECT_EQ(get(desc,
                          HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES,
                          &gotMovedBytes,
                          sizeof(gotMovedBytes)),
                      HIPTENSOR_STATUS_SUCCESS);
            EXPECT_EQ(gotMovedBytes, movedBytes);
        }

        // Reads PADDING_LEFT or PADDING_RIGHT of an output with `numModes` modes.
        std::vector<uint32_t> getPadding(hiptensorOperationDescriptor_t          desc,
                                         hiptensorOperationDescriptorAttribute_t attr,
                                         size_t                                  numModes)
        {
            std::vector<uint32_t> padding(numModes, 99u);
            EXPECT_EQ(get(desc, attr, padding.data(), padding.size() * sizeof(uint32_t)),
                      HIPTENSOR_STATUS_SUCCESS);
            return padding;
        }

        hiptensorHandle_t              mHandle      = nullptr;
        hiptensorTensorDescriptor_t    mDescA       = nullptr;
        hiptensorTensorDescriptor_t    mDescB       = nullptr;
        hiptensorTensorDescriptor_t    mDescD       = nullptr;
        hiptensorOperationDescriptor_t mPermutation = nullptr;
        hiptensorOperationDescriptor_t mReduction   = nullptr;
        hiptensorPlanPreference_t      mPref        = nullptr;
        hiptensorPlan_t                mPlan        = nullptr;
    };
} // namespace

// ---- hiptensorOperationDescriptorSetAttribute / GetAttribute ----

TEST_F(AttributeTest, OperationDescriptorScalarAttributesRoundTrip)
{
    uint32_t const tag        = 7u;
    float const    flops      = 1.5f;
    float const    movedBytes = 2.5f;
    EXPECT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, sizeof(flops)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(set(mPermutation,
                  HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES,
                  &movedBytes,
                  sizeof(movedBytes)),
              HIPTENSOR_STATUS_SUCCESS);
    expectScalarAttributes(mPermutation, tag, flops, movedBytes);
}

TEST_F(AttributeTest, OperationDescriptorNullArgumentsAreRejected)
{
    uint32_t tag = 0u;
    EXPECT_EQ(set(nullptr, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(get(nullptr, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorOperationDescriptorSetAttribute(
                  nullptr, mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorOperationDescriptorGetAttribute(
                  nullptr, mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);

    for(auto attr : {HIPTENSOR_OPERATION_DESCRIPTOR_TAG,
                     HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT,
                     HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE})
    {
        size_t const size = attr == HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT
                                ? 3 * sizeof(uint32_t)
                                : sizeof(uint32_t);
        EXPECT_EQ(set(mPermutation, attr, nullptr, size), HIPTENSOR_STATUS_INVALID_VALUE)
            << "attribute " << attr;
        EXPECT_EQ(get(mPermutation, attr, nullptr, size), HIPTENSOR_STATUS_INVALID_VALUE)
            << "attribute " << attr;
    }
}

TEST_F(AttributeTest, OperationDescriptorUnknownAttributeIsNotSupported)
{
    auto const unknown = static_cast<hiptensorOperationDescriptorAttribute_t>(64);
    uint32_t   value   = 0u;
    EXPECT_EQ(set(mPermutation, unknown, &value, sizeof(value)), HIPTENSOR_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(get(mPermutation, unknown, &value, sizeof(value)), HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// A buffer of another size than the attribute is rejected and copies nothing, so a larger one
// cannot overwrite the attributes stored next to it.
TEST_F(AttributeTest, OperationDescriptorWrongSizeIsRejected)
{
    uint64_t const wide = (uint64_t(HIPTENSOR_R_16F) << 32) | 7u;
    for(size_t size : {size_t(0), size_t(2), sizeof(wide)})
    {
        EXPECT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &wide, size),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << "size " << size;
        uint64_t readBack = 0;
        EXPECT_EQ(get(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &readBack, size),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << "size " << size;
    }

    double const wideValue = 0.25;
    EXPECT_EQ(set(mPermutation,
                  HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE,
                  &wideValue,
                  sizeof(wideValue)),
              HIPTENSOR_STATUS_INVALID_VALUE);

    expectScalarAttributes(mPermutation, 0u, 0.0f, 0.0f);
}

// PADDING_LEFT and PADDING_RIGHT hold one uint32_t per mode of the output tensor: three for
// the permutation's B, one for the reduction's D.
TEST_F(AttributeTest, OperationDescriptorPaddingSizeFollowsOutputRank)
{
    uint32_t const padding[4] = {1u, 2u, 3u, 4u};
    for(auto attr :
        {HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT})
    {
        for(size_t modes : {1u, 2u, 4u})
        {
            EXPECT_EQ(set(mPermutation, attr, padding, modes * sizeof(uint32_t)),
                      HIPTENSOR_STATUS_INVALID_VALUE)
                << "attribute " << attr << ", modes " << modes;
        }
        EXPECT_EQ(set(mReduction, attr, padding, 3 * sizeof(uint32_t)),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << "attribute " << attr;

        // Unset padding reads as zero.
        EXPECT_EQ(getPadding(mPermutation, attr, 3), (std::vector<uint32_t>{0u, 0u, 0u}));
        EXPECT_EQ(getPadding(mReduction, attr, 1), (std::vector<uint32_t>{0u}));
    }
}

// Setting the padding attributes keeps the other attributes, and the left and right padding
// are stored separately.
TEST_F(AttributeTest, OperationDescriptorPaddingRoundTrips)
{
    uint32_t const tag        = 5u;
    float const    flops      = 3.0f;
    float const    movedBytes = 4.0f;
    ASSERT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag)),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, sizeof(flops)),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(set(mPermutation,
                  HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES,
                  &movedBytes,
                  sizeof(movedBytes)),
              HIPTENSOR_STATUS_SUCCESS);

    uint32_t const left[3]  = {1u, 2u, 3u};
    uint32_t const right[3] = {4u, 5u, 6u};
    float const    value    = 0.5f;
    EXPECT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, left, sizeof(left)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, right, sizeof(right)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(
        set(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &value, sizeof(value)),
        HIPTENSOR_STATUS_SUCCESS);

    EXPECT_EQ(getPadding(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, 3),
              (std::vector<uint32_t>{1u, 2u, 3u}));
    EXPECT_EQ(getPadding(mPermutation, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, 3),
              (std::vector<uint32_t>{4u, 5u, 6u}));
    float gotValue = -1.0f;
    EXPECT_EQ(get(mPermutation,
                  HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE,
                  &gotValue,
                  sizeof(gotValue)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(gotValue, value);
    expectScalarAttributes(mPermutation, tag, flops, movedBytes);

    // The reduction's attributes are separate.
    EXPECT_EQ(getPadding(mReduction, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT, 1),
              (std::vector<uint32_t>{0u}));
    expectScalarAttributes(mReduction, 0u, 0.0f, 0.0f);
}

// PADDING_VALUE is one element of the output tensor's data type and reads as zero until set.
TEST_F(AttributeTest, OperationDescriptorPaddingValueDefaultsToZero)
{
    float got = -1.0f;
    EXPECT_EQ(get(mReduction, HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &got, sizeof(got)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(got, 0.0f);
}

// ---- hiptensorPlanPreferenceSetAttribute ----

TEST_F(AttributeTest, PlanPreferenceAttributesOfTheirSizeAreSet)
{
    auto const    autotune = HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;
    auto const    cache    = HIPTENSOR_CACHE_MODE_NONE;
    int32_t const count    = 4;
    auto const    algo     = HIPTENSOR_ALGO_DEFAULT;
    int32_t const rank     = 0;
    auto const    jit      = HIPTENSOR_JIT_MODE_NONE;

    EXPECT_EQ(
        hiptensorPlanPreferenceSetAttribute(
            mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE, &autotune, sizeof(autotune)),
        HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cache, sizeof(cache)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(
        hiptensorPlanPreferenceSetAttribute(
            mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT, &count, sizeof(count)),
        HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_ALGO, &algo, sizeof(algo)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK, &rank, sizeof(rank)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_JIT, &jit, sizeof(jit)),
              HIPTENSOR_STATUS_SUCCESS);

    // The preference is still usable.
    EXPECT_EQ(hiptensorCreatePlan(mHandle, &mPlan, mPermutation, mPref, 0),
              HIPTENSOR_STATUS_SUCCESS);
}

TEST_F(AttributeTest, PlanPreferenceInvalidArgumentsAreRejected)
{
    auto const cache = HIPTENSOR_CACHE_MODE_NONE;
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  nullptr, mPref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cache, sizeof(cache)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  mHandle, nullptr, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cache, sizeof(cache)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(
                  mHandle, mPref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, nullptr, sizeof(cache)),
              HIPTENSOR_STATUS_INVALID_VALUE);

    auto const unknown = static_cast<hiptensorPlanPreferenceAttribute_t>(64);
    EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(mHandle, mPref, unknown, &cache, sizeof(cache)),
              HIPTENSOR_STATUS_NOT_SUPPORTED);

    // A larger buffer would be copied over the attributes stored after this one.
    uint8_t const wide[32] = {};
    for(auto attr : {HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
                     HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE,
                     HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT,
                     HIPTENSOR_PLAN_PREFERENCE_ALGO,
                     HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK,
                     HIPTENSOR_PLAN_PREFERENCE_JIT})
    {
        for(size_t size : {size_t(0), size_t(2), size_t(8), sizeof(wide)})
        {
            EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(mHandle, mPref, attr, wide, size),
                      HIPTENSOR_STATUS_INVALID_VALUE)
                << "attribute " << attr << ", size " << size;
        }
    }
}

// ---- hiptensorPlanGetAttribute ----

TEST_F(AttributeTest, PlanAttributeChecksBufferAndSize)
{
    uint64_t const workspaceSizeLimit = 1024;
    ASSERT_EQ(hiptensorCreatePlan(mHandle, &mPlan, mPermutation, mPref, workspaceSizeLimit),
              HIPTENSOR_STATUS_SUCCESS);

    uint64_t workspace = ~uint64_t(0);
    EXPECT_EQ(hiptensorPlanGetAttribute(
                  mHandle, mPlan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &workspace, sizeof(workspace)),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_LE(workspace, workspaceSizeLimit);

    EXPECT_EQ(hiptensorPlanGetAttribute(
                  nullptr, mPlan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &workspace, sizeof(workspace)),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(
        hiptensorPlanGetAttribute(
            mHandle, nullptr, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &workspace, sizeof(workspace)),
        HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorPlanGetAttribute(
                  mHandle, mPlan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, nullptr, sizeof(workspace)),
              HIPTENSOR_STATUS_INVALID_VALUE);

    uint64_t wide[2] = {7u, 7u};
    for(size_t size : {size_t(0), sizeof(uint32_t), sizeof(wide)})
    {
        EXPECT_EQ(hiptensorPlanGetAttribute(
                      mHandle, mPlan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, wide, size),
                  HIPTENSOR_STATUS_INVALID_VALUE)
            << "size " << size;
    }
    EXPECT_EQ(wide[0], 7u);
    EXPECT_EQ(wide[1], 7u);

    auto const unknown = static_cast<hiptensorPlanAttribute_t>(64);
    EXPECT_EQ(hiptensorPlanGetAttribute(mHandle, mPlan, unknown, &workspace, sizeof(workspace)),
              HIPTENSOR_STATUS_NOT_SUPPORTED);
}
