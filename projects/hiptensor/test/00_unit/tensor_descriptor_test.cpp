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

// Argument validation of hiptensorCreateTensorDescriptor. No kernel is launched.

#include <cstdint>

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

namespace
{
    class TensorDescriptorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(mHandle != nullptr)
            {
                hiptensorDestroy(mHandle);
            }
        }

        // Creates a descriptor and expects `expected`. On failure, the output descriptor
        // must be left untouched.
        void expectCreate(hiptensorStatus_t expected,
                          uint32_t          numModes,
                          int64_t const*    lens,
                          int64_t const*    strides)
        {
            hiptensorTensorDescriptor_t desc = nullptr;
            EXPECT_EQ(hiptensorCreateTensorDescriptor(
                          mHandle, &desc, numModes, lens, strides, HIPTENSOR_R_32F, 0),
                      expected);
            if(expected == HIPTENSOR_STATUS_SUCCESS)
            {
                EXPECT_NE(desc, nullptr);
            }
            else
            {
                EXPECT_EQ(desc, nullptr);
            }
            if(desc != nullptr)
            {
                hiptensorDestroyTensorDescriptor(desc);
            }
        }

        hiptensorHandle_t mHandle = nullptr;
    };
} // namespace

TEST_F(TensorDescriptorTest, NullDescriptorReturnsNotInitialized)
{
    int64_t const lens[] = {4, 2};
    EXPECT_EQ(
        hiptensorCreateTensorDescriptor(mHandle, nullptr, 2, lens, nullptr, HIPTENSOR_R_32F, 0),
        HIPTENSOR_STATUS_NOT_INITIALIZED);
}

TEST_F(TensorDescriptorTest, NullHandleReturnsNotInitialized)
{
    int64_t const               lens[] = {4, 2};
    hiptensorTensorDescriptor_t desc   = nullptr;
    EXPECT_EQ(hiptensorCreateTensorDescriptor(nullptr, &desc, 2, lens, nullptr, HIPTENSOR_R_32F, 0),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TensorDescriptorTest, NullLensWithModesReturnsInvalidValue)
{
    expectCreate(HIPTENSOR_STATUS_INVALID_VALUE, 2, nullptr, nullptr);
}

TEST_F(TensorDescriptorTest, NonPositiveExtentReturnsInvalidValue)
{
    int64_t const zeroExtent[]     = {4, 0};
    int64_t const negativeExtent[] = {-4, 2};
    int64_t const strides[]        = {1, 4};

    expectCreate(HIPTENSOR_STATUS_INVALID_VALUE, 2, zeroExtent, nullptr);
    expectCreate(HIPTENSOR_STATUS_INVALID_VALUE, 2, negativeExtent, nullptr);
    expectCreate(HIPTENSOR_STATUS_INVALID_VALUE, 2, zeroExtent, strides);
    expectCreate(HIPTENSOR_STATUS_INVALID_VALUE, 2, negativeExtent, strides);
}

TEST_F(TensorDescriptorTest, NegativeStrideReturnsNotSupported)
{
    int64_t const lens[]    = {4, 2};
    int64_t const strides[] = {1, -4};
    expectCreate(HIPTENSOR_STATUS_NOT_SUPPORTED, 2, lens, strides);
}

TEST_F(TensorDescriptorTest, ValidDescriptorsAreCreated)
{
    int64_t const lens[]          = {4, 2};
    int64_t const strides[]       = {1, 4};
    int64_t const gappedStrides[] = {2, 16};
    int64_t const zeroStrides[]   = {1, 0};

    // Without strides, packed strides are derived from the extents.
    expectCreate(HIPTENSOR_STATUS_SUCCESS, 2, lens, nullptr);

    // Packed, gapped, and zero strides.
    expectCreate(HIPTENSOR_STATUS_SUCCESS, 2, lens, strides);
    expectCreate(HIPTENSOR_STATUS_SUCCESS, 2, lens, gappedStrides);
    expectCreate(HIPTENSOR_STATUS_SUCCESS, 2, lens, zeroStrides);

    // A rank-0 (scalar) tensor needs no extents or strides.
    expectCreate(HIPTENSOR_STATUS_SUCCESS, 0, nullptr, nullptr);
    expectCreate(HIPTENSOR_STATUS_SUCCESS, 0, lens, nullptr);
}
