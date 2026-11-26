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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// The CPU reduction reference sorts the modes of C and D by the modes of A. A mode of C that A
// does not carry cannot be sorted, and the reference must report that with a status.

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

#include <vector>

#include "reduction/reduction_cpu_reference.hpp"

namespace
{
    // Reduces a {m,k} FP32 tensor of ones over k into an output tensor of extent 4 whose mode is
    // `outputMode`, and returns the status of the CPU reference.
    hiptensorStatus_t reduceWithOutputMode(int32_t outputMode, std::vector<float>& d)
    {
        std::vector<int32_t> modeA{'m', 'k'};
        std::vector<int64_t> extentA{4, 3};
        std::vector<int32_t> modeC{outputMode};
        std::vector<int64_t> extentC{4};

        std::vector<float> a(12, 1.0f);
        std::vector<float> c(4, 0.0f);
        d.assign(4, -1.0f);
        float alpha = 1.0f;
        float beta  = 0.0f;

        hiptensorHandle_t           handle = nullptr;
        hiptensorTensorDescriptor_t descA  = nullptr;
        hiptensorTensorDescriptor_t descC  = nullptr;

        auto status = hiptensorCreate(&handle);
        if(status == HIPTENSOR_STATUS_SUCCESS)
        {
            status = hiptensorCreateTensorDescriptor(
                handle, &descA, modeA.size(), extentA.data(), nullptr, HIPTENSOR_R_32F, 0);
        }
        if(status == HIPTENSOR_STATUS_SUCCESS)
        {
            status = hiptensorCreateTensorDescriptor(
                handle, &descC, modeC.size(), extentC.data(), nullptr, HIPTENSOR_R_32F, 0);
        }
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << "setup failed";
        if(status == HIPTENSOR_STATUS_SUCCESS)
        {
            status = hiptensorReductionReference(&alpha,
                                                 a.data(),
                                                 descA,
                                                 modeA.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 &beta,
                                                 c.data(),
                                                 descC,
                                                 modeC.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 d.data(),
                                                 descC,
                                                 modeC.data(),
                                                 HIPTENSOR_OP_ADD,
                                                 HIPTENSOR_COMPUTE_DESC_32F,
                                                 nullptr);
        }

        if(descC)
        {
            hiptensorDestroyTensorDescriptor(descC);
        }
        if(descA)
        {
            hiptensorDestroyTensorDescriptor(descA);
        }
        if(handle)
        {
            hiptensorDestroy(handle);
        }
        return status;
    }
}

TEST(ReductionModeTest, OutputModeOfInputIsReduced)
{
    std::vector<float> d;
    ASSERT_EQ(reduceWithOutputMode('m', d), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(d, std::vector<float>(4, 3.0f));
}

TEST(ReductionModeTest, OutputModeMissingFromInputReturnsInvalidValue)
{
    std::vector<float> d;
    EXPECT_EQ(reduceWithOutputMode('x', d), HIPTENSOR_STATUS_INVALID_VALUE);
    // The modes are checked before C is copied, so a failed call leaves D untouched.
    EXPECT_EQ(d, std::vector<float>(4, -1.0f));
}
