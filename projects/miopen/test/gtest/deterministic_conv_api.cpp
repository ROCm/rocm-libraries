/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/convolution.hpp>

#include "gtest_common.hpp"

TEST(DeterministicConvApiTest, Test)
{
    miopenConvolutionDescriptor_t conv_desc;

    auto status = miopenCreateConvolutionDescriptor(&conv_desc);
    EXPECT_EQ(status, miopenStatusSuccess);

    status = miopenInitConvolutionDescriptor(conv_desc, miopenConvolutionMode_t::miopenConvolution, 0, 0, 1, 1, 1, 1);
    EXPECT_EQ(status, miopenStatusSuccess);

    const auto& desc = miopen::deref(conv_desc);
    EXPECT_EQ(desc.attribute.deterministic.Get(), 0);   // The default value should be false
    EXPECT_TRUE(!desc.attribute.deterministic);         // Check the bool operator

    const int val = 1;
    status = miopenSetConvolutionAttribute(conv_desc, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, val);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(desc.attribute.deterministic.Get(), 1);
    EXPECT_TRUE(desc.attribute.deterministic);

    int new_val = -1;
    status = miopenGetConvolutionAttribute(conv_desc, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &new_val);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(val, new_val);
}
