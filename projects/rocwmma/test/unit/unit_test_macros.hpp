/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_UNIT_TEST_MACROS_HPP
#define ROCWMMA_UNIT_TEST_MACROS_HPP

#include "common.hpp"

#define ROCWMMA_EXPECT_EQUAL(LHS, RHS, DEBUG_ON_FAIL) \
    rocwmma::expectEqual(LHS, RHS, DEBUG_ON_FAIL, __FILE__, __LINE__);

#define ROCWMMA_EXPECT_NOT_EQUAL(LHS, RHS, DEBUG_ON_FAIL) \
    rocwmma::expectNotEqual(LHS, RHS, DEBUG_ON_FAIL, __FILE__, __LINE__);


///
/// Unit test suite definition
/// @params
/// TestClassName: name of the unit test class
/// TestParamClassName: name of the params class name of unit test
///
#define ROCWMMA_GENERATE_UNIT_GTEST_SUITE(TestClassName, TestParamsClassName)                 \
    class TestClassName : public rocwmma::UnitTest                                            \
    {                                                                                         \
    };                                                                                        \
                                                                                              \
    TEST_P(TestClassName, RunKernel)                                                          \
    {                                                                                         \
        this->RunKernel();                                                                    \
    }                                                                                         \
                                                                                              \
    INSTANTIATE_TEST_SUITE_P(                                                                 \
        KernelTests,                                                                          \
        TestClassName,                                                                        \
        ::testing::Combine(::testing::ValuesIn(rocwmma::TestParamsClassName::kernels()),      \
                           ::testing::ValuesIn(rocwmma::TestParamsClassName::threadBlocks()), \
                           ::testing::ValuesIn(rocwmma::TestParamsClassName::problemSizes()), \
                           ::testing::ValuesIn(rocwmma::TestParamsClassName::param1s()),      \
                           ::testing::ValuesIn(rocwmma::TestParamsClassName::param2s())));

#endif // ROCWMMA_UNIT_TEST_MACROS_HPP
