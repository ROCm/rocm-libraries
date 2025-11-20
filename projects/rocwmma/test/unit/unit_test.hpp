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

#ifndef ROCWMMA_UNIT_TEST_BASE_HPP
#define ROCWMMA_UNIT_TEST_BASE_HPP

#include <gtest/gtest.h>

#include "unit_kernel_base.hpp"
#include "unit_test_params.hpp"

namespace rocwmma
{

    struct UnitTest
        : public ::testing::TestWithParam<std::tuple<typename UnitTestParams::KernelT,
                                                     typename UnitTestParams::ThreadBlockT,
                                                     typename UnitTestParams::ProblemSizeT,
                                                     typename UnitTestParams::Param1T,
                                                     typename UnitTestParams::Param2T>>
    {
        using Base = ::testing::TestWithParam<std::tuple<typename UnitTestParams::KernelT,
                                                         typename UnitTestParams::ThreadBlockT,
                                                         typename UnitTestParams::ProblemSizeT,
                                                         typename UnitTestParams::Param1T,
                                                         typename UnitTestParams::Param2T>>;

        void SetUp() override
        {
            // Construct ProblemParams from
            // incoming gtest parameterization
            auto param       = Base::GetParam();
            auto kernel      = std::get<0>(param);
            auto threadBlock = std::get<1>(param);
            auto problemSize = std::get<2>(param);
            auto param1      = std::get<3>(param);
            auto param2      = std::get<4>(param);

            ProblemParams params = {threadBlock, problemSize, param1, param2};

            // Walk through kernel workflow
            kernel->setup(params);

            // Mark skipped tests in GTest
            if(!kernel->runFlag())
            {
                std::cout << *kernel << std::endl;
                GTEST_SKIP();
            }
        }

        virtual void RunKernel()
        {
            // Construct ProblemParams from
            // incoming gtest parameterization
            auto param  = Base::GetParam();
            auto kernel = std::get<0>(param);

            kernel->exec();
            kernel->validateResults();
            kernel->reportResults();

            // Mark test failures in GTest
            EXPECT_TRUE(kernel->validationResult());
        }

        void TearDown() override
        {
            // Construct ProblemParams from
            // incoming gtest parameterization
            auto param  = Base::GetParam();
            auto kernel = std::get<0>(param);
            kernel->tearDown();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_UNIT_TEST_BASE_HPP
