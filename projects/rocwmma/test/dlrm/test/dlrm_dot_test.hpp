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

#ifndef DLRM_DOT_TEST_HPP
#define DLRM_DOT_TEST_HPP

#include <gtest/gtest.h>

#include "dlrm_kernel_base.hpp"
#include "dlrm_test_params.hpp"

namespace rocwmma
{
    struct DlrmDotTest
        : public ::testing::TestWithParam<std::tuple<typename DlrmTestParams::KernelT,
                                                     typename DlrmTestParams::ThreadBlockT,
                                                     typename DlrmTestParams::ProblemSizeT,
                                                     typename DlrmTestParams::PassDirectionT>>
    {
        using Base = ::testing::TestWithParam<std::tuple<typename DlrmTestParams::KernelT,
                                                         typename DlrmTestParams::ThreadBlockT,
                                                         typename DlrmTestParams::ProblemSizeT,
                                                         typename DlrmTestParams::PassDirectionT>>;

        void SetUp() override
        {
            // Construct ProblemParams from
            // incoming gtest parameterization
            auto param         = Base::GetParam();
            auto kernel        = std::get<0>(param);
            auto threadBlock   = std::get<1>(param);
            auto problemSize   = std::get<2>(param);
            auto passDirection = std::get<3>(param);

            // Cleanup previously used resources if data types change
            static KernelI* sLastKernelRun = nullptr;
            if(sLastKernelRun && sLastKernelRun->getResource() != kernel->getResource())
            {
                sLastKernelRun->getResource()->reset();
            }
            sLastKernelRun = kernel.get();

            ProblemParams params = {threadBlock, problemSize, passDirection};

            // Walk through kernel workflow
            kernel->setup(params);
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
        }

        virtual void Warmup()
        {
            auto param  = Base::GetParam();
            auto kernel = std::get<0>(param);
            kernel->exec();
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
    // pass enum template values through Base::<name>

} // namespace rocwmma

#endif // DLRM_DOT_TEST_HPP
