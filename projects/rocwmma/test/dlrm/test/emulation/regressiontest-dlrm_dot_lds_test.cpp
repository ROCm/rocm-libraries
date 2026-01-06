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

#include "../dlrm_dot_test.hpp"
#include "detail/dlrm_dot_lds.hpp"
#include "kernel_generator.hpp"
#include "regressiontest-emulation_dlrm_test_params.hpp"

namespace rocwmma
{
    struct TestParams : public DlrmTestParams
    {
        // Types: 32 and 16 bit float
        // Block Sizes: 16 x 16 x 16
        using Base      = DlrmTestParams;
        using Types     = typename Base::DataTypes;
        using TileSizes = typename Base::TileSizes;

        // Lds parameters
        using MappingLds = typename Base::TestMappingLds;

        using KernelParams = typename CombineLists<Types, TileSizes, MappingLds>::Result;

        using GeneratorImpl   = DlrmDotLdsGenerator;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }
    };

} // namespace rocwmma

class EmulationRegressionDlrmDotLdsTestBasic : public rocwmma::DlrmDotTest
{
};

TEST_P(EmulationRegressionDlrmDotLdsTestBasic, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    DlrmKernelTests,
    EmulationRegressionDlrmDotLdsTestBasic,
    ::testing::Combine(::testing::ValuesIn(rocwmma::TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::TestParams::passDirections())));
