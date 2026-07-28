/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// Regression test for SILOTIGER-709 / ALMIOPEN-2275:
// The naive reference kernel lacked instantiations for 3D INT8 forward convolutions with
// int8_t output (ncdhw and ndhwc layouts). This caused "named symbol not found" failures
// during MIOpenDriver verification even though the fast solver ran successfully.

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases()
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    // Regression for ALMIOPEN-2275: 3D int8 forward, both NCDHW and NDHWC layouts.
    // x: {N, C, D, H, W}, w: {K, C, D, H, W}, type_y=int8 (the previously missing output variant).
    // C is kept small (<=64) so the test data generator (conv_tensor_gen.hpp:num_add <
    // int8_max=127) doesn't overflow during input generation for the int8_t accumulator path.
    using TDP = miopen::unit_tests::TensorDescriptorParams;
    using CDP = miopen::unit_tests::ConvolutionDescriptorParams;
    return std::vector{
        // clang-format off
        // NCDHW layout (default): exercises naive_conv_..._ncdhw_int8_t_int32_t_int8_t_0
        TestCase{{1, 16, 5, 14, 14}, {8, 16, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, miopenInt8, miopenInt8, miopenInt8},
        // NDHWC layout: exercises naive_conv_..._ndhwc_int8_t_int32_t_int8_t_0
        TestCase{TDP{miopenInt8, miopenTensorNDHWC, {1, 16, 5, 14, 14}},
                 TDP{miopenInt8, miopenTensorNDHWC, {8, 16, 1, 1, 1}},
                 miopenInt8,
                 CDP{{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        p.UseCpuRef();
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvNaiveConv3dFwd_I8 = GPU_UnitTestConvSolverFwd_I8;

TEST_P(GPU_UnitTestConvNaiveConv3dFwd_I8, ConvDirectNaiveConvFwd3dInt8)
{
    this->RunTest(miopen::solver::conv::ConvDirectNaiveConvFwd{});
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvNaiveConv3dFwd_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases())));
