// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // Shape reported in #12334 and minimum supported channel count.
        TestCase{{2, 32, 16, 16}, {32, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, datatype},

        // YOLO26x depthwise shapes.
        TestCase{{1, 384, 80, 80}, {384, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 384, datatype},
        TestCase{{1, 768, 40, 40}, {768, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 768, datatype},
        TestCase{{1, 384, 20, 20}, {384, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 384, datatype},
        // Strided, unpadded, larger filter and odd-sized variants. The 33x17
        // case yields out_h = 17, a prime, so the row block cannot divide the
        // output height and the last row block runs short.
        TestCase{{2, 128, 33, 17}, {128, 1, 3, 3}, {1, 1}, {2, 2}, {1, 1}, 128, datatype},
        TestCase{{1, 96, 41, 41}, {96, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 96, datatype},
        TestCase{{1, 64, 56, 56}, {64, 1, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 64, datatype},
        // Dilated. Stride 1 is the atrous case, where no input row survives to
        // the next output row and the window is reloaded per row; stride 2 is
        // the case where rows are reused across the slide.
        TestCase{{1, 64, 28, 28}, {64, 1, 3, 3}, {2, 2}, {1, 1}, {2, 2}, 64, datatype},
        TestCase{{1, 64, 28, 28}, {64, 1, 3, 3}, {2, 2}, {2, 2}, {2, 2}, 64, datatype},
        // Asymmetric filter, padding, stride and dilation, so that swapping a
        // height constant for a width one cannot pass unnoticed.
        TestCase{{1, 64, 32, 32}, {64, 1, 5, 3}, {2, 1}, {1, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {1, 1}, {2, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {2, 1}, {1, 1}, {2, 1}, 64, datatype},
        // Largest register window the solver admits: 7x7 needs 105 of the 128
        // float slots kMaxWindowRegs allows.
        TestCase{{1, 64, 24, 24}, {64, 1, 7, 7}, {3, 3}, {1, 1}, {1, 1}, 64, datatype},
        // A single output pixel, over an input narrower than one lane's column
        // span, so every row load takes the bounds-checked path.
        TestCase{{1, 64, 3, 3}, {64, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 64, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx110X | Gpu::gfx115X |
                                                              Gpu::gfx120X);
        return p;
    }();
    return params;
}

float GetWti(std::string_view device_name,
             const miopen::unit_tests::ConvTestCase& test_case,
             const miopen::solver::conv::ConvSolverInterface& solver)
{
    const auto& devices = GetAllKnownDevices();
    const auto device   = std::find_if(devices.begin(), devices.end(), [&](const auto& entry) {
        return entry.second.name == device_name;
    });
    if(device == devices.end())
        throw std::runtime_error("Unknown test device");

    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::Forward);
    auto handle        = MockHandle{device->second, false};
    auto ctx           = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);

    return solver.GetWti(ctx, problem);
}

} // namespace

using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32  = GPU_UnitTestConvSolverFwd_FP32;

using CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE,
       ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DWTI_NONE, Gfx1151Fp16Gate)
{
    using TestCase       = miopen::unit_tests::ConvTestCase;
    constexpr auto worst = miopen::solver::conv::ConvSolverInterface::wti_approximate_worst;

    const auto exact =
        TestCase{{2, 32, 16, 16}, {32, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, miopenHalf};
    const auto batch_one =
        TestCase{{1, 32, 16, 16}, {32, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, miopenHalf};
    const auto bf16 =
        TestCase{{2, 32, 16, 16}, {32, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, miopenBFloat16};
    const auto small =
        TestCase{{2, 32, 15, 15}, {32, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, miopenHalf};
    const auto strided =
        TestCase{{2, 32, 32, 32}, {32, 1, 3, 3}, {1, 1}, {2, 2}, {1, 1}, 32, miopenHalf};

    const auto direct = miopen::solver::conv::ConvDirectDepthwiseFwd2D{};
    const auto gemm   = miopen::solver::conv::GemmFwdRest{};

    const auto exact_wti     = GetWti("gfx1151", exact, direct);
    const auto batch_one_wti = GetWti("gfx1151", batch_one, direct);
    EXPECT_FLOAT_EQ(exact_wti, 0.35f);
    EXPECT_FLOAT_EQ(batch_one_wti, 0.35f);
    EXPECT_GT(exact_wti, GetWti("gfx1151", exact, gemm));
    EXPECT_GT(batch_one_wti, GetWti("gfx1151", batch_one, gemm));

    EXPECT_FLOAT_EQ(GetWti("gfx1150", exact, direct), worst);
    EXPECT_FLOAT_EQ(GetWti("gfx1151", bf16, direct), worst);
    EXPECT_FLOAT_EQ(GetWti("gfx1151", small, direct), worst);
    EXPECT_FLOAT_EQ(GetWti("gfx1151", strided, direct), worst);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

#endif // MIOPEN_BACKEND_HIP
