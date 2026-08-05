// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/adapters/hipblaslt/HipblasltDataInitialization.hpp>
#include <roc/host_validation/adapters/hipblaslt/HipblasltReferenceGemm.hpp>
#include <roc/host_validation/adapters/hipblaslt/hipblaslt_init.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <span>
#include <vector>

TEST(HostValidationDataInitializationBridge, GeneratesComplexTrigonometricValues)
{
    std::array<std::complex<float>, 4> values{};
    roc::host_validation::hipblaslt_adapter::initialize(std::span<std::complex<float>>(values),
                                                        hipblaslt_initialization::trig_float,
                                                        roc::host_validation::DataPattern::Sine);

    for(size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_FLOAT_EQ(values[index].real(), std::sin(static_cast<float>(index)));
        EXPECT_FLOAT_EQ(values[index].imag(), std::cos(static_cast<float>(index)));
    }
}

TEST(HostValidationDataInitializationBridge, CounterBasedGenerationIsRepeatable)
{
    std::array<float, 16> first{};
    std::array<float, 16> second{};
    roc::host_validation::hipblaslt_adapter::initialize(std::span<float>(first),
                                                        hipblaslt_initialization::norm_dist);
    roc::host_validation::hipblaslt_adapter::initialize(std::span<float>(second),
                                                        hipblaslt_initialization::norm_dist);
    EXPECT_EQ(first, second);
}

TEST(HostValidationDataInitializationBridge, LegacyHostEntryPointsUseTensorLayouts)
{
    using Complex = std::complex<float>;
    std::array<Complex, 8> values;
    values.fill(Complex(-99, -99));

    hipblaslt_init_sin(values.data(), 2, 2, 3);
    EXPECT_EQ(values[0], Complex(std::sin(0.0f), std::cos(0.0f)));
    EXPECT_EQ(values[1], Complex(std::sin(1.0f), std::cos(1.0f)));
    EXPECT_EQ(values[3], Complex(std::sin(2.0f), std::cos(2.0f)));
    EXPECT_EQ(values[4], Complex(std::sin(3.0f), std::cos(3.0f)));
    EXPECT_EQ(values[2], Complex(-99, -99));

    hipblaslt_init_zero(values.data(), 2, 2, 3);
    EXPECT_EQ(values[0], Complex(0, 0));
    EXPECT_EQ(values[1], Complex(0, 0));
    EXPECT_EQ(values[3], Complex(0, 0));
    EXPECT_EQ(values[4], Complex(0, 0));
    EXPECT_EQ(values[2], Complex(-99, -99));
}

TEST(HostValidationCblasBridge, FloatGemm)
{
    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    std::array<float, 4>       d{1, 1, 1, 1};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    2,
                                    2,
                                    3,
                                    2,
                                    a.data(),
                                    2,
                                    b.data(),
                                    3,
                                    3,
                                    d.data(),
                                    2,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], 2 * 58 + 3);
    EXPECT_FLOAT_EQ(d[1], 2 * 139 + 3);
    EXPECT_FLOAT_EQ(d[2], 2 * 64 + 3);
    EXPECT_FLOAT_EQ(d[3], 2 * 154 + 3);
}

TEST(HostValidationCblasBridge, MixedHalfInputs)
{
    const std::array<hipblasLtHalf, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<hipblasLtHalf, 6> b{7, 9, 11, 8, 10, 12};
    std::array<float, 4>               d{};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    2,
                                    2,
                                    3,
                                    1,
                                    a.data(),
                                    2,
                                    b.data(),
                                    3,
                                    0,
                                    d.data(),
                                    2,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1,
                                    false,
                                    false,
                                    HIP_R_16F,
                                    HIP_R_16F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_16F,
                                    HIP_R_16F);

    EXPECT_FLOAT_EQ(d[0], 58);
    EXPECT_FLOAT_EQ(d[1], 139);
    EXPECT_FLOAT_EQ(d[2], 64);
    EXPECT_FLOAT_EQ(d[3], 154);
}

TEST(HostValidationCblasBridge, ComplexConjugateTranspose)
{
    using Complex = std::complex<float>;

    const std::array<Complex, 2> a{Complex(1, 2), Complex(3, -1)};
    const std::array<Complex, 2> b{Complex(2, -1), Complex(-4, 3)};
    std::array<Complex, 1>       d{Complex(0, 0)};

    hipblaslt_reference_gemm<Complex>(HIPBLAS_OP_C,
                                      HIPBLAS_OP_N,
                                      1,
                                      1,
                                      2,
                                      Complex(1, 0),
                                      a.data(),
                                      2,
                                      b.data(),
                                      2,
                                      Complex(0, 0),
                                      d.data(),
                                      1,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      Complex(1, 0),
                                      false,
                                      false,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F);

    const Complex expected = std::conj(a[0]) * b[0] + std::conj(a[1]) * b[1];
    EXPECT_FLOAT_EQ(d[0].real(), expected.real());
    EXPECT_FLOAT_EQ(d[0].imag(), expected.imag());
}

TEST(HostValidationCblasBridge, LargeProblemUsesAcceleratedBackend)
{
    constexpr int64_t  m = 601;
    std::vector<float> a(m);
    for(int64_t row = 0; row < m; ++row)
        a[row] = static_cast<float>(row % 7);
    const std::array<float, 1> b{2};
    std::vector<float>         d(m, 1);

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    m,
                                    1,
                                    1,
                                    3,
                                    a.data(),
                                    m,
                                    b.data(),
                                    1,
                                    4,
                                    d.data(),
                                    m,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    for(int64_t row = 0; row < m; ++row)
        EXPECT_FLOAT_EQ(d[row], 6 * a[row] + 4);
}
