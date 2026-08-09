// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cblas_interface.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <string>
#include <vector>

namespace
{
    uint16_t half_bits(hipblasLtHalf value)
    {
        uint16_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    hipblasLtHalf half_from_bits(uint16_t bits)
    {
        hipblasLtHalf value;
        std::memcpy(&value, &bits, sizeof(bits));
        return value;
    }

    void set_experimental_hb_env(const char* value)
    {
#ifdef _WIN32
        _putenv_s("HIPBLASLT_ENABLE_EXPERIMENTAL_HB", value ? value : "");
#else
        if(value)
            setenv("HIPBLASLT_ENABLE_EXPERIMENTAL_HB", value, 1);
        else
            unsetenv("HIPBLASLT_ENABLE_EXPERIMENTAL_HB");
#endif
    }

    class ExperimentalHBEnvGuard
    {
    public:
        ExperimentalHBEnvGuard()
        {
            if(const char* value = std::getenv("HIPBLASLT_ENABLE_EXPERIMENTAL_HB"))
            {
                m_wasSet = true;
                m_value  = value;
            }
        }

        ~ExperimentalHBEnvGuard()
        {
            set_experimental_hb_env(m_wasSet ? m_value.c_str() : nullptr);
        }

    private:
        bool        m_wasSet = false;
        std::string m_value;
    };
}

TEST(FP16ReferenceContract, OrdinaryFp16UsesFp16Accumulation)
{
    constexpr int64_t k = 256;
    std::vector<hipblasLtHalf> A(k, hipblasLtHalf(1.0f));
    std::vector<hipblasLtHalf> B(k, hipblasLtHalf(1.0f));
    B[0] = hipblasLtHalf(2048.0f);
    hipblasLtHalf output(0.0f);
    hipblasLtHalf scale(1.0f);

    cblas_gemm<hipblasLtHalf>(HIPBLAS_OP_N,
                              HIPBLAS_OP_N,
                              1,
                              1,
                              k,
                              hipblasLtHalf(1.0f),
                              A.data(),
                              1,
                              B.data(),
                              k,
                              hipblasLtHalf(0.0f),
                              &output,
                              1,
                              nullptr,
                              &scale,
                              &scale,
                              hipblasLtHalf(1.0f),
                              false,
                              false,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F);

    EXPECT_EQ(static_cast<float>(output), 2048.0f);
}

TEST(FP16ReferenceContract, ExplicitExactFp16Reference)
{
    constexpr int64_t k = 256;
    std::vector<hipblasLtHalf> A(k, hipblasLtHalf(1.0f));
    std::vector<hipblasLtHalf> B(k, hipblasLtHalf(1.0f));
    B[0] = hipblasLtHalf(2048.0f);
    hipblasLtHalf output(0.0f);

    cblas_gemm<hipblasLtHalf>(HIPBLAS_OP_N,
                              HIPBLAS_OP_N,
                              1,
                              1,
                              k,
                              hipblasLtHalf(1.0f),
                              A.data(),
                              1,
                              B.data(),
                              k,
                              hipblasLtHalf(0.0f),
                              &output,
                              1,
                              nullptr,
                              nullptr,
                              nullptr,
                              hipblasLtHalf(1.0f),
                              false,
                              false,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              HIP_R_16F,
                              false,
                              false,
                              false,
                              true);

    EXPECT_EQ(static_cast<float>(output), 2048.0f);
}

TEST(FP16ReferenceContract, PedanticModeIsEnvironmentGated)
{
    ExperimentalHBEnvGuard guard;
    hipblasLtMatmulDesc_t   descriptor = nullptr;

    set_experimental_hb_env(nullptr);
    EXPECT_EQ(hipblasLtMatmulDescCreate(
                  &descriptor, HIPBLAS_COMPUTE_16F_PEDANTIC, HIP_R_16F),
              HIPBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descriptor, nullptr);

    set_experimental_hb_env("1");
    ASSERT_EQ(hipblasLtMatmulDescCreate(
                  &descriptor, HIPBLAS_COMPUTE_16F_PEDANTIC, HIP_R_16F),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtMatmulDescDestroy(descriptor), HIPBLAS_STATUS_SUCCESS);
}

TEST(NativeFMA16Oracle, FractionalPatterns)
{
    int64_t m = 1, n = 1, k = 3;

    std::vector<hipblasLtHalf> A(k);
    std::vector<hipblasLtHalf> B(k);
    std::vector<hipblasLtHalf> C(1);

    // Pattern 1: Subnormal-to-normal boundary with FTZ emulation
    // 0.000061035f is < 2^-14 (6.103515625e-5f), so it's subnormal.
    // FTZ should flush it to zero before FMA.
    A[0] = hipblasLtHalf(0.000061f);
    B[0] = hipblasLtHalf(1.0f);

    A[1] = hipblasLtHalf(0.000062f); // > 2^-14, normal
    B[1] = hipblasLtHalf(1.0f);

    A[2] = hipblasLtHalf(0.5f);
    B[2] = hipblasLtHalf(0.000061f); // subnormal

    hipblasLtHalf alpha(1.0f);
    hipblasLtHalf beta(0.0f);

    cblas_gemm_native_fma16<hipblasLtHalf>(
        HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);

    // Since FTZ is active on inputs, A[0]*B[0] = 0 and A[2]*B[2] = 0.
    // Result should just be A[1]*B[1] = 0.000062f -> converted to half
    EXPECT_EQ(static_cast<float>(C[0]), static_cast<float>(hipblasLtHalf(0.000062f)));

    // Pattern 2: Rounding tie-points and sticky bits
    // We want to test RNE at a tie point (halfway between two representable values).
    // In fp16, 1.0f is 0x3C00. The next representable is 1.0009765625f (0x3C01).
    // The midpoint is 1.00048828125f = 1.0 + 2^-11.
    // By default, 1.0 + 2^-11 should round to EVEN, which is 1.0 (0x3C00).
    A[0] = hipblasLtHalf(1.0f);
    B[0] = hipblasLtHalf(1.0f);

    A[1] = hipblasLtHalf(0.03125f); // 2^-5
    B[1] = hipblasLtHalf(0.015625f); // 2^-6 => product is 2^-11

    A[2] = hipblasLtHalf(0.0f);
    B[2] = hipblasLtHalf(0.0f);

    cblas_gemm_native_fma16<hipblasLtHalf>(
        HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);
    // 1.0 + 2^-11 rounds to 1.0 (even)
    EXPECT_EQ(static_cast<float>(C[0]), 1.0f);

    // Pattern 3: Sticky-bit dynamics (per-MAC)
    // If the FMA result is slightly more than the midpoint, it must round UP.
    // We achieve this by making A[1]*B[1] slightly larger than 2^-11.
    // 2^-11 + 2^-14 = 0.00048828125 + 0.00006103515625
    A[0] = hipblasLtHalf(1.0f);
    B[0] = hipblasLtHalf(1.0f);

    A[1] = hipblasLtHalf(0.03125f); // 2^-5
    B[1] = hipblasLtHalf(0.017578125f); // 2^-6 + 2^-9

    A[2] = hipblasLtHalf(0.0f);
    B[2] = hipblasLtHalf(0.0f);

    cblas_gemm_native_fma16<hipblasLtHalf>(
        HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);

    // Now it's strictly greater than the midpoint (1.00054931640625 > 1.00048828125), so it must round UP to 1.0009765625f (0x3C01)
    EXPECT_EQ(static_cast<float>(C[0]), 1.0009765625f);
}

TEST(NativeFMA16Oracle, FlushesSubnormalInputsAndOutputs)
{
    const hipblasLtHalf smallest_normal   = half_from_bits(0x0400);
    const hipblasLtHalf largest_subnormal = half_from_bits(0x03ff);
    const hipblasLtHalf one(1.0f);
    hipblasLtHalf       output = largest_subnormal;

    cblas_gemm_native_fma16<hipblasLtHalf>(
        HIPBLAS_OP_N, HIPBLAS_OP_N, 1, 1, 1, one, &largest_subnormal, 1, &one, 1, one, &output, 1);
    EXPECT_EQ(half_bits(output), 0x0000);

    output = hipblasLtHalf(0.0f);
    cblas_gemm_native_fma16<hipblasLtHalf>(HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           1,
                                           1,
                                           1,
                                           one,
                                           &smallest_normal,
                                           1,
                                           &one,
                                           1,
                                           hipblasLtHalf(0.0f),
                                           &output,
                                           1);
    EXPECT_EQ(half_bits(output), 0x0400);
}

TEST(NativeFMA16Oracle, GeneratedScheduleIsStepRounded)
{
    // Sequential: round(2048 + 1) then add -2048 => 0.
    // Generated order: round(2048 + -2048) then add 1 => 1.
    const std::vector<hipblasLtHalf> A{
        hipblasLtHalf(2048.0f), hipblasLtHalf(1.0f), hipblasLtHalf(-2048.0f)};
    const std::vector<hipblasLtHalf> B(3, hipblasLtHalf(1.0f));
    hipblasLtHalf                    sequential(0.0f);
    hipblasLtHalf                    scheduled(0.0f);
    const int64_t                    generated_order[] = {0, 2, 1};

    cblas_gemm_native_fma16<hipblasLtHalf>(HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           1,
                                           1,
                                           3,
                                           hipblasLtHalf(1.0f),
                                           A.data(),
                                           1,
                                           B.data(),
                                           3,
                                           hipblasLtHalf(0.0f),
                                           &sequential,
                                           1);
    cblas_gemm_native_fma16<hipblasLtHalf>(HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           1,
                                           1,
                                           3,
                                           hipblasLtHalf(1.0f),
                                           A.data(),
                                           1,
                                           B.data(),
                                           3,
                                           hipblasLtHalf(0.0f),
                                           &scheduled,
                                           1,
                                           generated_order,
                                           3);
    EXPECT_EQ(static_cast<float>(sequential), 0.0f);
    EXPECT_EQ(static_cast<float>(scheduled), 1.0f);
}

TEST(NativeFMA16Oracle, SupportsTransposeAndValidatesSchedule)
{
    // A is 2x2 column-major. op(A)=transpose(A); select row zero dot B.
    const std::vector<hipblasLtHalf> A{
        hipblasLtHalf(1), hipblasLtHalf(2), hipblasLtHalf(3), hipblasLtHalf(4)};
    const std::vector<hipblasLtHalf> B{hipblasLtHalf(5), hipblasLtHalf(6)};
    hipblasLtHalf                    C(0);
    cblas_gemm_native_fma16<hipblasLtHalf>(HIPBLAS_OP_T,
                                           HIPBLAS_OP_N,
                                           1,
                                           1,
                                           2,
                                           hipblasLtHalf(1),
                                           A.data(),
                                           2,
                                           B.data(),
                                           2,
                                           hipblasLtHalf(0),
                                           &C,
                                           1);
    EXPECT_EQ(static_cast<float>(C), 17.0f);

    const int64_t invalid[] = {0, 2};
    EXPECT_THROW(cblas_gemm_native_fma16<hipblasLtHalf>(HIPBLAS_OP_N,
                                                        HIPBLAS_OP_N,
                                                        1,
                                                        1,
                                                        2,
                                                        hipblasLtHalf(1),
                                                        A.data(),
                                                        1,
                                                        B.data(),
                                                        2,
                                                        hipblasLtHalf(0),
                                                        &C,
                                                        1,
                                                        invalid,
                                                        2),
                 std::out_of_range);
}
