/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Host-only unit tests for the ULP (units-in-the-last-place) error helpers
// declared in clients/common/include/ulp.hpp. These exercise the pure math /
// dispatch logic and do not require a GPU.

#include <gtest/gtest.h>

#include "ulp.hpp"

#include <cmath>
#include <vector>

namespace
{
    // 2^exp as a double, for building exact ULP steps.
    inline double p2(int exp)
    {
        return std::ldexp(1.0, exp);
    }

    // ------------------------------------------------------------------
    // ulp_mantissa_bits
    // ------------------------------------------------------------------
    TEST(UlpMantissaBits, matches_expected_fraction_widths)
    {
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_64F), 52);
        EXPECT_EQ(ulp_mantissa_bits(HIP_C_64F), 52);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_32F), 23);
        EXPECT_EQ(ulp_mantissa_bits(HIP_C_32F), 23);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_16F), 10);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_16BF), 7);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_8F_E4M3), 3);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_8F_E4M3_FNUZ), 3);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_8F_E5M2), 2);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_8F_E5M2_FNUZ), 2);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_4F_E2M1), 1);
    }

    TEST(UlpMantissaBits, integer_types_are_zero)
    {
        // 0 marks an integer type where one ULP == 1.
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_32I), 0);
        EXPECT_EQ(ulp_mantissa_bits(HIP_R_8I), 0);
    }

    // ------------------------------------------------------------------
    // ulp_distance
    // ------------------------------------------------------------------
    TEST(UlpDistance, identical_values_are_zero)
    {
        EXPECT_DOUBLE_EQ(ulp_distance(1.0, 1.0, 23), 0.0);
        EXPECT_DOUBLE_EQ(ulp_distance(-3.5, -3.5, 23), 0.0);
        EXPECT_DOUBLE_EQ(ulp_distance(0.0, 0.0, 23), 0.0);
        // Early identical-return also covers non-finite equal inputs.
        const double inf = std::numeric_limits<double>::infinity();
        EXPECT_DOUBLE_EQ(ulp_distance(inf, inf, 23), 0.0);
    }

    TEST(UlpDistance, one_step_at_power_of_two_is_one_ulp_f32)
    {
        // At 1.0 the f32 spacing is 2^-23; one step away is exactly 1 ULP.
        EXPECT_DOUBLE_EQ(ulp_distance(1.0, 1.0 + p2(-23), 23), 1.0);
    }

    TEST(UlpDistance, one_step_at_power_of_two_is_one_ulp_f64)
    {
        // At 1.0 the f64 spacing is 2^-52.
        EXPECT_DOUBLE_EQ(ulp_distance(1.0, 1.0 + p2(-52), 52), 1.0);
    }

    TEST(UlpDistance, scales_linearly_with_difference)
    {
        EXPECT_DOUBLE_EQ(ulp_distance(1.0, 1.0 + 4.0 * p2(-23), 23), 4.0);
        EXPECT_DOUBLE_EQ(ulp_distance(1.0, 1.0 + 17.0 * p2(-23), 23), 17.0);
    }

    TEST(UlpDistance, boundary_fix_applies_at_powers_of_two)
    {
        // 2.0 is a power of two: frexp gives mantissa 0.5 / exp 2, and the e--
        // correction makes the spacing 2^-22 (the true ULP just above 2.0),
        // so a single 2^-22 step reads as exactly 1 ULP (not 0.5).
        EXPECT_DOUBLE_EQ(ulp_distance(2.0, 2.0 + p2(-22), 23), 1.0);
        EXPECT_DOUBLE_EQ(ulp_distance(4.0, 4.0 + p2(-21), 23), 1.0);
    }

    TEST(UlpDistance, is_symmetric_in_the_difference_sign)
    {
        // Distance depends on |exact - approx|, so under/over-shoot match.
        const double lo = ulp_distance(1.0, 1.0 - p2(-24), 23);
        const double hi = ulp_distance(1.0, 1.0 + p2(-24), 23);
        EXPECT_DOUBLE_EQ(lo, hi);
    }

    TEST(UlpDistance, large_gap_reports_many_ulps)
    {
        // From 1.0 to 2.0 is 2^23 f32 steps.
        EXPECT_DOUBLE_EQ(ulp_distance(1.0, 2.0, 23), p2(23));
    }

    // ------------------------------------------------------------------
    // ulp_as_double  (widening / promotion)
    // ------------------------------------------------------------------
    TEST(UlpAsDouble, exact_for_float_and_double)
    {
        EXPECT_DOUBLE_EQ(ulp_as_double(1.5f), 1.5);
        EXPECT_DOUBLE_EQ(ulp_as_double(-2.25), -2.25);
    }

    TEST(UlpAsDouble, promotes_half_and_bfloat16_exactly)
    {
        // 1.5 and -4.0 are exactly representable in fp16 and bf16.
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hipblasLtHalf>(1.5f)), 1.5);
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hip_bfloat16>(-4.0f)), -4.0);
    }

    TEST(UlpAsDouble, promotes_fp8_representable_values)
    {
        // 1.0 and 2.0 are representable in every fp8 flavor here.
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hipblaslt_f8>(1.0f)), 1.0);
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hipblaslt_bf8>(2.0f)), 2.0);
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hipblaslt_f8_fnuz>(1.0f)), 1.0);
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hipblaslt_bf8_fnuz>(2.0f)), 2.0);
    }

    TEST(UlpAsDouble, integer_types)
    {
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<int32_t>(7)), 7.0);
        EXPECT_DOUBLE_EQ(ulp_as_double(static_cast<hipblasLtInt8>(5)), 5.0);
    }

    // ------------------------------------------------------------------
    // ulp_accumulate_general  (strided / batched reduction)
    // ------------------------------------------------------------------
    TEST(UlpAccumulate, identical_matrices_report_zero)
    {
        std::vector<float> cpu = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> gpu = cpu;

        double max_ulp = 0.0, sum_ulp = 0.0;
        size_t count = 0;
        ulp_accumulate_general<float>(
            /*M*/ 2, /*N*/ 2, /*lda*/ 2, /*stride*/ 0, cpu.data(), gpu.data(),
            /*batch*/ 1, /*mant_bits*/ 23, max_ulp, sum_ulp, count);

        EXPECT_DOUBLE_EQ(max_ulp, 0.0);
        EXPECT_DOUBLE_EQ(sum_ulp, 0.0);
        EXPECT_EQ(count, 4u);
    }

    TEST(UlpAccumulate, empty_matrix_is_skipped)
    {
        std::vector<float> cpu = {1.0f};
        std::vector<float> gpu = {2.0f};

        double max_ulp = 0.0, sum_ulp = 0.0;
        size_t count = 0;
        ulp_accumulate_general<float>(
            /*M*/ 0, /*N*/ 4, /*lda*/ 1, /*stride*/ 0, cpu.data(), gpu.data(),
            /*batch*/ 1, /*mant_bits*/ 23, max_ulp, sum_ulp, count);

        EXPECT_EQ(count, 0u);
        EXPECT_DOUBLE_EQ(max_ulp, 0.0);
    }

    TEST(UlpAccumulate, ignores_leading_dimension_padding)
    {
        // Column-major 2x2 with lda=3: row index 2 in each column is padding
        // and must not be compared. Fill padding with a huge mismatch.
        const int64_t M = 2, N = 2, lda = 3;
        std::vector<float> cpu(lda * N, 1.0f);
        std::vector<float> gpu(lda * N, 1.0f);

        // Single 1-ULP error at (row=0, col=1): idx = 0 + 1*lda = 3.
        gpu[3] = 1.0f + static_cast<float>(p2(-23));

        // Padding rows (idx 2 and 5) get a large mismatch that should be ignored.
        cpu[2] = 5.0f;   gpu[2] = 999.0f;
        cpu[5] = 5.0f;   gpu[5] = -999.0f;

        double max_ulp = 0.0, sum_ulp = 0.0;
        size_t count = 0;
        ulp_accumulate_general<float>(
            M, N, lda, /*stride*/ 0, cpu.data(), gpu.data(),
            /*batch*/ 1, /*mant_bits*/ 23, max_ulp, sum_ulp, count);

        EXPECT_EQ(count, 4u);                 // only M*N real elements
        EXPECT_DOUBLE_EQ(max_ulp, 1.0);       // the single 1-ULP error
        EXPECT_DOUBLE_EQ(sum_ulp, 1.0);       // everything else identical
    }

    TEST(UlpAccumulate, walks_all_batches_with_stride)
    {
        // 2 batches of a 1x2 matrix (lda=1), stride = lda*N = 2.
        const int64_t M = 1, N = 2, lda = 1, stride = 2, batch = 2;
        std::vector<float> cpu(stride * batch, 1.0f);
        std::vector<float> gpu = cpu;

        // Put a 2-ULP error in the second batch only (base = 1*stride = 2).
        gpu[2 + 0] = 1.0f + 2.0f * static_cast<float>(p2(-23));

        double max_ulp = 0.0, sum_ulp = 0.0;
        size_t count = 0;
        ulp_accumulate_general<float>(
            M, N, lda, stride, cpu.data(), gpu.data(),
            batch, /*mant_bits*/ 23, max_ulp, sum_ulp, count);

        EXPECT_EQ(count, 4u);           // M*N*batch
        EXPECT_DOUBLE_EQ(max_ulp, 2.0);
        EXPECT_DOUBLE_EQ(sum_ulp, 2.0);
    }

    // ------------------------------------------------------------------
    // ulp_check_general  (type dispatch)
    // ------------------------------------------------------------------
    struct CheckResult
    {
        double max_ulp = 0.0;
        double sum_ulp = 0.0;
        size_t count   = 0;
    };

    template <typename T>
    CheckResult run_check(const std::vector<T>& cpu,
                          const std::vector<T>& gpu,
                          int64_t               M,
                          int64_t               N,
                          hipDataType           type)
    {
        CheckResult r;
        ulp_check_general(M,
                          N,
                          /*lda*/ M,
                          /*stride*/ 0,
                          const_cast<T*>(cpu.data()),
                          const_cast<T*>(gpu.data()),
                          /*batch*/ 1,
                          r.max_ulp,
                          r.sum_ulp,
                          r.count,
                          type);
        return r;
    }

    TEST(UlpCheckGeneral, float_identical_is_zero)
    {
        std::vector<float> cpu = {1.0f, 2.0f, 3.0f};
        std::vector<float> gpu = cpu;
        CheckResult        r   = run_check(cpu, gpu, 1, 3, HIP_R_32F);

        EXPECT_DOUBLE_EQ(r.max_ulp, 0.0);
        EXPECT_EQ(r.count, 3u);
    }

    TEST(UlpCheckGeneral, float_one_ulp_error)
    {
        std::vector<float> cpu = {1.0f, 1.0f};
        std::vector<float> gpu = {1.0f, 1.0f + static_cast<float>(p2(-23))};
        CheckResult        r   = run_check(cpu, gpu, 1, 2, HIP_R_32F);

        EXPECT_DOUBLE_EQ(r.max_ulp, 1.0);
        EXPECT_DOUBLE_EQ(r.sum_ulp, 1.0);
        EXPECT_EQ(r.count, 2u);
    }

    TEST(UlpCheckGeneral, double_dispatch)
    {
        std::vector<double> cpu = {1.0, 1.0};
        std::vector<double> gpu = {1.0, 1.0 + p2(-52)};
        CheckResult         r   = run_check(cpu, gpu, 1, 2, HIP_R_64F);

        EXPECT_DOUBLE_EQ(r.max_ulp, 1.0);
        EXPECT_EQ(r.count, 2u);
    }

    TEST(UlpCheckGeneral, half_and_bfloat16_dispatch_identical)
    {
        std::vector<hipblasLtHalf> h_cpu
            = {static_cast<hipblasLtHalf>(1.0f), static_cast<hipblasLtHalf>(2.0f)};
        std::vector<hipblasLtHalf> h_gpu = h_cpu;
        CheckResult                rh    = run_check(h_cpu, h_gpu, 1, 2, HIP_R_16F);
        EXPECT_DOUBLE_EQ(rh.max_ulp, 0.0);
        EXPECT_EQ(rh.count, 2u);

        std::vector<hip_bfloat16> b_cpu
            = {static_cast<hip_bfloat16>(1.0f), static_cast<hip_bfloat16>(2.0f)};
        std::vector<hip_bfloat16> b_gpu = b_cpu;
        CheckResult               rb    = run_check(b_cpu, b_gpu, 1, 2, HIP_R_16BF);
        EXPECT_DOUBLE_EQ(rb.max_ulp, 0.0);
        EXPECT_EQ(rb.count, 2u);
    }

    TEST(UlpCheckGeneral, int32_identical_dispatch)
    {
        std::vector<int32_t> cpu = {10, 20, 30, 40};
        std::vector<int32_t> gpu = cpu;
        CheckResult          r   = run_check(cpu, gpu, 2, 2, HIP_R_32I);

        EXPECT_DOUBLE_EQ(r.max_ulp, 0.0);
        EXPECT_EQ(r.count, 4u);
    }

    TEST(UlpCheckGeneral, unsupported_type_leaves_counters_untouched)
    {
        // HIP_C_32F is not handled by the switch; it hits the default branch,
        // logs an error, and must not touch the running counters.
        std::vector<float> cpu = {1.0f, 2.0f};
        std::vector<float> gpu = {9.0f, 9.0f};
        CheckResult        r   = run_check(cpu, gpu, 1, 2, HIP_C_32F);

        EXPECT_DOUBLE_EQ(r.max_ulp, 0.0);
        EXPECT_DOUBLE_EQ(r.sum_ulp, 0.0);
        EXPECT_EQ(r.count, 0u);
    }

} // namespace
