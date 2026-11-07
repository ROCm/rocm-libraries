// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the allclose check in allclose.hpp, which hipblaslt-bench -v runs
// against the CPU reference. See README.md in this directory for the HostUnit/smoke_ naming.
//
// The check tries a fixed list of (atol, rtol) pairs, tightest first, and reports the first pair
// under which every element satisfies |cpu - gpu| <= atol + rtol * |gpu|. A reported pair of 1
// means no pair in the list passed, and the bench prints "failed" in its atol and rtol columns.

#include "allclose.hpp"
#include "hipblaslt_datatype2string.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{
    template <typename F>
    void with_type(hipDataType type, F&& f)
    {
        switch(type)
        {
        case HIP_R_32F:
            return f(float{});
        case HIP_R_64F:
            return f(double{});
        case HIP_R_16F:
            return f(hipblasLtHalf{});
        case HIP_R_16BF:
            return f(hip_bfloat16{});
        case HIP_R_8F_E4M3:
            return f(hipblaslt_f8{});
        case HIP_R_8F_E4M3_FNUZ:
            return f(hipblaslt_f8_fnuz{});
        case HIP_R_8I:
            return f(int8_t{});
        case HIP_R_32I:
            return f(int32_t{});
        case HIP_C_32F:
            return f(std::complex<float>{});
        default:
            FAIL() << "type not covered by this test: " << hip_datatype_to_string(type);
        }
    }

    // Column-major M x N matrices, one per batch, ld elements between columns and ld * N
    // between batches. The values used below are small integers, which every type stores exactly.
    struct Matrices
    {
        hipDataType                type;
        int64_t                    m, n, ld, batches;
        std::vector<unsigned char> bytes;

        Matrices(hipDataType type, int64_t m, int64_t n, int64_t ld, int64_t batches)
            : type(type)
            , m(m)
            , n(n)
            , ld(ld)
            , batches(batches)
        {
            size_t element = 0;
            with_type(type, [&](auto tag) { element = sizeof(tag); });
            bytes.assign(size_t(stride() * batches) * element, 0);
            for(int64_t b = 0; b < batches; b++)
                for(int64_t col = 0; col < n; col++)
                    for(int64_t row = 0; row < m; row++)
                        set(b, row, col, (row + 2 * col + 3 * b) % 4);
        }

        int64_t stride() const
        {
            return ld * n;
        }

        void set(int64_t batch, int64_t row, int64_t col, double value)
        {
            const size_t i = size_t(batch * stride() + col * ld + row);
            with_type(type, [&](auto tag) {
                using T                               = decltype(tag);
                reinterpret_cast<T*>(bytes.data())[i] = T(float(value));
            });
        }

        void* data()
        {
            return bytes.data();
        }
    };

    // Runs the strided-batch check the bench uses on one cpu/gpu pair.
    bool check(Matrices& cpu, Matrices& gpu, double& atol, double& rtol)
    {
        return allclose_check_general('F',
                                      cpu.m,
                                      cpu.n,
                                      cpu.ld,
                                      cpu.stride(),
                                      cpu.data(),
                                      gpu.data(),
                                      cpu.batches,
                                      atol,
                                      rtol,
                                      cpu.type);
    }

    class HostUnitAllcloseTypes : public ::testing::TestWithParam<hipDataType>
    {
    };

    TEST_P(HostUnitAllcloseTypes, smoke_IdenticalBatchesPassAtTightestPair)
    {
        Matrices cpu(GetParam(), 3, 2, 4, 2);
        Matrices gpu(GetParam(), 3, 2, 4, 2);
        double   atol = 1, rtol = 1;
        EXPECT_TRUE(check(cpu, gpu, atol, rtol));
        EXPECT_EQ(atol, 1e-6);
        EXPECT_EQ(rtol, 1e-6);
    }

    // The case behind ROCM-32137: one wrong element in a batch after the first. 4 against 6
    // fails even the loosest pair, 0.1 + 0.1 * 6.
    TEST_P(HostUnitAllcloseTypes, smoke_WrongElementInLaterBatchFails)
    {
        Matrices cpu(GetParam(), 3, 2, 4, 3);
        Matrices gpu(GetParam(), 3, 2, 4, 3);
        cpu.set(2, 1, 1, 4);
        gpu.set(2, 1, 1, 6);
        double atol = 1, rtol = 1;
        EXPECT_FALSE(check(cpu, gpu, atol, rtol));
        EXPECT_EQ(atol, 1);
        EXPECT_EQ(rtol, 1);
    }

    TEST_P(HostUnitAllcloseTypes, smoke_WrongElementInFirstBatchFails)
    {
        Matrices cpu(GetParam(), 3, 2, 4, 3);
        Matrices gpu(GetParam(), 3, 2, 4, 3);
        cpu.set(0, 2, 0, 4);
        gpu.set(0, 2, 0, 6);
        double atol = 1, rtol = 1;
        EXPECT_FALSE(check(cpu, gpu, atol, rtol));
        EXPECT_EQ(atol, 1);
        EXPECT_EQ(rtol, 1);
    }

    INSTANTIATE_TEST_SUITE_P(EveryType,
                             HostUnitAllcloseTypes,
                             ::testing::Values(HIP_R_32F,
                                               HIP_R_64F,
                                               HIP_R_16F,
                                               HIP_R_16BF,
                                               HIP_R_8F_E4M3,
                                               HIP_R_8F_E4M3_FNUZ,
                                               HIP_R_8I,
                                               HIP_R_32I,
                                               HIP_C_32F),
                             [](const ::testing::TestParamInfo<hipDataType>& info) {
                                 return std::string(hip_datatype_to_string(info.param));
                             });

    // A later batch that needs a looser pair than batch 0 sets the reported pair. Here 0 against
    // 5e-5 needs atol 1e-4, while batch 0 passes at 1e-6.
    TEST(HostUnitAllclose, smoke_LooserLaterBatchSetsReportedPair)
    {
        Matrices cpu(HIP_R_32F, 3, 2, 4, 2);
        Matrices gpu(HIP_R_32F, 3, 2, 4, 2);
        cpu.set(1, 0, 1, 0);
        gpu.set(1, 0, 1, 5e-5);
        double atol = 1, rtol = 1;
        EXPECT_TRUE(check(cpu, gpu, atol, rtol));
        EXPECT_EQ(atol, 1e-4);
        EXPECT_EQ(rtol, 1e-6);
    }

    // The bench passes the same atol and rtol to every gemm of a grouped GEMM and to every
    // batch in pointer-array mode, so one call must not depend on what an earlier call left there.
    TEST(HostUnitAllclose, smoke_EarlierCallDoesNotDecideLaterCall)
    {
        Matrices good_cpu(HIP_R_32F, 3, 2, 4, 1);
        Matrices good_gpu(HIP_R_32F, 3, 2, 4, 1);
        Matrices bad_cpu(HIP_R_32F, 3, 2, 4, 1);
        Matrices bad_gpu(HIP_R_32F, 3, 2, 4, 1);
        bad_cpu.set(0, 1, 0, 4);
        bad_gpu.set(0, 1, 0, 6);

        double atol = 1, rtol = 1;
        EXPECT_TRUE(check(good_cpu, good_gpu, atol, rtol));
        EXPECT_FALSE(check(bad_cpu, bad_gpu, atol, rtol));
        EXPECT_EQ(atol, 1);
        EXPECT_EQ(rtol, 1);
    }

    TEST(HostUnitAllclose, smoke_EmptyMatrixPassesAndReportsZero)
    {
        Matrices        cpu(HIP_R_32F, 0, 2, 4, 2);
        Matrices        gpu(HIP_R_32F, 0, 2, 4, 2);
        double          atol = 1, rtol = 1;
        AllcloseFailure failure;
        EXPECT_TRUE(allclose_check_general(
            'F', 0, 2, 4, 8, cpu.data(), gpu.data(), 2, atol, rtol, HIP_R_32F, &failure));
        EXPECT_EQ(atol, 0);
        EXPECT_EQ(rtol, 0);
        EXPECT_EQ(failure.compared_count, 0u);
        EXPECT_EQ(failure.bad_count, 0u);
    }

    // Three wrong elements over three batches. Row 3 is padding (ld 4, M 3), so a difference
    // there is not counted.
    TEST(HostUnitAllclose, smoke_FailureCountsBadElementsAndFindsWorst)
    {
        Matrices cpu(HIP_R_32F, 3, 2, 4, 3);
        Matrices gpu(HIP_R_32F, 3, 2, 4, 3);
        cpu.set(0, 2, 0, 4);
        gpu.set(0, 2, 0, 6);
        cpu.set(2, 1, 1, 4);
        gpu.set(2, 1, 1, 9);
        cpu.set(2, 0, 0, 0);
        gpu.set(2, 0, 0, 1);
        gpu.set(1, 3, 0, 7);

        double          atol = 0, rtol = 0;
        AllcloseFailure failure;
        EXPECT_FALSE(allclose_check_general('F',
                                            cpu.m,
                                            cpu.n,
                                            cpu.ld,
                                            cpu.stride(),
                                            cpu.data(),
                                            gpu.data(),
                                            cpu.batches,
                                            atol,
                                            rtol,
                                            HIP_R_32F,
                                            &failure));
        EXPECT_EQ(atol, 1);
        EXPECT_EQ(failure.bad_count, 3u);
        EXPECT_EQ(failure.compared_count, 18u);
        EXPECT_EQ(failure.batch, 2);
        EXPECT_EQ(failure.row, 1);
        EXPECT_EQ(failure.col, 1);
        EXPECT_FALSE(failure.imag);
        EXPECT_EQ(failure.got, 9);
        EXPECT_EQ(failure.expected, 4);
        EXPECT_EQ(allclose_failure_message(failure),
                  "3 of 18 elements are outside atol 0.1 + rtol 0.1 * |result|; the worst is "
                  "batch 2, row 1, column 1: result 9, reference 4");
    }

    TEST(HostUnitAllclose, smoke_FailureInImaginaryPartIsLabeled)
    {
        Matrices cpu(HIP_C_32F, 3, 2, 4, 1);
        Matrices gpu(HIP_C_32F, 3, 2, 4, 1);
        reinterpret_cast<std::complex<float>*>(gpu.data())[1 * gpu.ld + 2].imag(3);

        double          atol = 0, rtol = 0;
        AllcloseFailure failure;
        EXPECT_FALSE(allclose_check_general(
            'F', 3, 2, 4, 8, cpu.data(), gpu.data(), 1, atol, rtol, HIP_C_32F, &failure));
        EXPECT_EQ(failure.bad_count, 1u);
        EXPECT_EQ(failure.row, 2);
        EXPECT_EQ(failure.col, 1);
        EXPECT_TRUE(failure.imag);
        EXPECT_EQ(failure.got, 3);
        EXPECT_EQ(failure.expected, 0);
        EXPECT_NE(allclose_failure_message(failure).find("column 1 (imaginary part)"),
                  std::string::npos);
    }

    // The pair order matches the search: a larger atol is looser whatever the rtol, and a pair
    // of 1 (failed) beats every pair from the list.
    TEST(HostUnitAllclose, smoke_MergePairKeepsLooserPair)
    {
        double atol = 0, rtol = 0;
        allclose_merge_pair(atol, rtol, 1e-6, 1e-3);
        EXPECT_EQ(atol, 1e-6);
        EXPECT_EQ(rtol, 1e-3);
        allclose_merge_pair(atol, rtol, 1e-4, 1e-6);
        EXPECT_EQ(atol, 1e-4);
        EXPECT_EQ(rtol, 1e-6);
        allclose_merge_pair(atol, rtol, 1e-6, 1e-1);
        EXPECT_EQ(atol, 1e-4);
        EXPECT_EQ(rtol, 1e-6);
        allclose_merge_pair(atol, rtol, 1, 1);
        allclose_merge_pair(atol, rtol, 1e-1, 1e-1);
        EXPECT_EQ(atol, 1);
        EXPECT_EQ(rtol, 1);
    }

    // The bench folds pointer-array batches one at a time; the worst element keeps its batch,
    // and a NaN result ranks worse than any finite error.
    TEST(HostUnitAllclose, smoke_MergeFailureKeepsWorstAndItsBatch)
    {
        AllcloseFailure total;
        AllcloseFailure one;
        one.compared_count = 6;
        one.bad_count      = 1;
        one.error          = 2;
        allclose_merge_failure(total, one, 0);
        one.bad_count = 0;
        allclose_merge_failure(total, one, 1);
        one.bad_count = 1;
        one.error     = std::numeric_limits<double>::quiet_NaN();
        allclose_merge_failure(total, one, 3);
        one.error = 100;
        allclose_merge_failure(total, one, 4);

        EXPECT_EQ(total.compared_count, 24u);
        EXPECT_EQ(total.bad_count, 3u);
        EXPECT_EQ(total.batch, 3);
        EXPECT_TRUE(std::isnan(total.error));
    }
}
