/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "client_utility.hpp"
#include "host_alloc.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "singletons.hpp"
#include <gtest/gtest-spi.h>

// host_mem_safe turns down an allocation that would not fit, and for managed memory it
// consults a count that the device containers keep. The count is keyed on the pointer, so
// setup and teardown have to name the same one: the padded block the container hands out,
// not the start of the allocation underneath it. When they disagreed the release missed, and
// a miss in that map is silent, so the count only ever rose and a long run of managed
// allocations began refusing ones that would have fit.
//
// The guard pad is set here rather than taken from whatever a previous test left in place,
// because a pad of zero puts both pointers at the same address and the disagreement cannot
// show up at all. Restored on the way out however this test leaves, since a skip or a failed
// allocation returns early and every later test in the process reads the same global.
namespace
{
    struct scoped_pad_length
    {
        explicit scoped_pad_length(size_t pad)
            : m_was(g_DVEC_PAD)
        {
            d_vector_set_pad_length(pad);
        }
        ~scoped_pad_length()
        {
            d_vector_set_pad_length(m_was);
        }

    private:
        size_t m_was;
    };
}

// Guard-detection tests: verify that device_vector_check catches writes into the
// guard regions. The allocation and corruption happen inside EXPECT_NONFATAL_FAILURE
// so the expected failure is captured and the test itself remains green. Each test
// covers one guard (post and pre) independently.
//
// The pad is forced nonzero so the regions exist; it is restored on exit. The
// pre-check with a probe allocation confirms device memory is available before
// entering EXPECT_NONFATAL_FAILURE, where a skip or fatal failure would escape
// the wrapper and leave the process in a misleading state.

TEST(host_alloc, guard_detects_post_overwrite)
{
    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; post-guard corruption cannot be detected";

    {
        device_vector<float> probe(1);
        if(probe.memcheck() != hipSuccess)
            GTEST_SKIP() << "device allocation unavailable";
    }

    // Overwrite one element in the post-guard with zeros. The guard pattern is
    // NaN bytes, so any zero write produces a detectable mismatch. Both the
    // allocation and the destruction (which triggers the check) are inside the
    // macro so the resulting non-fatal failure is captured.
    EXPECT_NONFATAL_FAILURE(
        {
            device_vector<float> dv(1024);
            // ASSERT (not EXPECT): if hipMemset fails, an EXPECT would emit a
            // nonfatal failure whose message contains "post-guard"; EXPECT_NONFATAL_FAILURE
            // would then see one matching nonfatal failure and pass, masking the fact
            // that the guard was never actually corrupted. ASSERT emits a fatal failure
            // (kFatalFailure), which SingleFailureChecker treats as a type mismatch,
            // so the test fails with a clear message rather than a false green.
            ASSERT_EQ(hipMemset(static_cast<float*>(dv) + dv.nmemb(), 0, sizeof(float)), hipSuccess)
                << "hipMemset failed; post-guard was never corrupted";
        },
        "post-guard");
}

TEST(host_alloc, guard_detects_pre_overwrite)
{
    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; pre-guard corruption cannot be detected";

    {
        device_vector<float> probe(1);
        if(probe.memcheck() != hipSuccess)
            GTEST_SKIP() << "device allocation unavailable";
    }

    // Overwrite the last element of the pre-guard with zeros. The user pointer
    // sits m_pad elements past the base allocation, so subtracting one element
    // lands inside the pre-guard without leaving the hipMalloc'd block.
    EXPECT_NONFATAL_FAILURE(
        {
            device_vector<float> dv(1024);
            // ASSERT (not EXPECT): same reasoning as guard_detects_post_overwrite.
            ASSERT_EQ(hipMemset(static_cast<float*>(dv) - 1, 0, sizeof(float)), hipSuccess)
                << "hipMemset failed; pre-guard was never corrupted";
        },
        "pre-guard");
}

TEST(host_alloc, guard_no_false_positive_on_clean_alloc)
{
    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; the false-positive check is meaningless with pad == 0";

    {
        device_vector<float> probe(1);
        if(probe.memcheck() != hipSuccess)
            GTEST_SKIP() << "device allocation unavailable";
    }

    // A clean alloc+free (guards intact) must produce exactly zero nonfatal GTest
    // failures. This catches regressions such as an uninitialised guard pattern that
    // matches the default content of freshly-allocated device memory.
    // GTest provides no EXPECT_NO_NONFATAL_FAILURE macro; ScopedFakeTestPartResultReporter
    // (from <gtest/gtest-spi.h>) is the only standard way to assert that a block emits zero
    // nonfatal failures.
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        device_vector<float> dv(1024);
        // Guards are not modified. dv is declared after reporter, so its
        // destructor runs first (reverse declaration order) while reporter
        // is still intercepting — any EXPECT from device_vector_check is captured.
    }
    EXPECT_EQ(failures.size(), 0)
        << "device_vector_check reported a spurious failure on an unmodified guard";
}

// Byte-diagnostic tests: verify that device_vector_check reports the correct differing-byte
// count and first-byte index via the report_guard_corruption lambda. Both tests rely on
// rocblas_init_nan producing 0x7FC00000 (quiet NaN) for float, which differs from 0xFF in
// all four bytes on little-endian IEEE 754 systems, giving a deterministic byte count and
// a predictable first-differing-byte index.

TEST(host_alloc, guard_reports_byte_count)
{
    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; the byte-count check is meaningless";

    {
        device_vector<float> probe(1);
        if(probe.memcheck() != hipSuccess)
            GTEST_SKIP() << "device allocation unavailable";
    }

    // 3 floats × 4 bytes = 12 guard bytes corrupted. 0xFF differs from every byte of
    // 0x7FC00000 on little-endian, so the diagnostic must report "12 post-guard".
    EXPECT_NONFATAL_FAILURE(
        {
            device_vector<float> dv(1024);
            ASSERT_EQ(hipMemset(static_cast<float*>(dv) + dv.nmemb(), 0xFF, 3 * sizeof(float)),
                      hipSuccess)
                << "hipMemset failed; byte-count diagnostic was never exercised";
        },
        "12 post-guard");
}

TEST(host_alloc, guard_reports_first_byte_index)
{
    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; the first-byte-index check is meaningless";

    {
        device_vector<float> probe(1);
        if(probe.memcheck() != hipSuccess)
            GTEST_SKIP() << "device allocation unavailable";
    }

    // Element 5 starts at byte offset 5 × sizeof(float) = 20 into the post-guard.
    // Bytes 0–19 remain intact; the diagnostic must report "first at byte 20".
    EXPECT_NONFATAL_FAILURE(
        {
            device_vector<float> dv(1024);
            ASSERT_EQ(hipMemset(static_cast<float*>(dv) + dv.nmemb() + 5, 0xFF, sizeof(float)),
                      hipSuccess)
                << "hipMemset failed; first-byte-index diagnostic was never exercised";
        },
        "first at byte 20");
}

// Verifies that device_vector_check reports both guard regions independently when both
// are corrupted. device_vector_check uses EXPECT (not ASSERT), so it continues past the
// first guard failure; both "post-guard" and "pre-guard" must appear in the output.
TEST(host_alloc, guard_detects_both_guards_corrupted)
{
    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; both-guard corruption cannot be detected";

    {
        device_vector<float> probe(1);
        if(probe.memcheck() != hipSuccess)
            GTEST_SKIP() << "device allocation unavailable";
    }

    // Capture failures from alloc+corrupt+destroy. hipMemset results are recorded
    // outside the reporter scope and checked after it closes so that an ASSERT inside
    // the intercepted scope cannot mask a real hipMemset failure.
    hipError_t                     post_err = hipSuccess, pre_err = hipSuccess;
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        device_vector<float> dv(1024);
        post_err = hipMemset(static_cast<float*>(dv) + dv.nmemb(), 0, sizeof(float));
        pre_err  = hipMemset(static_cast<float*>(dv) - 1, 0, sizeof(float));
        // dv destructs here; device_vector_check fires and both EXPECT failures are captured.
    }
    ASSERT_EQ(post_err, hipSuccess) << "hipMemset post-guard failed; test is inconclusive";
    ASSERT_EQ(pre_err, hipSuccess) << "hipMemset pre-guard failed; test is inconclusive";

    bool post_reported = false, pre_reported = false;
    for(size_t i = 0; i < failures.size() && (!post_reported || !pre_reported); ++i)
    {
        std::string msg = failures.GetTestPartResult(i).message();
        if(!post_reported && msg.find("post-guard") != std::string::npos)
            post_reported = true;
        if(!pre_reported && msg.find("pre-guard") != std::string::npos)
            pre_reported = true;
    }
    EXPECT_TRUE(post_reported) << "post-guard corruption was not reported";
    EXPECT_TRUE(pre_reported) << "pre-guard corruption was not reported";
}

TEST(host_alloc, hmm_count_returns_to_its_baseline)
{
    int device = 0;
    CHECK_HIP_ERROR(hipGetDevice(&device));

    int managed = 0;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(
        &managed, hipDeviceAttribute_t(hipDeviceAttributeManagedMemory), device));
    if(!managed)
        GTEST_SKIP() << HMM_NOT_SUPPORTED_STRING;

    scoped_pad_length pad(4096);
    ASSERT_EQ(g_DVEC_PAD, size_t(4096))
        << "guard pad was not set; the pointer mismatch this test targets cannot show up";

    const size_t baseline = host_bytes_allocated();
    {
        device_vector<float> dv(1024, 1 /* inc */, true /* HMM */);
        CHECK_DEVICE_ALLOCATION(dv.memcheck());

        EXPECT_GT(host_bytes_allocated(), baseline)
            << "the managed allocation was never counted, so the ceiling host_mem_safe "
               "enforces is not being tracked";
    }
    EXPECT_EQ(host_bytes_allocated(), baseline)
        << "the managed allocation was counted but not released, so every managed allocation "
           "in this process permanently consumes the ceiling host_mem_safe enforces";
}
