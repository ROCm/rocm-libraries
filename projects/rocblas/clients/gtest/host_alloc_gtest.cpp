/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas_data.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "singletons.hpp"
#include "type_dispatch.hpp"
#include <gtest/gtest-spi.h>

// host_mem_safe turns down an allocation that would not fit, and for managed memory it
// consults a count that the device containers keep. The count is keyed on the pointer, so
// setup and teardown have to name the same one: the padded block the container hands out,
// not the start of the allocation underneath it. When they disagreed the release missed, and
// a miss in that map is silent, so the count only ever rose and a long run of managed
// allocations began refusing ones that would have fit.
//
// The guard regions either side of the user allocation are sized in elements, so their byte
// offsets scale with sizeof(T). The suite is therefore run through the usual type dispatch
// rather than fixed at float, and the YAML lists the precisions to cover.
namespace
{
    // Guard pad in elements, forced for the duration of a test rather than taken from
    // whatever --pad a run was given: a pad of zero puts the user pointer and the base
    // allocation at the same address, so neither the guard regions nor the pointer
    // mismatch these tests target can exist at all. Restored however the test leaves,
    // since a skip or a failed allocation returns early and every later test in the
    // process reads the same global.
    constexpr size_t c_guard_pad = 4096;

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

    // Reported as a bool rather than skipping here: GTEST_SKIP only returns from the
    // function it appears in, so the decision has to be made in the test body.
    template <typename T>
    bool device_alloc_available()
    {
        device_vector<T> probe(1);
        return probe.memcheck() == hipSuccess;
    }

    size_t guarded_length(const Arguments& arg)
    {
        return arg.N > 0 ? size_t(arg.N) : 1024;
    }

    // rocblas_init_nan / rocblas_nan_rng only force a NaN exponent; sign and mantissa are
    // random. A fixed fill value can therefore match live guard bytes and under-count the
    // mismatches, so the byte-level diagnostics are exercised by inverting the actual
    // m_guard bytes: every byte written is then guaranteed to differ.
    template <typename T>
    hipError_t overwrite_post_guard_inverted(device_vector<T>& dv, size_t first_elem, size_t n_elem)
    {
        // Three elements of the widest supported type is the largest write any caller makes.
        constexpr size_t c_max_flip_bytes = 3 * sizeof(rocblas_double_complex);

        const size_t n_bytes = n_elem * sizeof(T);
        if(n_bytes > c_max_flip_bytes)
            return hipErrorInvalidValue;

        const auto* src
            = reinterpret_cast<const unsigned char*>(d_vector<T>::m_guard) + first_elem * sizeof(T);
        unsigned char flipped[c_max_flip_bytes];
        for(size_t i = 0; i < n_bytes; ++i)
            flipped[i] = static_cast<unsigned char>(~src[i]);

        return hipMemcpy(
            static_cast<T*>(dv) + dv.nmemb() + first_elem, flipped, n_bytes, hipMemcpyDefault);
    }

    // Guard-detection tests: verify that device_vector_check catches writes into the
    // guard regions. The allocation and the corruption both happen inside
    // EXPECT_NONFATAL_FAILURE so the expected failure is captured and the test itself
    // remains green. Each test covers one guard (post and pre) independently.

    template <typename T>
    void testing_guard_post_overwrite(const Arguments& arg)
    {
        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; post-guard corruption cannot be detected";

        // Checked before entering EXPECT_NONFATAL_FAILURE, where a skip or a fatal failure
        // would escape the wrapper and leave the process in a misleading state.
        if(!device_alloc_available<T>())
            GTEST_SKIP() << "device allocation unavailable";

        // Overwrite the first element of the post-guard with zeros. The guard pattern is
        // NaN bytes, which are never all zero for any supported type, so the write always
        // produces a detectable mismatch. Both the allocation and the destruction (which
        // triggers the check) are inside the macro so the resulting non-fatal failure is
        // captured.
        EXPECT_NONFATAL_FAILURE(
            {
                device_vector<T> dv(guarded_length(arg));
                // ASSERT (not EXPECT): if hipMemset fails, an EXPECT would emit a
                // nonfatal failure whose message contains "post-guard";
                // EXPECT_NONFATAL_FAILURE would then see one matching nonfatal failure
                // and pass, masking the fact that the guard was never actually corrupted.
                // ASSERT emits a fatal failure (kFatalFailure), which SingleFailureChecker
                // treats as a type mismatch, so the test fails with a clear message rather
                // than a false green.
                ASSERT_EQ(hipMemset(static_cast<T*>(dv) + dv.nmemb(), 0, sizeof(T)), hipSuccess)
                    << "hipMemset failed; post-guard was never corrupted";
            },
            "post-guard");
    }

    template <typename T>
    void testing_guard_pre_overwrite(const Arguments& arg)
    {
        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; pre-guard corruption cannot be detected";

        if(!device_alloc_available<T>())
            GTEST_SKIP() << "device allocation unavailable";

        // Overwrite the last element of the pre-guard with zeros. The user pointer sits
        // m_pad elements past the base allocation, so subtracting one element lands inside
        // the pre-guard without leaving the hipMalloc'd block.
        EXPECT_NONFATAL_FAILURE(
            {
                device_vector<T> dv(guarded_length(arg));
                // ASSERT (not EXPECT): same reasoning as testing_guard_post_overwrite.
                ASSERT_EQ(hipMemset(static_cast<T*>(dv) - 1, 0, sizeof(T)), hipSuccess)
                    << "hipMemset failed; pre-guard was never corrupted";
            },
            "pre-guard");
    }

    template <typename T>
    void testing_guard_clean_alloc(const Arguments& arg)
    {
        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; the false-positive check is meaningless with pad == 0";

        if(!device_alloc_available<T>())
            GTEST_SKIP() << "device allocation unavailable";

        // A clean alloc+free (guards intact) must produce exactly zero nonfatal GTest
        // failures. This catches regressions such as an uninitialised guard pattern that
        // matches the default content of freshly-allocated device memory.
        // GTest provides no EXPECT_NO_NONFATAL_FAILURE macro; ScopedFakeTestPartResultReporter
        // (from <gtest/gtest-spi.h>) is the only standard way to assert that a block emits zero
        // nonfatal failures.
        ::testing::TestPartResultArray failures;
        {
            ::testing::ScopedFakeTestPartResultReporter reporter(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD,
                &failures);
            device_vector<T> dv(guarded_length(arg));
            // If this allocation fails after the one-element probe succeeded, the
            // destructor is a no-op and an empty `failures` array would greenwash
            // the test. Record memcheck with EXPECT so a failed alloc shows up in
            // `failures` and the final size check does not treat it as a clean pass.
            EXPECT_EQ(dv.memcheck(), hipSuccess)
                << "device allocation failed; guard check was never exercised";
            // Guards are not modified. dv is declared after reporter, so its
            // destructor runs first (reverse declaration order) while reporter
            // is still intercepting — any EXPECT from device_vector_check is captured.
        }
        EXPECT_EQ(failures.size(), 0)
            << "device_vector_check reported a spurious failure on an unmodified guard";
    }

    // Byte-diagnostic tests: verify that device_vector_check reports the correct
    // differing-byte count and first-byte index. Both quantities are element-size
    // dependent, so the expected text is derived from sizeof(T) rather than hard-coded.

    template <typename T>
    void testing_guard_reports_byte_count(const Arguments& arg)
    {
        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; the byte-count check is meaningless";

        if(!device_alloc_available<T>())
            GTEST_SKIP() << "device allocation unavailable";

        // Three elements inverted, so the diagnostic must report 3 * sizeof(T) differing
        // bytes whatever the random NaN payload in m_guard happens to be.
        const std::string expected = std::to_string(3 * sizeof(T)) + " post-guard";

        EXPECT_NONFATAL_FAILURE(
            {
                device_vector<T> dv(guarded_length(arg));
                ASSERT_EQ(overwrite_post_guard_inverted<T>(dv, 0, 3), hipSuccess)
                    << "post-guard invert failed; byte-count diagnostic was never exercised";
            },
            expected);
    }

    template <typename T>
    void testing_guard_reports_first_byte_index(const Arguments& arg)
    {
        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; the first-byte-index check is meaningless";

        if(!device_alloc_available<T>())
            GTEST_SKIP() << "device allocation unavailable";

        // Element 5 starts at byte offset 5 * sizeof(T) into the post-guard. The bytes
        // before it are left intact, so that offset is the first mismatch.
        const std::string expected = "first at byte " + std::to_string(5 * sizeof(T));

        EXPECT_NONFATAL_FAILURE(
            {
                device_vector<T> dv(guarded_length(arg));
                ASSERT_EQ(overwrite_post_guard_inverted<T>(dv, 5, 1), hipSuccess)
                    << "post-guard invert failed; first-byte-index diagnostic was never exercised";
            },
            expected);
    }

    // Verifies that device_vector_check reports both guard regions independently when both
    // are corrupted. device_vector_check uses EXPECT (not ASSERT), so it continues past the
    // first guard failure; both "post-guard" and "pre-guard" must appear in the output.
    template <typename T>
    void testing_guard_detects_both_guards(const Arguments& arg)
    {
        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; both-guard corruption cannot be detected";

        if(!device_alloc_available<T>())
            GTEST_SKIP() << "device allocation unavailable";

        // Capture failures from alloc+corrupt+destroy. hipMemset results are recorded
        // outside the reporter scope and checked after it closes so that an ASSERT inside
        // the intercepted scope cannot mask a real hipMemset failure.
        hipError_t                     post_err = hipSuccess, pre_err = hipSuccess;
        ::testing::TestPartResultArray failures;
        {
            ::testing::ScopedFakeTestPartResultReporter reporter(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD,
                &failures);
            device_vector<T> dv(guarded_length(arg));
            post_err = hipMemset(static_cast<T*>(dv) + dv.nmemb(), 0, sizeof(T));
            pre_err  = hipMemset(static_cast<T*>(dv) - 1, 0, sizeof(T));
            // dv destructs here; device_vector_check fires and both EXPECT failures are captured.
        }
        ASSERT_EQ(post_err, hipSuccess) << "hipMemset post-guard failed; test is inconclusive";
        ASSERT_EQ(pre_err, hipSuccess) << "hipMemset pre-guard failed; test is inconclusive";

        bool post_reported = false, pre_reported = false;
        for(int i = 0; i < failures.size() && (!post_reported || !pre_reported); ++i)
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

    template <typename T>
    void testing_hmm_alloc_count(const Arguments& arg)
    {
        // The count only tracks managed allocations, so a run with HMM off would compare a
        // baseline against itself and pass no matter what the accounting did. The harness
        // has already skipped this test if the device has no managed memory.
        ASSERT_TRUE(arg.HMM) << "this test counts managed allocations only; set HMM: true in YAML";

        scoped_pad_length pad(c_guard_pad);
        ASSERT_EQ(g_DVEC_PAD, c_guard_pad)
            << "guard pad was not set; the pointer mismatch this test targets cannot show up";

        const size_t baseline = host_bytes_allocated();
        {
            device_vector<T> dv(guarded_length(arg), 1 /* inc */, arg.HMM);
            CHECK_DEVICE_ALLOCATION(dv.memcheck());

            EXPECT_GT(host_bytes_allocated(), baseline)
                << "the managed allocation was never counted, so the ceiling host_mem_safe "
                   "enforces is not being tracked";
        }
        EXPECT_EQ(host_bytes_allocated(), baseline)
            << "the managed allocation was counted but not released, so every managed allocation "
               "in this process permanently consumes the ceiling host_mem_safe enforces";
    }

    bool is_host_alloc_function(const char* fn)
    {
        return !strcmp(fn, "host_alloc_guard_post_overwrite")
               || !strcmp(fn, "host_alloc_guard_pre_overwrite")
               || !strcmp(fn, "host_alloc_guard_clean_alloc")
               || !strcmp(fn, "host_alloc_guard_reports_byte_count")
               || !strcmp(fn, "host_alloc_guard_reports_first_byte_index")
               || !strcmp(fn, "host_alloc_guard_detects_both_guards")
               || !strcmp(fn, "host_alloc_hmm_count");
    }

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct host_alloc_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct host_alloc_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "host_alloc_guard_post_overwrite"))
                testing_guard_post_overwrite<T>(arg);
            else if(!strcmp(arg.function, "host_alloc_guard_pre_overwrite"))
                testing_guard_pre_overwrite<T>(arg);
            else if(!strcmp(arg.function, "host_alloc_guard_clean_alloc"))
                testing_guard_clean_alloc<T>(arg);
            else if(!strcmp(arg.function, "host_alloc_guard_reports_byte_count"))
                testing_guard_reports_byte_count<T>(arg);
            else if(!strcmp(arg.function, "host_alloc_guard_reports_first_byte_index"))
                testing_guard_reports_first_byte_index<T>(arg);
            else if(!strcmp(arg.function, "host_alloc_guard_detects_both_guards"))
                testing_guard_detects_both_guards<T>(arg);
            else if(!strcmp(arg.function, "host_alloc_hmm_count"))
                testing_hmm_alloc_count<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct host_alloc : RocBLAS_Test<host_alloc, host_alloc_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return is_host_alloc_function(arg.function);
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<host_alloc> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            return std::move(name);
        }
    };

    // RUN_TEST_ON_THREADS_STREAMS, not CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES: the
    // managed-memory skip for YAML HMM: true only exists in this dispatch, and
    // host_alloc_hmm_count needs it.
    TEST_P(host_alloc, auxiliary)
    {
        RUN_TEST_ON_THREADS_STREAMS(rocblas_simple_dispatch<host_alloc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(host_alloc);

} // namespace
