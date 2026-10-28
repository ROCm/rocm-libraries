// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for how hipblaslt-test holds pinned host memory (AIHPBLAS-5352). The client pool keeps
// buffers that tests return; a large one left there stays pinned, and the kernel cannot
// reclaim it for the other test processes on the host. The _pre_checkin suffix puts the suite in
// the standard test tier. It pins at most about 3.3 GiB at a time and launches no kernels.

#include <gtest/gtest.h>

#include "hipBuffer.hpp"

#include <string>
#include <vector>

namespace
{
    // What the HostPoolReleaseListener in hipblaslt-test does when a test ends.
    void end_test()
    {
        HostPoolReleaseListener().OnTestEnd(
            *::testing::UnitTest::GetInstance()->current_test_info());
    }

    const size_t kSmall = size_t(1) << 20;

    TEST(HostMemory_pre_checkin, large_buffer_released_after_a_test_that_does_not_reuse_it)
    {
        memory_pool<h_memory>::ReleaseCached();
        {
            HipHostBuffer large(HIP_R_8I, kHostPoolRetainLimit);
            HipHostBuffer small(HIP_R_8I, kSmall);
            ASSERT_EQ(large.memcheck(), hipSuccess);
            ASSERT_EQ(small.memcheck(), hipSuccess);
        }

        // The next test may have the same shape, so the large buffer stays for it.
        end_test();
        EXPECT_GE(memory_pool<h_memory>::CachedBytes(), kHostPoolRetainLimit + kSmall);

        // That test did not take it, so it is released; the small one stays.
        end_test();
        EXPECT_LT(memory_pool<h_memory>::CachedBytes(), kHostPoolRetainLimit);
        EXPECT_GE(memory_pool<h_memory>::CachedBytes(), kSmall);
    }

    TEST(HostMemory_pre_checkin, next_test_of_the_same_shape_reuses_the_large_buffer)
    {
        memory_pool<h_memory>::ReleaseCached();
        auto first = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        ASSERT_NE(first.get(), nullptr);
        first.get()[0]                        = 42;
        first.get()[kHostPoolRetainLimit - 1] = 43;
        memory_pool<h_memory>::Restore(first);
        end_test();
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), kHostPoolRetainLimit);

        // Taken from the pool, not pinned anew: it still holds what the last test wrote.
        auto again = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        ASSERT_NE(again.get(), nullptr);
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), 0u);
        EXPECT_EQ(again.get()[0], 42);
        EXPECT_EQ(again.get()[kHostPoolRetainLimit - 1], 43);
        memory_pool<h_memory>::Restore(again);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, large_miss_releases_what_earlier_tests_left)
    {
        // A test of another shape needs a new large buffer; the previous test's ones are
        // released before more memory is pinned, not after this test ends. On any miss the
        // pool already drops the largest buffer that is too small, so two are left here.
        memory_pool<h_memory>::ReleaseCached();
        const size_t step    = kHostPoolRetainLimit / 10;
        auto         earlier = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        auto         bigger  = memory_pool<h_memory>::Get(kHostPoolRetainLimit + step);
        ASSERT_NE(earlier.get(), nullptr);
        ASSERT_NE(bigger.get(), nullptr);
        memory_pool<h_memory>::Restore(earlier);
        memory_pool<h_memory>::Restore(bigger);
        end_test();
        ASSERT_GE(memory_pool<h_memory>::CachedBytes(), 2 * kHostPoolRetainLimit);

        auto larger = memory_pool<h_memory>::Get(kHostPoolRetainLimit + 2 * step);
        ASSERT_NE(larger.get(), nullptr);
        EXPECT_EQ(memory_pool<h_memory>::CachedBytes(), 0u);
        memory_pool<h_memory>::Restore(larger);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, buffers_at_the_retain_limit_get_no_slack)
    {
        // The pool adds 20% to small requests so that a slightly larger one can reuse them. A
        // large buffer is reused only by the same shape, so the extra would only add pinned
        // memory.
        memory_pool<h_memory>::ReleaseCached();
        auto large = memory_pool<h_memory>::Get(kHostPoolRetainLimit);
        ASSERT_NE(large.get(), nullptr);
        EXPECT_EQ(large.capacity(), kHostPoolRetainLimit);
        memory_pool<h_memory>::Restore(large);

        auto small = memory_pool<h_memory>::Get(kSmall);
        ASSERT_NE(small.get(), nullptr);
        EXPECT_GT(small.capacity(), small.bytes());
        memory_pool<h_memory>::Restore(small);
        memory_pool<h_memory>::ReleaseCached();
    }

    TEST(HostMemory_pre_checkin, failed_host_allocation_is_reported_not_used)
    {
        // Far beyond any host: the allocation fails and leaves a null buffer.
        std::vector<HipHostBuffer> ok, failed;
        ok.emplace_back(HIP_R_8I, 4096);
        failed.emplace_back(HIP_R_8I, size_t(1) << 60);
        ASSERT_EQ(ok[0].memcheck(), hipSuccess);
        EXPECT_EQ(failed[0].buf(), nullptr);
        EXPECT_EQ(failed[0].memcheck(), hipErrorOutOfMemory);

        EXPECT_EQ(host_allocation_failure({&ok}), "");
        std::string why = host_allocation_failure({&ok, &failed});
        EXPECT_NE(why.find("could not allocate 1.07e+09 GiB of pinned host memory"),
                  std::string::npos)
            << why;
    }
} // namespace
