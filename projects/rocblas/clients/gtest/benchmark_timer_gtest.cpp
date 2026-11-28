// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "testing_common.hpp"

#include "benchmark.hpp"

#include <gtest/gtest.h>

namespace
{
    // Benchmark::run_timer used to chain three get_time_us_sync calls. With event timing (the
    // default), the first call of a pair records the start and returns 0 and the second returns
    // the time since then, so the third call opened a new pair: the hot time came out negative and
    // the next timing loop measured from that stray start.
    TEST(quick_benchmark_timer, run_timer_pairs_its_timer_calls)
    {
        constexpr size_t bytes  = size_t(64) << 20;
        void*            buffer = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&buffer, bytes));
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipSuccess, hipStreamCreate(&stream));

        Arguments arg{};
        arg.cold_iters = 2;
        arg.iters      = 5;
        auto memset    = [&](size_t) { (void)hipMemsetAsync(buffer, 0, bytes, stream); };

        for(int run = 0; run < 2; run++)
        {
            Benchmark<decltype(memset)> benchmark(memset, stream, arg, 1);
            benchmark.run_timer();
            EXPECT_GT(benchmark.get_cold_time(), 0) << "run " << run;
            EXPECT_GT(benchmark.get_hot_time(), 0) << "run " << run;
        }

        // run_timer closed its pairs: a following pair measures from its own start
        double start = get_time_us_sync(stream);
        (void)hipMemsetAsync(buffer, 0, bytes, stream);
        EXPECT_GT(get_time_us_sync(stream) - start, 0);

        EXPECT_EQ(hipSuccess, hipStreamDestroy(stream));
        EXPECT_EQ(hipSuccess, hipFree(buffer));
    }
}
