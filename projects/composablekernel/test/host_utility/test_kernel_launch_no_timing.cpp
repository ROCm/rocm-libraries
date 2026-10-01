// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exercise the compile-time untimed branch even in a timed CK build.
#undef CK_TIME_KERNEL
#define CK_TIME_KERNEL 0

#include <gtest/gtest.h>

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/library/utility/device_memory.hpp"

namespace {

__global__ void IncrementFlag(int* value) { ++*value; }

template <typename Launch>
void CheckUntimedPreprocess(Launch launch)
{
    ck::DeviceMem device(sizeof(int));
    const int dirty = 113;
    device.ToDevice(&dirty);

    hipStream_t stream;
    ck::hip_check_error(hipStreamCreate(&stream));
    const StreamConfig config{stream, false};
    int* value       = static_cast<int*>(device.GetDeviceBuffer());
    const auto clear = [&]() {
        ck::hip_check_error(hipMemsetAsync(value, 0, sizeof(int), config.stream_id_));
    };

    for(int run = 0; run < 2; ++run)
    {
        launch(config, clear, value);
        ck::hip_check_error(hipStreamSynchronize(stream));
        int result = 0;
        device.FromDevice(&result);
        EXPECT_EQ(result, 1) << "run=" << run;
    }
    ck::hip_check_error(hipStreamDestroy(stream));
}

TEST(TestKernelLaunchNoTiming, BasicLauncherPreprocess)
{
    CheckUntimedPreprocess([](const auto& config, const auto& clear, int* value) {
        ck::launch_and_time_kernel_with_preprocess(
            config, clear, IncrementFlag, dim3(1), dim3(1), 0, value);
    });
}

TEST(TestKernelLaunchNoTiming, FlushCacheLauncherPreprocess)
{
    CheckUntimedPreprocess([](const auto& config, const auto& clear, int* value) {
        ck::launch_and_time_kernel_with_preprocess_flush_cache(
            config, clear, IncrementFlag, dim3(1), dim3(1), 0, value);
    });
}

TEST(TestKernelLaunchNoTiming, UtilityLauncherPreprocess)
{
    CheckUntimedPreprocess([](const auto& config, const auto& clear, int* value) {
        ck::utility::launch_and_time_kernel_with_preprocess<false>(
            config, clear, IncrementFlag, dim3(1), dim3(1), 0, value);
    });
}

} // namespace
