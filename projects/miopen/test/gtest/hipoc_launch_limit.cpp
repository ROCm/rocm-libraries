// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <gtest/gtest.h>

#include "get_handle.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <utility>

namespace {

constexpr char kProgramName[] = "gtest_hipoc_launch_limit.cpp";
constexpr char kKernelName[]  = "gtest_hipoc_launch_limit";

constexpr std::size_t kOversizedGrid = 65537;

// Only the last workgroup writes, so the readback detects a silently truncated grid.
constexpr char kKernelSource[] = R"(
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

extern "C" __global__ void gtest_hipoc_launch_limit(float* output, unsigned last_y, unsigned last_z)
{
    if(blockIdx.y == last_y && blockIdx.z == last_z)
        *output = 1.0f;
}
)";

} // namespace

using GPU_HipocLaunchLimit_FP32 = testing::TestWithParam<std::size_t>;

TEST_P(GPU_HipocLaunchLimit_FP32, ChecksWorkgroupLimit)
{
    auto&& handle        = get_handle();
    const auto dimension = GetParam();
    const auto arch      = handle.GetDeviceName();
    const auto limit     = miopen::GetMaxWorkgroupsYZ(arch);
    const bool has_limit = limit != std::numeric_limits<std::size_t>::max();

    SCOPED_TRACE(testing::Message() << arch << ", grid dimension " << dimension);

    // Use the regular launch path: a cooperative grid must fit concurrently on the device,
    // so its occupancy limit prevents a successful launch at the Y/Z boundary.
    auto invoke = handle.AddKernel(
        "", "", kProgramName, kKernelName, {1, 1, 1}, {1, 1, 1}, "", 0, kKernelSource);
    std::array host_output{0.0f};
    const auto output = handle.Write(host_output);
    const auto launch = [&](std::size_t workgroups) {
        // One thread per workgroup; only the selected dimension exceeds one workgroup.
        std::array<std::size_t, 3> grid{1, 1, 1};
        grid[dimension] = workgroups;
        invoke.SetGlobalDims(grid[0], grid[1], grid[2]);
        const auto last_y = static_cast<unsigned>(grid[1] - 1);
        const auto last_z = static_cast<unsigned>(grid[2] - 1);
        invoke(output.get(), last_y, last_z);
    };

    ASSERT_NO_THROW(launch(has_limit ? limit : kOversizedGrid));
    ASSERT_NO_THROW(handle.Finish());
    handle.ReadToVec(output, host_output);
    EXPECT_EQ(host_output.front(), 1.0f);

    if(has_limit)
    {
        // Only the invocation is subject to the rejection check, not compilation or Finish().
        EXPECT_THROW(
            {
                try
                {
                    launch(kOversizedGrid);
                }
                catch(const miopen::Exception& e)
                {
                    EXPECT_EQ(e.status, miopenStatusInternalError) << e.what();
                    throw;
                }
            },
            miopen::Exception);
        // Drain the stream even if a regression let the oversized launch through.
        EXPECT_NO_THROW(handle.Finish());
    }
}

TEST_P(GPU_HipocLaunchLimit_FP32, RejectsOversizedCooperativeGrid)
{
    auto&& handle        = get_handle();
    const auto dimension = GetParam();
    const auto arch      = handle.GetDeviceName();
    if(miopen::GetMaxWorkgroupsYZ(arch) == std::numeric_limits<std::size_t>::max())
        GTEST_SKIP() << arch << " has no Y/Z workgroup limit to test";

    SCOPED_TRACE(testing::Message() << arch << ", grid dimension " << dimension);

    // AddKernel returns a regular invoker; fetch the cached kernel to opt in to cooperative launch.
    constexpr char kCooperativeConfig[] = "cooperative";
    handle.AddKernel(kKernelName,
                     kCooperativeConfig,
                     kProgramName,
                     kKernelName,
                     {1, 1, 1},
                     {1, 1, 1},
                     "",
                     0,
                     kKernelSource);
    auto kernels = handle.GetKernelsImpl(kKernelName, kCooperativeConfig);
    ASSERT_EQ(kernels.size(), 1);
    auto invoke = handle.Run(std::move(kernels.front()), /*coop_launch=*/true);

    const auto output = handle.Create<float>(1);
    std::array<std::size_t, 3> grid{1, 1, 1};
    grid[dimension] = kOversizedGrid;
    invoke.SetGlobalDims(grid[0], grid[1], grid[2]);
    const auto last_y = static_cast<unsigned>(grid[1] - 1);
    const auto last_z = static_cast<unsigned>(grid[2] - 1);

    // MIOpen validates the grid before HIP checks cooperative support or occupancy.
    // Only the invocation is subject to the rejection check, not compilation or Finish().
    EXPECT_THROW(
        {
            try
            {
                invoke(output.get(), last_y, last_z);
            }
            catch(const miopen::Exception& e)
            {
                EXPECT_EQ(e.status, miopenStatusInternalError) << e.what();
                throw;
            }
        },
        miopen::Exception);
    // Drain the stream even if a regression let the oversized launch through.
    EXPECT_NO_THROW(handle.Finish());
}

INSTANTIATE_TEST_SUITE_P(Standard, GPU_HipocLaunchLimit_FP32, testing::Values(1, 2));

#endif // MIOPEN_BACKEND_HIP
