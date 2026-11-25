// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblasLtCreate reports a failed HIP call as a status instead of ending the
// process (ROCM-32008).

#include "testing_multi_gpu.hpp"

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <vector>

namespace
{
    constexpr int kChildReturnedStatus = 0;
    constexpr int kChildDeviceVisible  = 3;
    constexpr int kChildTimeoutSec     = 120;
} // namespace

namespace hipblaslt_handle_create_errors
{
    // Runs in a copy of hipblaslt-test started with no visible device, before
    // main() looks for one. The library's old exit(EXIT_FAILURE) ends this
    // process with 1, which matches neither code below.
    int run_no_device_child()
    {
        hipblasLtHandle_t handle = nullptr;
        const auto        status = hipblasLtCreate(&handle);
        std::fprintf(stderr, "hipblasLtCreate returned %d\n", int(status));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            static_cast<void>(hipblasLtDestroy(handle));
            return kChildDeviceVisible;
        }
        return kChildReturnedStatus;
    }
} // namespace hipblaslt_handle_create_errors

namespace
{
    TEST(HandleCreateErrors, smoke_CreateWithNoVisibleDeviceReturnsStatus)
    {
        std::vector<pid_t> pids(1, -1);
        const int          spawned = hipblaslt_bench::spawn_rank(
            "--handle-create-no-device-child", 0, 1, 0, {{"HIP_VISIBLE_DEVICES", "-1"}}, pids[0]);
        ASSERT_EQ(spawned, 0) << "posix_spawn -> errno " << spawned;

        std::vector<int> codes;
        if(!hipblaslt_bench::wait_for_ranks(pids, codes, kChildTimeoutSec))
        {
            hipblaslt_bench::kill_ranks(pids);
            FAIL() << "the child did not finish within " << kChildTimeoutSec << "s";
        }
        if(codes[0] == kChildDeviceVisible)
            GTEST_SKIP() << "HIP_VISIBLE_DEVICES=-1 did not hide the device";
        EXPECT_EQ(codes[0], kChildReturnedStatus)
            << "exit code 1 means hipblasLtCreate ended the process instead of returning";
    }
} // namespace
