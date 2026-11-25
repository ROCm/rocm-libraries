// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "testing_multi_gpu.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
    constexpr int kChildReturnedStatus = 0;
    constexpr int kChildFailed         = 2;
    constexpr int kChildDeviceVisible  = 3;
    constexpr int kChildCaptureAllowed = 4;
    constexpr int kChildTimeoutSec     = 120;
} // namespace

namespace hipblaslt_handle_create_errors
{
    // Runs in a copy of hipblaslt-test started with no visible device, before
    // main() looks for one. The library's old exit(EXIT_FAILURE) ends this
    // process with 1, which matches none of the codes above.
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
    // HIP refuses hipMalloc and hipFree while a stream captures in global mode,
    // so both calls fail inside the library. A destroy that failed leaves the
    // handle valid, and destroying it again after the capture must succeed.
    // Direct hipMalloc and hipFree calls after the library's decide whether this
    // runtime refuses them, so a library that hides the failure cannot skip.
    void createAndDestroyWhileCapturingAndExit()
    {
        hipblasLtHandle_t handle = nullptr;
        void*             probe  = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS
           || hipMalloc(&probe, 256) != hipSuccess)
            std::exit(kChildFailed);
        hipStream_t stream = nullptr;
        if(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess
           || hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess)
            std::exit(kChildFailed);

        hipblasLtHandle_t other     = nullptr;
        const auto        created   = hipblasLtCreate(&other);
        const auto        destroyed = hipblasLtDestroy(handle);
        void*             probe2    = nullptr;
        const bool allowed = hipMalloc(&probe2, 256) == hipSuccess || hipFree(probe) == hipSuccess;

        hipGraph_t graph = nullptr;
        static_cast<void>(hipStreamEndCapture(stream, &graph));
        if(graph != nullptr)
            static_cast<void>(hipGraphDestroy(graph));
        static_cast<void>(hipGetLastError());
        std::fprintf(stderr,
                     "during capture: hipblasLtCreate returned %d, hipblasLtDestroy returned %d\n",
                     int(created),
                     int(destroyed));
        if(allowed)
            std::exit(kChildCaptureAllowed);

        if(created == HIPBLAS_STATUS_SUCCESS || other != nullptr
           || destroyed == HIPBLAS_STATUS_SUCCESS)
            std::exit(kChildFailed);
        const auto retried = hipblasLtDestroy(handle);
        std::fprintf(stderr, "after capture: hipblasLtDestroy returned %d\n", int(retried));
        std::exit(retried == HIPBLAS_STATUS_SUCCESS ? kChildReturnedStatus : kChildFailed);
    }

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

    TEST(HandleCreateErrors, smoke_CreateAndDestroyDuringCaptureReturnStatus)
    {
        // The library's old exit() would end the whole test run, so the case runs
        // in a newly started copy of this binary.
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        bool captureAllowed = false;
        auto passOrSkip     = [&captureAllowed](int status) {
            captureAllowed = ::testing::ExitedWithCode(kChildCaptureAllowed)(status);
            return captureAllowed || ::testing::ExitedWithCode(kChildReturnedStatus)(status);
        };
        EXPECT_EXIT(createAndDestroyWhileCapturingAndExit(), passOrSkip, "");
        if(captureAllowed)
            GTEST_SKIP() << "this HIP runtime allows hipMalloc or hipFree during global capture";
    }
} // namespace
