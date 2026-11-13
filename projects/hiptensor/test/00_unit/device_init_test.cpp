/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace
{
    constexpr char const* kNoDeviceChildEnv = "HIPTENSOR_DEVICE_INIT_TEST_NO_DEVICE_CHILD";

    int visibleDeviceCount()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess)
        {
            return 0;
        }
        return count;
    }

    // Creates a plan for a small f16 contraction D[m,n] = A[m,k] * B[n,k] using the given handle
    // and returns the status of hiptensorCreatePlan, which validates the handle's device.
    // 16F compute has matrix-core support on every architecture hipTensor supports. A failure in
    // any earlier step is reported as a test failure and returned as INTERNAL_ERROR, so that it
    // cannot be mistaken for the ARCH_MISMATCH a device check produces.
    hiptensorStatus_t createContractionPlan(hiptensorHandle_t handle)
    {
        int64_t lensA[2] = {8, 8};
        int64_t lensB[2] = {8, 8};
        int64_t lensD[2] = {8, 8};
        int32_t modeA[2] = {'m', 'k'};
        int32_t modeB[2] = {'n', 'k'};
        int32_t modeD[2] = {'m', 'n'};

        hiptensorTensorDescriptor_t    descA{}, descB{}, descD{};
        hiptensorOperationDescriptor_t opDesc{};
        hiptensorPlanPreference_t      pref{};
        hiptensorPlan_t                plan{};
        auto                           status = HIPTENSOR_STATUS_INTERNAL_ERROR;

        bool setupOk = hiptensorCreateTensorDescriptor(
                           handle, &descA, 2, lensA, nullptr, HIPTENSOR_R_16F, 128)
                           == HIPTENSOR_STATUS_SUCCESS
                       && hiptensorCreateTensorDescriptor(
                              handle, &descB, 2, lensB, nullptr, HIPTENSOR_R_16F, 128)
                              == HIPTENSOR_STATUS_SUCCESS
                       && hiptensorCreateTensorDescriptor(
                              handle, &descD, 2, lensD, nullptr, HIPTENSOR_R_16F, 128)
                              == HIPTENSOR_STATUS_SUCCESS
                       && hiptensorCreateContraction(handle,
                                                     &opDesc,
                                                     descA,
                                                     modeA,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descB,
                                                     modeB,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     nullptr,
                                                     nullptr,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descD,
                                                     modeD,
                                                     HIPTENSOR_COMPUTE_DESC_16F)
                              == HIPTENSOR_STATUS_SUCCESS
                       && hiptensorCreatePlanPreference(
                              handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE)
                              == HIPTENSOR_STATUS_SUCCESS;

        if(setupOk)
        {
            status = hiptensorCreatePlan(handle, &plan, opDesc, pref, 0);
        }
        else
        {
            ADD_FAILURE() << "Failed to set up the contraction before creating the plan";
        }

        hiptensorDestroyPlan(plan);
        hiptensorDestroyPlanPreference(pref);
        hiptensorDestroyOperationDescriptor(opDesc);
        hiptensorDestroyTensorDescriptor(descA);
        hiptensorDestroyTensorDescriptor(descB);
        hiptensorDestroyTensorDescriptor(descD);

        return status;
    }
} // namespace

TEST(DeviceInitTest, CreateOnCurrentDevice)
{
    if(std::getenv(kNoDeviceChildEnv) != nullptr)
    {
        GTEST_SKIP() << "Runs only in the parent process";
    }

    hiptensorHandle_t handle = reinterpret_cast<hiptensorHandle_t>(0x1);
    auto              status = hiptensorCreate(&handle);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        EXPECT_EQ(handle, nullptr);
        GTEST_SKIP() << "Cannot proceed: unsupported host device detected";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(createContractionPlan(handle), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorDestroy(handle), HIPTENSOR_STATUS_SUCCESS);
}

TEST(DeviceInitTest, HandleIsBoundToCreationDevice)
{
    if(std::getenv(kNoDeviceChildEnv) != nullptr)
    {
        GTEST_SKIP() << "Runs only in the parent process";
    }
    if(visibleDeviceCount() < 2)
    {
        GTEST_SKIP() << "Requires at least two visible devices";
    }

    hiptensorHandle_t handle0 = nullptr;
    hiptensorHandle_t handle1 = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    auto status0 = hiptensorCreate(&handle0);
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    auto status1 = hiptensorCreate(&handle1);
    if(status0 == HIPTENSOR_STATUS_ARCH_MISMATCH || status1 == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        hiptensorDestroy(handle0);
        hiptensorDestroy(handle1);
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        GTEST_SKIP() << "Requires devices 0 and 1 to be supported by hipTensor";
    }
    ASSERT_EQ(status0, HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(status1, HIPTENSOR_STATUS_SUCCESS);

    // handle0 succeeding on device 0 shows the contraction is supported there, so the
    // ARCH_MISMATCH for handle1 on device 0 can only come from the device check.
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    EXPECT_EQ(createContractionPlan(handle0), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(createContractionPlan(handle1), HIPTENSOR_STATUS_ARCH_MISMATCH);

    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    EXPECT_EQ(createContractionPlan(handle1), HIPTENSOR_STATUS_SUCCESS);

    EXPECT_EQ(hiptensorDestroy(handle0), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorDestroy(handle1), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
}

// Each thread selects its own device. An unsupported device must be rejected for that thread only,
// without affecting the threads that selected a supported device.
TEST(DeviceInitTest, ThreadsSelectTheirOwnDevice)
{
    if(std::getenv(kNoDeviceChildEnv) != nullptr)
    {
        GTEST_SKIP() << "Runs only in the parent process";
    }
    int numThreads = std::min(visibleDeviceCount(), 4);
    if(numThreads < 2)
    {
        GTEST_SKIP() << "Requires at least two visible devices";
    }

    std::vector<hiptensorStatus_t> createStatus(numThreads, HIPTENSOR_STATUS_INTERNAL_ERROR);
    std::vector<hiptensorStatus_t> planStatus(numThreads, HIPTENSOR_STATUS_INTERNAL_ERROR);
    std::vector<char>              handleCleared(numThreads, 0);
    std::vector<std::thread>       threads;

    for(int i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([i, &createStatus, &planStatus, &handleCleared]() {
            if(hipSetDevice(i) != hipSuccess)
            {
                return;
            }
            hiptensorHandle_t handle = nullptr;
            createStatus[i]          = hiptensorCreate(&handle);
            handleCleared[i]         = handle == nullptr;
            if(createStatus[i] == HIPTENSOR_STATUS_SUCCESS)
            {
                planStatus[i] = createContractionPlan(handle);
                hiptensorDestroy(handle);
            }
        });
    }
    for(auto& t : threads)
    {
        t.join();
    }

    int supported = 0;
    for(int i = 0; i < numThreads; ++i)
    {
        if(createStatus[i] == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            EXPECT_TRUE(handleCleared[i]) << "device " << i;
            continue;
        }
        ++supported;
        EXPECT_EQ(createStatus[i], HIPTENSOR_STATUS_SUCCESS) << "device " << i;
        EXPECT_EQ(planStatus[i], HIPTENSOR_STATUS_SUCCESS) << "device " << i;
    }
    if(supported == 0)
    {
        GTEST_SKIP() << "Cannot proceed: unsupported host device detected";
    }
}

// Reached only in a child process started with no visible devices. Getting here at all shows
// that loading hipTensor did not terminate the process before main().
TEST(DeviceInitTest, CreateWithoutDeviceReturnsError)
{
    if(std::getenv(kNoDeviceChildEnv) == nullptr)
    {
        GTEST_SKIP() << "Runs only in the no-device child process";
    }

    hiptensorHandle_t handle = reinterpret_cast<hiptensorHandle_t>(0x1);
    EXPECT_NE(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(handle, nullptr);
}

TEST(DeviceInitTest, LoadWithoutDeviceDoesNotTerminate)
{
#ifdef _WIN32
    GTEST_SKIP() << "Uses /proc/self/exe and POSIX environment handling";
#else
    if(std::getenv(kNoDeviceChildEnv) != nullptr)
    {
        GTEST_SKIP() << "Runs only in the parent process";
    }

    auto self = std::filesystem::read_symlink("/proc/self/exe").string();
    auto cmd  = std::string("HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 ") + kNoDeviceChildEnv
               + "=1 '" + self + "' --gtest_filter=DeviceInitTest.CreateWithoutDeviceReturnsError";

    int rc = std::system(cmd.c_str());
    ASSERT_NE(rc, -1);
    ASSERT_TRUE(WIFEXITED(rc)) << "child did not exit normally";
    EXPECT_EQ(WEXITSTATUS(rc), 0) << "child failed or was terminated by hipTensor";
#endif
}

// The shared test main exits when no device is visible, but the no-device child must reach its test.
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
