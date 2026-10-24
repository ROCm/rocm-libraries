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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hiptensor/hiptensor.h>

// This binary has its own main: the shared test main stops before any test runs when no GPU is
// visible or when the device is not supported, which are exactly the cases tested here.
//
// The library used to end the process from a static initializer when it was loaded on a host
// whose GPU it does not list, or on a host without a GPU. On those hosts this binary therefore
// failed before main. hiptensorCreate now reports both cases as a status.

namespace
{
    // ctest reports a test whose output contains "HIPTENSOR_STATUS_ARCH_MISMATCH" as skipped, and
    // gtest prints the expressions of a failed assertion, so the expected status is named
    // differently here to keep a failure reported as a failure.
    constexpr auto kArchMismatch = HIPTENSOR_STATUS_ARCH_MISMATCH;

    // Sets an environment variable for the lifetime of the object.
    class ScopedEnvironmentVariable
    {
    public:
        ScopedEnvironmentVariable(char const* name, char const* value)
            : mName(name)
        {
#ifdef _WIN32
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        }

        ~ScopedEnvironmentVariable()
        {
#ifdef _WIN32
            _putenv_s(mName.c_str(), "");
#else
            unsetenv(mName.c_str());
#endif
        }

        ScopedEnvironmentVariable(ScopedEnvironmentVariable const&)            = delete;
        ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable const&) = delete;

    private:
        std::string mName;
    };

    bool hasVisibleDevice()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }
} // namespace

// HIPTENSOR_TEST_FORCE_UNSUPPORTED_ARCH=ON makes the library classify the current device as an
// architecture it does not support, so that case can be simulated on any GPU.
TEST(DeviceSupportTest, UnsupportedArchitectureCreateReturnsArchMismatch)
{
    if(!hasVisibleDevice())
    {
        GTEST_SKIP() << "No GPU is visible";
    }

    {
        ScopedEnvironmentVariable arch("HIPTENSOR_TEST_FORCE_UNSUPPORTED_ARCH", "ON");

        hiptensorHandle_t handle = nullptr;
        EXPECT_EQ(hiptensorCreate(&handle), kArchMismatch);
        EXPECT_EQ(handle, nullptr);
    }

    // Positive control in the same process: without the override the handle is created.
    hiptensorHandle_t handle = nullptr;
    auto              status = hiptensorCreate(&handle);
    if(status == kArchMismatch)
    {
        GTEST_SKIP() << "The current GPU itself is not supported, so there is no positive control";
    }
    ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(hiptensorDestroy(handle), HIPTENSOR_STATUS_SUCCESS);
}

// The hook only takes support away when it is enabled.
TEST(DeviceSupportTest, DisabledUnsupportedArchitectureHookKeepsTheDevice)
{
    if(!hasVisibleDevice())
    {
        GTEST_SKIP() << "No GPU is visible";
    }

    hiptensorHandle_t handle = nullptr;
    if(hiptensorCreate(&handle) == kArchMismatch)
    {
        GTEST_SKIP() << "The current GPU itself is not supported";
    }
    ASSERT_EQ(hiptensorDestroy(handle), HIPTENSOR_STATUS_SUCCESS);

    ScopedEnvironmentVariable arch("HIPTENSOR_TEST_FORCE_UNSUPPORTED_ARCH", "OFF");
    handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(hiptensorDestroy(handle), HIPTENSOR_STATUS_SUCCESS);
}

// Runs for real only when no GPU is visible: on a GPU-less host, or as the
// device_support_test_no_visible_device ctest entry, which hides every GPU.
TEST(DeviceSupportTest, NoVisibleDeviceCreateReturnsHipError)
{
    if(hasVisibleDevice())
    {
        GTEST_SKIP() << "A GPU is visible; set HIP_VISIBLE_DEVICES=-1 to run this test";
    }

    hiptensorHandle_t handle = nullptr;
    EXPECT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_HIP_ERROR);
    EXPECT_EQ(handle, nullptr);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
