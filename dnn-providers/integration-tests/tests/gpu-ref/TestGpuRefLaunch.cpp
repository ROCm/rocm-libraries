// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace hipdnn_gpu_ref;

namespace
{
// All validation in the launch helpers happens before the kernel is handed to the
// HIP driver, so a null function is never dereferenced by these error-path tests.
hipFunction_t nullFunction = nullptr;
int dummyArgs = 0;

int64_t maxGridDim(int dim)
{
    int deviceId = 0;
    EXPECT_EQ(hipGetDevice(&deviceId), hipSuccess);
    hipDeviceProp_t props{};
    EXPECT_EQ(hipGetDeviceProperties(&props, deviceId), hipSuccess);
    return static_cast<int64_t>(props.maxGridSize[dim]);
}
} // namespace

// ============================================================================
// checkedNarrowToUInt — narrowing contract
// ============================================================================

TEST(TestGpuRefLaunch, CheckedNarrowAcceptsInRangeValues)
{
    EXPECT_EQ(detail::checkedNarrowToUInt(0), 0U);
    EXPECT_EQ(detail::checkedNarrowToUInt(1), 1U);
    EXPECT_EQ(detail::checkedNarrowToUInt(std::numeric_limits<unsigned int>::max()),
              std::numeric_limits<unsigned int>::max());
}

TEST(TestGpuRefLaunch, CheckedNarrowThrowsAboveUIntMax)
{
    constexpr int64_t TOO_LARGE
        = static_cast<int64_t>(std::numeric_limits<unsigned int>::max()) + 1;
    EXPECT_THROW(detail::checkedNarrowToUInt(TOO_LARGE, "X grid size"), std::runtime_error);
    EXPECT_THROW(detail::checkedNarrowToUInt(std::numeric_limits<int64_t>::max()),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, CheckedNarrowThrowsOnNegativeValues)
{
    EXPECT_THROW(detail::checkedNarrowToUInt(-1), std::runtime_error);
    EXPECT_THROW(detail::checkedNarrowToUInt(std::numeric_limits<int64_t>::min()),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, CheckedNarrowErrorMessageNamesTheDimension)
{
    constexpr int64_t TOO_LARGE
        = static_cast<int64_t>(std::numeric_limits<unsigned int>::max()) + 1;
    try
    {
        detail::checkedNarrowToUInt(TOO_LARGE, "Z block size");
        FAIL() << "expected checkedNarrowToUInt to throw";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("Z block size"), std::string::npos);
    }
}

// ============================================================================
// launchKernel — geometry validation before the driver call
// ============================================================================

TEST(TestGpuRefLaunch, LaunchKernelThrowsOnOutOfRangeGridDimension)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t TOO_LARGE
        = static_cast<int64_t>(std::numeric_limits<unsigned int>::max()) + 1;
    EXPECT_THROW(
        detail::launchKernel(nullFunction, {TOO_LARGE, 1, 1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
        std::runtime_error);
    EXPECT_THROW(
        detail::launchKernel(nullFunction, {1, TOO_LARGE, 1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
        std::runtime_error);
    EXPECT_THROW(
        detail::launchKernel(nullFunction, {1, 1, TOO_LARGE}, {1, 1, 1}, &dummyArgs, sizeof(int)),
        std::runtime_error);
}

TEST(TestGpuRefLaunch, LaunchKernelThrowsOnOutOfRangeBlockDimension)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t TOO_LARGE
        = static_cast<int64_t>(std::numeric_limits<unsigned int>::max()) + 1;
    EXPECT_THROW(
        detail::launchKernel(nullFunction, {1, 1, 1}, {TOO_LARGE, 1, 1}, &dummyArgs, sizeof(int)),
        std::runtime_error);
    EXPECT_THROW(detail::launchKernel(nullFunction, {1, 1, 1}, {1, 1, -1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, LaunchKernelThrowsOnZeroGridDimension)
{
    SKIP_IF_NO_DEVICES();

    EXPECT_THROW(detail::launchKernel(nullFunction, {0, 1, 1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernel(nullFunction, {1, 0, 1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernel(nullFunction, {1, 1, 0}, {1, 1, 1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, LaunchKernelThrowsOnNegativeGridDimension)
{
    SKIP_IF_NO_DEVICES();

    EXPECT_THROW(detail::launchKernel(nullFunction, {-1, 1, 1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernel(nullFunction, {1, -1, 1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernel(nullFunction, {1, 1, -1}, {1, 1, 1}, &dummyArgs, sizeof(int)),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, LaunchKernelThrowsWhenGridExceedsDeviceLimit)
{
    SKIP_IF_NO_DEVICES();

    // maxGridSize is expressed in blocks; one block past the limit must be rejected.
    for(int dim = 0; dim < 3; ++dim)
    {
        const int64_t limit = maxGridDim(dim);
        if(limit >= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
        {
            continue; // narrowing check already covers this dimension
        }
        std::array<int64_t, 3> grid{1, 1, 1};
        grid[static_cast<size_t>(dim)] = limit + 1;
        EXPECT_THROW(detail::launchKernel(nullFunction, grid, {1, 1, 1}, &dummyArgs, sizeof(int)),
                     std::runtime_error);
    }
}

// ============================================================================
// launchKernel1d / launchKernelForElements — element and block count contracts
// ============================================================================

TEST(TestGpuRefLaunch, LaunchKernel1dThrowsOnNonPositiveGrid)
{
    SKIP_IF_NO_DEVICES();

    EXPECT_THROW(detail::launchKernel1d(nullFunction, 0, 256, &dummyArgs, sizeof(int)),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernel1d(nullFunction, -1, 256, &dummyArgs, sizeof(int)),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, LaunchForElementsThrowsOnNonPositiveBlockSize)
{
    SKIP_IF_NO_DEVICES();

    EXPECT_THROW(detail::launchKernelForElements(nullFunction, 1024, &dummyArgs, sizeof(int), 0),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernelForElements(nullFunction, 1024, &dummyArgs, sizeof(int), -8),
                 std::runtime_error);
}

TEST(TestGpuRefLaunch, LaunchForElementsThrowsOnNonPositiveElementCount)
{
    SKIP_IF_NO_DEVICES();

    EXPECT_THROW(detail::launchKernelForElements(nullFunction, 0, &dummyArgs, sizeof(int)),
                 std::runtime_error);
    EXPECT_THROW(detail::launchKernelForElements(nullFunction, -1, &dummyArgs, sizeof(int)),
                 std::runtime_error);
}
