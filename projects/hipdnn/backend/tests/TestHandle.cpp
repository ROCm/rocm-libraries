// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "handle/Handle.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>

namespace hipdnn_backend
{

TEST(TestHandle, DefaultStreamIsNull)
{
    hipdnnHandle handle;
    EXPECT_EQ(handle.getStream(), nullptr) << "Default stream should be nullptr.";
}

TEST(TestGpuHandle, SetAndGetStream)
{
    SKIP_IF_NO_DEVICES();

    hipdnnHandle handle;

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    handle.setStream(stream);

    EXPECT_EQ(handle.getStream(), stream) << "Stream was not set correctly.";

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}

TEST(TestGpuHandle, SetStreamToNull)
{
    SKIP_IF_NO_DEVICES();

    hipdnnHandle handle;

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess) << "Failed to create HIP stream.";

    handle.setStream(stream);
    EXPECT_EQ(handle.getStream(), stream) << "Stream was not set correctly.";

    handle.setStream(nullptr);
    EXPECT_EQ(handle.getStream(), nullptr) << "Stream should be nullptr after being reset.";

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess) << "Failed to destroy HIP stream.";
}

}
