// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <memory>
#include <type_traits>

namespace
{

void destroyStream(hipStream_t stream)
{
    EXPECT_EQ(hipSuccess, hipStreamDestroy(stream));
}

class TestGpuDeviceQuery : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipSuccess, hipGetDeviceCount(&_deviceCount));
        ASSERT_EQ(hipSuccess, hipGetDevice(&_originalDevice));
    }

    ~TestGpuDeviceQuery() override
    {
        // The owned stream is created on the original device. Restore that device even
        // when a fatal assertion or exception interrupts a cross-device query.
        if(_originalDevice >= 0)
        {
            EXPECT_EQ(hipSuccess, hipSetDevice(_originalDevice));
        }
        _stream.reset();
    }

    int _deviceCount = 0;
    int _originalDevice = -1;
    std::unique_ptr<std::remove_pointer_t<hipStream_t>, decltype(&destroyStream)> _stream{
        nullptr, destroyStream};
};

TEST_F(TestGpuDeviceQuery, DefaultStreamsFollowLiveCurrentDevice)
{
    const std::array<hipStream_t, 3> streams{nullptr, hipStreamLegacy, hipStreamPerThread};

    // Visit every visible device and return to the original one, querying each token
    // after every switch so no token can retain a previously resolved ordinal.
    for(int step = 0; step <= _deviceCount; ++step)
    {
        const int currentDevice = (_originalDevice + step) % _deviceCount;
        ASSERT_EQ(hipSuccess, hipSetDevice(currentDevice));
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipSuccess, hipGetDeviceProperties(&properties, currentDevice));

        for(const auto stream : streams)
        {
            SCOPED_TRACE(::testing::Message()
                         << "device " << currentDevice << ", stream " << stream);
            EXPECT_TRUE(hipdnn_plugin_sdk::isDefaultStream(stream));
            hipDevice_t deviceId = -1;
            ASSERT_EQ(hipSuccess, hipdnn_plugin_sdk::getDeviceFromStream(stream, &deviceId));
            EXPECT_EQ(currentDevice, deviceId);
            EXPECT_EQ(hipdnn_plugin_sdk::getDeviceArch(stream), properties.gcnArchName);
        }
    }
}

TEST_F(TestGpuDeviceQuery, ConcreteStreamKeepsOwningDevice)
{
    if(_deviceCount < 2)
    {
        GTEST_SKIP() << "Concrete stream ownership requires at least two devices";
    }

    hipStream_t stream = nullptr;
    const hipError_t status = hipStreamCreate(&stream);
    _stream.reset(stream);
    ASSERT_EQ(hipSuccess, status);

    const int otherDevice = (_originalDevice + 1) % _deviceCount;
    ASSERT_EQ(hipSuccess, hipSetDevice(otherDevice));
    EXPECT_FALSE(hipdnn_plugin_sdk::isDefaultStream(_stream.get()));
    hipDevice_t deviceId = -1;
    ASSERT_EQ(hipSuccess, hipdnn_plugin_sdk::getDeviceFromStream(_stream.get(), &deviceId));
    EXPECT_EQ(_originalDevice, deviceId);

    int currentDevice = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&currentDevice));
    EXPECT_EQ(otherDevice, currentDevice);
}

} // namespace
