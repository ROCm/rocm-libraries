// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_data_sdk::utilities;

namespace
{

template <typename T>
void initBuffer(T* buffer, size_t size, T mult = 1)
{
    for(size_t i = 0; i < size; ++i)
    {
        buffer[i] = static_cast<T>(i) * mult;
    }
}

template <typename T>
void checkBuffer(const T* buffer, size_t size, T mult = 1)
{
    for(size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(buffer[i], static_cast<T>(i) * mult);
    }
}

template <typename T>
void checkBufferSynchronized(const T* buffer, size_t size, hipStream_t stream = nullptr, T mult = 1)
{
    const hipError_t error = hipStreamSynchronize(stream);
    EXPECT_EQ(error, hipSuccess) << "Error synchronizing stream";

    for(size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(buffer[i], static_cast<T>(i) * mult);
    }
}

} // namespace

TEST(TestMigratableMemory, NotInitialized)
{
    MigratableMemory<float> memory;

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.count(), 0);
    EXPECT_EQ(memory.location(), MemoryLocation::NONE);
    EXPECT_EQ(memory.hostData(), nullptr);
}

TEST(TestMigratableMemory, InitializeWithSize)
{
    MigratableMemory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);
    EXPECT_NE(memory.hostData(), nullptr);
}

TEST(TestMigratableMemory, MoveConstructor)
{
    MigratableMemory<float> memory1(10);
    auto* oldHostData = memory1.hostData();

    MigratableMemory<float> memory2(std::move(memory1));

    EXPECT_TRUE(memory1.empty()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(memory1.count(), 0);
    EXPECT_EQ(memory1.location(), MemoryLocation::NONE);
    EXPECT_EQ(memory1.hostData(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.count(), 10);
    EXPECT_EQ(memory2.location(), MemoryLocation::HOST);
    EXPECT_NE(memory2.hostData(), nullptr);
    EXPECT_EQ(memory2.hostData(), oldHostData);
}

TEST(TestMigratableMemory, MoveAssignment)
{
    MigratableMemory<float> memory1(10);
    auto* oldHostData = memory1.hostData();

    MigratableMemory<float> memory2;
    memory2 = std::move(memory1);

    EXPECT_TRUE(memory1.empty()); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(memory1.count(), 0);
    EXPECT_EQ(memory1.location(), MemoryLocation::NONE);
    EXPECT_EQ(memory1.hostData(), nullptr);

    EXPECT_FALSE(memory2.empty());
    EXPECT_EQ(memory2.count(), 10);
    EXPECT_EQ(memory2.location(), MemoryLocation::HOST);
    EXPECT_NE(memory2.hostData(), nullptr);
    EXPECT_EQ(memory2.hostData(), oldHostData);
}

TEST(TestMigratableMemory, Resize)
{
    MigratableMemory<float> memory(10);
    memory.resize(20);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 20);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);
    EXPECT_NE(memory.hostData(), nullptr);
}

TEST(TestMigratableMemory, MigrateToDevice)
{
    SKIP_IF_NO_DEVICES();

    MigratableMemory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);

    initBuffer(memory.hostData(), memory.count());

    EXPECT_NE(memory.deviceData(), nullptr);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    initBuffer(memory.hostData(), memory.count(), 0.0f);
    memory.markDeviceModified();
    checkBuffer(static_cast<float*>(memory.hostData()), memory.count());
}

TEST(TestMigratableMemory, MigrateToDeviceNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    MigratableMemory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);

    initBuffer(memory.hostData(), memory.count());

    EXPECT_NE(memory.deviceData(), nullptr);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    initBuffer(memory.hostData(), memory.count(), 0.0f);
    memory.markDeviceModified();
    checkBufferSynchronized(static_cast<float*>(memory.hostData()), memory.count(), stream);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(TestMigratableMemory, MigrateToDeviceAsyncNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    MigratableMemory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);

    initBuffer(memory.hostData(), memory.count());

    EXPECT_NE(memory.deviceDataAsync(), nullptr);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    initBuffer(memory.hostData(), memory.count(), 0.0f);
    memory.markDeviceModified();
    checkBufferSynchronized(static_cast<float*>(memory.hostDataAsync()), memory.count(), stream);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(TestMigratableMemory, MigrateToHost)
{
    SKIP_IF_NO_DEVICES();

    MigratableMemory<float> memory(10);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);

    initBuffer(memory.hostData(), memory.count());

    auto tmpPtr = memory.deviceData();
    EXPECT_NE(tmpPtr, nullptr);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    std::array<float, 10> array;
    initBuffer(array.data(), 10, 2.0f);
    const hipError_t err = hipMemcpy(
        memory.deviceData(), array.data(), memory.count() * sizeof(float), hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess);
    memory.markDeviceModified();
    EXPECT_EQ(memory.location(), MemoryLocation::DEVICE);

    checkBufferSynchronized(memory.hostData(), memory.count(), nullptr, 2.0f);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);
}

TEST(TestMigratableMemory, MigrateToHostNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    MigratableMemory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);

    initBuffer(memory.hostData(), memory.count());

    auto tmpPtr = memory.deviceData();
    EXPECT_NE(tmpPtr, nullptr);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    std::array<float, 10> array;
    initBuffer(array.data(), 10, 2.0f);
    const hipError_t err = hipMemcpyWithStream(memory.deviceData(),
                                               array.data(),
                                               memory.count() * sizeof(float),
                                               hipMemcpyHostToDevice,
                                               stream);
    EXPECT_EQ(err, hipSuccess);
    memory.markDeviceModified();
    EXPECT_EQ(memory.location(), MemoryLocation::DEVICE);

    checkBufferSynchronized(memory.hostData(), memory.count(), stream, 2.0f);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(TestMigratableMemory, MigrateToHostAsyncNonDefaultStream)
{
    SKIP_IF_NO_DEVICES();

    hipStream_t stream;
    hipError_t error = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    EXPECT_EQ(error, hipSuccess) << "Failed to create HIP stream";
    ASSERT_NE(stream, nullptr) << "Failed to create HIP stream";

    MigratableMemory<float> memory(10, stream);

    EXPECT_FALSE(memory.empty());
    EXPECT_EQ(memory.count(), 10);
    EXPECT_EQ(memory.location(), MemoryLocation::HOST);

    initBuffer(memory.hostData(), memory.count());

    auto tmpPtr = memory.deviceDataAsync();
    EXPECT_NE(tmpPtr, nullptr);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    std::array<float, 10> array;
    initBuffer(array.data(), 10, 2.0f);
    const hipError_t err = hipMemcpyWithStream(memory.deviceData(),
                                               array.data(),
                                               memory.count() * sizeof(float),
                                               hipMemcpyHostToDevice,
                                               stream);
    EXPECT_EQ(err, hipSuccess);
    memory.markDeviceModified();
    EXPECT_EQ(memory.location(), MemoryLocation::DEVICE);

    checkBufferSynchronized(memory.hostDataAsync(), memory.count(), stream, 2.0f);
    EXPECT_EQ(memory.location(), MemoryLocation::BOTH);

    error = hipStreamDestroy(stream);
    EXPECT_EQ(error, hipSuccess) << "Failed to destroy HIP stream";
}

TEST(TestMigratableMemory, Clear)
{
    SKIP_IF_NO_DEVICES();

    MigratableMemory<float> memory(10);
    memory.clear();

    EXPECT_TRUE(memory.empty());
    EXPECT_EQ(memory.count(), 0);
    EXPECT_EQ(memory.location(), MemoryLocation::NONE);
    EXPECT_EQ(memory.hostData(), nullptr);
    EXPECT_EQ(memory.deviceData(), nullptr);
}
