// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "core/Container.hpp"
#include "core/Handle.hpp"

TEST(TestHandle, ConstructsAndDestructsSuccessfully)
{
    const Handle handle;
}

TEST(TestHandle, SetAndGetStream)
{
    Handle handle;

    EXPECT_EQ(handle.getStream(), nullptr);

    // Use nullptr as a stand-in for a real stream (no GPU required)
    hipStream_t stream = nullptr;
    handle.setStream(stream);

    EXPECT_EQ(handle.getStream(), stream);
}

TEST(TestHandle, GetEngineManagerWithContainer)
{
    Handle handle;
    handle.container = std::make_shared<hip_kernel_provider::core::Container>();

    auto& engineManager = handle.getEngineManager();
    (void)engineManager;
}

TEST(TestHandle, StoreAndRemoveEngineDetailsBuffer)
{
    Handle handle;

    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>();
    const void* ptr = buffer->data();

    handle.storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));
    handle.removeEngineDetailsDetachedBuffer(ptr);
}

TEST(TestHandle, RemoveNonExistentBufferDoesNotThrow)
{
    Handle handle;

    const void* ptr = reinterpret_cast<const void*>(0x1234);
    EXPECT_NO_THROW(handle.removeEngineDetailsDetachedBuffer(ptr));
}
