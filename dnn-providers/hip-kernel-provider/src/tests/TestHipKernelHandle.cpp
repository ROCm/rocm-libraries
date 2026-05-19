// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipKernelContainer.hpp"
#include "HipKernelHandle.hpp"

TEST(TestHipKernelHandle, ConstructsAndDestructsSuccessfully)
{
    const HipKernelHandle handle;
}

TEST(TestHipKernelHandle, SetAndGetStream)
{
    HipKernelHandle handle;

    EXPECT_EQ(handle.getStream(), nullptr);

    // Use nullptr as a stand-in for a real stream (no GPU required)
    hipStream_t stream = nullptr;
    handle.setStream(stream);

    EXPECT_EQ(handle.getStream(), stream);
}

TEST(TestHipKernelHandle, GetEngineManagerWithContainer)
{
    HipKernelHandle handle;
    handle.container = std::make_shared<hip_kernel_provider::HipKernelContainer>();

    auto& engineManager = handle.getEngineManager();
    (void)engineManager;
}

TEST(TestHipKernelHandle, StoreAndRemoveEngineDetailsBuffer)
{
    HipKernelHandle handle;

    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>();
    const void* ptr = buffer->data();

    handle.storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));
    handle.removeEngineDetailsDetachedBuffer(ptr);
}

TEST(TestHipKernelHandle, RemoveNonExistentBufferDoesNotThrow)
{
    HipKernelHandle handle;

    const void* ptr = reinterpret_cast<const void*>(0x1234);
    EXPECT_NO_THROW(handle.removeEngineDetailsDetachedBuffer(ptr));
}
