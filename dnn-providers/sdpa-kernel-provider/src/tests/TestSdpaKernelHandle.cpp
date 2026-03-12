// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "SdpaKernelContainer.hpp"
#include "SdpaKernelHandle.hpp"

TEST(TestSdpaKernelHandle, ConstructsAndDestructsSuccessfully)
{
    SdpaKernelHandle handle;
}

TEST(TestSdpaKernelHandle, SetAndGetStream)
{
    SdpaKernelHandle handle;

    EXPECT_EQ(handle.getStream(), nullptr);

    // Use nullptr as a stand-in for a real stream (no GPU required)
    hipStream_t stream = nullptr;
    handle.setStream(stream);

    EXPECT_EQ(handle.getStream(), stream);
}

TEST(TestSdpaKernelHandle, GetEngineManagerWithContainer)
{
    SdpaKernelHandle handle;
    handle.container = std::make_shared<sdpa_kernel_provider::SdpaKernelContainer>();

    auto& engineManager = handle.getEngineManager();
    (void)engineManager;
}

TEST(TestSdpaKernelHandle, StoreAndRemoveEngineDetailsBuffer)
{
    SdpaKernelHandle handle;

    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>();
    const void* ptr = buffer->data();

    handle.storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));
    handle.removeEngineDetailsDetachedBuffer(ptr);
}

TEST(TestSdpaKernelHandle, RemoveNonExistentBufferDoesNotThrow)
{
    SdpaKernelHandle handle;

    const void* ptr = reinterpret_cast<const void*>(0x1234);
    EXPECT_NO_THROW(handle.removeEngineDetailsDetachedBuffer(ptr));
}
