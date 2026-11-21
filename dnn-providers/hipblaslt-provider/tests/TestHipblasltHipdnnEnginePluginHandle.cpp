// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "HipdnnEnginePluginHandle.hpp"

class TestHipblasltHipdnnEnginePluginHandle : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _handle = std::make_unique<HipdnnEnginePluginHandle>();
    }

    std::unique_ptr<HipdnnEnginePluginHandle> _handle;
};

TEST_F(TestHipblasltHipdnnEnginePluginHandle, DefaultConstruction)
{
    EXPECT_EQ(_handle->hipblasltHandle, nullptr);
    EXPECT_EQ(_handle->hipblasltContainer, nullptr);
    EXPECT_EQ(_handle->getStream(), nullptr);
}

TEST_F(TestHipblasltHipdnnEnginePluginHandle, StoreDetachedBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    auto createdString = builder.CreateString("test");
    builder.Finish(createdString);
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    const void* ptr = reinterpret_cast<const void*>(0x12354);

    _handle->storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));

    // Buffer should be stored, verify by trying to remove it
    _handle->removeEngineDetailsDetachedBuffer(ptr);
}

TEST_F(TestHipblasltHipdnnEnginePluginHandle, RemoveDetachedBuffer)
{
    flatbuffers::FlatBufferBuilder builder;
    auto createdString = builder.CreateString("test");
    builder.Finish(createdString);
    auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    const void* ptr = reinterpret_cast<const void*>(0x12354);

    _handle->storeEngineDetailsDetachedBuffer(ptr, std::move(buffer));
    _handle->removeEngineDetailsDetachedBuffer(ptr);

    // Should not crash when removing non-existent buffer
    _handle->removeEngineDetailsDetachedBuffer(ptr);
}

TEST_F(TestHipblasltHipdnnEnginePluginHandle, RemoveNonExistentBuffer)
{
    const void* fakePtr = reinterpret_cast<const void*>(0x12345678);

    // Should not crash when removing non-existent buffer
    EXPECT_NO_THROW(_handle->removeEngineDetailsDetachedBuffer(fakePtr));
}

TEST_F(TestHipblasltHipdnnEnginePluginHandle, MultipleBuffers)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto createdString1 = builder1.CreateString("test");
    builder1.Finish(createdString1);
    auto buffer1 = std::make_unique<flatbuffers::DetachedBuffer>(builder1.Release());
    const void* ptr1 = reinterpret_cast<const void*>(0x12354);

    flatbuffers::FlatBufferBuilder builder2;
    auto createdString2 = builder2.CreateString("test2");
    builder2.Finish(createdString2);
    auto buffer2 = std::make_unique<flatbuffers::DetachedBuffer>(builder2.Release());
    const void* ptr2 = reinterpret_cast<const void*>(0x54311);

    _handle->storeEngineDetailsDetachedBuffer(ptr1, std::move(buffer1));
    _handle->storeEngineDetailsDetachedBuffer(ptr2, std::move(buffer2));

    _handle->removeEngineDetailsDetachedBuffer(ptr1);
    _handle->removeEngineDetailsDetachedBuffer(ptr2);
}

TEST_F(TestHipblasltHipdnnEnginePluginHandle, SetStreamStoresStream)
{
    auto fakeStream = reinterpret_cast<hipStream_t>(0x123);

    EXPECT_NO_THROW(_handle->setStream(fakeStream));
    EXPECT_EQ(_handle->getStream(), fakeStream);
}

// Engine-details buffers are stored and removed from every thread that queries engine details
// through one handle. Unsynchronized access to the underlying map corrupts it (and crashes
// under this load); with the lock every key stored is removed and the map ends empty.
TEST_F(TestHipblasltHipdnnEnginePluginHandle, ConcurrentStoreAndRemoveOfDistinctBuffers)
{
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_PAIRS = 2000;
    // Removal trails the store by a window so the map holds entries (and rehashes) while the
    // other threads insert and erase.
    constexpr size_t WINDOW = 16;

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for(int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([this]() {
            // Keyed by the buffer's own address, as the engine does.
            std::deque<const void*> pending;
            for(int i = 0; i < NUM_PAIRS; ++i)
            {
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(builder.CreateString("test"));
                auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
                const void* key = buffer->data();
                _handle->storeEngineDetailsDetachedBuffer(key, std::move(buffer));
                pending.push_back(key);

                if(pending.size() > WINDOW)
                {
                    _handle->removeEngineDetailsDetachedBuffer(pending.front());
                    pending.pop_front();
                }
            }
            for(const void* key : pending)
            {
                _handle->removeEngineDetailsDetachedBuffer(key);
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(_handle->engineDetailsDetachedBufferCount(), 0U);
}
