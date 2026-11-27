// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <thread>

#include <hip/hip_runtime.h>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/GraphCapture.hpp"

// NOLINTBEGIN(readability-identifier-naming)

// HIP checks hipMalloc / hipFree against the calling thread's capture mode, so an
// allocation on one thread invalidates another thread's Global-mode capture unless the
// allocating thread is in Relaxed mode. These tests pin that property with raw HIP
// calls, because a plan build or destroy on another thread during a capture is only as
// safe as the allocations it makes.

namespace
{

using hipdnn_integration_tests::detail::abandonCapture;
using hipdnn_integration_tests::detail::HipGraph;
using hipdnn_integration_tests::detail::HipGraphExec;

struct CrossThreadAllocation
{
    hipError_t allocate = hipSuccess;
    hipError_t release = hipSuccess;
    hipError_t exchangeIn = hipSuccess;
    hipError_t exchangeOut = hipSuccess;
};

// Allocates and frees a small buffer on a second thread, optionally switching that
// thread to Relaxed capture mode around the allocation and restoring its mode after.
CrossThreadAllocation allocateOnOtherThread(int device, bool relaxed)
{
    CrossThreadAllocation outcome;
    std::thread worker([&outcome, device, relaxed] {
        static_cast<void>(hipSetDevice(device));

        hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
        if(relaxed)
        {
            // On return `mode` holds the thread's previous mode.
            outcome.exchangeIn = hipThreadExchangeStreamCaptureMode(&mode);
        }

        void* buffer = nullptr;
        outcome.allocate = hipMalloc(&buffer, 4096);
        if(outcome.allocate == hipSuccess)
        {
            outcome.release = hipFree(buffer);
        }

        if(relaxed && outcome.exchangeIn == hipSuccess)
        {
            outcome.exchangeOut = hipThreadExchangeStreamCaptureMode(&mode);
        }

        // A rejected allocation sets the thread's sticky error; keep it from leaking.
        static_cast<void>(hipGetLastError());
    });
    worker.join();
    return outcome;
}

class TestCrossThreadCapture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipMalloc(&_buffer, 256), hipSuccess);
    }

    void TearDown() override
    {
        if(_stream != nullptr)
        {
            abandonCapture(_stream);
            static_cast<void>(hipStreamDestroy(_stream));
            _stream = nullptr;
        }
        if(_buffer != nullptr)
        {
            static_cast<void>(hipFree(_buffer));
            _buffer = nullptr;
        }
    }

    hipStream_t _stream = nullptr;
    void* _buffer = nullptr;
};

// Every HIP result is recorded first and asserted after, so the capture is already
// closed when an assertion returns from the test body.
TEST_F(TestCrossThreadCapture, ConcurrentGlobalModeBuildInvalidatesCapture)
{
    const auto begin = hipStreamBeginCapture(_stream, hipStreamCaptureModeGlobal);
    const auto enqueue
        = begin == hipSuccess ? hipMemsetAsync(_buffer, 0, 256, _stream) : hipErrorUnknown;

    // This thread is Global mode (the default), as is any thread that builds or
    // destroys a plan without choosing otherwise.
    const auto other = allocateOnOtherThread(0, false);

    HipGraph graph;
    const auto end
        = begin == hipSuccess ? hipStreamEndCapture(_stream, graph.out()) : hipErrorUnknown;
    abandonCapture(_stream);

    ASSERT_EQ(begin, hipSuccess);
    ASSERT_EQ(enqueue, hipSuccess);
    EXPECT_NE(other.allocate, hipSuccess)
        << "a Global-mode hipMalloc during another thread's capture was accepted";
    EXPECT_EQ(end, hipErrorStreamCaptureInvalidated);
}

TEST_F(TestCrossThreadCapture, RelaxedModeBuildDoesNotInvalidateCapture)
{
    const auto begin = hipStreamBeginCapture(_stream, hipStreamCaptureModeGlobal);
    const auto enqueue
        = begin == hipSuccess ? hipMemsetAsync(_buffer, 0, 256, _stream) : hipErrorUnknown;

    const auto other = allocateOnOtherThread(0, true);

    HipGraph graph;
    HipGraphExec exec;
    const auto end
        = begin == hipSuccess ? hipStreamEndCapture(_stream, graph.out()) : hipErrorUnknown;
    const auto instantiate = end == hipSuccess
                                 ? hipGraphInstantiateWithFlags(exec.out(), graph.get(), 0)
                                 : hipErrorUnknown;
    abandonCapture(_stream);

    ASSERT_EQ(begin, hipSuccess);
    ASSERT_EQ(enqueue, hipSuccess);
    EXPECT_EQ(other.exchangeIn, hipSuccess);
    EXPECT_EQ(other.exchangeOut, hipSuccess);
    EXPECT_EQ(other.allocate, hipSuccess);
    EXPECT_EQ(other.release, hipSuccess);
    EXPECT_EQ(end, hipSuccess);
    EXPECT_EQ(instantiate, hipSuccess);
}

} // namespace

// NOLINTEND(readability-identifier-naming)
