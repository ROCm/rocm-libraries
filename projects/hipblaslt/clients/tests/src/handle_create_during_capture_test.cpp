// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for hipblasLtCreate ending the process during another
// thread's graph capture (ROCm/rocm-libraries#12895).
//
// hipblasLtCreate zeroes its synchronizer buffers before it returns. It used
// to do that with hipMemset, which runs on the legacy null stream, and HIP
// can refuse legacy-stream work while another stream in the process is
// capturing a graph, even a thread-local capture on another thread, on a
// stream created non-blocking. The refusal reached a macro that called
// exit(EXIT_FAILURE), so the whole test binary died on the hipblasLtCreate
// line.
//
// Here one thread holds a thread-local capture open while the test thread
// creates a handle. The create must return HIPBLAS_STATUS_SUCCESS, and the
// capture, which saw no work from it, must still end cleanly.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <atomic>
#include <thread>

namespace
{
    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // Holds a thread-local capture open on its own thread, on a non-blocking
    // stream, from construction until finish().
    class CaptureOnAnotherThread
    {
    public:
        explicit CaptureOnAnotherThread(hipStream_t stream)
            : stream_(stream)
            , thread_([this] { run(); })
        {
            while(state_.load() == State::Starting)
                std::this_thread::yield();
        }

        ~CaptureOnAnotherThread()
        {
            static_cast<void>(finish());
        }

        bool capturing() const
        {
            return state_.load() == State::Capturing;
        }

        // Ends the capture and joins the thread. Returns hipStreamEndCapture's
        // result, or the begin error if the capture never opened.
        hipError_t finish()
        {
            if(thread_.joinable())
            {
                release_.store(true);
                thread_.join();
            }
            return result_;
        }

    private:
        enum class State
        {
            Starting,
            Capturing,
            Failed
        };

        void run()
        {
            result_ = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
            if(result_ != hipSuccess)
            {
                state_.store(State::Failed);
                return;
            }
            state_.store(State::Capturing);
            while(!release_.load())
                std::this_thread::yield();
            hipGraph_t graph = nullptr;
            result_          = hipStreamEndCapture(stream_, &graph);
            if(graph != nullptr)
                static_cast<void>(hipGraphDestroy(graph));
        }

        hipStream_t        stream_;
        std::atomic<State> state_{State::Starting};
        std::atomic<bool>  release_{false};
        hipError_t         result_ = hipSuccess;
        std::thread        thread_;
    };

    TEST(HandleCreateDuringCapture, CreateWhileAnotherThreadCaptures)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        hipblasLtHandle_t handle = nullptr;
        hipblasStatus_t   status = HIPBLAS_STATUS_SUCCESS;
        hipError_t        ended  = hipSuccess;
        {
            CaptureOnAnotherThread capture(stream);
            if(!capture.capturing())
            {
                static_cast<void>(hipStreamDestroy(stream));
                GTEST_SKIP() << "hipStreamBeginCapture failed: "
                             << hipGetErrorName(capture.finish());
            }
            status = hipblasLtCreate(&handle);
            ended  = capture.finish();
        }

        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(ended, hipSuccess);
        if(status == HIPBLAS_STATUS_SUCCESS)
            EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
} // namespace
