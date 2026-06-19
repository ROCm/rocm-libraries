// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TensileLite::Client
{
    class HipStreamHandle
    {
    public:
        explicit HipStreamHandle(unsigned int flags = hipStreamDefault)
        {
            HIP_CHECK_EXC(hipStreamCreateWithFlags(&m_stream, flags));
        }

        ~HipStreamHandle() noexcept
        {
            destroyNoThrow();
        }

        HipStreamHandle(HipStreamHandle const&)            = delete;
        HipStreamHandle& operator=(HipStreamHandle const&) = delete;

        HipStreamHandle(HipStreamHandle&& other) noexcept
            : m_stream(std::exchange(other.m_stream, nullptr))
        {
        }

        HipStreamHandle& operator=(HipStreamHandle&& other) noexcept
        {
            if(this != &other)
            {
                destroyNoThrow();
                m_stream = std::exchange(other.m_stream, nullptr);
            }
            return *this;
        }

        hipStream_t get() const noexcept
        {
            return m_stream;
        }

        explicit operator bool() const noexcept
        {
            return m_stream != nullptr;
        }

        void synchronize() const
        {
            if(m_stream != nullptr)
                HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
        }

    private:
        void destroyNoThrow() noexcept
        {
            if(m_stream == nullptr)
                return;

            hipError_t err = hipStreamDestroy(m_stream);
            if(err != hipSuccess)
            {
                std::cerr << "HipStreamHandle: hipStreamDestroy failed: "
                          << hipGetErrorString(err) << std::endl;
            }
            m_stream = nullptr;
        }

        hipStream_t m_stream = nullptr;
    };

    class HipEventHandle
    {
    public:
        explicit HipEventHandle(unsigned int flags = hipEventDisableTiming)
        {
            HIP_CHECK_EXC(hipEventCreateWithFlags(&m_event, flags));
        }

        ~HipEventHandle() noexcept
        {
            destroyNoThrow();
        }

        HipEventHandle(HipEventHandle const&)            = delete;
        HipEventHandle& operator=(HipEventHandle const&) = delete;

        HipEventHandle(HipEventHandle&& other) noexcept
            : m_event(std::exchange(other.m_event, nullptr))
        {
        }

        HipEventHandle& operator=(HipEventHandle&& other) noexcept
        {
            if(this != &other)
            {
                destroyNoThrow();
                m_event = std::exchange(other.m_event, nullptr);
            }
            return *this;
        }

        hipEvent_t get() const noexcept
        {
            return m_event;
        }

        explicit operator bool() const noexcept
        {
            return m_event != nullptr;
        }

        void record(hipStream_t stream) const
        {
            if(m_event != nullptr)
                HIP_CHECK_EXC(hipEventRecord(m_event, stream));
        }

        void wait(hipStream_t stream) const
        {
            if(m_event != nullptr)
                HIP_CHECK_EXC(hipStreamWaitEvent(stream, m_event, 0));
        }

    private:
        void destroyNoThrow() noexcept
        {
            if(m_event == nullptr)
                return;

            hipError_t err = hipEventDestroy(m_event);
            if(err != hipSuccess)
            {
                std::cerr << "HipEventHandle: hipEventDestroy failed: "
                          << hipGetErrorString(err) << std::endl;
            }
            m_event = nullptr;
        }

        hipEvent_t m_event = nullptr;
    };

    template <typename T>
    class PinnedHostBuffer
    {
    public:
        PinnedHostBuffer() = default;

        explicit PinnedHostBuffer(size_t count)
        {
            allocate(count);
        }

        ~PinnedHostBuffer() noexcept
        {
            reset();
        }

        PinnedHostBuffer(PinnedHostBuffer const&)            = delete;
        PinnedHostBuffer& operator=(PinnedHostBuffer const&) = delete;

        PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
            : m_ptr(std::exchange(other.m_ptr, nullptr))
            , m_count(std::exchange(other.m_count, 0))
        {
        }

        PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept
        {
            if(this != &other)
            {
                reset();
                m_ptr   = std::exchange(other.m_ptr, nullptr);
                m_count = std::exchange(other.m_count, 0);
            }
            return *this;
        }

        void allocate(size_t count)
        {
            reset();
            if(count == 0)
                return;

            if(count > (std::numeric_limits<size_t>::max() / sizeof(T)))
            {
                throw std::overflow_error("PinnedHostBuffer allocation overflow.");
            }

            void* raw = nullptr;
            HIP_CHECK_EXC(hipHostMalloc(&raw, count * sizeof(T), 0));
            m_ptr   = static_cast<T*>(raw);
            m_count = count;
        }

        void reset() noexcept
        {
            if(m_ptr == nullptr)
                return;

            hipError_t err = hipHostFree(m_ptr);
            if(err != hipSuccess)
            {
                std::cerr << "PinnedHostBuffer: hipHostFree failed: "
                          << hipGetErrorString(err) << std::endl;
            }
            m_ptr   = nullptr;
            m_count = 0;
        }

        T* get() noexcept
        {
            return m_ptr;
        }

        T* get() const noexcept
        {
            return m_ptr;
        }

        size_t size() const noexcept
        {
            return m_count;
        }

        bool empty() const noexcept
        {
            return m_ptr == nullptr;
        }

        explicit operator bool() const noexcept
        {
            return m_ptr != nullptr;
        }

    private:
        T*     m_ptr   = nullptr;
        size_t m_count = 0;
    };

    class CopyEngine
    {
    public:
        enum class CopySubmissionMode
        {
            Sync,
            Async
        };

        virtual ~CopyEngine() = default;

        virtual hipStream_t stream() const noexcept                                      = 0;
        virtual void        copy(void*             dst,
                                 void const*       src,
                                 size_t            bytes,
                                 hipMemcpyKind     kind,
                                 hipStream_t       stream,
                                 CopySubmissionMode mode)                             = 0;
        virtual void synchronize(hipStream_t stream)                                     = 0;
        virtual void synchronizeDefaultStream()                                           = 0;
        virtual void recordCopyDone(size_t slot)                                           = 0;
        virtual void waitForCopyDone(size_t slot, hipStream_t computeStream)               = 0;
    };

    inline CopyEngine::CopySubmissionMode submissionModeForStream(hipStream_t stream) noexcept
    {
        return stream == nullptr ? CopyEngine::CopySubmissionMode::Sync
                                 : CopyEngine::CopySubmissionMode::Async;
    }

    class HipCopyEngine final : public CopyEngine
    {
    public:
        explicit HipCopyEngine(size_t eventCount, unsigned int streamFlags = hipStreamDefault)
            : m_copyStream(streamFlags)
        {
            m_copyDoneEvents.reserve(eventCount);
            for(size_t i = 0; i < eventCount; ++i)
                m_copyDoneEvents.emplace_back(hipEventDisableTiming);
        }

        ~HipCopyEngine() noexcept
        {
            try
            {
                synchronizeDefaultStream();
            }
            catch(std::exception const& e)
            {
                std::cerr << "HipCopyEngine: synchronizeDefaultStream failed: " << e.what()
                          << std::endl;
            }
            catch(...)
            {
                std::cerr << "HipCopyEngine: synchronizeDefaultStream failed: unknown error"
                          << std::endl;
            }
        }

        HipCopyEngine(HipCopyEngine const&)            = delete;
        HipCopyEngine& operator=(HipCopyEngine const&) = delete;

        hipStream_t stream() const noexcept override
        {
            return m_copyStream.get();
        }

        void copy(void*               dst,
                  void const*         src,
                  size_t              bytes,
                  hipMemcpyKind       kind,
                  hipStream_t         stream,
                  CopySubmissionMode   mode) override
        {
            if(mode == CopySubmissionMode::Sync)
            {
                if(stream != nullptr)
                {
                    throw std::invalid_argument(
                        "HipCopyEngine::copy Sync mode requires a null stream.");
                }
                HIP_CHECK_EXC(hipMemcpy(dst, src, bytes, kind));
            }
            else
            {
                HIP_CHECK_EXC(hipMemcpyAsync(dst, src, bytes, kind, stream));
            }
        }

        void synchronize(hipStream_t stream) override
        {
            if(stream != nullptr)
                HIP_CHECK_EXC(hipStreamSynchronize(stream));
        }

        void synchronizeDefaultStream() override
        {
            m_copyStream.synchronize();
        }

        void recordCopyDone(size_t slot) override
        {
            m_copyDoneEvents.at(slot).record(m_copyStream.get());
        }

        void waitForCopyDone(size_t slot, hipStream_t computeStream) override
        {
            m_copyDoneEvents.at(slot).wait(computeStream);
        }

    private:
        HipStreamHandle              m_copyStream;
        std::vector<HipEventHandle>  m_copyDoneEvents;
    };
} // namespace TensileLite::Client
