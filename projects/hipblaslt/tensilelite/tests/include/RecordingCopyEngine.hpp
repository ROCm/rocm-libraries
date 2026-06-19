// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipCopyEngine.hpp"

#include <cstdint>
#include <vector>

namespace TensileLite::testing
{
    class RecordingCopyEngine final : public TensileLite::Client::CopyEngine
    {
    public:
        using CopySubmissionMode = TensileLite::Client::CopyEngine::CopySubmissionMode;

        enum class CallType
        {
            Copy,
            Synchronize,
            SynchronizeDefaultStream,
            RecordCopyDone,
            WaitForCopyDone
        };

        struct Call
        {
            CallType         type            = CallType::Copy;
            void*            dst             = nullptr;
            void const*      src             = nullptr;
            size_t           bytes           = 0;
            hipMemcpyKind    copyKind        = hipMemcpyHostToHost;
            hipStream_t      stream          = nullptr;
            CopySubmissionMode submissionMode = CopySubmissionMode::Sync;
            size_t           slot            = 0;
            hipStream_t      computeStream   = nullptr;
        };

        explicit RecordingCopyEngine(
            hipStream_t stream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1)))
            : m_stream(stream)
        {
        }

        hipStream_t stream() const noexcept override
        {
            return m_stream;
        }

        void copy(void*               dst,
                  void const*         src,
                  size_t              bytes,
                  hipMemcpyKind       kind,
                  hipStream_t         stream,
                  CopySubmissionMode   mode) override
        {
            calls.push_back({CallType::Copy, dst, src, bytes, kind, stream, mode});
        }

        void synchronize(hipStream_t stream) override
        {
            calls.push_back({CallType::Synchronize,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             stream,
                             CopySubmissionMode::Sync});
        }

        void synchronizeDefaultStream() override
        {
            calls.push_back({CallType::SynchronizeDefaultStream,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             m_stream,
                             CopySubmissionMode::Sync});
        }

        void recordCopyDone(size_t slot) override
        {
            calls.push_back({CallType::RecordCopyDone,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             m_stream,
                             CopySubmissionMode::Sync,
                             slot});
        }

        void waitForCopyDone(size_t slot, hipStream_t computeStream) override
        {
            calls.push_back({CallType::WaitForCopyDone,
                             nullptr,
                             nullptr,
                             0,
                             hipMemcpyHostToHost,
                             nullptr,
                             CopySubmissionMode::Sync,
                             slot,
                             computeStream});
        }

        void clear()
        {
            calls.clear();
        }

        std::vector<Call> calls;

    private:
        hipStream_t m_stream = nullptr;
    };
} // namespace TensileLite::testing
