// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include <Tensile/hip/HipUtils.hpp>

namespace TensileLite
{
namespace testing
{
    class HipStreamGuard
    {
    public:
        explicit HipStreamGuard(unsigned int flags = hipStreamDefault)
        {
            HIP_CHECK_EXC(hipStreamCreateWithFlags(&m_stream, flags));
        }

        ~HipStreamGuard() noexcept
        {
            destroyNoThrow();
        }

        HipStreamGuard(HipStreamGuard const&)            = delete;
        HipStreamGuard& operator=(HipStreamGuard const&) = delete;

        HipStreamGuard(HipStreamGuard&& other) noexcept : m_stream(other.m_stream)
        {
            other.m_stream = nullptr;
        }

        HipStreamGuard& operator=(HipStreamGuard&& other) noexcept
        {
            if(this == &other)
                return *this;

            destroyNoThrow();
            m_stream       = other.m_stream;
            other.m_stream = nullptr;
            return *this;
        }

        hipStream_t get() const noexcept
        {
            return m_stream;
        }

        void synchronize() const
        {
            if(m_stream != nullptr)
                HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
        }

    private:
        void destroyNoThrow() noexcept
        {
            if(m_stream != nullptr)
            {
                (void)hipStreamDestroy(m_stream);
                m_stream = nullptr;
            }
        }

        hipStream_t m_stream = nullptr;
    };
} // namespace testing
} // namespace TensileLite
