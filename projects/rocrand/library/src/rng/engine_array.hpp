// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCRAND_RNG_ENGINE_ARRAY_H_
#define ROCRAND_RNG_ENGINE_ARRAY_H_

#include "common.hpp"

#include <rocrand/rocrand.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace rocrand_impl::host
{

/// \brief Owns the engine array of a generator.
///
/// Generation kernels captured into a HIP graph keep the raw engine pointer they were
/// launched with, so a graph launched after the generator is reinitialized (for example
/// after a seed, offset or ordering change) still accesses that pointer. To keep such
/// launches from touching freed memory, the array is reused when it is large enough, and
/// when it must grow after a generation was captured, the old allocation is kept alive
/// until the generator is destroyed.
template<class T, class System>
class engine_array
{
public:
    engine_array() = default;

    engine_array(const engine_array&) = delete;

    engine_array(engine_array&& other)
        : m_data(std::exchange(other.m_data, nullptr))
        , m_capacity(std::exchange(other.m_capacity, 0))
        , m_captured(std::exchange(other.m_captured, false))
        , m_retired(std::exchange(other.m_retired, {}))
    {}

    engine_array& operator=(const engine_array&) = delete;

    engine_array& operator=(engine_array&& other)
    {
        std::swap(m_data, other.m_data);
        std::swap(m_capacity, other.m_capacity);
        std::swap(m_captured, other.m_captured);
        std::swap(m_retired, other.m_retired);
        return *this;
    }

    ~engine_array()
    {
        if(m_data != nullptr)
        {
            System::free(m_data);
        }
        for(T* retired : m_retired)
        {
            System::free(retired);
        }
    }

    T* data() const
    {
        return m_data;
    }

    /// \brief Makes the array hold at least \p size engines. The contents are not preserved.
    rocrand_status reserve(size_t size)
    {
        if(m_data != nullptr && size <= m_capacity)
        {
            // Freeing the array used to wait for all pending work. Keep doing so, as work
            // queued on a previous stream may still use the engines that are about to be
            // reinitialized.
            if(hipDeviceSynchronize() != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
            return ROCRAND_STATUS_SUCCESS;
        }

        T*                   data   = nullptr;
        const rocrand_status status = System::alloc(&data, size);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        if(m_data != nullptr)
        {
            if(m_captured)
            {
                m_retired.push_back(m_data);
            }
            else
            {
                System::free(m_data);
            }
        }
        m_data     = data;
        m_capacity = size;
        return ROCRAND_STATUS_SUCCESS;
    }

    /// \brief Records whether work using the array is being captured into a graph on \p stream.
    /// Must be called by every operation that launches work using the array.
    void track_capture(hipStream_t stream)
    {
        if(m_captured)
        {
            return;
        }
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        if(hipStreamIsCapturing(stream, &capture_status) != hipSuccess)
        {
            // The query fails for example on the legacy default stream while another stream
            // is capturing. Assume capture, and clear the error so it isn't reported by a
            // subsequent kernel launch check.
            static_cast<void>(hipGetLastError());
            m_captured = true;
            return;
        }
        m_captured = capture_status != hipStreamCaptureStatusNone;
    }

private:
    T*              m_data     = nullptr;
    size_t          m_capacity = 0;
    bool            m_captured = false;
    std::vector<T*> m_retired;
};

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_ENGINE_ARRAY_H_
