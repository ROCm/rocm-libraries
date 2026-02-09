/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <memory>
#include <span>
#include <vector>

#include <hip/hip_runtime.h>

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

using namespace rocRoller;

template <typename T>
static std::shared_ptr<T> makeSharedDeviceUninitialized(size_t count)
{
    T* raw = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&raw), count * sizeof(T)));
    return std::shared_ptr<T>(raw, [](T* p) {
        hipError_t err = hipFree(p);
        if(err != hipSuccess)
        {
            Throw<FatalError>("hipFree failed in RotatingBuffer deleter");
        }
    });
}

template <typename T>
class RotatingBuffer
{
public:
    RotatingBuffer(const std::vector<T>& hostData, size_t rotatingBufferSizeBytes)
        : m_numElems(hostData.size())
        , m_rotatingBufferElems(0)
        , m_currentOffset(0)
    {
        if(hostData.empty())
        {
            Throw<FatalError>("RotatingBuffer: hostData is empty");
        }

        // If rotatingBufferSize == 0, disable rotation
        if(rotatingBufferSizeBytes == 0)
        {
            m_buffer = makeSharedDeviceUninitialized<T>(m_numElems);
            HIP_CHECK(hipMemcpy(
                m_buffer.get(), hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice));
            return;
        }

        // Total size of one tensor copy
        size_t tensorBytes = m_numElems * sizeof(T);

        if(rotatingBufferSizeBytes < tensorBytes)
        {
            m_rotatingBufferElems = 0; // mark rotation disabled
            m_buffer              = makeSharedDeviceUninitialized<T>(m_numElems);
            HIP_CHECK(hipMemcpy(
                m_buffer.get(), hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice));
            return;
        }

        // Align rotation buffer to multiples of tensor size
        size_t alignedBytes   = (rotatingBufferSizeBytes / tensorBytes) * tensorBytes;
        m_rotatingBufferElems = alignedBytes / sizeof(T);

        // If still no room for rotation, just allocate one copy
        if(m_rotatingBufferElems < m_numElems)
        {
            m_rotatingBufferElems = 0; // mark rotation disabled
            m_buffer              = makeSharedDeviceUninitialized<T>(m_numElems);
            HIP_CHECK(hipMemcpy(
                m_buffer.get(), hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice));
        }
        else
        {
            m_buffer = makeSharedDeviceUninitialized<T>(m_rotatingBufferElems);

            size_t numCopies = m_rotatingBufferElems / m_numElems;
            for(size_t r = 0; r < numCopies; ++r)
            {
                T* dst = m_buffer.get() + r * m_numElems;
                HIP_CHECK(
                    hipMemcpy(dst, hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice));
            }
        }
    }

    std::span<T> next()
    {
        // No rotation case (either disabled or fell back to single buffer)
        if(m_rotatingBufferElems == 0)
        {
            m_currentOffset = 0;
            return std::span<T>(m_buffer.get(), m_numElems);
        }

        if(m_numElems < m_rotatingBufferElems)
        {
            m_currentOffset = (m_currentOffset + m_numElems) % m_rotatingBufferElems;
            //  if offset + size would overflow, just reset to base
            if(m_currentOffset + m_numElems > m_rotatingBufferElems)
            {
                m_currentOffset = 0;
            }
        }
        else
        {
            m_currentOffset = 0; // no rotation possible
        }

        return std::span<T>(m_buffer.get() + m_currentOffset, m_numElems);
    }

private:
    size_t m_numElems; // number of elements in one matrix
    size_t
        m_rotatingBufferElems; // rotating buffer size, in elements (0 means rotation is disabled)
    size_t             m_currentOffset;
    std::shared_ptr<T> m_buffer;
};
