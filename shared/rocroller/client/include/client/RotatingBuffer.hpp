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


#include <memory>
#include <vector>
#include <span>

#include <hip/hip_runtime.h>

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Utils.hpp>

template <typename T>
class RotatingBuffer
{
public:
    RotatingBuffer(const std::vector<T>& hostData, size_t rotatingBufferSizeBytes)
        : m_numElems(hostData.size()),
          m_rotatingBufferElems(rotatingBufferSizeBytes > 0 ? rotatingBufferSizeBytes / sizeof(T) : 0),
          m_currentOffset(0)
    {
        if(hostData.empty())
        {
            Throw<FatalError>("RotatingBuffer: hostData is empty");
        }

        // If rotatingBufferSize == 0, disable rotation
        if(rotatingBufferSizeBytes == 0)
        {
            m_buffer = make_shared_device<T>(m_numElems);
            HIP_CHECK(hipMemcpy(m_buffer.get(), hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice));
            return;
        }

        if(m_rotatingBufferElems == 0)
        {
            Throw<FatalError>("RotatingBuffer: rotatingBufferSizeBytes too small for element size");
        }

        if(m_numElems >= m_rotatingBufferElems)
        {
            m_buffer = make_shared_device<T>(m_numElems);
            hipMemcpy(m_buffer.get(), hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice);
        }
        else
        {
            m_buffer = make_shared_device<T>(m_rotatingBufferElems);
            size_t numCopies = m_rotatingBufferElems / m_numElems;
            for(size_t r = 0; r < numCopies; ++r)
            {
                T* dst = m_buffer.get() + r * m_numElems;
                HIP_CHECK(hipMemcpy(dst, hostData.data(), m_numElems * sizeof(T), hipMemcpyHostToDevice));
            }
        }
    }

    std::span<T> next()
    {
        if(m_rotatingBufferElems == 0)
        {
            m_currentOffset = 0;
            return std::span<T>(m_buffer.get(), m_numElems);
        }

        if(m_numElems < m_rotatingBufferElems)
        {
            m_currentOffset = (m_currentOffset + m_numElems) % m_rotatingBufferElems;
        }
        else
        {
            m_currentOffset = 0; // always return base
        }
        if(m_currentOffset + m_numElems > (m_numElems < m_rotatingBufferElems ? m_rotatingBufferElems : m_numElems))
        {
            Throw<FatalError>("RotatingBuffer::next: computed offset out of bounds");
        }

        return std::span<T>(m_buffer.get() + m_currentOffset, m_numElems);
    }

private:
    size_t m_numElems;                // number of elements in one matrix
    size_t m_rotatingBufferElems;     // rotating buffer size, in elements (0 means rotation is disabled)
    size_t m_currentOffset;
    std::shared_ptr<T> m_buffer;
};
