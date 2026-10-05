/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>

namespace TensileLite
{
    struct RotatingUnitInfo
    {
        std::vector<size_t> sizes;
        size_t totalSize;
        size_t rotatingNum;
    };

    struct RotatingMemoryUnit
    {
        std::shared_ptr<void> data;
        size_t size;
    };

    class RotatingMemory
    {
    public:
        explicit RotatingMemory(size_t num) : m_rotatingBufferNum(num) {}
        ~RotatingMemory() {}
        void addRotatingSize(std::vector<size_t> sizes);
        void createRotatingMemory(int32_t mode, size_t rotatingSize);
        std::vector<std::vector<RotatingMemoryUnit>> getRotatingMemory() const;
        std::shared_ptr<void> getData() const;
        size_t getDataSize() const;
        size_t getDataLargestUnitSize() const;
    private:
        size_t m_rotatingBufferNum;
        size_t m_rotatingSize;
        std::vector<RotatingUnitInfo> m_rotatingInfo;
        std::vector<std::vector<RotatingMemoryUnit>> m_rotatingMemory;
        std::shared_ptr<void> m_data;
        size_t m_size;
        size_t m_largestUnitSize;
    };

    /// Clamp the requested number of extra rotating-buffer copies to the number
    /// that actually fits in the allocated rotating pool.
    ///
    /// The requested count is derived from the user-requested rotating buffer
    /// size, which can slightly exceed what RotatingMemory actually allocated
    /// (e.g. beta==0 tensors dropped, guard-page rounding), so the two size
    /// computations diverge and the naive count can over-provision.
    ///
    /// @param rotatingNum           Requested number of extra rotating copies.
    /// @param rotatingSize          Size in bytes of one rotating copy.
    /// @param rotatingAllocatedSize Bytes available for rotating copies, i.e.
    ///                              RotatingMemory::getDataSize() minus
    ///                              getDataLargestUnitSize().
    /// @return min(rotatingNum, rotatingAllocatedSize / rotatingSize), never
    ///         negative. All arithmetic is 64-bit on purpose:
    ///         rotatingNum * rotatingSize exceeds INT32_MAX once the rotating
    ///         buffer is >= ~2 GiB, which previously overflowed a signed 32-bit
    ///         accumulator and spuriously aborted the benchmark.
    int32_t
        clampRotatingNum(int32_t rotatingNum, size_t rotatingSize, size_t rotatingAllocatedSize);
}
