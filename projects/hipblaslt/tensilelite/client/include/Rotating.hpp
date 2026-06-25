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
        std::vector<size_t> sizes;       // pristine footprint (slot 0 aliases the pristine pool)
        std::vector<size_t> actualSizes; // per-problem actual footprint (rotation copies, slot > 0)
        size_t              totalSize;
        size_t              actualTotalSize;
        size_t              rotatingNum;
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
        void addRotatingSize(std::vector<size_t> sizes, std::vector<size_t> actualSizes);
        void createRotatingMemory(int32_t mode, size_t rotatingSize, size_t maxRotatingBufferNum);
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
}
