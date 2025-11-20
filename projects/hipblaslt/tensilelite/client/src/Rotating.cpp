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

#include "Rotating.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>

namespace TensileLite
{
    void RotatingMemory::addRotatingSize(std::vector<size_t> sizes)
    {
        if(m_rotatingBufferNum != sizes.size())
        {
            throw std::runtime_error("Rotating buffer number is not equal to size number");
        }
        size_t totalSize = 0;
        for(auto size : sizes)
        {
            totalSize += size;
        }
        m_rotatingInfo.push_back(RotatingUnitInfo{sizes, totalSize, 0});
    }

    void RotatingMemory::createRotatingMemory(int32_t mode, size_t rotatingSize)
    {
        // Check how many rotating units are needed
        m_rotatingSize = rotatingSize;
        size_t maxNumRotate = 0;
        for (auto& unit : m_rotatingInfo)
        {
            size_t num = std::ceil((float)rotatingSize / unit.totalSize);
            unit.rotatingNum = num;
            maxNumRotate = std::max(maxNumRotate, num);
        }

        m_rotatingMemory.resize(maxNumRotate);
        for(size_t i = 0; i < m_rotatingMemory.size(); i++)
        {
            m_rotatingMemory[i].resize(m_rotatingBufferNum);
            for(auto& unit : m_rotatingInfo)
            {
                if(unit.rotatingNum > i)
                {
                    size_t j = 0;
                    for(auto size : unit.sizes)
                    {
                        m_rotatingMemory[i][j].size = std::max(size, m_rotatingMemory[i][j].size);
                        j++;
                    }
                }
            }
        }

        size_t totalSize = 0;
        size_t largestUnitSize = 0;
        if(mode == 0)
        {
            for(auto info : m_rotatingInfo)
            {
                totalSize = std::max(totalSize, info.totalSize * (info.rotatingNum - 1));
            }
            for(auto unit : m_rotatingMemory[0])
            {
                largestUnitSize += unit.size;
            }
            totalSize += largestUnitSize;
        }
        else if(mode == 1)
        {
            for(auto rotatingUnit : m_rotatingMemory)
            {
                for(auto unit : rotatingUnit)
                {
                    totalSize += unit.size;
                }
                for(auto unit : m_rotatingMemory[0])
                {
                    largestUnitSize += unit.size;
                }
            }
        }
        else
        {
            throw std::runtime_error("Unsupported mode");
        }
        m_size = totalSize;
        m_largestUnitSize = largestUnitSize;

        void* ptr   = nullptr;
        static_cast<void>(hipMalloc(&ptr, totalSize));
        m_data = std::shared_ptr<void>(ptr, hipFree);
        std::cout << "Rotating memory size: " << totalSize << std::endl;

        size_t limit = (mode == 1) ? m_rotatingMemory.size() : 1;
        size_t offset = 0;
        for(size_t i = 0; i < limit; i++)
        {
            for(auto& rotatingUnit : m_rotatingMemory[i])
            {
                rotatingUnit.data = std::shared_ptr<void>(
                        m_data, (void*)((uint8_t*)m_data.get() + offset));
                offset += rotatingUnit.size;
                if(offset > totalSize)
                {
                    throw std::runtime_error("Rotating memory offset exceeds total size");
                }
            }
        }
    }

    std::vector<std::vector<RotatingMemoryUnit>> RotatingMemory::getRotatingMemory() const
    {
        return m_rotatingMemory;
    }

    std::shared_ptr<void> RotatingMemory::getData() const
    {
        return m_data;
    }

    size_t RotatingMemory::getDataSize() const
    {
        return m_size;
    }

    size_t RotatingMemory::getDataLargestUnitSize() const
    {
        return m_largestUnitSize;
    }
}
