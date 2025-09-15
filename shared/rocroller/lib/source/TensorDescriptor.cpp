/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>

#include <rocRoller/TensorDescriptor.hpp>

namespace rocRoller
{
    int64_t TensorDescriptor::dimensionPadding(size_t dim) const
    {
        AssertFatal(dim < dimensions(), ShowValue(dim), ShowValue(dimensions()));

        if(dim == 0)
            return m_strides[0] - 1;

        return m_strides[dim] - (m_strides[dim - 1] * m_sizes[dim - 1]);
    }

    void TensorDescriptor::collapseDims(size_t begin, size_t end)
    {
        AssertFatal(end >= begin, ShowValue(begin), ShowValue(end));
        AssertFatal(begin < dimensions(), ShowValue(begin), ShowValue(dimensions()));
        AssertFatal(end <= dimensions(), ShowValue(begin), ShowValue(dimensions()));

        if(end <= (begin + 1))
            return;

        for(size_t i = begin + 1; i < end; i++)
            AssertFatal(dimensionPadding(i) == 0, ShowValue(i), ShowValue(dimensionPadding(i)));

        size_t newDimensionSize = 1;
        for(size_t i = begin; i < end; i++)
            newDimensionSize *= m_sizes[i];

        m_sizes.erase(m_sizes.begin() + (begin + 1), m_sizes.begin() + end);
        m_sizes[begin] = newDimensionSize;

        m_strides.erase(m_strides.begin() + (begin + 1), m_strides.begin() + end);

        calculate();
    }

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
    {
        return stream << t.toString();
    }

}