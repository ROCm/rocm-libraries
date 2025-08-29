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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>

#include <Tensile/Comparison.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    const size_t TensorDescriptor::UseDefaultStride = static_cast<size_t>(-1);

    bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
    {
        return m_dataType == rhs.m_dataType && m_sizes == rhs.m_sizes && m_strides == rhs.m_strides;
    }

    bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const
    {
        return !(*this == rhs);
    }

    void TensorDescriptor::appendDim(size_t size)
    {
        appendDim(size, m_totalAllocatedElements);
    }

    void TensorDescriptor::appendDim(size_t size, size_t stride)
    {
        m_sizes.push_back(size);
        m_strides.push_back(stride);

        calculate();
    }

    int64_t TensorDescriptor::dimensionPadding(size_t dim) const
    {
        TENSILE_ASSERT_EXC(dim < dimensions());

        if(dim == 0)
            return m_strides[0] - 1;

        return m_strides[dim] - (m_strides[dim - 1] * m_sizes[dim - 1]);
    }

    void TensorDescriptor::collapseDims(size_t begin, size_t end)
    {
        TENSILE_ASSERT_EXC(end >= begin);
        TENSILE_ASSERT_EXC(begin < dimensions());
        TENSILE_ASSERT_EXC(end <= dimensions());

        if(end <= (begin + 1))
            return;

        for(size_t i = begin + 1; i < end; i++)
            TENSILE_ASSERT_EXC(dimensionPadding(i) == 0);

        size_t newDimensionSize = 1;
        for(size_t i = begin; i < end; i++)
            newDimensionSize *= m_sizes[i];

        m_sizes.erase(m_sizes.begin() + (begin + 1), m_sizes.begin() + end);
        m_sizes[begin] = newDimensionSize;

        m_strides.erase(m_strides.begin() + (begin + 1), m_strides.begin() + end);

        calculate();
    }

    std::string TensorDescriptor::ToString() const
    {
        std::ostringstream result;

        if(!getName().empty())
        {
            result << "[" << getName() << "]";
        }

        result << dimensions() << "-tensor<" << dataType() << ">"
               << "( sizes(";
        streamJoin(result, m_sizes, ", ");

        result << "), strides(";
        streamJoin(result, m_strides, ", ");
        result << ") )";
        return result.str();
    }

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
    {
        return stream << t.ToString();
    }

} // namespace TensileLite
