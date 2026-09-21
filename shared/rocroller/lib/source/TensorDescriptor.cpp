// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
