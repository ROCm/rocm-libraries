// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_UTILS_THREEDIM_ITERATOR_
#define ROCRAND_RNG_UTILS_THREEDIM_ITERATOR_

#include <hip/hip_runtime.h>

#include <cassert>
#include <iterator>

#include <stdint.h>

namespace rocrand_impl::cpp_utils
{

/// \brief A random access iterator that converts linear indices to three-dimensional indexing.
class threedim_iterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = ptrdiff_t;
    using value_type        = dim3;
    using reference         = dim3;
    using pointer           = dim3*;

    /// \brief Constructs a new instance of `threedim_iterator`.
    threedim_iterator() : m_dimensions(dim3()), m_index(0) {}

    /// \brief Constructs a new instance of `threedim_iterator`.
    /// \param dimensions The extents of the 3D grid which specifies how the linear indices are transformed.
    /// \param index The starting linear index.
    explicit threedim_iterator(const dim3 dimensions, const size_t index = 0)
        : m_dimensions(dimensions), m_index(index)
    {
        assert(m_dimensions.x != 0);
        assert(m_dimensions.y != 0);
        assert(m_dimensions.z != 0);
    }

    /// \brief Constructs a `threedim_iterator` at the beginning of a range specified by `dimensions`.
    /// The constructed iterator "points" to 3D indices `{0, 0, 0}`.
    /// \param dimensions The extents of the 3D grid.
    [[nodiscard]] static threedim_iterator begin(const dim3 dimensions)
    {
        return threedim_iterator(dimensions);
    }

    /// \brief Constructs a `threedim_iterator` at the end of a range specified by `dimensions`.
    /// \param dimensions The extents of the 3D grid.
    [[nodiscard]] static threedim_iterator end(const dim3 dimensions)
    {
        return threedim_iterator(dimensions,
                                 static_cast<size_t>(dimensions.x) * dimensions.y * dimensions.z);
    }

    [[nodiscard]] bool operator!=(const threedim_iterator& other) const
    {
        return m_index != other.m_index || m_dimensions.x != other.m_dimensions.x
               || m_dimensions.y != other.m_dimensions.y || m_dimensions.z != other.m_dimensions.z;
    }

    [[nodiscard]] bool operator==(const threedim_iterator& other) const
    {
        return !this->operator!=(other);
    }

    [[nodiscard]] bool operator>(const threedim_iterator& other) const
    {
        return m_index > other.m_index;
    }

    [[nodiscard]] bool operator>=(const threedim_iterator& other) const
    {
        return m_index >= other.m_index;
    }

    [[nodiscard]] bool operator<(const threedim_iterator& other) const
    {
        return m_index < other.m_index;
    }

    [[nodiscard]] bool operator<=(const threedim_iterator& other) const
    {
        return m_index <= other.m_index;
    }

    [[maybe_unused]] threedim_iterator& operator++()
    {
        ++m_index;
        return *this;
    }

    [[maybe_unused]] threedim_iterator& operator+=(const difference_type diff)
    {
        m_index += diff;
        return *this;
    }

    [[nodiscard]] threedim_iterator operator++(int)
    {
        const auto tmp = *this;
        this->     operator++();
        return tmp;
    }

    [[nodiscard]] threedim_iterator operator+(const difference_type diff) const
    {
        return threedim_iterator(m_dimensions, m_index + diff);
    }

    [[nodiscard]] friend threedim_iterator operator+(const difference_type    diff,
                                                     const threedim_iterator& iter)
    {
        return iter + diff;
    }

    [[maybe_unused]] threedim_iterator& operator--()
    {
        --m_index;
        return *this;
    }

    [[maybe_unused]] threedim_iterator& operator-=(const difference_type diff)
    {
        m_index -= diff;
        return *this;
    }

    [[nodiscard]] threedim_iterator operator--(int)
    {
        const auto tmp = *this;
        this->     operator--();
        return tmp;
    }

    [[nodiscard]] difference_type operator-(const threedim_iterator& other) const
    {
        return static_cast<difference_type>(m_index) - static_cast<difference_type>(other.m_index);
    }

    [[nodiscard]] threedim_iterator operator-(const difference_type diff) const
    {
        return threedim_iterator(m_dimensions, m_index - diff);
    }

    [[nodiscard]] reference operator*() const
    {
        dim3 ret{0, 0, 0};
        ret.x = static_cast<uint32_t>(m_index % m_dimensions.x);
        ret.y = static_cast<uint32_t>((m_index / m_dimensions.x) % m_dimensions.y);
        ret.z = static_cast<uint32_t>(m_index / m_dimensions.x / m_dimensions.y);
        return ret;
    }

    [[nodiscard]] reference operator[](const difference_type diff) const
    {
        return *this->operator+(diff);
    }

private:
    dim3   m_dimensions;
    size_t m_index;
};

} // end namespace rocrand_impl::cpp_utils

#endif // ROCRAND_RNG_UTILS_THREEDIM_ITERATOR_
